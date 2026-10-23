// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createECDH } from 'node:crypto'

import { looksLikeSecret } from './audit.js'
import { HubIdentityError, newId, sha256Hex, type IdentityContext } from './core.js'
import type { PushSubscriptionRow, UserRow } from './store.js'
import { MAX_PUSH_PLAINTEXT_BYTES, createVapidJwt, encryptWebPush, vapidAuthorization } from './webPush.js'

/**
 * Push subscriptions and delivery.
 *
 * Lock-screen text is deliberately minimal: a short title, a short body and a
 * relative URL. Transcript text and tool arguments never belong here. Delivery
 * is only reported when the push service answers 201 or 202.
 */

/** Push services a browser may hand us. Anything else is refused (SSRF guard). */
export const DEFAULT_PUSH_HOSTS: readonly string[] = [
  'fcm.googleapis.com',
  'push.services.mozilla.com',
  'push.apple.com',
  'notify.windows.com',
]

const MAX_ENDPOINT_LENGTH = 2048
const VAPID_LIFETIME_SECONDS = 12 * 60 * 60

export interface PushMessage {
  readonly title: string
  readonly body: string
  /** Same-origin path the notification opens, for example `/task/abc`. */
  readonly url: string
  /** Notifications with the same key are sent at most once per dedupe window. Default: hash of the message. */
  readonly dedupeKey?: string
}

export type PushDeliveryReason = 'accepted' | 'gone' | 'rejected' | 'network_error' | 'endpoint_not_allowed' | 'encryption_failed'

export interface PushDelivery {
  readonly subscriptionId: string
  readonly deviceId: string
  readonly ok: boolean
  readonly reason: PushDeliveryReason
  /** HTTP status from the push service, when one was received. */
  readonly status?: number
}

export type PushFailure = 'no_subscriptions' | 'deduped' | 'user_inactive' | 'vapid_subject_missing' | 'all_failed'

export interface PushResult {
  /** True only when at least one push service answered 201/202. */
  readonly delivered: boolean
  readonly attempted: number
  readonly accepted: number
  readonly failure?: PushFailure
  readonly deliveries: readonly PushDelivery[]
}

function hostAllowed(hosts: readonly string[], hostname: string): boolean {
  return hosts.some(host => hostname === host || hostname.endsWith(`.${host}`))
}

export function validateEndpoint(ctx: Pick<IdentityContext, 'pushHosts'>, endpoint: unknown): URL {
  if (typeof endpoint !== 'string' || endpoint.length === 0 || endpoint.length > MAX_ENDPOINT_LENGTH) throw new HubIdentityError('invalid_request', 'endpoint must be a URL')
  let url: URL
  try {
    url = new URL(endpoint)
  } catch {
    throw new HubIdentityError('invalid_request', 'endpoint must be a URL')
  }
  if (url.protocol !== 'https:' || url.username || url.password || (url.port !== '' && url.port !== '443')) {
    throw new HubIdentityError('invalid_request', 'endpoint must be an https push service URL')
  }
  if (/^[\d.]+$/.test(url.hostname) || url.hostname.includes(':') || !hostAllowed(ctx.pushHosts, url.hostname)) {
    throw new HubIdentityError('forbidden', 'endpoint is not an allowed push service')
  }
  return url
}

function decodeKey(value: unknown, field: string, length: number): Buffer {
  if (typeof value !== 'string' || !/^[A-Za-z0-9_-]+$/.test(value)) throw new HubIdentityError('invalid_request', `${field} must be base64url`)
  const bytes = Buffer.from(value, 'base64url')
  if (bytes.length !== length) throw new HubIdentityError('invalid_request', `${field} must decode to ${length} bytes`)
  return bytes
}

export function subscribePush(
  ctx: IdentityContext,
  who: { readonly userId: string; readonly deviceId: string },
  input: { readonly endpoint: unknown; readonly keys: unknown },
): PushSubscriptionRow {
  const url = validateEndpoint(ctx, input.endpoint)
  const keys = typeof input.keys === 'object' && input.keys !== null ? (input.keys as Record<string, unknown>) : {}
  const p256dh = decodeKey(keys.p256dh, 'keys.p256dh', 65)
  const auth = decodeKey(keys.auth, 'keys.auth', 16)
  try {
    createECDH('prime256v1').generateKeys()
    const probe = createECDH('prime256v1')
    probe.generateKeys()
    probe.computeSecret(p256dh)
  } catch {
    throw new HubIdentityError('invalid_request', 'keys.p256dh is not a valid P-256 point')
  }
  const id = newId(ctx, 'push')
  ctx.store.transaction(() => {
    ctx.store.run('DELETE FROM hub_push_subscriptions WHERE device_id = ? OR endpoint = ?', who.deviceId, url.href)
    ctx.store.run(
      'INSERT INTO hub_push_subscriptions (id, user_id, device_id, endpoint, p256dh, auth, created_at, last_success_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)',
      id,
      who.userId,
      who.deviceId,
      url.href,
      p256dh.toString('base64url'),
      auth.toString('base64url'),
      ctx.clock.now(),
    )
  })
  const row = ctx.store.one<PushSubscriptionRow>('SELECT * FROM hub_push_subscriptions WHERE id = ?', id)
  if (!row) throw new HubIdentityError('internal', 'subscription was not stored')
  return row
}

export function unsubscribePush(ctx: IdentityContext, who: { readonly deviceId: string }, endpoint?: unknown): number {
  if (endpoint === undefined) return ctx.store.run('DELETE FROM hub_push_subscriptions WHERE device_id = ?', who.deviceId)
  const url = validateEndpoint(ctx, endpoint)
  return ctx.store.run('DELETE FROM hub_push_subscriptions WHERE device_id = ? AND endpoint = ?', who.deviceId, url.href)
}

function checkMessage(message: PushMessage): void {
  const fields: [string, string, number][] = [['title', message.title, 80], ['body', message.body, 160], ['url', message.url, 200]]
  for (const [name, value, max] of fields) {
    if (typeof value !== 'string' || value.length === 0 || value.length > max) {
      throw new HubIdentityError('invalid_request', `push ${name} must be 1-${max} characters`)
    }
    if (looksLikeSecret(value)) throw new HubIdentityError('invalid_request', `push ${name} looks like a secret`)
  }
  if (!message.url.startsWith('/') || message.url.startsWith('//')) throw new HubIdentityError('invalid_request', 'push url must be a same-origin path')
}

/** Send one notification to every subscription of `userId`. Never throws for delivery problems. */
export async function sendPush(ctx: IdentityContext, userId: string, message: PushMessage): Promise<PushResult> {
  checkMessage(message)
  const user = ctx.store.one<UserRow>('SELECT * FROM hub_users WHERE id = ?', userId)
  if (!user || user.revoked_at !== null || user.kind !== 'user') return { delivered: false, attempted: 0, accepted: 0, failure: 'user_inactive', deliveries: [] }
  if (ctx.vapidSubject === undefined) return { delivered: false, attempted: 0, accepted: 0, failure: 'vapid_subject_missing', deliveries: [] }
  const subscriptions = ctx.store.all<PushSubscriptionRow>(
    `SELECT s.* FROM hub_push_subscriptions s JOIN hub_devices d ON d.id = s.device_id
     WHERE s.user_id = ? AND d.status = 'active' ORDER BY s.created_at ASC`,
    userId,
  )
  if (subscriptions.length === 0) return { delivered: false, attempted: 0, accepted: 0, failure: 'no_subscriptions', deliveries: [] }

  const now = ctx.clock.now()
  const key = `${userId}:${message.dedupeKey ?? sha256Hex(`${message.title}\0${message.body}\0${message.url}`)}`
  for (const [entry, at] of ctx.pushDedupe) if (now - at >= ctx.limits.pushDedupeMs) ctx.pushDedupe.delete(entry)
  if (ctx.pushDedupe.has(key)) return { delivered: false, attempted: 0, accepted: 0, failure: 'deduped', deliveries: [] }
  ctx.pushDedupe.set(key, now)

  const payload = new TextEncoder().encode(JSON.stringify({ title: message.title, body: message.body, url: message.url }))
  if (payload.length > MAX_PUSH_PLAINTEXT_BYTES) {
    ctx.pushDedupe.delete(key)
    throw new HubIdentityError('invalid_request', 'push payload is too large')
  }
  const deliveries: PushDelivery[] = []
  for (const subscription of subscriptions) deliveries.push(await deliver(ctx, subscription, payload))
  const accepted = deliveries.filter(delivery => delivery.ok).length
  // A message nobody accepted must stay retryable.
  if (accepted === 0) ctx.pushDedupe.delete(key)
  return {
    delivered: accepted > 0,
    attempted: deliveries.length,
    accepted,
    ...(accepted === 0 ? { failure: 'all_failed' as const } : {}),
    deliveries,
  }
}

async function deliver(ctx: IdentityContext, subscription: PushSubscriptionRow, payload: Uint8Array): Promise<PushDelivery> {
  const base = { subscriptionId: subscription.id, deviceId: subscription.device_id }
  let url: URL
  try {
    url = validateEndpoint(ctx, subscription.endpoint)
  } catch {
    return { ...base, ok: false, reason: 'endpoint_not_allowed' }
  }
  let body: Uint8Array
  let authorization: string
  try {
    body = encryptWebPush({ payload, p256dh: Buffer.from(subscription.p256dh, 'base64url'), auth: Buffer.from(subscription.auth, 'base64url') })
    const jwt = createVapidJwt({
      privateKeyPem: ctx.vapid.privateKeyPem,
      audience: url.origin,
      subject: ctx.vapidSubject ?? '',
      expiresAtSeconds: Math.floor(ctx.clock.now() / 1000) + VAPID_LIFETIME_SECONDS,
    })
    authorization = vapidAuthorization(jwt, ctx.vapid.publicKey)
  } catch {
    return { ...base, ok: false, reason: 'encryption_failed' }
  }
  let status: number
  try {
    status = (
      await ctx.fetch(url.href, {
        method: 'POST',
        headers: { authorization, 'content-encoding': 'aes128gcm', 'content-type': 'application/octet-stream', ttl: '3600', urgency: 'normal' },
        body,
      })
    ).status
  } catch {
    return { ...base, ok: false, reason: 'network_error' }
  }
  if (status === 201 || status === 202) {
    ctx.store.run('UPDATE hub_push_subscriptions SET last_success_at = ? WHERE id = ?', ctx.clock.now(), subscription.id)
    return { ...base, ok: true, reason: 'accepted', status }
  }
  if (status === 404 || status === 410) {
    ctx.store.run('DELETE FROM hub_push_subscriptions WHERE id = ?', subscription.id)
    return { ...base, ok: false, reason: 'gone', status }
  }
  return { ...base, ok: false, reason: 'rejected', status }
}
