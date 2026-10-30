// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubHttp } from '../api.js'
import { HUB_HTTP } from '../api.js'

/**
 * Web Push opt-in. It must be started from a user gesture (a tap): the
 * permission prompt is the first thing `enable` does, before any network
 * round trip, so the browser still counts the gesture. Notifications carry no
 * actions; they only open the task.
 */

export type PushState = 'unsupported' | 'blocked' | 'available' | 'enabled'
export type PushOutcome = { readonly ok: true } | { readonly ok: false; readonly message: string }

export interface PushService {
  state(): Promise<PushState>
  enable(): Promise<PushOutcome>
  /**
   * Re-register this browser's existing subscription with the Hub. A re-pair or a device revoke
   * deletes the Hub's copy while the browser still holds one, so the app would say notifications
   * are on while nothing arrives. Idempotent; never prompts. `skipped` when there is nothing to send.
   */
  resync(): Promise<PushOutcome | { readonly ok: 'skipped' }>
}

export interface PushSubscriptionLike {
  toJSON(): unknown
}

export interface PushRegistrationLike {
  readonly pushManager: {
    getSubscription(): Promise<PushSubscriptionLike | null>
    subscribe(options: { readonly userVisibleOnly: boolean; readonly applicationServerKey: BufferSource }): Promise<PushSubscriptionLike>
  }
}

export interface PushEnv {
  readonly http: Pick<HubHttp, 'json' | 'csrf' | 'refresh'> & Partial<Pick<HubHttp, 'ensureFresh'>>
  /** Undefined when the browser has no service worker or Push API. */
  readonly ready?: (() => Promise<PushRegistrationLike>) | undefined
  readonly permission: () => NotificationPermission | 'unsupported'
  readonly requestPermission: () => Promise<NotificationPermission>
}

/** Base64url (VAPID public key) to bytes. */
export function urlBase64ToBytes(value: string): Uint8Array<ArrayBuffer> {
  const padded = value.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(value.length / 4) * 4, '=')
  const binary = atob(padded)
  const bytes = new Uint8Array(new ArrayBuffer(binary.length))
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index)
  return bytes
}

function failure(message: string): PushOutcome {
  return { ok: false, message }
}

const EXPIRED = 'Your session with the Hub expired. Try again in a moment.'

function errorMessage(body: Readonly<Record<string, unknown>>): string | undefined {
  const nested = typeof body.error === 'object' && body.error !== null ? (body.error as { message?: unknown }).message : undefined
  return typeof nested === 'string' && nested.length > 0 ? nested : undefined
}

export function createPushService(env: PushEnv): PushService {
  /** A fresh browser session (and its CSRF token) before any authenticated push call. */
  async function freshSession(): Promise<boolean> {
    const outcome = env.http.ensureFresh ? await env.http.ensureFresh() : (env.http.csrf() === undefined ? await env.http.refresh() : { status: 'ok' as const })
    return outcome.status === 'ok'
  }

  async function save(subscription: PushSubscriptionLike): Promise<PushOutcome> {
    try {
      // The Hub takes the bare PushSubscription JSON: {endpoint, keys:{p256dh, auth}}.
      const json = subscription.toJSON()
      const saved = await env.http.json('POST', HUB_HTTP.pushSubscribe, typeof json === 'object' && json !== null ? (json as Record<string, unknown>) : {})
      if (saved.status === 401) return failure(EXPIRED)
      if (saved.status < 200 || saved.status >= 300) {
        return failure(errorMessage(saved.body) ?? `The Hub did not save the subscription (${saved.status}).`)
      }
    } catch (error) {
      return failure(error instanceof Error ? error.message : 'Could not reach the Hub.')
    }
    return { ok: true }
  }

  return {
    async resync() {
      if (env.ready === undefined || env.permission() !== 'granted') return { ok: 'skipped' }
      let subscription: PushSubscriptionLike | null
      try {
        subscription = await (await env.ready()).pushManager.getSubscription()
      } catch {
        return { ok: 'skipped' }
      }
      if (subscription === null) return { ok: 'skipped' }
      if (!await freshSession()) return failure(EXPIRED)
      return save(subscription)
    },
    async state() {
      if (env.ready === undefined || env.permission() === 'unsupported') return 'unsupported'
      if (env.permission() === 'denied') return 'blocked'
      if (env.permission() === 'granted') {
        try {
          const registration = await env.ready()
          return (await registration.pushManager.getSubscription()) ? 'enabled' : 'available'
        } catch {
          return 'available'
        }
      }
      return 'available'
    },
    async enable() {
      if (env.ready === undefined || env.permission() === 'unsupported') {
        return failure('This browser cannot receive notifications. On iPhone, add Xerxes to the Home Screen first (iOS 16.4 or later).')
      }
      // First await in the gesture: the permission prompt.
      const permission = await env.requestPermission()
      if (permission !== 'granted') {
        return failure(permission === 'denied' ? 'Notifications are blocked for Xerxes. Allow them in your browser or system settings.' : 'Notifications were not enabled.')
      }
      let key: string | undefined
      try {
        // The 5-minute browser session may have lapsed while the socket stayed up.
        if (!await freshSession()) return failure(EXPIRED)
        const vapid = await env.http.json('GET', HUB_HTTP.pushVapid)
        if (vapid.status === 401) return failure(EXPIRED)
        const body = vapid.body
        const candidate = body.public_key ?? body.key ?? body.vapid_public_key
        key = typeof candidate === 'string' && candidate.length > 0 ? candidate : undefined
        if (vapid.status < 200 || vapid.status >= 300 || key === undefined) {
          return failure('This Hub is not set up for push notifications.')
        }
      } catch (error) {
        return failure(error instanceof Error ? error.message : 'Could not reach the Hub.')
      }
      let subscription: PushSubscriptionLike
      try {
        const registration = await env.ready()
        subscription = (await registration.pushManager.getSubscription()) ?? await registration.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: urlBase64ToBytes(key),
        })
      } catch (error) {
        return failure(error instanceof Error ? error.message : 'The browser refused the push subscription.')
      }
      if (env.http.csrf() === undefined) await env.http.refresh()
      return save(subscription)
    },
  }
}
