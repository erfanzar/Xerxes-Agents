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
  readonly http: Pick<HubHttp, 'json' | 'csrf' | 'refresh'>
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

export function createPushService(env: PushEnv): PushService {
  return {
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
        const vapid = await env.http.json('GET', HUB_HTTP.pushVapid)
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
      try {
        if (env.http.csrf() === undefined) await env.http.refresh()
        // The Hub takes the bare PushSubscription JSON: {endpoint, keys:{p256dh, auth}}.
        const json = subscription.toJSON()
        const saved = await env.http.json('POST', HUB_HTTP.pushSubscribe, typeof json === 'object' && json !== null ? (json as Record<string, unknown>) : {})
        if (saved.status < 200 || saved.status >= 300) {
          const nested = typeof saved.body.error === 'object' && saved.body.error !== null ? (saved.body.error as { message?: unknown }).message : undefined
          return failure(typeof nested === 'string' ? nested : `The Hub did not save the subscription (${saved.status}).`)
        }
      } catch (error) {
        return failure(error instanceof Error ? error.message : 'Could not reach the Hub.')
      }
      return { ok: true }
    },
  }
}
