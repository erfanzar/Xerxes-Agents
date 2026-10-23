// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The renderer's side of the Hub cockpit.
 *
 * The renderer never holds a Hub cookie, an SSH key or a pairing secret. It asks
 * the main process (through `window.xerxes.hubCockpit`) to sign in and to open
 * the Hub WebSocket, and the frames come back as IPC events. This file turns that
 * into the two things the shared client wants: a `SocketFactory` for
 * `HubConnection` and an `ensureSession` check.
 */

import type { SessionCheck, SocketFactory, SocketHandlers } from '../../client/hubConnection.js'
import type { CockpitAction, CockpitFailure, CockpitProgress, CockpitReplyMap, HubSocketEvent } from '../hubShared.js'
import type { XerxesBridge } from './types.js'

export interface CockpitApi {
  call(action: string, params?: Record<string, unknown>): Promise<unknown>
  onSocket(handler: (event: HubSocketEvent) => void): () => void
  onProgress(handler: (progress: CockpitProgress) => void): () => void
}

/** `undefined` outside the desktop app (for example the VS Code panel), which has no cockpit. */
export function cockpitApiFrom(bridge: XerxesBridge | undefined): CockpitApi | undefined {
  if (bridge?.hubCockpit === undefined || bridge.onHubSocket === undefined || bridge.onHubProgress === undefined) return undefined
  return {
    call: (action, params) => bridge.hubCockpit!(action, params),
    onSocket: handler => bridge.onHubSocket!(handler),
    onProgress: handler => bridge.onHubProgress!(handler),
  }
}

export function windowCockpitApi(): CockpitApi | undefined {
  return typeof window === 'undefined' ? undefined : cockpitApiFrom(window.xerxes)
}

export type CockpitReply<A extends CockpitAction> = ({ readonly ok: true } & CockpitReplyMap[A]) | CockpitFailure

function failure(code: CockpitFailure['code'], message: string): CockpitFailure {
  return { ok: false, code, message }
}

/** One cockpit call. A rejected IPC call or an unreadable reply becomes a failure value, never a throw. */
export async function cockpitCall<A extends CockpitAction>(api: CockpitApi, action: A, params: Record<string, unknown> = {}): Promise<CockpitReply<A>> {
  let value: unknown
  try {
    value = await api.call(action, params)
  } catch (error) {
    return failure('cli_failed', error instanceof Error ? error.message.slice(0, 300) : 'The desktop could not reach its Hub service.')
  }
  if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
    const item = value as Record<string, unknown>
    if (item.ok === true) return item as unknown as CockpitReply<A>
    if (item.ok === false && typeof item.code === 'string' && typeof item.message === 'string') return item as unknown as CockpitFailure
  }
  return failure('bad_output', 'The desktop answered with something this page cannot read.')
}

function randomSocketId(): string {
  const bytes = new Uint8Array(12)
  globalThis.crypto.getRandomValues(bytes)
  return `hs${Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')}`
}

export interface CockpitSockets {
  readonly factory: SocketFactory
  dispose(): void
}

/**
 * A `SocketFactory` whose sockets live in the main process. The URL argument is
 * ignored on purpose: main opens the Hub address it saved, with the Hub's own
 * `Origin` and cookies, so the renderer cannot point it elsewhere.
 */
export function createCockpitSockets(api: CockpitApi, newId: () => string = randomSocketId): CockpitSockets {
  const live = new Map<string, SocketHandlers>()
  let stop: (() => void) | undefined

  const finish = (id: string, code: number, reason: string): void => {
    const handlers = live.get(id)
    if (handlers === undefined) return
    live.delete(id)
    handlers.close(code, reason)
  }
  const route = (event: HubSocketEvent): void => {
    const handlers = live.get(event.id)
    if (handlers === undefined) return
    switch (event.type) {
      case 'open': handlers.open(); return
      case 'message': handlers.message(event.data); return
      case 'binary': handlers.binary(); return
      case 'error': handlers.error(); return
      case 'close': finish(event.id, event.code, event.reason); return
    }
  }

  const factory: SocketFactory = (_url, handlers) => {
    stop ??= api.onSocket(route)
    const id = newId()
    live.set(id, handlers)
    void cockpitCall(api, 'socket.open', { id }).then(result => {
      if (!result.ok) finish(id, 1006, result.message)
    })
    return {
      send: text => {
        void cockpitCall(api, 'socket.send', { id, data: text }).then(result => {
          if (!result.ok) finish(id, 1006, result.message)
        })
      },
      close: (code, reason) => {
        void cockpitCall(api, 'socket.close', { id, ...(code === undefined ? {} : { code }), ...(reason === undefined ? {} : { reason }) }).then(result => {
          if (!result.ok) finish(id, code ?? 1000, reason ?? '')
        })
      },
    }
  }

  return {
    factory,
    dispose() {
      stop?.()
      stop = undefined
      live.clear()
    },
  }
}

/**
 * Asked before every connect and reconnect. Main answers from its cookie jar,
 * refreshes the session, or pairs this desktop again; only a Hub that refuses
 * this desktop ends the connection as revoked.
 */
export function cockpitEnsureSession(api: CockpitApi): () => Promise<SessionCheck> {
  return async () => {
    const result = await cockpitCall(api, 'signin')
    if (result.ok) return 'ok'
    return result.denied === true ? 'denied' : 'unavailable'
  }
}

/** Fired by the Settings panel when the saved Hub changes, so an open XERXES tab starts over. */
export const HUB_CHANGED_EVENT = 'xerxes:hub-changed'
