// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ServerWebSocket } from 'bun'

import type { AuthenticatedContext, HubHttpRequestInfo, HubIdentityPort } from '../identityPort.js'
import { VoiceBudget } from './budgets.js'
import type { VoiceRuntimeConfig } from './config.js'
import { VOICE_LIMITS, VOICE_PATH } from './protocol.js'
import { VoiceSession, type VoiceTransport } from './session.js'
import { SpeakerLeases } from './speakerLease.js'
import { ExactlyOnceSubmitter, type SupervisorVoiceBridge } from './supervisorBridge.js'

/**
 * Mounts the voice endpoint into the Hub's `Bun.serve`.
 *
 * Integration contract for server.ts:
 *  1. On a GET to `route.path`, check the Origin exactly as the main Hub socket
 *     does, then `const context = await route.authenticateUpgrade(request, info)`.
 *     `null` means answer 401.
 *  2. `server.upgrade(request, { data: voiceSocketData(context) })`.
 *  3. Merge `route.websocket` handlers into the Bun websocket handlers for those
 *     sockets (dispatch on `ws.data.kind === 'voice'`), and use `route.limits`
 *     for `maxPayloadLength` / `backpressureLimit`.
 *  4. Call `route.close()` on shutdown.
 */

export interface VoiceSocketData {
  readonly kind: 'voice'
  readonly context: AuthenticatedContext
  session?: VoiceSession | undefined
}

export function voiceSocketData(context: AuthenticatedContext): VoiceSocketData {
  return { kind: 'voice', context }
}

export interface HubVoiceRouteOptions {
  readonly identity: HubIdentityPort
  readonly supervisor: SupervisorVoiceBridge
  readonly config: VoiceRuntimeConfig
  readonly clock?: () => number
  readonly maxConnections?: number
  readonly maxConnectionsPerPrincipal?: number
  readonly tickMs?: number
  readonly onError?: (error: unknown, context: string) => void
}

export interface HubVoiceRoute {
  readonly path: typeof VOICE_PATH
  readonly limits: { readonly maxPayloadLength: number; readonly backpressureLimit: number; readonly idleTimeout: number }
  readonly sessionCount: number
  authenticateUpgrade(request: Request, info: HubHttpRequestInfo): Promise<AuthenticatedContext | null>
  readonly websocket: {
    open(ws: ServerWebSocket<VoiceSocketData>): void
    message(ws: ServerWebSocket<VoiceSocketData>, message: string | Buffer): void
    close(ws: ServerWebSocket<VoiceSocketData>, code: number, reason: string): void
  }
  close(): void
}

export function createHubVoiceRoute(options: HubVoiceRouteOptions): HubVoiceRoute {
  const clock = options.clock ?? Date.now
  const { settings } = options.config
  const sessions = new Set<VoiceSession>()
  const budget = new VoiceBudget({
    clock,
    limits: {
      ...(settings.maxSessionUsd === undefined ? {} : { maxSessionUsd: settings.maxSessionUsd }),
      ...(settings.maxDailyUserUsd === undefined ? {} : { maxDailyUserUsd: settings.maxDailyUserUsd }),
    },
  })
  const leases = new SpeakerLeases({ ttlMs: VOICE_LIMITS.leaseTtlMs, clock })
  const submitter = new ExactlyOnceSubmitter(options.supervisor)
  const maxConnections = options.maxConnections ?? 64
  const maxPerPrincipal = options.maxConnectionsPerPrincipal ?? 4
  let closed = false

  const report = (error: unknown, context: string): void => options.onError?.(error, context)

  const offRevoked = options.identity.onRevoked(event => {
    for (const session of [...sessions]) {
      const matches = event.browserSessionId !== undefined
        ? session.browserSessionId === event.browserSessionId
        : session.principalId === event.principalId && (event.deviceId === undefined || session.deviceId === event.deviceId)
      if (matches) session.end('access revoked', 1008)
    }
  })

  const timer = setInterval(() => {
    for (const session of [...sessions]) {
      try { session.tick() } catch (error) { report(error, 'voice tick') }
    }
  }, options.tickMs ?? 1000)
  timer.unref?.()

  function transportFor(ws: ServerWebSocket<VoiceSocketData>): VoiceTransport {
    return {
      // Bun returns 0 when a frame was dropped and -1 when it was queued under backpressure.
      sendText: text => ws.readyState === 1 && ws.send(text) !== 0,
      sendBinary: bytes => ws.readyState === 1 && ws.send(bytes) !== 0,
      bufferedAmount: () => ws.getBufferedAmount(),
      close: (code, reason) => { try { ws.close(code, reason) } catch (error) { report(error, 'close voice socket') } },
    }
  }

  return {
    path: VOICE_PATH,
    limits: {
      maxPayloadLength: VOICE_LIMITS.headerBytes + VOICE_LIMITS.maxFramePayloadBytes + 1024,
      backpressureLimit: VOICE_LIMITS.maxBufferedBytes,
      idleTimeout: 120,
    },
    get sessionCount() { return sessions.size },

    async authenticateUpgrade(request, info) {
      if (closed) return null
      if (new URL(request.url).pathname !== VOICE_PATH) return null
      const context = await options.identity.authenticate(info)
      // Voice is for signed-in people only; an agent principal never speaks.
      if (context === null || context.principal.kind !== 'user') return null
      return context
    },

    websocket: {
      open(ws) {
        if (closed) { ws.close(1001, 'Hub shutting down'); return }
        const context = ws.data.context
        const own = [...sessions].filter(session => session.principalId === context.principal.id).length
        if (sessions.size >= maxConnections || own >= maxPerPrincipal) { ws.close(1013, 'too many voice connections'); return }
        const session = new VoiceSession({
          id: crypto.randomUUID(),
          context,
          config: options.config,
          supervisor: options.supervisor,
          submitter,
          leases,
          budget,
          clock,
          transport: transportFor(ws),
          onError: report,
        })
        ws.data.session = session
        sessions.add(session)
      },
      message(ws, message) {
        const session = ws.data.session
        if (session === undefined) return
        if (typeof message === 'string') {
          session.handleText(message).catch(error => report(error, 'voice message'))
        } else {
          session.handleBinary(message)
        }
      },
      close(ws) {
        const session = ws.data.session
        if (session === undefined) return
        sessions.delete(session)
        session.dispose()
        ws.data.session = undefined
      },
    },

    close() {
      if (closed) return
      closed = true
      clearInterval(timer)
      offRevoked()
      for (const session of [...sessions]) session.end('Hub shutting down', 1001)
      sessions.clear()
    },
  }
}
