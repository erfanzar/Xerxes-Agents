// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { realpath } from 'node:fs/promises'
import { basename, extname, resolve, sep } from 'node:path'
import type { ServerWebSocket } from 'bun'

import {
  canSeeWorkspace, effectiveScopes, grantedSessionIds, hasScope,
  type AccessTarget, type HubGrant, type HubScope,
} from '../security/principal.js'
import { cursorScopeKey, encodeCursor, resolveClientCursor, type CursorScope } from './cursor.js'
import { resolveSessionKey, type CommandBackend, type HubCommandRouter } from './commandRouter.js'
import { HubEventStoreError, type HubEventStore } from './eventStore.js'
import { originAllowed, type HubExposure } from './exposure.js'
import type { AuthenticatedContext, HubHttpRequestInfo, HubIdentityPort } from './identityPort.js'
import { aggregateUsage, type WorkspaceProjection } from './projections.js'
import {
  HUB_LIMITS, HUB_PROTOCOL_VERSION, HubProtocolError, encodeHubFrame, hubError, hubEvent, hubResponse,
  isRecord, parseHubRequestFrame, readId, readIdList, utf8ByteLength,
  type HubCommandResult, type HubEventType, type HubMethod, type HubRequestFrame, type HubResponseFrame, type HubServerFrame,
} from './protocol.js'
import type { PoolHealth } from './runtimePool.js'
import type { SubscriptionManager, SubscriptionSink } from './subscriptions.js'
import { voiceSocketData, type HubVoiceRoute, type VoiceSocketData } from './voice/route.js'

/**
 * Bun.serve front door of the Hub: HTTP (health, identity routes, static phone
 * bundle) and one authenticated WebSocket per client speaking Hub protocol v1.
 *
 * Authority comes only from `identity.authenticate`. Grants are re-read for
 * every request, so a revoked or narrowed grant takes effect immediately.
 * `X-Forwarded-*` headers are never consulted.
 */

export const HUB_WS_PATH = '/api/hub/v1/ws'
const HUB_TARGET: AccessTarget = Object.freeze({ workspaceId: '\u0000hub' })
const MAX_SUBSCRIPTIONS_PER_CONNECTION = 8
const MAX_INFLIGHT_PER_CONNECTION = 16
const MAX_PRESENCE_ENTRIES = 5000
const PRESENCE_STATES = ['viewing', 'typing', 'idle'] as const

export interface HubWorkspaceView {
  list(): readonly { readonly id: string; readonly label: string }[]
  health(workspaceId: string): PoolHealth | undefined
  projection(workspaceId: string): WorkspaceProjection | undefined
  /** True when the workspace's daemon cannot verify signed commands; omitted means writable. */
  readOnly?(workspaceId: string): boolean
}

export interface HubMethodContext {
  readonly context: AuthenticatedContext
  readonly grants: readonly HubGrant[]
}
/** Plug-in for methods owned by the identity or supervisor subsystems. */
export type HubMethodHandler = (context: HubMethodContext, params: Readonly<Record<string, unknown>>) => Record<string, unknown> | Promise<Record<string, unknown>>

export interface RateWindow { readonly limit: number; readonly windowMs: number }

export interface HubServerOptions {
  readonly hubId: string
  readonly exposure: HubExposure
  readonly store: HubEventStore
  readonly subscriptions: SubscriptionManager
  readonly workspaces: HubWorkspaceView
  readonly backend: Pick<CommandBackend, 'call'>
  /** Absent means fail closed: every request except GET /healthz is refused. */
  readonly identity?: HubIdentityPort
  readonly router?: HubCommandRouter
  /** Directory of the built phone bundle (index.html plus hashed assets). */
  readonly staticRoot?: string
  readonly clock?: () => number
  /** Outbound bytes a socket may have buffered before it is closed with 1013. */
  readonly maxBufferedBytes?: number
  readonly maxConnections?: number
  readonly maxConnectionsPerPrincipal?: number
  readonly rateLimit?: { readonly principal?: RateWindow; readonly api?: RateWindow; readonly assets?: RateWindow }
  readonly presenceTtlMs?: number
  readonly methodHandlers?: Readonly<Partial<Record<HubMethod, HubMethodHandler>>>
  /** Voice endpoint; absent means the voice path is not served (the phone hides its talk button). */
  readonly voice?: HubVoiceRoute
  readonly ready?: () => boolean
  readonly onError?: (error: unknown, context: string) => void
}

interface SubscriptionScope { readonly workspaceIds: readonly string[]; readonly sessionIds?: readonly string[] }
interface Conn {
  readonly kind: 'hub'
  readonly id: string
  readonly context: AuthenticatedContext
  readonly address: string
  ws: ServerWebSocket<Conn> | undefined
  readonly subs: Map<string, SubscriptionScope>
  inflight: number
  closed: boolean
}
/** One Bun.serve carries both the Hub protocol socket and the voice socket. */
type SocketData = Conn | VoiceSocketData
function isHubSocket(ws: ServerWebSocket<SocketData>): ws is ServerWebSocket<Conn> { return ws.data.kind === 'hub' }
function isVoiceSocket(ws: ServerWebSocket<SocketData>): ws is ServerWebSocket<VoiceSocketData> { return ws.data.kind === 'voice' }
interface Handled { readonly result: Record<string, unknown>; readonly after?: () => void }
interface PresenceEntry {
  readonly principalKind: 'user' | 'agent'
  readonly principalId: string
  readonly displayName: string
  readonly state: string
  readonly updatedAt: number
  readonly expiresAt: number
  readonly connId: string
}

class SlidingWindow {
  private readonly hits = new Map<string, number[]>()
  allow(key: string, window: RateWindow, now: number): boolean {
    const fresh = (this.hits.get(key) ?? []).filter(time => now - time < window.windowMs)
    if (fresh.length >= window.limit) { this.hits.set(key, fresh); return false }
    fresh.push(now)
    this.hits.set(key, fresh)
    if (this.hits.size > 5000) {
      for (const [other, times] of this.hits) {
        const newest = times.at(-1)
        if (newest === undefined || now - newest >= window.windowMs) this.hits.delete(other)
      }
    }
    return true
  }
}

function json(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json; charset=utf-8', ...headers } })
}

/** Files whose name carries a content hash can be cached forever; nothing else can. */
function isHashedAsset(name: string): boolean {
  return /[-.][A-Za-z0-9_]{8,}\.(?:js|mjs|css|woff2?|png|jpe?g|svg|webp|ico|map)$/.test(name)
}

export class HubServer {
  private server: Bun.Server<SocketData> | undefined
  private readonly conns = new Set<Conn>()
  /** Open voice sockets, tracked so shutdown waits for their close frames too. */
  private readonly voiceSockets = new Set<ServerWebSocket<VoiceSocketData>>()
  private readonly presence = new Map<string, Map<string, PresenceEntry>>()
  private readonly limiter = new SlidingWindow()
  private readonly clock: () => number
  private readonly maxBufferedBytes: number
  private readonly presenceTtlMs: number
  private sweepTimer: ReturnType<typeof setInterval> | undefined
  private offRevoked: (() => void) | undefined
  private allClosed: (() => void) | undefined
  private stopping = false

  constructor(private readonly options: HubServerOptions) {
    this.clock = options.clock ?? Date.now
    this.maxBufferedBytes = options.maxBufferedBytes ?? HUB_LIMITS.maxFrameBytes
    this.presenceTtlMs = options.presenceTtlMs ?? 30_000
    if (!Number.isSafeInteger(this.maxBufferedBytes) || this.maxBufferedBytes < 1) throw new RangeError('maxBufferedBytes must be positive')
    if (!Number.isSafeInteger(this.presenceTtlMs) || this.presenceTtlMs < 1) throw new RangeError('presenceTtlMs must be positive')
  }

  get port(): number {
    if (!this.server?.port) throw new Error('Hub server is not listening')
    return this.server.port
  }
  get connectionCount(): number { return this.conns.size }
  get identityConfigured(): boolean { return this.options.identity !== undefined }
  get ready(): boolean { return this.server !== undefined && !this.stopping && this.identityConfigured && (this.options.ready?.() ?? true) }
  get allowedOrigins(): readonly string[] { return this.options.exposure.allowedOrigins(this.port) }

  start(): { readonly port: number; readonly hostname: string; readonly origins: readonly string[] } {
    if (this.server) throw new Error('Hub server already started')
    const exposure = this.options.exposure
    const voice = this.options.voice
    this.server = Bun.serve<SocketData>({
      hostname: exposure.hostname,
      port: exposure.port,
      ...(exposure.tls === undefined ? {} : { tls: { cert: exposure.tls.cert, key: exposure.tls.key } }),
      fetch: (request, server) => this.fetch(request, server),
      websocket: {
        open: ws => {
          if (isVoiceSocket(ws)) {
            if (this.stopping) { ws.close(1001, 'Hub shutting down'); return }
            this.voiceSockets.add(ws)
            voice?.websocket.open(ws)
          } else if (isHubSocket(ws)) this.onOpen(ws)
        },
        message: (ws, message) => {
          if (isVoiceSocket(ws)) voice?.websocket.message(ws, message)
          else if (isHubSocket(ws)) this.onMessage(ws.data, message)
        },
        close: (ws, code, reason) => {
          if (isVoiceSocket(ws)) {
            this.voiceSockets.delete(ws)
            voice?.websocket.close(ws, code, reason)
            this.checkDrained()
          } else if (isHubSocket(ws)) this.onClose(ws.data)
        },
        // Frames over the Hub cap are answered with frame_too_large rather than a reset;
        // the voice session enforces its own (smaller) per-frame bound.
        maxPayloadLength: Math.max(HUB_LIMITS.maxFrameBytes * 2, voice?.limits.maxPayloadLength ?? 0),
        idleTimeout: 120,
        sendPings: true,
      },
    })
    this.offRevoked = this.options.identity?.onRevoked(event => this.revoke(event))
    this.sweepTimer = setInterval(() => this.sweepPresence(), Math.max(1000, Math.floor(this.presenceTtlMs / 3)))
    this.sweepTimer.unref?.()
    return { port: this.port, hostname: exposure.hostname, origins: this.allowedOrigins }
  }

  async stop(): Promise<void> {
    const server = this.server
    if (!server || this.stopping) return
    this.stopping = true
    clearInterval(this.sweepTimer)
    this.offRevoked?.()
    try { this.options.voice?.close() } catch (error) { this.options.onError?.(error, 'close voice route') }
    const drained = new Promise<void>(resolveDrained => { this.allClosed = resolveDrained })
    this.checkDrained()
    for (const conn of this.conns) {
      try { conn.ws?.close(1001, 'Hub shutting down') } catch (error) { this.options.onError?.(error, 'close socket') }
    }
    // Let close frames reach peers, but never let a stalled peer block shutdown.
    await Promise.race([drained, Bun.sleep(250)])
    await Promise.race([server.stop(true).catch(() => undefined), Bun.sleep(250)])
    this.server = undefined
  }

  // ---------------------------------------------------------------- HTTP

  private baseHeaders(): Record<string, string> {
    const connect = ["'self'", ...this.allowedOrigins.map(origin => origin.replace(/^http/, 'ws'))].join(' ')
    return {
      'content-security-policy': [
        "default-src 'self'", "script-src 'self'", "style-src 'self'", "img-src 'self' data:", "font-src 'self'",
        `connect-src ${connect}`, "manifest-src 'self'", "worker-src 'self'", "object-src 'none'",
        "base-uri 'none'", "form-action 'self'", "frame-ancestors 'none'",
      ].join('; '),
      'x-content-type-options': 'nosniff',
      'referrer-policy': 'no-referrer',
      'x-frame-options': 'DENY',
      'cross-origin-opener-policy': 'same-origin',
    }
  }

  private secure(response: Response, cache = 'no-store'): Response {
    const headers = new Headers(response.headers)
    for (const [name, value] of Object.entries(this.baseHeaders())) headers.set(name, value)
    if (!headers.has('cache-control')) headers.set('cache-control', cache)
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers })
  }

  private httpInfo(request: Request, address: string): HubHttpRequestInfo {
    const cookie = request.headers.get('cookie')
    const origin = request.headers.get('origin')
    const host = request.headers.get('host')
    const csrf = request.headers.get('x-xerxes-csrf')
    return {
      method: request.method,
      url: request.url,
      ...(cookie === null ? {} : { cookie }),
      ...(origin === null ? {} : { origin }),
      ...(host === null ? {} : { host }),
      ...(csrf === null ? {} : { csrfToken: csrf }),
      remoteAddress: address,
    }
  }

  private healthz(request: Request): Response {
    if (request.method !== 'GET' && request.method !== 'HEAD') return this.secure(json(405, { error: 'method_not_allowed' }, { allow: 'GET, HEAD' }))
    const rows = this.options.workspaces.list()
    let online = 0
    let degraded = 0
    for (const row of rows) {
      const health = this.options.workspaces.health(row.id)
      if (health?.state === 'online') online += 1
      if (health === undefined || health.state !== 'online' || health.degraded) degraded += 1
    }
    const body = {
      status: this.identityConfigured ? (degraded > 0 ? 'degraded' : 'ok') : 'identity_unavailable',
      live: true,
      ready: this.ready,
      identity: this.identityConfigured ? 'configured' : 'missing',
      workspaces: { total: rows.length, online, degraded },
    }
    return this.secure(request.method === 'HEAD' ? new Response(null, { status: 200 }) : json(200, body))
  }

  private async fetch(request: Request, server: Bun.Server<SocketData>): Promise<Response | undefined> {
    try {
      const url = new URL(request.url)
      const path = url.pathname
      if (path === '/healthz') return this.healthz(request)
      const identity = this.options.identity
      if (identity === undefined) {
        return this.secure(json(503, { error: 'identity_unavailable', message: 'The Hub has no identity subsystem; it refuses every request except /healthz.' }))
      }
      if (this.stopping) return this.secure(json(503, { error: 'shutting_down' }))
      const address = server.requestIP(request)?.address ?? 'unknown'
      const api = path.startsWith('/api/')
      const limits = this.options.rateLimit
      const window = api ? (limits?.api ?? { limit: 240, windowMs: 60_000 }) : (limits?.assets ?? { limit: 1200, windowMs: 60_000 })
      if (!this.limiter.allow(`${api ? 'api' : 'asset'}:${address}`, window, this.clock())) {
        return this.secure(json(429, { error: 'rate_limited' }, { 'retry-after': String(Math.ceil(window.windowMs / 1000)) }))
      }
      const info = this.httpInfo(request, address)
      if (path === HUB_WS_PATH) return await this.upgrade(request, server, info, address)
      if (this.options.voice !== undefined && path === this.options.voice.path) return await this.upgradeVoice(request, server, info, this.options.voice)
      const handled = await identity.handleHttp(request, info)
      if (handled !== undefined) return this.secure(handled)
      if (api) return this.secure(json(404, { error: 'not_found' }))
      return await this.serveStatic(request, path)
    } catch (error) {
      this.options.onError?.(error, 'http')
      return this.secure(json(500, { error: 'internal' }))
    }
  }

  private async upgradeVoice(request: Request, server: Bun.Server<SocketData>, info: HubHttpRequestInfo, voice: HubVoiceRoute): Promise<Response | undefined> {
    if (request.method !== 'GET' || request.headers.get('upgrade')?.toLowerCase() !== 'websocket') {
      return this.secure(json(426, { error: 'websocket_required' }, { upgrade: 'websocket' }))
    }
    if (!originAllowed(request.headers.get('origin'), this.allowedOrigins)) return this.secure(json(403, { error: 'origin_not_allowed' }))
    let context: AuthenticatedContext | null
    try { context = await voice.authenticateUpgrade(request, info) } catch (error) {
      this.options.onError?.(error, 'authenticate voice')
      return this.secure(json(503, { error: 'authentication_unavailable' }))
    }
    if (context === null) return this.secure(json(401, { error: 'unauthorized' }))
    if (!server.upgrade(request, { data: voiceSocketData(context) })) return this.secure(json(400, { error: 'upgrade_failed' }))
    return undefined
  }

  private async upgrade(request: Request, server: Bun.Server<SocketData>, info: HubHttpRequestInfo, address: string): Promise<Response | undefined> {
    const identity = this.options.identity!
    if (request.method !== 'GET' || request.headers.get('upgrade')?.toLowerCase() !== 'websocket') {
      return this.secure(json(426, { error: 'websocket_required' }, { upgrade: 'websocket' }))
    }
    if (!originAllowed(request.headers.get('origin'), this.allowedOrigins)) return this.secure(json(403, { error: 'origin_not_allowed' }))
    let context: AuthenticatedContext | null
    try { context = await identity.authenticate(info) } catch (error) {
      this.options.onError?.(error, 'authenticate')
      return this.secure(json(503, { error: 'authentication_unavailable' }))
    }
    if (context === null) return this.secure(json(401, { error: 'unauthorized' }))
    const perPrincipal = [...this.conns].filter(conn => conn.context.principal.id === context!.principal.id).length
    if (this.conns.size >= (this.options.maxConnections ?? 512) || perPrincipal >= (this.options.maxConnectionsPerPrincipal ?? 16)) {
      return this.secure(json(429, { error: 'too_many_connections' }, { 'retry-after': '5' }))
    }
    const conn: Conn = { kind: 'hub', id: crypto.randomUUID(), context, address, ws: undefined, subs: new Map(), inflight: 0, closed: false }
    if (!server.upgrade(request, { data: conn })) return this.secure(json(400, { error: 'upgrade_failed' }))
    return undefined
  }

  private async serveStatic(request: Request, path: string): Promise<Response> {
    if (request.method !== 'GET' && request.method !== 'HEAD') return this.secure(json(405, { error: 'method_not_allowed' }, { allow: 'GET, HEAD' }))
    const root = this.options.staticRoot
    if (root === undefined) return this.secure(json(404, { error: 'not_found', message: 'The phone bundle is not installed on this Hub.' }))
    let decoded: string
    try { decoded = decodeURIComponent(path) } catch { return this.secure(json(400, { error: 'bad_path' })) }
    if (decoded.includes('\0') || decoded.includes('\\') || decoded.split('/').includes('..')) return this.secure(json(400, { error: 'bad_path' }))
    const realRoot = await realpath(resolve(root)).catch(() => undefined)
    if (realRoot === undefined) return this.secure(json(404, { error: 'not_found', message: 'The phone bundle is not installed on this Hub.' }))
    const wanted = decoded === '/' ? '/index.html' : decoded
    let file = await this.insideRoot(realRoot, wanted)
    if (file === undefined) {
      // App routes (no extension, or an HTML navigation) fall back to the shell; missing assets stay 404.
      const navigation = extname(wanted) === '' || (request.headers.get('accept') ?? '').includes('text/html')
      if (!navigation || extname(wanted) !== '' && !(request.headers.get('accept') ?? '').includes('text/html')) {
        return this.secure(json(404, { error: 'not_found' }))
      }
      file = await this.insideRoot(realRoot, '/index.html')
      if (file === undefined) return this.secure(json(404, { error: 'not_found', message: 'The phone bundle has no index.html.' }))
    }
    const name = basename(file)
    const cache = name === 'index.html' ? 'no-store' : isHashedAsset(name) ? 'public, max-age=31536000, immutable' : 'no-cache'
    const bun = Bun.file(file)
    const headers: Record<string, string> = { 'content-type': bun.type, 'cache-control': cache }
    return this.secure(request.method === 'HEAD' ? new Response(null, { status: 200, headers }) : new Response(bun, { status: 200, headers }), cache)
  }

  private async insideRoot(realRoot: string, requested: string): Promise<string | undefined> {
    const candidate = resolve(realRoot, `.${requested}`)
    if (candidate !== realRoot && !candidate.startsWith(realRoot + sep)) return undefined
    let real: string
    try { real = await realpath(candidate) } catch { return undefined }
    if (!real.startsWith(realRoot + sep)) return undefined
    const file = Bun.file(real)
    return (await file.exists()) && file.size >= 0 && !(await isDirectory(real)) ? real : undefined
  }

  // ----------------------------------------------------------- WebSocket

  private onOpen(ws: ServerWebSocket<Conn>): void {
    const conn = ws.data
    conn.ws = ws
    if (this.stopping) { ws.close(1001, 'Hub shutting down'); return }
    this.conns.add(conn)
  }

  private onClose(conn: Conn): void {
    conn.closed = true
    for (const id of conn.subs.keys()) {
      try { this.options.subscriptions.unsubscribe(id) } catch (error) { this.options.onError?.(error, 'unsubscribe on close') }
    }
    conn.subs.clear()
    this.conns.delete(conn)
    const touched: string[] = []
    for (const [target, entries] of this.presence) {
      for (const [key, entry] of entries) if (entry.connId === conn.id) { entries.delete(key); if (!touched.includes(target)) touched.push(target) }
      if (entries.size === 0) this.presence.delete(target)
    }
    for (const target of touched) { const [workspaceId, sessionId] = target.split('\0'); if (workspaceId && sessionId) this.broadcastPresence(workspaceId, sessionId) }
    this.checkDrained()
  }

  /** During shutdown, resolve once every Hub and voice socket has closed. */
  private checkDrained(): void {
    if (this.stopping && this.conns.size === 0 && this.voiceSockets.size === 0) this.allClosed?.()
  }

  private sendFrame(conn: Conn, frame: HubServerFrame): boolean {
    const ws = conn.ws
    if (!ws || conn.closed || ws.readyState !== 1) return false
    let text: string
    try { text = encodeHubFrame(frame) } catch (error) {
      if (frame.kind !== 'response') { this.options.onError?.(error, 'event frame'); return false }
      text = encodeHubFrame(hubError(frame.id, 'frame_too_large', 'the response exceeds the Hub frame limit'))
    }
    if (ws.getBufferedAmount() + utf8ByteLength(text) > this.maxBufferedBytes) {
      ws.close(1013, 'slow client; resume from the last applied cursor')
      return false
    }
    if (ws.send(text) === 0) { ws.close(1013, 'slow client; resume from the last applied cursor'); return false }
    return true
  }

  private grants(conn: Conn): readonly HubGrant[] {
    return this.options.identity?.grantsFor(conn.context.principal.id) ?? []
  }

  private onMessage(conn: Conn, raw: string | Buffer): void {
    if (typeof raw !== 'string') { conn.ws?.close(1003, 'text frames only'); return }
    let frame: HubRequestFrame
    try { frame = parseHubRequestFrame(raw) } catch (error) {
      if (error instanceof HubProtocolError) this.sendFrame(conn, hubError(error.requestId, error.code, error.message))
      else this.sendFrame(conn, hubError(null, 'invalid_request', 'unreadable frame'))
      return
    }
    const window = this.options.rateLimit?.principal ?? { limit: 240, windowMs: 10_000 }
    if (!this.limiter.allow(`principal:${conn.context.principal.id}`, window, this.clock())) {
      this.sendFrame(conn, hubError(frame.id, 'rate_limited', 'too many requests; slow down'))
      return
    }
    if (conn.inflight >= MAX_INFLIGHT_PER_CONNECTION) {
      this.sendFrame(conn, hubError(frame.id, 'rate_limited', 'too many requests in flight'))
      return
    }
    let outcome: Handled | Promise<Handled>
    try { outcome = this.handle(conn, frame) } catch (error) { this.sendFrame(conn, this.errorFrame(frame.id, error)); return }
    if (outcome instanceof Promise) {
      conn.inflight += 1
      outcome.then(
        handled => this.deliver(conn, frame.id, handled),
        (error: unknown) => { this.sendFrame(conn, this.errorFrame(frame.id, error)) },
      ).finally(() => { conn.inflight -= 1 })
    } else {
      // Synchronous methods answer before anything they scheduled (subscribe's first batch).
      this.deliver(conn, frame.id, outcome)
    }
  }

  private deliver(conn: Conn, id: string, handled: Handled): void {
    if (this.sendFrame(conn, hubResponse(id, handled.result))) handled.after?.()
  }

  private errorFrame(id: string, error: unknown): HubResponseFrame {
    if (error instanceof HubProtocolError) return hubError(id, error.code, error.message)
    if (error instanceof HubEventStoreError) {
      this.options.onError?.(error, 'event store')
      return hubError(id, 'unavailable', 'the event journal is unavailable')
    }
    this.options.onError?.(error, 'request')
    return hubError(id, 'internal', 'internal error')
  }

  private access(conn: Conn, target: AccessTarget, scope: HubScope): boolean {
    return hasScope(conn.context.principal, this.grants(conn), target, scope, this.clock())
  }

  private require(conn: Conn, target: AccessTarget, scope: HubScope): void {
    if (!this.access(conn, target, scope)) throw new HubProtocolError('forbidden', `not permitted: ${scope} is required`)
  }

  private handle(conn: Conn, frame: HubRequestFrame): Handled | Promise<Handled> {
    const params = frame.params
    const custom = this.options.methodHandlers?.[frame.method]
    if (custom !== undefined) {
      return Promise.resolve(custom({ context: conn.context, grants: this.grants(conn) }, params)).then(result => ({ result }))
    }
    switch (frame.method) {
      case 'hub.hello': return { result: this.hello(conn) }
      case 'hub.workspaces.list': return { result: this.listWorkspaces(conn) }
      case 'hub.snapshot': return { result: this.snapshot(conn, params) }
      case 'hub.events.subscribe': return { result: this.subscribe(conn, params) }
      case 'hub.events.read': return { result: this.readEvents(conn, params) }
      case 'hub.events.ack': {
        this.exactKeys(params, ['subscription_id', 'cursor'])
        const id = this.ownSubscription(conn, params.subscription_id)
        if (typeof params.cursor !== 'string') throw new HubProtocolError('invalid_request', 'cursor must be a string')
        this.options.subscriptions.ack(id, params.cursor)
        return { result: {} }
      }
      case 'hub.events.unsubscribe': {
        this.exactKeys(params, ['subscription_id'])
        const id = this.ownSubscription(conn, params.subscription_id)
        this.options.subscriptions.unsubscribe(id)
        conn.subs.delete(id)
        return { result: {} }
      }
      case 'hub.history': return this.history(conn, params).then(result => ({ result }))
      case 'hub.usage': return this.usage(conn, params).then(result => ({ result }))
      case 'hub.command': return this.command(conn, params)
      case 'hub.command.status': {
        if (!this.options.router) throw new HubProtocolError('unsupported', 'commands are not available on this Hub')
        return { result: { ...this.options.router.status(conn.context, params) } }
      }
      case 'hub.presence.update': return this.updatePresence(conn, params)
      default:
        throw new HubProtocolError('unsupported', `${frame.method} is not available on this endpoint`)
    }
  }

  private exactKeys(params: Readonly<Record<string, unknown>>, allowed: readonly string[]): void {
    for (const key of Object.keys(params)) {
      if (!allowed.includes(key)) throw new HubProtocolError('invalid_request', `unknown parameter ${JSON.stringify(key.slice(0, 40))}`)
    }
  }

  private ownSubscription(conn: Conn, value: unknown): string {
    const id = readId(value, 'subscription_id')
    if (!conn.subs.has(id)) throw new HubProtocolError('not_found', 'subscription not found')
    return id
  }

  // ------------------------------------------------------------- methods

  private hello(conn: Conn): Record<string, unknown> {
    const principal = conn.context.principal
    const scopes = [...effectiveScopes(principal, this.grants(conn), HUB_TARGET, this.clock())].sort()
    return {
      protocol_version: HUB_PROTOCOL_VERSION,
      hub_id: this.options.hubId,
      stream_id: this.options.store.streamId,
      principal: {
        kind: principal.kind,
        id: principal.id,
        display_name: principal.displayName,
        ...(principal.kind === 'user' ? { device_id: principal.deviceId, hub_owner: principal.hubOwner } : {}),
      },
      hub_scopes: scopes,
      limits: { max_frame_bytes: HUB_LIMITS.maxFrameBytes, normal_frame_bytes: HUB_LIMITS.normalFrameBytes, max_batch_events: HUB_LIMITS.maxBatchEvents },
      presence_ttl_ms: this.presenceTtlMs,
      server_time: new Date(this.clock()).toISOString(),
    }
  }

  /** Workspace ids the caller may see, with whether access is workspace-wide or limited to named sessions. */
  private visibleWorkspaces(conn: Conn): { id: string; label: string; broad: boolean; sessionIds: string[] }[] {
    const grants = this.grants(conn)
    const principal = conn.context.principal
    const now = this.clock()
    const rows: { id: string; label: string; broad: boolean; sessionIds: string[] }[] = []
    for (const workspace of this.options.workspaces.list()) {
      const broad = canSeeWorkspace(principal, grants, workspace.id, now)
      const sessionIds = broad ? [] : grantedSessionIds(principal, grants, workspace.id, now)
      if (broad || sessionIds.length > 0) rows.push({ id: workspace.id, label: workspace.label, broad, sessionIds })
    }
    return rows
  }

  private listWorkspaces(conn: Conn): Record<string, unknown> {
    const grants = this.grants(conn)
    const now = this.clock()
    const workspaces = this.visibleWorkspaces(conn).map(row => {
      const health = this.options.workspaces.health(row.id)
      return {
        id: row.id,
        label: row.label,
        health: health?.state ?? 'offline',
        degraded: health?.degraded ?? false,
        read_only: health?.state === 'online' && this.options.workspaces.readOnly?.(row.id) === true,
        access: row.broad ? 'workspace' : 'sessions',
        ...(row.broad
          ? { scopes: [...effectiveScopes(conn.context.principal, grants, { workspaceId: row.id }, now)].sort() }
          : { session_ids: row.sessionIds }),
      }
    })
    return { workspaces }
  }

  private snapshot(conn: Conn, params: Readonly<Record<string, unknown>>): Record<string, unknown> {
    this.exactKeys(params, ['workspace_ids', 'limit'])
    const requested = readIdList(params.workspace_ids, 'workspace_ids')
    let limit = 100
    if (params.limit !== undefined) {
      if (typeof params.limit !== 'number' || !Number.isInteger(params.limit) || params.limit < 1 || params.limit > 200) {
        throw new HubProtocolError('invalid_request', 'limit must be an integer from 1 to 200')
      }
      limit = params.limit
    }
    const visible = this.visibleWorkspaces(conn).filter(row => requested.length === 0 || requested.includes(row.id))
    if (requested.length > 0 && visible.length === 0) throw new HubProtocolError('forbidden', 'no readable workspaces in the request')
    const build = (cap: number): { workspaces: unknown[]; truncated: boolean } => {
      let truncated = false
      const workspaces = visible.map(row => {
        const projection = this.options.workspaces.projection(row.id)
        const all = Object.values(projection?.sessions ?? {})
          .filter(session => row.broad || row.sessionIds.includes(session.id))
          .sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at))
        if (all.length > cap) truncated = true
        const health = this.options.workspaces.health(row.id)
        return { id: row.id, label: row.label, health: health?.state ?? 'offline', degraded: health?.degraded ?? false, sessions: all.slice(0, cap) }
      })
      return { workspaces, truncated }
    }
    let cap = limit
    let built = build(cap)
    // Keep the snapshot inside one normal frame by shrinking the per-workspace session cap.
    while (utf8ByteLength(JSON.stringify(built)) > HUB_LIMITS.normalFrameBytes && cap > 1) {
      cap = Math.max(1, Math.floor(cap / 2))
      built = build(cap)
    }
    return { workspaces: built.workspaces, truncated: built.truncated || cap < limit, generated_at: new Date(this.clock()).toISOString(), watermark: encodeCursor({ streamId: this.options.store.streamId, sequence: this.options.store.watermark(), scopeKey: 'snapshot' }) }
  }

  private parseScope(params: Readonly<Record<string, unknown>>): CursorScope {
    const workspaceIds = readIdList(params.workspace_ids, 'workspace_ids', { required: true })
    if (workspaceIds.length === 0) throw new HubProtocolError('invalid_request', 'at least one workspace is required')
    const sessionIds = params.session_ids === undefined ? undefined : readIdList(params.session_ids, 'session_ids')
    return { workspaceIds, ...(sessionIds === undefined ? {} : { sessionIds }) }
  }

  private assertReadable(conn: Conn, scope: CursorScope): void {
    const grants = this.grants(conn)
    const principal = conn.context.principal
    const now = this.clock()
    for (const workspaceId of scope.workspaceIds) {
      const broad = canSeeWorkspace(principal, grants, workspaceId, now) && hasScope(principal, grants, { workspaceId }, 'session.read', now)
      if (broad) continue
      const allowed = grantedSessionIds(principal, grants, workspaceId, now)
      if (allowed.length === 0 || (scope.sessionIds !== undefined && !scope.sessionIds.some(id => allowed.includes(id)))) {
        throw new HubProtocolError('forbidden', 'no readable sessions in the requested workspaces')
      }
    }
  }

  private subscribe(conn: Conn, params: Readonly<Record<string, unknown>>): Record<string, unknown> {
    this.exactKeys(params, ['workspace_ids', 'session_ids', 'after_cursor'])
    const scope = this.parseScope(params)
    if (params.after_cursor !== undefined && typeof params.after_cursor !== 'string') throw new HubProtocolError('invalid_request', 'after_cursor must be a string')
    if (conn.subs.size >= MAX_SUBSCRIPTIONS_PER_CONNECTION) throw new HubProtocolError('rate_limited', `at most ${MAX_SUBSCRIPTIONS_PER_CONNECTION} subscriptions per connection`)
    const sink: SubscriptionSink = {
      send: frame => { this.sendFrame(conn, frame) },
      close: (code, reason) => { try { conn.ws?.close(code, reason) } catch (error) { this.options.onError?.(error, 'close subscription') } },
    }
    const result = this.options.subscriptions.subscribe(conn.context.principal, this.grants(conn), {
      workspaceIds: scope.workspaceIds,
      ...(scope.sessionIds === undefined ? {} : { sessionIds: scope.sessionIds }),
      ...(params.after_cursor === undefined ? {} : { afterCursor: params.after_cursor }),
    }, sink)
    conn.subs.set(result.subscriptionId, scope)
    return { subscription_id: result.subscriptionId, watermark: result.watermark, oldest_cursor: result.oldest, reset_required: result.resetRequired }
  }

  private readEvents(conn: Conn, params: Readonly<Record<string, unknown>>): Record<string, unknown> {
    this.exactKeys(params, ['workspace_ids', 'session_ids', 'after_cursor', 'limit'])
    const scope = this.parseScope(params)
    this.assertReadable(conn, scope)
    if (params.after_cursor !== undefined && typeof params.after_cursor !== 'string') throw new HubProtocolError('invalid_request', 'after_cursor must be a string')
    let limit = 100
    if (params.limit !== undefined) {
      if (typeof params.limit !== 'number' || !Number.isInteger(params.limit) || params.limit < 1 || params.limit > HUB_LIMITS.maxBatchEvents) {
        throw new HubProtocolError('invalid_request', `limit must be an integer from 1 to ${HUB_LIMITS.maxBatchEvents}`)
      }
      limit = params.limit
    }
    const store = this.options.store
    const scopeKey = cursorScopeKey(scope)
    const cursor = (sequence: number): string => encodeCursor({ streamId: store.streamId, sequence, scopeKey })
    const resolved = resolveClientCursor(params.after_cursor, { streamId: store.streamId, scopeKey })
    const watermark = store.watermark()
    if (resolved.kind === 'stream_replaced') {
      return { events: [], through_cursor: cursor(watermark), has_more: false, gap: 'stream_replaced', watermark: cursor(watermark) }
    }
    const after = resolved.kind === 'after' ? resolved.sequence : Math.max(0, store.oldestSequence() - 1)
    if (after > watermark) throw new HubProtocolError('invalid_request', 'cursor is ahead of this Hub stream')
    const page = store.read(after, scope, limit, HUB_LIMITS.normalFrameBytes - 1024)
    if (page.gap) {
      return { events: [], through_cursor: cursor(watermark), has_more: false, gap: 'retention', oldest_cursor: cursor(page.oldestSequence - 1), watermark: cursor(watermark) }
    }
    const grants = this.grants(conn)
    const now = this.clock()
    const events = page.events.filter(event => hasScope(conn.context.principal, grants, {
      workspaceId: event.workspace_id, ...(event.session_id === undefined ? {} : { sessionId: event.session_id }),
    }, 'session.read', now))
    return { events, through_cursor: cursor(page.throughSequence), has_more: page.hasMore, watermark: cursor(page.watermark) }
  }

  private async history(conn: Conn, params: Readonly<Record<string, unknown>>): Promise<Record<string, unknown>> {
    this.exactKeys(params, ['workspace_id', 'session_id', 'before', 'limit'])
    const workspaceId = readId(params.workspace_id, 'workspace_id')
    const sessionId = readId(params.session_id, 'session_id')
    this.require(conn, { workspaceId, sessionId }, 'session.read')
    if (!this.options.workspaces.list().some(row => row.id === workspaceId)) throw new HubProtocolError('not_found', 'unknown workspace')
    let limit = 10
    if (params.limit !== undefined) {
      if (typeof params.limit !== 'number' || !Number.isInteger(params.limit) || params.limit < 1 || params.limit > 25) {
        throw new HubProtocolError('invalid_request', 'limit must be an integer from 1 to 25')
      }
      limit = params.limit
    }
    const before = params.before
    if (before !== undefined && !(typeof before === 'number' && Number.isFinite(before)) && !(typeof before === 'string' && before.length > 0 && before.length <= HUB_LIMITS.maxIdLength)) {
      throw new HubProtocolError('invalid_request', 'before must be a short string or number')
    }
    const key = await resolveSessionKey(this.options.backend, workspaceId, sessionId)
    if (key === undefined) throw new HubProtocolError('not_found', 'the session is not active in this workspace')
    // Shrink the page rather than ever sending an oversized frame.
    for (;;) {
      const outcome = await this.options.backend.call(workspaceId, 'session.history', { session_key: key, history_limit: limit, ...(before === undefined ? {} : { before }) })
      if (outcome.state !== 'accepted') throw new HubProtocolError('unavailable', outcome.error.message)
      const result = { history: outcome.result.history ?? null, session_id: sessionId }
      if (utf8ByteLength(JSON.stringify(result)) <= HUB_LIMITS.normalFrameBytes) return result
      if (limit === 1) throw new HubProtocolError('frame_too_large', 'a single history page exceeds the Hub frame budget')
      limit = Math.max(1, Math.floor(limit / 2))
    }
  }

  private async usage(conn: Conn, params: Readonly<Record<string, unknown>>): Promise<Record<string, unknown>> {
    this.exactKeys(params, ['workspace_ids'])
    const requested = readIdList(params.workspace_ids, 'workspace_ids')
    const known = this.options.workspaces.list().map(row => row.id)
    const candidates = requested.length === 0 ? known : requested.filter(id => known.includes(id))
    const allowed = candidates.filter(id => this.access(conn, { workspaceId: id }, 'usage.read'))
    if (requested.length > 0 && allowed.length === 0) throw new HubProtocolError('forbidden', 'not permitted: usage.read is required')
    // Workspaces served by one daemon share one usage ledger: ask each runtime once.
    const groups = new Map<string, string[]>()
    for (const id of allowed) {
      const runtime = this.options.workspaces.health(id)?.runtimeId ?? `workspace:${id}`
      groups.set(runtime, [...(groups.get(runtime) ?? []), id])
    }
    const unavailable: string[] = []
    const rows: Record<string, unknown>[] = []
    const merged = new Map<string, { model: string; profile?: string; calls: number; input_tokens: number; output_tokens: number; cost_usd?: number; cost_basis?: string }>()
    await Promise.all([...groups.entries()].map(async ([runtime, ids]) => {
      const first = ids[0]!
      const sessionKey = await this.anySessionKey(first)
      const outcome = await this.options.backend.call(first, 'usage.report', { session_key: sessionKey })
      if (outcome.state !== 'accepted') { unavailable.push(...ids); return }
      const models = Array.isArray(outcome.result.models) ? outcome.result.models : []
      for (const model of models) {
        if (!isRecord(model) || typeof model.model !== 'string') continue
        const basis = model.cost_basis === 'subscription' ? 'subscription' : undefined
        const profile = typeof model.profile === 'string' ? model.profile : undefined
        const key = JSON.stringify([model.model, profile ?? null, basis ?? null])
        const previous = merged.get(key) ?? { model: model.model, ...(profile === undefined ? {} : { profile }), calls: 0, input_tokens: 0, output_tokens: 0, ...(basis === undefined ? {} : { cost_basis: basis }) }
        const number = (value: unknown): number => (typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : 0)
        previous.calls += number(model.calls)
        previous.input_tokens += number(model.input_tokens)
        previous.output_tokens += number(model.output_tokens)
        if (typeof model.cost_usd === 'number' && Number.isFinite(model.cost_usd) && model.cost_usd >= 0) previous.cost_usd = (previous.cost_usd ?? 0) + model.cost_usd
        merged.set(key, previous)
        rows.push({ id: `${runtime}\0${key}`, input_tokens: number(model.input_tokens), output_tokens: number(model.output_tokens), ...(typeof model.cost_usd === 'number' ? { cost_usd: model.cost_usd } : {}), ...(basis === undefined ? {} : { cost_basis: basis }) })
      }
    }))
    // Plan-paid cost stays separate from billable cost; aggregateUsage never mixes them.
    const totals = aggregateUsage(rows)
    return { totals, by_model: [...merged.values()].sort((a, b) => b.input_tokens + b.output_tokens - (a.input_tokens + a.output_tokens)), runtimes: groups.size, unavailable_workspaces: unavailable.sort() }
  }

  private async anySessionKey(workspaceId: string): Promise<string> {
    const listed = await this.options.backend.call(workspaceId, 'session.active_list', {})
    if (listed.state === 'accepted' && Array.isArray(listed.result.sessions)) {
      for (const row of listed.result.sessions) if (isRecord(row) && typeof row.key === 'string' && row.key) return row.key
    }
    return 'hub:usage'
  }

  private command(conn: Conn, params: Readonly<Record<string, unknown>>): Promise<Handled> {
    const router = this.options.router
    if (!router) throw new HubProtocolError('unsupported', 'commands are not available on this Hub')
    return router.execute(conn.context, params).then(result => ({
      result: { ...result },
      after: () => this.emitCommandResult(conn, params, result),
    }))
  }

  private emitCommandResult(conn: Conn, params: Readonly<Record<string, unknown>>, result: HubCommandResult): void {
    const frame = hubEvent('hub.command_result', {
      ...(typeof params.workspace_id === 'string' ? { workspace_id: params.workspace_id } : {}),
      ...(typeof params.session_id === 'string' ? { session_id: params.session_id } : {}),
      ...result,
    })
    for (const other of this.conns) if (other.context.principal.id === conn.context.principal.id) this.sendFrame(other, frame)
  }

  // ------------------------------------------------------------ presence

  private updatePresence(conn: Conn, params: Readonly<Record<string, unknown>>): Handled {
    // Unknown keys are refused, not dropped, so a client can never believe a draft was shared.
    this.exactKeys(params, ['workspace_id', 'session_id', 'state'])
    const workspaceId = readId(params.workspace_id, 'workspace_id')
    const sessionId = readId(params.session_id, 'session_id')
    const state = params.state === undefined ? 'viewing' : params.state
    if (typeof state !== 'string' || !(PRESENCE_STATES as readonly string[]).includes(state)) {
      throw new HubProtocolError('invalid_request', `state must be one of ${PRESENCE_STATES.join(', ')}`)
    }
    this.require(conn, { workspaceId, sessionId }, 'presence.read')
    const target = `${workspaceId}\0${sessionId}`
    let total = 0
    for (const entries of this.presence.values()) total += entries.size
    const principal = conn.context.principal
    const key = `${principal.id}\0${conn.id}`
    if (total >= MAX_PRESENCE_ENTRIES && !this.presence.get(target)?.has(key)) throw new HubProtocolError('rate_limited', 'too many presence entries')
    const entries = this.presence.get(target) ?? new Map<string, PresenceEntry>()
    const now = this.clock()
    entries.set(key, { principalKind: principal.kind, principalId: principal.id, displayName: principal.displayName, state, updatedAt: now, expiresAt: now + this.presenceTtlMs, connId: conn.id })
    this.presence.set(target, entries)
    return { result: { ttl_ms: this.presenceTtlMs }, after: () => this.broadcastPresence(workspaceId, sessionId) }
  }

  private participants(target: string): Record<string, unknown>[] {
    const now = this.clock()
    const latest = new Map<string, PresenceEntry>()
    for (const entry of this.presence.get(target)?.values() ?? []) {
      if (entry.expiresAt <= now) continue
      const previous = latest.get(entry.principalId)
      if (!previous || entry.updatedAt >= previous.updatedAt) latest.set(entry.principalId, entry)
    }
    return [...latest.values()].sort((a, b) => a.principalId.localeCompare(b.principalId)).map(entry => ({
      principal_kind: entry.principalKind, principal_id: entry.principalId, display_name: entry.displayName,
      state: entry.state, updated_at: new Date(entry.updatedAt).toISOString(),
    }))
  }

  private broadcastPresence(workspaceId: string, sessionId: string): void {
    const frame = hubEvent('hub.presence', { workspace_id: workspaceId, session_id: sessionId, participants: this.participants(`${workspaceId}\0${sessionId}`) })
    for (const conn of this.conns) {
      if (conn.closed) continue
      const covered = [...conn.subs.values()].some(scope => scope.workspaceIds.includes(workspaceId) && (scope.sessionIds === undefined || scope.sessionIds.includes(sessionId)))
      if (covered && this.access(conn, { workspaceId, sessionId }, 'presence.read')) this.sendFrame(conn, frame)
    }
  }

  /**
   * Push one event frame to every open connection of the named principals.
   * Used by the supervisor for thread and confirmation events; the caller has
   * already decided who may see the payload. Returns how many sockets got it.
   */
  sendEventToPrincipals(principalIds: readonly string[], type: HubEventType, payload: Readonly<Record<string, unknown>>): number {
    const wanted = new Set(principalIds)
    const frame = hubEvent(type, payload)
    let sent = 0
    for (const conn of this.conns) {
      if (conn.closed || !wanted.has(conn.context.principal.id)) continue
      if (this.sendFrame(conn, frame)) sent += 1
    }
    return sent
  }

  /**
   * Push one event to every connected person (not agents). The payload is built per
   * recipient, so it can differ by what that person may do. Returns how many sockets got it.
   */
  sendEventToAll(type: HubEventType, build: (context: AuthenticatedContext) => Readonly<Record<string, unknown>> | undefined): number {
    let sent = 0
    for (const conn of this.conns) {
      if (conn.closed || conn.context.principal.kind !== 'user') continue
      // A builder that returns undefined skips that person (they may not see this event).
      const payload = build(conn.context)
      if (payload === undefined) continue
      if (this.sendFrame(conn, hubEvent(type, payload))) sent += 1
    }
    return sent
  }

  /** Drop expired presence and tell subscribers. Runs on a timer; exposed for deterministic tests. */
  sweepPresence(): void {
    const now = this.clock()
    for (const [target, entries] of [...this.presence]) {
      let removed = false
      for (const [key, entry] of entries) if (entry.expiresAt <= now) { entries.delete(key); removed = true }
      if (entries.size === 0) this.presence.delete(target)
      if (removed) { const [workspaceId, sessionId] = target.split('\0'); if (workspaceId && sessionId) this.broadcastPresence(workspaceId, sessionId) }
    }
  }

  // ---------------------------------------------------------- revocation

  private revoke(event: { readonly principalId: string; readonly deviceId?: string; readonly browserSessionId?: string; readonly reason: string }): void {
    for (const conn of [...this.conns]) {
      const principal = conn.context.principal
      if (principal.id !== event.principalId) continue
      if (event.deviceId !== undefined && !(principal.kind === 'user' && principal.deviceId === event.deviceId)) continue
      if (event.browserSessionId !== undefined && conn.context.browserSessionId !== event.browserSessionId) continue
      // Tell the client why before closing; the reason text stays internal.
      this.sendFrame(conn, hubEvent('hub.revoked' satisfies HubEventType, { principal_id: principal.id, reason: 'revoked' }))
      for (const id of conn.subs.keys()) this.options.subscriptions.unsubscribe(id)
      conn.subs.clear()
      conn.ws?.close(1008, 'access revoked')
    }
  }
}

async function isDirectory(path: string): Promise<boolean> {
  try { return (await import('node:fs/promises').then(fs => fs.stat(path))).isDirectory() } catch { return false }
}
