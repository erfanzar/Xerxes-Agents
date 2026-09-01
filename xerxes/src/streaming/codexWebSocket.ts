// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Codex WebSocket transport: one long-lived `wss` socket per session instead
 * of one SSE HTTP request per turn.
 *
 * Wire contract (mirrors the Codex backend's `/codex/responses` WebSocket
 * endpoint):
 * - The client sends exactly one UNCOMPRESSED text frame per request,
 *   `{ type: 'response.create', ...body }`.
 * - The server streams JSON event objects (the Responses event vocabulary).
 *   `response.completed` / `response.done` / `response.incomplete` are
 *   terminal; `type:'error'` and `response.failed` are typed failures.
 * - SSE request bodies may be zstd-compressed via `content-encoding: zstd`;
 *   WebSocket frames are never compressed.
 *
 * Sockets are pooled per (sessionId, accountKey) and reused across turns:
 * a pooled socket is evicted at 55 minutes of age, and an idle pooled socket
 * is closed after 5 minutes. Busy sessions get throwaway sockets that are
 * never cached. No keepalive pings — the idle timer is the liveness policy.
 *
 * Retry and SSE-fallback policy belongs to the caller; this module only
 * classifies errors and records a sticky fallback flag it can consult.
 */

import * as nodeZlib from 'node:zlib'

import { ProviderError } from '../core/errors.js'

/** Minimal structural WebSocket so tests inject fakes without a real socket. */
export interface WebSocketLike {
  readonly readyState: number
  send(data: string): void
  close(code?: number, reason?: string): void
  addEventListener(
    type: 'close' | 'error' | 'message' | 'open',
    listener: (event: WebSocketEventLike) => void,
  ): void
  removeEventListener?(
    type: 'close' | 'error' | 'message' | 'open',
    listener: (event: WebSocketEventLike) => void,
  ): void
}

/** Narrowed subset of the DOM/Bun WebSocket event fields this module reads. */
export interface WebSocketEventLike {
  readonly type?: string
  readonly data?: unknown
  readonly code?: number
  readonly reason?: string
  readonly error?: unknown
}

export interface CodexWebSocketOptions {
  readonly baseUrl: string
  /** Full auth headers (Authorization, account id, …); passed to the socket verbatim. */
  readonly headers: Record<string, string>
  /** Pools and reuses one socket per (sessionId, accountKey) when present. */
  readonly sessionId?: string
  /** Stable account identity for the pool key; falls back to Authorization. */
  readonly accountId?: string
  readonly transport?: 'auto' | 'sse' | 'websocket' | 'websocket-cached'
  readonly connectTimeoutMs?: number
  /** Max gap between server events before the socket is closed as stalled. */
  readonly idleTimeoutMs?: number
  /** Pool eviction override (tests/embedders); default 55 minutes. */
  readonly poolMaxAgeMs?: number
  /** Pool idle-close override (tests/embedders); default 5 minutes. */
  readonly poolIdleTimeoutMs?: number
  readonly signal?: AbortSignal
  readonly webSocketFactory?: (url: string, init: { headers: Record<string, string> }) => WebSocketLike
}

/** The provider answered with an event-level failure; `code` is its identity. */
export class CodexWsApiError extends ProviderError {
  readonly code: string
  readonly closeCode: number | undefined

  constructor(code: string, message: string, closeCode?: number) {
    super('codex', `Codex WebSocket ${code}: ${message}`, undefined, {
      code,
      ...(closeCode === undefined ? {} : { closeCode }),
    })
    this.code = code
    this.closeCode = closeCode
  }
}

/** The stream violated the framing contract (unparseable JSON, wrong shape). */
export class CodexWsProtocolError extends ProviderError {
  constructor(message: string) {
    super('codex', `Codex WebSocket protocol error: ${message}`)
  }
}

const CONNECT_TIMEOUT_DEFAULT_MS = 15_000
const IDLE_TIMEOUT_DEFAULT_MS = 120_000
const POOL_MAX_AGE_MS = 55 * 60_000
const POOL_IDLE_CLOSE_MS = 5 * 60_000

const TERMINAL_EVENT_TYPES: ReadonlySet<string> = new Set([
  'response.completed',
  'response.done',
  'response.incomplete',
])

// ── URL derivation ──────────────────────────────────────────────────────

/** `https://host/backend-api` → `wss://host/backend-api/codex/responses`. */
export function resolveCodexWebSocketUrl(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, '')
  if (!trimmed) throw new CodexWsApiError('invalid_base_url', 'baseUrl must not be empty')
  const url = new URL(trimmed)
  const path = url.pathname.replace(/\/+$/, '')
  if (!path.endsWith('/codex') && !path.endsWith('/codex/responses')) {
    url.pathname = `${path}/codex/responses`
  }
  if (url.protocol === 'https:') url.protocol = 'wss:'
  else if (url.protocol === 'http:') url.protocol = 'ws:'
  return url.toString()
}

// ── zstd request compression (SSE path) ─────────────────────────────────

/** Header that accompanies a zstd-compressed SSE request body. */
export const CODEX_SSE_COMPRESSION_HEADER: Readonly<Record<string, string>> = Object.freeze({
  'content-encoding': 'zstd',
})

type BunGlobals = {
  Bun?: {
    zstdCompressSync?: (data: string | Uint8Array, level?: number) => Uint8Array
  }
}

type NodeZstdCompress = (data: Uint8Array, options?: { level?: number }) => Uint8Array

/**
 * Zstd-compress a JSON request body: Bun's native codec first, then
 * `node:zlib` at level 3, `undefined` when neither can produce bytes.
 */
export function compressRequestBodyZstd(body: string): Uint8Array | undefined {
  const input = Buffer.from(body, 'utf8')
  try {
    const bunCompress = (globalThis as BunGlobals).Bun?.zstdCompressSync
    if (typeof bunCompress === 'function') {
      const compressed = bunCompress.call((globalThis as BunGlobals).Bun, input)
      if (compressed instanceof Uint8Array && compressed.length > 0) return compressed
    }
  } catch {
    // Fall through to the node codec.
  }
  try {
    const nodeCompress = (nodeZlib as { zstdCompressSync?: NodeZstdCompress }).zstdCompressSync
    if (typeof nodeCompress === 'function') {
      const compressed = nodeCompress(input, { level: 3 })
      if (compressed instanceof Uint8Array && compressed.length > 0) return compressed
    }
  } catch {
    // Both codecs failed; the caller sends the body uncompressed.
  }
  return undefined
}

// ── Event plumbing ──────────────────────────────────────────────────────

type SocketSignal =
  | { readonly kind: 'open' }
  | { readonly kind: 'message'; readonly data: unknown }
  | { readonly kind: 'close'; readonly code: number; readonly reason: string }
  | { readonly kind: 'error'; readonly error: unknown }

/** Queues socket signals so the generator can await them one at a time. */
class SocketHandle {
  readonly socket: WebSocketLike
  private readonly queued: SocketSignal[] = []
  private readonly resolvers: Array<(signal: SocketSignal) => void> = []

  constructor(socket: WebSocketLike) {
    this.socket = socket
    socket.addEventListener('open', () => this.push({ kind: 'open' }))
    socket.addEventListener('message', event => this.push({ kind: 'message', data: event.data }))
    socket.addEventListener('close', event => this.push({
      kind: 'close',
      code: typeof event.code === 'number' ? event.code : 1006,
      reason: typeof event.reason === 'string' ? event.reason : '',
    }))
    socket.addEventListener('error', event => this.push({ kind: 'error', error: event.error }))
  }

  private push(signal: SocketSignal): void {
    const resolver = this.resolvers.shift()
    if (resolver) resolver(signal)
    else this.queued.push(signal)
  }

  /** Events read before a previous turn ended (e.g. post-terminal usage). */
  drain(): void {
    this.queued.length = 0
  }

  /** Cancellation-aware next(); a cancelled waiter re-queues its signal. */
  next(): { promise: Promise<SocketSignal>; cancel: () => void } {
    const queued = this.queued.shift()
    if (queued) return { promise: Promise.resolve(queued), cancel: () => undefined }
    let resolver!: (signal: SocketSignal) => void
    const promise = new Promise<SocketSignal>(resolve => {
      resolver = resolve
    })
    this.resolvers.push(resolver)
    const cancel = (): void => {
      const index = this.resolvers.indexOf(resolver)
      if (index >= 0) this.resolvers.splice(index, 1)
    }
    return { promise, cancel }
  }
}

type WaitOutcome =
  | { readonly kind: 'signal'; readonly signal: SocketSignal }
  | { readonly kind: 'timeout' }
  | { readonly kind: 'aborted' }

interface XerxesTimer {
  unref?: () => void
}

function createTimer(ms: number): { promise: Promise<'timeout'>; cancel: () => void } {
  let timer: ReturnType<typeof setTimeout> | undefined
  const promise = new Promise<'timeout'>(resolve => {
    timer = setTimeout(() => resolve('timeout'), ms)
  })
  ;(timer as unknown as XerxesTimer | undefined)?.unref?.()
  return {
    promise,
    cancel: () => {
      if (timer !== undefined) clearTimeout(timer)
    },
  }
}

async function waitOutcome(
  handle: SocketHandle,
  timeoutMs: number,
  signal: AbortSignal | undefined,
): Promise<WaitOutcome> {
  if (signal?.aborted) return { kind: 'aborted' }
  const request = handle.next()
  const timer = createTimer(timeoutMs)
  let onAbort: (() => void) | undefined
  const abortOutcome = signal
    ? new Promise<WaitOutcome>(resolve => {
        onAbort = () => resolve({ kind: 'aborted' })
        signal.addEventListener('abort', onAbort, { once: true })
      })
    : undefined
  try {
    const races: ReadonlyArray<Promise<WaitOutcome>> = [
      request.promise.then((value): WaitOutcome => ({ kind: 'signal', signal: value })),
      timer.promise.then((): WaitOutcome => ({ kind: 'timeout' })),
      ...(abortOutcome ? [abortOutcome] : []),
    ]
    return await Promise.race(races)
  } finally {
    request.cancel()
    timer.cancel()
    if (signal && onAbort) signal.removeEventListener('abort', onAbort)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringAt(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key]
  return typeof value === 'string' && value ? value : undefined
}

function closeSocket(socket: WebSocketLike, code: number, reason: string): void {
  try {
    socket.close(code, reason)
  } catch {
    // Closing a dying socket must never mask the original failure.
  }
}

// ── Session pool ────────────────────────────────────────────────────────

interface PoolEntry {
  readonly handle: SocketHandle
  busy: boolean
  createdAt: number
  idleTimer: ReturnType<typeof setTimeout> | undefined
}

const poolBySession = new Map<string, Map<string, PoolEntry>>()
const sseFallbackSessions = new Set<string>()

interface CodexWebSocketStats {
  connectionsCreated: number
  connectionsReused: number
  failures: number
}

const stats: CodexWebSocketStats = { connectionsCreated: 0, connectionsReused: 0, failures: 0 }

export interface CodexWebSocketDebugStats {
  readonly connectionsCreated: number
  readonly connectionsReused: number
  readonly failures: number
  /** Sessions currently pinned to the sticky SSE fallback. */
  readonly fallbackSessions: number
}

function accountKeyOf(options: CodexWebSocketOptions): string {
  return options.accountId?.trim()
    || options.headers.Authorization
    || options.headers.authorization
    || ''
}

function entriesFor(sessionKey: string): Map<string, PoolEntry> {
  let entries = poolBySession.get(sessionKey)
  if (!entries) {
    entries = new Map()
    poolBySession.set(sessionKey, entries)
  }
  return entries
}

function clearIdleTimer(entry: PoolEntry): void {
  if (entry.idleTimer !== undefined) {
    clearTimeout(entry.idleTimer)
    entry.idleTimer = undefined
  }
}

/** Take a reusable pooled socket, or clean up a dead/stale entry. */
type PoolAcquisition =
  | { readonly kind: 'reuse'; readonly handle: SocketHandle }
  | { readonly kind: 'fresh-cacheable' }
  | { readonly kind: 'fresh-throwaway' }

function acquireFromPool(
  sessionKey: string,
  accountKey: string,
  options: CodexWebSocketOptions,
): PoolAcquisition {
  const entry = poolBySession.get(sessionKey)?.get(accountKey)
  if (!entry) return { kind: 'fresh-cacheable' }
  const maxAgeMs = options.poolMaxAgeMs ?? POOL_MAX_AGE_MS
  // A busy entry means another turn is mid-stream on that socket: this turn
  // gets its own connection, and the throwaway is never cached over it.
  if (entry.busy) return { kind: 'fresh-throwaway' }
  if (entry.handle.socket.readyState !== 1) {
    clearIdleTimer(entry)
    poolBySession.get(sessionKey)?.delete(accountKey)
    closeSocket(entry.handle.socket, 1000, 'stale')
    return { kind: 'fresh-cacheable' }
  }
  if (Date.now() - entry.createdAt >= maxAgeMs) {
    clearIdleTimer(entry)
    poolBySession.get(sessionKey)?.delete(accountKey)
    closeSocket(entry.handle.socket, 1000, 'evicted')
    return { kind: 'fresh-cacheable' }
  }
  clearIdleTimer(entry)
  entry.busy = true
  entry.handle.drain()
  return { kind: 'reuse', handle: entry.handle }
}

function cacheInPool(sessionKey: string, accountKey: string, handle: SocketHandle): void {
  entriesFor(sessionKey).set(accountKey, { handle, busy: true, createdAt: Date.now(), idleTimer: undefined })
}

/** Hand a finished socket back to the pool (or close it if it aged out). */
function releaseToPool(
  sessionKey: string,
  accountKey: string,
  handle: SocketHandle,
  options: CodexWebSocketOptions,
): void {
  const entry = poolBySession.get(sessionKey)?.get(accountKey)
  if (!entry || entry.handle !== handle) return
  const maxAgeMs = options.poolMaxAgeMs ?? POOL_MAX_AGE_MS
  if (Date.now() - entry.createdAt >= maxAgeMs) {
    discardFromPool(sessionKey, accountKey, handle)
    closeSocket(handle.socket, 1000, 'evicted')
    return
  }
  entry.busy = false
  const idleMs = options.poolIdleTimeoutMs ?? POOL_IDLE_CLOSE_MS
  const timer = setTimeout(() => {
    entry.idleTimer = undefined
    if (entry.busy) return
    discardFromPool(sessionKey, accountKey, handle)
    closeSocket(handle.socket, 1000, 'idle_timeout')
  }, idleMs)
  ;(timer as unknown as XerxesTimer).unref?.()
  entry.idleTimer = timer
}

function discardFromPool(sessionKey: string | undefined, accountKey: string, handle: SocketHandle): void {
  if (!sessionKey) return
  const entries = poolBySession.get(sessionKey)
  const entry = entries?.get(accountKey)
  if (entry && entry.handle === handle) {
    clearIdleTimer(entry)
    entries?.delete(accountKey)
  }
  if (entries && entries.size === 0) poolBySession.delete(sessionKey)
}

/** Close every pooled socket for one session (or all sessions). */
export function closeCodexWebSocketSessions(sessionId?: string): void {
  const targets = sessionId !== undefined ? [sessionId] : [...poolBySession.keys()]
  for (const sessionKey of targets) {
    const entries = poolBySession.get(sessionKey)
    poolBySession.delete(sessionKey)
    for (const entry of entries?.values() ?? []) {
      clearIdleTimer(entry)
      closeSocket(entry.handle.socket, 1000, 'session_closed')
    }
  }
}

export function getCodexWebSocketDebugStats(_sessionId?: string): CodexWebSocketDebugStats {
  void _sessionId
  return {
    connectionsCreated: stats.connectionsCreated,
    connectionsReused: stats.connectionsReused,
    failures: stats.failures,
    fallbackSessions: sseFallbackSessions.size,
  }
}

// ── Retry / fallback classification (policy stays with the caller) ──────

/** Connection-limit rejections and oversized frames deserve a fresh attempt. */
export function isCodexRetryableWebSocketError(error: unknown): boolean {
  if (!(error instanceof CodexWsApiError)) return false
  return error.code === 'websocket_connection_limit_reached' || error.closeCode === 1009
}

export function recordCodexWebSocketFallback(sessionId: string): void {
  const key = sessionId.trim()
  if (key) sseFallbackSessions.add(key)
}

export function codexWebSocketFallbackActive(sessionId: string): boolean {
  return sseFallbackSessions.has(sessionId.trim())
}

export function clearCodexWebSocketFallback(sessionId?: string): void {
  if (sessionId === undefined) sseFallbackSessions.clear()
  else sseFallbackSessions.delete(sessionId.trim())
}

// ── Streaming ───────────────────────────────────────────────────────────

type WebSocketConstructor = new (
  url: string,
  init?: { headers?: Record<string, string> },
) => WebSocketLike

function defaultWebSocketFactory(url: string, init: { headers: Record<string, string> }): WebSocketLike {
  const constructor = (globalThis as { WebSocket?: WebSocketConstructor }).WebSocket
  if (typeof constructor !== 'function') {
    throw new CodexWsApiError('websocket_unavailable', 'no global WebSocket implementation')
  }
  return new constructor(url, { headers: init.headers })
}

function apiErrorFromEvent(event: Record<string, unknown>): CodexWsApiError {
  const error = isRecord(event.error) ? event.error : {}
  const fallback = stringAt(event, 'type') === 'response.failed' ? 'response_failed' : 'unknown_error'
  const code = stringAt(event, 'code') ?? stringAt(error, 'code') ?? fallback
  const message = stringAt(event, 'message') ?? stringAt(error, 'message')
    ?? JSON.stringify(event).slice(0, 500)
  return new CodexWsApiError(code, message)
}

/**
 * Stream one Codex response over a (possibly pooled) WebSocket.
 *
 * Sends `{ type: 'response.create', ...requestBody }` as a single plain text
 * frame, then yields each parsed server event until a terminal event, which
 * is yielded as-is (`response.done` / `response.incomplete` are not renamed —
 * the caller maps them onto the neutral finish vocabulary).
 */
export async function* streamCodexWebSocket(
  requestBody: Record<string, unknown>,
  options: CodexWebSocketOptions,
): AsyncGenerator<Record<string, unknown>> {
  if (options.transport === 'sse') {
    throw new CodexWsApiError('transport_mismatch', "streamCodexWebSocket cannot serve transport 'sse'")
  }
  const url = resolveCodexWebSocketUrl(options.baseUrl)
  const sessionKey = options.sessionId?.trim() || undefined
  const accountKey = accountKeyOf(options)
  const idleTimeoutMs = options.idleTimeoutMs ?? IDLE_TIMEOUT_DEFAULT_MS

  let handle: SocketHandle | undefined
  let pooled = false
  /**
   * True once this socket was already closed/discarded by the failure path —
   * the finally block must not double-close on an early consumer return.
   */
  let settled = false

  const teardown = (reason: string): void => {
    if (!handle || settled) return
    settled = true
    discardFromPool(sessionKey, accountKey, handle)
    closeSocket(handle.socket, 1000, reason)
  }

  try {
    const acquisition = sessionKey ? acquireFromPool(sessionKey, accountKey, options) : undefined
    if (acquisition?.kind === 'reuse') {
      handle = acquisition.handle
      pooled = true
      stats.connectionsReused += 1
    }
    if (!handle) {
      const socket = (options.webSocketFactory ?? defaultWebSocketFactory)(url, { headers: options.headers })
      handle = new SocketHandle(socket)
      stats.connectionsCreated += 1
      await connectSocket(handle, options)
      if (sessionKey && acquisition?.kind === 'fresh-cacheable') {
        cacheInPool(sessionKey, accountKey, handle)
        pooled = true
      }
    }

    handle.socket.send(JSON.stringify({ type: 'response.create', ...requestBody }))

    for (;;) {
      const outcome = await waitOutcome(handle, idleTimeoutMs, options.signal)
      if (outcome.kind === 'aborted') {
        stats.failures += 1
        settled = true
        discardFromPool(sessionKey, accountKey, handle)
        closeSocket(handle.socket, 1000, 'aborted')
        throw new Error('aborted')
      }
      if (outcome.kind === 'timeout') {
        stats.failures += 1
        settled = true
        discardFromPool(sessionKey, accountKey, handle)
        closeSocket(handle.socket, 1000, 'idle_timeout')
        throw new CodexWsApiError('idle_timeout', `no event for ${idleTimeoutMs}ms`)
      }
      const signal = outcome.signal
      if (signal.kind === 'open') continue
      if (signal.kind === 'close') {
        stats.failures += 1
        settled = true
        discardFromPool(sessionKey, accountKey, handle)
        throw new CodexWsApiError(
          'websocket_closed',
          `connection closed before a terminal event (code ${signal.code}${signal.reason ? ` ${signal.reason}` : ''})`,
          signal.code,
        )
      }
      if (signal.kind === 'error') {
        stats.failures += 1
        settled = true
        discardFromPool(sessionKey, accountKey, handle)
        closeSocket(handle.socket, 1000, 'stream_error')
        throw new CodexWsApiError('websocket_error', 'socket errored mid-stream')
      }

      let event: unknown
      const frame = typeof signal.data === 'string' ? signal.data : ''
      try {
        event = JSON.parse(frame)
      } catch {
        stats.failures += 1
        settled = true
        discardFromPool(sessionKey, accountKey, handle)
        closeSocket(handle.socket, 1000, 'protocol_error')
        throw new CodexWsProtocolError(`malformed JSON frame: ${frame.slice(0, 200)}`)
      }
      if (!isRecord(event)) {
        stats.failures += 1
        settled = true
        discardFromPool(sessionKey, accountKey, handle)
        closeSocket(handle.socket, 1000, 'protocol_error')
        throw new CodexWsProtocolError(`event is not an object: ${frame.slice(0, 200)}`)
      }
      const type = stringAt(event, 'type') ?? ''
      if (type === 'error' || type === 'response.failed') {
        stats.failures += 1
        settled = true
        discardFromPool(sessionKey, accountKey, handle)
        closeSocket(handle.socket, 1000, 'api_error')
        throw apiErrorFromEvent(event)
      }
      yield event
      if (TERMINAL_EVENT_TYPES.has(type)) {
        settled = true
        if (pooled && sessionKey) {
          releaseToPool(sessionKey, accountKey, handle, options)
        } else {
          closeSocket(handle.socket, 1000, 'done')
        }
        return
      }
    }
  } finally {
    // Consumer broke out early (`break`/`return`) without a terminal event.
    teardown('consumer_detached')
  }
}

async function connectSocket(handle: SocketHandle, options: CodexWebSocketOptions): Promise<void> {
  const timeoutMs = options.connectTimeoutMs ?? CONNECT_TIMEOUT_DEFAULT_MS
  for (;;) {
    if (handle.socket.readyState === 1) return
    if (handle.socket.readyState === 3) {
      stats.failures += 1
      throw new CodexWsApiError('websocket_closed', 'connection closed before it opened')
    }
    const outcome = await waitOutcome(handle, timeoutMs, options.signal)
    if (outcome.kind === 'aborted') {
      closeSocket(handle.socket, 1000, 'aborted')
      throw new Error('aborted')
    }
    if (outcome.kind === 'timeout') {
      stats.failures += 1
      closeSocket(handle.socket, 1000, 'connect_timeout')
      throw new CodexWsApiError('connect_timeout', `no open frame within ${timeoutMs}ms`)
    }
    const signal = outcome.signal
    if (signal.kind === 'open') return
    if (signal.kind === 'close') {
      stats.failures += 1
      throw new CodexWsApiError(
        'websocket_closed',
        `connection closed before it opened (code ${signal.code})`,
        signal.code,
      )
    }
    if (signal.kind === 'error') {
      stats.failures += 1
      throw new CodexWsApiError('websocket_error', 'connection errored before it opened')
    }
    // A message before open is out of contract; keep waiting for open.
  }
}

// ── Delta / cached-context continuation ─────────────────────────────────

/** Everything needed to extend the previous turn instead of resending it. */
export interface CodexWsContinuation {
  readonly lastRequestBody: Record<string, unknown>
  readonly lastResponseId: string
  readonly lastResponseItems: readonly Record<string, unknown>[]
}

/** Remember the request/response pair a delta continuation would extend. */
export function continuationFromResponse(
  body: Record<string, unknown>,
  responseId: string,
  assistantItems: readonly Record<string, unknown>[],
): CodexWsContinuation | undefined {
  const trimmed = responseId.trim()
  if (!trimmed) return undefined
  return {
    lastRequestBody: body,
    lastResponseId: trimmed,
    lastResponseItems: [...assistantItems],
  }
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue)
  if (isRecord(value)) {
    const sorted: Record<string, unknown> = {}
    for (const key of Object.keys(value).sort()) sorted[key] = stableValue(value[key])
    return sorted
  }
  return value
}

function stableStringify(value: unknown): string {
  return JSON.stringify(stableValue(value))
}

/**
 * Rewrite `body` as a delta continuation when it is a strict prefix-extension
 * of the remembered turn: everything except `input`/`previous_response_id`
 * must be unchanged, and `body.input` must extend
 * `lastRequestBody.input ++ lastResponseItems`. Otherwise the full body is
 * returned untouched.
 */
export function buildCachedWebSocketRequestBody(
  body: Record<string, unknown>,
  continuation: CodexWsContinuation,
): { body: Record<string, unknown>; usedDelta: boolean } {
  const withoutVolatile = (value: Record<string, unknown>): Record<string, unknown> => {
    const { input: _input, previous_response_id: _previousResponseId, ...rest } = value
    return rest
  }
  try {
    if (stableStringify(withoutVolatile(body)) !== stableStringify(withoutVolatile(continuation.lastRequestBody))) {
      return { body, usedDelta: false }
    }
    const baseInput = Array.isArray(continuation.lastRequestBody.input) ? continuation.lastRequestBody.input : []
    const prefix = [...baseInput, ...continuation.lastResponseItems]
    const input = body.input
    if (!Array.isArray(input) || input.length <= prefix.length) {
      return { body, usedDelta: false }
    }
    for (const [index, prefixItem] of prefix.entries()) {
      if (stableStringify(input[index]) !== stableStringify(prefixItem)) {
        return { body, usedDelta: false }
      }
    }
    return {
      body: {
        ...body,
        previous_response_id: continuation.lastResponseId,
        input: input.slice(prefix.length),
      },
      usedDelta: true,
    }
  } catch {
    // Cyclic or otherwise unserializable bodies fall back to the full request.
    return { body, usedDelta: false }
  }
}
