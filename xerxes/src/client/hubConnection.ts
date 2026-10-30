// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  HUB_LIMITS,
  HUB_PROTOCOL_VERSION,
  HubProtocolError,
  encodeHubFrame,
  isHubMethod,
  isRecord,
  parseHubServerFrame,
  type HubActor,
  type HubCommandAction,
  type HubCommandResult,
  type HubCommandState,
  type HubErrorCode,
  type HubEvent,
  type HubEventType,
  type HubGapReason,
  type HubMethod,
  type HubRequestFrame,
} from '../hub/protocol.js'
import { EventCursorTracker, newCommandId, type ApplyOutcome, type BridgeScope, type ClientBridge, type ConnectionStatus } from './bridge.js'
import { arr, num, rec, str } from './parse.js'

/**
 * One WebSocket to the Hub, shared by every view. It owns request correlation,
 * reconnect with capped jittered backoff, resumable subscriptions and command
 * outcome resolution. Browser-safe: the socket and timers are injected.
 *
 * Delivery guarantees the rest of the app can lean on:
 * - events reach `apply` once, in order; duplicates are dropped;
 * - a sequence jump, an expired or foreign cursor, or a failed apply triggers
 *   a snapshot reset (`handlers.reset`) instead of guessing;
 * - acks are sent only after `apply` returned;
 * - a mutation is sent at most once. If its acknowledgement is lost the
 *   outcome is asked for by command id (`hub.command.status`), never replayed.
 */

export type RequestFailure = HubErrorCode | 'timeout' | 'disconnected' | 'offline'

export class HubRequestError extends Error {
  readonly code: RequestFailure
  /** True when the request may have reached the Hub: the outcome is unknown. */
  readonly uncertain: boolean
  constructor(code: RequestFailure, message: string, uncertain = false) {
    super(message)
    this.name = 'HubRequestError'
    this.code = code
    this.uncertain = uncertain
  }
}

export interface SocketHandlers {
  open(): void
  message(data: string): void
  /** A frame the socket could not deliver as text. */
  binary(): void
  close(code: number, reason: string): void
  error(): void
}

export interface SocketLike {
  send(data: string): void
  close(code?: number, reason?: string): void
}

export type SocketFactory = (url: string, handlers: SocketHandlers) => SocketLike

export interface Timers {
  setTimeout(callback: () => void, ms: number): unknown
  clearTimeout(handle: unknown): void
}

export type SessionCheck = 'ok' | 'denied' | 'unavailable'

export interface BackoffOptions {
  readonly baseMs: number
  readonly maxMs: number
}

export interface HubConnectionOptions {
  /** Same-origin `wss://host/api/hub/v1/ws`. Never carries a token. */
  readonly url: string
  readonly socketFactory?: SocketFactory
  /** Called before every reconnect; `denied` ends the connection as revoked. */
  readonly ensureSession?: () => Promise<SessionCheck>
  readonly requestTimeoutMs?: number
  readonly backoff?: BackoffOptions
  readonly random?: () => number
  readonly timers?: Timers
  /** How long a lost acknowledgement waits for the link before reporting `unknown`. */
  readonly commandStatusWaitMs?: number
  readonly maxFrameBytes?: number
  readonly onProtocolError?: (error: Error) => void
  /** Treat a sequence jump as a gap. Default false: the Hub's sequences are global, so scoped streams have holes. */
  readonly contiguousSequences?: boolean
  /** How long one attempt may take to reach `live` before it is dropped and retried. Default 20 s. */
  readonly connectTimeoutMs?: number
  /** Timeout of the liveness probe sent by `verifyLink`. Default 4 s. */
  readonly probeTimeoutMs?: number
  /**
   * While visible and live, probe the link when no frame arrived for this long.
   * The Hub pings at the WebSocket layer, which a page cannot observe, so a
   * silent link is checked with a request. 0 or absent disables it.
   */
  readonly heartbeatMs?: number
}

/** Result of a liveness check: the link answered, a reconnect was started, or nothing was checked. */
export type LinkCheck = 'live' | 'reconnecting' | 'skipped'

export interface ResetBoundary {
  readonly sequence?: number
  readonly cursor?: string
  /** Replay the scope from the start of the journal (used to recover pending interactions). */
  readonly fromStart?: boolean
}

export type ResetReason = HubGapReason | 'sequence_gap' | 'apply_failed'

export interface SubscriptionSpec {
  readonly workspaceIds: readonly string[]
  readonly sessionIds?: readonly string[]
}

export interface SubscriptionHandlers {
  /** `replay` is true while the initial replay of a cursor-less subscription is still being delivered. */
  apply(event: HubEvent, info: { readonly replay: boolean }): void
  /** Re-read a snapshot and return the boundary it names, when it names one. */
  reset(info: { readonly reason: ResetReason }): Promise<ResetBoundary | undefined> | ResetBoundary | undefined
  error?(error: HubRequestError): void
  /** Called once the initial replay (a subscription started without a cursor) has been applied. */
  caughtUp?(): void
}

export interface SubscriptionHandle {
  dispose(): void
  /** The cursor a reconnect would resume from. */
  cursor(): string | undefined
}

export interface CommandInput {
  readonly workspaceId: string
  readonly sessionId?: string
  readonly action: HubCommandAction
  readonly args: Readonly<Record<string, unknown>>
  readonly commandId?: string
  readonly expectedRevision?: number
}

export interface HelloInfo {
  readonly userId?: string
  readonly displayName?: string
}

type NoticeListener = (type: HubEventType, payload: Readonly<Record<string, unknown>>) => void

interface Pending {
  readonly method: string
  readonly resolve: (value: Readonly<Record<string, unknown>>) => void
  readonly reject: (error: HubRequestError) => void
  readonly timer: unknown
}

interface Sub {
  readonly spec: SubscriptionSpec
  readonly handlers: SubscriptionHandlers
  readonly tracker: EventCursorTracker
  baseCursor: string | undefined
  /** `through_cursor` of the last batch fully applied. */
  applied: string | undefined
  /** Watermark the Hub reported when this subscription (re)started. */
  watermark: string | undefined
  /** True while the initial replay of a cursor-less subscription is still arriving. */
  replaying: boolean
  serverId: string | undefined
  resetting: boolean
  resetPromise: Promise<void> | undefined
  disposed: boolean
  starting: Promise<void> | undefined
}

/**
 * The Hub numbers events globally, so a subscription scoped to some
 * workspaces sees holes in `sequence` for events it is not allowed to see.
 * This tracker still dedupes by id and by sequence but treats a jump as normal;
 * real loss is announced by the Hub as `hub.stream_gap`. Use the plain
 * `EventCursorTracker` (option `contiguousSequences`) against a Hub that
 * numbers each scope contiguously.
 */
export class ScopedCursorTracker extends EventCursorTracker {
  override classify(event: HubEvent): ApplyOutcome {
    const outcome = super.classify(event)
    return outcome.kind === 'gap' ? { kind: 'applied' } : outcome
  }
}

/**
 * A phone on a flaky network should not wait half a minute for its next try,
 * and a hidden page schedules no retries at all (see `setVisible`), so a low
 * cap costs nothing in the background.
 */
const DEFAULT_BACKOFF: BackoffOptions = { baseMs: 500, maxMs: 8_000 }
/** Socket open + hello + subscription restarts; generous because a reset may load a snapshot. */
const DEFAULT_CONNECT_TIMEOUT_MS = 20_000
const DEFAULT_PROBE_TIMEOUT_MS = 4_000
/** Consecutive request timeouts on a live link after which the socket is treated as dead. */
const DEAD_LINK_TIMEOUTS = 2

/** Capped exponential backoff with jitter in [half, full] of the step. */
export function backoffDelay(attempt: number, options: BackoffOptions = DEFAULT_BACKOFF, random: () => number = Math.random): number {
  const step = Math.min(options.maxMs, options.baseMs * 2 ** Math.max(0, attempt - 1))
  return Math.round(step / 2 + random() * (step / 2))
}

export function hubWebSocketUrl(location: { readonly protocol: string; readonly host: string }): string {
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/hub/v1/ws`
}

export const browserSocketFactory: SocketFactory = (url, handlers) => {
  const socket = new WebSocket(url)
  socket.addEventListener('open', () => handlers.open())
  socket.addEventListener('message', event => {
    if (typeof event.data === 'string') handlers.message(event.data)
    else handlers.binary()
  })
  socket.addEventListener('close', event => handlers.close(event.code, event.reason))
  socket.addEventListener('error', () => handlers.error())
  return { send: text => socket.send(text), close: (code, reason) => socket.close(code, reason) }
}

const browserTimers: Timers = {
  setTimeout: (callback, ms) => globalThis.setTimeout(callback, ms),
  clearTimeout: handle => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
}

export function asActor(value: unknown): HubActor | undefined {
  const item = rec(value)
  const kind = item.principal_kind
  const id = str(item.principal_id)
  const name = str(item.display_name)
  if ((kind !== 'user' && kind !== 'agent') || !id || !name) return undefined
  const userId = str(item.user_id)
  const deviceId = str(item.device_id)
  const requestedBy = str(item.requested_by)
  const confirmedBy = str(item.confirmed_by)
  return {
    principal_kind: kind,
    principal_id: id,
    display_name: name,
    ...(userId ? { user_id: userId } : {}),
    ...(deviceId ? { device_id: deviceId } : {}),
    ...(requestedBy ? { requested_by: requestedBy } : {}),
    ...(confirmedBy ? { confirmed_by: confirmedBy } : {}),
  }
}

/** Validate one untrusted event from a batch. */
export function asHubEvent(raw: unknown): HubEvent | undefined {
  const item = rec(raw)
  const eventId = str(item.event_id)
  const cursor = str(item.cursor)
  const workspaceId = str(item.workspace_id)
  const type = str(item.type)
  const sequence = num(item.sequence)
  if (!eventId || !cursor || !workspaceId || !type || sequence === undefined || !isRecord(item.payload)) return undefined
  const sessionId = str(item.session_id)
  const actor = asActor(item.actor)
  return {
    event_id: eventId,
    sequence,
    cursor,
    workspace_id: workspaceId,
    type,
    payload: item.payload,
    occurred_at: str(item.occurred_at) ?? new Date(0).toISOString(),
    ...(sessionId ? { session_id: sessionId } : {}),
    ...(actor ? { actor } : {}),
  }
}

const COMMAND_STATES: ReadonlySet<string> = new Set<HubCommandState>(['accepted', 'rejected', 'unknown', 'pending_confirmation'])

export function parseCommandResult(value: Readonly<Record<string, unknown>>, commandId: string): HubCommandResult {
  const state = typeof value.state === 'string' && COMMAND_STATES.has(value.state) ? (value.state as HubCommandState) : 'unknown'
  const error = rec(value.error)
  const message = str(error.message)
  const result = isRecord(value.result) ? value.result : undefined
  return {
    ok: value.ok === true && state !== 'rejected' && state !== 'unknown',
    command_id: str(value.command_id) ?? commandId,
    state,
    ...(result ? { result } : {}),
    ...(message ? { error: { code: (str(error.code) ?? 'internal') as HubErrorCode, message } } : {}),
  }
}

const SCOPED_METHODS: ReadonlySet<string> = new Set(['hub.history', 'hub.command', 'hub.presence.update'])

export class HubConnection {
  private readonly options: HubConnectionOptions
  private readonly factory: SocketFactory
  private readonly timers: Timers
  private readonly random: () => number
  private readonly maxFrameBytes: number
  private readonly pending = new Map<string, Pending>()
  private readonly subs = new Set<Sub>()
  private readonly byServerId = new Map<string, Sub>()
  private readonly statusListeners = new Set<(status: ConnectionStatus) => void>()
  private readonly eventListeners = new Set<(event: HubEvent) => void>()
  private readonly noticeListeners = new Set<NoticeListener>()
  private current: ConnectionStatus = { state: 'closed' }
  private socket: SocketLike | undefined
  private generation = 0
  private attempt = 0
  private nextRequest = 0
  private reconnectTimer: unknown
  private stopped = true
  private revoked = false
  /** One refresh-and-retry after an unauthorized hello; reset once the connection goes live. */
  private helloRetried = false
  private lastError: string | undefined
  private hello: HelloInfo = {}
  /** False while the page is hidden: no reconnect timer runs then; `resume` reconnects at once. */
  private visible = true
  /** A reconnect that was due while hidden. */
  private parked = false
  private connectTimer: unknown
  private consecutiveTimeouts = 0
  private heartbeatTimer: unknown
  private framesSinceBeat = false
  private probe: Promise<LinkCheck> | undefined
  /** Set when a caller of the running probe asked to notify `onLinkVerified` listeners. */
  private probeNotifies = false
  private readonly resumeListeners = new Set<() => void>()

  constructor(options: HubConnectionOptions) {
    this.options = options
    this.factory = options.socketFactory ?? browserSocketFactory
    this.timers = options.timers ?? browserTimers
    this.random = options.random ?? Math.random
    this.maxFrameBytes = options.maxFrameBytes ?? HUB_LIMITS.maxFrameBytes
  }

  status(): ConnectionStatus {
    return this.current
  }

  helloInfo(): HelloInfo {
    return this.hello
  }

  onStatus(listener: (status: ConnectionStatus) => void): () => void {
    this.statusListeners.add(listener)
    return () => { this.statusListeners.delete(listener) }
  }

  /** Every event applied to any subscription, after it was applied. */
  onEvent(listener: (event: HubEvent) => void): () => void {
    this.eventListeners.add(listener)
    return () => { this.eventListeners.delete(listener) }
  }

  /** Non-batch Hub events: presence, workspace status, command results, supervisor frames. */
  onNotice(listener: NoticeListener): () => void {
    this.noticeListeners.add(listener)
    return () => { this.noticeListeners.delete(listener) }
  }

  connect(): void {
    if (this.revoked || !this.stopped) return
    this.stopped = false
    this.attempt = 0
    this.setStatus({ state: 'connecting' })
    void this.open()
  }

  /** Drop the link now and reconnect immediately, with the backoff reset (used by "Retry"). */
  reconnectNow(): void {
    if (this.revoked) return
    this.stopped = false
    this.parked = false
    this.cancelReconnect()
    this.cancelConnectTimer()
    this.cancelHeartbeat()
    this.generation += 1
    this.dropSocket(1000, 'client reconnect')
    this.failPending(new HubRequestError('disconnected', 'Connection restarted', true))
    this.attempt = 0
    this.consecutiveTimeouts = 0
    this.setStatus({ state: 'connecting' })
    void this.open()
  }

  /**
   * Page visibility. While hidden no reconnect timer is kept (a suspended
   * phone would only pile them up); a reconnect that falls due is parked
   * until `resume`.
   */
  setVisible(visible: boolean): void {
    this.visible = visible
    if (visible) return
    this.cancelHeartbeat()
    if (this.reconnectTimer !== undefined) {
      this.cancelReconnect()
      this.parked = true
    }
  }

  /**
   * The app came back (foreground, page restored, network back). A broken
   * link reconnects at once with the backoff reset; a link that looks live is
   * probed, because a socket that died while the phone slept still reads live.
   */
  resume(): void {
    this.visible = true
    if (this.revoked || this.stopped) return
    const state = this.current.state
    if (state === 'reconnecting' || this.parked) { this.reconnectNow(); return }
    if (state === 'live') void this.verifyLink(undefined, { notify: true })
  }

  /**
   * Probe the live link with a cheap request and a short timeout. When it does
   * not answer the socket is dropped and a new one opened, so a dead socket
   * never keeps showing `live`. Concurrent calls share one probe.
   *
   * `notify: true` (a return to the foreground: `resume`, the controller's
   * `resync`) tells `onLinkVerified` listeners when the link answered, so they
   * re-read what may have changed meanwhile. The heartbeat's own probes stay
   * silent: a quiet, visible app must not re-read everything every beat.
   */
  verifyLink(timeoutMs?: number, options: { readonly notify?: boolean } = {}): Promise<LinkCheck> {
    if (this.revoked || this.stopped) return Promise.resolve('skipped')
    if (this.current.state === 'reconnecting') { this.reconnectNow(); return Promise.resolve('reconnecting') }
    if (this.current.state !== 'live') return Promise.resolve('skipped')
    // A shared probe notifies when any of its callers asked for it.
    if (options.notify === true) this.probeNotifies = true
    if (this.probe) return this.probe
    const generation = this.generation
    const probe = this.rawRequest('hub.hello', { protocol: HUB_PROTOCOL_VERSION, client: 'xerxes-mobile' }, timeoutMs ?? this.options.probeTimeoutMs ?? DEFAULT_PROBE_TIMEOUT_MS)
      .then((): LinkCheck => 'live', (error: unknown): LinkCheck => {
        const silent = !(error instanceof HubRequestError) || error.code === 'timeout' || error.code === 'disconnected' || error.code === 'offline'
        // Any answer from the Hub, even a refusal, proves the socket carries frames.
        if (!silent) return 'live'
        if (this.stopped || this.revoked) return 'skipped'
        // The dead-link rule may already have restarted the link for this very timeout.
        if (generation !== this.generation) return 'reconnecting'
        this.lastError = 'The connection stopped answering'
        this.reconnectNow()
        return 'reconnecting'
      })
      .then(result => {
        const notify = this.probeNotifies
        this.probeNotifies = false
        if (result === 'live') {
          if (this.heartbeatTimer === undefined) this.scheduleHeartbeat()
          if (notify) for (const listener of [...this.resumeListeners]) listener()
        }
        return result
      })
      .finally(() => { if (this.probe === probe) this.probe = undefined })
    this.probe = probe
    return probe
  }

  /**
   * Called after a foreground probe (`verifyLink(…, { notify: true })`, `resume`) confirmed the
   * existing link (no reconnect, so no `live` transition). Heartbeat probes never call it.
   */
  onLinkVerified(listener: () => void): () => void {
    this.resumeListeners.add(listener)
    return () => { this.resumeListeners.delete(listener) }
  }

  close(): void {
    this.stopped = true
    this.parked = false
    this.generation += 1
    this.cancelReconnect()
    this.cancelConnectTimer()
    this.cancelHeartbeat()
    this.dropSocket(1000, 'client closed')
    this.failPending(new HubRequestError('disconnected', 'Connection closed', true))
    for (const sub of this.subs) this.detach(sub)
    if (!this.revoked) this.setStatus({ state: 'closed' })
  }

  /** Send one request on the live link. Fails fast when offline. */
  request(method: HubMethod, params: Readonly<Record<string, unknown>> = {}, options: { readonly timeoutMs?: number } = {}): Promise<Readonly<Record<string, unknown>>> {
    if (this.current.state !== 'live' && this.current.state !== 'replaying') {
      return Promise.reject(new HubRequestError('offline', 'Offline. Reconnect to continue.'))
    }
    return this.rawRequest(method, params, options.timeoutMs)
  }

  /**
   * Send one mutation exactly once. When the acknowledgement is lost the
   * outcome is asked for by command id; the command is never sent again.
   */
  async command(input: CommandInput): Promise<HubCommandResult> {
    const commandId = input.commandId ?? newCommandId()
    if (this.current.state !== 'live') {
      return { ok: false, command_id: commandId, state: 'rejected', error: { code: 'unavailable', message: 'Offline. Nothing was sent.' } }
    }
    const params: Record<string, unknown> = {
      workspace_id: input.workspaceId,
      command_id: commandId,
      action: input.action,
      args: input.args,
      ...(input.sessionId === undefined ? {} : { session_id: input.sessionId }),
      ...(input.expectedRevision === undefined ? {} : { expected_revision: input.expectedRevision }),
    }
    try {
      return parseCommandResult(await this.rawRequest('hub.command', params), commandId)
    } catch (error) {
      if (!(error instanceof HubRequestError)) throw error
      if (!error.uncertain) {
        return { ok: false, command_id: commandId, state: 'rejected', error: { code: error.code as HubErrorCode, message: error.message } }
      }
      return this.commandOutcome(commandId, { workspaceId: input.workspaceId, ...(input.sessionId === undefined ? {} : { sessionId: input.sessionId }) })
    }
  }

  /** Ask the Hub what happened to a command. Never re-sends it. */
  async commandOutcome(commandId: string, scope: { readonly workspaceId: string; readonly sessionId?: string }): Promise<HubCommandResult> {
    const statusParams = { workspace_id: scope.workspaceId, command_id: commandId, ...(scope.sessionId === undefined ? {} : { session_id: scope.sessionId }) }
    const unknown: HubCommandResult = {
      ok: false,
      command_id: commandId,
      state: 'unknown',
      error: { code: 'unavailable', message: 'Could not confirm whether this was applied. Check the task before trying again.' },
    }
    if (!(await this.waitForLive(this.options.commandStatusWaitMs ?? 20_000))) return unknown
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        const result = parseCommandResult(await this.rawRequest('hub.command.status', statusParams), commandId)
        if (result.state !== 'unknown') return result
        return unknown
      } catch (error) {
        if (!(error instanceof HubRequestError) || !error.uncertain) return unknown
        if (!(await this.waitForLive(this.options.commandStatusWaitMs ?? 20_000))) return unknown
      }
    }
    return unknown
  }

  subscribe(spec: SubscriptionSpec, handlers: SubscriptionHandlers, initial?: { readonly cursor?: string }): SubscriptionHandle {
    const sub: Sub = {
      spec,
      handlers,
      tracker: this.options.contiguousSequences === true ? new EventCursorTracker() : new ScopedCursorTracker(),
      baseCursor: initial?.cursor,
      applied: undefined,
      watermark: undefined,
      replaying: initial?.cursor === undefined,
      serverId: undefined,
      resetting: false,
      resetPromise: undefined,
      disposed: false,
      starting: undefined,
    }
    this.subs.add(sub)
    if (this.current.state === 'live' || this.current.state === 'replaying') void this.ensureStarted(sub)
    return {
      dispose: () => {
        if (sub.disposed) return
        const serverId = sub.serverId
        this.detach(sub)
        if (serverId && this.socket) {
          void this.rawRequest('hub.events.unsubscribe', { subscription_id: serverId }).catch(() => undefined)
        }
      },
      cursor: () => sub.applied ?? sub.tracker.cursor ?? sub.baseCursor,
    }
  }

  /**
   * The position "now" in a scope: subscribe, read the watermark, unsubscribe.
   * Events after it can be replayed by cursor, so a snapshot taken after this
   * call plus a subscription resumed from it loses nothing.
   */
  async watermark(spec: SubscriptionSpec): Promise<string> {
    // events.read answers with a cursor bound to this exact scope, which a
    // subscription over the same scope accepts. One event at most is returned.
    const result = await this.request('hub.events.read', {
      workspace_ids: spec.workspaceIds,
      ...(spec.sessionIds === undefined ? {} : { session_ids: spec.sessionIds }),
      limit: 1,
    })
    const mark = str(result.watermark)
    if (!mark) throw new HubRequestError('internal', 'The Hub did not report a stream position.')
    return mark
  }

  /** A bridge view for one workspace or session; shares this connection. */
  bridge(scope: BridgeScope): ClientBridge {
    const unsubscribers = new Set<() => void>()
    const track = (unsubscribe: () => void): (() => void) => {
      unsubscribers.add(unsubscribe)
      return () => { unsubscribers.delete(unsubscribe); unsubscribe() }
    }
    return {
      scope,
      call: (method, params = {}) => {
        if (!isHubMethod(method)) return Promise.reject(new HubRequestError('invalid_request', `unknown hub method ${method}`))
        const scoped = SCOPED_METHODS.has(method)
          ? { workspace_id: scope.workspaceId, ...(scope.sessionId === undefined ? {} : { session_id: scope.sessionId }), ...params }
          : params
        return this.request(method, scoped)
      },
      onEvent: listener => track(this.onEvent(event => {
        if (event.workspace_id !== scope.workspaceId) return
        if (scope.sessionId !== undefined && event.session_id !== scope.sessionId) return
        listener(event)
      })),
      onStatus: listener => track(this.onStatus(listener)),
      status: () => this.current,
      dispose: () => { for (const unsubscribe of [...unsubscribers]) unsubscribe(); unsubscribers.clear() },
    }
  }

  // ── connection lifecycle ───────────────────────────────────────────

  private async open(): Promise<void> {
    const generation = ++this.generation
    if (this.options.ensureSession) {
      let check: SessionCheck
      try { check = await this.options.ensureSession() } catch { check = 'unavailable' }
      if (generation !== this.generation || this.stopped) return
      if (check === 'denied') {
        this.markRevoked('Signed out. Pair this device again.')
        return
      }
    }
    if (generation !== this.generation || this.stopped) return
    this.cancelConnectTimer()
    // A socket that never opens (a half-dead network, a captive portal) must not hold
    // 'connecting' forever: the whole handshake has to reach 'live' within the deadline.
    this.connectTimer = this.timers.setTimeout(() => {
      this.connectTimer = undefined
      if (generation !== this.generation || this.stopped || this.revoked || this.current.state === 'live') return
      this.generation += 1
      this.lastError = 'The Hub did not answer'
      this.dropSocket(1000, 'connect timeout')
      this.failPending(new HubRequestError('disconnected', 'Connection timed out', true))
      this.scheduleReconnect()
    }, this.options.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS)
    this.socket = this.factory(this.options.url, {
      open: () => { if (generation === this.generation) void this.onOpen(generation) },
      message: data => { if (generation === this.generation) this.onMessage(data) },
      binary: () => { if (generation === this.generation) this.report(new HubProtocolError('invalid_request', 'binary frames are not part of the hub protocol')) },
      close: (code, reason) => { if (generation === this.generation) this.onClose(code, reason) },
      error: () => { if (generation === this.generation) this.lastError = 'Connection error' },
    })
  }

  private cancelConnectTimer(): void {
    if (this.connectTimer !== undefined) {
      this.timers.clearTimeout(this.connectTimer)
      this.connectTimer = undefined
    }
  }

  /** While visible and live, probe the link whenever no frame arrived during one heartbeat period. */
  private scheduleHeartbeat(): void {
    const period = this.options.heartbeatMs
    if (period === undefined || period <= 0) return
    this.cancelHeartbeat()
    this.framesSinceBeat = false
    this.heartbeatTimer = this.timers.setTimeout(() => {
      this.heartbeatTimer = undefined
      if (this.stopped || this.revoked || this.current.state !== 'live' || !this.visible) return
      // A quiet link is probed; a successful probe schedules the next beat.
      if (!this.framesSinceBeat) void this.verifyLink()
      else this.scheduleHeartbeat()
    }, period)
  }

  private cancelHeartbeat(): void {
    if (this.heartbeatTimer !== undefined) {
      this.timers.clearTimeout(this.heartbeatTimer)
      this.heartbeatTimer = undefined
    }
  }

  private async onOpen(generation: number): Promise<void> {
    this.setStatus({ state: 'replaying' })
    try {
      const result = await this.rawRequest('hub.hello', { protocol: HUB_PROTOCOL_VERSION, client: 'xerxes-mobile' })
      const principal = rec(result.principal)
      const userId = str(principal.id)
      const displayName = str(principal.display_name)
      this.hello = { ...(userId ? { userId } : {}), ...(displayName ? { displayName } : {}) }
    } catch (error) {
      if (generation !== this.generation) return
      if (error instanceof HubRequestError && (error.code === 'unauthorized' || error.code === 'forbidden')) {
        // The 5-minute browser session can lapse between the refresh and the hello (a suspended
        // phone). Refresh once and retry before deciding the device is signed out.
        if (this.options.ensureSession && !this.helloRetried) {
          this.helloRetried = true
          let check: SessionCheck
          try { check = await this.options.ensureSession() } catch { check = 'unavailable' }
          if (generation !== this.generation || this.stopped) return
          if (check === 'denied') { this.markRevoked('Signed out. Pair this device again.'); return }
          this.lastError = 'Session expired; reconnecting'
          this.dropSocket(1000, 'session refresh')
          this.scheduleReconnect()
          return
        }
        this.markRevoked('Signed out. Pair this device again.')
        return
      }
      this.lastError = error instanceof Error ? error.message : 'Hello failed'
      this.dropSocket(1000, 'hello failed')
      this.scheduleReconnect()
      return
    }
    if (generation !== this.generation) return
    await Promise.all([...this.subs].map(async sub => {
      await this.ensureStarted(sub)
      await sub.resetPromise
    }))
    if (generation !== this.generation || this.stopped) return
    this.cancelConnectTimer()
    this.attempt = 0
    this.consecutiveTimeouts = 0
    this.helloRetried = false
    this.lastError = undefined
    this.setStatus({ state: 'live' })
    this.scheduleHeartbeat()
  }

  private onClose(code: number, reason: string): void {
    this.cancelConnectTimer()
    this.cancelHeartbeat()
    this.socket = undefined
    this.byServerId.clear()
    for (const sub of this.subs) { sub.serverId = undefined }
    this.failPending(new HubRequestError('disconnected', 'Connection lost', true))
    if (this.stopped || this.revoked) {
      if (!this.revoked) this.setStatus({ state: 'closed' })
      return
    }
    this.lastError = reason || `Connection closed (${code})`
    this.scheduleReconnect()
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.revoked) return
    this.cancelReconnect()
    this.attempt += 1
    this.cancelHeartbeat()
    this.cancelConnectTimer()
    this.setStatus({ state: 'reconnecting', attempt: this.attempt, ...(this.lastError === undefined ? {} : { lastError: this.lastError }) })
    if (!this.visible) {
      // A hidden page keeps no timer: `resume` reconnects at once when the app comes back.
      this.parked = true
      return
    }
    const delay = backoffDelay(this.attempt, this.options.backoff ?? DEFAULT_BACKOFF, this.random)
    this.reconnectTimer = this.timers.setTimeout(() => {
      this.reconnectTimer = undefined
      if (!this.visible) { this.parked = true; return }
      void this.open()
    }, delay)
  }

  private cancelReconnect(): void {
    if (this.reconnectTimer !== undefined) {
      this.timers.clearTimeout(this.reconnectTimer)
      this.reconnectTimer = undefined
    }
  }

  private markRevoked(reason: string): void {
    this.revoked = true
    this.stopped = true
    this.generation += 1
    this.parked = false
    this.cancelReconnect()
    this.cancelConnectTimer()
    this.cancelHeartbeat()
    this.dropSocket(1000, 'revoked')
    this.failPending(new HubRequestError('disconnected', 'Connection closed', true))
    for (const sub of this.subs) this.detach(sub)
    this.setStatus({ state: 'revoked', reason })
  }

  private dropSocket(code: number, reason: string): void {
    const socket = this.socket
    this.socket = undefined
    this.byServerId.clear()
    for (const sub of this.subs) sub.serverId = undefined
    if (socket) {
      try { socket.close(code, reason) } catch { /* already closing */ }
    }
  }

  private setStatus(next: ConnectionStatus): void {
    const previous = this.current
    if (previous.state === next.state && JSON.stringify(previous) === JSON.stringify(next)) return
    this.current = next
    for (const listener of [...this.statusListeners]) listener(next)
  }

  private waitForLive(timeoutMs: number): Promise<boolean> {
    if (this.current.state === 'live') return Promise.resolve(true)
    if (this.current.state === 'revoked' || this.current.state === 'closed') return Promise.resolve(false)
    return new Promise(resolve => {
      let timer: unknown
      const done = (value: boolean): void => {
        this.timers.clearTimeout(timer)
        unsubscribe()
        resolve(value)
      }
      const unsubscribe = this.onStatus(status => {
        if (status.state === 'live') done(true)
        else if (status.state === 'revoked' || status.state === 'closed') done(false)
      })
      timer = this.timers.setTimeout(() => done(false), timeoutMs)
    })
  }

  // ── requests ────────────────────────────────────────────────────────

  private rawRequest(method: HubMethod, params: Readonly<Record<string, unknown>>, timeoutMs?: number): Promise<Readonly<Record<string, unknown>>> {
    const socket = this.socket
    if (!socket) return Promise.reject(new HubRequestError('offline', 'Offline. Reconnect to continue.'))
    const id = `c${++this.nextRequest}`
    const frame: HubRequestFrame = { v: HUB_PROTOCOL_VERSION, kind: 'request', id, method, params }
    let text: string
    try {
      text = encodeHubFrame(frame, this.maxFrameBytes)
    } catch (error) {
      const code = error instanceof HubProtocolError ? error.code : 'invalid_request'
      return Promise.reject(new HubRequestError(code, error instanceof Error ? error.message : 'invalid frame'))
    }
    return new Promise((resolve, reject) => {
      const generation = this.generation
      const timer = this.timers.setTimeout(() => {
        this.pending.delete(id)
        reject(new HubRequestError('timeout', `${method} timed out`, true))
        // Two silent requests in a row on a link that reads live: the socket is dead.
        if (generation !== this.generation || this.current.state !== 'live') return
        this.consecutiveTimeouts += 1
        if (this.consecutiveTimeouts >= DEAD_LINK_TIMEOUTS && !this.stopped && !this.revoked) {
          this.lastError = 'The connection stopped answering'
          this.reconnectNow()
        }
      }, timeoutMs ?? this.options.requestTimeoutMs ?? 15_000)
      this.pending.set(id, { method, resolve, reject, timer })
      try {
        socket.send(text)
      } catch {
        this.timers.clearTimeout(timer)
        this.pending.delete(id)
        reject(new HubRequestError('disconnected', 'Connection lost', true))
      }
    })
  }

  private failPending(error: HubRequestError): void {
    const entries = [...this.pending.values()]
    this.pending.clear()
    for (const entry of entries) {
      this.timers.clearTimeout(entry.timer)
      entry.reject(error)
    }
  }

  // ── inbound frames ──────────────────────────────────────────────────

  private onMessage(text: string): void {
    let frame
    try {
      frame = parseHubServerFrame(text, this.maxFrameBytes)
    } catch (error) {
      this.report(error instanceof Error ? error : new Error('unreadable frame'))
      return
    }
    // Any frame proves the link carries traffic.
    this.framesSinceBeat = true
    this.consecutiveTimeouts = 0
    if (frame.kind === 'response') {
      if (frame.id === null) {
        if (!frame.ok) this.report(new HubProtocolError(frame.error.code, frame.error.message))
        return
      }
      const entry = this.pending.get(frame.id)
      if (!entry) return
      this.pending.delete(frame.id)
      this.timers.clearTimeout(entry.timer)
      if (frame.ok) entry.resolve(frame.result)
      else entry.reject(new HubRequestError(frame.error.code, frame.error.message))
      return
    }
    switch (frame.type) {
      case 'hub.event_batch': this.onBatch(frame.payload); return
      case 'hub.stream_gap': this.onGap(frame.payload); return
      case 'hub.revoked':
        this.markRevoked(str(frame.payload.reason) ?? 'This device was signed out.')
        return
      default:
        for (const listener of [...this.noticeListeners]) {
          try { listener(frame.type, frame.payload) } catch (error) { this.report(error instanceof Error ? error : new Error('notice listener failed')) }
        }
    }
  }

  private onBatch(payload: Readonly<Record<string, unknown>>): void {
    const sub = this.byServerId.get(str(payload.subscription_id) ?? '')
    if (!sub || sub.resetting || sub.disposed) return
    for (const raw of arr(payload.events)) {
      const event = asHubEvent(raw)
      if (!event) {
        this.report(new HubProtocolError('invalid_request', 'malformed event in batch'))
        void this.reset(sub, 'apply_failed')
        return
      }
      const outcome = sub.tracker.classify(event)
      if (outcome.kind === 'duplicate') continue
      if (outcome.kind === 'gap') {
        void this.reset(sub, 'sequence_gap')
        return
      }
      try {
        sub.handlers.apply(event, { replay: sub.replaying })
      } catch (error) {
        this.report(error instanceof Error ? error : new Error('apply failed'))
        void this.reset(sub, 'apply_failed')
        return
      }
      sub.tracker.commit(event)
      for (const listener of [...this.eventListeners]) {
        try { listener(event) } catch (error) { this.report(error instanceof Error ? error : new Error('event listener failed')) }
      }
    }
    // Ack only after every event in the batch was applied.
    const through = str(payload.through_cursor)
    if (through) sub.applied = through
    if (through && sub.replaying && through === sub.watermark) {
      sub.replaying = false
      sub.handlers.caughtUp?.()
    }
    const serverId = sub.serverId
    if (through && serverId) {
      void this.rawRequest('hub.events.ack', { subscription_id: serverId, cursor: through }).catch(() => undefined)
    }
  }

  private onGap(payload: Readonly<Record<string, unknown>>): void {
    const sub = this.byServerId.get(str(payload.subscription_id) ?? '')
    if (!sub || sub.resetting || sub.disposed) return
    const reason = str(payload.reason)
    const known: ReadonlySet<string> = new Set(['retention', 'stream_replaced', 'journal_unavailable', 'scope_changed'])
    void this.reset(sub, (reason && known.has(reason) ? reason : 'retention') as HubGapReason, str(payload.watermark))
  }

  // ── subscriptions ───────────────────────────────────────────────────

  private ensureStarted(sub: Sub, resume?: { readonly preferred: string | undefined; readonly fallback: string | undefined }): Promise<void> {
    if (sub.disposed || sub.resetting || sub.serverId) return Promise.resolve()
    sub.starting ??= this.startSub(sub, resume).finally(() => { sub.starting = undefined })
    return sub.starting
  }

  private async startSub(sub: Sub, resume?: { readonly preferred: string | undefined; readonly fallback: string | undefined }): Promise<void> {
    const preferred = resume ? resume.preferred : sub.applied ?? sub.tracker.cursor ?? sub.baseCursor
    const attempt = async (cursor: string | undefined): Promise<Readonly<Record<string, unknown>>> => {
      const params: Record<string, unknown> = {
        workspace_ids: sub.spec.workspaceIds,
        ...(sub.spec.sessionIds === undefined ? {} : { session_ids: sub.spec.sessionIds }),
        ...(cursor === undefined ? {} : { after_cursor: cursor }),
      }
      return this.rawRequest('hub.events.subscribe', params)
    }
    let result: Readonly<Record<string, unknown>>
    try {
      try {
        result = await attempt(preferred)
      } catch (error) {
        // A cursor from another scope is refused; fall back to the watermark we know.
        if (error instanceof HubRequestError && error.code === 'invalid_request' && preferred !== undefined) {
          // Never retry without a cursor: that would replay the whole journal.
          if (resume === undefined) {
            void this.reset(sub, 'scope_changed')
            return
          }
          if (resume.fallback === undefined || resume.fallback === preferred) throw error
          result = await attempt(resume.fallback)
        } else throw error
      }
    } catch (error) {
      if (error instanceof HubRequestError) sub.handlers.error?.(error)
      return
    }
    const serverId = str(result.subscription_id)
    if (sub.disposed) {
      // Left while the subscribe was in flight: cancel it on the Hub so it does not leak there.
      if (serverId) void this.rawRequest('hub.events.unsubscribe', { subscription_id: serverId }).catch(() => undefined)
      return
    }
    if (!serverId) {
      sub.handlers.error?.(new HubRequestError('internal', 'subscribe returned no subscription id'))
      return
    }
    sub.serverId = serverId
    this.byServerId.set(serverId, sub)
    const watermark = str(result.watermark)
    sub.watermark = watermark
    // An empty journal has nothing to replay: its oldest anchor is the watermark itself.
    if (sub.replaying && watermark !== undefined && str(result.oldest_cursor) === watermark) {
      sub.replaying = false
      sub.handlers.caughtUp?.()
    }
    if (result.reset_required === true && resume === undefined) {
      // Not awaited: the reset re-enters ensureStarted, which would wait on this very call.
      void this.reset(sub, 'retention', str(result.watermark))
    }
  }

  private reset(sub: Sub, reason: ResetReason, watermark?: string): Promise<void> {
    if (sub.resetting || sub.disposed) return sub.resetPromise ?? Promise.resolve()
    sub.resetting = true
    const run: Promise<void> = this.runReset(sub, reason, watermark).finally(() => {
      if (sub.resetPromise === run) sub.resetPromise = undefined
    })
    sub.resetPromise = run
    return run
  }

  private async runReset(sub: Sub, reason: ResetReason, watermark?: string): Promise<void> {
    const oldId = sub.serverId
    if (oldId) {
      this.byServerId.delete(oldId)
      sub.serverId = undefined
      void this.rawRequest('hub.events.unsubscribe', { subscription_id: oldId }).catch(() => undefined)
    }
    let boundary: ResetBoundary | undefined
    try {
      boundary = await sub.handlers.reset({ reason })
    } catch (error) {
      this.report(error instanceof Error ? error : new Error('snapshot reset failed'))
    }
    if (sub.disposed) return
    if (boundary?.sequence !== undefined && boundary.cursor !== undefined) {
      sub.tracker.reset({ sequence: boundary.sequence, cursor: boundary.cursor })
    } else {
      sub.tracker.reset()
    }
    const fromStart = boundary?.fromStart === true
    sub.baseCursor = fromStart ? undefined : boundary?.cursor ?? watermark
    sub.replaying = sub.baseCursor === undefined
    sub.applied = undefined
    sub.resetting = false
    // A reset triggered by the subscribe reply itself runs while that subscribe is still settling;
    // restarting before it settles would be skipped as "already starting".
    await sub.starting
    if (sub.disposed) return
    if (this.current.state === 'live' || this.current.state === 'replaying') {
      await this.ensureStarted(sub, { preferred: sub.baseCursor, fallback: fromStart ? undefined : watermark })
    }
  }

  private detach(sub: Sub): void {
    sub.disposed = true
    if (sub.serverId) this.byServerId.delete(sub.serverId)
    sub.serverId = undefined
    this.subs.delete(sub)
  }

  private report(error: Error): void {
    this.options.onProtocolError?.(error)
  }
}
