// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { basename } from 'node:path'

import { SHARED_LESSON_LIMITS } from '../memory/sharedLessonWire.js'
import { HubDaemonClient, DaemonRpcError, DaemonTransportError } from './daemonClient.js'
import type { DaemonCallOptions, DaemonHandshake, HubDaemonClientOptions } from './daemonClient.js'
import { HubOperatorError, isPrivateProviderType, parseUpstreamEvent } from './ports.js'
import type { HubEventSink, HubTunnel, UpstreamEvent } from './ports.js'
import { HUB_LIMITS, isRecord } from './protocol.js'
import type { WorkspaceHealth } from './protocol.js'
import { emptyProjection, markProjectionInterrupted, markProjectionStale, reduceProjectionEvent, replaceSessionSummaries } from './projections.js'
import type { WorkspaceProjection } from './projections.js'
import { openSshTunnel } from './sshTunnel.js'

export type PoolEndpoint = { readonly kind: 'local'; readonly socketPath: string } | { readonly kind: 'ssh'; readonly sshAlias: string; readonly remoteSocketPath?: string }
/** One way to reach a daemon. `primary` is the entry's own endpoint; the others are fallbacks. */
export interface PoolRoute {
  readonly via: 'primary' | 'hub' | 'mac'
  readonly endpoint: PoolEndpoint
}
export interface PoolWorkspace {
  readonly id: string
  readonly label: string
  readonly endpoint: PoolEndpoint
  /**
   * Fallback routes to the same daemon (machine entries only), best first. With routes, every
   * (re)connect tries the endpoint and then each route in order; a route that reaches a different
   * runtime than the one already verified is never used.
   */
  readonly routes?: readonly PoolRoute[]
  readonly projectDir: string
  readonly autoConnect: boolean
  readonly bootstrapConsent?: boolean
  readonly updateConsent?: boolean
  /**
   * A machine entry: its projectDir (`/`) is never used for routing. The pool asks the daemon which
   * project folders it knows and registers one derived workspace per folder (see `derivedWorkspaceId`).
   */
  readonly discover?: boolean
}
export interface PoolHealth {
  readonly state: WorkspaceHealth
  readonly degraded: boolean
  readonly message?: string
  readonly runtimeId?: string
  readonly instanceId?: string
}
export interface PoolGapNotice {
  readonly runtimeId: string
  readonly workspaceIds: readonly string[]
  readonly reason: string
  readonly watermark?: string
}
export interface PoolInterruptedNotice {
  readonly runtimeId: string
  readonly workspaceIds: readonly string[]
  readonly previousInstanceId: string
  readonly instanceId: string
}
export type PoolCallResult =
  | { readonly state: 'accepted'; readonly result: Readonly<Record<string, unknown>> }
  | { readonly state: 'rejected' | 'unknown'; readonly error: { readonly code: string; readonly message: string } }

/** One daemon the Hub reaches: a `discover` machine entry, or the group of per-project entries that share a daemon. */
export interface PoolMachine {
  /** The discover entry's id, or the smallest workspace id of a group of per-project entries. */
  readonly id: string
  readonly label: string
  readonly health: WorkspaceHealth
  /** The daemon cannot verify signed Hub commands. */
  readonly readOnly: boolean
  /** The daemon advertised `hub_models_v1_supported`. */
  readonly modelsSupported: boolean
  /** Routable workspaces on this machine (never the discover entry itself). */
  readonly workspaceIds: readonly string[]
  /** The route in use, when the machine has fallback routes. */
  readonly route?: PoolRoute['via']
  /** The daemon's verified runtime identity, once known. */
  readonly runtimeId?: string
}
export interface PoolDerivedWorkspace {
  readonly id: string
  readonly label: string
  readonly projectDir: string
  readonly machineId: string
}

/** Internal RPC ceiling, not an authorization substitute. No shell/slash/provider/config/shutdown passthrough. */
export const HUB_DAEMON_METHODS = [
  'runtime.status', 'session.active_list', 'session.list', 'session.status', 'session.history', 'session.search',
  'usage.report', 'subagent.inspect', 'session.open', 'turn.submit', 'turn.steer', 'turn.cancel',
  'permission_response', 'question_response', 'session.goal', 'set_model', 'subagent.steer',
] as const
export type HubDaemonMethod = (typeof HUB_DAEMON_METHODS)[number]
/** Session-less methods that run the Hub's XERXES supervisor model on a daemon's own provider profiles. */
export const HUB_MODEL_DAEMON_METHODS = ['hub.models.catalog', 'hub.models.options', 'hub.model.complete'] as const
const MODEL_METHODS: ReadonlySet<string> = new Set(HUB_MODEL_DAEMON_METHODS)
/** A completion is capped at 120 s on the daemon; leave room for the round trip. */
const MODEL_CALL_TIMEOUT_MS = 130_000
/** Separate from generic RPCs: only the lesson synchronizer may use this path. */
export const HUB_LESSON_DAEMON_METHODS = ['lessons.shared.list', 'lessons.shared.apply'] as const
export type HubLessonDaemonMethod = (typeof HUB_LESSON_DAEMON_METHODS)[number]
const LESSON_METHODS: ReadonlySet<string> = new Set(HUB_LESSON_DAEMON_METHODS)
export const LESSON_CALL_TIMEOUT_MS = 30_000
export interface PoolLessonWorkspace {
  readonly id: string
  readonly projectDir: string
  readonly runtimeId: string
}
/** Most derived workspaces one machine may hold, and how many saved sessions one discovery pass reads. */
export const MAX_DERIVED_WORKSPACES = 100
const DISCOVERY_SESSION_LIMIT = 300
const MACHINE_POLL_MS = 10_000

/**
 * Stable id of the workspace derived from a machine entry and a project folder. Deterministic, so grants,
 * invites and bookmarks keep working across Hub restarts.
 */
export function derivedWorkspaceId(machineId: string, projectDir: string): string {
  return `m-${machineId.slice(0, 8)}-${createHash('sha256').update(projectDir).digest('hex').slice(0, 12)}`
}
/** An absolute, single-line project folder from a daemon row, without a trailing slash; the root itself is no project. */
function cleanDirectory(value: unknown): string | undefined {
  if (typeof value !== 'string' || !value.startsWith('/') || value.length > 4096 || /[\0\r\n]/.test(value)) return undefined
  const trimmed = value.length > 1 ? value.replace(/\/+$/, '') : value
  return trimmed === '' || trimmed === '/' ? undefined : trimmed
}
const METHODS: ReadonlySet<string> = new Set(HUB_DAEMON_METHODS)
const READS: ReadonlySet<string> = new Set(['runtime.status', 'session.active_list', 'session.list', 'session.status', 'session.history', 'session.search', 'usage.report', 'subagent.inspect'])
const UNSCOPED: ReadonlySet<string> = new Set(['runtime.status', 'session.active_list', 'session.list', 'session.search'])
const SUMMARIES: ReadonlySet<string> = new Set(['session.active_list', 'session.status', 'session.open', 'session.list'])
/** Opening a session only attaches a lane; everything else outside READS changes daemon state. */
const ATTACH_ONLY: ReadonlySet<string> = new Set(['session.open'])
export const READ_ONLY_MESSAGE = 'This workspace runs a Xerxes version that cannot verify signed Hub commands, so the Hub only reads it. Update Xerxes on that machine to send, steer, approve or change it from here.'
export interface HubRuntimePoolOptions {
  readonly sink: HubEventSink
  readonly clientOptions?: Omit<HubDaemonClientOptions, 'socketPath' | 'resolveSocketPath' | 'projectDir' | 'sessionKey'>
  readonly pollIntervalMs?: number
  /** Metadata is shared; bounded, lazy session lanes keep interaction ownership independent. */
  readonly maxSessionConnections?: number
  readonly openTunnel?: (workspace: PoolWorkspace, signal: AbortSignal, onFailure: (error: Error) => void) => Promise<HubTunnel>
  readonly onHealth?: (workspaceId: string, health: PoolHealth) => void
  readonly onProjection?: (workspaceId: string, projection: WorkspaceProjection) => void
  readonly onGap?: (notice: PoolGapNotice) => void
  readonly onInterrupted?: (notice: PoolInterruptedNotice) => void
  /** How often a machine on a fallback route checks whether a better route is back (default 30 s). */
  readonly betterRouteProbeMs?: number
  /** Live text/thinking deltas of one session are merged for up to this long before they are stored (default 150 ms). */
  readonly coalesceMs?: number
}
/**
 * Delta event types merged by the Hub. Every spelling is normalized to the canonical
 * `text_part {text}` / `think_part {think}` the phone renders; `sources` lists the payload
 * fields a spelling may carry its text in, first match wins.
 */
interface DeltaKind { readonly type: 'text_part' | 'think_part'; readonly field: 'text' | 'think'; readonly sources: readonly string[] }
const DELTA_KINDS: Readonly<Record<string, DeltaKind>> = {
  text_part: { type: 'text_part', field: 'text', sources: ['text'] },
  text_delta: { type: 'text_part', field: 'text', sources: ['text', 'delta'] },
  think_part: { type: 'think_part', field: 'think', sources: ['think'] },
  thinking_part: { type: 'think_part', field: 'think', sources: ['think', 'thinking', 'text'] },
  thinking_delta: { type: 'think_part', field: 'think', sources: ['thinking', 'think', 'delta', 'text'] },
}
/** Text deltas change no projection field but the row's timestamps; refresh those at most this often per task. */
const DELTA_PROJECTION_INTERVAL_MS = 5000
/** The canonical delta an upstream event carries, or undefined when it is not a text/thinking delta. */
export function normalizeDelta(event: UpstreamEvent): { readonly event: UpstreamEvent; readonly field: 'text' | 'think'; readonly text: string } | undefined {
  const kind = DELTA_KINDS[event.type]
  if (kind === undefined) return undefined
  const source = kind.sources.find(name => typeof event.payload[name] === 'string')
  if (source === undefined) return undefined
  const text = event.payload[source] as string
  const rest: Record<string, unknown> = {}
  for (const [name, value] of Object.entries(event.payload)) if (!kind.sources.includes(name)) rest[name] = value
  return { event: { ...event, type: kind.type, payload: { ...rest, [kind.field]: text } }, field: kind.field, text }
}
/** A merged delta is stored once it holds this many characters, whatever its age. */
const COALESCE_MAX_CHARS = 16_000
interface DeltaBuffer {
  readonly key: string
  readonly field: 'text' | 'think'
  readonly first: UpstreamEvent
  last: UpstreamEvent
  text: string
  readonly generation: number
}
interface SessionLane {
  readonly client: HubDaemonClient
  inFlight: number
  usedAt: number
}
interface Lane {
  readonly client: HubDaemonClient
  readonly controls: Map<string, SessionLane>
  socketPath: string | undefined
  readonly workspaces: Map<string, PoolWorkspace>
  readonly abort: AbortController
  readonly endpointKeys: Set<string>
  active: boolean
  everOnline: boolean
  runtimeId: string
  verified: boolean
  /** The daemon advertised principal_auth_v1: it verifies hub_auth on every mutation. */
  signedCommands: boolean
  /** The daemon advertised hub_models_v1: it can run XERXES's model calls. */
  modelsSupported: boolean
  /** The daemon advertised lessons_sync_v1_supported. */
  lessonsSupported: boolean
  /** Last health published for this lane; derived workspaces start from it. */
  health: PoolHealth | undefined
  /** Project dirs the current event subscription covers; a change means resubscribe. */
  subscribedKey: string | undefined
  instanceId: string | undefined
  subscriptionId: string | undefined
  streamId: string | undefined
  degraded: boolean
  tail: Promise<void>
  poll: ReturnType<typeof setTimeout> | undefined
  tunnel: HubTunnel | undefined
  generation: number
  queuedEventBytes: number
  /**
   * Text/thinking deltas received but not stored yet. Their cursor is not committed, so a
   * reconnect drops the buffer and the daemon replays them from the committed cursor.
   */
  deltas: DeltaBuffer | undefined
  deltaTimer: ReturnType<typeof setTimeout> | undefined
  /** The entry this lane's client was opened for; its routes are the ones tried. */
  readonly entry: PoolWorkspace
  /** Index into `routesOf(entry)` of the route this lane last picked. */
  routeIndex: number
  /** The picked route reached a verified, online daemon. Until then the next pick starts after it. */
  routeConfirmed: boolean
  /** Route index -> time until which it is skipped (it reached a different runtime). */
  readonly badRoutes: Map<number, number>
  /** Earliest time to check whether a better route than the current one is back. */
  betterProbeAt: number
}

/** The entry's own endpoint first, then its fallback routes. */
export function routesOf(workspace: PoolWorkspace): readonly PoolRoute[] {
  return [{ via: 'primary', endpoint: workspace.endpoint }, ...(workspace.routes ?? [])]
}
const BETTER_ROUTE_PROBE_MS = 30_000
const BAD_ROUTE_MS = 5 * 60_000

/** Resolves when something accepts a connection on the Unix socket; a stale tunnel socket refuses. */
export async function probeSocket(path: string, timeoutMs = 1500): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    let settled = false
    const done = (error?: Error): void => {
      if (settled) return
      settled = true; clearTimeout(timer)
      if (error) reject(error); else resolve()
    }
    const timer = setTimeout(() => done(new Error('the socket did not answer')), timeoutMs)
    void Bun.connect<undefined>({ unix: path, socket: {
      open(socket) { socket.end(); done() },
      data() {},
      error(_socket, error) { done(error) },
      connectError(_socket, error) { done(error) },
    } }).catch((error: unknown) => done(error instanceof Error ? error : new Error(String(error))))
  })
}

/** One metadata/event lane per VERIFIED runtime; endpoint equality alone never merges distinct runtimes. */
export class HubRuntimePool {
  private readonly lanes = new Set<Lane>()
  private readonly runtimes = new Map<string, Lane>()
  private readonly endpoints = new Map<string, Lane>()
  private readonly workspaces = new Map<string, PoolWorkspace>()
  private readonly owners = new Map<string, Lane>()
  /** machine entry id -> ids of the workspaces derived from it. */
  private readonly derived = new Map<string, Set<string>>()
  private readonly healthByWorkspace = new Map<string, PoolHealth>()
  private readonly projections = new Map<string, WorkspaceProjection>()
  /** workspace + task -> when a text/thinking delta last reached its projection (bounded LRU). */
  private readonly deltaProjectedAt = new Map<string, number>()
  private stopped = false
  constructor(private readonly options: HubRuntimePoolOptions) {}
  get runtimeCount(): number { return this.lanes.size }
  health(workspaceId: string): PoolHealth | undefined { return this.healthByWorkspace.get(workspaceId) }
  projection(workspaceId: string): WorkspaceProjection | undefined { return this.projections.get(workspaceId) }
  /** True while the workspace's daemon cannot verify signed commands (reads only). */
  readOnly(workspaceId: string): boolean {
    const lane = this.owners.get(workspaceId)
    return lane === undefined || !lane.signedCommands
  }

  async start(workspaces: readonly PoolWorkspace[]): Promise<void> {
    await this.addWorkspaces(workspaces)
  }

  /** Register workspaces (all validated first) and connect the auto-connect ones. Used at start and on registry changes. */
  async addWorkspaces(workspaces: readonly PoolWorkspace[]): Promise<void> {
    if (this.stopped) throw new Error('Runtime pool has stopped')
    const ids = new Set<string>()
    for (const workspace of workspaces) {
      if (this.workspaces.has(workspace.id) || ids.has(workspace.id)) throw new Error(`Duplicate workspace id: ${workspace.id}`)
      if (!workspace.projectDir.startsWith('/') || /[\0\r\n]/.test(workspace.projectDir)) throw new Error('Workspace projectDir must be an absolute path')
      ids.add(workspace.id)
    }
    // Register before discovery, then connect independently: one unreachable
    // SSH host must not delay healthy local/remote workspaces coming online.
    for (const workspace of workspaces) {
      this.workspaces.set(workspace.id, workspace); this.projections.set(workspace.id, emptyProjection(workspace.id))
      this.setHealth(workspace.id, { state: 'offline', degraded: false })
    }
    await Promise.all(workspaces.filter(workspace => workspace.autoConnect).map(workspace => this.connectWorkspace(workspace.id)))
  }

  /**
   * Forget a workspace removed from the registry. Its lane is closed only when no
   * other workspace still shares it; the daemon itself is never told to stop.
   */
  async removeWorkspace(workspaceId: string): Promise<void> {
    if (!this.workspaces.has(workspaceId)) return
    const lane = this.owners.get(workspaceId)
    // A machine takes the workspaces derived from it along.
    for (const id of this.derived.get(workspaceId) ?? []) {
      this.workspaces.delete(id); this.owners.delete(id); this.healthByWorkspace.delete(id); this.projections.delete(id)
      lane?.workspaces.delete(id)
    }
    this.derived.delete(workspaceId)
    this.workspaces.delete(workspaceId); this.owners.delete(workspaceId)
    this.healthByWorkspace.delete(workspaceId); this.projections.delete(workspaceId)
    if (!lane) return
    lane.workspaces.delete(workspaceId)
    if (lane.workspaces.size > 0) return
    lane.active = false; lane.abort.abort(); clearTimeout(lane.poll); lane.client.close()
    for (const control of lane.controls.values()) control.client.close()
    lane.controls.clear()
    this.lanes.delete(lane)
    for (const key of lane.endpointKeys) if (this.endpoints.get(key) === lane) this.endpoints.delete(key)
    if (lane.verified && this.runtimes.get(lane.runtimeId) === lane) this.runtimes.delete(lane.runtimeId)
    await lane.tunnel?.close()
  }
  workspaceIds(): readonly string[] { return [...this.workspaces.keys()] }
  async connectWorkspace(workspaceId: string): Promise<void> {
    const workspace = this.workspaces.get(workspaceId)
    if (!workspace || this.stopped) return
    const endpointKey = JSON.stringify(workspace.endpoint)
    const existing = this.endpoints.get(endpointKey)
    if (existing) {
      existing.workspaces.set(workspaceId, workspace); this.owners.set(workspaceId, existing)
      if (existing.everOnline && existing.client.online) this.enqueue(existing, () => this.configure(existing))
      else {
        try { await existing.client.reconnect() } catch { /* explicit retry reports health; no command is replayed */ }
      }
      await existing.tail; return
    }
    const abort = new AbortController()
    let lane: Lane
    const routed = (workspace.routes?.length ?? 0) > 0
    const client = new HubDaemonClient({ ...this.options.clientOptions, projectDir: workspace.projectDir,
      ...(workspace.endpoint.kind === 'local' && !routed ? { socketPath: workspace.endpoint.socketPath } : {
        resolveSocketPath: async () => {
          if (routed) return this.pickRoute(lane, workspace, abort.signal)
          await lane.tunnel?.close(); lane.tunnel = undefined
          lane.tunnel = await this.openTunnelFor(lane, workspace, abort.signal)
          lane.socketPath = lane.tunnel.socketPath
          return lane.tunnel.socketPath
        },
      }) })
    lane = { client, controls: new Map(), socketPath: workspace.endpoint.kind === 'local' && !routed ? workspace.endpoint.socketPath : undefined,
      workspaces: new Map([[workspace.id, workspace]]), abort, endpointKeys: new Set([endpointKey]), active: true,
      everOnline: false, runtimeId: `unverified:${workspace.id}`, verified: false, signedCommands: false, modelsSupported: false, lessonsSupported: false, health: undefined, subscribedKey: undefined, instanceId: undefined, subscriptionId: undefined, streamId: undefined,
      degraded: true, tail: Promise.resolve(), poll: undefined, tunnel: undefined, generation: 0, queuedEventBytes: 0,
      deltas: undefined, deltaTimer: undefined,
      entry: workspace, routeIndex: 0, routeConfirmed: true, badRoutes: new Map(), betterProbeAt: 0 }
    this.lanes.add(lane); this.endpoints.set(endpointKey, lane); this.owners.set(workspace.id, lane)
    client.onState((state, error) => {
      if (!lane.active || this.stopped) return
      if (state === 'online') return // setup, journal resume and snapshot must finish before admission
      lane.generation++
      clearTimeout(lane.poll); lane.poll = undefined; lane.subscriptionId = undefined
      // Uncommitted deltas come back from the committed cursor after the reconnect.
      this.dropDeltas(lane)
      if (state === 'needs_operator') this.setLaneHealth(lane, 'needs_operator', error?.message)
      else if (state === 'disconnected') this.setLaneHealth(lane, lane.everOnline ? 'stale' : 'offline', error?.message)
      else if (state === 'connecting') this.setLaneHealth(lane, lane.everOnline ? 'stale' : 'connecting')
      for (const id of lane.workspaces.keys()) this.publish(id, markProjectionStale(this.projections.get(id)!))
    })
    client.onHandshake(handshake => {
      const generation = lane.generation
      this.enqueue(lane, async () => { if (generation === lane.generation) await this.handshake(lane, handshake) })
    })
    client.onEvent((type, payload) => {
      if (type !== 'event_batch' && type !== 'event_gap') return
      const generation = lane.generation, bytes = Buffer.byteLength(JSON.stringify(payload))
      if (lane.queuedEventBytes + bytes > 4 * HUB_LIMITS.maxFrameBytes) {
        client.interrupt(new Error('Hub upstream event queue exceeded its byte limit; resuming from the committed cursor'))
        return
      }
      lane.queuedEventBytes += bytes
      this.enqueue(lane, async () => {
        try {
          if (generation !== lane.generation || payload.subscription_id !== lane.subscriptionId) return
          if (type === 'event_gap') { await this.gap(lane, payload); await this.backfill(lane, typeof payload.watermark === 'string' ? payload.watermark : undefined) }
          else await this.ingest(lane, payload)
        } finally { lane.queuedEventBytes -= bytes }
      })
    })
    try { await client.connect(); await lane.tail } catch { /* client state carries the error; retries stay isolated to this lane */ }
  }

  async call(workspaceId: string, method: string, params: Readonly<Record<string, unknown>> = {}, options: DaemonCallOptions = {}): Promise<PoolCallResult> {
    const reject = (code: string, message: string): PoolCallResult => ({ state: 'rejected', error: { code, message } })
    if (!METHODS.has(method)) return reject('forbidden', `Daemon method is not in the Hub allowlist: ${method}`)
    const workspace = this.workspaces.get(workspaceId), lane = this.owners.get(workspaceId)
    if (!workspace || !lane || this.stopped || !lane.client.online || this.health(workspaceId)?.state !== 'online') return reject('unavailable', 'Workspace is not online; reconnect before sending commands')
    // Every session operation is explicitly scoped. Navigation can never retarget
    // another person's command through a shared connection's active session.
    if (!UNSCOPED.has(method) && (typeof params.session_key !== 'string' || !params.session_key)) return reject('invalid_request', 'An explicit internal session_key is required')
    // A daemon that cannot verify hub_auth would run any mutation unattributed and
    // unauthorized. Refuse before anything is written (fail closed, retry-safe).
    if (!READS.has(method) && !ATTACH_ONLY.has(method) && !lane.signedCommands) return reject('read_only', READ_ONLY_MESSAGE)
    const scoped = { ...params, project_dir: workspace.projectDir, ...(SUMMARIES.has(method) ? { history_limit: 0 } : {}) }
    let control: SessionLane | undefined
    let submitted = false
    try {
      if (options.signal?.aborted) return reject('cancelled', 'Command cancelled before submission')
      if (!UNSCOPED.has(method)) {
        control = this.sessionLane(lane, workspace, String(params.session_key))
        control.inFlight++; control.usedAt = Date.now()
        const handshake = await control.client.connect()
        if (lane.verified && handshake.status.runtime_id !== lane.runtimeId) {
          control.client.close()
          throw new Error('Session lane runtime identity does not match its verified metadata lane')
        }
      }
      // Opening a session lane is only a handshake, never a mutation replay.
      if (!lane.client.online || this.health(workspaceId)?.state !== 'online') return reject('unavailable', 'Workspace disconnected during command admission')
      submitted = true
      const result = await (control?.client ?? lane.client).call(method, scoped, options)
      if (result.ok === false) return reject(typeof result.code === 'string' ? result.code : 'daemon_rejected', typeof result.error === 'string' ? result.error : 'Daemon rejected command')
      return { state: 'accepted', result }
    } catch (error) {
      if (submitted && !READS.has(method) && error instanceof DaemonTransportError && error.written) return { state: 'unknown', error: { code: 'outcome_unknown', message: 'The request was written but no definitive answer arrived. Query its receipt; do not replay this mutation.' } }
      return reject(error instanceof DaemonRpcError ? 'daemon_rejected' : 'unavailable', error instanceof Error ? error.message : String(error))
    } finally { if (control) control.inFlight-- }
  }
  private sessionLane(lane: Lane, workspace: PoolWorkspace, sessionKey: string): SessionLane {
    const key = JSON.stringify([workspace.id, sessionKey])
    const existing = lane.controls.get(key)
    if (existing) return existing
    if (lane.controls.size >= (this.options.maxSessionConnections ?? 8)) {
      const oldest = [...lane.controls].filter(([, control]) => control.inFlight === 0).sort((a, b) => a[1].usedAt - b[1].usedAt)[0]
      if (!oldest) throw new Error('All bounded session-control lanes are busy; retry after one request settles')
      oldest[1].client.close(); lane.controls.delete(oldest[0])
    }
    const client = new HubDaemonClient({ ...this.options.clientOptions, projectDir: workspace.projectDir, sessionKey,
      resolveSocketPath: async () => {
        if (!lane.active || !lane.client.online || !lane.socketPath) throw new Error('Metadata transport is reconnecting')
        return lane.socketPath
      } })
    const control: SessionLane = { client, inFlight: 0, usedAt: Date.now() }
    lane.controls.set(key, control)
    return control
  }
  async stop(): Promise<void> {
    if (this.stopped) return
    this.stopped = true
    const closing = [...this.lanes].map(async lane => {
      lane.active = false; lane.abort.abort(); clearTimeout(lane.poll); this.dropDeltas(lane); lane.client.close()
      for (const control of lane.controls.values()) control.client.close()
      lane.controls.clear()
      await lane.tunnel?.close(); await lane.tail
    })
    await Promise.all(closing)
    this.lanes.clear(); this.runtimes.clear(); this.endpoints.clear(); this.owners.clear()
    for (const id of this.workspaces.keys()) { this.setHealth(id, { state: 'offline', degraded: false }); this.publish(id, markProjectionStale(this.projections.get(id)!)) }
  }
  /** Open the SSH tunnel for one endpoint; a later transport failure interrupts the lane. */
  private openTunnelFor(lane: Lane, entry: PoolWorkspace, signal: AbortSignal): Promise<HubTunnel> {
    const open = this.options.openTunnel ?? ((target, abort, onFailure) => {
      if (target.endpoint.kind !== 'ssh') throw new Error('SSH endpoint required')
      return openSshTunnel({ sshAlias: target.endpoint.sshAlias,
        ...(target.endpoint.remoteSocketPath === undefined ? {} : { remoteSocketPath: target.endpoint.remoteSocketPath }),
        bootstrapConsent: target.bootstrapConsent === true, signal: abort, onFailure })
    })
    const routed = (lane.entry.routes?.length ?? 0) > 0
    return open(entry, signal, error => {
      if (!lane.active) return
      // With other routes left, an operator condition on one route is not a reason to stop trying.
      const reported = routed && error instanceof HubOperatorError ? new Error(error.message) : error
      for (const control of lane.controls.values()) control.client.interrupt(reported)
      lane.client.interrupt(reported)
    })
  }

  /**
   * The socket of the first route that answers, best first. After a pick that never came online the
   * search starts at the next route, so one broken daemon behind a live socket cannot pin the lane.
   * Never throws an operator error: the client keeps retrying while any route might come back.
   */
  private async pickRoute(lane: Lane, workspace: PoolWorkspace, signal: AbortSignal): Promise<string> {
    const all = routesOf(workspace)
    await lane.tunnel?.close(); lane.tunnel = undefined; lane.socketPath = undefined
    const now = Date.now()
    const start = lane.routeConfirmed ? 0 : (lane.routeIndex + 1) % all.length
    const problems: string[] = []
    for (let step = 0; step < all.length; step++) {
      const index = (start + step) % all.length
      const route = all[index]!
      if ((lane.badRoutes.get(index) ?? 0) > now) { problems.push(`${route.via}: reaches another runtime`); continue }
      signal.throwIfAborted()
      try {
        let socketPath: string
        if (route.endpoint.kind === 'local') {
          await probeSocket(route.endpoint.socketPath)
          socketPath = route.endpoint.socketPath
        } else {
          lane.tunnel = await this.openTunnelFor(lane, { ...workspace, endpoint: route.endpoint }, signal)
          socketPath = lane.tunnel.socketPath
        }
        lane.routeIndex = index; lane.routeConfirmed = false; lane.socketPath = socketPath
        return socketPath
      } catch (error) {
        if (signal.aborted) throw error
        problems.push(`${route.via}: ${error instanceof Error ? error.message.slice(0, 160) : String(error)}`)
      }
    }
    lane.routeConfirmed = true
    throw new Error(`No route to ${workspace.label} answered (${problems.join('; ')})`)
  }

  /** On a fallback route: whether a better local route answers again (checked at most every 30 s). */
  private async betterRouteBack(lane: Lane): Promise<boolean> {
    if (lane.routeIndex === 0 || (lane.entry.routes?.length ?? 0) === 0 || Date.now() < lane.betterProbeAt) return false
    lane.betterProbeAt = Date.now() + (this.options.betterRouteProbeMs ?? BETTER_ROUTE_PROBE_MS)
    const all = routesOf(lane.entry)
    for (let index = 0; index < lane.routeIndex; index++) {
      const route = all[index]!
      if (route.endpoint.kind !== 'local' || (lane.badRoutes.get(index) ?? 0) > Date.now()) continue
      try { await probeSocket(route.endpoint.socketPath); return true } catch { /* still down */ }
    }
    return false
  }

  private enqueue(lane: Lane, operation: () => Promise<void>): void {
    lane.tail = lane.tail.then(async () => { if (lane.active && !this.stopped) await operation() }).catch((error: unknown) => {
      if (lane.active && !this.stopped) lane.client.interrupt(error instanceof Error ? error : new Error(String(error)))
    })
  }
  private async handshake(lane: Lane, handshake: DaemonHandshake): Promise<void> {
    const runtimeId = typeof handshake.status.runtime_id === 'string' && handshake.status.runtime_id ? handshake.status.runtime_id : undefined
    // A fallback route must reach the daemon this machine already is. Anything else (a reused tunnel
    // socket, a different host behind an alias) is skipped for a while; the next route is tried.
    if (runtimeId && lane.verified && runtimeId !== lane.runtimeId && lane.routeIndex > 0 && (lane.entry.routes?.length ?? 0) > 0) {
      lane.badRoutes.set(lane.routeIndex, Date.now() + BAD_ROUTE_MS)
      throw new Error(`The ${routesOf(lane.entry)[lane.routeIndex]?.via ?? 'fallback'} route to ${lane.entry.label} reaches a different Xerxes runtime; it is skipped`)
    }
    if (runtimeId) {
      const existing = this.runtimes.get(runtimeId)
      if (existing && existing !== lane && existing.client.online) {
        lane.active = false; lane.abort.abort(); lane.client.close(); await lane.tunnel?.close(); this.lanes.delete(lane)
        for (const [id, workspace] of lane.workspaces) { existing.workspaces.set(id, workspace); this.owners.set(id, existing) }
        for (const key of lane.endpointKeys) { existing.endpointKeys.add(key); this.endpoints.set(key, existing) }
        this.enqueue(existing, () => this.configure(existing)); await existing.tail; return
      }
      if (existing && existing !== lane) {
        // The lane that owned this runtime cannot reach it (its socket moved, e.g. a
        // re-linked tunnel); this one just did. Keep the live lane and fold the dead
        // one into it. Folding the other way closed the only working connection.
        existing.active = false; existing.abort.abort(); clearTimeout(existing.poll); existing.client.close()
        for (const control of existing.controls.values()) control.client.close()
        existing.controls.clear(); this.lanes.delete(existing)
        await existing.tunnel?.close()
        for (const [id, workspace] of existing.workspaces) { lane.workspaces.set(id, workspace); this.owners.set(id, lane) }
        for (const key of existing.endpointKeys) { lane.endpointKeys.add(key); this.endpoints.set(key, lane) }
        this.runtimes.delete(runtimeId)
      }
      if (lane.verified && lane.runtimeId !== runtimeId) this.runtimes.delete(lane.runtimeId)
      lane.runtimeId = runtimeId; lane.verified = true; this.runtimes.set(runtimeId, lane)
    }
    const instance = typeof handshake.status.instance_id === 'string' ? handshake.status.instance_id : undefined
    if (instance && lane.instanceId && instance !== lane.instanceId) {
      for (const id of lane.workspaces.keys()) this.publish(id, markProjectionInterrupted(this.projections.get(id)!))
      this.options.onInterrupted?.({ runtimeId: lane.runtimeId, workspaceIds: [...lane.workspaces.keys()], previousInstanceId: lane.instanceId, instanceId: instance })
    }
    lane.instanceId = instance
    lane.signedCommands = handshake.status.principal_auth_v1_supported === true
    lane.modelsSupported = handshake.status.hub_models_v1_supported === true
    lane.lessonsSupported = handshake.status.lessons_sync_v1_supported === true
    lane.degraded = handshake.status.event_stream_supported !== true || !lane.verified
    await this.configure(lane)
  }
  private async configure(lane: Lane): Promise<void> {
    if (!lane.active || !lane.client.online) return
    clearTimeout(lane.poll); lane.poll = undefined
    // Resubscribing resumes from the committed cursor: store buffered deltas first so none is received twice.
    await this.flushDeltas(lane)
    await this.summaries(lane)
    // Machine entries ('/') are never subscribed: the daemon matches project dirs exactly. With no
    // project known yet there is nothing to subscribe to; polling finds the first one.
    const dirs = this.subscriptionDirs(lane)
    lane.subscribedKey = dirs.join('\0')
    if (!lane.degraded && dirs.length > 0) {
      if (lane.subscriptionId) await lane.client.call('events.unsubscribe', { subscription_id: lane.subscriptionId })
      lane.subscriptionId = undefined
      const cursor = this.options.sink.upstreamCursor(lane.runtimeId)
      let subscribed: Record<string, unknown>
      // Conversation detail: phones show assistant text and thinking live. The Hub merges the
      // deltas (see ingest) so the journal and the phones receive a bounded number of events.
      try { subscribed = await lane.client.call('events.subscribe', { project_dirs: dirs, detail: 'conversation', max_batch_bytes: HUB_LIMITS.normalFrameBytes, ...(cursor === undefined ? {} : { after_cursor: cursor }) }) }
      catch (error) { if (error instanceof DaemonRpcError && error.code === -32601) subscribed = { ok: false }; else throw error }
      if (subscribed.ok !== true || typeof subscribed.subscription_id !== 'string') {
        if (subscribed.ok === false && (subscribed.error === undefined || /unknown method|unsupported/i.test(String(subscribed.error)))) lane.degraded = true
        else throw new Error('Daemon refused event subscription')
      } else {
        if (typeof subscribed.stream_id !== 'string' || !subscribed.stream_id) throw new Error('Daemon subscription has no stream identity')
        lane.subscriptionId = subscribed.subscription_id; lane.streamId = subscribed.stream_id
        if (subscribed.reset_required === true) await this.gap(lane, { ...subscribed, reason: 'stream_replaced' })
        const after = subscribed.reset_required === true ? subscribed.watermark : cursor
        await this.backfill(lane, typeof after === 'string' ? after : undefined)
      }
    }
    lane.everOnline = true; lane.routeConfirmed = true; this.setLaneHealth(lane, 'online')
    this.schedulePoll(lane)
  }
  private async backfill(lane: Lane, after: string | undefined): Promise<void> {
    if (!lane.subscriptionId) return
    let cursor = after
    // Byte-bounded pages and a bounded catch-up budget. If the journal keeps
    // growing faster than the sink, reconnect from the committed cursor rather
    // than advertise a caught-up lane or retain unbounded in-memory events.
    for (let page = 0; page < 256; page++) {
      const backfill = await lane.client.call('events.read', { subscription_id: lane.subscriptionId,
        limit: HUB_LIMITS.maxBatchEvents, max_bytes: HUB_LIMITS.normalFrameBytes, ...(cursor === undefined ? {} : { after_cursor: cursor }) })
      if (backfill.ok !== true) throw new Error('Daemon event backfill failed')
      if (backfill.reset_required === true) {
        await this.gap(lane, backfill)
        if (typeof backfill.watermark !== 'string' || backfill.watermark === cursor) throw new Error('Daemon journal cannot recover after a gap')
        cursor = backfill.watermark
        continue
      }
      await this.ingest(lane, backfill)
      if (backfill.has_more !== true) return
      const next = backfill.next_cursor ?? backfill.through_cursor
      if (typeof next !== 'string' || next === cursor) throw new Error('Daemon backfill did not advance its cursor')
      cursor = next
    }
    throw new Error('Daemon journal catch-up exceeded the bounded page budget')
  }
  private schedulePoll(lane: Lane): void {
    if (!lane.active || this.stopped || !lane.client.online) return
    lane.poll = setTimeout(() => {
      lane.poll = undefined
      this.enqueue(lane, async () => {
        // Back on the best route as soon as it answers: reconnecting re-picks from the top.
        if (await this.betterRouteBack(lane)) { lane.client.interrupt(new Error('A better route is back; switching to it')); return }
        await this.summaries(lane)
        // A project found by this poll joins the event subscription.
        if (!lane.degraded && this.subscriptionDirs(lane).join('\0') !== lane.subscribedKey) await this.configure(lane)
        else this.schedulePoll(lane)
      })
    }, this.options.pollIntervalMs ?? (lane.degraded ? 5000 : this.hasMachine(lane) ? MACHINE_POLL_MS : 30_000))
    lane.poll.unref?.()
  }
  private hasMachine(lane: Lane): boolean { return [...lane.workspaces.values()].some(row => row.discover === true) }
  private subscriptionDirs(lane: Lane): string[] {
    return [...new Set([...lane.workspaces.values()].filter(row => row.discover !== true).map(row => row.projectDir))].sort()
  }

  /**
   * A machine entry knows no project folders up front: every distinct absolute cwd of its daemon's open and
   * saved chats becomes a workspace of its own (id from `derivedWorkspaceId`, same endpoint, same lane).
   * Subagent sessions are skipped (their cwd is often a worktree), and a folder an explicit per-project
   * entry already covers is not derived twice.
   */
  private async discover(lane: Lane, active: readonly unknown[]): Promise<void> {
    const machine = [...lane.workspaces.values()].filter(row => row.discover === true).sort((a, b) => a.id.localeCompare(b.id))[0]
    if (!machine) return
    let saved: readonly unknown[] = []
    try {
      const listed = await lane.client.call('session.list', { scope: 'global', limit: DISCOVERY_SESSION_LIMIT })
      if (listed.ok === true && Array.isArray(listed.sessions)) saved = listed.sessions
    } catch (error) {
      // An older daemon may not offer the global list; open chats still find projects.
      if (!(error instanceof DaemonRpcError)) throw error
    }
    // Any folder this lane already routes (explicit, or derived from another machine entry that
    // reaches the same runtime) is not derived again, so one runtime never lists a project twice.
    const explicit = new Set([...lane.workspaces.values()].filter(row => row.discover !== true).map(row => row.projectDir))
    const known = this.derived.get(machine.id) ?? new Set<string>()
    const fresh = new Map<string, PoolWorkspace>()
    for (const row of [...active, ...saved]) {
      if (!isRecord(row)) continue
      if ((typeof row.parent_session_id === 'string' && row.parent_session_id) || row.kind === 'subagent' || row.session_kind === 'subagent') continue
      const cwd = cleanDirectory(row.cwd)
      if (cwd === undefined || explicit.has(cwd)) continue
      const id = derivedWorkspaceId(machine.id, cwd)
      if (this.workspaces.has(id) || fresh.has(id)) continue
      if (known.size + fresh.size >= MAX_DERIVED_WORKSPACES) break
      fresh.set(id, {
        id, label: `${machine.label} · ${basename(cwd) || cwd}`, endpoint: machine.endpoint, projectDir: cwd, autoConnect: true,
        ...(machine.bootstrapConsent === undefined ? {} : { bootstrapConsent: machine.bootstrapConsent }),
        ...(machine.updateConsent === undefined ? {} : { updateConsent: machine.updateConsent }),
      })
    }
    if (fresh.size === 0) return
    for (const [id, workspace] of fresh) {
      known.add(id)
      this.workspaces.set(id, workspace); this.owners.set(id, lane); lane.workspaces.set(id, workspace)
      this.projections.set(id, emptyProjection(id))
      this.setHealth(id, lane.health ?? { state: 'connecting', degraded: lane.degraded })
    }
    this.derived.set(machine.id, known)
  }
  /** The folder of a workspace found on a machine entry; undefined for any other id. */
  derivedProjectDir(workspaceId: string): string | undefined {
    return this.isDerived(workspaceId) ? this.workspaces.get(workspaceId)?.projectDir : undefined
  }
  private isDerived(workspaceId: string): boolean {
    for (const ids of this.derived.values()) if (ids.has(workspaceId)) return true
    return false
  }

  /** Workspaces found on machine entries, with the machine they came from. */
  derivedWorkspaces(): readonly PoolDerivedWorkspace[] {
    const rows: PoolDerivedWorkspace[] = []
    for (const [machineId, ids] of this.derived) {
      for (const id of ids) {
        const workspace = this.workspaces.get(id)
        if (workspace) rows.push({ id, label: workspace.label, projectDir: workspace.projectDir, machineId })
      }
    }
    return rows
  }

  /** The daemons the Hub reaches. Per-project entries that share a daemon are one machine. */
  machines(): readonly PoolMachine[] {
    const rows: PoolMachine[] = []
    for (const lane of this.lanes) {
      const members = [...lane.workspaces.values()]
      if (members.length === 0) continue
      const entries = members.filter(row => row.discover === true)
      const head = [...(entries.length > 0 ? entries : members)].sort((a, b) => a.id.localeCompare(b.id))[0]!
      const route = (lane.entry.routes?.length ?? 0) > 0 ? routesOf(lane.entry)[lane.routeIndex]?.via : undefined
      rows.push({
        id: head.id, label: head.label, health: this.health(head.id)?.state ?? 'offline',
        readOnly: !lane.signedCommands, modelsSupported: lane.modelsSupported,
        workspaceIds: members.filter(row => row.discover !== true).map(row => row.id).sort(),
        ...(route === undefined ? {} : { route }), ...(lane.verified ? { runtimeId: lane.runtimeId } : {}),
      })
    }
    // A machine that has not connected (yet) is still a machine.
    for (const workspace of this.workspaces.values()) {
      if (workspace.discover !== true || this.owners.has(workspace.id)) continue
      rows.push({ id: workspace.id, label: workspace.label, health: 'offline', readOnly: true, modelsSupported: false, workspaceIds: [] })
    }
    return rows
  }

  /**
   * One of the three `hub.model*` daemon methods on a machine's daemon. The caller signs the request
   * (`params.hub_auth`); this method only routes it, bounds it in time and never reports `unknown`:
   * a completion is not a mutation, so a lost answer is simply an error.
   */
  async callModel(machineId: string, method: string, params: Readonly<Record<string, unknown>>, options: DaemonCallOptions = {}): Promise<PoolCallResult> {
    const reject = (code: string, message: string): PoolCallResult => ({ state: 'rejected', error: { code, message } })
    if (!MODEL_METHODS.has(method)) return reject('forbidden', `Daemon method is not a Hub model method: ${method}`)
    const lane = this.owners.get(machineId)
    if (!lane || this.stopped || !lane.client.online || this.health(machineId)?.state !== 'online') return reject('unavailable', 'The machine is not online')
    if (!lane.modelsSupported) return reject('unsupported', "This machine runs a Xerxes version that cannot run XERXES's model calls. Update Xerxes there.")
    try {
      const result = await lane.client.call(method, params, { timeoutMs: MODEL_CALL_TIMEOUT_MS, ...options })
      if (result.ok === false) return reject(typeof result.code === 'string' ? result.code : 'daemon_rejected', typeof result.error === 'string' ? result.error : 'The daemon refused the model request')
      return { state: 'accepted', result }
    } catch (error) {
      return reject(error instanceof DaemonRpcError ? 'daemon_rejected' : 'unavailable', error instanceof Error ? error.message : String(error))
    }
  }
  /** Online, verified, writable lesson lanes. Machine discovery entries are never projects. */
  lessonWorkspaces(): readonly PoolLessonWorkspace[] {
    const rows: PoolLessonWorkspace[] = []
    for (const [id, workspace] of this.workspaces) {
      const lane = this.owners.get(id)
      if (workspace.discover === true || !lane || this.stopped || !lane.verified || !lane.signedCommands
        || !lane.lessonsSupported || !lane.client.online || this.health(id)?.state !== 'online') continue
      rows.push({ id, projectDir: workspace.projectDir, runtimeId: lane.runtimeId })
    }
    return rows
  }

  /**
   * Session-less lesson RPCs only. The synchronizer includes project_dir BEFORE signing;
   * pinning the same value here prevents navigation or supplied params from retargeting it.
   * Applies are idempotent merges: a lost answer is an error, never an automatic replay.
   */
  async callLessons(workspaceId: string, method: string, params: Readonly<Record<string, unknown>>, options: DaemonCallOptions = {}): Promise<PoolCallResult> {
    const reject = (code: string, message: string): PoolCallResult => ({ state: 'rejected', error: { code, message } })
    if (!LESSON_METHODS.has(method)) return reject('forbidden', 'Daemon method is not a Hub lesson method')
    const workspace = this.workspaces.get(workspaceId), lane = this.owners.get(workspaceId)
    if (!workspace || workspace.discover === true || !lane || this.stopped || !lane.verified || !lane.client.online
      || this.health(workspaceId)?.state !== 'online') return reject('unavailable', 'Workspace is not online and verified')
    if (!lane.signedCommands) return reject('read_only', READ_ONLY_MESSAGE)
    if (!lane.lessonsSupported) return reject('unsupported', 'This daemon does not support shared lesson sync')
    if (params.project_dir !== workspace.projectDir || !isRecord(params.hub_auth)) return reject('invalid_request', 'Lesson requests must be signed with the workspace project_dir')
    const scoped = { ...params, project_dir: workspace.projectDir }
    if (method === 'lessons.shared.apply' && (!Array.isArray(params.lessons) || params.lessons.length > SHARED_LESSON_LIMITS.applyLessons)) return reject('invalid_request', 'Too many lessons in an apply batch')
    try {
      if (Buffer.byteLength(JSON.stringify(scoped)) > SHARED_LESSON_LIMITS.payloadBytes) return reject('invalid_request', 'Lesson request exceeds 1 MiB')
      const timeoutMs = options.timeoutMs === undefined || !Number.isFinite(options.timeoutMs)
        ? LESSON_CALL_TIMEOUT_MS : Math.max(1, Math.min(options.timeoutMs, LESSON_CALL_TIMEOUT_MS))
      const result = await lane.client.call(method, scoped, { ...options, timeoutMs })
      if (Buffer.byteLength(JSON.stringify(result)) > SHARED_LESSON_LIMITS.payloadBytes) return reject('invalid_response', 'Lesson reply exceeds 1 MiB')
      if (result.ok === false) return reject(typeof result.code === 'string' ? result.code : 'daemon_rejected', typeof result.error === 'string' ? result.error : 'Daemon refused lesson sync')
      return { state: 'accepted', result }
    } catch (error) {
      return reject(error instanceof DaemonRpcError ? 'daemon_rejected' : 'unavailable', error instanceof Error ? error.message : String(error))
    }
  }

  private async summaries(lane: Lane): Promise<void> {
    const result = await lane.client.call('session.active_list', { history_limit: 0 })
    if (result.ok !== true || !Array.isArray(result.sessions)) throw new Error('Daemon did not return session summaries')
    await this.discover(lane, result.sessions)
    for (const [id, workspace] of lane.workspaces) {
      const summaries = result.sessions.filter(row => isRecord(row) && (row.cwd === workspace.projectDir || row.project_dir === workspace.projectDir))
      this.publish(id, replaceSessionSummaries(this.projections.get(id)!, summaries, new Date().toISOString()))
    }
  }
  private async ingest(lane: Lane, payload: Record<string, unknown>): Promise<void> {
    if (!Array.isArray(payload.events)) throw new Error('Daemon event batch has no events array')
    let previousSequence = -1
    for (const value of payload.events) {
      if (isRecord(value) && typeof value.type === 'string' && (isPrivateProviderType(value.type)
        || (isRecord(value.payload) && value.payload.flow === 'provider'))) continue
      const event = parseUpstreamEvent(value)
      if (!event) throw new Error('Malformed daemon journal event; refusing to advance the upstream cursor')
      if (event.sequence <= previousSequence) throw new Error('Daemon journal batch is not ordered')
      previousSequence = event.sequence
      if (event.runtime_id !== lane.runtimeId) throw new Error('Daemon journal runtime identity changed without a new handshake')
      const normalized = normalizeDelta(event)
      if (normalized !== undefined) {
        const { field, text: delta, event: canonical } = normalized
        const { [field]: _text, ...rest } = canonical.payload
        const key = JSON.stringify([canonical.type, canonical.project_dir, canonical.session_id ?? null, rest])
        if (lane.deltas !== undefined && lane.deltas.key !== key) await this.flushDeltas(lane)
        if (lane.deltas === undefined) lane.deltas = { key, field, first: canonical, last: canonical, text: delta, generation: lane.generation }
        else { lane.deltas.last = canonical; lane.deltas.text += delta }
        if (lane.deltas.text.length >= COALESCE_MAX_CHARS) await this.flushDeltas(lane)
        continue
      }
      // Anything else keeps upstream order: buffered text is stored before it.
      await this.flushDeltas(lane)
      await this.store(lane, event)
    }
    this.scheduleDeltaFlush(lane)
  }
  /**
   * Persist and project one event. Preserve upstream order across workspace partitions: persisting
   * all of A before B could checkpoint past an uncommitted B event on a sink failure.
   */
  private async store(lane: Lane, event: UpstreamEvent): Promise<void> {
    for (const [id, workspace] of lane.workspaces) {
      if (event.project_dir !== workspace.projectDir) continue
      await this.options.sink.ingest(lane.runtimeId, id, [{ ...event, ...(lane.streamId === undefined ? {} : { stream_id: lane.streamId }) }])
      if (this.skipDeltaProjection(id, event)) continue
      this.publish(id, reduceProjectionEvent(this.projections.get(id)!, event, workspace.projectDir))
    }
  }
  /**
   * A merged text/thinking delta only moves its task's timestamps in the projection, yet
   * reducing it copies the whole projection (every ~150 ms per streaming task). Reduce one
   * such event per task per interval; any other event of the task reduces as before.
   * Watch's last activity is read from the stored events, not from here.
   */
  private skipDeltaProjection(workspaceId: string, event: UpstreamEvent): boolean {
    const key = `${workspaceId}\0${event.session_id ?? ''}`
    if (event.type !== 'text_part' && event.type !== 'think_part') { this.deltaProjectedAt.delete(key); return false }
    const now = Date.now()
    const last = this.deltaProjectedAt.get(key)
    if (last !== undefined && now - last < DELTA_PROJECTION_INTERVAL_MS) return true
    this.deltaProjectedAt.delete(key)
    this.deltaProjectedAt.set(key, now)
    if (this.deltaProjectedAt.size > 4096) this.deltaProjectedAt.delete(this.deltaProjectedAt.keys().next().value!)
    return false
  }
  /**
   * Store the buffered deltas as ONE event: the first delta's payload with the joined text, and the
   * last delta's id, sequence and cursor, so committing it commits every delta it holds.
   */
  private async flushDeltas(lane: Lane): Promise<void> {
    const buffer = lane.deltas
    clearTimeout(lane.deltaTimer); lane.deltaTimer = undefined
    if (buffer === undefined) return
    lane.deltas = undefined
    if (buffer.generation !== lane.generation) return
    await this.store(lane, { ...buffer.last, payload: { ...buffer.first.payload, [buffer.field]: buffer.text } })
  }
  private scheduleDeltaFlush(lane: Lane): void {
    if (lane.deltas === undefined || lane.deltaTimer !== undefined) return
    const generation = lane.generation
    lane.deltaTimer = setTimeout(() => {
      lane.deltaTimer = undefined
      this.enqueue(lane, async () => { if (generation === lane.generation) await this.flushDeltas(lane) })
    }, this.options.coalesceMs ?? 150)
    lane.deltaTimer.unref?.()
  }
  private dropDeltas(lane: Lane): void {
    clearTimeout(lane.deltaTimer); lane.deltaTimer = undefined; lane.deltas = undefined
  }
  private async gap(lane: Lane, payload: Record<string, unknown>): Promise<void> {
    await this.flushDeltas(lane)
    await this.summaries(lane)
    this.options.onGap?.({ runtimeId: lane.runtimeId, workspaceIds: [...lane.workspaces.keys()], reason: typeof payload.reason === 'string' ? payload.reason : 'retention',
      ...(typeof payload.watermark === 'string' ? { watermark: payload.watermark } : {}) })
  }
  private publish(id: string, projection: WorkspaceProjection): void { this.projections.set(id, projection); this.options.onProjection?.(id, projection) }
  private setHealth(id: string, health: PoolHealth): void {
    const before = this.healthByWorkspace.get(id)
    this.healthByWorkspace.set(id, health)
    if (JSON.stringify(before) !== JSON.stringify(health)) this.options.onHealth?.(id, health)
  }
  private setLaneHealth(lane: Lane, state: WorkspaceHealth, message?: string): void {
    const health: PoolHealth = { state, degraded: lane.degraded,
      ...(lane.verified ? { runtimeId: lane.runtimeId } : {}), ...(lane.instanceId === undefined ? {} : { instanceId: lane.instanceId }), ...(message === undefined ? {} : { message }) }
    lane.health = health
    for (const id of lane.workspaces.keys()) this.setHealth(id, health)
  }
}
