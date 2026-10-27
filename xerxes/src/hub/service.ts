// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { daemonPaths } from '../daemon/paths.js'
import { HubCommandRouter } from './commandRouter.js'
import { HubEventStore, type HubEventStoreOptions, type UpstreamEvent as StoredUpstreamEvent } from './eventStore.js'
import {
  resolveHubExposure, type HubExposure, type HubExposureConfig, type HubTunnelHandle, type HubTunnelRequest,
} from './exposure.js'
import { canSeeWorkspace, grantedSessionIds, hasScope, type HubGrant } from '../security/principal.js'
import type { AuthenticatedContext, HubIdentityPort } from './identityPort.js'
import type { HubHandle, HubWorkspaceHealth } from './lifecycle.js'
import { HubLessonSync } from './lessonSync.js'
import type { HubPaths } from './paths.js'
import type { HubEventSink, UpstreamEvent } from './ports.js'
import { HubRuntimePool, type PoolWorkspace, type HubRuntimePoolOptions, type PoolHealth } from './runtimePool.js'
import { HubServer, type HubServerOptions } from './server.js'
import type { HubSigner } from './signer.js'
import { SubscriptionManager, type SubscriptionManagerOptions } from './subscriptions.js'
import {
  SUPERVISOR_SETUP_HINT, SupervisorGrantOverlay, createHubSupervisor, createUnsupportedSupervisorHandlers, withDefaultSpendBound,
  type HubSupervisor, type PushLike, type SupervisorModelPort, type SupervisorQuotas,
} from './supervisor/index.js'
import { SupervisorModelControl } from './supervisor/modelControl.js'
import { messageToWire } from './supervisor/threads.js'
import { WorkspaceRegistry, type HubWorkspace } from './workspaceRegistry.js'
import { allowedModes, type VoiceRuntimeConfig } from './voice/config.js'
import { createHubVoiceRoute, type HubVoiceRoute } from './voice/route.js'

/**
 * Composition root of the Hub: WorkspaceRegistry -> HubRuntimePool (sink = the
 * durable HubEventStore) -> SubscriptionManager -> HubServer, plus the command
 * router. Identity and the signing key are injected; nothing here discovers
 * credentials.
 *
 * Without an identity port the service still serves GET /healthz but refuses
 * every other request (fail closed) and says so in `status()`.
 */

export type HubServiceLifecycle = Pick<HubHandle, 'registerShutdownHook' | 'setHttpReady'>

/**
 * The XERXES supervisor. `configured` carries an explicitly chosen cheap model;
 * `unconfigured` (or leaving the option out) makes every `supervisor.*` method
 * answer `unsupported` with a setup hint. There is never a fallback model.
 */
export type HubServiceSupervisorConfig =
  | {
      readonly kind: 'configured'
      readonly model: SupervisorModelPort
      readonly quotas?: Partial<SupervisorQuotas>
      readonly persona?: string
      /** Only when the runtime guarantees `model.set` is session-only. */
      readonly modelSetSessionOnly?: boolean
    }
  | { readonly kind: 'unconfigured'; readonly reason?: string }

export interface HubServiceOptions {
  readonly paths: HubPaths
  readonly signer: HubSigner
  /** Absent means fail closed. */
  readonly identity?: HubIdentityPort
  readonly exposure?: HubExposureConfig
  /** Explicitly configured tunnel (Tailscale Serve, Cloudflare). Never started implicitly. */
  readonly tunnel?: HubTunnelRequest
  /** Directory holding the built phone bundle. */
  readonly staticRoot?: string
  readonly registry?: WorkspaceRegistry
  /** From `runHub`: lets the service register its shutdown hook and report HTTP readiness. */
  readonly lifecycle?: HubServiceLifecycle
  readonly pool?: Partial<Omit<HubRuntimePoolOptions, 'sink' | 'onHealth' | 'onGap' | 'onInterrupted'>>
  readonly server?: Partial<Pick<HubServerOptions, 'clock' | 'maxBufferedBytes' | 'maxConnections' | 'maxConnectionsPerPrincipal' | 'rateLimit' | 'presenceTtlMs' | 'methodHandlers'>>
  readonly subscriptions?: Omit<SubscriptionManagerOptions, 'grantsForPrincipal'>
  readonly store?: HubEventStoreOptions
  readonly supervisor?: HubServiceSupervisorConfig
  /**
   * Voice ports and settings (createVoiceRuntimeConfig). The voice socket is served
   * only when identity, a configured supervisor and at least one enabled voice mode
   * are all present; otherwise the path is not mounted and the phone hides its talk button.
   */
  readonly voice?: VoiceRuntimeConfig
  /** Minimal-text push to a user (identity.admin.sendPush). Absent: no push is attempted. */
  readonly sendPush?: PushLike
  /** Live Hub-owner check (identity.admin.ownerUserId). XERXES Autopilot stays active only while its enabler is the owner. */
  readonly isHubOwner?: (userId: string) => boolean
  /** XERXES Autopilot timings; `checkInMs` comes from XERXES_HUB_AUTOPILOT_CHECKIN_MS (0 disables). */
  readonly autopilot?: { readonly checkInMs?: number; readonly cooldownMs?: number; readonly maxWakesPerHour?: number }
  /** XERXES Watch timings; `stallMs` comes from XERXES_HUB_WATCH_STALL_MS (at least 60000). */
  readonly watch?: {
    readonly stallMs?: number; readonly maxAnomalyWakesPerHour?: number; readonly maxReviewWakesPerHour?: number; readonly updateIntervalMs?: number
    readonly maxHubWakesPerHour?: number; readonly anomalyCooldownMs?: number
  }
  /** How often the workspace registry is re-read so added/removed workspaces apply live. 0 disables. Default 3000 ms. */
  readonly registryPollMs?: number
  readonly onError?: (error: unknown, context: string) => void
}

export interface HubServiceInfo {
  readonly port: number
  readonly hostname: string
  readonly origins: readonly string[]
  readonly publicOrigin?: string
}

export interface HubServiceStatus {
  readonly state: 'created' | 'running' | 'stopped'
  readonly identity: 'configured' | 'missing'
  readonly fail_closed: boolean
  readonly listening: boolean
  readonly message?: string
  readonly workspaces: readonly { readonly id: string; readonly health: string; readonly degraded: boolean }[]
}

export interface HubService {
  start(): Promise<HubServiceInfo>
  /** Idempotent. Never sends a daemon shutdown. */
  stop(): Promise<void>
  readonly listening: boolean
  /** Settles when the initial workspace connection attempts have finished. */
  readonly workspacesSettled: Promise<void>
  workspaceHealth(): readonly HubWorkspaceHealth[]
  status(): HubServiceStatus
  readonly server: HubServer | undefined
  readonly pool: HubRuntimePool | undefined
  readonly store: HubEventStore | undefined
  /** Internal, model-free shared lesson synchronization (not a phone RPC). */
  readonly lessonSync: HubLessonSync | undefined
  /** Present only when the supervisor is configured and the Hub has an identity subsystem. */
  readonly supervisor: HubSupervisor | undefined
}

/** Locate the built phone bundle next to the source tree or next to a bundled CLI. */
export function defaultStaticRoot(): string | undefined {
  for (const relative of ['../../dist/mobile', './mobile']) {
    const candidate = fileURLToPath(new URL(relative, import.meta.url))
    if (existsSync(candidate)) return candidate
  }
  return undefined
}

/** Adapt the durable store to the pool's sink contract (the pool tags each batch with its stream id). */
export function storeSink(store: HubEventStore): HubEventSink {
  return {
    ingest(runtimeId, workspaceId, events: readonly UpstreamEvent[]) {
      const last = events.at(-1)
      if (!last) return
      const streamId = last.stream_id
      if (streamId === undefined) throw new Error('Daemon journal batch has no stream identity; refusing to persist it')
      const batch: StoredUpstreamEvent[] = events.map(event => ({
        event_id: event.event_id, sequence: event.sequence, cursor: event.cursor, type: event.type, payload: event.payload,
        occurred_at: event.occurred_at, ...(event.session_id === undefined ? {} : { session_id: event.session_id }),
      }))
      store.ingest(runtimeId, workspaceId, { stream_id: streamId, events: batch, through_cursor: last.cursor, through_sequence: last.sequence })
    },
    upstreamCursor: runtimeId => store.upstreamCursor(runtimeId)?.last_cursor,
  }
}

/** Registry entry to pool workspace. Exported for tests: the SSH socket path must survive this step. */
export function toPoolWorkspace(workspace: HubWorkspace): PoolWorkspace {
  const endpoint: PoolWorkspace['endpoint'] = workspace.endpoint.kind === 'local'
    ? { kind: 'local', socketPath: workspace.endpoint.socketPath ?? daemonPaths(workspace.projectDir).socketPath }
    : {
        kind: 'ssh', sshAlias: workspace.endpoint.sshAlias,
        ...(workspace.endpoint.remoteSocketPath === undefined ? {} : { remoteSocketPath: workspace.endpoint.remoteSocketPath }),
      }
  return {
    id: workspace.id, label: workspace.label, endpoint, projectDir: workspace.projectDir,
    autoConnect: workspace.connection.autoConnect, bootstrapConsent: workspace.consent.bootstrap, updateConsent: workspace.consent.update,
    ...(workspace.connection.discover === true ? { discover: true } : {}),
    ...(workspace.routes === undefined || workspace.routes.length === 0 ? {} : {
      routes: workspace.routes.map(route => ({ via: route.via, endpoint: route.endpoint })),
    }),
  }
}

export function createHubService(options: HubServiceOptions): HubService {
  const exposure: HubExposure = resolveHubExposure(options.exposure)
  const registry = options.registry ?? new WorkspaceRegistry(options.paths.workspacesPath)
  const report = (error: unknown, context: string): void => {
    if (options.onError) options.onError(error, context)
    else console.error(`Hub ${context} failed`, error)
  }

  let state: 'created' | 'running' | 'stopped' = 'created'
  let store: HubEventStore | undefined
  let subscriptions: SubscriptionManager | undefined
  let pool: HubRuntimePool | undefined
  let router: HubCommandRouter | undefined
  let server: HubServer | undefined
  let supervisor: HubSupervisor | undefined
  let modelControl: SupervisorModelControl | undefined
  let lessonSync: HubLessonSync | undefined
  let tunnel: HubTunnelHandle | undefined
  let settled: Promise<void> = Promise.resolve()
  let stopping: Promise<void> | undefined
  interface KnownEntry { readonly id: string; readonly label: string; readonly projectDir: string; readonly machine: boolean }
  const known = new Map<string, KnownEntry>()
  /** Workspaces phones see: registry projects plus the ones found on machine entries. The machine entry itself is never listed. */
  const listed = (): { readonly id: string; readonly label: string }[] => [
    ...[...known.values()].filter(row => !row.machine).map(row => ({ id: row.id, label: row.label })),
    ...(pool?.derivedWorkspaces() ?? []).map(row => ({ id: row.id, label: row.label })),
  ]
  const projectDirOf = (workspaceId: string): string | undefined => {
    const entry = known.get(workspaceId)
    if (entry !== undefined) return entry.machine ? undefined : entry.projectDir
    return pool?.derivedProjectDir(workspaceId)
  }
  /** Pool-relevant fingerprint per workspace: a changed endpoint or project means reconnect. */
  const signatures = new Map<string, string>()
  let registryRevision = -1
  let registryTimer: ReturnType<typeof setInterval> | undefined
  let registrySync: Promise<void> = Promise.resolve()
  const signature = (workspace: HubWorkspace): string => JSON.stringify(toPoolWorkspace(workspace))
  const remember = (workspace: HubWorkspace): void => {
    known.set(workspace.id, { id: workspace.id, label: workspace.label, projectDir: workspace.projectDir, machine: workspace.connection.discover === true })
    signatures.set(workspace.id, signature(workspace))
  }

  /** `hub.machines.list`: the owner sees every machine; anyone else sees machines holding a workspace they can read. */
  const machinesList = (context: AuthenticatedContext, grants: readonly HubGrant[]): Record<string, unknown> => {
    const owner = context.principal.kind === 'user' && context.principal.hubOwner
    const now = Date.now()
    const machines = (pool?.machines() ?? []).flatMap(machine => {
      // A session-only grant reveals that session, never the machine or its workspace inventory.
      const visible = machine.workspaceIds.filter(id => owner || canSeeWorkspace(context.principal, grants, id, now))
      if (!owner && visible.length === 0) return []
      return [{
        id: machine.id, label: machine.label, health: machine.health, read_only: machine.readOnly,
        models_supported: machine.modelsSupported, workspace_ids: visible,
      }]
    })
    return { machines }
  }

  /** Apply registry changes to a running pool: removed and changed workspaces leave, new and changed ones join. */
  async function syncRegistry(runtimePool: HubRuntimePool): Promise<void> {
    const snapshot = await registry.load()
    if (snapshot.revision === registryRevision || state !== 'running') return
    registryRevision = snapshot.revision
    const next = new Map(snapshot.workspaces.map(workspace => [workspace.id, workspace]))
    for (const id of [...known.keys()]) {
      const row = next.get(id)
      if (row !== undefined && signatures.get(id) === signature(row)) {
        known.set(id, { id, label: row.label, projectDir: row.projectDir, machine: row.connection.discover === true })
        continue
      }
      known.delete(id); signatures.delete(id)
      await runtimePool.removeWorkspace(id)
      append(id, { health: 'offline', removed: row === undefined })
    }
    const added = snapshot.workspaces.filter(workspace => !known.has(workspace.id))
    for (const workspace of added) remember(workspace)
    if (added.length > 0) await runtimePool.addWorkspaces(added.map(toPoolWorkspace))
  }

  const append = (workspaceId: string, payload: Record<string, unknown>): void => {
    try { store?.append({ workspace_id: workspaceId, type: 'hub.workspace_status', payload }) } catch (error) { report(error, 'workspace status event') }
  }
  const onHealth = (workspaceId: string, health: PoolHealth): void => {
    append(workspaceId, { health: health.state, degraded: health.degraded })
    if (health.state === 'online') lessonSync?.workspaceOnline()
  }

  async function teardown(): Promise<void> {
    const errors: unknown[] = []
    const attempt = async (step: () => unknown): Promise<void> => {
      try { await step() } catch (error) { errors.push(error) }
    }
    // Clients first (WebSocket close 1001), then the daemon relays, then the journal.
    clearInterval(registryTimer); registryTimer = undefined
    await attempt(() => registrySync)
    await attempt(() => server?.stop())
    await attempt(() => tunnel?.close())
    await attempt(() => lessonSync?.close())
    await attempt(() => pool?.stop())
    await attempt(() => settled)
    await attempt(() => supervisor?.close())
    await attempt(() => modelControl?.close())
    await attempt(() => subscriptions?.close())
    await attempt(() => router?.close())
    await attempt(() => store?.close())
    if (errors.length > 0) throw new AggregateError(errors, `Hub service shutdown failed: ${errors.map(error => (error instanceof Error ? error.message : String(error))).join('; ')}`)
  }

  const service: HubService = {
    get listening() { return state === 'running' },
    get workspacesSettled() { return settled },
    get server() { return server },
    get pool() { return pool },
    get store() { return store },
    get lessonSync() { return lessonSync },
    get supervisor() { return supervisor },

    workspaceHealth() {
      const machines = new Map((pool?.machines() ?? []).map(machine => [machine.id, machine]))
      return [
        // Per-project entries keep their original `{ id, health }` shape; only machine rows carry identity.
        ...[...known.values()].filter(row => !row.machine).map(row => ({ id: row.id, health: pool?.health(row.id)?.state ?? 'offline' })),
        ...(pool?.derivedWorkspaces() ?? []).map(row => {
          const machine = machines.get(row.machineId)
          return {
            id: row.id, health: pool?.health(row.id)?.state ?? 'offline', machine_id: row.machineId, project_dir: row.projectDir, label: row.label,
            ...(machine?.runtimeId === undefined ? {} : { runtime_id: machine.runtimeId }),
            ...(machine?.route === undefined ? {} : { route: machine.route }),
          }
        }),
      ]
    },

    status() {
      const identity = options.identity === undefined ? 'missing' : 'configured'
      return {
        state,
        identity,
        fail_closed: options.identity === undefined,
        listening: state === 'running',
        ...(options.identity === undefined ? { message: 'No identity subsystem is configured; the Hub refuses every request except GET /healthz.' } : {}),
        workspaces: listed().map(row => {
          const health = pool?.health(row.id)
          return { id: row.id, health: health?.state ?? 'offline', degraded: health?.degraded ?? false }
        }),
      }
    },

    async start() {
      if (state !== 'created') throw new Error('Hub service can only be started once')
      state = 'running'
      try {
        const snapshot = await registry.load()
        registryRevision = snapshot.revision
        for (const workspace of snapshot.workspaces) remember(workspace)
        store = new HubEventStore(options.paths.databasePath, options.store)
        const identity = options.identity
        subscriptions = new SubscriptionManager(store, {
          ...options.subscriptions,
          ...(identity === undefined ? {} : { grantsForPrincipal: (principalId: string) => identity.grantsFor(principalId) }),
        })
        const runtimePool = new HubRuntimePool({
          ...options.pool,
          sink: storeSink(store),
          onHealth,
          onGap: notice => { for (const id of notice.workspaceIds) append(id, { gap_reason: notice.reason }) },
          onInterrupted: notice => { for (const id of notice.workspaceIds) append(id, { interrupted: true }) },
        })
        pool = runtimePool
        lessonSync = new HubLessonSync({
          databasePath: options.paths.databasePath, signer: options.signer, backend: runtimePool, onError: report,
        })
        const backend = {
          call: (workspaceId: string, method: string, params: Readonly<Record<string, unknown>> = {}, callOptions = {}) => runtimePool.call(workspaceId, method, params, callOptions),
          projectDir: projectDirOf,
        }
        // The supervisor has no stored grants: while it executes one confirmed
        // action the router sees the confirming human's delegated scopes, and
        // nothing otherwise (fail closed).
        const overlay = new SupervisorGrantOverlay()
        if (identity !== undefined) {
          router = new HubCommandRouter({
            databasePath: options.paths.databasePath, backend, identity: overlay.wrapIdentity(identity), signer: options.signer,
            overlayGrants: () => overlay.current(),
            ...(options.server?.clock === undefined ? {} : { clock: options.server.clock }),
            onError: report,
          })
        }
        let supervisorHandlers: ReturnType<typeof createUnsupportedSupervisorHandlers>
        const supervisorConfig = options.supervisor
        let hubServerRef: HubServer | undefined
        let supervisorStartError: string | undefined
        if (identity !== undefined && router !== undefined) try {
          const commandRouter = router
          // XERXES exists whenever identity does. Its model is the stored choice (machine, profile, model,
          // effort); the XERXES_HUB_SUPERVISOR_* model is only a fallback while nothing is stored.
          modelControl = new SupervisorModelControl({
            databasePath: options.paths.databasePath,
            signer: options.signer,
            machines: runtimePool,
            audit: record => identity.audit(record),
            ...(supervisorConfig?.kind === 'configured' ? { fallback: supervisorConfig.model } : {}),
            broadcast: build => { hubServerRef?.sendEventToAll('supervisor.config_changed', build) },
            // Say it in the conversation, not only in settings: a different machine is answering.
            onFailover: change => {
              const running = supervisor
              if (running === undefined) return
              const text = change.to === undefined
                ? `${change.from} is back; XERXES answers there again.`
                : `${change.from} is offline; XERXES is answering on ${change.to} with the same model.`
              const { message } = running.threads.append({ threadId: 'shared', role: 'system', text })
              running.threads.emit('shared', 'supervisor.thread_event', { message: messageToWire(message) })
            },
            ...(options.server?.clock === undefined ? {} : { clock: options.server.clock }),
            onError: report,
          })
          const control = modelControl
          supervisor = createHubSupervisor({
            databasePath: options.paths.databasePath,
            identity,
            overlay,
            execute: (context, params) => commandRouter.execute(context, params),
            workspaces: {
              list: listed,
              health: id => runtimePool.health(id),
              projection: id => runtimePool.projection(id),
            },
            backend,
            events: {
              read: (after, filter, limit, maxBytes, through) => (store as HubEventStore).read(after, filter, limit, maxBytes, through),
              watermark: () => (store as HubEventStore).watermark(),
              onEvents: observer => (store as HubEventStore).onEvents(observer),
            },
            model: control.port,
            configStatus: () => control.describe(),
            // A default token bound per 24 h window applies unless the operator configured one.
            quotas: withDefaultSpendBound(supervisorConfig?.kind === 'configured' ? supervisorConfig.quotas : undefined),
            ...(supervisorConfig?.kind !== 'configured' || supervisorConfig.persona === undefined ? {} : { persona: supervisorConfig.persona }),
            ...(supervisorConfig?.kind !== 'configured' || supervisorConfig.modelSetSessionOnly === undefined ? {} : { modelSetSessionOnly: supervisorConfig.modelSetSessionOnly }),
            ...(options.sendPush === undefined ? {} : { sendPush: options.sendPush }),
            ...(options.isHubOwner === undefined ? {} : { isHubOwner: options.isHubOwner }),
            ...(options.autopilot === undefined ? {} : { autopilot: options.autopilot }),
            // Only people who can see the workspace learn about its Autopilot row.
            broadcastAutopilot: (workspaceId, payload) => {
              hubServerRef?.sendEventToAll('supervisor.autopilot_changed', recipient =>
                canSeeWorkspace(recipient.principal, identity.grantsFor(recipient.principal.id), workspaceId, Date.now()) ? payload : undefined)
            },
            // Only people who can read the watched task get its live card.
            broadcastWatch: (workspaceId, taskId, payload) => {
              hubServerRef?.sendEventToAll('supervisor.watch_update', recipient =>
                hasScope(recipient.principal, identity.grantsFor(recipient.principal.id), { workspaceId, sessionId: taskId }, 'session.read', Date.now())
                  ? payload : undefined)
            },
            ...(options.watch === undefined ? {} : { watch: options.watch }),
            ...(options.server?.clock === undefined ? {} : { clock: options.server.clock }),
            onError: report,
          })
          supervisorHandlers = { ...supervisor.handlers, ...control.handlers, 'hub.machines.list': ctx => machinesList(ctx.context, ctx.grants) }
        } catch (error) {
          // XERXES failing to start (a missing persona file, a bad stored row) must not take
          // the Hub down: tasks, approvals and usage keep working and the app says why.
          report(error, 'supervisor start')
          try { modelControl?.close() } catch { /* already reported */ }
          modelControl = undefined
          supervisor = undefined
          supervisorStartError = `XERXES could not start on this Hub: ${error instanceof Error ? error.message : String(error)}`
          supervisorHandlers = createUnsupportedSupervisorHandlers(supervisorStartError)
        } else {
          const reason = supervisorConfig?.kind === 'unconfigured' && supervisorConfig.reason !== undefined ? supervisorConfig.reason : SUPERVISOR_SETUP_HINT
          supervisorHandlers = createUnsupportedSupervisorHandlers(reason)
        }
        let voiceRoute: HubVoiceRoute | undefined
        if (identity !== undefined && supervisor !== undefined && options.voice !== undefined && allowedModes(options.voice).length > 0) {
          voiceRoute = createHubVoiceRoute({
            identity,
            supervisor: supervisor.voice,
            config: options.voice,
            ...(options.server?.clock === undefined ? {} : { clock: options.server.clock }),
            onError: report,
          })
        }
        const staticRoot = options.staticRoot ?? defaultStaticRoot()
        server = new HubServer({
          ...options.server,
          ...(voiceRoute === undefined ? {} : { voice: voiceRoute }),
          methodHandlers: { ...options.server?.methodHandlers, ...supervisorHandlers },
          hubId: options.signer.hubId,
          exposure,
          store,
          subscriptions,
          workspaces: {
            list: listed,
            health: id => runtimePool.health(id),
            projection: id => runtimePool.projection(id),
            readOnly: id => runtimePool.readOnly(id),
          },
          backend,
          ...(identity === undefined ? {} : { identity }),
          ...(router === undefined ? {} : { router }),
          ...(staticRoot === undefined ? {} : { staticRoot }),
          ready: () => state === 'running',
          onError: report,
        })
        const hubServer = server
        hubServerRef = hubServer
        supervisor?.attachSink({ send: (principalIds, type, payload) => { hubServer.sendEventToPrincipals(principalIds, type, payload) } })
        const listening = server.start()
        let publicOrigin: string | undefined
        if (options.tunnel !== undefined) {
          const target = `http://${listening.hostname.includes(':') ? `[${listening.hostname}]` : listening.hostname}:${listening.port}`
          tunnel = options.tunnel.kind === 'tailscale'
            ? await options.tunnel.port.serve({ target, httpsPort: options.tunnel.httpsPort ?? 443 })
            : await options.tunnel.port.expose({ target, hostname: options.tunnel.hostname })
          publicOrigin = tunnel.publicOrigin
          if (publicOrigin !== undefined && !listening.origins.includes(publicOrigin)) {
            throw new Error(`Tunnel origin ${publicOrigin} is not in the configured public origins; add it explicitly or stop the tunnel`)
          }
        }
        options.lifecycle?.registerShutdownHook(() => service.stop())
        options.lifecycle?.setHttpReady(true)
        lessonSync.start()
        // Connect in the background: one unreachable SSH host must not delay HTTP readiness.
        settled = runtimePool.start([...snapshot.workspaces.map(toPoolWorkspace)]).catch(error => report(error, 'workspace start'))
        const pollMs = options.registryPollMs ?? 3000
        if (pollMs > 0) {
          registryTimer = setInterval(() => {
            // Serialised: a slow reconnect never overlaps the next registry read.
            registrySync = registrySync.then(() => settled).then(() => syncRegistry(runtimePool)).catch(error => report(error, 'workspace registry reload'))
          }, pollMs)
          registryTimer.unref?.()
        }
        return { ...listening, ...(publicOrigin === undefined ? {} : { publicOrigin }) }
      } catch (error) {
        try { await teardown() } catch (cleanup) { report(cleanup, 'startup cleanup') }
        state = 'stopped'
        throw error
      }
    },

    stop() {
      if (stopping) return stopping
      options.lifecycle?.setHttpReady(false)
      state = 'stopped'
      stopping = teardown()
      return stopping
    },
  }
  return service
}
