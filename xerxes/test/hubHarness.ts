// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { generateKeyPairSync } from 'node:crypto'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { HubCommandRouter } from '../src/hub/commandRouter.js'
import { HubEventStore } from '../src/hub/eventStore.js'
import { resolveHubExposure } from '../src/hub/exposure.js'
import type { AuthenticatedContext, HubAuditInput, HubHttpRequestInfo, HubIdentityPort, RevocationListener } from '../src/hub/identityPort.js'
import { emptyProjection, replaceSessionSummaries, type WorkspaceProjection } from '../src/hub/projections.js'
import {
  HUB_PROTOCOL_VERSION, parseHubServerFrame,
  type HubEventFrame, type HubEventType, type HubMethod, type HubResponseFrame, type HubServerFrame,
} from '../src/hub/protocol.js'
import type { PoolCallResult, PoolHealth } from '../src/hub/runtimePool.js'
import { HubServer, type HubServerOptions } from '../src/hub/server.js'
import { createHubSigner, type HubSigner } from '../src/hub/signer.js'
import { SubscriptionManager } from '../src/hub/subscriptions.js'
import type { HubGrant, HubPrincipal } from '../src/security/principal.js'

const cleanups: Array<() => Promise<void> | void> = []
export function registerCleanup(fn: () => Promise<void> | void): void { cleanups.push(fn) }
export async function disposeAll(): Promise<void> {
  for (const fn of cleanups.splice(0).reverse()) await fn()
}

export async function until<T>(read: () => T | undefined | false, timeout = 3000): Promise<T> {
  const deadline = Date.now() + timeout
  for (;;) {
    const value = read()
    if (value !== undefined && value !== false) return value
    if (Date.now() > deadline) throw new Error('condition timed out')
    await Bun.sleep(5)
  }
}

export function testSigner(): HubSigner {
  const { privateKey } = generateKeyPairSync('ed25519')
  return createHubSigner(privateKey.export({ type: 'pkcs8', format: 'pem' }).toString())
}

export async function tempDir(prefix = 'xh-srv-'): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), prefix))
  registerCleanup(() => rm(dir, { recursive: true, force: true }))
  return dir
}

export const userPrincipal = (id: string, hubOwner = false): HubPrincipal => ({ kind: 'user', id, displayName: id.toUpperCase(), deviceId: `dev-${id}`, hubOwner })
export const agentPrincipal = (id: string): HubPrincipal => ({ kind: 'agent', id, displayName: id, onBehalfOf: 'alice' })
export const workspaceGrant = (principalId: string, workspaceId: string, role: 'viewer' | 'steer' | 'operator' | 'owner'): HubGrant =>
  ({ principalId, resource: { type: 'workspace', workspaceId }, effect: 'allow', role })
export const sessionGrant = (principalId: string, workspaceId: string, sessionId: string, role: 'viewer' | 'steer' | 'operator' | 'owner'): HubGrant =>
  ({ principalId, resource: { type: 'session', workspaceId, sessionId }, effect: 'allow', role })

/** Identity fake: the `who=<name>` cookie selects a registered user. */
export class FakeIdentity implements HubIdentityPort {
  readonly users = new Map<string, { principal: HubPrincipal; grants: HubGrant[] }>()
  readonly audits: HubAuditInput[] = []
  failAudit = false
  authenticateCalls = 0
  private readonly listeners = new Set<RevocationListener>()

  add(principal: HubPrincipal, grants: HubGrant[] = []): this {
    this.users.set(principal.id, { principal, grants })
    return this
  }
  context(name: string): AuthenticatedContext {
    const entry = this.users.get(name)
    if (!entry) throw new Error(`unknown test user ${name}`)
    const principal = entry.principal
    return {
      principal, grants: entry.grants, browserSessionId: `bs-${name}`,
      actor: {
        principal_kind: principal.kind, principal_id: principal.id, display_name: principal.displayName,
        ...(principal.kind === 'user' ? { user_id: principal.id, device_id: principal.deviceId } : {}),
      },
    }
  }
  async authenticate(request: HubHttpRequestInfo): Promise<AuthenticatedContext | null> {
    this.authenticateCalls += 1
    const name = /(?:^|;\s*)who=([^;]+)/.exec(request.cookie ?? '')?.[1]
    return name !== undefined && this.users.has(name) ? this.context(name) : null
  }
  grantsFor(principalId: string): readonly HubGrant[] {
    for (const entry of this.users.values()) if (entry.principal.id === principalId) return entry.grants
    return []
  }
  onRevoked(listener: RevocationListener): () => void {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }
  revoke(event: Parameters<RevocationListener>[0]): void { for (const listener of [...this.listeners]) listener(event) }
  async audit(record: HubAuditInput): Promise<void> {
    if (this.failAudit) throw new Error('audit log unavailable')
    this.audits.push(record)
  }
  async handleHttp(request: Request): Promise<Response | undefined> {
    return new URL(request.url).pathname === '/api/identity/ping' ? new Response('pong') : undefined
  }
}

/** owner (hub owner), alice (operator w1), viewer (viewer w1), guest (session-only w1/s1), robot (agent, operator w1). */
export function standardUsers(identity: FakeIdentity): FakeIdentity {
  return identity
    .add(userPrincipal('owner', true))
    .add(userPrincipal('alice'), [workspaceGrant('alice', 'w1', 'operator')])
    .add(userPrincipal('viewer'), [workspaceGrant('viewer', 'w1', 'viewer')])
    .add(userPrincipal('guest'), [sessionGrant('guest', 'w1', 's1', 'viewer')])
    .add(agentPrincipal('robot'), [workspaceGrant('robot', 'w1', 'operator')])
}

export interface BackendCall { readonly workspaceId: string; readonly method: string; readonly params: Record<string, unknown> }
export class FakeBackend {
  readonly calls: BackendCall[] = []
  sessions: Record<string, unknown>[] = [{ id: 's1', key: 'k1' }, { id: 's2', key: 'k2' }]
  handler: ((call: BackendCall) => PoolCallResult | undefined | Promise<PoolCallResult | undefined>) | undefined
  async call(workspaceId: string, method: string, params: Readonly<Record<string, unknown>> = {}): Promise<PoolCallResult> {
    const call: BackendCall = { workspaceId, method, params: { ...params } }
    this.calls.push(call)
    const custom = await this.handler?.(call)
    if (custom) return custom
    if (method === 'session.active_list') return { state: 'accepted', result: { ok: true, sessions: this.sessions } }
    return { state: 'accepted', result: { ok: true } }
  }
  projectDir(workspaceId: string): string | undefined { return workspaceId === 'nope' ? undefined : `/proj/${workspaceId}` }
  callsTo(method: string): BackendCall[] { return this.calls.filter(call => call.method === method) }
}

export class TestClient {
  readonly frames: HubServerFrame[] = []
  closeCode: number | undefined
  private sequence = 0
  private constructor(readonly ws: WebSocket) {}

  static async connect(port: number, who: string | undefined, options: { origin?: string | null } = {}): Promise<TestClient | undefined> {
    const origin = options.origin === undefined ? `http://127.0.0.1:${port}` : options.origin
    const headers: Record<string, string> = {}
    if (origin !== null) headers.origin = origin
    if (who !== undefined) headers.cookie = `who=${who}`
    // Bun's WebSocket accepts request headers; the DOM typings only know about protocols.
    const BunWebSocket = WebSocket as unknown as new (url: string, options: { headers: Record<string, string> }) => WebSocket
    const ws = new BunWebSocket(`ws://127.0.0.1:${port}/api/hub/v1/ws`, { headers })
    const client = new TestClient(ws)
    ws.addEventListener('message', event => { client.frames.push(parseHubServerFrame(String(event.data))) })
    ws.addEventListener('close', event => { client.closeCode = event.code })
    const opened = await new Promise<boolean>(resolve => {
      ws.addEventListener('open', () => resolve(true))
      ws.addEventListener('error', () => resolve(false))
      ws.addEventListener('close', () => resolve(false))
    })
    if (!opened) return undefined
    registerCleanup(() => client.close())
    return client
  }

  sendRaw(text: string): void { this.ws.send(text) }
  async request(method: HubMethod, params: Record<string, unknown> = {}): Promise<HubResponseFrame> {
    const id = `r${++this.sequence}`
    this.ws.send(JSON.stringify({ v: HUB_PROTOCOL_VERSION, kind: 'request', id, method, params }))
    return await this.waitFor(frame => frame.kind === 'response' && frame.id === id) as HubResponseFrame
  }
  /** Result of a request that must succeed. */
  async ok(method: HubMethod, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    const response = await this.request(method, params)
    if (!response.ok) throw new Error(`${method} failed: ${response.error.code} ${response.error.message}`)
    return { ...response.result }
  }
  /** Error code of a request that must fail. */
  async fails(method: HubMethod, params: Record<string, unknown> = {}): Promise<string> {
    const response = await this.request(method, params)
    if (response.ok) throw new Error(`${method} unexpectedly succeeded`)
    return response.error.code
  }
  async waitFor(predicate: (frame: HubServerFrame) => boolean, timeout = 3000): Promise<HubServerFrame> {
    return until(() => this.frames.find(predicate), timeout)
  }
  async event(type: HubEventType, predicate: (payload: Record<string, unknown>) => boolean = () => true, timeout = 3000): Promise<HubEventFrame> {
    return await this.waitFor(frame => frame.kind === 'event' && frame.type === type && predicate({ ...frame.payload }), timeout) as HubEventFrame
  }
  events(type: HubEventType): HubEventFrame[] {
    return this.frames.filter((frame): frame is HubEventFrame => frame.kind === 'event' && frame.type === type)
  }
  async closed(timeout = 3000): Promise<number> { return until(() => this.closeCode, timeout) }
  close(): void { try { this.ws.close() } catch { /* already closed */ } }
}

export interface FixtureOptions {
  readonly identity?: boolean
  readonly router?: boolean
  readonly staticRoot?: string
  readonly server?: Partial<HubServerOptions>
}
export interface Fixture {
  readonly dir: string
  readonly identity: FakeIdentity
  readonly backend: FakeBackend
  readonly store: HubEventStore
  readonly subscriptions: SubscriptionManager
  readonly server: HubServer
  readonly router: HubCommandRouter | undefined
  readonly signer: HubSigner
  readonly port: number
  readonly time: { now: number }
  readonly health: Map<string, PoolHealth>
  readonly projections: Map<string, WorkspaceProjection>
  url(path: string): string
  client(who: string, options?: { origin?: string | null }): Promise<TestClient>
}

export async function startFixture(options: FixtureOptions = {}): Promise<Fixture> {
  const dir = await tempDir()
  const identity = standardUsers(new FakeIdentity())
  const backend = new FakeBackend()
  const signer = testSigner()
  const time = { now: Date.now() }
  const databasePath = join(dir, 'hub.sqlite')
  const store = new HubEventStore(databasePath)
  const subscriptions = new SubscriptionManager(store, { grantsForPrincipal: id => identity.grantsFor(id) })
  const withIdentity = options.identity !== false
  const router = withIdentity && options.router !== false
    ? new HubCommandRouter({ databasePath, backend, identity, signer, clock: () => time.now })
    : undefined
  const labels = [{ id: 'w1', label: 'One' }, { id: 'w2', label: 'Two' }]
  const health = new Map<string, PoolHealth>([['w1', { state: 'online', degraded: false, runtimeId: 'rt1' }], ['w2', { state: 'online', degraded: false, runtimeId: 'rt1' }]])
  const projections = new Map<string, WorkspaceProjection>([
    ['w1', replaceSessionSummaries(emptyProjection('w1'), [{ id: 's1', title: 'First task', status: 'idle' }, { id: 's2', title: 'Second task', status: 'working' }], new Date().toISOString())],
    ['w2', emptyProjection('w2')],
  ])
  const server = new HubServer({
    hubId: signer.hubId,
    exposure: resolveHubExposure({ port: 0 }),
    store, subscriptions,
    workspaces: { list: () => labels, health: id => health.get(id), projection: id => projections.get(id) },
    backend,
    ...(withIdentity ? { identity } : {}),
    ...(router === undefined ? {} : { router }),
    ...(options.staticRoot === undefined ? {} : { staticRoot: options.staticRoot }),
    clock: () => time.now,
    ...options.server,
  })
  const started = server.start()
  registerCleanup(async () => {
    await server.stop()
    subscriptions.close()
    router?.close()
    store.close()
  })
  return {
    dir, identity, backend, store, subscriptions, server, router, signer, port: started.port, time, health, projections,
    url: path => `http://127.0.0.1:${started.port}${path}`,
    async client(who, clientOptions) {
      const client = await TestClient.connect(started.port, who, clientOptions)
      if (!client) throw new Error(`websocket for ${who} was refused`)
      return client
    },
  }
}
