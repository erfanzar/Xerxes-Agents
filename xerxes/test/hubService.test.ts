// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { chmod, stat } from 'node:fs/promises'
import { join } from 'node:path'

import { runHubCli } from '../src/hub/cli.js'
import { HubEventStore } from '../src/hub/eventStore.js'
import { getHubStatus, runHub, stopHub } from '../src/hub/lifecycle.js'
import { ensureHubDirectories, hubPaths, type HubPaths } from '../src/hub/paths.js'
import { createHubService, type HubService } from '../src/hub/service.js'
import { HUB_KEY_FILENAME, HubSignerError, createHubSigner, loadOrCreateHubSigner } from '../src/hub/signer.js'
import { WorkspaceRegistry } from '../src/hub/workspaceRegistry.js'
import { verifyHubAuth, type HubAuthClaims } from '../src/security/hubAuthEnvelope.js'
import { FakeIdentity, TestClient, disposeAll, registerCleanup, standardUsers, tempDir, testSigner, until } from './hubHarness.js'

afterEach(disposeAll)

type Frame = { id: number; method: string; params: Record<string, unknown> }

/** Minimal v35 daemon on a unix socket: enough handshake for the pool, and it records every request. */
async function fakeDaemon(dir: string, projectDir: string) {
  const socketPath = join(dir, 'd.sock')
  const requests: Frame[] = []
  const sockets = new Set<Bun.Socket<{ buffer: string }>>()
  const server = Bun.listen<{ buffer: string }>({
    unix: socketPath,
    socket: {
      open(socket) { sockets.add(socket); socket.data = { buffer: '' } },
      close(socket) { sockets.delete(socket) },
      data(socket, bytes) {
        socket.data.buffer += bytes.toString()
        for (;;) {
          const newline = socket.data.buffer.indexOf('\n')
          if (newline < 0) return
          const line = socket.data.buffer.slice(0, newline)
          socket.data.buffer = socket.data.buffer.slice(newline + 1)
          const frame = JSON.parse(line) as Frame
          requests.push(frame)
          let result: Record<string, unknown> = { ok: true }
          if (frame.method === 'initialize') result = { ok: true, daemon_protocol: 35, session: { id: 'metadata', key: frame.params.session_key }, connection_lease_supported: true }
          if (frame.method === 'connection.lease') result = { ok: true, token: 'lease' }
          if (frame.method === 'runtime.status') result = { ok: true, daemon_protocol: 35, runtime_id: 'rt', instance_id: 'i1', event_stream_supported: true, principal_auth_v1_supported: true }
          if (frame.method === 'session.active_list') result = { ok: true, sessions: [{ id: 's1', key: 'k1', cwd: projectDir, title: 'Task', status: 'idle', model: 'm' }] }
          if (frame.method === 'events.subscribe') result = { ok: true, subscription_id: 'sub', stream_id: 'stream', watermark: 'c0', oldest_cursor: 'c0', reset_required: false }
          if (frame.method === 'events.read') result = { ok: true, events: [], through_cursor: frame.params.after_cursor ?? 'c0' }
          socket.write(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }) + '\n')
        }
      },
    },
  })
  registerCleanup(() => { for (const socket of sockets) socket.terminate(); server.stop(true) })
  return {
    socketPath, requests,
    push(type: string, payload: Record<string, unknown>) {
      for (const socket of sockets) socket.write(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type, payload } }) + '\n')
    },
  }
}

async function serviceFixture(options: { identity?: boolean; lifecycle?: Parameters<typeof createHubService>[0]['lifecycle']; registryPollMs?: number } = {}) {
  const home = await tempDir('xh-svc-')
  const paths = hubPaths({ XERXES_HOME: home })
  await ensureHubDirectories(paths)
  const projectDir = join(home, 'project')
  const daemon = await fakeDaemon(home, projectDir)
  const registry = new WorkspaceRegistry(paths.workspacesPath, { generateId: () => 'w1' })
  await registry.add({ label: 'One', projectDir, endpoint: { kind: 'local', socketPath: daemon.socketPath }, connection: { autoConnect: true }, consent: { bootstrap: false, update: false } }, 0)
  const identity = standardUsers(new FakeIdentity())
  const signer = testSigner()
  const errors: string[] = []
  const service = createHubService({
    paths, signer, registry, exposure: { port: 0 },
    ...(options.identity === false ? {} : { identity }),
    ...(options.lifecycle === undefined ? {} : { lifecycle: options.lifecycle }),
    ...(options.registryPollMs === undefined ? {} : { registryPollMs: options.registryPollMs }),
    pool: { clientOptions: { retryBaseMs: 10, retryMaxMs: 25, random: () => 0.5, requestTimeoutMs: 2000 }, pollIntervalMs: 5000 },
    onError: (_error, context) => { errors.push(context) },
  })
  registerCleanup(() => service.stop())
  const info = await service.start()
  return { home, paths, projectDir, daemon, identity, signer, service, info, errors, url: (path: string) => `http://127.0.0.1:${info.port}${path}` }
}

describe('Hub service', () => {
  test('composes registry, pool, store and server; reports real workspace health and journals status changes', async () => {
    const t = await serviceFixture()
    expect(t.info.hostname).toBe('127.0.0.1')
    expect(t.service.status()).toMatchObject({ state: 'running', identity: 'configured', fail_closed: false, listening: true })
    await t.service.workspacesSettled
    await until(() => t.service.workspaceHealth()[0]?.health === 'online' || undefined)
    expect(t.service.workspaceHealth()).toEqual([{ id: 'w1', health: 'online' }])
    expect(t.service.status().workspaces).toEqual([{ id: 'w1', health: 'online', degraded: false }])
    expect(await (await fetch(t.url('/healthz'))).json()).toMatchObject({ status: 'ok', ready: true, workspaces: { total: 1, online: 1, degraded: 0 } })

    const events = t.service.store!.read(0, { workspaceIds: ['w1'] }).events
    expect(events.some(event => event.type === 'hub.workspace_status' && event.payload.health === 'online')).toBe(true)
    // The daemon only ever saw read-only summary traffic.
    expect(t.daemon.requests.every(request => request.method !== 'shutdown')).toBe(true)
    expect(t.errors).toEqual([])
  })

  test('applies registry changes live: an added workspace connects and a removed one leaves without a restart', async () => {
    const t = await serviceFixture({ registryPollMs: 30 })
    await t.service.workspacesSettled
    await until(() => t.service.workspaceHealth()[0]?.health === 'online' || undefined)
    const registry = new WorkspaceRegistry(t.paths.workspacesPath)
    let snapshot = await registry.load()
    snapshot = await registry.add({ label: 'Two', projectDir: join(t.home, 'two'), endpoint: { kind: 'local', socketPath: t.daemon.socketPath }, connection: { autoConnect: true }, consent: { bootstrap: false, update: false } }, snapshot.revision)
    const second = snapshot.workspaces.at(-1)!.id
    await until(() => t.service.workspaceHealth().find(row => row.id === second)?.health === 'online' || undefined)
    await registry.remove('w1', snapshot.revision)
    await until(() => (t.service.workspaceHealth().length === 1 && t.service.workspaceHealth()[0]!.id === second) || undefined)
    expect(t.service.pool?.health('w1')).toBeUndefined()
    expect(t.daemon.requests.some(request => request.method === 'shutdown')).toBe(false)
    expect(t.errors).toEqual([])
  })

  test('streams daemon events through the durable store to an authorized phone, and runs a signed command end to end', async () => {
    const t = await serviceFixture()
    await t.service.workspacesSettled
    await until(() => t.service.pool?.health('w1')?.state === 'online' || undefined)
    const alice = await TestClient.connect(t.info.port, 'alice')
    expect(alice).toBeDefined()
    const listed = await alice!.ok('hub.workspaces.list')
    expect(listed.workspaces).toMatchObject([{ id: 'w1', label: 'One', health: 'online' }])
    await alice!.ok('hub.events.subscribe', { workspace_ids: ['w1'] })
    t.daemon.push('event_batch', {
      subscription_id: 'sub',
      events: [{ event_id: 'e1', sequence: 1, cursor: 'c1', runtime_id: 'rt', instance_id: 'i1', project_dir: t.projectDir, session_id: 's1', type: 'turn_begin', payload: {}, occurred_at: new Date().toISOString() }],
      through_cursor: 'c1',
    })
    await alice!.event('hub.event_batch', payload => (payload.events as Array<Record<string, unknown>>).some(event => event.type === 'turn_begin'))

    const result = await alice!.ok('hub.command', { workspace_id: 'w1', session_id: 's1', command_id: 'c1', action: 'turn.submit', args: { text: 'hi' } })
    expect(result).toMatchObject({ ok: true, state: 'accepted' })
    const sent = t.daemon.requests.find(request => request.method === 'turn.submit' && request.params.hub_auth !== undefined)!
    expect(sent.params.session_key).toBe('k1')
    const { hub_auth: envelope, ...params } = sent.params
    const verified = verifyHubAuth(envelope, { publicKey: t.signer.publicKeyPem, method: 'turn.submit', params })
    expect(verified.ok).toBe(true)
    if (verified.ok) expect(verified.claims).toMatchObject({ principal_id: 'alice', workspace_id: 'w1', hub_id: t.signer.hubId })
    // Replaying the same command id returns the recorded outcome without a second daemon call.
    expect(await alice!.ok('hub.command', { workspace_id: 'w1', session_id: 's1', command_id: 'c1', action: 'turn.submit', args: { text: 'hi' } })).toEqual(result)
    expect(t.daemon.requests.filter(request => request.method === 'turn.submit')).toHaveLength(1)
  })

  test('without an identity port the service serves only /healthz and says so', async () => {
    const t = await serviceFixture({ identity: false })
    expect(t.service.status()).toMatchObject({ identity: 'missing', fail_closed: true, message: expect.stringContaining('/healthz') })
    expect(await (await fetch(t.url('/healthz'))).json()).toMatchObject({ status: 'identity_unavailable', ready: false, identity: 'missing' })
    for (const path of ['/', '/api/identity/ping', '/api/hub/v1/ws']) expect((await fetch(t.url(path))).status).toBe(503)
    expect(await TestClient.connect(t.info.port, 'alice')).toBeUndefined()
  })

  test('stop closes the server and sockets, releases the store, and never shuts a daemon down; it is idempotent', async () => {
    const t = await serviceFixture()
    await t.service.workspacesSettled
    const alice = await TestClient.connect(t.info.port, 'alice')
    await alice!.ok('hub.hello')
    const before = t.daemon.requests.length
    await t.service.stop()
    await t.service.stop()
    expect(t.service.status().state).toBe('stopped')
    expect(t.service.listening).toBe(false)
    await expect(fetch(t.url('/healthz'))).rejects.toThrow()
    expect(await alice!.closed()).toBeNumber()
    const methods = t.daemon.requests.slice(before).map(request => request.method)
    expect(methods).not.toContain('shutdown')
    expect(t.daemon.requests.some(request => request.method === 'shutdown' || request.method === 'turn.cancel')).toBe(false)
    expect(() => new HubEventStore(t.paths.databasePath).close()).not.toThrow()
  })

  test('registers with runHub: reports HTTP readiness and live workspace health, and stops with the Hub', async () => {
    const home = await tempDir('xh-svc-run-')
    const paths = hubPaths({ XERXES_HOME: home })
    let service: HubService | undefined
    const handle = await runHub({ paths, installSignalHandlers: false, workspaceHealth: () => service?.workspaceHealth() ?? [] })
    registerCleanup(() => handle.stop())
    const projectDir = join(home, 'project')
    const daemon = await fakeDaemon(home, projectDir)
    const registry = new WorkspaceRegistry(paths.workspacesPath, { generateId: () => 'w1' })
    await registry.add({ label: 'One', projectDir, endpoint: { kind: 'local', socketPath: daemon.socketPath }, connection: { autoConnect: true }, consent: { bootstrap: false, update: false } }, 0)
    expect(handle.status().listeners.http).toBe(false)
    service = createHubService({
      paths, signer: testSigner(), registry, identity: standardUsers(new FakeIdentity()), exposure: { port: 0 }, lifecycle: handle,
      pool: { clientOptions: { retryBaseMs: 10, retryMaxMs: 25, random: () => 0.5 } },
    })
    registerCleanup(() => service!.stop())
    await service.start()
    expect(handle.status().listeners.http).toBe(true)
    await service.workspacesSettled
    await until(() => handle.status().workspaces.some(row => row.id === 'w1' && row.health === 'online') || undefined)
    await handle.stop()
    await handle.closed
    expect(service.status().state).toBe('stopped')
    expect(daemon.requests.some(request => request.method === 'shutdown')).toBe(false)
  })

  test('a start failure cleans up and leaves nothing listening', async () => {
    const home = await tempDir('xh-svc-bad-')
    const paths: HubPaths = hubPaths({ XERXES_HOME: join(home, 'missing', 'deeper') })
    // The state directory does not exist, so opening the event store fails.
    const service = createHubService({ paths, signer: testSigner(), exposure: { port: 0 }, onError: () => undefined })
    await expect(service.start()).rejects.toThrow()
    expect(service.status().state).toBe('stopped')
    await expect(service.start()).rejects.toThrow(/only be started once/)
  })
})

describe('hub run wiring', () => {
  test('xerxes hub run starts the service with identity configured and asks for an owner on a fresh home', async () => {
    const home = await tempDir('xh-cli-')
    const paths = hubPaths({ XERXES_HOME: home })
    const lines: string[] = []
    const errors: string[] = []
    const environment = { XERXES_HOME: home, XERXES_HUB_PORT: '0' }
    const running = runHubCli(['run'], { paths, lifecycle: { paths, environment }, stdout: line => lines.push(line), stderr: line => errors.push(line) })
    registerCleanup(async () => { await stopHub({ paths }).catch(() => undefined); await running.catch(() => undefined) })
    const listening = await until(() => lines.map(line => JSON.parse(line) as Record<string, unknown>).find(row => row.event === 'hub.listening'), 10_000)
    expect(typeof listening.port).toBe('number')
    expect(listening.hostname).toBe('127.0.0.1')
    expect(listening.identity).toBe('configured')
    expect(listening.fail_closed).toBe(false)
    const health = await (await fetch(`http://127.0.0.1:${String(listening.port)}/healthz`)).json() as Record<string, unknown>
    expect(health.live).toBe(true)
    expect((await getHubStatus({ paths }))?.listeners.http).toBe(true)
    expect((await stat(join(paths.stateDir, HUB_KEY_FILENAME))).mode & 0o777).toBe(0o600)
    expect(await stopHub({ paths })).toEqual({ stopped: true })
    expect(await running).toBe(0)
    expect(errors).toEqual([expect.stringContaining('xerxes hub owner init')])
  })
})

describe('Hub signer', () => {
  const claims = (hubId: string): HubAuthClaims => ({
    hub_id: hubId, principal_id: 'alice', principal_kind: 'user', display_name: 'Alice', workspace_id: 'w1', method: 'turn.submit',
    params_hash: 'h', request_id: 'r', nonce: 'n', issued_at: Date.now(), expires_at: Date.now() + 60_000, scopes: ['session.submit'],
  })

  test('creates the key privately once, reloads the same identity, and refuses a group-readable key', async () => {
    const home = await tempDir('xh-key-')
    const paths = hubPaths({ XERXES_HOME: home })
    const first = await loadOrCreateHubSigner({ paths })
    const keyPath = join(paths.stateDir, HUB_KEY_FILENAME)
    expect((await stat(keyPath)).mode & 0o777).toBe(0o600)
    expect(first.publicKeyPem).toContain('PUBLIC KEY')
    expect(JSON.stringify(first)).not.toContain('PRIVATE')
    const second = await loadOrCreateHubSigner({ paths })
    expect(second.hubId).toBe(first.hubId)
    expect(second.fingerprint).toBe(first.fingerprint)
    expect(first.fingerprint).toMatch(/^[0-9a-f]{64}$/)

    const envelope = first.sign(claims(first.hubId))
    expect(verifyHubAuth(envelope, { publicKey: first.publicKeyPem, method: 'turn.submit', params: {} })).toMatchObject({ ok: false, reason: 'params_mismatch' })
    expect(() => first.sign(claims('hub-someone-else'))).toThrow(HubSignerError)

    await chmod(keyPath, 0o644)
    await expect(loadOrCreateHubSigner({ paths })).rejects.toThrow(HubSignerError)
  })

  test('rejects a non-Ed25519 or malformed key', async () => {
    expect(() => createHubSigner('not a key')).toThrow(HubSignerError)
    const { generateKeyPairSync } = await import('node:crypto')
    const rsa = generateKeyPairSync('rsa', { modulusLength: 2048 }).privateKey.export({ type: 'pkcs8', format: 'pem' }).toString()
    expect(() => createHubSigner(rsa)).toThrow(/Ed25519/)
  })
})
