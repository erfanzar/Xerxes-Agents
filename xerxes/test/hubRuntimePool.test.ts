// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { HubDaemonClient, DaemonTransportError } from '../src/hub/daemonClient.js'
import { HubRuntimePool } from '../src/hub/runtimePool.js'
import { HubOperatorError } from '../src/hub/ports.js'
import type { HubEventSink, UpstreamEvent } from '../src/hub/ports.js'
import type { PoolWorkspace } from '../src/hub/runtimePool.js'
import { DaemonServer } from '../src/daemon/server.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'

const cleanup: Array<() => Promise<void> | void> = []
afterEach(async () => { for (const close of cleanup.splice(0).reverse()) await close() })
async function until(condition: () => boolean, timeout = 2000): Promise<void> {
  const deadline = Date.now() + timeout
  while (!condition()) { if (Date.now() > deadline) throw new Error('condition timed out'); await Bun.sleep(5) }
}
type Frame = { id: number; method: string; params: Record<string, unknown> }
async function fakeDaemon(runtimeId = 'r', socketOverride?: string) {
  const dir = await mkdtemp(join(tmpdir(), 'xh-'))
  const socketPath = socketOverride ?? join(dir, 'd.sock')
  let instance = 'i1', streaming = true, unsupported = false, dropMutation = false, connections = 0, signed = true
  const requests: Frame[] = []
  let response: ((frame: Frame) => Record<string, unknown> | undefined) | undefined
  const silent = new Set<string>()
  const sockets = new Set<Bun.Socket<{ buffer: string }>>()
  const server = Bun.listen<{ buffer: string }>({ unix: socketPath, socket: {
    open(socket) { connections++; sockets.add(socket); socket.data = { buffer: '' } },
    close(socket) { sockets.delete(socket) },
    data(socket, bytes) {
      socket.data.buffer += bytes.toString()
      for (;;) {
        const nl = socket.data.buffer.indexOf('\n'); if (nl < 0) return
        const line = socket.data.buffer.slice(0, nl); socket.data.buffer = socket.data.buffer.slice(nl + 1)
        const frame = JSON.parse(line) as Frame; requests.push(frame)
        let result: Record<string, unknown> = { ok: true }
        if (frame.method === 'initialize') result = { ok: true, daemon_protocol: 35, session: { id: 'metadata', key: frame.params.session_key }, connection_lease_supported: true }
        if (frame.method === 'connection.lease') result = { ok: true, token: 'lease' }
        if (frame.method === 'runtime.status') result = { ok: true, daemon_protocol: 35, runtime_id: runtimeId, instance_id: instance, event_stream_supported: streaming, ...(signed ? { principal_auth_v1_supported: true } : {}) }
        if (frame.method === 'session.active_list') result = { ok: true, sessions: ['/a', '/b'].map((cwd, index) => ({ id: `s${index}`, key: `k${index}`, cwd, title: cwd, status: 'idle', model: 'm' })) }
        if (frame.method === 'events.subscribe') result = unsupported ? { ok: false, error: 'Unknown method events.subscribe' } : { ok: true, subscription_id: 'sub', stream_id: 'stream', watermark: 'c0', oldest_cursor: 'c0', reset_required: false }
        if (frame.method === 'events.read') result = { ok: true, events: [], through_cursor: frame.params.after_cursor ?? 'c0' }
        if (frame.method === 'turn.submit' && dropMutation) { socket.terminate(); continue }
        if (frame.method === 'hang' || silent.has(frame.method)) continue
        result = response?.(frame) ?? result
        socket.write(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }) + '\n')
      }
    },
  } })
  cleanup.push(async () => { for (const socket of sockets) socket.terminate(); server.stop(true); await rm(dir, { recursive: true, force: true }) })
  return { socketPath, requests, get connections() { return connections }, get live() { return sockets.size },
    setInstance(value: string) { instance = value }, setStreaming(value: boolean) { streaming = value }, setSigned(value: boolean) { signed = value }, setUnsupported() { unsupported = true }, dropMutations() { dropMutation = true },
    drop() { for (const socket of sockets) socket.terminate() },
    /** The socket disappears for good (a re-linked tunnel moved it). */
    close() { for (const socket of sockets) socket.terminate(); server.stop(true) },
    push(type: string, payload: Record<string, unknown>) { for (const socket of sockets) socket.write(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type, payload } }) + '\n') },
    raw(frame: unknown) { for (const socket of sockets) socket.write(JSON.stringify(frame) + '\n') },
    text(value: string) { for (const socket of sockets) socket.write(value) },
    respond(handler: (frame: Frame) => Record<string, unknown> | undefined) { response = handler },
    silence(method: string) { silent.add(method) },
  }
}
function workspace(id: string, socketPath: string, projectDir = '/a'): PoolWorkspace { return { id, label: id, endpoint: { kind: 'local', socketPath }, projectDir, autoConnect: true } }
function sink() {
  const rows: UpstreamEvent[] = []; const cursors = new Map<string, string>()
  const port: HubEventSink = { ingest(runtime, _workspace, events) { rows.push(...events); const last = events.at(-1); if (last) cursors.set(runtime, last.cursor) }, upstreamCursor(runtime) { return cursors.get(runtime) } }
  return { port, rows }
}
const retry = { retryBaseMs: 10, retryMaxMs: 25, random: () => 0.5, requestTimeoutMs: 300 }

test('N runtimes/M workspaces share one verified runtime lane, isolate outages and force summary-only reads', async () => {
  const first = await fakeDaemon('r1'), alias = await fakeDaemon('r1'), second = await fakeDaemon('r2')
  const store = sink()
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, pollIntervalMs: 30 })
  cleanup.push(() => pool.stop())
  await pool.start([workspace('a', first.socketPath), workspace('b', first.socketPath, '/b'), workspace('alias', alias.socketPath), workspace('c', second.socketPath)])
  expect(pool.runtimeCount).toBe(2)
  expect(first.live + alias.live).toBe(1)
  expect(pool.health('a')?.state).toBe('online')
  expect(pool.projection('b')?.sessions.s1?.title).toBe('/b')
  expect(pool.projection('b')?.sessions.s0).toBeUndefined()
  for (const daemon of [first, alias, second]) for (const request of daemon.requests.filter(row => ['initialize', 'session.active_list', 'session.status', 'session.open'].includes(row.method))) expect(request.params.history_limit).toBe(0)
  first.drop()
  expect(pool.health('c')?.state).toBe('online')
  await pool.stop()
  expect([...first.requests, ...second.requests].some(row => row.method === 'shutdown')).toBe(false)
})

test('SSH loss marks stale, rebuilds tunnel, restores lease and resumes committed cursor; restarts mark interrupted', async () => {
  const daemon = await fakeDaemon(), store = sink(), states: string[] = [], interrupted: string[] = []
  let tunnels = 0
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, pollIntervalMs: 1000,
    openTunnel: async () => { tunnels++; return { socketPath: daemon.socketPath, close() {} } },
    onHealth: (_id, status) => states.push(status.state), onInterrupted: info => interrupted.push(info.instanceId) })
  cleanup.push(() => pool.stop())
  await pool.start([{ id: 'w', label: 'Remote', projectDir: '/a', autoConnect: true, endpoint: { kind: 'ssh', sshAlias: 'server', remoteSocketPath: '/tmp/remote.sock' } }])
  daemon.push('event_batch', { subscription_id: 'sub', events: [{ event_id: 'e1', sequence: 1, cursor: 'c1', runtime_id: 'r', instance_id: 'i1', project_dir: '/a', session_id: 's0', type: 'turn_begin', payload: {}, occurred_at: new Date().toISOString() }], through_cursor: 'c1' })
  await until(() => store.rows.length === 1)
  daemon.setInstance('i2'); daemon.drop()
  await until(() => interrupted.length === 1 && pool.health('w')?.state === 'online')
  expect(states).toContain('stale')
  expect(tunnels).toBeGreaterThan(1)
  expect(daemon.requests.filter(row => row.method === 'events.subscribe').at(-1)?.params.after_cursor).toBe('c1')
  expect(daemon.requests.filter(row => row.method === 'connection.lease').some(row => row.params.token === 'lease')).toBe(true)
  expect(pool.projection('w')?.sessions.s0?.interrupted).toBe(true)
})

test('unsupported subscriptions and old runtimes degrade to bounded summary polling', async () => {
  for (const advertised of [true, false]) {
    const daemon = await fakeDaemon(`r${advertised}`), store = sink()
    daemon.setStreaming(advertised); daemon.setUnsupported()
    const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, pollIntervalMs: 15 })
    cleanup.push(() => pool.stop())
    await pool.start([workspace('w', daemon.socketPath)])
    await until(() => daemon.requests.filter(row => row.method === 'session.active_list').length >= 2)
    expect(pool.health('w')).toMatchObject({ state: 'online', degraded: true })
    for (const request of daemon.requests.filter(row => row.method === 'session.active_list')) expect(request.params.history_limit).toBe(0)
    await pool.stop()
  }
})

test('dropped mutation outcome is unknown and never automatically replayed; allowlist rejects generic RPC', async () => {
  const daemon = await fakeDaemon(), store = sink()
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry })
  cleanup.push(() => pool.stop())
  await pool.start([workspace('w', daemon.socketPath)])
  daemon.dropMutations()
  expect(await pool.call('w', 'turn.submit', { session_key: 'k0', text: 'hello' })).toMatchObject({ state: 'unknown' })
  await until(() => pool.health('w')?.state === 'online')
  await Bun.sleep(60) // several reconnect budgets: an uncertain mutation must remain absent
  expect(daemon.requests.filter(row => row.method === 'turn.submit')).toHaveLength(1)
  expect(await pool.call('w', 'shutdown', {})).toMatchObject({ state: 'rejected' })
  expect(await pool.call('w', 'slash', { command: '!bad' })).toMatchObject({ state: 'rejected' })
  expect(await pool.call('w', 'turn.cancel', {})).toMatchObject({ state: 'rejected' })
  expect(daemon.requests.some(row => row.method === 'shutdown' || row.method === 'slash')).toBe(false)
})

test('a daemon that cannot verify hub_auth is read only: reads work, mutations are refused before anything is written', async () => {
  const daemon = await fakeDaemon(), store = sink()
  daemon.setSigned(false)
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry })
  cleanup.push(() => pool.stop())
  await pool.start([workspace('w', daemon.socketPath)])
  expect(pool.health('w')?.state).toBe('online')
  expect(pool.readOnly('w')).toBe(true)
  expect((await pool.call('w', 'session.open', { session_key: 'k0' })).state).toBe('accepted')
  expect((await pool.call('w', 'session.status', { session_key: 'k0' })).state).toBe('accepted')
  for (const method of ['turn.submit', 'turn.steer', 'turn.cancel', 'permission_response', 'session.goal', 'set_model']) {
    expect(await pool.call('w', method, { session_key: 'k0', text: 'x' })).toMatchObject({ state: 'rejected', error: { code: 'read_only' } })
  }
  expect(daemon.requests.some(row => ['turn.submit', 'turn.steer', 'turn.cancel', 'permission_response', 'session.goal', 'set_model'].includes(row.method))).toBe(false)
  expect(pool.readOnly('unknown')).toBe(true)
})

test('workspaces join and leave a running pool; a shared lane closes only with its last workspace', async () => {
  const daemon = await fakeDaemon(), store = sink()
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry })
  cleanup.push(() => pool.stop())
  await pool.start([workspace('a', daemon.socketPath)])
  await pool.addWorkspaces([workspace('b', daemon.socketPath, '/b')])
  expect(pool.health('b')?.state).toBe('online')
  expect(pool.projection('b')?.sessions.s1?.title).toBe('/b')
  expect(pool.runtimeCount).toBe(1)
  await expect(pool.addWorkspaces([workspace('b', daemon.socketPath)])).rejects.toThrow('Duplicate')
  await pool.removeWorkspace('a')
  expect(pool.workspaceIds()).toEqual(['b'])
  expect(pool.health('a')).toBeUndefined()
  expect(pool.health('b')?.state).toBe('online')
  await pool.removeWorkspace('b')
  expect(pool.runtimeCount).toBe(0)
  await until(() => daemon.live === 0)
  expect(daemon.requests.some(row => row.method === 'shutdown')).toBe(false)
})

test('a second endpoint to the same runtime takes over when the first one died; the live connection is kept', async () => {
  // A re-linked tunnel: the old entry's socket is gone, a new entry reaches the same daemon.
  const old = await fakeDaemon('mac'), relinked = await fakeDaemon('mac'), store = sink()
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry })
  cleanup.push(() => pool.stop())
  await pool.start([workspace('old', old.socketPath)])
  expect(pool.health('old')?.state).toBe('online')
  old.close()
  await until(() => pool.health('old')?.state !== 'online')
  await pool.addWorkspaces([workspace('new', relinked.socketPath, '/b')])
  await until(() => pool.health('old')?.state === 'online' && pool.health('new')?.state === 'online')
  expect(pool.runtimeCount).toBe(1)
  expect(relinked.live).toBeGreaterThan(0)
  expect((await pool.call('old', 'session.status', { session_key: 'k0' })).state).toBe('accepted')
})

test('a machine with routes comes online on a fallback route and moves back to its own endpoint when that returns', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xh-routes-'))
  cleanup.push(() => rm(dir, { recursive: true, force: true }))
  const primaryPath = join(dir, 'primary.sock')
  const relay = await fakeDaemon('mac'), store = sink()
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, pollIntervalMs: 20, betterRouteProbeMs: 20 })
  cleanup.push(() => pool.stop())
  const machine: PoolWorkspace = { ...workspace('m', primaryPath), routes: [{ via: 'mac', endpoint: { kind: 'local', socketPath: relay.socketPath } }] }
  await pool.start([machine])
  await until(() => pool.health('m')?.state === 'online')
  expect(pool.machines()[0]).toMatchObject({ id: 'm', route: 'mac', runtimeId: 'mac' })
  expect((await pool.call('m', 'session.status', { session_key: 'k0' })).state).toBe('accepted')
  // The machine's own endpoint answers again: the pool switches back without being told.
  const direct = await fakeDaemon('mac', primaryPath)
  await until(() => pool.machines()[0]?.route === 'primary' && pool.health('m')?.state === 'online', 4000)
  expect(direct.live).toBeGreaterThan(0)
  expect((await pool.call('m', 'session.status', { session_key: 'k1' })).state).toBe('accepted')
  expect(pool.runtimeCount).toBe(1)
})

test('a fallback route that reaches a different runtime is never used', async () => {
  const own = await fakeDaemon('tpu'), stranger = await fakeDaemon('someone-else'), store = sink()
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, pollIntervalMs: 20, betterRouteProbeMs: 20 })
  cleanup.push(() => pool.stop())
  const machine: PoolWorkspace = { ...workspace('m', own.socketPath), routes: [{ via: 'hub', endpoint: { kind: 'local', socketPath: stranger.socketPath } }] }
  await pool.start([machine])
  await until(() => pool.health('m')?.state === 'online')
  own.close()
  await until(() => pool.health('m')?.state !== 'online')
  // Give the lane time to try the stranger's socket and refuse it.
  await until(() => stranger.requests.some(frame => frame.method === 'runtime.status'), 3000)
  await Bun.sleep(300)
  expect(pool.health('m')?.state).not.toBe('online')
  expect(pool.machines()[0]?.runtimeId).toBe('tpu')
  expect((await pool.call('m', 'session.status', { session_key: 'k0' })).state).toBe('rejected')
  // Nothing that reached the stranger ever subscribed to its journal or wrote to it.
  expect(stranger.requests.some(frame => frame.method === 'events.subscribe' || frame.method === 'session.active_list')).toBe(false)
})

test('a daemon that advertises principal_auth_v1 accepts mutations', async () => {
  const daemon = await fakeDaemon(), store = sink()
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry })
  cleanup.push(() => pool.stop())
  await pool.start([workspace('w', daemon.socketPath)])
  expect(pool.readOnly('w')).toBe(false)
  expect((await pool.call('w', 'turn.submit', { session_key: 'k0', text: 'hi' })).state).toBe('accepted')
})

test('event gap resyncs summaries and emits a Hub gap notice; private provider events never reach the sink', async () => {
  const daemon = await fakeDaemon(), store = sink(), gaps: string[] = []
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, onGap: notice => gaps.push(notice.reason) })
  cleanup.push(() => pool.stop())
  await pool.start([workspace('w', daemon.socketPath)])
  const before = daemon.requests.filter(row => row.method === 'session.active_list').length
  daemon.raw({ jsonrpc: '2.0', method: 'provider.remote.request', params: { secret: 'do-not-forward' } })
  daemon.push('event_batch', { subscription_id: 'sub', events: [{ event_id: 'private', sequence: 1, cursor: 'c1', runtime_id: 'r', instance_id: 'i1', project_dir: '/a', type: 'provider.remote.request', payload: { secret: true }, occurred_at: new Date().toISOString() }] })
  daemon.push('event_gap', { subscription_id: 'sub', reason: 'retention', watermark: 'c5', oldest_cursor: 'c2' })
  await until(() => gaps.length > 0)
  expect(gaps).toEqual(['retention'])
  expect(store.rows).toHaveLength(0)
  expect(daemon.requests.filter(row => row.method === 'session.active_list').length).toBeGreaterThan(before)
})

test('Bun daemon client has request deadlines, pre-write cancellation and ignores private request frames', async () => {
  const daemon = await fakeDaemon(), events: string[] = []
  const client = new HubDaemonClient({ socketPath: daemon.socketPath, projectDir: '/a', ...retry })
  cleanup.push(() => client.close())
  client.onEvent(type => events.push(type)); await client.connect()
  await expect(client.call('hang', {}, { timeoutMs: 15 })).rejects.toBeInstanceOf(DaemonTransportError)
  await expect(client.call('hang', {}, { signal: AbortSignal.abort() })).rejects.toBeInstanceOf(DaemonTransportError)
  expect(daemon.requests.filter(row => row.method === 'hang')).toHaveLength(1)
  daemon.raw({ jsonrpc: '2.0', method: 'provider.remote.request', id: 9999, params: { secret: 'private' } })
  daemon.push('provider.remote.request', { secret: 'private' })
  daemon.push('public', { ok: true })
  await until(() => events.includes('public'))
  expect(events).not.toContain('provider.remote.request')
})

test('client works with the real in-memory DaemonServer without providers or UI imports', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xh-real-')), socketPath = join(directory, 'd.sock')
  const server = new DaemonServer({ socketPath, projectDirectory: directory, runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') }) })
  await server.start()
  cleanup.push(async () => { await server.stop(); await rm(directory, { recursive: true, force: true }) })
  const client = new HubDaemonClient({ socketPath, projectDir: directory, ...retry, requestTimeoutMs: 5000 })
  cleanup.push(() => client.close())
  const handshake = await client.connect()
  expect(handshake.status.daemon_protocol).toBe(35)
  expect((await client.call('session.active_list', { history_limit: 0 })).ok).toBe(true)
})

function upstream(sequence: number, project_dir = '/a', instance_id = 'i1'): UpstreamEvent {
  return { event_id: `e${sequence}`, sequence, cursor: `c${sequence}`, runtime_id: 'r', instance_id, project_dir, session_id: 's0', type: 'turn_begin', payload: {}, occurred_at: new Date().toISOString() }
}
function delta(sequence: number, type: 'text_part' | 'think_part' | 'tool_call', value: string, session = 's0'): UpstreamEvent {
  const payload = type === 'text_part' ? { text: value } : type === 'think_part' ? { think: value } : { name: value, arguments: '{}' }
  return { ...upstream(sequence), session_id: session, type, payload }
}

test('the Hub subscribes with conversation detail and merges text/thinking deltas into one stored event per run', async () => {
  const daemon = await fakeDaemon(), store = sink()
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, pollIntervalMs: 5000, coalesceMs: 200 })
  cleanup.push(() => pool.stop()); await pool.start([workspace('w', daemon.socketPath)])
  expect(daemon.requests.find(row => row.method === 'events.subscribe')?.params.detail).toBe('conversation')
  daemon.push('event_batch', { subscription_id: 'sub', events: [
    delta(1, 'think_part', 'hmm '), delta(2, 'think_part', 'ok'),
    delta(3, 'text_part', 'Hel'), delta(4, 'text_part', 'lo'), delta(5, 'text_part', ' there', 's1'),
    delta(6, 'tool_call', 'exec_command'),
  ] })
  await until(() => store.rows.length === 4)
  // Upstream order is kept; each merged event commits the cursor of the last delta it holds.
  expect(store.rows.map(row => [row.type, row.sequence, row.cursor, row.session_id])).toEqual([
    ['think_part', 2, 'c2', 's0'], ['text_part', 4, 'c4', 's0'], ['text_part', 5, 'c5', 's1'], ['tool_call', 6, 'c6', 's0'],
  ])
  expect(store.rows.map(row => row.payload.think ?? row.payload.text ?? row.payload.name)).toEqual(['hmm ok', 'Hello', ' there', 'exec_command'])

  // A trailing run with nothing after it is stored once the coalescing window passes.
  daemon.push('event_batch', { subscription_id: 'sub', events: [delta(7, 'text_part', 'a'), delta(8, 'text_part', 'b')] })
  daemon.push('event_batch', { subscription_id: 'sub', events: [delta(9, 'text_part', 'c')] })
  await until(() => store.rows.length === 5)
  await Bun.sleep(300)
  expect(store.rows).toHaveLength(5)
  expect(store.rows[4]).toMatchObject({ type: 'text_part', sequence: 9, cursor: 'c9', payload: { text: 'abc' } })
})

test('thinking_part, text_delta and thinking_delta are normalized to text_part / think_part and merged', async () => {
  const daemon = await fakeDaemon(), store = sink()
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, pollIntervalMs: 5000, coalesceMs: 200 })
  cleanup.push(() => pool.stop()); await pool.start([workspace('w', daemon.socketPath)])
  const raw = (sequence: number, type: string, payload: Record<string, unknown>): UpstreamEvent => ({ ...upstream(sequence), type, payload })
  daemon.push('event_batch', { subscription_id: 'sub', events: [
    raw(1, 'thinking_part', { think: 'a' }), raw(2, 'thinking_delta', { thinking: 'b' }), raw(3, 'think_part', { think: 'c' }),
    raw(4, 'text_delta', { delta: 'Hel' }), raw(5, 'text_part', { text: 'lo' }),
    delta(6, 'tool_call', 'exec_command'),
  ] })
  await until(() => store.rows.length === 3)
  expect(store.rows.map(row => [row.type, row.sequence, row.payload])).toEqual([
    ['think_part', 3, { think: 'abc' }], ['text_part', 5, { text: 'Hello' }], ['tool_call', 6, { name: 'exec_command', arguments: '{}' }],
  ])
})

test('streamed text deltas do not re-project the workspace on every merge; other events still do', async () => {
  const daemon = await fakeDaemon(), store = sink(), projected: string[] = []
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, pollIntervalMs: 60_000, coalesceMs: 5,
    onProjection: (_id, projection) => { projected.push(projection.recent_event_ids.at(-1) ?? '') } })
  cleanup.push(() => pool.stop()); await pool.start([workspace('w', daemon.socketPath)])
  daemon.push('event_batch', { subscription_id: 'sub', events: [upstream(1)] })
  await until(() => store.rows.length === 1)
  projected.length = 0
  // Five separately stored text runs (each flushed by the coalescing timer).
  for (let sequence = 2; sequence <= 6; sequence += 1) {
    daemon.push('event_batch', { subscription_id: 'sub', events: [delta(sequence, 'text_part', `t${sequence}`)] })
    await until(() => store.rows.length === sequence)
  }
  expect(projected.filter(id => ['e2', 'e3', 'e4', 'e5', 'e6'].includes(id))).toEqual(['e2'])
  daemon.push('event_batch', { subscription_id: 'sub', events: [delta(7, 'tool_call', 'exec_command')] })
  await until(() => store.rows.length === 7)
  expect(projected.at(-1)).toBe('e7')
  expect(pool.projection('w')?.sessions.s0?.status).toBe('working')
})

test('buffered deltas are never committed by a lost connection: the resubscribe resumes before them', async () => {
  const daemon = await fakeDaemon(), store = sink()
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, pollIntervalMs: 5000, coalesceMs: 60_000 })
  cleanup.push(() => pool.stop()); await pool.start([workspace('w', daemon.socketPath)])
  daemon.push('event_batch', { subscription_id: 'sub', events: [upstream(1), delta(2, 'text_part', 'never stored')] })
  await until(() => store.rows.length === 1)
  daemon.drop()
  await until(() => daemon.requests.filter(row => row.method === 'events.subscribe').length === 2 && pool.health('w')?.state === 'online')
  expect(daemon.requests.filter(row => row.method === 'events.subscribe').at(-1)?.params.after_cursor).toBe('c1')
  await Bun.sleep(30)
  expect(store.rows.map(row => row.sequence)).toEqual([1])
})

test('journal backfill drains bounded pages, including events from the preceding daemon instance', async () => {
  const daemon = await fakeDaemon(), store = sink()
  daemon.setInstance('i2')
  daemon.respond(frame => frame.method === 'events.read' ? (frame.params.after_cursor === 'c1'
    ? { ok: true, events: [upstream(2, '/a', 'i2')], next_cursor: 'c2', has_more: false }
    : { ok: true, events: [upstream(1)], next_cursor: 'c1', has_more: true }) : undefined)
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry })
  cleanup.push(() => pool.stop()); await pool.start([workspace('w', daemon.socketPath)])
  expect(store.rows.map(row => row.sequence)).toEqual([1, 2])
  expect(daemon.requests.filter(row => row.method === 'events.read').map(row => row.params.max_bytes)).toEqual([256 * 1024, 256 * 1024])
  expect(pool.health('w')?.state).toBe('online')
})

test('journal partitions commit in upstream order across workspaces, not workspace order', async () => {
  const daemon = await fakeDaemon(), store = sink()
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry })
  cleanup.push(() => pool.stop()); await pool.start([workspace('a', daemon.socketPath), workspace('b', daemon.socketPath, '/b')])
  daemon.push('event_batch', { subscription_id: 'sub', events: [upstream(1, '/b'), upstream(2, '/a'), upstream(3, '/b')] })
  await until(() => store.rows.length === 3)
  expect(store.rows.map(row => row.sequence)).toEqual([1, 2, 3])
})

test('post-write cancellation produces unknown without cancelling daemon work or replaying a mutation', async () => {
  const daemon = await fakeDaemon(), store = sink(), controller = new AbortController()
  daemon.silence('turn.submit')
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry })
  cleanup.push(() => pool.stop()); await pool.start([workspace('w', daemon.socketPath)])
  const pending = pool.call('w', 'turn.submit', { session_key: 'k0', text: 'one' }, { signal: controller.signal })
  await until(() => daemon.requests.some(row => row.method === 'turn.submit'))
  controller.abort()
  expect(await pending).toMatchObject({ state: 'unknown' })
  expect(daemon.requests.filter(row => row.method === 'turn.submit')).toHaveLength(1)
  expect(daemon.requests.some(row => row.method === 'turn.cancel')).toBe(false)
  expect(pool.health('w')?.state).toBe('online')
})

test('independent navigation always uses explicit session keys and overrides summary history limits', async () => {
  const daemon = await fakeDaemon(), store = sink()
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry })
  cleanup.push(() => pool.stop()); await pool.start([workspace('a', daemon.socketPath), workspace('b', daemon.socketPath, '/b')])
  await Promise.all([pool.call('a', 'session.open', { session_key: 'alice', history_limit: 1000 }), pool.call('b', 'session.open', { session_key: 'bob' })])
  await pool.call('a', 'turn.steer', { session_key: 'alice', text: 'Alice only' })
  expect(daemon.requests.filter(row => row.method === 'session.open').map(row => row.params)).toEqual([
    { session_key: 'alice', project_dir: '/a', history_limit: 0 }, { session_key: 'bob', project_dir: '/b', history_limit: 0 },
  ])
  expect(daemon.requests.find(row => row.method === 'turn.steer')?.params.session_key).toBe('alice')
})

test('operator-only SSH failures do not retry, healthy workspaces stay online, explicit reconnect is possible', async () => {
  const daemon = await fakeDaemon(), store = sink()
  let attempts = 0, fixed = false
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, openTunnel: async () => {
    attempts++; if (!fixed) throw new HubOperatorError('SSH host-key verification failed')
    return { socketPath: daemon.socketPath, close() {} }
  } })
  cleanup.push(() => pool.stop())
  await pool.start([{ id: 'remote', label: 'Remote', endpoint: { kind: 'ssh', sshAlias: 'prod', remoteSocketPath: '/tmp/d.sock' }, projectDir: '/a', autoConnect: true }, workspace('local', daemon.socketPath)])
  await Bun.sleep(60)
  expect(attempts).toBe(1)
  expect(pool.health('remote')?.state).toBe('needs_operator')
  expect(pool.health('local')?.state).toBe('online')
  fixed = true; await pool.connectWorkspace('remote')
  expect(attempts).toBe(2)
  expect(pool.health('remote')?.state).toBe('online')
})

test('SSH operator failure during initialize remains needs_operator rather than turning into a retry storm', async () => {
  const daemon = await fakeDaemon(), store = sink()
  daemon.silence('initialize')
  let attempts = 0
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, openTunnel: async (_workspace, _signal, fail) => {
    attempts++; setTimeout(() => fail(new HubOperatorError('Remote daemon socket is missing; bootstrap consent is absent')), 10)
    return { socketPath: daemon.socketPath, close() {} }
  } })
  cleanup.push(() => pool.stop())
  await pool.start([{ id: 'remote', label: 'Remote', endpoint: { kind: 'ssh', sshAlias: 'prod', remoteSocketPath: '/tmp/d.sock' }, projectDir: '/a', autoConnect: true }])
  await Bun.sleep(60)
  expect(attempts).toBe(1)
  expect(pool.health('remote')?.state).toBe('needs_operator')
})

test('client correlates out-of-order responses, handles split UTF-8 and refuses oversized inbound frames', async () => {
  const daemon = await fakeDaemon(), events: string[] = [], states: string[] = []
  const client = new HubDaemonClient({ socketPath: daemon.socketPath, projectDir: '/a', ...retry, maxFrameBytes: 512 })
  cleanup.push(() => client.close()); await client.connect()
  client.onEvent((type, payload) => { if (type === 'public') events.push(String(payload.text)) })
  client.onState(state => states.push(state))
  const a = client.call('hang', { request: 'a' }), b = client.call('hang', { request: 'b' })
  await until(() => daemon.requests.filter(row => row.method === 'hang').length === 2)
  const requests = daemon.requests.filter(row => row.method === 'hang')
  daemon.raw({ jsonrpc: '2.0', id: requests[1]!.id, result: { ok: true, value: 'b' } })
  daemon.raw({ jsonrpc: '2.0', id: requests[0]!.id, result: { ok: true, value: 'a' } })
  expect((await a).value).toBe('a'); expect((await b).value).toBe('b')
  daemon.text('{"jsonrpc":"2.0","method":"event","params":{"type":"public",')
  daemon.text('"payload":{"text":"سلام"}}}\n')
  await until(() => events.length === 1)
  expect(events).toEqual(['سلام'])
  daemon.text('x'.repeat(513))
  await until(() => states.includes('disconnected'))
  expect(states).toContain('disconnected')
})

test('missing remote socket consent is surfaced as needs_operator without reaching a real SSH host', async () => {
  const store = sink(), pool = new HubRuntimePool({ sink: store.port, clientOptions: retry })
  cleanup.push(() => pool.stop())
  await pool.start([{ id: 'w', label: 'Remote', endpoint: { kind: 'ssh', sshAlias: 'never-contact-this-host' }, projectDir: '/a', autoConnect: true }])
  expect(pool.health('w')).toMatchObject({ state: 'needs_operator', message: expect.stringContaining('consent') })
})

test('legacy daemons without verified runtime identities are summary-polled but never merged by a guessed identity', async () => {
  const first = await fakeDaemon(), second = await fakeDaemon(), store = sink()
  for (const daemon of [first, second]) daemon.respond(frame => frame.method === 'runtime.status' ? { ok: true, daemon_protocol: 35 } : undefined)
  const pool = new HubRuntimePool({ sink: store.port, clientOptions: retry, pollIntervalMs: 15 })
  cleanup.push(() => pool.stop()); await pool.start([workspace('a', first.socketPath), workspace('b', second.socketPath)])
  expect(pool.runtimeCount).toBe(2)
  expect(pool.health('a')).toMatchObject({ state: 'online', degraded: true })
  expect(pool.health('a')?.runtimeId).toBeUndefined()
  expect(first.requests.some(row => row.method === 'events.subscribe')).toBe(false)
})

test('failed persistence does not publish the uncommitted event and reconnect resumes the previously committed cursor', async () => {
  const daemon = await fakeDaemon(), states: string[] = []
  let writes = 0
  const port: HubEventSink = { ingest() { writes++; throw new Error('disk unavailable') }, upstreamCursor() { return 'committed' } }
  const pool = new HubRuntimePool({ sink: port, clientOptions: retry, onHealth: (_id, health) => states.push(health.state) })
  cleanup.push(() => pool.stop()); await pool.start([workspace('w', daemon.socketPath)])
  daemon.push('event_batch', { subscription_id: 'sub', events: [upstream(1)] })
  await until(() => writes === 1 && daemon.requests.filter(row => row.method === 'events.subscribe').length === 2)
  expect(states).toContain('stale')
  expect(pool.projection('w')?.sessions.s0?.status).toBe('idle')
  expect(daemon.requests.filter(row => row.method === 'events.subscribe').at(-1)?.params.after_cursor).toBe('committed')
})

test('session-control lanes are bounded and real session navigation does not retarget a subsequent task', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xh-pool-')), socketPath = join(directory, 'd.sock')
  const server = new DaemonServer({ socketPath, projectDirectory: directory, runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') }) })
  await server.start()
  cleanup.push(async () => { await server.stop(); await rm(directory, { recursive: true, force: true }) })
  const pool = new HubRuntimePool({ sink: sink().port, maxSessionConnections: 1, clientOptions: { ...retry, requestTimeoutMs: 5000 } })
  cleanup.push(() => pool.stop()); await pool.start([workspace('w', socketPath, directory)])
  expect(pool.health('w')?.state).toBe('online')
  expect((await pool.call('w', 'session.open', { session_key: 'alice' })).state).toBe('accepted')
  expect((await pool.call('w', 'session.open', { session_key: 'bob' })).state).toBe('accepted')
  const alice = await pool.call('w', 'session.status', { session_key: 'alice' })
  expect(alice.state).toBe('accepted')
  if (alice.state === 'accepted') expect(alice.result.session).toMatchObject({ key: 'alice' })
})
