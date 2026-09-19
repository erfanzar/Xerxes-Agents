// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createServer, type Server, type Socket as NetSocket } from 'node:net'
import { spawn } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, expect, test } from 'bun:test'

import { DaemonRpc, MAX_FRAME_BYTES } from '../src/desktop/main/daemon.js'

test('slow daemon startup retries reuse the child and recover when its socket appears', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xds-'))
  const path = join(directory, 'daemon.sock')
  const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' })
  const exited = new Promise<void>(resolve => child.once('exit', () => resolve()))
  let launches = 0
  const rpc = new DaemonRpc({
    projectDir: directory, env: { XERXES_DAEMON_SOCKET: path }, startupTimeoutMs: 25,
    launch: () => { launches += 1; return child },
  })
  const daemon = new FakeDaemon(path, ['runtime.status'])
  try {
    await expect(rpc.call('runtime.status', {})).rejects.toThrow('still starting')
    await expect(rpc.call('runtime.status', {})).rejects.toThrow('still starting')
    expect(launches).toBe(1)
    await daemon.listen()
    expect(await rpc.call<{ ok: boolean }>('runtime.status', {})).toEqual({ ok: true })
    expect(launches).toBe(1)
  } finally {
    rpc.dispose()
    daemon.close()
    child.kill('SIGTERM')
    await exited
    rmSync(directory, { recursive: true, force: true })
  }
})

// Transport contract for the fresh DaemonRpc, exercised against an
// in-process fake daemon on a temp socket. Live auto-launch is covered by the
// manual smoke; here everything is deterministic.

class FakeDaemon {
  readonly connections: NetSocket[] = []
  requests: Array<{ id: unknown; method: string; params?: Record<string, unknown> }> = []
  private server: Server
  /** Methods answered automatically with `{ok:true}`; everything else needs an explicit reply. */
  private readonly autoReply: Set<string>

  constructor(readonly socketPath: string, autoReply: string[] = []) {
    this.autoReply = new Set(autoReply)
    this.server = createServer(socket => {
      this.connections.push(socket)
      socket.setEncoding('utf8')
      socket.on('error', () => {})
      let buffer = ''
      socket.on('data', chunk => {
        buffer += chunk
        let nl = buffer.indexOf('\n')
        while (nl !== -1) {
          const line = buffer.slice(0, nl)
          buffer = buffer.slice(nl + 1)
          if (!line.trim()) continue
          try {
            const parsed = JSON.parse(line) as { id?: unknown; method?: string; params?: Record<string, unknown> }
            const method = String(parsed.method ?? '')
            this.requests.push({ id: parsed.id, method, ...(parsed.params ? { params: parsed.params } : {}) })
            if (parsed.id !== undefined && this.autoReply.has(method)) {
              // Deferred a tick: replying synchronously from inside this same
              // socket's data handler wedges subsequent delivery under Bun.
              setImmediate(() => this.reply(parsed.id, { ok: true }))
            }
          } catch {
            // Oversized probes intentionally fail to parse.
          }
          nl = buffer.indexOf('\n')
        }
      })
    })
  }

  listen(): Promise<void> {
    return new Promise(resolve => this.server.listen(this.socketPath, resolve))
  }

  reply(id: unknown, result: unknown): void {
    this.send(`${JSON.stringify({ jsonrpc: '2.0', id, result })}\n`)
  }

  failWith(id: unknown, code: number, message: string): void {
    this.send(`${JSON.stringify({ jsonrpc: '2.0', id, error: { code, message } })}\n`)
  }

  event(type: string, payload: Record<string, unknown>): void {
    this.send(`${JSON.stringify({ method: 'event', params: { type, payload } })}\n`)
  }

  raw(text: string): void {
    this.send(text)
  }

  /**
   * One framed write plus an explicit flush. Bun defers same-process socket
   * flushes on an internal timer (observed multi-second stalls in these
   * tests); the flush is a no-op on runtimes without it.
   */
  private send(text: string): void {
    const conn = this.connections.findLast(connection => !connection.destroyed)
    if (!conn) return
    conn.write(text)
    ;(conn as NetSocket & { flush?: () => void }).flush?.()
  }

  close(): void {
    for (const connection of this.connections) connection.destroy()
    this.server.close()
  }
}

let dir: string
let socketPath: string

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'xd')) // short: macOS caps socket paths
  socketPath = join(dir, 'd.sock')
})

afterEach(() => {
  rmSync(dir, { recursive: true, force: true })
})

const env = (): Record<string, string> => ({
  XERXES_DAEMON_SOCKET: socketPath,
  XERXES_HOME: join(dir, 'home'),
})

function client(deadlineMs = 2000): DaemonRpc {
  return new DaemonRpc({
    projectDir: join(dir, 'project'),
    env: env(),
    deadlineMs,
  })
}

const until = async (check: () => boolean, what: string, budgetMs = 3000): Promise<void> => {
  const deadline = Date.now() + budgetMs
  while (!check()) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`)
    await new Promise(resolve => setTimeout(resolve, 10))
  }
}

test('requests correlate by id; results resolve', async () => {
  const daemon = new FakeDaemon(socketPath)
  await daemon.listen()
  const rpc = client()
  const pending = rpc.call<{ ok: boolean }>('ping', { n: 1 })
  await until(() => daemon.requests.length === 1, 'request frame')
  daemon.reply(daemon.requests[0]!.id, { ok: true })
  expect(await pending).toEqual({ ok: true })
  rpc.dispose()
  daemon.close()
})

test('rpc-level errors reject with code and message', async () => {
  const daemon = new FakeDaemon(socketPath)
  await daemon.listen()
  const rpc = client()
  const pending = rpc.call('nope')
  await until(() => daemon.requests.length === 1, 'request frame')
  daemon.failWith(daemon.requests[0]!.id, -32000, 'no active session')
  await expect(pending).rejects.toThrow(/rpc -32000: no active session/)
  rpc.dispose()
  daemon.close()
})

test('events fan out with type and payload', async () => {
  const daemon = new FakeDaemon(socketPath, ['bootstrap'])
  await daemon.listen()
  const rpc = client()
  const seen: string[] = []
  rpc.onEvent(type => seen.push(type))
  await rpc.call('bootstrap', {})
  daemon.event('turn_begin', { session_id: 's' })
  daemon.event('text_part', { text: 'hi' })
  await until(() => seen.length === 2, 'two events')
  expect(seen).toEqual(['turn_begin', 'text_part'])
  rpc.dispose()
  daemon.close()
})

test('frames split across writes reassemble; neighbouring frames survive', async () => {
  const daemon = new FakeDaemon(socketPath, ['warm'])
  await daemon.listen()
  const rpc = client()
  const events: string[] = []
  rpc.onEvent(type => events.push(type))
  await rpc.call('warm', {})
  // Feed the split at the framing layer: cross-process delivery is covered by
  // the live smoke, same-process loopback second-writes stall under Bun.
  const feed = (rpc as unknown as { onData: (chunk: string) => void }).onData.bind(rpc)
  const response = JSON.stringify({ jsonrpc: '2.0', id: 7, result: { part: true } })
  feed(response.slice(0, 12))
  feed(
    `${response.slice(12)}\n${JSON.stringify({ method: 'event', params: { type: 'joined', payload: {} } })}\n`,
  )
  await until(() => events.includes('joined'), 'cross-frame event', 3000)
  rpc.dispose()
  daemon.close()
})

test('oversized frames reject the waiters and drop the connection', async () => {
  const daemon = new FakeDaemon(socketPath, ['warm'])
  await daemon.listen()
  const rpc = client(2000)
  const errors: string[] = []
  rpc.on('protocol_error', payload => errors.push(String((payload as { message: string }).message)))
  await rpc.call('warm', {})
  const pending = rpc.call('big')
  // Attach the rejection watcher before feeding: the guard fires mid-flight.
  const rejected = { done: false, message: '' }
  pending.catch(error => {
    rejected.done = true
    rejected.message = error.message
  })
  await until(() => daemon.requests.length === 1, 'request frame')
  const feed = (rpc as unknown as { onData: (chunk: string) => void }).onData.bind(rpc)
  feed(`x${'é'.repeat(MAX_FRAME_BYTES + 64)}\n`)
  await until(() => rejected.done && rejected.message.includes('maximum size'), 'waiter rejection', 3000)
  await until(() => errors.some(m => m.includes('maximum size')), 'protocol error', 3000)
  expect(rpc.online).toBe(false)
  rpc.dispose()
  daemon.close()
})

test('daemon hangup rejects in-flight requests', async () => {
  const daemon = new FakeDaemon(socketPath, ['warm'])
  await daemon.listen()
  const rpc = client()
  await rpc.call('warm', {})
  const pending = rpc.call('dies')
  await until(() => daemon.requests.length >= 2, 'second request frame')
  daemon.connections[0]?.destroy()
  await expect(pending).rejects.toThrow(/connection closed/)
  rpc.dispose()
  daemon.close()
})

test('dispose stops retry loops and rejects new calls immediately', async () => {
  const daemon = new FakeDaemon(socketPath, ['warm'])
  await daemon.listen()
  const rpc = client()
  await rpc.call('warm', {})
  rpc.dispose()
  expect(rpc.online).toBe(false)
  await expect(rpc.call('after')).rejects.toThrow(/disposed/)
  daemon.close()
})

// ── Launch helpers ──────────────────────────────────────────────────────

import { daemonArgv, daemonEntryOf, bunBinaryOf } from '../src/desktop/main/spawn.js'

/** Create an empty file including its parent directories. */
function touch(path: string): string {
  mkdirSync(join(path, '..'), { recursive: true })
  writeFileSync(path, '')
  return path
}

test('daemon entry resolution: source first, dist fallback, explicit override', () => {
  const source = touch(join(dir, 'p', 'xerxes', 'src', 'cli.ts'))
  expect(daemonEntryOf(join(dir, 'p'), env())).toBe(source)

  const dist = touch(join(dir, 'q', 'xerxes', 'dist', 'cli.js'))
  expect(daemonEntryOf(join(dir, 'q'), env())).toBe(dist)

  const custom = touch(join(dir, 'custom.ts'))
  expect(daemonEntryOf(dir, { XERXES_TUI_BUN_DAEMON: custom })).toBe(custom)
  expect(() => daemonEntryOf(dir, { XERXES_TUI_BUN_DAEMON: '/missing/cli.ts' })).toThrow(/does not exist/)
})

test('daemon entry resolution falls back to the app checkout for any workspace', () => {
  // Built layout: <checkout>/dist/desktop — the app serves a daemon for a
  // workspace that has no runtime of its own.
  const builtCli = touch(join(dir, 'checkout', 'src', 'cli.ts'))
  const builtAppDir = join(dir, 'checkout', 'dist', 'desktop')
  expect(daemonEntryOf(join(dir, 'some-workspace'), {}, builtAppDir)).toBe(builtCli)

  // Source layout: <checkout>/src/desktop/main.
  const sourceAppDir = join(dir, 'checkout', 'src', 'desktop', 'main')
  expect(daemonEntryOf(join(dir, 'other-workspace'), {}, sourceAppDir)).toBe(builtCli)

  // Monorepo: the app ships from <repo>/xerxes, one level deeper.
  const monoCli = touch(join(dir, 'repo', 'xerxes', 'src', 'cli.ts'))
  const monoAppDir = join(dir, 'repo', 'xerxes', 'dist', 'desktop')
  expect(daemonEntryOf(join(dir, 'elsewhere'), {}, monoAppDir)).toBe(monoCli)

  // Workspace-relative still wins over app-relative.
  const ownCli = touch(join(dir, 'own', 'xerxes', 'src', 'cli.ts'))
  expect(daemonEntryOf(join(dir, 'own'), {}, join(dir, 'checkout', 'dist', 'desktop'))).toBe(ownCli)

  // Packaged bundle: Resources/app/main.js resolves the runtime copied to
  // Resources/runtime/cli.js.
  const packagedCli = touch(join(dir, 'packaged', 'Contents', 'Resources', 'runtime', 'cli.js'))
  const packagedAppDir = join(dir, 'packaged', 'Contents', 'Resources', 'app')
  expect(daemonEntryOf(join(dir, 'chosen-workspace'), {}, packagedAppDir)).toBe(packagedCli)

  // Nothing anywhere: the actionable error stands.
  expect(() => daemonEntryOf(dir, {}, join(dir, 'nowhere'))).toThrow(/Could not locate the Bun daemon entry/)
})

test('argv matches the frozen daemon launch contract', () => {
  const project = touch(join(dir, 'argv', 'xerxes', 'src', 'cli.ts'))
  const root = join(project, '..', '..', '..')
  const { binary, args } = daemonArgv(root, '/s/d.sock', '/s/d.pid', {})
  expect(binary).toBe('bun')
  expect(args.slice(1)).toEqual([
    'daemon', '--project-dir', root, '--socket', '/s/d.sock', '--pid-file', '/s/d.pid',
  ])
  expect(bunBinaryOf({ XERXES_TUI_BUN: '/opt/bun' })).toBe('/opt/bun')
})

test('SSH socket attachment keeps the remote project path and never launches locally',async()=>{
 const fake=new FakeDaemon(socketPath,['runtime.status']);await fake.listen()
 const rpc=new DaemonRpc({projectDir:'/remote/project',socketPath,env:{XERXES_BUN:'/must-not-launch'},deadlineMs:2000})
 try{expect(await rpc.call<Record<string,unknown>>('runtime.status')).toEqual({ok:true});expect(rpc.projectDir).toBe('/remote/project')}
 finally{rpc.dispose();fake.close()}
 const missing=new DaemonRpc({projectDir:'/remote/project',socketPath:join(dir,'missing.sock'),env:{XERXES_BUN:'/must-not-launch'}})
 try{await expect(missing.call('runtime.status')).rejects.toThrow('SSH transport unavailable')}
 finally{missing.dispose()}
})

test('runtime update refuses remote transports without issuing a shutdown', async () => {
  const rpc = new DaemonRpc({ projectDir: '/remote', socketPath: '/missing' })
  expect(await rpc.restartRuntime()).toMatchObject({ ok: false, error: expect.stringContaining('remote machine') })
  rpc.dispose()
})

test('managed SSH update uses its host callback and reports the installed remote build', async () => {
  const fake = new FakeDaemon(socketPath, ['initialize'])
  await fake.listen()
  let updates = 0
  const rpc = new DaemonRpc({ projectDir: '/remote', socketPath,
    expectedRemoteBuildId: () => 'abcdef0123456789',
    remoteUpdate: async () => { updates++; return { ok: false, busy: true } },
  })
  try {
    expect(await rpc.call('initialize')).toMatchObject({ desktop_expected_daemon_build_id: 'abcdef0123456789' })
    expect(await rpc.restartRuntime()).toEqual({ ok: false, busy: true })
    expect(updates).toBe(1)
    expect(fake.requests.map(row => row.method)).toEqual(['initialize'])
  } finally { rpc.dispose(); fake.close() }
})

test('runtime update leaves busy and unsupported daemons connected', async () => {
  const daemon = new FakeDaemon(socketPath)
  await daemon.listen()
  const rpc = client()
  try {
    const pending = rpc.restartRuntime()
    await until(() => daemon.requests.length === 1, 'restart request')
    expect(daemon.requests[0]!.method).toBe('runtime.restart_if_idle')
    daemon.reply(daemon.requests[0]!.id, { ok: false, busy: true })
    expect(await pending).toEqual({ ok: false, busy: true })
    expect(rpc.online).toBe(true)
    const unsupported = rpc.restartRuntime()
    await until(() => daemon.requests.length === 2, 'second restart request')
    daemon.reply(daemon.requests[1]!.id, { ok: false, error: 'Unknown method' })
    expect(await unsupported).toMatchObject({ ok: false, error: expect.stringContaining('older runtime') })
    expect(rpc.online).toBe(true)
  } finally { rpc.dispose(); daemon.close() }
})

test('runtime update waits for the old connection to close before reconnecting', async () => {
  const daemon = new FakeDaemon(socketPath)
  await daemon.listen()
  const rpc = client()
  let completed = false
  try {
    const pending = rpc.restartRuntime().then(result => { completed = true; return result })
    await until(() => daemon.requests.length === 1, 'restart request')
    daemon.reply(daemon.requests[0]!.id, { ok: true })
    await new Promise(resolve => setTimeout(resolve, 40))
    expect(completed).toBe(false)
    // Keep the listener available, as a supervisor would, but replace the connection.
    daemon.connections[0]!.destroy()
    expect(await pending).toEqual({ ok: true })
    expect(daemon.connections.length).toBeGreaterThan(1)
  } finally { rpc.dispose(); daemon.close() }
})

test('explicit legacy restart checks activity and shuts down before reconnecting', async () => {
  const daemon = new FakeDaemon(socketPath)
  await daemon.listen()
  const rpc = client()
  try {
    const pending = rpc.restartRuntime(true)
    const replies: Array<[string, Record<string, unknown>]> = [
      ['runtime.restart_if_idle', { ok: false, error: 'Unknown method: runtime.restart_if_idle' }],
      ['runtime.status', { ok: true, active_subagents: 0, channels_configured: false }],
      ['session.active_list', { ok: true, sessions: [{ key: 'session', status: 'idle', active_turn_id: '' }] }],
      ['terminal.list', { ok: true, terminals: [] }],
      ['monitor.list', { ok: true, monitors: [] }],
      ['shutdown', { ok: true }],
    ]
    for (const [index, [method, reply]] of replies.entries()) {
      await until(() => daemon.requests.length > index, method)
      expect(daemon.requests[index]!.method).toBe(method)
      daemon.reply(daemon.requests[index]!.id, reply)
    }
    await new Promise(resolve => setTimeout(resolve, 30))
    daemon.connections[0]!.destroy()
    expect(await pending).toEqual({ ok: true })
  } finally { rpc.dispose(); daemon.close() }
})

test('explicit legacy restart refuses another session that is working', async () => {
  const daemon = new FakeDaemon(socketPath)
  await daemon.listen()
  const rpc = client()
  try {
    const pending = rpc.restartRuntime(true)
    await until(() => daemon.requests.length === 1, 'restart probe')
    daemon.reply(daemon.requests[0]!.id, { ok: false, error: 'Unknown method' })
    await until(() => daemon.requests.length === 2, 'status')
    daemon.reply(daemon.requests[1]!.id, { ok: true, active_subagents: 0, channels_configured: false })
    await until(() => daemon.requests.length === 3, 'sessions')
    daemon.reply(daemon.requests[2]!.id, { ok: true, sessions: [{ key: 'other', status: 'working' }] })
    expect(await pending).toEqual({ ok: false, busy: true })
    expect(daemon.requests.some(row => row.method === 'shutdown')).toBe(false)
  } finally { rpc.dispose(); daemon.close() }
})


test('opening and resuming a desktop session uses its window target on the shared daemon', async () => {
  const fake = new FakeDaemon(socketPath, ['initialize', 'session.open', 'runtime.status']); await fake.listen()
  const rpc = new DaemonRpc({ projectDir: '/second/workspace', socketPath, deadlineMs: 2000 })
  try {
    await rpc.call('initialize', { session_key: 'second' })
    await rpc.call('initialize', { resume_session_id: 'saved', project_dir: '/wrong' })
    await rpc.call('session.open', { session_key: 'another' })
    await rpc.call('runtime.status')
    expect(fake.requests.filter(row => row.method === 'initialize' || row.method === 'session.open').map(row => row.params)).toEqual([
      { session_key: 'second', project_dir: '/second/workspace' },
      { resume_session_id: 'saved', project_dir: '/second/workspace' },
      { session_key: 'another', project_dir: '/second/workspace' },
    ])
    expect(fake.requests.findLast(row => row.method === 'runtime.status')?.params).toEqual({})
  } finally { rpc.dispose(); fake.close() }
})

test('initialize uses structured history without duplicate legacy replay events', async () => {
  const fake = new FakeDaemon(socketPath); await fake.listen()
  const rpc = new DaemonRpc({ projectDir: '/project', socketPath, deadlineMs: 2000 })
  const events: string[] = []; rpc.onEvent((_type, payload) => events.push(String(payload.body)))
  try {
    const initializing = rpc.call('initialize', { resume_session_id: 'saved' })
    for (let n = 0; n < 100 && !fake.requests.some(row => row.method === 'initialize'); n++) await Bun.sleep(5)
    const request = fake.requests.find(row => row.method === 'initialize')!
    fake.raw(JSON.stringify({ method: 'event', params: { type: 'notification', payload: { category: 'history', type: 'replay_user', body: 'Stored message' } } }) + '\n')
    fake.raw(JSON.stringify({ method: 'event', params: { type: 'notification', payload: { category: 'runtime', body: 'Live warning' } } }) + '\n')
    fake.reply(request.id, { ok: true, session: { messages: [{ role: 'user', content: 'Stored message' }] } })
    expect(await initializing).toMatchObject({ session: { messages: [{ content: 'Stored message' }] } })
    expect(events).toEqual(['Live warning'])
    fake.raw(JSON.stringify({ method: 'event', params: { type: 'notification', payload: { category: 'history', body: 'Explicit slash replay' } } }) + '\n')
    for (let n = 0; n < 100 && events.length < 2; n++) await Bun.sleep(5)
    expect(events).toEqual(['Live warning', 'Explicit slash replay'])
  } finally { rpc.dispose(); fake.close() }
})

test('dead SSH tunnel is rebuilt once for concurrent requests without a local launch', async () => {
  const fake = new FakeDaemon(socketPath, ['initialize', 'runtime.status']); await fake.listen()
  let reconnects = 0
  const rpc = new DaemonRpc({ projectDir: '/remote/project', socketPath: join(dir, 'dead.sock'),
    launch: () => { throw new Error('Must not launch locally') },
    reconnectRemote: async () => { reconnects++; await Bun.sleep(10); return socketPath },
  })
  try {
    const [initialized, status] = await Promise.all([rpc.call('initialize', { resume_session_id: 'original' }), rpc.call('runtime.status')])
    expect(initialized).toEqual({ ok: true }); expect(status).toEqual({ ok: true })
    expect(reconnects).toBe(1)
    expect(fake.requests.find(row => row.method === 'initialize')?.params).toMatchObject({ resume_session_id: 'original', project_dir: '/remote/project' })
  } finally { rpc.dispose(); fake.close() }
})
test('SSH recovery errors remain actionable and disposal cancels an outstanding reconnect', async () => {
  const rejected = new DaemonRpc({ socketPath: join(dir, 'dead.sock'), reconnectRemote: async () => { throw new Error('Host key verification failed') } })
  try { await expect(rejected.call('runtime.status')).rejects.toThrow('Host key verification failed') } finally { rejected.dispose() }
  let started = false, aborted = false
  const rpc = new DaemonRpc({ socketPath: join(dir, 'dead.sock'), reconnectRemote: signal => new Promise((_resolve, reject) => {
    started = true
    signal.addEventListener('abort', () => { aborted = true; reject(new Error('Connection cancelled')) }, { once: true })
  }) })
  const pending = rpc.call('runtime.status')
  const result = pending.catch(error => error)
  await until(() => started, 'SSH recovery start')
  rpc.dispose(); expect((await result).message).toBe('Connection cancelled'); expect(aborted).toBe(true); expect(rpc.online).toBe(false)
})

test('desktop negotiates a lease and reclaims it before resuming after a socket loss', async () => {
  const fake = new FakeDaemon(socketPath); await fake.listen()
  const rpc = new DaemonRpc({ projectDir: '/remote/project', socketPath, deadlineMs: 2000 })
  try {
    const opening = rpc.call('initialize', { resume_session_id: 'saved' })
    await until(() => fake.requests.length === 1, 'initialize')
    fake.reply(fake.requests[0]!.id, { ok: true, connection_lease_supported: true })
    await until(() => fake.requests.length === 2, 'enable lease')
    expect(fake.requests[1]!.method).toBe('connection.lease')
    fake.reply(fake.requests[1]!.id, { ok: true, token: 'private-lease' })
    await opening
    fake.connections.at(-1)!.destroy()
    await until(() => !rpc.online, 'disconnected')
    const resuming = rpc.call('initialize', { resume_session_id: 'saved' })
    await until(() => fake.requests.length === 3, 'reclaim lease')
    expect(fake.requests[2]).toMatchObject({ method: 'connection.lease', params: { token: 'private-lease', project_dir: '/remote/project' } })
    fake.reply(fake.requests[2]!.id, { ok: true, token: 'private-lease' })
    await until(() => fake.requests.length === 4, 'resume initialize')
    expect(fake.requests[3]).toMatchObject({ method: 'initialize', params: { resume_session_id: 'saved' } })
    fake.reply(fake.requests[3]!.id, { ok: true, connection_lease_supported: true })
    await resuming
    expect(fake.requests).toHaveLength(4)
  } finally { rpc.dispose(); fake.close() }
})
