// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createServer, type Server, type Socket as NetSocket } from 'node:net'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, expect, test } from 'bun:test'

import { DaemonRpc, MAX_FRAME_BYTES } from '../src/desktop/main/daemon.js'

// Transport contract for the fresh DaemonRpc, exercised against an
// in-process fake daemon on a temp socket. Live auto-launch is covered by the
// manual smoke; here everything is deterministic.

class FakeDaemon {
  readonly connections: NetSocket[] = []
  requests: Array<{ id: unknown; method: string }> = []
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
            const parsed = JSON.parse(line) as { id?: unknown; method?: string }
            const method = String(parsed.method ?? '')
            this.requests.push({ id: parsed.id, method })
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
    const conn = this.connections[0]
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
