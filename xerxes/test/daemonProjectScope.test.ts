// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { DaemonServer } from '../src/daemon/server.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'

// Regression: a daemon launched with --project-dir from ANOTHER cwd must
// bind new sessions to that project, not to the process's launch directory.
test('initialize without project_dir uses the daemon project, not process cwd', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-proj-scope-'))
  const socketPath = join(directory, 'daemon.sock')
  const project = join(directory, 'the-project')
  await Bun.write(join(project, 'marker.txt'), 'x')
  const server = new DaemonServer({
    socketPath,
    projectDirectory: project,
    runtime: new InMemoryDaemonRuntime(undefined, {
      model: 'claude-code/default',
      sessionDirectory: join(directory, 'sessions'),
    }),
  })
  await server.start()
  const client = await SocketLite.connect(socketPath)
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { session_key: 'k1' } })
    const frame = await client.next((f) => f.id === 1)
    // resolveProjectDirectory realpaths; on macOS /var -> /private/var.
    const { realpathSync } = await import('node:fs')
    expect((frame.result as { cwd?: string } | undefined)?.cwd).toBe(realpathSync(project))
  } finally {
    client.close()
    await server.stop()
    await rm(directory, { recursive: true, force: true })
  }
})

test('a transcript deleted under a live daemon does not wedge initialize', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-flush-poison-'))
  const socketPath = join(directory, 'daemon.sock')
  const sessionDirectory = join(directory, 'sessions')
  const gate = { release: (): void => {} }
  const runner = {
    async *run(): AsyncGenerator<{ type: string; payload: Record<string, unknown> }> {
      yield { type: 'text_part', payload: { text: 'real exchange' } }
      await new Promise<void>((resolve) => { gate.release = resolve })
      yield { type: 'text_part', payload: { text: 'done' } }
    },
  }
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(runner, {
      model: 'claude-code/default',
      sessionDirectory,
    }),
  })
  await server.start()
  const client = await SocketLite.connect(socketPath)
  try {
    // A session with a completed exchange becomes flushable state.
    client.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { session_key: 'live-key' } })
    await client.next((f) => f.id === 1)
    client.send({ jsonrpc: '2.0', id: 2, method: 'turn.submit', params: { session_key: 'live-key', text: 'speak' } })
    gate.release()
    const settled = await client.next((f) => f.id === 2)
    expect((settled.error as { message?: string } | undefined)?.message).toBeUndefined()

    // The transcript is deleted OUT FROM UNDER the running daemon —
    // exactly what an external purge does.
    const { rm } = await import('node:fs/promises')
    await rm(join(sessionDirectory, 'live-key.json'), { force: true })
    await rm(join(sessionDirectory, 'live-key.jsonl'), { force: true })

    // A brand-new client binds a new session: the stale in-memory session
    // must be dropped at flush, not fail the bind.
    const checker = await SocketLite.connect(socketPath)
    checker.send({ jsonrpc: '2.0', id: 3, method: 'initialize', params: { session_key: 'fresh-key' } })
    const frame = await checker.next((f) => f.id === 3)
    expect((frame.error as { message?: string } | undefined)?.message).toBeUndefined()
    expect((frame.result as { ok?: boolean } | undefined)?.ok).toBe(true)
    checker.close()
  } finally {
    client.close()
    await server.stop()
    await rm(directory, { recursive: true, force: true })
  }
})

/** Minimal NDJSON client — keeps this test independent of the big suite's helpers. */
class SocketLite {
  private frames: Array<Record<string, unknown>> = []
  private waiters: Array<{ p: (f: Record<string, unknown>) => boolean; r: (f: Record<string, unknown>) => void }> = []
  private socket: import('node:net').Socket

  private constructor(socket: import('node:net').Socket) {
    this.socket = socket
    let buf = ''
    socket.on('data', (d) => {
      buf += d
      let i
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i)
        buf = buf.slice(i + 1)
        if (!line.trim()) continue
        const f = JSON.parse(line) as Record<string, unknown>
        const idx = this.waiters.findIndex((w) => w.p(f))
        if (idx >= 0) this.waiters.splice(idx, 1)[0]!.r(f)
        else this.frames.push(f)
      }
    })
  }

  static connect(path: string): Promise<SocketLite> {
    return new Promise((resolve) => {
      const { connect } = require('node:net') as typeof import('node:net')
      const s = connect(path)
      s.once('connect', () => resolve(new SocketLite(s)))
    })
  }

  next(p: (f: Record<string, unknown>) => boolean): Promise<Record<string, unknown>> {
    const i = this.frames.findIndex(p)
    if (i >= 0) return Promise.resolve(this.frames.splice(i, 1)[0]!)
    return new Promise((r) => this.waiters.push({ p, r }))
  }

  send(f: Record<string, unknown>): void {
    this.socket.write(`${JSON.stringify(f)}\n`)
  }

  close(): void {
    this.socket.destroy()
  }
}
