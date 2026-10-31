// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { HubDaemonClient } from '../src/hub/daemonClient.js'

interface Frame { readonly id?: number; readonly method?: string; readonly params?: { readonly type?: string }; readonly result?: Record<string, unknown> }

class Client {
  private buffer = ''
  private readonly frames: Frame[] = []
  private readonly waiters: Array<{ predicate: (frame: Frame) => boolean; resolve: (frame: Frame) => void }> = []
  private constructor(private readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', chunk => this.receive(String(chunk)))
  }
  static async connect(path: string): Promise<Client> {
    const socket = connect({ path })
    await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) })
    return new Client(socket)
  }
  close(): void { this.socket.destroy() }
  async call(id: number, method: string, params: Record<string, unknown>): Promise<Record<string, unknown>> {
    this.socket.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
    return (await this.next(frame => frame.id === id)).result ?? {}
  }
  next(predicate: (frame: Frame) => boolean): Promise<Frame> {
    const index = this.frames.findIndex(predicate)
    if (index >= 0) return Promise.resolve(this.frames.splice(index, 1)[0]!)
    return new Promise(resolve => this.waiters.push({ predicate, resolve }))
  }
  private receive(chunk: string): void {
    this.buffer += chunk
    for (let newline = this.buffer.indexOf('\n'); newline >= 0; newline = this.buffer.indexOf('\n')) {
      const frame = JSON.parse(this.buffer.slice(0, newline)) as Frame
      this.buffer = this.buffer.slice(newline + 1)
      const waiter = this.waiters.findIndex(entry => entry.predicate(frame))
      if (waiter >= 0) this.waiters.splice(waiter, 1)[0]!.resolve(frame)
      else this.frames.push(frame)
    }
  }
}

async function fixture() {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-bare-init-')))
  const socketPath = join(directory, 'daemon.sock')
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, model: 'm', sessionDirectory: join(directory, 'sessions') })
  const server = new DaemonServer({ runtime, socketPath })
  await server.start()
  const owner = await Client.connect(socketPath)
  // A chat with history, then resumed: the live session is bound under its own id.
  const opened = await owner.call(1, 'initialize', { project_dir: directory, session_key: 'seed-key' })
  const id = String((opened.session as { id?: string } | undefined)?.id ?? '')
  await owner.call(2, 'turn.submit', { session_key: 'seed-key', text: 'remember this turn' })
  await owner.next(frame => frame.method === 'event' && frame.params?.type === 'turn_end')
  // `/resume <id>` rebinds the chat under its own id, as on the real host.
  await owner.call(3, 'slash', { command: `/resume ${id}` })
  const messages = runtime.sessionStatus(id)?.messages.length ?? 0
  return { directory, socketPath, runtime, server, owner, id, messages,
    async close() { owner.close(); await server.stop(); await rm(directory, { recursive: true, force: true }) } }
}

test('a bare initialize whose key is a chat id attaches that chat instead of replacing it with an empty session', async () => {
  const f = await fixture()
  const other = await Client.connect(f.socketPath)
  try {
    expect(f.id).toMatch(/^[0-9a-f]{8,32}$/)
    expect(f.runtime.sessionStatus(f.id)?.sessionKey).toBe(f.id)
    expect(f.messages).toBeGreaterThan(0)
    // What a Hub per-task lane used to send on every (re)connect.
    const attached = await other.call(1, 'initialize', { project_dir: f.directory, session_key: f.id, history_limit: 0 })
    expect(attached.ok).not.toBe(false)
    expect((attached.session as { id?: string } | undefined)?.id).toBe(f.id)
    const live = f.runtime.listSessions().filter(session => session.id === f.id)
    expect(live).toHaveLength(1)
    expect(live[0]!.messages.length).toBe(f.messages)
    expect(live[0]!.turnCount).toBeGreaterThan(0)
  } finally { other.close(); await f.close() }
})

test('a Hub per-task lane keyed by a chat id resumes it by id and leaves its history intact', async () => {
  const f = await fixture()
  const lane = new HubDaemonClient({ socketPath: f.socketPath, projectDir: f.directory, sessionKey: f.id, requestTimeoutMs: 5000, retryBaseMs: 60_000 })
  try {
    const handshake = await lane.connect()
    expect((handshake.initialized.session as { id?: string } | undefined)?.id).toBe(f.id)
    expect(f.runtime.sessionStatus(f.id)?.messages.length).toBe(f.messages)
  } finally { lane.close(); await f.close() }
})

test('a bare initialize on a non-id slot key still starts that slot fresh', async () => {
  const f = await fixture()
  const other = await Client.connect(f.socketPath)
  try {
    await other.call(1, 'initialize', { project_dir: f.directory, session_key: 'slot-key' })
    await other.call(2, 'turn.submit', { session_key: 'slot-key', text: 'hello' })
    await other.next(frame => frame.method === 'event' && frame.params?.type === 'turn_end')
    const before = f.runtime.sessionStatus('slot-key')?.id
    await other.call(3, 'initialize', { project_dir: f.directory, session_key: 'slot-key' })
    const after = f.runtime.sessionStatus('slot-key')
    expect(after?.messages.length).toBe(0)
    expect(after?.id).not.toBe(before)
  } finally { other.close(); await f.close() }
})
