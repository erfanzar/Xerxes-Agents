// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { connect, type Socket } from 'node:net'
import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'
import { nativeWebSocketMonitorSource } from '../src/runtime/websocketMonitorSource.js'

class SocketClient {
  private buffer = ''
  private readonly frames: Array<Record<string, unknown>> = []
  private readonly waiters: Array<() => void> = []
  private constructor(private readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', chunk => {
      this.buffer += String(chunk)
      let index = this.buffer.indexOf('\n')
      while (index >= 0) {
        const line = this.buffer.slice(0, index).trim()
        this.buffer = this.buffer.slice(index + 1)
        if (line) this.frames.push(JSON.parse(line) as Record<string, unknown>)
        index = this.buffer.indexOf('\n')
      }
      for (const wake of this.waiters.splice(0)) wake()
    })
  }
  static connect(path: string): Promise<SocketClient> {
    return new Promise((resolve, reject) => {
      const socket = connect(path)
      socket.once('connect', () => resolve(new SocketClient(socket)))
      socket.once('error', reject)
    })
  }
  send(id: number, method: string, params: Record<string, unknown>): void {
    this.socket.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
  }
  async response(id: number): Promise<Record<string, unknown>> {
    const deadline = Date.now() + 3_000
    for (;;) {
      const index = this.frames.findIndex(frame => frame.id === id)
      if (index >= 0) return this.frames.splice(index, 1)[0]!
      if (Date.now() >= deadline) throw new Error(`response ${id} not received`)
      await new Promise<void>(resolve => { this.waiters.push(resolve); setTimeout(resolve, 20) })
    }
  }
  close(): void { this.socket.destroy() }
}

function result(frame: Record<string, unknown>): Record<string, unknown> { return (frame.result ?? {}) as Record<string, unknown> }

async function waitFor<T>(read: () => T | Promise<T>, predicate: (value: T) => boolean): Promise<T> {
  const deadline = Date.now() + 3_000
  let value = await read()
  while (!predicate(value)) {
    if (Date.now() >= deadline) throw new Error(`condition not met: ${JSON.stringify(value)}`)
    await Bun.sleep(30)
    value = await read()
  }
  return value
}

function localWebSocketServer(onSocket: (socket: Bun.ServerWebSocket<unknown>) => void, onClose: () => void): { readonly url: string; readonly server: Bun.Server<unknown> } {
  const server = Bun.serve<unknown>({
    port: 0,
    fetch(request, instance) {
      if (instance.upgrade(request, { data: undefined })) return
      return new Response('websocket only', { status: 426 })
    },
    websocket: { open: onSocket, close: onClose, message() {} },
  })
  return { url: `ws://127.0.0.1:${server.port}/events`, server }
}

async function fixture(socketPath: string, projectDirectory: string, runtime: InMemoryDaemonRuntime, history: RunHistory, terminals: TerminalRegistry, source: { open: typeof nativeWebSocketMonitorSource.open }): Promise<{ server: DaemonServer; monitors: TerminalMonitors }> {
  const monitors = new TerminalMonitors(terminals, history, undefined, undefined, undefined, undefined, {
    source,
    resolveWorkspace: owner => runtime.listSessions().find(session => session.id === owner)?.cwd ?? (() => { throw new Error('WebSocket monitor owner session is unavailable') })(),
  })
  const server = new DaemonServer({ socketPath, projectDirectory, runtime, runHistory: history, terminalRegistry: terminals, monitors, autoTitle: false })
  await server.start()
  return { server, monitors }
}

test('websocket monitor RPC filters and deduplicates frames and enforces ownership', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-websocket-monitor-rpc-'))
  const workspace = join(root, 'workspace')
  await mkdir(workspace, { recursive: true })
  const sockets: Bun.ServerWebSocket<unknown>[] = []
  let closed = 0
  const websocket = localWebSocketServer(socket => sockets.push(socket), () => { closed += 1 })
  const history = new RunHistory(join(root, 'runs.sqlite'))
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: workspace, sessionDirectory: join(root, 'sessions') })
  const terminals = new TerminalRegistry({ runHistory: history })
  const socketPath = join(root, 'daemon.sock')
  const { server, monitors } = await fixture(socketPath, workspace, runtime, history, terminals, nativeWebSocketMonitorSource)
  const owner = await SocketClient.connect(socketPath)
  const foreign = await SocketClient.connect(socketPath)
  try {
    owner.send(1, 'initialize', { session_key: 'monitor-owner', project_dir: workspace })
    foreign.send(2, 'initialize', { session_key: 'monitor-foreign', project_dir: workspace })
    expect(result(await owner.response(1)).ok).toBe(true)
    await foreign.response(2)

    owner.send(3, 'monitor.create', { source_kind: 'websocket', websocket_url: websocket.url, file_path: 'wrong.txt', trigger: 'output', match: 'error', duration_seconds: 60 })
    expect(result(await owner.response(3)).ok).toBe(false)
    owner.send(4, 'monitor.create', { source_kind: 'websocket', websocket_url: websocket.url, trigger: 'output', match: 'error', duration_seconds: 60 })
    const created = result(await owner.response(4))
    expect(created.ok).toBe(true)
    const monitor = created.monitor as Record<string, unknown>
    const id = String(monitor.id)
    await waitFor(() => sockets[0], socket => socket !== undefined)
    sockets[0]!.send('info')
    sockets[0]!.send('error')
    sockets[0]!.send('error')
    const inspected = await waitFor(async () => {
      owner.send(5, 'monitor.inspect', { monitor_id: id })
      return result(await owner.response(5)).monitor as Record<string, unknown>
    }, value => Array.isArray(value.events) && value.events.length === 1)
    expect(inspected.events).toEqual([expect.objectContaining({ text: expect.stringContaining('error') })])

    foreign.send(6, 'monitor.inspect', { monitor_id: id })
    expect((await foreign.response(6)).error).toBeDefined()
    foreign.send(7, 'monitor.stop', { monitor_id: id })
    expect((await foreign.response(7)).error).toBeDefined()
    owner.send(8, 'monitor.stop', { monitor_id: id })
    expect(result(await owner.response(8))).toMatchObject({ ok: true, monitor: { state: 'stopped', sourceStatus: 'Source closed (stopped).' } })
    await waitFor(() => closed, value => value === 1)
  } finally {
    owner.close(); foreign.close()
    monitors.close()
    await server.stop()
    websocket.server.stop(true)
    history.close()
    await rm(root, { recursive: true, force: true })
  }
})

test('websocket monitor RPC reports interrupted source after restart', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-websocket-monitor-restart-'))
  const workspace = join(root, 'workspace')
  await mkdir(workspace, { recursive: true })
  const websocket = localWebSocketServer(() => {}, () => {})
  const history = new RunHistory(join(root, 'runs.sqlite'))
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: workspace, sessionDirectory: join(root, 'sessions') })
  const terminals = new TerminalRegistry({ runHistory: history })
  const socketPath = join(root, 'daemon.sock')
  const restartSocketPath = join(root, 'daemon-restart.sock')
  let server: DaemonServer | undefined
  let monitors: TerminalMonitors | undefined
  let restartedServer: DaemonServer | undefined
  let restartedMonitors: TerminalMonitors | undefined
  let client: SocketClient | undefined
  let restartedClient: SocketClient | undefined
  try {
    ({ server, monitors } = await fixture(socketPath, workspace, runtime, history, terminals, nativeWebSocketMonitorSource))
    client = await SocketClient.connect(socketPath)
    client.send(1, 'initialize', { session_key: 'restart-owner', project_dir: workspace })
    await client.response(1)
    client.send(2, 'monitor.create', { source_kind: 'websocket', websocket_url: websocket.url, trigger: 'output', match: 'error', duration_seconds: 60 })
    const created = result(await client.response(2))
    const createdMonitor = created.monitor as Record<string, unknown>
    const id = String(createdMonitor.id)
    monitors.close();
    ({ server: restartedServer, monitors: restartedMonitors } = await fixture(restartSocketPath, workspace, runtime, history, terminals, nativeWebSocketMonitorSource))
    restartedClient = await SocketClient.connect(restartSocketPath)
    restartedClient.send(3, 'monitor.inspect', { session_key: 'restart-owner', monitor_id: id })
    const inspected = await restartedClient.response(3)
    expect(result(inspected)).toMatchObject({ ok: true, monitor: { state: 'interrupted', sourceStatus: expect.stringContaining('downtime') } })
  } finally {
    client?.close()
    restartedClient?.close()
    monitors?.close()
    restartedMonitors?.close()
    if (server) await server.stop()
    if (restartedServer) await restartedServer.stop()
    websocket.server.stop(true)
    history.close()
    await rm(root, { recursive: true, force: true })
  }
})
