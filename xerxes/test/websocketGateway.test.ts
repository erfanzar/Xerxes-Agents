// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { connect } from 'node:net'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonInteractionBoard } from '../src/daemon/interactions.js'
import { DaemonServer } from '../src/daemon/server.js'
import { websocketOriginAllowed, websocketRequestAuthorized } from '../src/daemon/websocketGateway.js'
import type { DaemonEvent, DaemonSession, TurnRunner } from '../src/daemon/runtime.js'
import type { PermissionRequest } from '../src/streaming/events.js'

test('WebSocket authentication accepts bearer or query tokens without accepting mismatches', () => {
  const expected = 'correct horse battery staple'
  expect(websocketRequestAuthorized(new Request('http://localhost/rpc'), expected)).toBe(false)
  expect(websocketRequestAuthorized(new Request('http://localhost/rpc', {
    headers: { Authorization: `Bearer ${expected}` },
  }), expected)).toBe(true)
  expect(websocketRequestAuthorized(new Request(`http://localhost/rpc?token=${encodeURIComponent(expected)}`), expected)).toBe(true)
  expect(websocketRequestAuthorized(new Request(`http://localhost/rpc?token=${encodeURIComponent(expected)}`, {
    headers: { Authorization: 'Bearer wrong-token' },
  }), expected)).toBe(true)
  expect(websocketRequestAuthorized(new Request('http://localhost/rpc?token=wrong-token'), expected)).toBe(false)
  expect(websocketRequestAuthorized(new Request('http://localhost/rpc'), undefined)).toBe(true)
})

test('daemon WebSocket transport dispatches v35 JSON-RPC and broadcasts events', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-websocket-'))
  const daemon = new DaemonServer({
    socketPath: join(directory, 'daemon.sock'),
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, 'sessions'),
    }),
    websocket: {
      authToken: 'socket-token',
      host: '127.0.0.1',
      path: '/rpc',
      port: 0,
    },
  })
  await daemon.start()
  const endpoint = daemon.websocketUrl
  if (!endpoint) {
    throw new Error('WebSocket gateway did not start')
  }
  endpoint.searchParams.set('token', 'socket-token')
  const client = await WebSocketTestClient.connect(endpoint)
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'runtime.status', params: {} })
    expect((await client.next(frame => frame.id === 1)).result).toMatchObject({
      ok: true,
      runtime_ready: false,
      daemon_protocol: 35,
      runtime: 'bun-typescript',
    })

    client.send({
      jsonrpc: '2.0',
      id: 2,
      method: 'initialize',
      params: { model: 'websocket-model', session_key: 'remote-session' },
    })
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({
      ok: true,
      session: { key: 'remote-session', status: 'idle' },
    })
    expect((await client.next(eventFrame('init_done'))).params?.payload).toMatchObject({
      session_id: expect.any(String),
      mode: 'code',
    })

    daemon.broadcast('gateway_notice', { scope: 'remote' })
    expect((await client.next(eventFrame('gateway_notice'))).params?.payload).toEqual({ scope: 'remote' })

    client.send({ jsonrpc: '2.0', id: 3, method: 'turn.submit', params: { text: 'hello from WebSocket' } })
    expect((await client.next(frame => frame.id === 3)).result).toEqual({ ok: true })
    expect((await client.next(eventFrame('turn_begin'))).params?.payload).toMatchObject({ text: 'hello from WebSocket' })
    expect((await client.next(eventFrame('turn_end'))).params?.payload).toMatchObject({ cancelled: false })
  } finally {
    client.close()
    await daemon.stop()
    await rm(directory, { recursive: true, force: true })
  }
})

test('closing a WebSocket client cancels its pending interaction turn', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-websocket-disconnect-'))
  const interactions = new DaemonInteractionBoard()
  const runtime = new InMemoryDaemonRuntime(new WebSocketApprovalRunner(interactions), {
    currentProjectDirectory: directory,
    interactions,
    sessionDirectory: join(directory, 'sessions'),
  })
  const daemon = new DaemonServer({
    interactions,
    runtime,
    socketPath: join(directory, 'daemon.sock'),
    websocket: { host: '127.0.0.1', port: 0 },
  })
  await daemon.start()
  const endpoint = daemon.websocketUrl
  if (!endpoint) {
    throw new Error('WebSocket gateway did not start')
  }
  const client = await WebSocketTestClient.connect(endpoint)
  try {
    client.send({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: { model: 'websocket-approval-model', session_key: 'remote-wait' },
    })
    await client.next(frame => frame.id === 1)
    await client.next(eventFrame('init_done'))
    await client.next(eventFrame('status_update'))
    client.send({ jsonrpc: '2.0', id: 2, method: 'turn.submit', params: { text: 'wait for approval' } })
    await client.next(frame => frame.id === 2)
    await client.next(eventFrame('turn_begin'))
    await client.next(eventFrame('approval_request'))
    expect(interactions.pendingPermissionIds()).toEqual(['websocket-approval'])

    client.close()
    await client.waitForClose()
    await waitFor(() =>
      interactions.pendingPermissionIds().length === 0 &&
      runtime.sessionStatus('remote-wait')?.activeTurnId === '',
    )
    expect(runtime.sessionStatus('remote-wait')?.cancelRequested).toBe(true)
  } finally {
    client.close()
    await daemon.stop()
    await rm(directory, { recursive: true, force: true })
  }
})

test('daemon keeps its Unix v35 service alive when an optional WebSocket port is unavailable', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-websocket-fallback-'))
  const blocker = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: () => new Response('occupied') })
  const blockedPort = blocker.port
  if (blockedPort === undefined) {
    throw new Error('Bun did not report the occupied WebSocket port')
  }
  const daemon = new DaemonServer({
    socketPath: join(directory, 'daemon.sock'),
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, 'sessions'),
    }),
    websocket: { host: '127.0.0.1', port: blockedPort },
  })
  await daemon.start()
  try {
    expect(daemon.websocketUrl).toBeUndefined()
    const socket = connect({ path: join(directory, 'daemon.sock') })
    await new Promise<void>((resolve, reject) => {
      socket.once('connect', resolve)
      socket.once('error', reject)
    })
    socket.destroy()
  } finally {
    await daemon.stop()
    await blocker.stop(true)
    await rm(directory, { recursive: true, force: true })
  }
})

test('daemon WebSocket transport closes oversized requests', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-websocket-limit-'))
  const daemon = new DaemonServer({
    socketPath: join(directory, 'daemon.sock'),
    websocket: { host: '127.0.0.1', maxMessageBytes: 1024, port: 0 },
  })
  await daemon.start()
  const endpoint = daemon.websocketUrl
  if (!endpoint) {
    throw new Error('WebSocket gateway did not start')
  }
  const client = await WebSocketTestClient.connect(endpoint)
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'runtime.status', params: { padding: 'x'.repeat(2_000) } })
    expect((await client.waitForClose()).code).toBe(1009)
  } finally {
    client.close()
    await daemon.stop()
    await rm(directory, { recursive: true, force: true })
  }
})

test('WebSocket origin policy accepts loopback and missing origins while rejecting foreign pages', () => {
  const at = (origin?: string) => new Request('http://localhost/rpc', {
    headers: origin === undefined ? {} : { Origin: origin },
  })
  expect(websocketOriginAllowed(at())).toBe(true)
  expect(websocketOriginAllowed(at('http://localhost:3000'))).toBe(true)
  expect(websocketOriginAllowed(at('https://localhost'))).toBe(true)
  expect(websocketOriginAllowed(at('http://127.0.0.1:5173'))).toBe(true)
  expect(websocketOriginAllowed(at('http://[::1]:8080'))).toBe(true)
  expect(websocketOriginAllowed(at('https://evil.example'))).toBe(false)
  expect(websocketOriginAllowed(at('http://localhost.evil.example'))).toBe(false)
  expect(websocketOriginAllowed(at('not a url'))).toBe(false)
})

test('daemon WebSocket transport rejects cross-origin browser upgrades with 403', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-websocket-origin-'))
  const daemon = new DaemonServer({
    socketPath: join(directory, 'daemon.sock'),
    websocket: { host: '127.0.0.1', path: '/rpc', port: 0 },
  })
  await daemon.start()
  const endpoint = daemon.websocketUrl
  if (!endpoint) {
    throw new Error('WebSocket gateway did not start')
  }
  try {
    const httpEndpoint = new URL(endpoint)
    httpEndpoint.protocol = 'http:'
    const rejected = await fetch(httpEndpoint, {
      headers: {
        Connection: 'upgrade',
        Origin: 'https://evil.example',
        'Sec-WebSocket-Key': 'dGhlIHNhbXBsZSBub25jZQ==',
        'Sec-WebSocket-Version': '13',
        Upgrade: 'websocket',
      },
    })
    expect(rejected.status).toBe(403)

    const client = await WebSocketTestClient.connect(endpoint, { Origin: 'http://localhost:3000' })
    try {
      client.send({ jsonrpc: '2.0', id: 1, method: 'runtime.status', params: {} })
      expect((await client.next(frame => frame.id === 1)).result).toMatchObject({ ok: true })
    } finally {
      client.close()
    }
  } finally {
    await daemon.stop()
    await rm(directory, { recursive: true, force: true })
  }
})

interface Frame {
  readonly id?: number | string | null
  readonly method?: string
  readonly params?: {
    readonly payload?: Record<string, unknown>
    readonly type?: string
  }
  readonly result?: Record<string, unknown>
}

class WebSocketApprovalRunner implements TurnRunner {
  constructor(private readonly interactions: DaemonInteractionBoard) {}

  async *run(
    session: DaemonSession,
    _text: string,
    signal: AbortSignal,
  ): AsyncGenerator<DaemonEvent> {
    const request: PermissionRequest = {
      requestId: 'websocket-approval',
      description: 'Wait for a WebSocket approval.',
      inputs: {},
      toolCall: {
        id: 'websocket-tool',
        type: 'function',
        function: { name: 'WriteFile', arguments: {} },
      },
    }
    yield {
      type: 'approval_request',
      payload: { id: request.requestId, request_id: request.requestId },
    }
    const decision = await this.interactions
      .permissionBroker(session.id)
      .request(request, signal)
    yield { type: 'text_part', payload: { text: `approval:${decision}` } }
  }
}

function eventFrame(type: string): (frame: Frame) => boolean {
  return frame => frame.method === 'event' && frame.params?.type === type
}

async function waitFor(predicate: () => boolean, timeout = 2_000): Promise<void> {
  const deadline = Date.now() + timeout
  while (!predicate()) {
    if (Date.now() >= deadline) {
      throw new Error('Timed out waiting for WebSocket daemon state')
    }
    await Bun.sleep(5)
  }
}

class WebSocketTestClient {
  private closeEvent: CloseEvent | undefined
  private readonly frames: Frame[] = []
  private constructor(private readonly socket: WebSocket) {
    socket.addEventListener('message', event => {
      if (typeof event.data === 'string') {
        this.frames.push(JSON.parse(event.data) as Frame)
      }
    })
    socket.addEventListener('close', event => {
      this.closeEvent = event
    })
  }

  static async connect(url: URL, headers?: Record<string, string>): Promise<WebSocketTestClient> {
    // Bun accepts an options bag here, while TypeScript selects the DOM
    // constructor overload because this project also includes `lib.dom`.
    const BunWebSocket = WebSocket as unknown as {
      new (endpoint: string | URL, options?: Bun.WebSocketOptions): WebSocket
    }
    const socket = new BunWebSocket(url, headers ? { headers } : undefined)
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('WebSocket connection timed out')), 2_000)
      socket.addEventListener('open', () => {
        clearTimeout(timeout)
        resolve()
      }, { once: true })
      socket.addEventListener('error', () => {
        clearTimeout(timeout)
        reject(new Error('WebSocket connection failed'))
      }, { once: true })
    })
    return new WebSocketTestClient(socket)
  }

  close(): void {
    if (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING) {
      this.socket.close()
    }
  }

  send(frame: object): void {
    this.socket.send(JSON.stringify(frame))
  }

  async next(predicate: (frame: Frame) => boolean): Promise<Frame> {
    const deadline = Date.now() + 2_000
    while (Date.now() < deadline) {
      const index = this.frames.findIndex(predicate)
      if (index >= 0) {
        const frame = this.frames.splice(index, 1)[0]
        if (frame) {
          return frame
        }
      }
      await Bun.sleep(5)
    }
    throw new Error('Timed out waiting for WebSocket frame')
  }

  async waitForClose(): Promise<CloseEvent> {
    const deadline = Date.now() + 2_000
    while (Date.now() < deadline) {
      if (this.closeEvent) {
        return this.closeEvent
      }
      await Bun.sleep(5)
    }
    throw new Error('Timed out waiting for WebSocket close')
  }
}
