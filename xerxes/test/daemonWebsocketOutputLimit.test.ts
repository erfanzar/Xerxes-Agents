// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'

interface Frame {
  readonly error?: { readonly code: number; readonly message: string }
  readonly id?: number | string | null
  readonly method?: string
  readonly params?: {
    readonly payload?: Record<string, unknown>
    readonly type?: string
  }
  readonly result?: Record<string, unknown>
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

  static async connect(url: URL): Promise<WebSocketTestClient> {
    // Bun accepts an options bag here, while TypeScript selects the DOM
    // constructor overload because this project also includes `lib.dom`.
    const BunWebSocket = WebSocket as unknown as {
      new (endpoint: string | URL, options?: Bun.WebSocketOptions): WebSocket
    }
    const socket = new BunWebSocket(url)
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
    throw new Error('Timed out waiting for a matching WebSocket frame')
  }

  async waitForClose(): Promise<CloseEvent> {
    const deadline = Date.now() + 2_000
    while (!this.closeEvent) {
      if (Date.now() >= deadline) {
        throw new Error('Timed out waiting for the WebSocket to close')
      }
      await Bun.sleep(5)
    }
    return this.closeEvent
  }
}

test("oversized responses tell the requester why before closing, mirroring the Unix transport", async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-ws-output-limit-'))
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, 'sessions'),
  })
  const daemon = new DaemonServer({
    runtime,
    socketPath: join(directory, 'daemon.sock'),
    websocket: {
      host: '127.0.0.1',
      maxMessageBytes: 1024,
      port: 0,
    },
  })
  await daemon.start()
  const endpoint = daemon.websocketUrl
  if (!endpoint) {
    throw new Error('WebSocket gateway did not start')
  }
  const client = await WebSocketTestClient.connect(endpoint)
  try {
    // A response that fits the tiny cap still flows normally.
    client.send({ jsonrpc: '2.0', id: 1, method: 'runtime.status', params: {} })
    expect((await client.next(frame => frame.id === 1)).result).toMatchObject({ ok: true })

    // An initialize echoes a full session payload, far past maxMessageBytes:
    // the client must receive the same correlated -32000 failure the Unix
    // socket emits, then the 1009 close — not a bare drop. Any over-limit
    // daemon events along the way are skipped silently and must not close
    // the socket ahead of the correlated error.
    client.send({
      jsonrpc: '2.0',
      id: 2,
      method: 'initialize',
      params: { model: 'ws-limit-model', session_key: 'ws-flood' },
    })
    const failure = await client.next(frame => frame.id === 2 && frame.error !== undefined)
    expect(failure.error?.code).toBe(-32000)
    expect(failure.error?.message).toBe('response exceeds socket output limit')

    const closed = await client.waitForClose()
    expect(closed.code).toBe(1009)
    expect(closed.reason).toBe('outbound message exceeds limit')
  } finally {
    client.close()
    await daemon.stop()
    await rm(directory, { recursive: true, force: true })
  }
})
