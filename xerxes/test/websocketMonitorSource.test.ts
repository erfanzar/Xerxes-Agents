// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { nativeWebSocketMonitorSource, type WebSocketMonitorEvent } from '../src/runtime/websocketMonitorSource.js'

function waitFor<T>(read: () => T | undefined, timeoutMs = 2_000): Promise<T> {
  return new Promise((resolve, reject) => {
    let done = false
    const timer = setTimeout(() => { done = true; reject(new Error('timed out waiting for websocket event')) }, timeoutMs)
    const check = (): void => {
      if (done) return
      const value = read()
      if (value !== undefined) { done = true; clearTimeout(timer); resolve(value); return }
      setTimeout(check, 10)
    }
    check()
  })
}

function localServer(onOpen: (socket: Bun.ServerWebSocket<unknown>) => void, onMessage: () => void = () => undefined): { url: string; server: Bun.Server<unknown> } {
  const server = Bun.serve<unknown>({
    port: 0,
    fetch(_request, server) {
      if (server.upgrade(_request, { data: undefined })) return
      return new Response('websocket only', { status: 426 })
    },
    websocket: { open: onOpen, message: onMessage },
  })
  return { url: `ws://127.0.0.1:${server.port}/events`, server }
}

test('websocket monitor receives text frames with deterministic identities and no outgoing frames', async () => {
  const received: Bun.ServerWebSocket<unknown>[] = []
  let outgoing = 0
  const { url, server } = localServer(socket => received.push(socket), () => { outgoing += 1 })
  const events: WebSocketMonitorEvent[] = []
  const statuses: string[] = []
  const monitor = await nativeWebSocketMonitorSource.open(url, event => events.push(event), status => statuses.push(status), error => { throw error })
  try {
    await waitFor(() => received[0])
    received[0]?.send('hello')
    const first = await waitFor(() => events.find(event => event.kind === 'message'))
    received[0]?.send('hello')
    const second = await waitFor(() => events.filter(event => event.kind === 'message')[1])
    expect(first.identity).toBe(second.identity)
    expect(first.text).toBe('hello')
    expect(statuses).toContain('connected')
    expect(outgoing).toBe(0)
  } finally { monitor.close(); server.stop(true) }
})

test('websocket monitor emits a gap and reconnects after an established disconnect', async () => {
  const sockets: Bun.ServerWebSocket<unknown>[] = []
  const { url, server } = localServer(socket => sockets.push(socket))
  const events: WebSocketMonitorEvent[] = []
  const statuses: string[] = []
  const monitor = await nativeWebSocketMonitorSource.open(url, event => events.push(event), status => statuses.push(status), error => { throw error })
  try {
    await waitFor(() => sockets[0])
    sockets[0]?.close(1000, 'test disconnect')
    await waitFor(() => events.find(event => event.kind === 'gap'))
    await waitFor(() => sockets[1])
    sockets[1]?.send('after reconnect')
    await waitFor(() => events.find(event => event.text === 'after reconnect'))
    expect(statuses).toContain('reconnecting')
    expect(statuses.filter(status => status === 'connected').length).toBeGreaterThanOrEqual(2)
  } finally { monitor.close(); server.stop(true) }
})

test('websocket monitor rejects unsafe URLs and surfaces binary or oversized frames', async () => {
  await expect(nativeWebSocketMonitorSource.open('ws://example.com/events', () => undefined, () => undefined, () => undefined)).rejects.toThrow(/loopback/)
  await expect(nativeWebSocketMonitorSource.open('ws://127.0.0.1/events?token=secret', () => undefined, () => undefined, () => undefined)).rejects.toThrow(/query/)
  await expect(nativeWebSocketMonitorSource.open('ws://user:pass@127.0.0.1/events', () => undefined, () => undefined, () => undefined)).rejects.toThrow(/credentials/)

  const { url, server } = localServer(socket => {
    setTimeout(() => socket.send(new Uint8Array([1, 2, 3])), 20)
  })
  const errors: unknown[] = []
  const monitor = await nativeWebSocketMonitorSource.open(url, () => undefined, () => undefined, error => errors.push(error))
  try {
    await waitFor(() => errors[0])
    expect(String(errors[0])).toMatch(/binary/i)
  } finally { monitor.close(); server.stop(true) }

  const oversized = localServer(socket => {
    setTimeout(() => socket.send('x'.repeat(64 * 1024 + 1)), 20)
  })
  const oversizedErrors: unknown[] = []
  const oversizedMonitor = await nativeWebSocketMonitorSource.open(oversized.url, () => undefined, () => undefined, error => oversizedErrors.push(error))
  try {
    await waitFor(() => oversizedErrors[0])
    expect(String(oversizedErrors[0])).toMatch(/exceeds/)
  } finally { oversizedMonitor.close(); oversized.server.stop(true) }
})

test('websocket monitor rejects an already aborted signal', async () => {
  const controller = new AbortController()
  controller.abort(new Error('cancelled'))
  await expect(nativeWebSocketMonitorSource.open('ws://127.0.0.1/events', () => undefined, () => undefined, () => undefined, controller.signal)).rejects.toThrow('cancelled')
})

test('websocket monitor reports refused reconnect exhaustion and aborts after connection', async () => {
  const sockets: Bun.ServerWebSocket<unknown>[] = []
  const { url, server } = localServer(socket => sockets.push(socket))
  const errors: unknown[] = []
  const monitor = await nativeWebSocketMonitorSource.open(url, () => undefined, () => undefined, error => errors.push(error))
  await waitFor(() => sockets[0])
  server.stop(true)
  try {
    await waitFor(() => errors[0], 7_500)
    expect(String(errors[0])).toMatch(/exhausted 3 reconnect attempts/)
  } finally { monitor.close() }

  const second = localServer(() => undefined)
  const controller = new AbortController()
  const connected = await nativeWebSocketMonitorSource.open(second.url, () => undefined, () => undefined, error => { throw error }, controller.signal)
  controller.abort(new Error('after connection cancelled'))
  connected.close()
  second.server.stop(true)
}, { timeout: 10_000 })

test('websocket monitor survives more disconnects than its retry budget when every reconnect succeeds', async () => {
  const sockets: Bun.ServerWebSocket<unknown>[] = []
  const { url, server } = localServer(socket => sockets.push(socket))
  const events: WebSocketMonitorEvent[] = []
  const errors: unknown[] = []
  const monitor = await nativeWebSocketMonitorSource.open(url, event => events.push(event), () => undefined, error => errors.push(error))
  try {
    // Each drop reconnects on the first attempt; the budget bounds consecutive
    // failures, so a long watch outlives any number of healthy reconnects.
    for (let drop = 0; drop < 5; drop += 1) {
      await waitFor(() => sockets[drop])
      sockets[drop]?.close(1000, `drop ${drop}`)
      await waitFor(() => events.filter(event => event.kind === 'gap')[drop])
    }
    await waitFor(() => sockets[5])
    sockets[5]?.send('still watching')
    await waitFor(() => events.find(event => event.text === 'still watching'))
    expect(errors).toEqual([])
  } finally { monitor.close(); server.stop(true) }
})
