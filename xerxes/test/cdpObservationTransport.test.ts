// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { BunCdpConnectionFactory, type CdpConnection } from '../src/operators/cdpBrowser.js'

/** Owned loopback protocol fixture, never an actual browser or external endpoint. */
function fixture(frame?: Readonly<Record<string, unknown>>, reply = false, binary = false) {
  let opens = 0
  const server = Bun.serve({ hostname: '127.0.0.1', port: 0,
    fetch(request, server) { return server.upgrade(request) ? undefined : new Response(null, { status: 400 }) },
    websocket: { open(socket) { opens += 1; if (frame) socket.send(binary ? Buffer.from(JSON.stringify(frame)) : JSON.stringify(frame)) },
      message(socket, message) { if (reply) { const request: unknown = JSON.parse(String(message)); if (typeof request === 'object' && request !== null && 'id' in request) socket.send(JSON.stringify({ id: request.id, result: { ok: true } })) } } },
  })
  return { endpoint: `ws://127.0.0.1:${server.port}/fixture`, opens: () => opens, stop: () => server.stop(true) }
}

async function promptly(work: Promise<unknown>) {
  return Promise.race([work.then(() => 'resolved', () => 'refused'), Bun.sleep(100).then(() => 'hung')])
}

test('a pre-aborted CDP attachment never opens a socket', async () => {
  const rig = fixture(); const controller = new AbortController(); controller.abort()
  let connection: CdpConnection | undefined
  try {
    const work = new BunCdpConnectionFactory().connect(rig.endpoint, { signal: controller.signal }).then(value => { connection = value; return value })
    expect(await promptly(work)).toBe('refused')
    expect(rig.opens()).toBe(0)
  } finally { connection?.close(); await rig.stop() }
})

test('caller abort also cancels an attachment whose WebSocket upgrade never completes', async () => {
  let begin = (): void => {}; const started = new Promise<void>(resolve => { begin = resolve })
  let finish = (): void => {}; const held = new Promise<void>(resolve => { finish = resolve })
  const server = Bun.serve({ hostname: '127.0.0.1', port: 0, async fetch() { begin(); await held; return new Response(null, { status: 503 }) } })
  const controller = new AbortController()
  const work = new BunCdpConnectionFactory().connect(`ws://127.0.0.1:${server.port}/fixture`, { signal: controller.signal })
  try { await started; controller.abort(); expect(await promptly(work)).toBe('refused') }
  finally { finish(); await server.stop(true); await work.catch(() => {}) }
})

test('caller abort closes the CDP lane and rejects pending and future commands promptly', async () => {
  const rig = fixture(); const controller = new AbortController()
  const connection = await new BunCdpConnectionFactory().connect(rig.endpoint, { signal: controller.signal })
  const work = connection.command('Page.enable')
  try {
    controller.abort()
    expect(await promptly(work)).toBe('refused')
    await expect(connection.command('Page.captureScreenshot')).rejects.toThrow()
  } finally { connection.close(); await work.catch(() => {}); await rig.stop() }
})

test('bounded CDP lanes deliver page lifecycle events without treating them as command responses', async () => {
  const event = { method: 'Page.frameNavigated', params: { frame: { id: 'frame', url: 'https://xerxes.invalid/capture' } }, sessionId: 'capture-session' }
  const rig = fixture(event)
  let delivered = (): void => {}; const ready = new Promise<void>(resolve => { delivered = resolve })
  const events: unknown[] = []
  const connection = await new BunCdpConnectionFactory().connect(rig.endpoint, { maxMessageBytes: 1024, onEvent: frame => { events.push(frame); delivered() } })
  try {
    expect(await promptly(ready)).toBe('resolved')
    expect(events).toEqual([event])
  } finally { connection.close(); await rig.stop() }
})

test('a lifecycle guard failure fences later command responses on text and binary lanes', async () => {
  for (const binary of [false, true]) {
    const rig = fixture({ method: 'Page.frameNavigated', params: { frame: { url: 'https://other.invalid/' } } }, true, binary)
    let events = 0
    const connection = await new BunCdpConnectionFactory().connect(rig.endpoint, { maxMessageBytes: 1024,
      onEvent: () => { events += 1; throw new Error('capture identity changed') } })
    const work = connection.command('Page.captureScreenshot')
    try { expect(await promptly(work)).toBe('refused'); expect(events).toBe(1) }
    finally { connection.close(); await work.catch(() => {}); await rig.stop() }
  }
})

test('a delayed binary lifecycle notification cannot be overtaken by a text capture response', async () => {
  const rig = fixture({ method: 'Page.frameNavigated', params: { frame: { url: 'https://other.invalid/' } } }, true, true)
  let finish = (): void => {}; const held = new Promise<void>(resolve => { finish = resolve }); let constructed = 0
  const factory = new BunCdpConnectionFactory((endpoint: string) => {
    constructed += 1
    const socket = new WebSocket(endpoint); socket.binaryType = 'blob'
    socket.addEventListener('message', event => {
      if (event.data instanceof Blob) {
        const original = event.data.text.bind(event.data)
        event.data.text = async () => { await held; return original() }
      }
    })
    return socket
  })
  const connection = await factory.connect(rig.endpoint, { maxMessageBytes: 1024, onEvent: () => { throw new Error('capture identity changed') } })
  const work = connection.command('Page.captureScreenshot')
  try {
    expect(constructed).toBe(1)
    expect(await promptly(work)).not.toBe('resolved')
    finish()
    expect(await promptly(work)).toBe('refused')
  } finally { finish(); connection.close(); await work.catch(() => {}); await rig.stop() }
})

test('a closed CDP lane discards already delivered frames before reading or decoding their data', async () => {
  const rig = fixture(); let socket: WebSocket | undefined; let reads = 0
  const factory = new BunCdpConnectionFactory((endpoint: string) => { socket = new WebSocket(endpoint); return socket })
  const connection = await factory.connect(rig.endpoint, { maxMessageBytes: 1024 })
  try {
    connection.close()
    const frame = new MessageEvent('message')
    Object.defineProperty(frame, 'data', { get() { reads += 1; return new Blob(['unused']) } })
    socket?.dispatchEvent(frame)
    expect(socket).toBeDefined()
    expect(reads).toBe(0)
  } finally { connection.close(); await rig.stop() }
})

test('an invalid lifecycle payload closes a guarded lane instead of hiding an identity change', async () => {
  const rig = fixture({ method: 'Page.frameNavigated', params: null }, true)
  let events = 0
  const connection = await new BunCdpConnectionFactory().connect(rig.endpoint, { onEvent: () => { events += 1 } })
  const work = connection.command('Page.captureScreenshot')
  try { expect(await promptly(work)).toBe('refused'); expect(events).toBe(0) }
  finally { connection.close(); await work.catch(() => {}); await rig.stop() }
})

test.each([
  { method: 'Page.frameNavigated', params: {}, sessionId: 42 },
  { method: '', params: {}, sessionId: 'capture-session' },
  { method: 42, params: {} },
  { id: 1, method: 'Page.frameNavigated', params: {} },
])('malformed lifecycle routing closes the lane instead of silently dropping identity fields: %j', async frame => {
  const rig = fixture(frame, true); let events = 0
  const connection = await new BunCdpConnectionFactory().connect(rig.endpoint, { onEvent: () => { events += 1 } })
  const work = connection.command('Page.captureScreenshot')
  try { expect(await promptly(work)).toBe('refused'); expect(events).toBe(0) }
  finally { connection.close(); await work.catch(() => {}); await rig.stop() }
})

test('invalid byte controls refuse before opening any CDP socket', async () => {
  const rig = fixture()
  try {
    for (const maxMessageBytes of [0, -1, NaN, 1.5, 64 * 1024 * 1024 + 1]) {
      await expect(new BunCdpConnectionFactory().connect(rig.endpoint, { maxMessageBytes })).rejects.toThrow('cdp_max_message_bytes')
    }
    expect(rig.opens()).toBe(0)
  } finally { await rig.stop() }
})

test('an oversized CDP frame closes a bounded lane rather than parsing or forwarding its payload', async () => {
  const rig = fixture({ method: 'Page.frameNavigated', params: { text: 'x'.repeat(2048) } })
  let events = 0
  const connection = await new BunCdpConnectionFactory().connect(rig.endpoint, { maxMessageBytes: 1024, onEvent: () => { events += 1 } })
  const work = connection.command('Page.enable')
  try { expect(await promptly(work)).toBe('refused'); expect(events).toBe(0) }
  finally { connection.close(); await work.catch(() => {}); await rig.stop() }
})
