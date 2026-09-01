// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  CODEX_SSE_COMPRESSION_HEADER,
  CodexWsApiError,
  CodexWsProtocolError,
  type CodexWebSocketOptions,
  buildCachedWebSocketRequestBody,
  clearCodexWebSocketFallback,
  closeCodexWebSocketSessions,
  codexWebSocketFallbackActive,
  compressRequestBodyZstd,
  continuationFromResponse,
  getCodexWebSocketDebugStats,
  isCodexRetryableWebSocketError,
  recordCodexWebSocketFallback,
  resolveCodexWebSocketUrl,
  streamCodexWebSocket,
  type WebSocketEventLike,
  type WebSocketLike,
} from '../src/streaming/codexWebSocket.js'

// ── Fake socket ─────────────────────────────────────────────────────────

class FakeWebSocket implements WebSocketLike {
  readonly url: string
  readonly headers: Record<string, string>
  readonly sent: string[] = []
  readonly closeCalls: Array<{ code: number | undefined; reason: string | undefined }> = []
  readyState = 0
  private readonly listeners = new Map<string, Array<(event: WebSocketEventLike) => void>>()

  constructor(
    url: string,
    headers: Record<string, string>,
    private readonly onSend?: (socket: FakeWebSocket, data: string) => void,
  ) {
    this.url = url
    this.headers = headers
  }

  addEventListener(
    type: 'close' | 'error' | 'message' | 'open',
    listener: (event: WebSocketEventLike) => void,
  ): void {
    const list = this.listeners.get(type) ?? []
    list.push(listener)
    this.listeners.set(type, list)
  }

  emit(type: 'close' | 'error' | 'message' | 'open', event: WebSocketEventLike = {}): void {
    for (const listener of [...(this.listeners.get(type) ?? [])]) listener({ type, ...event })
  }

  open(): void {
    if (this.readyState !== 0) return
    this.readyState = 1
    this.emit('open')
  }

  messageJson(value: Record<string, unknown>): void {
    this.emit('message', { data: JSON.stringify(value) })
  }

  terminate(code = 1008, reason = ''): void {
    if (this.readyState === 3) return
    this.readyState = 3
    this.emit('close', { code, reason })
  }

  send(data: string): void {
    this.sent.push(data)
    this.onSend?.(this, data)
  }

  close(code?: number, reason?: string): void {
    this.closeCalls.push({ code, reason })
    this.terminate(code ?? 1000, reason ?? '')
  }
}

interface Harness {
  readonly sockets: FakeWebSocket[]
  readonly factory: NonNullable<CodexWebSocketOptions['webSocketFactory']>
}

function harness(onSend?: (socket: FakeWebSocket, data: string) => void): Harness {
  const sockets: FakeWebSocket[] = []
  return {
    sockets,
    factory: (url, init) => {
      const socket = new FakeWebSocket(url, init.headers, onSend)
      sockets.push(socket)
      queueMicrotask(() => socket.open())
      return socket
    },
  }
}

function silentHarness(): Harness {
  return harness(() => undefined)
}

function failOpenHarness(): {
  sockets: FakeWebSocket[]
  factory: NonNullable<CodexWebSocketOptions['webSocketFactory']>
} {
  const sockets: FakeWebSocket[] = []
  return {
    sockets,
    factory: (url, init) => {
      const socket = new FakeWebSocket(url, init.headers)
      sockets.push(socket)
      return socket
    },
  }
}

async function collect(generator: AsyncGenerator<Record<string, unknown>>): Promise<Record<string, unknown>[]> {
  const events: Record<string, unknown>[] = []
  for await (const event of generator) events.push(event)
  return events
}

function baseOptions(
  overrides: Partial<CodexWebSocketOptions> = {},
): CodexWebSocketOptions {
  return {
    baseUrl: 'https://chatgpt.test/backend-api',
    headers: { Authorization: 'Bearer acct-a' },
    ...overrides,
  }
}

function simpleBody(): Record<string, unknown> {
  return {
    model: 'gpt-5.3-codex',
    stream: true,
    store: false,
    input: [{ type: 'message', role: 'user', content: [{ type: 'input_text', text: 'hi' }] }],
  }
}

function terminal(responseId: string): Record<string, unknown> {
  return { type: 'response.completed', response: { id: responseId } }
}

async function failureOf(pending: Promise<unknown>): Promise<unknown> {
  try {
    await pending
  } catch (error) {
    return error
  }
  throw new Error('expected the stream to reject')
}

// ── URL derivation ──────────────────────────────────────────────────────

test('resolveCodexWebSocketUrl derives the codex/responses socket endpoint', () => {
  expect(resolveCodexWebSocketUrl('https://chatgpt.com/backend-api')).toBe(
    'wss://chatgpt.com/backend-api/codex/responses',
  )
  expect(resolveCodexWebSocketUrl('https://chatgpt.com/backend-api/')).toBe(
    'wss://chatgpt.com/backend-api/codex/responses',
  )
  expect(resolveCodexWebSocketUrl('https://chatgpt.com/backend-api/codex')).toBe(
    'wss://chatgpt.com/backend-api/codex',
  )
  expect(resolveCodexWebSocketUrl('https://chatgpt.com/backend-api/codex/responses')).toBe(
    'wss://chatgpt.com/backend-api/codex/responses',
  )
  expect(resolveCodexWebSocketUrl('http://127.0.0.1:8080')).toBe('ws://127.0.0.1:8080/codex/responses')
})

test('the socket factory receives the derived URL and headers verbatim', async () => {
  const { sockets, factory } = harness(socket => socket.messageJson(terminal('resp_auto')))
  await collect(streamCodexWebSocket(simpleBody(), baseOptions({
    headers: { Authorization: 'Bearer tok', 'chatgpt-account-id': 'acct-9' },
    webSocketFactory: factory,
  })))

  expect(sockets).toHaveLength(1)
  expect(sockets[0]!.url).toBe(resolveCodexWebSocketUrl('https://chatgpt.test/backend-api'))
  expect(sockets[0]!.headers).toEqual({ Authorization: 'Bearer tok', 'chatgpt-account-id': 'acct-9' })
})

// ── Framing and streaming ───────────────────────────────────────────────

test('one uncompressed response.create text frame goes out per request', async () => {
  const { sockets, factory } = silentHarness()
  const pending = collect(streamCodexWebSocket(simpleBody(), baseOptions({ webSocketFactory: factory })))
  await Bun.sleep(5)
  const socket = sockets[0]!
  socket.messageJson(terminal('resp_1'))
  await pending

  expect(socket.sent).toHaveLength(1)
  expect(JSON.parse(socket.sent[0]!)).toEqual({ type: 'response.create', ...simpleBody() })
})

test('events stream parsed and in order until the terminal event', async () => {
  const { sockets, factory } = silentHarness()
  const pending = collect(streamCodexWebSocket(simpleBody(), baseOptions({ webSocketFactory: factory })))
  await Bun.sleep(5)
  const socket = sockets[0]!
  socket.messageJson({ type: 'response.output_text.delta', delta: 'he' })
  socket.messageJson({ type: 'response.output_text.delta', delta: 'y' })
  socket.messageJson(terminal('resp_1'))

  expect(await pending).toEqual([
    { type: 'response.output_text.delta', delta: 'he' },
    { type: 'response.output_text.delta', delta: 'y' },
    terminal('resp_1'),
  ])
})

test('response.done and response.incomplete are yielded as-is and end the stream', async () => {
  for (const terminalType of ['response.done', 'response.incomplete']) {
    const { sockets, factory } = silentHarness()
    const pending = collect(streamCodexWebSocket(simpleBody(), baseOptions({ webSocketFactory: factory })))
    await Bun.sleep(5)
    sockets[0]!.messageJson({ type: terminalType, response: { id: 'r' } })
    expect(await pending).toEqual([{ type: terminalType, response: { id: 'r' } }])
  }
})

// ── Typed failures ──────────────────────────────────────────────────────

test("an error event raises CodexWsApiError carrying the event's code", async () => {
  const { sockets, factory } = silentHarness()
  const pending = collect(streamCodexWebSocket(simpleBody(), baseOptions({ webSocketFactory: factory })))
  await Bun.sleep(5)
  sockets[0]!.messageJson({
    type: 'error',
    code: 'websocket_connection_limit_reached',
    message: 'connection limit reached',
  })

  const failure = await failureOf(pending)
  expect(failure).toBeInstanceOf(CodexWsApiError)
  expect((failure as CodexWsApiError).code).toBe('websocket_connection_limit_reached')
  expect((failure as Error).message).toContain('connection limit reached')
  expect(isCodexRetryableWebSocketError(failure)).toBe(true)
})

test('response.failed raises CodexWsApiError with the nested error code', async () => {
  const { sockets, factory } = silentHarness()
  const pending = collect(streamCodexWebSocket(simpleBody(), baseOptions({ webSocketFactory: factory })))
  await Bun.sleep(5)
  sockets[0]!.messageJson({ type: 'response.failed', error: { code: 'server_error', message: 'boom' } })

  const failure = await failureOf(pending)
  expect((failure as CodexWsApiError).code).toBe('server_error')
  expect(isCodexRetryableWebSocketError(failure)).toBe(false)
})

test('a close code of 1009 classifies the failure as retryable', async () => {
  const { sockets, factory } = silentHarness()
  const pending = collect(streamCodexWebSocket(simpleBody(), baseOptions({ webSocketFactory: factory })))
  await Bun.sleep(5)
  sockets[0]!.terminate(1009, 'frame too large')

  const failure = await failureOf(pending)
  expect(failure).toBeInstanceOf(CodexWsApiError)
  expect((failure as CodexWsApiError).closeCode).toBe(1009)
  expect(isCodexRetryableWebSocketError(failure)).toBe(true)
})

test('malformed JSON frames raise the protocol error', async () => {
  const { sockets, factory } = silentHarness()
  const pending = collect(streamCodexWebSocket(simpleBody(), baseOptions({ webSocketFactory: factory })))
  await Bun.sleep(5)
  sockets[0]!.emit('message', { data: 'not json {' })

  const failure = await failureOf(pending)
  expect(failure).toBeInstanceOf(CodexWsProtocolError)
  expect((failure as Error).message).toContain('malformed JSON')
})

// ── Timeouts and abort ──────────────────────────────────────────────────

test('connect timeout closes the socket and reports connect_timeout', async () => {
  const { sockets, factory } = failOpenHarness()
  const pending = collect(streamCodexWebSocket(simpleBody(), baseOptions({
    connectTimeoutMs: 25,
    webSocketFactory: factory,
  })))

  await expect(pending).rejects.toThrow('connect_timeout')
  expect(sockets[0]!.closeCalls).toContainEqual({ code: 1000, reason: 'connect_timeout' })
})

test('premature close mid-stream reports the close code', async () => {
  const { sockets, factory } = silentHarness()
  const pending = collect(streamCodexWebSocket(simpleBody(), baseOptions({ webSocketFactory: factory })))
  await Bun.sleep(5)
  sockets[0]!.terminate(1011, 'server gone')

  const failure = await failureOf(pending)
  expect((failure as CodexWsApiError).closeCode).toBe(1011)
  expect((failure as Error).message).toContain('before a terminal event')
})

test('an idle gap closes the socket and reports idle_timeout', async () => {
  const { sockets, factory } = silentHarness()
  const pending = collect(streamCodexWebSocket(simpleBody(), baseOptions({
    idleTimeoutMs: 30,
    webSocketFactory: factory,
  })))
  await Bun.sleep(5)

  await expect(pending).rejects.toThrow('idle_timeout')
  expect(sockets[0]!.closeCalls).toContainEqual({ code: 1000, reason: 'idle_timeout' })
})

test('an abort closes the socket and throws the plain aborted error', async () => {
  const controller = new AbortController()
  const { sockets, factory } = silentHarness()
  const pending = collect(streamCodexWebSocket(simpleBody(), baseOptions({
    signal: controller.signal,
    webSocketFactory: factory,
  })))
  await Bun.sleep(5)
  controller.abort()

  const failure = await failureOf(pending)
  expect(failure).toBeInstanceOf(Error)
  expect((failure as Error).message).toBe('aborted')
  expect(sockets[0]!.closeCalls).toContainEqual({ code: 1000, reason: 'aborted' })
})

test("transport 'sse' is refused by the WebSocket stream", async () => {
  await expect(collect(streamCodexWebSocket(simpleBody(), baseOptions({ transport: 'sse' }))))
    .rejects.toThrow('sse')
})

// ── Session pooling ─────────────────────────────────────────────────────

test('pooled sockets are reused per session and account across turns', async () => {
  const sessionId = 'ws-reuse-test'
  const { sockets, factory } = harness(socket => socket.messageJson(terminal('resp_auto')))
  const options = baseOptions({ sessionId, webSocketFactory: factory })
  const before = getCodexWebSocketDebugStats()

  await collect(streamCodexWebSocket(simpleBody(), options))
  await collect(streamCodexWebSocket(simpleBody(), options))

  const after = getCodexWebSocketDebugStats()
  expect(after.connectionsCreated - before.connectionsCreated).toBe(1)
  expect(after.connectionsReused - before.connectionsReused).toBe(1)
  expect(sockets).toHaveLength(1)
  expect(sockets[0]!.sent).toHaveLength(2)

  // A different account on the same session gets its own socket.
  await collect(streamCodexWebSocket(simpleBody(), baseOptions({
    sessionId,
    headers: { Authorization: 'Bearer acct-b' },
    webSocketFactory: factory,
  })))
  expect(sockets).toHaveLength(2)
  closeCodexWebSocketSessions(sessionId)
})

test('a busy session gets a throwaway socket that is never cached', async () => {
  const sessionId = 'ws-busy-test'
  const { sockets, factory } = silentHarness()
  const options = baseOptions({ sessionId, webSocketFactory: factory })

  // Turn A connects and stays mid-stream: consumed one delta, waiting for more.
  const pendingA = collect(streamCodexWebSocket(simpleBody(), options))
  await Bun.sleep(5)
  sockets[0]!.messageJson({ type: 'response.output_text.delta', delta: 'partial' })
  await Bun.sleep(5)

  // Turn B cannot reuse the busy socket: fresh throwaway connection.
  const pendingB = collect(streamCodexWebSocket(simpleBody(), options))
  await Bun.sleep(5)
  expect(sockets).toHaveLength(2)
  sockets[1]!.messageJson(terminal('resp_b'))
  await pendingB
  expect(sockets[1]!.closeCalls.some(call => call.reason === 'done')).toBe(true)

  // Finish turn A; its socket returns to the pool.
  sockets[0]!.messageJson(terminal('resp_a'))
  await pendingA

  // Turn C reuses A's socket — the throwaway never displaced it.
  const pendingC = collect(streamCodexWebSocket(simpleBody(), options))
  await Bun.sleep(5)
  expect(sockets).toHaveLength(2)
  expect(sockets[0]!.sent).toHaveLength(2)
  sockets[0]!.messageJson(terminal('resp_c'))
  await pendingC
  closeCodexWebSocketSessions(sessionId)
})

test('pooled sockets are evicted once they exceed the max age', async () => {
  const sessionId = 'ws-evict-test'
  const { sockets, factory } = harness(socket => socket.messageJson(terminal('resp_auto')))
  const options = baseOptions({ sessionId, poolMaxAgeMs: 5, webSocketFactory: factory })

  await collect(streamCodexWebSocket(simpleBody(), options))
  await Bun.sleep(15)
  await collect(streamCodexWebSocket(simpleBody(), options))

  expect(sockets).toHaveLength(2)
  expect(sockets[0]!.closeCalls.some(call => call.reason === 'evicted')).toBe(true)
  closeCodexWebSocketSessions(sessionId)
})

test('an idle pooled socket closes itself after the idle window', async () => {
  const sessionId = 'ws-idle-test'
  const { sockets, factory } = harness(socket => socket.messageJson(terminal('resp_auto')))
  const options = baseOptions({ sessionId, poolIdleTimeoutMs: 20, webSocketFactory: factory })

  await collect(streamCodexWebSocket(simpleBody(), options))
  expect(sockets[0]!.closeCalls.some(call => call.reason === 'idle_timeout')).toBe(false)
  await Bun.sleep(80)
  expect(sockets[0]!.closeCalls.some(call => call.reason === 'idle_timeout')).toBe(true)
  closeCodexWebSocketSessions(sessionId)
})

test('closeCodexWebSocketSessions closes pooled sockets for one session or all', async () => {
  const { sockets, factory } = harness(socket => socket.messageJson(terminal('resp_auto')))
  await collect(streamCodexWebSocket(simpleBody(), baseOptions({ sessionId: 'ws-close-a', webSocketFactory: factory })))
  await collect(streamCodexWebSocket(simpleBody(), baseOptions({ sessionId: 'ws-close-b', webSocketFactory: factory })))
  expect(sockets.every(socket => socket.readyState === 1)).toBe(true)

  closeCodexWebSocketSessions('ws-close-a')
  expect(sockets[0]!.closeCalls.some(call => call.reason === 'session_closed')).toBe(true)
  expect(sockets[1]!.readyState).toBe(1)

  closeCodexWebSocketSessions()
  expect(sockets[1]!.closeCalls.some(call => call.reason === 'session_closed')).toBe(true)
})

// ── Sticky SSE fallback ─────────────────────────────────────────────────

test('the SSE fallback flag is sticky per session until cleared', () => {
  clearCodexWebSocketFallback()
  recordCodexWebSocketFallback('sess-x')
  expect(codexWebSocketFallbackActive('sess-x')).toBe(true)
  expect(codexWebSocketFallbackActive('sess-y')).toBe(false)

  clearCodexWebSocketFallback('sess-x')
  expect(codexWebSocketFallbackActive('sess-x')).toBe(false)

  recordCodexWebSocketFallback('sess-y')
  clearCodexWebSocketFallback()
  expect(codexWebSocketFallbackActive('sess-y')).toBe(false)
})

// ── Delta continuation ──────────────────────────────────────────────────

test('continuationFromResponse rejects empty response ids', () => {
  expect(continuationFromResponse(simpleBody(), '  ', [])).toBeUndefined()
})

test('a strict prefix-extension becomes a delta; anything else stays full', () => {
  const body = simpleBody()
  const assistantItems = [{
    type: 'message',
    role: 'assistant',
    content: [{ type: 'output_text', text: 'hello' }],
  }]
  const continuation = continuationFromResponse(body, 'resp_1', assistantItems)
  expect(continuation).toBeDefined()

  const nextUser = { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'more' }] }
  const bodyInput = body.input as unknown[]
  // Next-turn input = last request input ++ assistant items ++ one new item.
  const extended = { ...body, input: [...bodyInput, ...assistantItems, nextUser] }
  const delta = buildCachedWebSocketRequestBody(extended, continuation!)
  expect(delta.usedDelta).toBe(true)
  expect(delta.body.previous_response_id).toBe('resp_1')
  expect(delta.body.input).toEqual([nextUser])

  // Same length as the remembered prefix: not a strict extension.
  const prefixOnly = { ...body, input: [...bodyInput, ...assistantItems] }
  expect(buildCachedWebSocketRequestBody(prefixOnly, continuation!).usedDelta).toBe(false)

  // Unrelated request shape: full body.
  expect(buildCachedWebSocketRequestBody({ ...extended, model: 'other-model' }, continuation!)).toEqual({
    body: { ...extended, model: 'other-model' },
    usedDelta: false,
  })

  // Diverging prefix (replay fork): full body.
  const diverged = {
    ...body,
    input: [{ type: 'message', role: 'user', content: 'rewritten' }, ...assistantItems, nextUser],
  }
  expect(buildCachedWebSocketRequestBody(diverged, continuation!).usedDelta).toBe(false)
})

// ── zstd compression ────────────────────────────────────────────────────

test('zstd compression round-trips and ships the SSE content-encoding header', () => {
  const payload = JSON.stringify({ model: 'gpt-5.3-codex', input: 'x'.repeat(2_048) })
  const compressed = compressRequestBodyZstd(payload)

  expect(compressed).toBeInstanceOf(Uint8Array)
  expect(compressed!.length).toBeLessThan(Buffer.byteLength(payload))
  expect(Bun.zstdDecompressSync(compressed!).toString()).toBe(payload)
  expect(CODEX_SSE_COMPRESSION_HEADER).toEqual({ 'content-encoding': 'zstd' })
})
