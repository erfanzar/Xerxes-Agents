// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, expect, test } from 'bun:test'

import { StreamTruncatedError } from '../src/core/errors.js'
import { ResponsesApiClient, type CompletionRequest, type LlmDelta } from '../src/llms/client.js'
import { classifyError } from '../src/runtime/errorClassifier.js'
import {
  clearCodexWebSocketFallback,
  closeCodexWebSocketSessions,
  codexWebSocketFallbackActive,
  type WebSocketEventLike,
} from '../src/streaming/codexWebSocket.js'

type Script = (socket: ScriptedSocket, frame: Record<string, unknown>) => void

/** Global WebSocket stand-in: each instance answers its frames with the active script. */
class ScriptedSocket {
  static script: Script = () => undefined
  static readonly created: ScriptedSocket[] = []

  readonly sent: Record<string, unknown>[] = []
  readyState = 0
  private readonly listeners = new Map<string, Array<(event: WebSocketEventLike) => void>>()

  constructor(readonly url: string) {
    ScriptedSocket.created.push(this)
    queueMicrotask(() => {
      this.readyState = 1
      this.emit('open')
    })
  }

  addEventListener(type: string, listener: (event: WebSocketEventLike) => void): void {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener])
  }

  emit(type: string, event: WebSocketEventLike = {}): void {
    for (const listener of [...(this.listeners.get(type) ?? [])]) listener({ type, ...event })
  }

  event(value: Record<string, unknown>): void {
    queueMicrotask(() => this.emit('message', { data: JSON.stringify(value) }))
  }

  drop(code: number): void {
    queueMicrotask(() => {
      this.readyState = 3
      this.emit('close', { code, reason: '' })
    })
  }

  send(data: string): void {
    const frame = JSON.parse(data) as Record<string, unknown>
    this.sent.push(frame)
    ScriptedSocket.script(this, frame)
  }

  close(code = 1000, reason = ''): void {
    if (this.readyState === 3) return
    this.readyState = 3
    this.emit('close', { code, reason })
  }
}

const globals = globalThis as { WebSocket?: unknown }
let savedWebSocket: unknown

beforeEach(() => {
  savedWebSocket = globals.WebSocket
  globals.WebSocket = ScriptedSocket
  ScriptedSocket.created.length = 0
})

afterEach(() => {
  globals.WebSocket = savedWebSocket
  closeCodexWebSocketSessions()
  clearCodexWebSocketFallback()
})

function codexClient(fetchCalls: { count: number } = { count: 0 }): ResponsesApiClient {
  return new ResponsesApiClient({
    providerName: 'openai-codex',
    apiKey: 'fixture',
    baseUrl: 'https://chatgpt.test/backend-api',
    codexTransport: 'auto',
    fetchImplementation: async () => {
      fetchCalls.count += 1
      throw new Error('the SSE fallback must not run')
    },
  })
}

function request(sessionId: string, messages: CompletionRequest['messages']): CompletionRequest {
  return { model: 'gpt-5.3-codex', sessionId, messages }
}

async function drain(stream: AsyncIterable<LlmDelta>): Promise<LlmDelta[]> {
  const deltas: LlmDelta[] = []
  for await (const delta of stream) deltas.push(delta)
  return deltas
}

function answer(text: string, responseId: string): Script {
  return socket => {
    socket.event({ type: 'response.created', response: { id: responseId } })
    socket.event({ type: 'response.output_text.delta', delta: text })
    socket.event({ type: 'response.completed', response: { id: responseId, status: 'completed' } })
  }
}

test('a socket that drops mid-reply surfaces as a retryable truncated stream', async () => {
  ScriptedSocket.script = socket => {
    socket.event({ type: 'response.created', response: { id: 'resp_1' } })
    socket.event({ type: 'response.output_text.delta', delta: 'Running' })
    socket.drop(1006)
  }
  const seen: LlmDelta[] = []

  const failure = await (async () => {
    for await (const delta of codexClient().stream(request('drop-session', [{ role: 'user', content: 'go' }]))) {
      seen.push(delta)
    }
  })().then(() => undefined, (error: unknown) => error)

  expect(seen).toEqual([{ content: 'Running' }])
  expect(failure).toBeInstanceOf(StreamTruncatedError)
  expect((failure as Error).message).toContain('1006')
  expect(classifyError(failure).retryable).toBe(true)
  // Goal rounds classify the failure from its message text alone.
  expect(classifyError(new Error((failure as Error).message)).retryable).toBe(true)
})

test('cancelling a turn before any output keeps the session on the WebSocket transport', async () => {
  // The model is still reasoning: nothing comes back before the user stops.
  ScriptedSocket.script = () => undefined
  const fetchCalls = { count: 0 }
  const controller = new AbortController()
  const stream = codexClient(fetchCalls).stream(request('cancel-session', [{ role: 'user', content: 'go' }]), controller.signal)
  setTimeout(() => controller.abort(), 20)

  await expect(drain(stream)).rejects.toThrow()

  expect(codexWebSocketFallbackActive('cancel-session')).toBe(false)
  expect(fetchCalls.count).toBe(0)
})

test("a session's cached continuation does not outlive its pooled socket", async () => {
  const first = [{ role: 'user' as const, content: 'hi' }]
  const second = [...first, { role: 'assistant' as const, content: 'Hello' }, { role: 'user' as const, content: 'more' }]

  // Control: on the same pooled socket the second turn extends the first.
  ScriptedSocket.script = answer('Hello', 'resp_kept')
  await drain(codexClient().stream(request('kept-session', first)))
  ScriptedSocket.script = answer('More', 'resp_kept_2')
  await drain(codexClient().stream(request('kept-session', second)))
  expect(ScriptedSocket.created).toHaveLength(1)
  expect(ScriptedSocket.created[0]?.sent[1]?.previous_response_id).toBe('resp_kept')

  // Once the session's sockets are gone, nothing can extend the response:
  // the next turn sends the full context and the stale body is not kept.
  ScriptedSocket.created.length = 0
  ScriptedSocket.script = answer('Hello', 'resp_closed')
  await drain(codexClient().stream(request('closed-session', first)))
  closeCodexWebSocketSessions('closed-session')
  ScriptedSocket.script = answer('More', 'resp_closed_2')
  await drain(codexClient().stream(request('closed-session', second)))
  expect(ScriptedSocket.created).toHaveLength(2)
  expect(ScriptedSocket.created[1]?.sent[0]?.previous_response_id).toBeUndefined()
})
