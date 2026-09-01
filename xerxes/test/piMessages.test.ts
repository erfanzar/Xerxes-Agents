// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  PiMessagesClient,
  PiMessagesResponseError,
  piMessagesContext,
  readPiMessagesEvents,
} from '../src/llms/piMessages.js'
import type { ChatMessage } from '../src/types/messages.js'

const HISTORY: ChatMessage[] = [
  { role: 'system', content: 'be brief' },
  { role: 'user', content: 'deploy it' },
  {
    role: 'assistant',
    content: '',
    thinking: 'checking pipeline',
    thinking_signature: 'sig-1',
    tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'deploy', arguments: { name: 'api' } } }],
  },
  { role: 'tool', tool_call_id: 'call_1', content: 'deployed', added_tool_names: ['rollback'] },
]

function client(overrides: Record<string, unknown> = {}): PiMessagesClient {
  return new PiMessagesClient('radius/claude-sonnet', {
    apiKey: 'radius-token',
    baseUrl: 'https://gateway.example/',
    providerName: 'radius',
    ...overrides,
  })
}

function sseResponse(frames: (string | object)[]): Response {
  const encoder = new TextEncoder()
  return new Response(new ReadableStream({
    start(controller) {
      for (const frame of frames) {
        const text = typeof frame === 'string'
          ? (frame.endsWith('\n\n') ? frame : `${frame}\n\n`)
          : `data: ${JSON.stringify(frame)}\n\n`
        controller.enqueue(encoder.encode(text))
      }
      controller.close()
    },
  }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
}

const DONE_FRAME = 'data: {"type":"done","reason":"stop","usage":{"input":1,"output":1,"cacheRead":0,"cacheWrite":0,"totalTokens":2,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}}}\n\n'

const capture = () => {
  const state: { url?: string; init?: RequestInit } = {}
  return {
    state,
    fetch: async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      state.url = String(input)
      if (init !== undefined) state.init = init
      return sseResponse([DONE_FRAME])
    },
  }
}

test('request is a single POST of model, context, and options to <baseUrl>/messages', async () => {
  const { state, fetch } = capture()
  await client({ fetchImplementation: fetch }).complete({
    model: 'radius/claude-sonnet',
    messages: HISTORY,
    temperature: 0.4,
    maxTokens: 256,
    tools: [{
      type: 'function',
      function: { name: 'deploy', description: 'Deploy.', parameters: { type: 'object', properties: {} } },
    }],
    toolChoice: 'any',
    thinking: { effort: 'high' },
    sessionId: 'sess-9',
  })

  expect(state.url).toBe('https://gateway.example/messages')
  expect(state.init?.method).toBe('POST')
  const headers = state.init?.headers as Record<string, string>
  expect(headers.authorization).toBe('Bearer radius-token')
  expect(headers.accept).toBe('text/event-stream')
  expect(headers['content-type']).toBe('application/json')

  const payload = JSON.parse(String(state.init?.body)) as Record<string, unknown>
  expect(payload.model).toBe('claude-sonnet')
  const options = payload.options as Record<string, unknown>
  expect(options.temperature).toBe(0.4)
  expect(options.maxTokens).toBe(256)
  expect(options.reasoning).toBe('high')
  expect(options.toolChoice).toBe('required')
  expect(options.sessionId).toBe('sess-9')
  expect(options.cacheRetention).toBeUndefined()

  const context = payload.context as Record<string, unknown>
  expect(context.systemPrompt).toBe('be brief')
  expect(context.tools).toEqual([{
    name: 'deploy',
    description: 'Deploy.',
    parameters: { type: 'object', properties: {} },
  }])
  const messages = context.messages as Record<string, unknown>[]
  expect(messages).toEqual([
    { role: 'user', content: 'deploy it', timestamp: expect.any(Number) },
    {
      role: 'assistant',
      api: 'pi-messages',
      provider: 'radius',
      model: 'claude-sonnet',
      content: [
        { type: 'thinking', thinking: 'checking pipeline', thinkingSignature: 'sig-1' },
        { type: 'toolCall', id: 'call_1', name: 'deploy', arguments: { name: 'api' } },
      ],
      usage: expect.any(Object),
      stopReason: 'toolUse',
      timestamp: expect.any(Number),
    },
    {
      role: 'toolResult',
      toolCallId: 'call_1',
      toolName: 'deploy',
      content: [{ type: 'text', text: 'deployed' }],
      addedToolNames: ['rollback'],
      isError: false,
      timestamp: expect.any(Number),
    },
  ])
})

test('sse events map onto neutral deltas: text, thinking, tool calls, and done usage', async () => {
  const events = [
    'data: {"type":"start"}',
    'data: {"type":"thinking_start","contentIndex":0}',
    'data: {"type":"thinking_delta","contentIndex":0,"delta":"thin"}',
    'data: {"type":"thinking_end","contentIndex":0,"content":"thinking","contentSignature":"tsig"}',
    'data: {"type":"text_start","contentIndex":1}',
    'data: {"type":"text_delta","contentIndex":1,"delta":"he"}',
    'data: {"type":"text_delta","contentIndex":1,"delta":"llo"}',
    'data: {"type":"text_end","contentIndex":1,"content":"hello"}',
    'data: {"type":"toolcall_start","contentIndex":2,"id":"call_9","toolName":"deploy"}',
    'data: {"type":"toolcall_delta","contentIndex":2,"delta":"{\\"name\\":"}',
    'data: {"type":"toolcall_delta","contentIndex":2,"delta":"\\"api\\"}"}',
    {
      type: 'toolcall_end',
      contentIndex: 2,
      toolCall: { type: 'toolCall', id: 'call_9', name: 'deploy', arguments: { name: 'api' } },
    },
    {
      type: 'done',
      reason: 'toolUse',
      usage: {
        input: 10, output: 5, cacheRead: 3, cacheWrite: 2, totalTokens: 15,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      responseId: 'resp_1',
    },
  ]
  const deltas = []
  for await (const delta of client({
    fetchImplementation: async () => sseResponse(events),
  }).stream({ model: 'radius/claude-sonnet', messages: [{ role: 'user', content: 'hi' }] })) {
    deltas.push(delta)
  }

  expect(deltas).toEqual([
    { thinking: 'thin' },
    // text/thinking endings are authoritative: only the un-streamed remainder.
    { thinking: 'king', thinkingSignature: 'tsig' },
    { content: 'he' },
    { content: 'llo' },
    { toolCalls: [{ id: 'call_9', type: 'function', function: { name: 'deploy', arguments: { name: 'api' } } }] },
    {
      finishReason: 'tool_calls',
      usage: {
        cacheReadTokens: 3,
        cacheCreationTokens: 2,
        inputTokens: 10,
        outputTokens: 5,
      },
    },
  ])
})

test('an authoritative text_end without deltas emits the full content once', async () => {
  const events = [
    'data: {"type":"start"}',
    'data: {"type":"text_start","contentIndex":0}',
    'data: {"type":"text_end","contentIndex":0,"content":"whole answer"}',
    'data: {"type":"done","reason":"stop","usage":{"input":1,"output":1,"cacheRead":0,"cacheWrite":0,"totalTokens":2,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}}}',
  ]
  const deltas = []
  for await (const delta of client({
    fetchImplementation: async () => sseResponse(events),
  }).stream({ model: 'radius/claude-sonnet', messages: [{ role: 'user', content: 'hi' }] })) {
    deltas.push(delta)
  }
  expect(deltas).toEqual([
    { content: 'whole answer' },
    {
      finishReason: 'stop',
      usage: { cacheReadTokens: 0, cacheCreationTokens: 0, inputTokens: 1, outputTokens: 1 },
    },
  ])
})

test('a terminal error event surfaces the backend message as a ProviderError', async () => {
  const events = [
    'data: {"type":"text_delta","contentIndex":0,"delta":"par"}',
    'data: {"type":"error","reason":"error","errorMessage":"policy rejected the request","usage":{"input":1,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":1,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}}}',
  ]
  const run = client({
    fetchImplementation: async () => sseResponse(events),
  }).stream({ model: 'radius/claude-sonnet', messages: [{ role: 'user', content: 'hi' }] })

  await expect((async () => {
    for await (const _delta of run) { /* consume */ }
  })()).rejects.toThrow('policy rejected the request')
})

test('a stream without a terminal event fails instead of ending silently', async () => {
  const run = client({
    fetchImplementation: async () => sseResponse(['data: {"type":"start"}\n\n']),
  }).stream({ model: 'radius/claude-sonnet', messages: [{ role: 'user', content: 'hi' }] })

  await expect((async () => {
    for await (const _delta of run) { /* consume */ }
  })()).rejects.toThrow('stream ended without a terminal event')
})

test('http error bodies become PiMessagesResponseError with the backend code', async () => {
  const clientUnderTest = client({
    fetchImplementation: async () => new Response(
      JSON.stringify({ error: { message: 'quota exhausted', code: 'rate_limited', details: { tier: 'free' } } }),
      { status: 429, statusText: 'Too Many Requests' },
    ),
  })

  await expect(clientUnderTest.complete({
    model: 'radius/claude-sonnet',
    messages: [{ role: 'user', content: 'hi' }],
  })).rejects.toThrow(PiMessagesResponseError)

  try {
    await clientUnderTest.complete({ model: 'radius/claude-sonnet', messages: [{ role: 'user', content: 'hi' }] })
  } catch (error) {
    const responseError = error as PiMessagesResponseError
    expect(responseError.code).toBe('rate_limited')
    expect(responseError.message).toBe('Client radius: 429 Too Many Requests: quota exhausted (rate_limited)')
    expect(responseError.diagnosticDetails.status).toBe(429)
    expect(responseError.diagnosticDetails.model).toBe('claude-sonnet')
  }
})

test('a non-JSON error body falls back to the raw text and truncates diagnostics', async () => {
  const longBody = 'x'.repeat(9_000)
  const clientUnderTest = client({
    fetchImplementation: async () => new Response(longBody, { status: 500, statusText: 'Server Error' }),
  })
  try {
    await clientUnderTest.complete({ model: 'radius/m', messages: [{ role: 'user', content: 'hi' }] })
    expect.unreachable()
  } catch (error) {
    const responseError = error as PiMessagesResponseError
    // Message keeps the raw (untruncated) body per pi-ai; diagnostics truncate.
    expect(responseError.message.startsWith('Client radius: 500 Server Error: ')).toBe(true)
    expect((responseError.diagnosticDetails.body as string).length).toBeLessThanOrEqual(8_193)
    expect(responseError.diagnosticDetails.body).not.toContain('"error"')
  }
})

test('debug appends the query flag and trailing slashes on the base URL collapse', async () => {
  const { state, fetch } = capture()
  await new PiMessagesClient('m', {
    apiKey: 'k',
    baseUrl: 'https://gw.example///',
    debug: true,
    fetchImplementation: fetch,
  }).complete({ model: 'm', messages: [{ role: 'user', content: 'hi' }] })
  expect(state.url).toBe('https://gw.example/messages?debug=1')
})

test('SSE frames split on CRLF and parse a trailing unterminated block', async () => {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      // A CRLF-separated block, then one final block with no trailing blank line.
      controller.enqueue(encoder.encode('data: {"type":"start"}\r\n\r\n'))
      controller.enqueue(encoder.encode('data: {"type":"text_delta","contentIndex":0,"delta":"end"}'))
      controller.close()
    },
  })
  const events = []
  for await (const event of readPiMessagesEvents(stream)) events.push(event)
  expect(events).toEqual([
    { type: 'start' },
    { type: 'text_delta', contentIndex: 0, delta: 'end' },
  ])
})

test('context serialization carries image parts and degrades remote urls to text', () => {
  const context = piMessagesContext({
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: 'look' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,QUJD' } },
        { type: 'image_url', image_url: { url: 'https://cdn.example/x.png' } },
      ],
    }],
  }, { modelId: 'm', now: () => 42, providerId: 'radius' })

  expect(context.messages).toEqual([{
    role: 'user',
    content: [
      { type: 'text', text: 'look' },
      { type: 'image', data: 'QUJD', mimeType: 'image/png' },
      { type: 'text', text: '[Image: https://cdn.example/x.png]' },
    ],
    timestamp: 42,
  }])
  expect(context.systemPrompt).toBeUndefined()
})
