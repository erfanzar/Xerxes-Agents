// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, spyOn, test } from 'bun:test'

import { CortexCompletionService, type CortexStreamEvent } from '../src/api-server/cortexCompletionService.js'
import { OpenAiApiServer } from '../src/api-server/server.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'

class RecordingClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  constructor(private readonly deltas: readonly LlmDelta[]) {}

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    yield* this.deltas
  }
}

const fixedNow = () => 1_700_000_000_000

test('health and model discovery expose the registered OpenAI-compatible models', async () => {
  const server = new OpenAiApiServer({
    llm: new RecordingClient([]),
    models: ['gpt-4o', 'qwen-max'],
    now: fixedNow,
  })

  const health = await server.fetch(new Request('http://xerxes.test/health'))
  expect(health.status).toBe(200)
  expect(await health.json()).toEqual({ status: 'healthy', agents: 2 })

  const models = await server.fetch(new Request('http://xerxes.test/v1/models'))
  expect(models.status).toBe(200)
  expect(await models.json()).toEqual({
    object: 'list',
    data: [
      { id: 'gpt-4o', object: 'model', created: 1_700_000_000, owned_by: 'xerxes' },
      { id: 'qwen-max', object: 'model', created: 1_700_000_000, owned_by: 'xerxes' },
    ],
  })
})

test('non-streaming chat completions aggregate LlmClient deltas without leaking thinking', async () => {
  const client = new RecordingClient([
    { content: 'Hello <thi' },
    { content: 'nk>private rationale</think>world', usage: { inputTokens: 11, outputTokens: 3 } },
    { finishReason: 'stop' },
  ])
  const server = new OpenAiApiServer({
    llm: client,
    models: ['gpt-4o'],
    now: fixedNow,
    responseId: () => 'chatcmpl-test',
  })

  const response = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({
      model: 'gpt-4o',
      metadata: 'opaque caller field',
      messages: [
        { role: 'system', content: 'Be concise.' },
        { role: 'user', content: 'Say hello.' },
      ],
      max_tokens: 42,
      temperature: 0.2,
      top_p: 0.8,
      stop: ['END'],
      tools: [{
        type: 'function',
        function: {
          name: 'ReadFile',
          description: 'Read a workspace file.',
          parameters: { type: 'object', properties: { path: { type: 'string' } } },
        },
      }],
      tool_choice: 'required',
    }),
  }))

  expect(response.status).toBe(200)
  expect(await response.json()).toEqual({
    id: 'chatcmpl-test',
    object: 'chat.completion',
    created: 1_700_000_000,
    model: 'gpt-4o',
    choices: [{
      index: 0,
      message: { role: 'assistant', content: 'Hello world' },
      finish_reason: 'stop',
    }],
    usage: { prompt_tokens: 11, completion_tokens: 3, total_tokens: 14 },
  })
  expect(client.requests).toEqual([{
    model: 'gpt-4o',
    messages: [
      { role: 'system', content: 'Be concise.' },
      { role: 'user', content: 'Say hello.' },
    ],
    maxTokens: 42,
    temperature: 0.2,
    topP: 0.8,
    stop: ['END'],
    tools: [{
      type: 'function',
      function: {
        name: 'ReadFile',
        description: 'Read a workspace file.',
        parameters: { type: 'object', properties: { path: { type: 'string' } } },
      },
    }],
    toolChoice: 'any',
  }])
})

test('streaming chat completions use OpenAI SSE framing and retain tool calls', async () => {
  const client = new RecordingClient([
    { content: 'one <thi' },
    { content: 'nk>private</think>two' },
    {
      toolCalls: [{
        id: 'call_123',
        type: 'function',
        function: { name: 'ReadFile', arguments: { path: 'README.md' } },
      }],
      finishReason: 'tool_calls',
      usage: { inputTokens: 7, outputTokens: 2 },
    },
  ])
  const server = new OpenAiApiServer({
    llm: client,
    models: ['gpt-4o'],
    now: fixedNow,
    responseId: () => 'chatcmpl-stream',
  })

  const response = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'Inspect the readme.' }],
      stream: true,
    }),
  }))

  expect(response.headers.get('content-type')).toContain('text/event-stream')
  const body = await response.text()
  expect(body).toContain('data: [DONE]')
  expect(body).not.toContain('private')
  const frames = parseSseFrames(body)
  expect(frames).toEqual([
    {
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: 1_700_000_000,
      model: 'gpt-4o',
      choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
    },
    {
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: 1_700_000_000,
      model: 'gpt-4o',
      choices: [{ index: 0, delta: { content: 'one ' }, finish_reason: null }],
    },
    {
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: 1_700_000_000,
      model: 'gpt-4o',
      choices: [{ index: 0, delta: { content: 'two' }, finish_reason: null }],
    },
    {
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: 1_700_000_000,
      model: 'gpt-4o',
      choices: [{
        index: 0,
        delta: {
          tool_calls: [{
            index: 0,
            id: 'call_123',
            type: 'function',
            function: { name: 'ReadFile', arguments: '{"path":"README.md"}' },
          }],
        },
        finish_reason: null,
      }],
    },
    {
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: 1_700_000_000,
      model: 'gpt-4o',
      choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }],
      usage: { prompt_tokens: 7, completion_tokens: 2, total_tokens: 9 },
    },
  ])
})

test('invalid requests and unavailable models return OpenAI-shaped errors', async () => {
  const server = new OpenAiApiServer({ llm: new RecordingClient([]), models: ['gpt-4o'] })

  const malformed = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({ model: 'gpt-4o', messages: 'not an array' }),
  }))
  expect(malformed.status).toBe(400)
  expect(await malformed.json()).toEqual({
    error: {
      message: 'messages must be an array.',
      type: 'invalid_request_error',
      param: 'messages',
      code: null,
    },
  })

  const missingModel = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({ model: 'not-real', messages: [{ role: 'user', content: 'Hi.' }] }),
  }))
  expect(missingModel.status).toBe(404)
  expect(await missingModel.json()).toMatchObject({
    error: { type: 'invalid_request_error', param: 'model', code: 'model_not_found' },
  })
})

test('Cortex model ids route through the explicit native Cortex service with validated metadata', async () => {
  const taskRequests: Array<{
    readonly background?: string
    readonly model: string
    readonly processType: string
    readonly prompt: string
  }> = []
  const cortex = new CortexCompletionService({
    execution: {
      executeTask: async request => {
        taskRequests.push(request)
        return { output: 'Native Cortex task result.' }
      },
      executeInstruction: async () => ({ output: 'unused' }),
      streamTask: emptyCortexStream,
      streamInstruction: emptyCortexStream,
    },
    now: fixedNow,
    responseId: () => 'chatcmpl-cortex-api',
  })
  const client = new RecordingClient([{ content: 'The portable LLM client must not run.' }])
  const server = new OpenAiApiServer({
    llm: client,
    cortex,
    models: ['gpt-4o', 'cortex-task-parallel'],
  })

  const response = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({
      model: 'cortex-task-parallel',
      messages: [
        { role: 'user', content: 'Old request.' },
        { role: 'assistant', content: 'Acknowledged.' },
        { role: 'user', content: 'Run the native task.' },
      ],
      metadata: { task_mode: true, process_type: 'hierarchical', background: 'Use source evidence.' },
    }),
  }))

  expect(response.status).toBe(200)
  expect(await response.json()).toMatchObject({
    id: 'chatcmpl-cortex-api',
    model: 'cortex-task-parallel',
    choices: [{ message: { content: 'Native Cortex task result.' } }],
  })
  expect(taskRequests).toEqual([{
    model: 'cortex-task-parallel',
    prompt: 'Run the native task.',
    processType: 'hierarchical',
    background: 'Use source evidence.',
  }])
  expect(client.requests).toEqual([])

  const disabled = new OpenAiApiServer({ llm: client, models: ['cortex-task'] })
  const disabledResponse = await disabled.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({ model: 'cortex-task', messages: [{ role: 'user', content: 'Run it.' }] }),
  }))
  expect(disabledResponse.status).toBe(404)
  expect(await disabledResponse.json()).toMatchObject({ error: { code: 'cortex_unavailable' } })

  const malformedMetadata = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({
      model: 'cortex-task-parallel',
      messages: [{ role: 'user', content: 'Validate metadata.' }],
      metadata: 'not-an-object',
    }),
  }))
  expect(malformedMetadata.status).toBe(400)
  expect(await malformedMetadata.json()).toMatchObject({
    error: { param: 'metadata', type: 'invalid_request_error' },
  })
})

test('Cortex streaming models retain service-provided SSE metadata through the HTTP handler', async () => {
  const cortex = new CortexCompletionService({
    execution: {
      executeTask: async () => ({ output: 'unused' }),
      executeInstruction: async () => ({ output: 'unused' }),
      streamTask: emptyCortexStream,
      streamInstruction: () => cortexEvents([
        { type: 'function_detection', message: 'lookup' },
        { type: 'stream_chunk', content: 'Done.' },
      ]),
    },
    now: fixedNow,
    responseId: () => 'chatcmpl-cortex-sse',
  })
  const server = new OpenAiApiServer({
    llm: new RecordingClient([]),
    cortex,
    models: ['cortex'],
  })

  const response = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({ model: 'cortex', stream: true, messages: [{ role: 'user', content: 'Research.' }] }),
  }))

  expect(response.headers.get('content-type')).toContain('text/event-stream')
  const frames = parseSseFrames(await response.text()) as Array<Record<string, unknown>>
  expect(frames).toHaveLength(3)
  expect(frames[0]).toMatchObject({ id: 'chatcmpl-cortex-sse', metadata: { event: 'function_detection' } })
  expect(frames[0]).not.toHaveProperty('usage')
  expect(frames[1]).toMatchObject({ choices: [{ delta: { content: 'Done.' } }] })
  expect(frames[2]).toMatchObject({ choices: [{ finish_reason: 'stop' }] })
  expect(frames[2]).not.toHaveProperty('usage')
})

test('CORS is opt-in and configured origins receive browser preflight headers', async () => {
  const server = new OpenAiApiServer({
    llm: new RecordingClient([]),
    models: ['gpt-4o'],
    cors: {
      origins: ['https://console.example'],
      allowCredentials: true,
      allowHeaders: ['Authorization', 'Content-Type', 'X-Trace-Id'],
      maxAge: 600,
    },
  })

  const preflight = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'OPTIONS',
    headers: {
      Origin: 'https://console.example',
      'Access-Control-Request-Method': 'POST',
      'Access-Control-Request-Headers': 'authorization, x-trace-id',
    },
  }))
  expect(preflight.status).toBe(204)
  expect(preflight.headers.get('access-control-allow-origin')).toBe('https://console.example')
  expect(preflight.headers.get('access-control-allow-credentials')).toBe('true')
  expect(preflight.headers.get('access-control-allow-methods')).toBe('GET, POST, OPTIONS')
  expect(preflight.headers.get('access-control-allow-headers')).toBe('Authorization, Content-Type, X-Trace-Id')
  expect(preflight.headers.get('access-control-max-age')).toBe('600')

  const allowed = await server.fetch(new Request('http://xerxes.test/v1/models', {
    headers: { Origin: 'https://console.example' },
  }))
  expect(allowed.headers.get('access-control-allow-origin')).toBe('https://console.example')

  const denied = await server.fetch(new Request('http://xerxes.test/health', {
    headers: { Origin: 'https://untrusted.example' },
  }))
  expect(denied.status).toBe(200)
  expect(denied.headers.get('access-control-allow-origin')).toBeNull()

  const unconfigured = new OpenAiApiServer({ llm: new RecordingClient([]), models: ['gpt-4o'] })
  const defaultResponse = await unconfigured.fetch(new Request('http://xerxes.test/health', {
    headers: { Origin: 'https://console.example' },
  }))
  expect(defaultResponse.headers.get('access-control-allow-origin')).toBeNull()
})

test('configured bearer authentication protects API routes while keeping health probes available', async () => {
  const server = new OpenAiApiServer({
    llm: new RecordingClient([]),
    models: ['gpt-4o'],
    auth: { token: 'test-api-token' },
  })

  const health = await server.fetch(new Request('http://xerxes.test/health'))
  expect(health.status).toBe(200)

  const missingCredentials = await server.fetch(new Request('http://xerxes.test/v1/models'))
  expect(missingCredentials.status).toBe(401)
  expect(missingCredentials.headers.get('www-authenticate')).toBe('Bearer')
  expect(await missingCredentials.json()).toEqual({
    error: {
      message: 'Invalid authentication credentials.',
      type: 'authentication_error',
      param: null,
      code: 'invalid_api_key',
    },
  })

  const authorized = await server.fetch(new Request('http://xerxes.test/v1/models', {
    headers: { Authorization: 'Bearer test-api-token' },
  }))
  expect(authorized.status).toBe(200)

  const sameLengthWrongToken = await server.fetch(new Request('http://xerxes.test/v1/models', {
    headers: { Authorization: 'Bearer test-api-tokeN' },
  }))
  expect(sameLengthWrongToken.status).toBe(401)
})

test('configured body limits reject oversized chat-completion payloads before invoking the LLM', async () => {
  const client = new RecordingClient([{ content: 'ok' }])
  const server = new OpenAiApiServer({
    llm: client,
    models: ['gpt-4o'],
    maxRequestBodyBytes: 256,
  })
  const oversizedPayload = JSON.stringify({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'x'.repeat(512) }],
  })

  const oversized = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: oversizedPayload,
  }))
  expect(oversized.status).toBe(413)
  expect(await oversized.json()).toEqual({
    error: {
      message: 'Request body too large.',
      type: 'invalid_request_error',
      param: null,
      code: 'request_too_large',
    },
  })
  expect(client.requests).toHaveLength(0)

  const declaredOversized = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Length': '257' },
    body: '{}',
  }))
  expect(declaredOversized.status).toBe(413)
  expect(client.requests).toHaveLength(0)
})

test('configured request rate limits are per caller and reset after their window elapses', async () => {
  let currentTime = 1_000
  const server = new OpenAiApiServer({
    llm: new RecordingClient([]),
    models: ['gpt-4o'],
    now: () => currentTime,
    rateLimit: { maxRequests: 2, windowMs: 1_000 },
  })
  const request = () => new Request('http://xerxes.test/health')

  const first = await server.fetch(request(), '198.51.100.10')
  expect(first.status).toBe(200)
  expect(first.headers.get('x-ratelimit-limit')).toBe('2')
  expect(first.headers.get('x-ratelimit-remaining')).toBe('1')

  const second = await server.fetch(request(), '198.51.100.10')
  expect(second.status).toBe(200)
  expect(second.headers.get('x-ratelimit-remaining')).toBe('0')

  const limited = await server.fetch(request(), '198.51.100.10')
  expect(limited.status).toBe(429)
  expect(limited.headers.get('retry-after')).toBe('1')
  expect(await limited.json()).toMatchObject({
    error: { type: 'rate_limit_error', code: 'rate_limit_exceeded' },
  })

  const otherCaller = await server.fetch(request(), '198.51.100.11')
  expect(otherCaller.status).toBe(200)

  currentTime += 1_001
  const reset = await server.fetch(request(), '198.51.100.10')
  expect(reset.status).toBe(200)
  expect(reset.headers.get('x-ratelimit-remaining')).toBe('1')
})

test('empty message lists are rejected with a 400 before any model lookup', async () => {
  const server = new OpenAiApiServer({ llm: new RecordingClient([]), models: ['gpt-4o'] })

  const response = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({ model: 'gpt-4o', messages: [] }),
  }))
  expect(response.status).toBe(400)
  expect(await response.json()).toEqual({
    error: {
      message: 'messages must contain at least one message.',
      type: 'invalid_request_error',
      param: 'messages',
      code: null,
    },
  })
})

test('frequency and presence penalties are forwarded to the LLM client', async () => {
  const client = new RecordingClient([{ content: 'ok', finishReason: 'stop' }])
  const server = new OpenAiApiServer({ llm: client, models: ['gpt-4o'] })

  const response = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'Hi.' }],
      frequency_penalty: 0.5,
      presence_penalty: -0.25,
    }),
  }))
  expect(response.status).toBe(200)
  expect(client.requests[0]).toMatchObject({ frequencyPenalty: 0.5, presencePenalty: -0.25 })

  const invalid = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'Hi.' }],
      frequency_penalty: 'high',
    }),
  }))
  expect(invalid.status).toBe(400)
  expect(await invalid.json()).toMatchObject({
    error: { type: 'invalid_request_error', param: 'frequency_penalty' },
  })
})

test('provider finish reasons are normalized to the chat-completions enum', async () => {
  const cases = [
    ['completed', 'stop'],
    ['end_turn', 'stop'],
    ['stop_sequence', 'stop'],
    ['incomplete', 'length'],
    ['max_tokens', 'length'],
    ['tool_use', 'tool_calls'],
    ['safety', 'content_filter'],
    ['recitation', 'content_filter'],
    ['error', 'stop'],
    ['other', 'stop'],
  ] as const

  for (const [providerReason, expected] of cases) {
    const server = new OpenAiApiServer({
      llm: new RecordingClient([{ content: 'x', finishReason: providerReason }]),
      models: ['gpt-4o'],
    })
    const response = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify({ model: 'gpt-4o', messages: [{ role: 'user', content: 'Hi.' }] }),
    }))
    expect(response.status).toBe(200)
    const body = await response.json() as { choices: [{ finish_reason: string }] }
    expect(body.choices[0].finish_reason).toBe(expected)
  }
})

test('streaming finish chunks normalize provider finish reasons', async () => {
  const server = new OpenAiApiServer({
    llm: new RecordingClient([{ content: 'partial', finishReason: 'incomplete' }]),
    models: ['gpt-4o'],
    now: fixedNow,
    responseId: () => 'chatcmpl-normalized',
  })

  const response = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'Hi.' }],
      stream: true,
    }),
  }))
  const frames = parseSseFrames(await response.text()) as Array<{ choices: [{ finish_reason: string | null }] }>
  expect(frames[frames.length - 1]?.choices[0]?.finish_reason).toBe('length')
})

test('streaming omits usage the provider never reported, even when include_usage is set', async () => {
  const server = new OpenAiApiServer({
    llm: new RecordingClient([{ content: 'done', finishReason: 'stop' }]),
    models: ['gpt-4o'],
    now: fixedNow,
    responseId: () => 'chatcmpl-no-usage',
  })

  const response = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'Hi.' }],
      stream: true,
      stream_options: { include_usage: true },
    }),
  }))
  const frames = parseSseFrames(await response.text()) as Array<Record<string, unknown>>
  const final = frames[frames.length - 1]
  expect(final).toMatchObject({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })
  expect(final).not.toHaveProperty('usage')
})

test('stream_options.include_usage set to false suppresses provider-reported usage', async () => {
  const server = new OpenAiApiServer({
    llm: new RecordingClient([{
      content: 'done',
      finishReason: 'stop',
      usage: { inputTokens: 5, outputTokens: 1 },
    }]),
    models: ['gpt-4o'],
    now: fixedNow,
    responseId: () => 'chatcmpl-suppressed-usage',
  })

  const response = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'Hi.' }],
      stream: true,
      stream_options: { include_usage: false },
    }),
  }))
  const frames = parseSseFrames(await response.text()) as Array<Record<string, unknown>>
  expect(frames[frames.length - 1]).not.toHaveProperty('usage')
})

test('request bodies default to a 16 MiB limit that can be disabled', async () => {
  const client = new RecordingClient([{ content: 'ok', finishReason: 'stop' }])
  const server = new OpenAiApiServer({ llm: client, models: ['gpt-4o'] })
  const oversizedPayload = JSON.stringify({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'x'.repeat(16 * 1024 * 1024) }],
  })

  const oversized = await server.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: oversizedPayload,
  }))
  expect(oversized.status).toBe(413)
  expect(await oversized.json()).toMatchObject({ error: { code: 'request_too_large' } })
  expect(client.requests).toHaveLength(0)

  const unbounded = new OpenAiApiServer({ llm: client, models: ['gpt-4o'], maxRequestBodyBytes: 0 })
  const accepted = await unbounded.fetch(new Request('http://xerxes.test/v1/chat/completions', {
    method: 'POST',
    body: oversizedPayload,
  }))
  expect(accepted.status).toBe(200)
  expect(client.requests).toHaveLength(1)
})

test('a rate-limit key failure returns an OpenAI-shaped error instead of throwing', async () => {
  const server = new OpenAiApiServer({
    llm: new RecordingClient([]),
    models: ['gpt-4o'],
    rateLimit: { maxRequests: 1, key: () => '' },
  })

  const response = await server.fetch(new Request('http://xerxes.test/health'))
  expect(response.status).toBe(500)
  expect(await response.json()).toEqual({
    error: { message: 'Internal server error.', type: 'api_error', param: null, code: null },
  })
})

test('rate limiter evicts the idlest keys once the tracked-key bound is exceeded', async () => {
  let currentTime = 1_000
  const server = new OpenAiApiServer({
    llm: new RecordingClient([]),
    models: ['gpt-4o'],
    now: () => currentTime,
    rateLimit: { maxRequests: 1, windowMs: 60_000 },
  })
  const request = () => new Request('http://xerxes.test/health')

  expect((await server.fetch(request(), '198.51.100.1')).status).toBe(200)
  expect((await server.fetch(request(), '198.51.100.1')).status).toBe(429)

  for (let index = 0; index < 10_001; index += 1) {
    currentTime += 1
    expect((await server.fetch(request(), `203.0.113.${index}`)).status).toBe(200)
  }

  // The earliest key was evicted to keep the map bounded, so it is allowed again.
  expect((await server.fetch(request(), '198.51.100.1')).status).toBe(200)
})

test('listen binds loopback without warnings and warns on unauthenticated public binds', async () => {
  const warn = spyOn(console, 'warn').mockImplementation(() => {})
  const api = new OpenAiApiServer({ llm: new RecordingClient([]), models: ['gpt-4o'] })
  const authenticated = new OpenAiApiServer({
    llm: new RecordingClient([]),
    models: ['gpt-4o'],
    auth: { token: 'secret-token' },
  })
  const loopback = api.listen({ port: 0 })
  const authenticatedPublic = authenticated.listen({ hostname: '0.0.0.0', port: 0 })
  const publicServer = api.listen({ hostname: '0.0.0.0', port: 0 })
  try {
    expect(loopback.hostname).toBe('127.0.0.1')
    expect(warn).toHaveBeenCalledTimes(1)
    expect(String(warn.mock.calls[0]?.[0])).toContain('without bearer authentication')
  } finally {
    loopback.stop(true)
    authenticatedPublic.stop(true)
    publicServer.stop(true)
    warn.mockRestore()
  }
})

test("slow completions survive idle gaps longer than Bun's default 10s timeout", async () => {
  const delayedClient: LlmClient = {
    async *stream() {
      yield { content: 'start' }
      await Bun.sleep(11_500)
      yield { content: 'end', finishReason: 'stop' }
    },
  }
  const api = new OpenAiApiServer({ llm: delayedClient, models: ['gpt-4o'] })
  const server = api.listen({ port: 0 })
  const completion = (stream: boolean) => fetch(`http://127.0.0.1:${server.port}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'Take your time.' }],
      stream,
    }),
  })
  try {
    const [streamed, buffered] = await Promise.all([completion(true), completion(false)])
    expect(streamed.status).toBe(200)
    const body = await streamed.text()
    expect(body).toContain('"content":"end"')
    expect(body).toContain('data: [DONE]')
    expect(buffered.status).toBe(200)
    expect(await buffered.json()).toMatchObject({ choices: [{ message: { content: 'startend' } }] })
  } finally {
    server.stop(true)
  }
}, 25_000)

function parseSseFrames(body: string): unknown[] {
  return body
    .split('\n\n')
    .filter(frame => frame.startsWith('data: {'))
    .map(frame => JSON.parse(frame.slice('data: '.length)) as unknown)
}

async function* emptyCortexStream(): AsyncGenerator<CortexStreamEvent> {}

async function* cortexEvents(events: readonly CortexStreamEvent[]): AsyncGenerator<CortexStreamEvent> {
  for (const event of events) {
    yield event
  }
}
