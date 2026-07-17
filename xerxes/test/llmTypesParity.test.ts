// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { CortexAgent } from '../src/cortex/agents/agent.js'
import { LLMConfig } from '../src/core/config.js'
import { ProviderError } from '../src/core/errors.js'
import {
  OpenAiCompatibleClient,
  closeLlmClient,
  completeLlm,
  formatLlmMessages,
  getLlmModelInfo,
  processLlmStream,
  type CompletionRequest,
  type LlmClient,
  type LlmDelta,
  withLlmClient,
} from '../src/llms/client.js'
import {
  PROVIDERS,
  detectProvider,
  getContextLimit,
  providerModel,
  resolveProvider,
} from '../src/llms/providerRegistry.js'
import { messagesFromOpenAi } from '../src/streaming/messages.js'
import { messageText } from '../src/types/messages.js'
import {
  OpenAiProtocolValidationError,
  parseOpenAiProtocolMessage,
  toolDefinitionFromOpenAi,
  toolDefinitionToOpenAi,
} from '../src/types/oaiProtocols.js'

const EMPTY_LLM: LlmClient = {
  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {},
}

test('native LLM configuration keeps validated defaults and explicit sampling controls', () => {
  const defaults = new LLMConfig({}, {})
  expect(defaults).toMatchObject({
    model: 'gpt-4',
    temperature: 0.6,
    topK: 64,
    maxTokens: 2_048,
    enableStreaming: true,
  })

  const configured = new LLMConfig({
    model: 'claude-3',
    temperature: 0.5,
    max_tokens: 4_096,
    top_p: 0.8,
    top_k: 20,
    frequency_penalty: 0.4,
    presence_penalty: 0.3,
    repetition_penalty: 1.15,
    enable_streaming: false,
  }, {})
  expect(configured).toMatchObject({
    model: 'claude-3',
    temperature: 0.5,
    maxTokens: 4_096,
    topP: 0.8,
    topK: 20,
    frequencyPenalty: 0.4,
    presencePenalty: 0.3,
    repetitionPenalty: 1.15,
    enableStreaming: false,
  })

  expect(() => new LLMConfig({ model: '' }, {})).toThrow('llm.model')
  expect(() => new LLMConfig({ max_tokens: -1 }, {})).toThrow('llm.maxTokens')
  expect(new LLMConfig({ max_tokens: 1_000_000 }, {}).maxTokens).toBe(1_000_000)
  expect(() => new LLMConfig({ max_tokens: 1_000_001 }, {})).toThrow('llm.maxTokens')
  expect(() => new LLMConfig({ top_p: 0 }, {})).not.toThrow()
  expect(() => new LLMConfig({ top_p: -0.1 }, {})).toThrow('llm.topP')
})

test('native completions collect stream-only clients, honor cleanup, and expose registry-backed model metadata', async () => {
  let closed = 0
  const client: LlmClient = {
    async *stream(): AsyncGenerator<LlmDelta> {
      yield { content: 'Hello ', usage: { inputTokens: 9, outputTokens: 0 } }
      yield {
        content: 'world.',
        thinking: 'Keep it concise.',
        usage: { inputTokens: 0, outputTokens: 4 },
        toolCalls: [{ id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
      }
      yield { finishReason: 'stop', usage: { inputTokens: 9, outputTokens: 4 } }
    },
    close: () => { closed += 1 },
  }

  const completion = await completeLlm(client, {
    model: 'anthropic/claude-sonnet-4-6',
    messages: [{ role: 'user', content: 'Say hello.' }],
  })
  expect(completion).toEqual({
    content: 'Hello world.',
    thinking: 'Keep it concise.',
    finishReason: 'stop',
    usage: { inputTokens: 9, outputTokens: 4 },
    toolCalls: [{ id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
  })

  const streamedText: string[] = []
  await expect(processLlmStream(client.stream({
    model: 'anthropic/claude-sonnet-4-6',
    messages: [{ role: 'user', content: 'Say hello.' }],
  }), content => { streamedText.push(content) })).resolves.toBe('Hello world.')
  expect(streamedText).toEqual(['Hello ', 'world.'])

  await withLlmClient(client, async () => 'done')
  await closeLlmClient({ async *stream(): AsyncGenerator<LlmDelta> {} })
  expect(closed).toBe(1)
  expect(getLlmModelInfo('anthropic/claude-sonnet-4-6', {
    temperature: 0.25,
    maxTokens: 750_000,
    stream: true,
  })).toEqual({
    provider: 'anthropic',
    model: 'anthropic/claude-sonnet-4-6',
    temperature: 0.25,
    maxTokens: 750_000,
    maxModelLen: 1_000_000,
    stream: true,
  })
  expect(getLlmModelInfo('unprefixed-model', {}, { provider: 'custom' }).maxModelLen).toBe(0)
  expect(formatLlmMessages([{ role: 'user', content: 'Hello' }], 'Be concise.')).toEqual([
    { role: 'system', content: 'Be concise.' },
    { role: 'user', content: 'Hello' },
  ])
})

test('OpenAI-compatible native completion requests do not require SSE and normalize tool and usage data', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new OpenAiCompatibleClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://api.openai.com/v1',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return Response.json({
        choices: [{
          finish_reason: 'tool_calls',
          message: {
            content: 'I will read it.',
            reasoning_content: 'Inspect the repository.',
            tool_calls: [{ id: 'call-1', function: { name: 'ReadFile', arguments: '{"path":"README.md"}' } }],
          },
        }],
        usage: {
          prompt_tokens: 12,
          completion_tokens: 5,
          completion_tokens_details: { reasoning_tokens: 2 },
        },
      })
    },
  })

  const completion = await completeLlm(client, {
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'Read the README.' }],
  })

  expect(payload).toEqual({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'Read the README.' }],
    stream: false,
  })
  expect(completion).toEqual({
    content: 'I will read it.',
    thinking: 'Inspect the repository.',
    finishReason: 'tool_calls',
    usage: { inputTokens: 12, outputTokens: 5, reasoningTokens: 2 },
    toolCalls: [{ id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
  })
})

test('OpenAI-compatible streams retain reasoning without guessing custom-gateway sampling extensions', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new OpenAiCompatibleClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'http://localhost:11556/v1/',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return sseResponse([
        { choices: [{ delta: { reasoning_content: 'Inspect the request first.' } }] },
        { choices: [{ delta: { content: 'Final answer.' }, finish_reason: 'stop' }] },
      ])
    },
  })

  const events = await collect(client.stream({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'Hello' }],
    temperature: 0.5,
    maxTokens: 4_096,
    topP: 0.8,
    topK: 20,
    minP: 0.05,
    repetitionPenalty: 1.15,
    presencePenalty: 0.3,
    frequencyPenalty: 0.4,
    stop: ['END'],
    extraBody: { chat_template_kwargs: { enable_thinking: true } },
  }))

  expect(payload).toEqual({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'Hello' }],
    stream: true,
    temperature: 0.5,
    max_tokens: 4_096,
    top_p: 0.8,
    frequency_penalty: 0.4,
    presence_penalty: 0.3,
    stop: ['END'],
    chat_template_kwargs: { enable_thinking: true },
    stream_options: { include_usage: true },
  })
  expect(payload).not.toHaveProperty('top_k')
  expect(payload).not.toHaveProperty('min_p')
  expect(payload).not.toHaveProperty('repetition_penalty')
  expect(events).toEqual([
    { thinking: 'Inspect the request first.' },
    { content: 'Final answer.', finishReason: 'stop' },
  ])
})

test('OpenRouter receives its documented extended sampling fields', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new OpenAiCompatibleClient({
    providerName: 'openrouter',
    apiKey: 'test-key',
    baseUrl: 'https://openrouter.ai/api/v1',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return sseResponse([])
    },
  })

  await collect(client.stream({
    model: 'anthropic/claude-sonnet-4.6',
    messages: [{ role: 'user', content: 'Hello' }],
    topK: 20,
    minP: 0.05,
    repetitionPenalty: 1.15,
  }))

  expect(payload).toMatchObject({
    top_k: 20,
    min_p: 0.05,
    repetition_penalty: 1.15,
  })
})

test('vendor and custom OpenAI-compatible endpoints omit non-standard sampling fields', async () => {
  const providers = ['custom', 'deepseek', 'qwen', 'zhipu', 'minimax'] as const

  for (const providerName of providers) {
    let payload: Record<string, unknown> | undefined
    const client = new OpenAiCompatibleClient({
      providerName,
      apiKey: 'test-key',
      baseUrl: `https://${providerName}.example.test/v1`,
      fetchImplementation: async (_input, init) => {
        payload = JSON.parse(String(init?.body)) as Record<string, unknown>
        return sseResponse([])
      },
    })

    await collect(client.stream({
      model: 'test-model',
      messages: [{ role: 'user', content: 'Hello' }],
      topK: 64,
      minP: 0.05,
      repetitionPenalty: 1.15,
    }))

    expect(payload).not.toHaveProperty('top_k')
    expect(payload).not.toHaveProperty('min_p')
    expect(payload).not.toHaveProperty('repetition_penalty')
  }
})

test('official OpenAI endpoints retain standard penalties while omitting incompatible sampling fields', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new OpenAiCompatibleClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://api.openai.com/v1',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return sseResponse([])
    },
  })

  await collect(client.stream({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'Hello' }],
    topK: 20,
    minP: 0.05,
    repetitionPenalty: 1.15,
    presencePenalty: 0.3,
    frequencyPenalty: 0.4,
  }))

  expect(payload).toMatchObject({
    presence_penalty: 0.3,
    frequency_penalty: 0.4,
  })
  expect(payload).not.toHaveProperty('top_k')
  expect(payload).not.toHaveProperty('min_p')
  expect(payload).not.toHaveProperty('repetition_penalty')
})

test('Kimi Code keeps its fixed temperature and omits unsupported extended sampling fields', async () => {
  const payloads: Record<string, unknown>[] = []
  const client = new OpenAiCompatibleClient({
    providerName: 'kimi-code',
    apiKey: 'test-key',
    baseUrl: 'https://api.kimi.com/coding/v1',
    fetchImplementation: async (_input, init) => {
      payloads.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
      return sseResponse([])
    },
  })

  const request = {
    model: 'kimi-for-coding',
    messages: [{ role: 'user' as const, content: 'Hello' }],
    topK: 64,
  }
  await collect(client.stream({ ...request, temperature: 0.6 }))
  await collect(client.stream({ ...request, temperature: 1 }))

  expect(payloads[0]).not.toHaveProperty('temperature')
  expect(payloads[0]).not.toHaveProperty('top_k')
  expect(payloads[1]).toMatchObject({ temperature: 1 })
  expect(payloads[1]).not.toHaveProperty('top_k')
})

test('provider registry retains Kimi, Claude Code, OpenRouter, and context-window routing', () => {
  expect(resolveProvider('kimi/kimi-latest', { base_url: 'https://api.kimi.com/coding/v1' })).toBe('kimi-code')
  expect(resolveProvider('kimi/kimi-latest', { base_url: 'https://api.moonshot.cn/v1' })).toBe('kimi')
  expect(detectProvider('claude-code/sonnet')).toBe('claude-code')
  expect(resolveProvider('sonnet', { provider: 'claude-code' })).toBe('claude-code')
  expect(resolveProvider('sonnet', { base_url: 'claude-code://local' })).toBe('claude-code')
  expect(providerModel('claude-code/opus', 'claude-code')).toBe('opus')
  expect(PROVIDERS.openrouter).toMatchObject({
    transport: 'openai',
    apiKeyEnv: 'OPENROUTER_API_KEY',
    baseUrl: 'https://openrouter.ai/api/v1',
  })
  expect(providerModel('anthropic/claude-sonnet-4.5', 'openrouter')).toBe('anthropic/claude-sonnet-4.5')
  expect(getContextLimit('kimi/kimi-for-coding')).toBe(262_144)
  expect(getContextLimit('claude-code/opus')).toBe(1_000_000)
  expect(getContextLimit('claude-code/custom')).toBe(200_000)
  expect(getContextLimit('anthropic/claude-sonnet-4-6')).toBe(1_000_000)
})

test('native OpenAI message and tool conversion covers every supported role with strict wire validation', () => {
  expect(messagesFromOpenAi([
    { role: 'system', content: 'Be helpful.' },
    { role: 'user', content: 'Hello' },
    { role: 'assistant', content: 'Hi there', reasoning_content: 'Be concise.' },
    { role: 'tool', content: 'result', tool_call_id: 'tc_1', name: 'Search' },
  ], { preserveSystemRole: true })).toEqual([
    { role: 'system', content: 'Be helpful.' },
    { role: 'user', content: 'Hello' },
    { role: 'assistant', content: 'Hi there', thinking: 'Be concise.' },
    { role: 'tool', content: 'result', tool_call_id: 'tc_1', name: 'Search' },
  ])

  const openAiTool = {
    type: 'function' as const,
    function: {
      name: 'search',
      description: 'Search the web',
      parameters: { type: 'object', properties: { query: { type: 'string' } } },
    },
  }
  expect(toolDefinitionToOpenAi(toolDefinitionFromOpenAi(openAiTool))).toEqual(openAiTool)
  expect(messageText({
    role: 'user',
    content: [
      { type: 'text', text: 'Inspect ' },
      { type: 'image_url', image_url: { url: 'https://example.test/image.png' } },
      { type: 'text', text: 'this.' },
    ],
  })).toBe('Inspect this.')
  expect(() => parseOpenAiProtocolMessage({ role: 'unknown', content: 'test' })).toThrow(OpenAiProtocolValidationError)
})

test('native Cortex agents retain explicit identity and sampling fields while rejecting duplicate tools', () => {
  const agent = new CortexAgent({
    id: 'test_agent',
    role: 'Test Agent',
    goal: 'Verify native agent values.',
    backstory: 'A precise test agent.',
    model: 'gpt-4',
    instructions: 'Test instructions',
    capabilities: [{ name: 'function_calling', description: 'Can call functions' }],
    llm: EMPTY_LLM,
    maxTokens: 1_024,
    temperature: 0.5,
    topP: 0.9,
  })
  expect(agent).toMatchObject({
    id: 'test_agent',
    name: 'Test Agent',
    model: 'gpt-4',
    instructions: 'Test instructions',
    maxTokens: 1_024,
    temperature: 0.5,
    topP: 0.9,
  })
  expect(agent.capabilities).toEqual([{ name: 'function_calling', description: 'Can call functions' }])

  const duplicateTool = {
    type: 'function' as const,
    function: { name: 'search', description: 'Search the web', parameters: { type: 'object' } },
  }
  expect(() => new CortexAgent({
    role: 'Duplicate Tool Agent',
    goal: 'Reject duplicate names.',
    backstory: 'A strict native agent.',
    model: 'gpt-4',
    llm: EMPTY_LLM,
    tools: [duplicateTool, duplicateTool],
  })).toThrow('Duplicate CortexAgent tool: search')
})

test('OpenAI-compatible streams throw on in-stream error payloads instead of skipping them', async () => {
  const structured = new OpenAiCompatibleClient({
    providerName: 'openrouter',
    apiKey: 'test-key',
    baseUrl: 'https://openrouter.ai/api/v1',
    fetchImplementation: async () => sseResponse([
      { choices: [{ delta: { content: 'partial' } }] },
      { error: { code: 502, message: 'Provider returned error' } },
    ]),
  })
  await expect(collect(structured.stream({
    model: 'anthropic/claude-sonnet-4.5',
    messages: [{ role: 'user', content: 'hi' }],
  }))).rejects.toThrow('stream returned API error (502): Provider returned error')

  const plain = new OpenAiCompatibleClient({
    providerName: 'openrouter',
    apiKey: 'test-key',
    baseUrl: 'https://openrouter.ai/api/v1',
    fetchImplementation: async () => sseResponse([{ error: 'stream broke' }]),
  })
  await expect(collect(plain.stream({
    model: 'anthropic/claude-sonnet-4.5',
    messages: [{ role: 'user', content: 'hi' }],
  }))).rejects.toThrow('stream returned API error: stream broke')
})

test('OpenAI-compatible streams append tool-call continuations that omit the chunk index', async () => {
  const client = new OpenAiCompatibleClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://api.openai.com/v1',
    fetchImplementation: async () => sseResponse([
      {
        choices: [{
          delta: { tool_calls: [{ index: 0, id: 'call-1', function: { name: 'ReadFile', arguments: '{"path":' } }] },
        }],
      },
      { choices: [{ delta: { tool_calls: [{ function: { arguments: '"README.md"}' } }] } }] },
      { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
    ]),
  })

  const events = await collect(client.stream({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'read the file' }],
  }))

  expect(events).toContainEqual({
    finishReason: 'tool_calls',
    toolCalls: [{ id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
  })
})

test('OpenAI-compatible HTTP failures cap quoted provider error bodies', async () => {
  const client = new OpenAiCompatibleClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://api.openai.com/v1',
    fetchImplementation: async () => new Response('x'.repeat(8_192), { status: 500 }),
  })

  const failure = await collect(client.stream({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'hi' }],
  })).catch(error => error)
  expect(failure).toBeInstanceOf(ProviderError)
  expect(failure.message).toBe(`Client openai: stream request failed (500): ${'x'.repeat(4_096)}`)
})

test('OpenAI-compatible streams cancel the response body when the consumer exits early', async () => {
  let cancelled = false
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'))
    },
    cancel() {
      cancelled = true
    },
  })
  const client = new OpenAiCompatibleClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://api.openai.com/v1',
    fetchImplementation: async () => new Response(body, { headers: { 'Content-Type': 'text/event-stream' } }),
  })

  for await (const event of client.stream({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'hi' }],
  })) {
    void event
    break
  }

  expect(cancelled).toBe(true)
})

async function collect(stream: AsyncIterable<LlmDelta>): Promise<LlmDelta[]> {
  const events: LlmDelta[] = []
  for await (const event of stream) {
    events.push(event)
  }
  return events
}

function sseResponse(events: readonly Record<string, unknown>[]): Response {
  const body = `${events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('')}data: [DONE]\n\n`
  return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } })
}
