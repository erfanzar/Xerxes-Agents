// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ProviderError } from '../src/core/errors.js'
import { MistralClient, createMistralToolCallIdNormalizer, mistralMessages, mistralPayload } from '../src/llms/mistral.js'
import type { CompletionRequest } from '../src/llms/client.js'
import { detectProvider } from '../src/llms/providerRegistry.js'

const normalize = createMistralToolCallIdNormalizer()

test('the tool-call id normalizer yields distinct nine-character alphanumeric ids', () => {
  const first = normalize('call_abc.123-def')
  expect(first).toMatch(/^[a-zA-Z0-9]{9}$/)
  const second = normalize('call_abc.123-def')
  expect(second).toBe(first)
  const other = normalize('other-id')
  expect(other).toMatch(/^[a-zA-Z0-9]{9}$/)
  expect(other).not.toBe(first)
})

test('mistralMessages replays the Mistral dialect exactly', () => {
  const request: CompletionRequest = {
    model: 'mistral/mistral-small-latest',
    messages: [
      { role: 'system', content: 'be brief' },
      { role: 'user', content: 'hi' },
      {
        role: 'assistant',
        content: 'working',
        thinking: 'internal plan',
        tool_calls: [{
          id: 'call_abc.123-def',
          type: 'function',
          function: { name: 'deploy', arguments: { name: 'api' } },
        }],
      },
      {
        role: 'tool',
        tool_call_id: 'call_abc.123-def',
        content: '',
        is_error: true,
        name: 'deploy',
      },
    ],
  }
  const messages = mistralMessages(request, normalize)
  expect(messages[0]).toEqual({ content: 'be brief', role: 'system' })
  expect(messages[1]).toEqual({ content: 'hi', role: 'user' })
  const assistant = messages[2]!
  expect(assistant.role).toBe('assistant')
  expect(assistant.prefix).toBe(false)
  expect(assistant.content).toEqual([
    { text: 'working', type: 'text' },
    { thinking: [{ text: 'internal plan', type: 'text' }], type: 'thinking' },
  ])
  expect(assistant.tool_calls?.[0]).toEqual({
    function: { arguments: '{"name":"api"}', name: 'deploy' },
    id: normalize('call_abc.123-def'),
    index: 0,
    type: 'function',
  })
  const tool = messages[3]!
  expect(tool.role).toBe('tool')
  expect(tool.tool_call_id).toBe(normalize('call_abc.123-def'))
  expect(tool.name).toBe('deploy')
  expect(tool.content).toEqual([{ text: '[tool error] (no tool output)', type: 'text' }])
})

test('mistralPayload maps sampling, reasoning mode, and cache routing per model', () => {
  const base: CompletionRequest = {
    model: 'mistral/mistral-small-latest',
    messages: [{ role: 'user', content: 'hi' }],
    maxTokens: 512,
    temperature: 0.4,
    thinking: { effort: 'high' },
    sessionId: 'session-1',
    toolChoice: 'none',
    tools: [{
      type: 'function',
      function: { name: 'deploy', description: 'd', parameters: { type: 'object', properties: {} } },
    }],
    topP: 0.9,
  }

  // mistral-small-latest takes reasoning_effort (pi-ai usesReasoningEffort).
  const effortPayload = mistralPayload(base, { promptCaching: true })
  expect(effortPayload.model).toBe('mistral-small-latest')
  expect(effortPayload.stream).toBe(true)
  expect(effortPayload.max_tokens).toBe(512)
  expect(effortPayload.temperature).toBe(0.4)
  expect(effortPayload.top_p).toBe(0.9)
  expect(effortPayload.reasoning_effort).toBe('high')
  expect(effortPayload.prompt_cache_key).toBe('session-1')
  expect(effortPayload.tool_choice).toBe('none')
  expect(effortPayload.tools).toEqual([{
    type: 'function',
    function: { name: 'deploy', description: 'd', parameters: { type: 'object', properties: {} }, strict: false },
  }])

  // Every other reasoning model takes prompt_mode: "reasoning".
  const magistral = mistralPayload({ ...base, model: 'mistral/magistral-medium-latest' }, { promptCaching: true })
  expect(magistral.prompt_mode).toBe('reasoning')
  expect(magistral.reasoning_effort).toBeUndefined()

  // Caching disabled drops the affinity key entirely.
  const noCache = mistralPayload(base, { promptCaching: false })
  expect(noCache.prompt_cache_key).toBeUndefined()
})

function sseResponse(events: string[]): Response {
  const encoder = new TextEncoder()
  return new Response(new ReadableStream({
    start(controller) {
      for (const event of events) controller.enqueue(encoder.encode(event))
      controller.close()
    },
  }), { headers: { 'Content-Type': 'text/event-stream' } })
}

test('streaming assembles content, thinking, tool calls, usage, and finish reason', async () => {
  const requests: { url: string; init?: RequestInit }[] = []
  const client = new MistralClient({
    apiKey: 'k',
    fetchImplementation: async (input, init) => {
      requests.push({ url: String(input), ...(init === undefined ? {} : { init }) })
      return sseResponse([
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"text":"plan"}]}]}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"abc123456","function":{"name":"deploy","arguments":"{\\"na"}}]}}]}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"me\\":\\"api\\"}"}}]}}]}\n\n',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_tokens_details":{"cached_tokens":40}}}\n\n',
        'data: [DONE]\n\n',
      ])
    },
  })

  const deltas = []
  for await (const delta of client.stream({
    model: 'mistral/mistral-small-latest',
    messages: [{ role: 'user', content: 'hi' }],
    sessionId: 'session-9',
  })) deltas.push(delta)

  expect(requests[0]?.url).toBe('https://api.mistral.ai/v1/chat/completions')
  const headers = requests[0]?.init?.headers as Record<string, string>
  expect(headers.Authorization).toBe('Bearer k')
  expect(headers['x-affinity']).toBe('session-9')
  expect(headers.Accept).toBe('text/event-stream')

  expect(deltas).toEqual([
    { content: 'Hel' },
    { thinking: 'plan' },
    { content: 'lo' },
    {
      toolCalls: [{
        id: 'abc123456',
        type: 'function',
        function: { name: 'deploy', arguments: { name: 'api' } },
      }],
    },
    { usage: { inputTokens: 60, outputTokens: 10, cacheReadTokens: 40 } },
    { finishReason: 'tool_calls' },
  ])
})

test('non-ok responses surface the Mistral error body with status', async () => {
  const client = new MistralClient({
    apiKey: 'k',
    fetchImplementation: async () =>
      new Response('{"message":"unauthorized"}', { status: 401, statusText: 'Unauthorized' }),
  })
  await expect(client.complete({
    model: 'mistral/mistral-small-latest',
    messages: [{ role: 'user', content: 'hi' }],
  })).rejects.toThrow(/Mistral API error \(401\)/)
})

test('finish reasons map and unknown reasons are fatal', async () => {
  const lengthClient = new MistralClient({
    apiKey: 'k',
    fetchImplementation: async () => sseResponse([
      'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"model_length"}]}\n\n',
      'data: [DONE]\n\n',
    ]),
  })
  const deltas = []
  for await (const delta of lengthClient.stream({
    model: 'mistral/mistral-small-latest',
    messages: [{ role: 'user', content: 'hi' }],
  })) deltas.push(delta)
  expect(deltas.at(-1)).toEqual({ finishReason: 'length' })

  const errorClient = new MistralClient({
    apiKey: 'k',
    fetchImplementation: async () => sseResponse([
      'data: {"choices":[{"delta":{},"finish_reason":"weird"}]}\n\n',
      'data: [DONE]\n\n',
    ]),
  })
  await expect((async () => {
    for await (const _ of errorClient.stream({
      model: 'mistral/mistral-small-latest',
      messages: [{ role: 'user', content: 'hi' }],
    })) { /* consume */ }
  })()).rejects.toBeInstanceOf(ProviderError)
})

test('model routing recognizes mistral families', () => {
  expect(detectProvider('mistral/mistral-small-latest')).toBe('mistral')
  expect(detectProvider('codestral-latest')).toBe('mistral')
  expect(detectProvider('pixtral-large-latest')).toBe('mistral')
  expect(detectProvider('open-mixtral-8x22b')).toBe('mistral')
})
