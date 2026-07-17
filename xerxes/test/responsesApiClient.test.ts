// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ResponsesApiClient,
  createLlmClient,
  type CompletionRequest,
  type LlmDelta,
} from '../src/llms/client.js'

test('Responses API client maps request tools and streamed events into neutral deltas', async () => {
  let endpoint = ''
  let payload: Record<string, unknown> | undefined
  const client = new ResponsesApiClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async (input, init) => {
      endpoint = String(input)
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return sseResponse([
        { type: 'response.output_text.delta', delta: 'Inspecting.' },
        {
          type: 'response.output_item.added',
          item: { type: 'function_call', id: 'call_1', name: 'ReadFile' },
        },
        { type: 'response.function_call_arguments.delta', item_id: 'call_1', delta: '{"path":"README.md"}' },
        { type: 'response.output_item.done', item: { type: 'function_call', id: 'call_1', name: 'ReadFile' } },
        {
          type: 'response.completed',
          response: { status: 'completed', usage: { input_tokens: 8, output_tokens: 3 } },
        },
      ])
    },
  })
  const request: CompletionRequest = {
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'Inspect the project.' }],
    toolChoice: 'any',
    tools: [{
      type: 'function',
      function: {
        name: 'ReadFile',
        description: 'Read a path.',
        parameters: { type: 'object', properties: {} },
      },
    }],
  }
  const events: LlmDelta[] = []
  for await (const event of client.stream(request)) events.push(event)

  expect(endpoint).toBe('https://example.invalid/v1/responses')
  expect(payload).toMatchObject({
    model: 'gpt-4o',
    stream: true,
    tool_choice: 'required',
    tools: [{
      type: 'function',
      name: 'ReadFile',
      parameters: { type: 'object', properties: {} },
    }],
  })
  expect(events).toEqual([
    { content: 'Inspecting.' },
    {
      finishReason: 'completed',
      usage: { inputTokens: 8, outputTokens: 3 },
      toolCalls: [{
        id: 'call_1',
        type: 'function',
        function: { name: 'ReadFile', arguments: { path: 'README.md' } },
      }],
    },
  ])
})

test('createLlmClient chooses the Responses API only through explicit configuration', () => {
  const client = createLlmClient('gpt-4o', {}, {
    apiKey: 'test-key',
    baseUrl: 'https://example.invalid/v1',
    responsesApi: true,
  })
  expect(client).toBeInstanceOf(ResponsesApiClient)
})

test('Responses API client supports a native non-streaming completion response', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new ResponsesApiClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return Response.json({
        status: 'completed',
        output: [
          { type: 'reasoning', summary: [{ text: 'Inspect source.' }] },
          { type: 'message', content: [{ type: 'output_text', text: 'I will read it.' }] },
          { type: 'function_call', call_id: 'call-1', name: 'ReadFile', arguments: '{"path":"README.md"}' },
        ],
        usage: {
          input_tokens: 14,
          output_tokens: 6,
          input_tokens_details: { cached_tokens: 3 },
        },
      })
    },
  })

  const completion = await client.complete({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'Read the README.' }],
  })

  expect(payload).toEqual({
    model: 'gpt-4o',
    input: [{ role: 'user', content: 'Read the README.' }],
    stream: false,
  })
  expect(completion).toEqual({
    content: 'I will read it.',
    thinking: 'Inspect source.',
    finishReason: 'completed',
    usage: { inputTokens: 14, outputTokens: 6, cacheReadTokens: 3 },
    toolCalls: [{ id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
  })
})

test('Responses API client translates tool history into native input items', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new ResponsesApiClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return sseResponse([
        { type: 'response.completed', response: { status: 'completed' } },
      ])
    },
  })
  const events: LlmDelta[] = []
  for await (const event of client.stream({
    model: 'gpt-4o',
    messages: [
      { role: 'system', content: 'Be concise.' },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Inspect this ' },
          { type: 'image_url', image_url: { url: 'https://example.invalid/a.png', detail: 'high' } },
        ],
      },
      {
        role: 'assistant',
        content: 'Reading it.',
        tool_calls: [{
          id: 'call-1',
          type: 'function',
          function: { name: 'ReadFile', arguments: { path: 'README.md' } },
        }],
      },
      { role: 'tool', tool_call_id: 'call-1', content: '# Xerxes' },
    ],
  })) events.push(event)

  expect(payload?.input).toEqual([
    { role: 'system', content: 'Be concise.' },
    {
      role: 'user',
      content: [
        { type: 'input_text', text: 'Inspect this ' },
        { type: 'input_image', image_url: 'https://example.invalid/a.png', detail: 'high' },
      ],
    },
    { role: 'assistant', content: 'Reading it.' },
    { type: 'function_call', call_id: 'call-1', name: 'ReadFile', arguments: '{"path":"README.md"}' },
    { type: 'function_call_output', call_id: 'call-1', output: '# Xerxes' },
  ])
})

test('Responses API client surfaces mid-stream provider failures as errors', async () => {
  const client = new ResponsesApiClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async () => sseResponse([
      { type: 'response.output_text.delta', delta: 'partial' },
      {
        type: 'response.failed',
        response: { status: 'failed', error: { code: 'server_error', message: 'Model exploded' } },
      },
    ]),
  })

  await expect(collect(client.stream({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'hi' }],
  }))).rejects.toThrow('stream returned API error (server_error): Model exploded')
})

async function collect(stream: AsyncIterable<LlmDelta>): Promise<LlmDelta[]> {
  const events: LlmDelta[] = []
  for await (const event of stream) {
    events.push(event)
  }
  return events
}

function sseResponse(events: readonly Record<string, unknown>[]): Response {
  const encoder = new TextEncoder()
  return new Response(new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode('data: ' + JSON.stringify(event) + '\n\n'))
      }
      controller.enqueue(encoder.encode('data: [DONE]\n\n'))
      controller.close()
    },
  }), { headers: { 'Content-Type': 'text/event-stream' } })
}
