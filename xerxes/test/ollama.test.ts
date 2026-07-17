// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { OllamaClient, messagesToOllama } from '../src/llms/ollama.js'
import { createLlmClient, type CompletionRequest } from '../src/llms/client.js'

const READ_FILE = {
  type: 'function' as const,
  function: {
    name: 'ReadFile',
    description: 'Read one file.',
    parameters: { type: 'object', properties: { path: { type: 'string' } } },
  },
}

test('Ollama message conversion preserves text, thinking, and prior tool calls', () => {
  expect(messagesToOllama([
    { role: 'system', content: 'Be concise.' },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Read ' },
        { type: 'image_url', image_url: { url: 'https://image.test/a.png' } },
      ],
    },
    {
      role: 'assistant',
      content: 'I will read it.',
      thinking: 'Need the source.',
      tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
    },
    { role: 'tool', tool_call_id: 'call-1', name: 'ReadFile', content: '# Xerxes' },
  ])).toEqual([
    { role: 'system', content: 'Be concise.' },
    { role: 'user', content: 'Read [Image: https://image.test/a.png]' },
    {
      role: 'assistant',
      content: 'I will read it.',
      thinking: 'Need the source.',
      tool_calls: [{ function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
    },
    { role: 'tool', content: '# Xerxes' },
  ])
})

test('Ollama client posts direct chat NDJSON and normalizes content, tool, and final deltas', async () => {
  let endpoint: URL | undefined
  let payload: Record<string, unknown> | undefined
  const client = new OllamaClient({
    baseUrl: 'http://ollama.test/v1',
    topK: 42,
    fetchImplementation: async (input, init) => {
      endpoint = new URL(String(input))
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return ndjsonResponse([
        { message: { role: 'assistant', content: 'Hello ' } },
        {
          message: {
            role: 'assistant',
            content: 'world',
            thinking: 'Compose the answer.',
            tool_calls: [{ function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
          },
        },
        { done: true, done_reason: 'stop', prompt_eval_count: 11, eval_count: 7 },
      ], [5, 11, 3])
    },
  })
  const request: CompletionRequest = {
    model: 'ollama/llama3.3',
    messages: [{ role: 'system', content: 'Be concise.' }, { role: 'user', content: 'Read the README.' }],
    temperature: 0.2,
    topK: 64,
    topP: 0.7,
    maxTokens: 99,
    stop: ['<stop>'],
    tools: [READ_FILE],
  }

  const events = await collect(client.stream(request))

  expect(endpoint?.toString()).toBe('http://ollama.test/api/chat')
  expect(payload).toEqual({
    model: 'llama3.3',
    messages: [
      { role: 'system', content: 'Be concise.' },
      { role: 'user', content: 'Read the README.' },
    ],
    stream: true,
    options: {
      temperature: 0.2,
      top_p: 0.7,
      num_predict: 99,
      stop: ['<stop>'],
      top_k: 64,
    },
    tools: [READ_FILE],
  })
  expect(events).toContainEqual({ content: 'Hello ' })
  expect(events).toContainEqual({
    content: 'world',
    thinking: 'Compose the answer.',
    toolCalls: [expect.objectContaining({
      type: 'function',
      function: { name: 'ReadFile', arguments: { path: 'README.md' } },
    })],
  })
  expect(events).toContainEqual({ finishReason: 'stop', usage: { inputTokens: 11, outputTokens: 7 } })
})

test('the native client factory selects direct Ollama unless the Responses API is explicitly requested', () => {
  const direct = createLlmClient('llama3.3', {}, { baseUrl: 'http://ollama.test/v1' })
  expect(direct).toBeInstanceOf(OllamaClient)
})

test('Ollama native completion posts stream false and returns the complete response', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new OllamaClient({
    baseUrl: 'http://ollama.test/v1',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return Response.json({
        message: {
          role: 'assistant',
          content: 'I will read it.',
          thinking: 'Inspect source.',
          tool_calls: [{ function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
        },
        done: true,
        done_reason: 'stop',
        prompt_eval_count: 13,
        eval_count: 5,
      })
    },
  })

  const completion = await client.complete({
    model: 'llama3.3',
    messages: [{ role: 'user', content: 'Read the README.' }],
  })

  expect(payload).toEqual({
    model: 'llama3.3',
    messages: [{ role: 'user', content: 'Read the README.' }],
    stream: false,
  })
  expect(completion).toEqual({
    content: 'I will read it.',
    thinking: 'Inspect source.',
    finishReason: 'stop',
    usage: { inputTokens: 13, outputTokens: 5 },
    toolCalls: [expect.objectContaining({ function: { name: 'ReadFile', arguments: { path: 'README.md' } } })],
  })
})

test('Ollama client reports direct HTTP failures without treating them as stream chunks', async () => {
  const client = new OllamaClient({
    fetchImplementation: async () => new Response(
      'daemon unavailable',
      { status: 503, statusText: 'Service Unavailable' },
    ),
  })

  await expect(collect(client.stream({ model: 'llama3.3', messages: [{ role: 'user', content: 'hello' }] })))
    .rejects.toThrow('chat stream request failed (503 Service Unavailable): daemon unavailable')
})

test('Ollama client rejects NDJSON records above the configured byte limit', async () => {
  const client = new OllamaClient({
    maxLineBytes: 12,
    fetchImplementation: async () => ndjsonResponse([{ message: { role: 'assistant', content: 'too large' } }]),
  })

  await expect(collect(client.stream({ model: 'llama3.3', messages: [{ role: 'user', content: 'hello' }] })))
    .rejects.toThrow('NDJSON line exceeded maximum size of 12 bytes')
})

test('Ollama client rejects malformed NDJSON instead of silently losing a delta', async () => {
  const client = new OllamaClient({
    fetchImplementation: async () => textResponse('{bad json}\n'),
  })

  await expect(collect(client.stream({ model: 'llama3.3', messages: [{ role: 'user', content: 'hello' }] })))
    .rejects.toThrow('invalid NDJSON JSON: {bad json}')
})

test('Ollama client cancels the response body when the consumer exits early', async () => {
  let cancelled = false
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('{"message":{"role":"assistant","content":"Hello"}}\n'))
    },
    cancel() {
      cancelled = true
    },
  })
  const client = new OllamaClient({
    fetchImplementation: async () => new Response(body, { headers: { 'Content-Type': 'application/x-ndjson' } }),
  })

  for await (const event of client.stream({ model: 'llama3.3', messages: [{ role: 'user', content: 'hello' }] })) {
    void event
    break
  }

  expect(cancelled).toBe(true)
})

async function collect(stream: AsyncIterable<unknown>): Promise<unknown[]> {
  const events: unknown[] = []
  for await (const event of stream) {
    events.push(event)
  }
  return events
}

function ndjsonResponse(events: readonly Record<string, unknown>[], chunkSizes: readonly number[] = []): Response {
  const text = `${events.map(event => JSON.stringify(event)).join('\n')}\n`
  return textResponse(text, chunkSizes)
}

function textResponse(text: string, chunkSizes: readonly number[] = []): Response {
  const encoded = new TextEncoder().encode(text)
  return new Response(new ReadableStream({
    start(controller) {
      let offset = 0
      let index = 0
      while (offset < encoded.byteLength) {
        const size = chunkSizes[index] ?? encoded.byteLength
        controller.enqueue(encoded.slice(offset, offset + size))
        offset += size
        index += 1
      }
      controller.close()
    },
  }), { headers: { 'Content-Type': 'application/x-ndjson' } })
}
