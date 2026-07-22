// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ProviderError } from '../src/core/errors.js'
import { GeminiClient, messagesToGemini } from '../src/llms/gemini.js'
import { createLlmClient, type CompletionRequest } from '../src/llms/client.js'

const READ_FILE = {
  type: 'function' as const,
  function: {
    name: 'ReadFile',
    description: 'Read one file.',
    parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
  },
}

test('Gemini conversion separates system instructions and preserves native tool context', () => {
  expect(messagesToGemini([
    { role: 'system', content: 'Be concise.' },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Inspect ' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,aGVsbG8=' } },
        { type: 'image_url', image_url: { url: 'https://images.test/reference.png' } },
      ],
    },
    {
      role: 'assistant',
      content: 'I will inspect it.',
      thinking: 'Need a source read.',
      thinking_signature: 'thought-1',
      tool_calls: [{
        id: 'call-1',
        type: 'function',
        function: { name: 'ReadFile', arguments: { path: 'README.md' } },
      }],
    },
    { role: 'tool', tool_call_id: 'call-1', content: '# Xerxes' },
    { role: 'tool', tool_call_id: 'call-2', name: 'RunTests', content: 'failed', is_error: true },
  ])).toEqual({
    systemInstruction: { parts: [{ text: 'Be concise.' }] },
    contents: [
      {
        role: 'user',
        parts: [
          { text: 'Inspect ' },
          { inlineData: { mimeType: 'image/png', data: 'aGVsbG8=' } },
          { text: '[Image: https://images.test/reference.png]' },
        ],
      },
      {
        role: 'model',
        parts: [
          { text: 'Need a source read.', thought: true, thoughtSignature: 'thought-1' },
          { text: 'I will inspect it.' },
          { functionCall: { id: 'call-1', name: 'ReadFile', args: { path: 'README.md' } } },
        ],
      },
      {
        role: 'user',
        parts: [
          { functionResponse: { id: 'call-1', name: 'ReadFile', response: { result: '# Xerxes' } } },
          { functionResponse: { id: 'call-2', name: 'RunTests', response: { error: 'failed' } } },
        ],
      },
    ],
  })
})

test('Gemini direct REST stream sends native settings and normalizes all shared deltas', async () => {
  let endpoint: URL | undefined
  let payload: Record<string, unknown> | undefined
  let apiKey: string | undefined
  const client = new GeminiClient({
    apiKey: 'test-key',
    baseUrl: 'https://gemini.test/v1beta',
    generationConfig: { responseMimeType: 'application/json', topK: 42 },
    safetySettings: [{ category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH' }],
    fetchImplementation: async (input, init) => {
      endpoint = new URL(String(input))
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      apiKey = new Headers(init?.headers).get('x-goog-api-key') ?? undefined
      return sseResponse([
        {
          candidates: [{
            content: {
              parts: [
                { text: 'Hello ' },
                { text: 'Inspect source.', thought: true, thoughtSignature: 'thought-2' },
              ],
            },
          }],
        },
        {
          candidates: [{
            content: {
              parts: [
                { text: 'world' },
                { functionCall: { id: 'gemini-call-1', name: 'ReadFile', args: { path: 'README.md' } } },
              ],
            },
            finishReason: 'STOP',
          }],
          usageMetadata: {
            promptTokenCount: 11,
            candidatesTokenCount: 7,
            cachedContentTokenCount: 3,
            thoughtsTokenCount: 2,
          },
        },
      ], [7, 13, 5])
    },
  })
  const request: CompletionRequest = {
    model: 'gemini/gemini-2.0-flash',
    messages: [{ role: 'system', content: 'Be concise.' }, { role: 'user', content: 'Read the README.' }],
    maxTokens: 99,
    temperature: 0.2,
    topK: 64,
    topP: 0.7,
    stop: ['<stop>'],
    tools: [READ_FILE],
    toolChoice: 'any',
  }

  const events = await collect(client.stream(request))

  expect(endpoint?.toString()).toBe('https://gemini.test/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse')
  expect(apiKey).toBe('test-key')
  expect(payload).toEqual({
    systemInstruction: { parts: [{ text: 'Be concise.' }] },
    contents: [{ role: 'user', parts: [{ text: 'Read the README.' }] }],
    generationConfig: {
      responseMimeType: 'application/json',
      topK: 64,
      maxOutputTokens: 99,
      temperature: 0.2,
      topP: 0.7,
      stopSequences: ['<stop>'],
    },
    safetySettings: [{ category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH' }],
    tools: [{ functionDeclarations: [READ_FILE.function] }],
    toolConfig: { functionCallingConfig: { mode: 'ANY' } },
  })
  expect(events).toContainEqual({ content: 'Hello ', thinking: 'Inspect source.', thinkingSignature: 'thought-2' })
  expect(events).toContainEqual({
    content: 'world',
    finishReason: 'stop',
    usage: { inputTokens: 11, outputTokens: 7, cacheReadTokens: 3, reasoningTokens: 2 },
    toolCalls: [{
      id: 'gemini-call-1',
      type: 'function',
      function: { name: 'ReadFile', arguments: { path: 'README.md' } },
    }],
  })
})

test('the native client factory selects direct Gemini and normalizes the official compatibility root', async () => {
  let endpoint = ''
  const client = createLlmClient('gemini-2.0-flash', {}, {
    apiKey: 'test-key',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
    fetchImplementation: async input => {
      endpoint = String(input)
      return sseResponse([{
        candidates: [{ content: { parts: [{ text: 'native' }] }, finishReason: 'STOP' }],
      }])
    },
  })
  expect(client).toBeInstanceOf(GeminiClient)
  await collect(client.stream(simpleRequest()))
  expect(endpoint).toBe([
    'https://generativelanguage.googleapis.com/v1beta/',
    'models/gemini-2.0-flash:streamGenerateContent?alt=sse',
  ].join(''))
})

test('Gemini native completion sends generateContent and normalizes a complete candidate', async () => {
  let endpoint = ''
  let payload: Record<string, unknown> | undefined
  const client = new GeminiClient({
    apiKey: 'test-key',
    baseUrl: 'https://gemini.test/v1beta',
    fetchImplementation: async (input, init) => {
      endpoint = String(input)
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return Response.json({
        candidates: [{
          content: {
            parts: [
              { text: 'Inspect source.', thought: true, thoughtSignature: 'thought-1' },
              { text: 'I will read it.' },
              { functionCall: { id: 'call-1', name: 'ReadFile', args: { path: 'README.md' } } },
            ],
          },
          finishReason: 'STOP',
        }],
        usageMetadata: {
          promptTokenCount: 12,
          candidatesTokenCount: 6,
          cachedContentTokenCount: 2,
          thoughtsTokenCount: 3,
        },
      })
    },
  })

  const completion = await client.complete({
    model: 'gemini-2.0-flash',
    messages: [{ role: 'user', content: 'Read the README.' }],
  })

  expect(endpoint).toBe('https://gemini.test/v1beta/models/gemini-2.0-flash:generateContent')
  expect(payload).toEqual({ contents: [{ role: 'user', parts: [{ text: 'Read the README.' }] }] })
  expect(completion).toEqual({
    content: 'I will read it.',
    thinking: 'Inspect source.',
    thinkingSignature: 'thought-1',
    finishReason: 'stop',
    usage: { inputTokens: 12, outputTokens: 6, cacheReadTokens: 2, reasoningTokens: 3 },
    toolCalls: [{ id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
  })
})

test('Gemini client exposes HTTP failures and malformed SSE JSON as provider errors', async () => {
  const unavailable = new GeminiClient({
    apiKey: 'test-key',
    fetchImplementation: async () => new Response('quota exhausted', { status: 429, statusText: 'Too Many Requests' }),
  })
  await expect(collect(unavailable.stream(simpleRequest()))).rejects.toThrow(
    'stream request failed (429 Too Many Requests): quota exhausted',
  )

  const malformed = new GeminiClient({
    apiKey: 'test-key',
    fetchImplementation: async () => textSseResponse('data: {not JSON}\n\n'),
  })
  await expect(collect(malformed.stream(simpleRequest()))).rejects.toThrow('invalid Gemini SSE JSON: {not JSON}')
})

test('Gemini fetch aborts surface unchanged instead of being wrapped as provider errors', async () => {
  const abort = new DOMException('The operation was aborted', 'AbortError')
  const client = new GeminiClient({
    apiKey: 'test-key',
    baseUrl: 'https://gemini.test/v1beta',
    fetchImplementation: async () => {
      throw abort
    },
  })

  await expect(client.complete(simpleRequest())).rejects.toBe(abort)
  await expect(collect(client.stream(simpleRequest()))).rejects.toBe(abort)

  const failure = new TypeError('fetch failed')
  const failing = new GeminiClient({
    apiKey: 'test-key',
    baseUrl: 'https://gemini.test/v1beta',
    fetchImplementation: async () => {
      throw failure
    },
  })
  const wrapped = await failing.complete(simpleRequest()).catch((error: unknown) => error)
  expect(wrapped).toBeInstanceOf(ProviderError)
  expect((wrapped as ProviderError).cause).toBe(failure)
})

test('Gemini conversion rejects tool replies with no resolvable function name', () => {
  expect(() => messagesToGemini([
    { role: 'user', content: 'run a tool' },
    { role: 'tool', tool_call_id: 'unknown-call', content: 'result' },
  ])).toThrow('Gemini tool response unknown-call is missing its function name')
})

test('Gemini stream keeps duplicate identical function calls with distinct ids', async () => {
  const client = new GeminiClient({
    apiKey: 'test-key',
    baseUrl: 'https://gemini.test/v1beta',
    fetchImplementation: async () => sseResponse([
      {
        candidates: [{
          content: { parts: [{ functionCall: { name: 'ReadFile', args: { path: 'README.md' } } }] },
        }],
      },
      {
        candidates: [{
          content: { parts: [{ functionCall: { name: 'ReadFile', args: { path: 'README.md' } } }] },
          finishReason: 'STOP',
        }],
      },
    ]),
  })

  const events = await collect(client.stream(simpleRequest()))
  const toolCalls = (events as { toolCalls?: { id: string; function: { name: string } }[] }[])
    .flatMap(event => event.toolCalls ?? [])

  expect(toolCalls).toHaveLength(2)
  expect(toolCalls[0]?.id).not.toBe(toolCalls[1]?.id)
  expect(toolCalls[0]).toMatchObject({ type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } })
  expect(toolCalls[1]).toMatchObject({ type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } })
})

function simpleRequest(): CompletionRequest {
  return { model: 'gemini-2.0-flash', messages: [{ role: 'user', content: 'hello' }] }
}

async function collect(stream: AsyncIterable<unknown>): Promise<unknown[]> {
  const events: unknown[] = []
  for await (const event of stream) {
    events.push(event)
  }
  return events
}

function sseResponse(events: readonly Record<string, unknown>[], chunkSizes: readonly number[] = []): Response {
  const body = `${events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('')}data: [DONE]\n\n`
  return textSseResponse(body, chunkSizes)
}

function textSseResponse(text: string, chunkSizes: readonly number[] = []): Response {
  const bytes = new TextEncoder().encode(text)
  return new Response(new ReadableStream({
    start(controller) {
      let offset = 0
      let index = 0
      while (offset < bytes.byteLength) {
        const size = chunkSizes[index] ?? bytes.byteLength
        controller.enqueue(bytes.slice(offset, offset + size))
        offset += size
        index += 1
      }
      controller.close()
    },
  }), { headers: { 'Content-Type': 'text/event-stream' } })
}
