// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  OpenAiCompatibleClient,
  collectLlmCompletion,
  type CompletionRequest,
  type LlmDelta,
} from '../src/llms/client.js'
import type { ToolCall } from '../src/types/toolCalls.js'

function openAiClient(fetchImplementation: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>) {
  return new OpenAiCompatibleClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://api.openai.com/v1',
    fetchImplementation,
  })
}

function request(): CompletionRequest {
  return { model: 'gpt-4o', messages: [{ role: 'user', content: 'hi' }] }
}

function sseResponse(chunks: readonly Record<string, unknown>[]): Response {
  const encoder = new TextEncoder()
  return new Response(new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`))
      }
      controller.enqueue(encoder.encode('data: [DONE]\n\n'))
      controller.close()
    },
  }), { headers: { 'Content-Type': 'text/event-stream' } })
}

async function collect(stream: AsyncIterable<unknown>): Promise<unknown[]> {
  const events: unknown[] = []
  for await (const event of stream) {
    events.push(event)
  }
  return events
}

test('collectLlmCompletion keeps identical id-less tool calls instead of collapsing them', async () => {
  // Id-less calls receive a deterministic id derived from name+arguments, so
  // two genuinely repeated calls share an id and must both survive collection.
  const repeated: ToolCall = {
    id: 'deterministic-call-id',
    type: 'function',
    function: { name: 'ReadFile', arguments: { path: 'README.md' } },
  }
  async function* stream(): AsyncGenerator<LlmDelta> {
    yield { toolCalls: [repeated, repeated] }
    yield { content: 'done', finishReason: 'tool_calls' }
  }

  const completion = await collectLlmCompletion(stream())

  expect(completion.toolCalls).toHaveLength(2)
  expect(completion.toolCalls[0]).toEqual(repeated)
  expect(completion.toolCalls[1]).toEqual(repeated)
  expect(completion.content).toBe('done')
})

test('index-less tool deltas carrying a new id or name start a new call instead of merging', async () => {
  const client = openAiClient(async () => sseResponse([
    {
      choices: [{
        delta: { tool_calls: [{ id: 'call-1', function: { name: 'ReadFile', arguments: '{"path":"a.md"}' } }] },
      }],
    },
    // A provider streaming parallel calls without `index` used to merge this
    // into the first entry, corrupting arguments and losing the name.
    {
      choices: [{
        delta: { tool_calls: [{ id: 'call-2', function: { name: 'WriteFile', arguments: '{"path":"b.md"}' } }] },
      }],
    },
    { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
  ]))

  const events = await collect(client.stream(request()))
  const toolCalls = (events as { toolCalls?: readonly ToolCall[] }[]).flatMap(event => event.toolCalls ?? [])

  expect(toolCalls).toEqual([
    { id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'a.md' } } },
    { id: 'call-2', type: 'function', function: { name: 'WriteFile', arguments: { path: 'b.md' } } },
  ])
})

test('index-less continuation deltas without id or name still append to the current call', async () => {
  const client = openAiClient(async () => sseResponse([
    {
      choices: [{
        delta: { tool_calls: [{ id: 'call-1', function: { name: 'ReadFile', arguments: '{"path"' } }] },
      }],
    },
    { choices: [{ delta: { tool_calls: [{ function: { arguments: ':"a.md"}' } }] } }] },
    { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
  ]))

  const events = await collect(client.stream(request()))
  const toolCalls = (events as { toolCalls?: readonly ToolCall[] }[]).flatMap(event => event.toolCalls ?? [])

  expect(toolCalls).toEqual([
    { id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'a.md' } } },
  ])
})

test('chat-completions usage maps prompt_tokens_details.cached_tokens to cacheReadTokens', async () => {
  const usage = {
    prompt_tokens: 20,
    completion_tokens: 5,
    prompt_tokens_details: { cached_tokens: 12 },
    completion_tokens_details: { reasoning_tokens: 3 },
  }
  const streaming = openAiClient(async () => sseResponse([
    { choices: [{ delta: { content: 'hi' } }] },
    { choices: [{ delta: {}, finish_reason: 'stop' }], usage },
  ]))
  const events = await collect(streaming.stream(request()))
  expect(events).toContainEqual({
    finishReason: 'stop',
    usage: { inputTokens: 20, outputTokens: 5, cacheReadTokens: 12, reasoningTokens: 3 },
  })

  const completing = openAiClient(async () => Response.json({
    choices: [{ finish_reason: 'stop', message: { content: 'hi' } }],
    usage,
  }))
  const completion = await completing.complete(request())
  expect(completion.usage).toEqual({ inputTokens: 20, outputTokens: 5, cacheReadTokens: 12, reasoningTokens: 3 })
})
