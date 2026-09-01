// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ProviderError } from '../src/core/errors.js'
import {
  CompletionDeadlineError,
  OpenAiCompatibleClient,
  collectLlmCompletion,
  completeLlm,
  type CompletionRequest,
  type LlmClient,
  type LlmDelta,
} from '../src/llms/client.js'
import { classifyError } from '../src/runtime/errorClassifier.js'
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

test('chat-completions usage reports cached prompt tokens apart from fresh input tokens', async () => {
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
    usage: { inputTokens: 8, outputTokens: 5, cacheReadTokens: 12, reasoningTokens: 3 },
  })

  const completing = openAiClient(async () => Response.json({
    choices: [{ finish_reason: 'stop', message: { content: 'hi' } }],
    usage,
  }))
  const completion = await completing.complete(request())
  expect(completion.usage).toEqual({ inputTokens: 8, outputTokens: 5, cacheReadTokens: 12, reasoningTokens: 3 })
})

test('chat-completions normalizes GLM cache-hit usage fields', async () => {
  const usage = {
    prompt_tokens: 10_000,
    completion_tokens: 50,
    prompt_cache_hit_tokens: 9_700,
    prompt_cache_miss_tokens: 300,
  }
  const streaming = openAiClient(async () => sseResponse([
    { choices: [{ delta: { content: 'hi' } }] },
    { choices: [{ delta: {}, finish_reason: 'stop' }], usage },
  ]))
  expect(await collect(streaming.stream(request()))).toContainEqual({
    finishReason: 'stop',
    usage: { inputTokens: 300, outputTokens: 50, cacheReadTokens: 9_700 },
  })

  const completing = openAiClient(async () => Response.json({
    choices: [{ finish_reason: 'stop', message: { content: 'hi' } }],
    usage,
  }))
  expect((await completing.complete(request())).usage).toEqual({
    inputTokens: 300,
    outputTokens: 50,
    cacheReadTokens: 9_700,
  })
})

test('chat-completions mirrors Pi cache placement and cache-write accounting', async () => {
  const client = openAiClient(async () => Response.json({
    choices: [{ finish_reason: 'stop', message: { content: 'done' } }],
    usage: {
      prompt_tokens: 1_000,
      completion_tokens: 20,
      cached_tokens: 700,
      prompt_tokens_details: { cache_write_tokens: 100 },
    },
  }))
  expect((await client.complete(request())).usage).toEqual({
    inputTokens: 200,
    outputTokens: 20,
    cacheReadTokens: 700,
    cacheCreationTokens: 100,
  })
})

test('chat-completions propagates Retry-After delta-seconds as structured classifier metadata', async () => {
  for (const operation of ['complete', 'stream'] as const) {
    const client = openAiClient(async () => new Response('slow down', {
      status: 429,
      headers: { 'Retry-After': '2.5' },
    }))
    const failure = await (operation === 'complete'
      ? client.complete(request())
      : collect(client.stream(request()))).catch(error => error as unknown)

    expect(failure).toBeInstanceOf(ProviderError)
    expect((failure as ProviderError).details).toMatchObject({ status: 429, retryAfterSeconds: 2.5 })
    expect(classifyError(failure)).toMatchObject({
      kind: 'rate_limit',
      retryable: true,
      suggestedBackoffSeconds: 2.5,
    })
  }
})

test('chat-completions parses Retry-After HTTP dates and ignores malformed values', async () => {
  const retryAt = new Date(Date.now() + 30_000).toUTCString()
  const dated = openAiClient(async () => new Response('unavailable', {
    status: 503,
    headers: { 'Retry-After': retryAt },
  }))
  const datedFailure = await dated.complete(request()).then(
    () => { throw new Error('expected request to fail') },
    error => error as ProviderError,
  )
  expect(datedFailure.details.status).toBe(503)
  const datedBackoff = classifyError(datedFailure).suggestedBackoffSeconds
  expect(datedBackoff).toBeGreaterThanOrEqual(28)
  expect(datedBackoff).toBeLessThanOrEqual(30)

  const malformed = openAiClient(async () => new Response('slow down', {
    status: 429,
    headers: { 'Retry-After': 'next Tuesday-ish' },
  }))
  const malformedFailure = await collect(malformed.stream(request())).then(
    () => { throw new Error('expected request to fail') },
    error => error as ProviderError,
  )
  expect(malformedFailure.details).toEqual({ status: 429 })
  expect(classifyError(malformedFailure).suggestedBackoffSeconds).toBeUndefined()
})

test('chat-completions SSE requires an explicit terminal finish event', async () => {
  const encoder = new TextEncoder()
  const client = openAiClient(async () => new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'))
      controller.close()
    },
  })))

  await expect(collect(client.stream(request()))).rejects.toThrow(
    'stream ended before a terminal completion event',
  )
})

/** A client whose completion never settles, ignoring whatever signal it is handed. */
function hungClient(observed: { signal?: AbortSignal | undefined }): LlmClient {
  return {
    stream() {
      throw new Error('unused by these tests')
    },
    complete: (_request, signal) => {
      observed.signal = signal
      return new Promise(() => {})
    },
  }
}

test('completeLlm enforces its deadline even when the client ignores the abort signal', async () => {
  // A stalled upstream is exactly the failure that may never observe the
  // signal, so the caller must still be released promptly.
  const observed: { signal?: AbortSignal } = {}
  const started = Date.now()

  await expect(completeLlm(hungClient(observed), request(), undefined, { timeoutMs: 25 }))
    .rejects.toBeInstanceOf(CompletionDeadlineError)
  expect(Date.now() - started).toBeLessThan(2_000)
  // The transport was asked to stop through the combined signal as well.
  expect(observed.signal?.aborted).toBeTrue()
})

test('completeLlm applies the same deadline when collecting a stream-only client', async () => {
  const client: LlmClient = {
    stream: async function* (): AsyncGenerator<LlmDelta> {
      await new Promise<never>(() => {})
      yield {}
    },
  }

  const failure = await completeLlm(client, request(), undefined, { timeoutMs: 25 }).then(
    () => { throw new Error('expected deadline rejection') },
    error => error,
  )
  expect(failure).toBeInstanceOf(CompletionDeadlineError)
  expect((failure as CompletionDeadlineError).timeoutMs).toBe(25)
})

test('a caller abort is honored immediately and independently of the deadline', async () => {
  const controller = new AbortController()
  const observed: { signal?: AbortSignal } = {}
  // completeLlm wires its abort wiring synchronously before suspending, so
  // aborting right after starting the call races nothing.
  const pending = completeLlm(hungClient(observed), request(), controller.signal, { timeoutMs: 60_000 })
  controller.abort()

  const reason = await pending.then(
    () => { throw new Error('expected the call to reject') },
    error => error,
  )
  // The deadline was far away; only the caller's signal can have stopped this.
  expect((reason as Error).name).toBe('AbortError')
  expect(observed.signal?.aborted).toBeTrue()
})
