// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createHash } from 'node:crypto'

import { ProviderError, StreamTruncatedError } from '../src/core/errors.js'
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
import { reportedModelCapability } from '../src/llms/modelsDev.js'
import type { ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

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

test('Kimi chat-completions aliases invalid names consistently and restores completed and streamed calls', async () => {
  const dottedName = 'mcp.tools/search'
  // A legitimate tool may occupy the first deterministic alias candidate.
  const reservedAlias = 'xerxes_' + createHash('sha256').update(dottedName).digest('hex').slice(0, 48)
  const names = [dottedName, 'mcp_tools_search', 'tool with spaces', '工具', '_private', '1tool', '-tool', 'x'.repeat(65), 'Valid-tool_2', reservedAlias]
  const historyOnly = 'retired.tool'
  const resultOnly = 'result.only'
  const tools: ToolDefinition[] = names.map(name => ({
    type: 'function',
    function: { name, description: 'tool', parameters: { type: 'object', properties: {} } },
  }))
  const input: CompletionRequest = {
    model: 'kimi-for-coding',
    tools,
    messages: [
      { role: 'assistant', content: '', tool_calls: [
        { id: 'previous', type: 'function', function: { name: dottedName, arguments: {} } },
        { id: 'retired', type: 'function', function: { name: historyOnly, arguments: {} } },
      ] },
      { role: 'tool', name: dottedName, tool_call_id: 'previous', content: 'done' },
      { role: 'tool', name: resultOnly, tool_call_id: 'retired', content: 'done' },
    ],
  }
  const before = JSON.stringify(input)
  for (const providerName of ['kimi', 'kimi-code', 'moonshotai', 'moonshotai-cn'] as const) {
    for (const operation of ['complete', 'stream', 'stream-done'] as const) {
      let previousNames: (string | undefined)[] | undefined
      const client = new OpenAiCompatibleClient({
        providerName,
        apiKey: 'test',
        baseUrl: 'https://example.invalid/v1',
        fetchImplementation: async (_url, init) => {
          const payload = JSON.parse(String(init?.body)) as {
            tools: { function: { name: string } }[]
            messages: { name?: string; tool_calls?: { function: { name: string } }[] }[]
          }
          const wireNames = payload.tools.map(tool => tool.function.name)
          expect(new Set(wireNames).size).toBe(names.length)
          for (const name of wireNames) expect(name).toMatch(/^[a-zA-Z][a-zA-Z0-9_-]{0,63}$/)
          for (const [index, name] of names.entries()) {
            if (/^[a-zA-Z][a-zA-Z0-9_-]{0,63}$/.test(name)) expect(wireNames[index]).toBe(name)
          }
          expect(wireNames[0]).not.toBe(reservedAlias)
          expect(payload.messages[0]?.tool_calls?.[0]?.function.name).toBe(wireNames[0])
          expect(payload.messages[1]?.name).toBe(wireNames[0])
          const retired = payload.messages[0]?.tool_calls?.[1]?.function.name
          const result = payload.messages[2]?.name
          expect(retired).toMatch(/^[a-zA-Z][a-zA-Z0-9_-]{0,63}$/)
          expect(result).toMatch(/^[a-zA-Z][a-zA-Z0-9_-]{0,63}$/)
          const allNames = [...wireNames, retired, result]
          expect(new Set(allNames).size).toBe(names.length + 2)
          if (previousNames) expect(allNames).toEqual(previousNames)
          previousNames = allNames
          const calls = [wireNames[0], retired, result, 'Valid-tool_2'].map((name, index) => ({
            index,
            id: `call-${index}`,
            type: 'function',
            function: { name, arguments: '{}' },
          }))
          return operation === 'complete'
            ? Response.json({ choices: [{ message: { content: '', tool_calls: calls }, finish_reason: 'tool_calls' }] })
            : sseResponse([
              { choices: [{ delta: { tool_calls: calls.map(call => ({ ...call, function: { ...call.function, arguments: '{' } })) } }] },
              { choices: [{ delta: { tool_calls: calls.map(call => ({ index: call.index, function: { arguments: '}' } })) } }] },
              ...(operation === 'stream' ? [{ choices: [{ delta: {}, finish_reason: 'tool_calls' }] }] : []),
            ])
        },
      })
      // A retry must produce the identical aliases without mutating the transcript.
      for (let retry = 0; retry < 2; retry += 1) {
        const completion = operation === 'complete'
          ? await client.complete(input)
          : await collectLlmCompletion(client.stream(input))
        expect(completion.toolCalls.map(call => call.function.name)).toEqual([dottedName, historyOnly, resultOnly, 'Valid-tool_2'])
      }
    }
  }
  expect(JSON.stringify(input)).toBe(before)
})

test('non-Kimi chat-completions leaves tool names unchanged', async () => {
  const name = 'mcp.tools/search'
  const client = openAiClient(async (_url, init) => {
    const payload = JSON.parse(String(init?.body)) as {
      tools: { function: { name: string } }[]
      messages: { name?: string; tool_calls?: { function: { name: string } }[] }[]
    }
    expect(payload.tools[0]?.function.name).toBe(name)
    expect(payload.messages[0]?.tool_calls?.[0]?.function.name).toBe(name)
    expect(payload.messages[1]?.name).toBe(name)
    return Response.json({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] })
  })
  await client.complete({
    ...request(),
    tools: [{ type: 'function', function: { name, description: 'tool', parameters: {} } }],
    messages: [
      { role: 'assistant', content: '', tool_calls: [{ id: 'old', type: 'function', function: { name, arguments: {} } }] },
      { role: 'tool', tool_call_id: 'old', name, content: 'ok' },
    ],
  })
})

test('Kimi usage keeps cache reads, writes and uncached input disjoint without masking top-level reads', async () => {
  const usage = {
    prompt_tokens: 1_000,
    completion_tokens: 20,
    cached_tokens: 700,
    prompt_tokens_details: { cached_tokens: 0, cache_write_tokens: 100 },
  }
  for (const operation of ['complete', 'stream'] as const) {
    const client = new OpenAiCompatibleClient({
      providerName: 'kimi-code',
      apiKey: 'test',
      baseUrl: 'https://example.invalid/v1',
      fetchImplementation: async () => operation === 'complete'
        ? Response.json({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }], usage })
        : sseResponse([
          { choices: [{ delta: { content: 'ok' }, finish_reason: 'stop' }] },
          { choices: [], usage },
        ]),
    })
    const completion = operation === 'complete'
      ? await client.complete(request())
      : await collectLlmCompletion(client.stream(request()))
    expect(completion.usage).toEqual({ inputTokens: 200, outputTokens: 20, cacheReadTokens: 700, cacheCreationTokens: 100 })
  }
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

test('a size refusal that states the window is an overflow and records that window for the model', async () => {
  const refusal = JSON.stringify({ error: { message: 'Invalid request: Your request exceeded model token limit: 1048576 (requested: 1232443)', type: 'invalid_request_error' } })
  for (const operation of ['complete', 'stream'] as const) {
    // A model id of its own, so the shared capability map stays clean for other tests.
    const model = `window-probe-${operation}`
    const client = openAiClient(async () => new Response(refusal, { status: 400 }))
    const input: CompletionRequest = { model, messages: [{ role: 'user', content: 'hi' }] }
    const failure = await (operation === 'complete' ? client.complete(input) : collect(client.stream(input))).catch(error => error as unknown)
    expect(failure).toBeInstanceOf(ProviderError)
    expect(classifyError(failure)).toMatchObject({ kind: 'context_overflow' })
    expect(reportedModelCapability('openai', model)?.contextLimit).toBe(1_048_576)
  }
  // A plain bad request records nothing.
  const plain = openAiClient(async () => new Response('{"error":{"message":"max_tokens must be at most 32768"}}', { status: 400 }))
  await plain.complete({ model: 'window-probe-plain', messages: [{ role: 'user', content: 'hi' }] }).catch(() => undefined)
  expect(reportedModelCapability('openai', 'window-probe-plain')).toBeUndefined()
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

test('a stream cut short before its terminal event is a retryable transport fault', async () => {
  const encoder = new TextEncoder()
  const client = openAiClient(async () => new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'))
      controller.close()
    },
  })))

  const failure = await collect(client.stream(request())).then(() => undefined, (error: unknown) => error)

  expect(failure).toBeInstanceOf(StreamTruncatedError)
  expect(classifyError(failure)).toMatchObject({ kind: 'transient', retryable: true })
  // Goal rounds classify the failure from its message text alone.
  expect(classifyError(new Error((failure as Error).message)).retryable).toBe(true)
})

test('documented provider finish reasons map onto the neutral vocabulary instead of failing the reply', async () => {
  const finishWith = (reason: string) => openAiClient(async () => sseResponse([
    { choices: [{ delta: { content: 'Hello' } }] },
    { choices: [{ delta: { content: ' world' }, finish_reason: reason }] },
  ]))

  // Together ends a finished completion with `eos`.
  expect(await collect(finishWith('eos').stream(request()))).toEqual([
    { content: 'Hello' },
    { content: ' world', finishReason: 'stop' },
  ])
  expect(await collect(finishWith('model_context_window_exceeded').stream(request()))).toContainEqual({
    content: ' world',
    finishReason: 'length',
  })

  // A server that gave up mid-reply asks for a retry.
  for (const reason of ['insufficient_system_resource', 'network_error']) {
    const failure = await collect(finishWith(reason).stream(request())).then(() => undefined, (error: unknown) => error)
    expect(classifyError(failure).retryable).toBe(true)
    expect(classifyError(new Error((failure as Error).message)).retryable).toBe(true)
  }

  // A refusal stays a failure.
  await expect(collect(finishWith('content_filter').stream(request()))).rejects.toThrow('provider finish_reason: content_filter')
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
