// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ConfigurationError,
  ContextCompressor,
  CostTracker,
  QueryEngine,
  naiveSummarizer,
  type CompletionRequest,
  type LlmClient,
  type LlmDelta,
} from '../src/index.js'

class ReplyClient implements LlmClient {
  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    yield { content: 'A deliberately useful assistant response.' }
  }
}

class UsageClient implements LlmClient {
  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    yield {
      content: 'A priced response.',
      usage: {
        inputTokens: 1_000,
        outputTokens: 500,
        cacheReadTokens: 100,
        cacheCreationTokens: 20,
      },
    }
  }
}

function deferred(): { readonly promise: Promise<void>; readonly resolve: () => void } {
  let resolve!: () => void
  const promise = new Promise<void>(done => {
    resolve = done
  })
  return { promise, resolve }
}

class OverlapClient implements LlmClient {
  readonly firstStarted = deferred()
  readonly releaseFirst = deferred()
  calls = 0

  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.calls += 1
    const call = this.calls
    if (call === 1) {
      this.firstStarted.resolve()
      await this.releaseFirst.promise
    }
    yield {
      content: `response-${call}`,
      usage: { inputTokens: call, outputTokens: call },
    }
  }
}

test('query engine applies an injected context compressor at its configured turn boundary', async () => {
  const engine = new QueryEngine({ llm: new ReplyClient() }, {
    config: { compactAfterTurns: 2, model: 'configured-test-model' },
    contextCompressor: new ContextCompressor({
      contextWindow: 1,
      threshold: 0.5,
      protectFirst: 1,
      protectLast: 1,
      summaryMinTokens: 1,
      summarizer: naiveSummarizer,
    }),
  })
  await engine.submit('first context')
  await engine.submit('second context')

  expect(engine.config.permissionMode).toBe('accept-all')
  expect(engine.state.messages).toHaveLength(3)
  expect(engine.state.messages[1]).toMatchObject({ role: 'user', content: expect.stringContaining('CONTEXT COMPACTION') })
  expect(engine.state.metadata.lastCompaction).toMatchObject({ strategy: 'first-pass', compressed_count: 2 })
})

test('query engine rejects execution without an explicitly configured model', () => {
  expect(() => new QueryEngine({ llm: new ReplyClient() }))
    .toThrow(ConfigurationError)
  expect(() => new QueryEngine({ llm: new ReplyClient() }))
    .toThrow('select a provider model or pass an explicit model')
})

test('query engine serializes overlapping submissions and preserves cancellation while queued', async () => {
  const client = new OverlapClient()
  const tracker = new CostTracker({ now: () => new Date('2026-08-03T10:00:00.000Z') })
  const engine = new QueryEngine({ llm: client }, {
    config: { model: 'configured-test-model' },
    costTracker: tracker,
  })

  const first = engine.submit('first')
  await client.firstStarted.promise
  const second = engine.submit('second')
  const cancelled = new AbortController()
  const reason = new Error('cancel queued query')
  const third = engine.submit('third', cancelled.signal).catch(error => error as unknown)
  cancelled.abort(reason)

  await Bun.sleep(0)
  expect(client.calls).toBe(1)
  expect(engine.state.messages).toEqual([{ role: 'user', content: 'first' }])

  client.releaseFirst.resolve()
  expect(await first).toMatchObject({ prompt: 'first', output: 'response-1' })
  expect(await second).toMatchObject({ prompt: 'second', output: 'response-2' })
  expect(await third).toBe(reason)

  expect(client.calls).toBe(2)
  expect(engine.state.messages).toHaveLength(4)
  expect(engine.state.messages).toMatchObject([
    { role: 'user', content: 'first' },
    { role: 'assistant', content: 'response-1' },
    { role: 'user', content: 'second' },
    { role: 'assistant', content: 'response-2' },
  ])
  expect(tracker.events.map(event => event.label)).toEqual(['turn_1', 'turn_2'])
})

test('query engine records each completed provider turn in its session cost ledger', async () => {
  const tracker = new CostTracker({ now: () => new Date('2026-07-13T10:00:00.000Z') })
  const engine = new QueryEngine({ llm: new UsageClient() }, {
    config: { agentId: 'planner', model: 'gpt-4o' },
    costTracker: tracker,
    sessionId: 'cost-session',
  })

  const result = await engine.submit('price this response')
  expect(result).toMatchObject({ inputTokens: 1_000, outputTokens: 500 })
  expect(tracker.asRecords()).toMatchObject([{
    model: 'gpt-4o',
    in_tokens: 1_000,
    out_tokens: 500,
    cache_read_tokens: 100,
    cache_creation_tokens: 20,
    session_id: 'cost-session',
    agent_id: 'planner',
  }])
  expect(engine.totalCost).toBeCloseTo(0.0075875, 12)
})
