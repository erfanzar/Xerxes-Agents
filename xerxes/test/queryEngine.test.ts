// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
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

test('query engine applies an injected context compressor at its configured turn boundary', async () => {
  const engine = new QueryEngine({ llm: new ReplyClient() }, {
    config: { compactAfterTurns: 2 },
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
