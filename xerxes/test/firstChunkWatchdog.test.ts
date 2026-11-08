// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import {
  DEFAULT_STREAM_INACTIVITY_TIMEOUT_MS,
  FIRST_CHUNK_MS_PER_PROMPT_TOKEN,
  MAX_FIRST_CHUNK_TIMEOUT_MS,
  firstChunkTimeoutMs,
  runTurn,
} from '../src/streaming/loop.js'

async function collect(events: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const result: StreamEvent[] = []
  for await (const event of events) result.push(event)
  return result
}

const sleep = (ms: number, signal?: AbortSignal) => new Promise<void>((resolve, reject) => {
  const timer = setTimeout(resolve, ms)
  signal?.addEventListener('abort', () => { clearTimeout(timer); reject(signal.reason) }, { once: true })
})

test('the first-chunk budget grows with the prompt, within bounds', () => {
  // 11,900 tokens read at ~70 tokens a second took ~170 s: past the 2-minute
  // inactivity budget, well inside the first-chunk budget.
  expect(firstChunkTimeoutMs(DEFAULT_STREAM_INACTIVITY_TIMEOUT_MS, 11_900)).toBe(11_900 * FIRST_CHUNK_MS_PER_PROMPT_TOKEN)
  expect(firstChunkTimeoutMs(DEFAULT_STREAM_INACTIVITY_TIMEOUT_MS, 500)).toBe(DEFAULT_STREAM_INACTIVITY_TIMEOUT_MS)
  expect(firstChunkTimeoutMs(DEFAULT_STREAM_INACTIVITY_TIMEOUT_MS, 10_000_000)).toBe(MAX_FIRST_CHUNK_TIMEOUT_MS)
  expect(firstChunkTimeoutMs(0, 11_900)).toBe(0)
  expect(firstChunkTimeoutMs(Number.POSITIVE_INFINITY, 11_900)).toBe(Number.POSITIVE_INFINITY)
  expect(firstChunkTimeoutMs(25, Number.NaN)).toBe(25)
})

test('a slow first chunk on a long prompt is waited for instead of being sent again', async () => {
  let calls = 0
  const client: LlmClient = {
    async *stream(_request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
      calls += 1
      await sleep(150, signal) // reading the prompt
      yield { content: 'read it' }
    },
  }
  const state = createAgentState()
  // ~2,500 estimated tokens: a budget far above the 150 ms wait.
  const events = await collect(runTurn(
    { model: 'test', state, userMessage: 'x'.repeat(10_000) },
    { llm: client, retryDelays: [0], streamInactivityTimeoutMs: 40 },
  ))
  expect(calls).toBe(1)
  expect(events.some(event => event.type === 'provider_retry')).toBe(false)
  expect(state.messages.at(-1)).toMatchObject({ role: 'assistant', content: 'read it' })
})

test('after the first chunk, a stall still trips the inactivity budget', async () => {
  let calls = 0
  const client: LlmClient = {
    async *stream(_request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
      calls += 1
      if (calls === 1) {
        yield { content: 'partial' }
        await sleep(5_000, signal)
      }
      yield { content: 'second try' }
    },
  }
  const state = createAgentState()
  const events = await collect(runTurn(
    { model: 'test', state, userMessage: 'x'.repeat(10_000) },
    { llm: client, retryDelays: [0], streamInactivityTimeoutMs: 40 },
  ))
  expect(calls).toBe(2)
  const retry = events.find(event => event.type === 'provider_retry') as { error?: string } | undefined
  expect(String(retry?.error)).toContain('within 40ms')
  expect(state.messages.at(-1)).toMatchObject({ role: 'assistant', content: 'second try' })
})
