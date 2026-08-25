// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'

async function collectLive(
  turn: AsyncIterable<StreamEvent>,
  onEvent?: (event: StreamEvent) => void,
): Promise<StreamEvent[]> {
  const events: StreamEvent[] = []
  for await (const event of turn) {
    events.push(event)
    onEvent?.(event)
  }
  return events
}

test('a slow multi-chunk stream delivers text before the provider round completes', async () => {
  let releaseSecondChunk!: () => void
  const secondChunk = new Promise<void>(resolve => {
    releaseSecondChunk = resolve
  })

  class GatedSlowClient implements LlmClient {
    async *stream(): AsyncGenerator<LlmDelta> {
      yield { content: 'first half ' }
      // The round cannot complete until the consumer reacts to the first
      // chunk, so buffering whole rounds inside runTurn would deadlock this
      // exchange instead of failing an assertion.
      await secondChunk
      yield { content: 'second half', usage: { inputTokens: 2, outputTokens: 1 } }
    }
  }

  const events: StreamEvent[] = []
  let watchdog: ReturnType<typeof setTimeout> | undefined
  try {
    await Promise.race([
      collectLive(runTurn(
        { model: 'gpt-4o', state: createAgentState(), userMessage: 'write something' },
        { llm: new GatedSlowClient(), retryDelays: [] },
      ), event => {
        events.push(event)
        if (event.type === 'text' && event.text.startsWith('first half')) {
          releaseSecondChunk()
        }
      }),
      new Promise<never>((_, reject) => {
        watchdog = setTimeout(
          () => reject(new Error('no text event arrived before the provider round completed')),
          5_000,
        )
      }),
    ])
  } finally {
    clearTimeout(watchdog)
  }
  // The gate proves the first chunk was yielded while the provider was still
  // mid-round; these assertions pin the full delivered payload.
  expect(events.filter(event => event.type === 'text').map(event => event.text).join('')).toBe(
    'first half second half',
  )
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'completed' })
})

test('cancelling mid-stream keeps the partial text received so far and ends aborted', async () => {
  const controller = new AbortController()

  class CancelsMidStreamClient implements LlmClient {
    async *stream(_request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
      yield { content: 'partial before cancel' }
      await new Promise<void>(resolve => {
        signal?.addEventListener('abort', () => resolve(), { once: true })
      })
      // Mirror a real transport: the aborted fetch surfaces as AbortError.
      throw new DOMException('The operation was aborted', 'AbortError')
    }
  }

  const state = createAgentState()
  const events = await collectLive(runTurn(
    { model: 'gpt-4o', state, userMessage: 'start writing' },
    { llm: new CancelsMidStreamClient(), retryDelays: [] },
    controller.signal,
  ), event => {
    if (event.type === 'text') controller.abort(new Error('user hit escape'))
  })

  // The partial text streamed before the cancel stays delivered...
  expect(events.filter(event => event.type === 'text').map(event => event.text)).toEqual([
    'partial before cancel',
  ])
  // ...with no fabricated error output, and cancellation is not misreported
  // as a provider failure downstream.
  expect(events.some(event => event.type === 'text' && event.text.startsWith('[Error:'))).toBeFalse()
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'aborted' })
  // The interrupted round persists nothing as assistant content.
  expect(state.messages.filter(message => message.role === 'assistant')).toEqual([])
  // The terminal provider_retry still records why the attempt sequence ended.
  expect(events.filter(event => event.type === 'provider_retry')).toEqual([
    expect.objectContaining({ final: true, error: 'user hit escape' }),
  ])
})

test('an abort during retry backoff reports aborted without synthetic error text', async () => {
  class TransientClient implements LlmClient {
    calls = 0

    async *stream(): AsyncGenerator<LlmDelta> {
      this.calls += 1
      throw new Error('transient connection drop')
    }
  }

  const controller = new AbortController()
  const client = new TransientClient()
  const state = createAgentState()
  const events = await collectLive(runTurn(
    { model: 'gpt-4o', state, userMessage: 'retry then abort' },
    {
      delay: async (_milliseconds, signal) => {
        controller.abort(new Error('user interrupt during backoff'))
        return new Promise<void>((_, reject) => {
          if (signal?.aborted) {
            reject(signal.reason)
            return
          }
          signal?.addEventListener('abort', () => reject(signal.reason), { once: true })
        })
      },
      llm: client,
      retryDelays: [5_000],
    },
    controller.signal,
  ))

  expect(client.calls).toBe(1)
  expect(events.at(-1)).toMatchObject({
    type: 'turn_done',
    reason: 'aborted',
    apiCallsCount: 1,
  })
  expect(events.some(event => event.type === 'text' && event.text.startsWith('[Error:'))).toBeFalse()
  expect(events.filter(event => event.type === 'turn_done')).toHaveLength(1)
  expect(events.filter(event => event.type === 'provider_retry').at(-1)).toMatchObject({
    attempt: 2,
    final: true,
    error: 'user interrupt during backoff',
  })
})

test('a replayed cross-tool-round prefix stays suppressed while streaming diverging chunks live', async () => {
  class ReplayThenDivergeClient implements LlmClient {
    private calls = 0

    async *stream(): AsyncGenerator<LlmDelta> {
      this.calls += 1
      if (this.calls === 1) {
        yield { content: 'Sentinel prefix stays hidden.' }
        yield {
          toolCalls: [{
            id: 'call-replay-live',
            type: 'function',
            function: { name: 'ReadFile', arguments: { path: 'a.ts' } },
          }],
        }
        return
      }
      // The provider replays the previous round's text verbatim, chunked
      // differently, then diverges mid-stream.
      yield { content: 'Sentinel' }
      yield { content: ' prefix stays' }
      yield { content: ' hidden. Now something new.' }
    }
  }

  const registryLikeExecutor = {
    execute: async (): Promise<string> => 'file body',
  }
  const events = await collectLive(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state: createAgentState(),
    userMessage: 'inspect twice',
    tools: [{
      type: 'function',
      function: { name: 'ReadFile', description: 'Read a file.', parameters: {} },
    }],
  }, {
    llm: new ReplayThenDivergeClient(),
    toolExecutor: registryLikeExecutor,
  }))

  // The replayed prefix is emitted exactly once (round one); the second round
  // contributes only its diverging tail.
  expect(events.filter(event => event.type === 'text').map(event => event.text).join('')).toBe(
    'Sentinel prefix stays hidden. Now something new.',
  )
})
