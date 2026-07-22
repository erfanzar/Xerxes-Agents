// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { HookRunner } from '../src/extensions/hooks.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { MAX_UNCONFIGURED_ONLY_ROUNDS, runTurn } from '../src/streaming/loop.js'
import type { ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

const READ_FILE: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ReadFile',
    description: 'Read a file.',
    parameters: { type: 'object', properties: { path: { type: 'string' } } },
  },
}

function hiddenCall(id: string): ToolCall {
  return {
    id,
    type: 'function',
    function: { name: 'hidden_tool', arguments: {} },
  }
}

async function collect(events: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const result: StreamEvent[] = []
  for await (const event of events) result.push(event)
  return result
}

/** Mirror of the loop's default backoff: rejects immediately on an aborted signal. */
function defaultAbortableDelay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason)
      return
    }
    const timer = setTimeout(resolve, milliseconds)
    signal?.addEventListener('abort', () => {
      clearTimeout(timer)
      reject(signal.reason)
    }, { once: true })
  })
}

test('an abort during retry backoff still completes the epilogue with exactly one turn_done', async () => {
  class TransientClient implements LlmClient {
    calls = 0

    async *stream(): AsyncGenerator<LlmDelta> {
      this.calls += 1
      throw new Error('transient connection drop')
    }
  }

  const controller = new AbortController()
  const hookRunner = new HookRunner()
  const hooks: string[] = []
  hookRunner.register('on_turn_end', () => {
    hooks.push('on_turn_end')
  })
  const client = new TransientClient()
  const state = createAgentState()
  const events = await collect(runTurn(
    { model: 'gpt-4o', state, userMessage: 'retry then abort' },
    {
      delay: async (milliseconds, signal) => {
        controller.abort(new Error('user interrupt during backoff'))
        return defaultAbortableDelay(milliseconds, signal)
      },
      hookRunner,
      llm: client,
      retryDelays: [5_000],
    },
    controller.signal,
  ))

  expect(client.calls).toBe(1)
  const retries = events.filter(event => event.type === 'provider_retry')
  expect(retries).toEqual([
    expect.objectContaining({ attempt: 1, final: false }),
    expect.objectContaining({ attempt: 2, final: true, error: 'user interrupt during backoff' }),
  ])
  expect(events.filter(event => event.type === 'turn_done')).toHaveLength(1)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', apiCallsCount: 1 })
  expect(hooks).toEqual(['on_turn_end'])
  expect(state.totalApiCalls).toBe(1)
  expect(state.usageComplete).toBe(false)
})

test('a rejecting permission broker still ends the turn with exactly one turn_done', async () => {
  let executions = 0
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'manual',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'read the file',
  }, {
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        yield {
          toolCalls: [{
            id: 'call-broker',
            type: 'function',
            function: { name: 'ReadFile', arguments: { path: 'a.ts' } },
          }],
        }
      },
    },
    permissionBroker: {
      async request(): Promise<'approve' | 'reject'> {
        throw new Error('broker transport lost')
      },
    },
    toolExecutor: {
      async execute(): Promise<string> {
        executions += 1
        return 'unreachable'
      },
    },
  }))

  expect(executions).toBe(0)
  expect(events.filter(event => event.type === 'turn_done')).toHaveLength(1)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
  expect(events).toContainEqual(expect.objectContaining({
    type: 'provider_retry',
    error: 'broker transport lost',
    final: true,
  }))
})

test('a rejecting subagent join still ends the turn with exactly one turn_done', async () => {
  const state = createAgentState()
  const events = await collect(runTurn(
    { model: 'gpt-4o', state, userMessage: 'wait for children' },
    {
      awaitAgentEvents: async () => {
        throw new Error('cohort wait failed')
      },
      llm: {
        async *stream(): AsyncGenerator<LlmDelta> {
          yield { content: 'children are running' }
        },
      },
    },
  ))

  expect(events.filter(event => event.type === 'turn_done')).toHaveLength(1)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
  expect(events).toContainEqual(expect.objectContaining({
    type: 'provider_retry',
    error: 'cohort wait failed',
    final: true,
  }))
})

test('unconfigured-only tool rounds stop with an explicit error after the consecutive-round cap', async () => {
  const requests: CompletionRequest[] = []
  const state = createAgentState()
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [READ_FILE],
    userMessage: 'finish without hidden tools',
  }, {
    llm: {
      async *stream(request): AsyncGenerator<LlmDelta> {
        requests.push(request)
        yield { toolCalls: [hiddenCall(`call-hidden-${requests.length}`)] }
      },
    },
  }))

  expect(requests).toHaveLength(MAX_UNCONFIGURED_ONLY_ROUNDS)
  expect(events.filter(event => event.type === 'tool_end')).toHaveLength(MAX_UNCONFIGURED_ONLY_ROUNDS)
  const texts = events.filter(event => event.type === 'text').map(event => event.text)
  expect(texts.at(-1)).toContain('only unconfigured tools')
  expect(events.at(-1)).toMatchObject({
    type: 'turn_done',
    toolCallsCount: MAX_UNCONFIGURED_ONLY_ROUNDS,
  })
})

test('a configured recovery round resets the consecutive unconfigured-only counter', async () => {
  let calls = 0
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'recover from hidden tools',
  }, {
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        calls += 1
        if (calls <= MAX_UNCONFIGURED_ONLY_ROUNDS - 1) {
          yield { toolCalls: [hiddenCall(`call-hidden-${calls}`)] }
          return
        }
        if (calls === MAX_UNCONFIGURED_ONLY_ROUNDS) {
          yield {
            toolCalls: [{
              id: 'call-visible',
              type: 'function',
              function: { name: 'ReadFile', arguments: { path: 'ok.ts' } },
            }],
          }
          return
        }
        yield { content: 'recovered' }
      },
    },
    toolExecutor: {
      async execute(): Promise<string> {
        return 'file body'
      },
    },
  }))

  expect(calls).toBe(MAX_UNCONFIGURED_ONLY_ROUNDS + 1)
  expect(events.filter(event => event.type === 'text').map(event => event.text)).toEqual(['recovered'])
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
})

test('a terminal provider error is not persisted as assistant content or retried by the objective guard', async () => {
  class AlwaysAuthFailureClient implements LlmClient {
    calls = 0

    async *stream(): AsyncGenerator<LlmDelta> {
      this.calls += 1
      throw new Error('stream request failed (401): invalid api key')
    }
  }

  const client = new AlwaysAuthFailureClient()
  const state = createAgentState()
  const events = await collect(runTurn(
    {
      interactionMode: 'objective',
      model: 'gpt-4o',
      state,
      userMessage: 'finish the task',
    },
    { delay: async () => undefined, llm: client, retryDelays: [0, 0] },
  ))

  // The objective guard must not re-call a terminally failed provider.
  expect(client.calls).toBe(1)
  expect(events.filter(event => event.type === 'provider_retry')).toEqual([
    expect.objectContaining({ attempt: 1, final: true }),
  ])
  expect(events.filter(event => event.type === 'text').map(event => event.text)).toEqual([
    '[Error: stream request failed (401): invalid api key]',
  ])
  expect(state.messages.filter(message => message.role === 'assistant')).toEqual([])
  expect(state.messages.filter(message => message.role === 'user')).toHaveLength(1)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
})

test('a fully empty round persists no assistant message for the next provider request', async () => {
  const requests: CompletionRequest[] = []
  const state = createAgentState()
  const events = await collect(runTurn(
    { model: 'gpt-4o', state, userMessage: 'say nothing' },
    {
      llm: {
        async *stream(request): AsyncGenerator<LlmDelta> {
          requests.push(request)
          yield { usage: { inputTokens: 2, outputTokens: 0 } }
        },
      },
    },
  ))

  expect(state.messages.filter(message => message.role === 'assistant')).toEqual([])
  expect(state.messages).toEqual([{ role: 'user', content: 'say nothing' }])
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
})

test('usage merging preserves a legitimate zero-token reading from a fully cached round', async () => {
  const state = createAgentState()
  const events = await collect(runTurn(
    { model: 'gpt-4o', state, userMessage: 'cached round' },
    {
      llm: {
        async *stream(): AsyncGenerator<LlmDelta> {
          yield { content: 'answer', usage: { inputTokens: 5, outputTokens: 4 } }
          yield { usage: { inputTokens: 0, outputTokens: 0 } }
        },
      },
    },
  ))

  expect(events.at(-1)).toMatchObject({
    type: 'turn_done',
    usage: { inputTokens: 0, outputTokens: 0 },
  })
  expect(state.totalInputTokens).toBe(0)
  expect(state.totalOutputTokens).toBe(0)
})

test('held-back text is never reordered after thinking that arrived later in the stream', async () => {
  class ThinkingInterleavedClient implements LlmClient {
    private calls = 0

    async *stream(): AsyncGenerator<LlmDelta> {
      this.calls += 1
      if (this.calls === 1) {
        yield { content: 'Sentinel value-123' }
        yield {
          toolCalls: [{
            id: 'call-think',
            type: 'function',
            function: { name: 'ReadFile', arguments: { path: 'a.ts' } },
          }],
        }
        return
      }
      yield { content: 'Sentinel value' }
      yield { thinking: 'pondering the result' }
      yield { content: ' differs now', usage: { inputTokens: 1, outputTokens: 1 } }
    }
  }

  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'inspect twice',
  }, {
    llm: new ThinkingInterleavedClient(),
    toolExecutor: {
      async execute(): Promise<string> {
        return 'file body'
      },
    },
  }))

  const incremental = events
    .filter(event => event.type === 'text' || event.type === 'thinking')
    .map(event => `${event.type}:${event.text}`)
  expect(incremental).toEqual([
    'text:Sentinel value-123',
    'text:Sentinel value',
    'thinking:pondering the result',
    'text: differs now',
  ])
})

test('the cross-round deduper stays linear and exact on large replayed text', async () => {
  const repeated = 'lorem ipsum dolor sit amet, consectetur adipiscing elit. '.repeat(1_000)
  const replayed = repeated.slice(-10_000)
  const chunks: string[] = []
  for (let index = 0; index < replayed.length; index += 100) {
    chunks.push(replayed.slice(index, index + 100))
  }

  class LargeReplayClient implements LlmClient {
    private calls = 0

    async *stream(): AsyncGenerator<LlmDelta> {
      this.calls += 1
      if (this.calls === 1) {
        yield { content: repeated }
        yield {
          toolCalls: [{
            id: 'call-large',
            type: 'function',
            function: { name: 'ReadFile', arguments: { path: 'big.ts' } },
          }],
          usage: { inputTokens: 1, outputTokens: 1 },
        }
        return
      }
      for (const chunk of chunks) {
        yield { content: chunk }
      }
      yield { content: ' NEW', usage: { inputTokens: 1, outputTokens: 1 } }
    }
  }

  const state = createAgentState()
  const startedAt = performance.now()
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [READ_FILE],
    userMessage: 'repeat the large text',
  }, {
    llm: new LargeReplayClient(),
    toolExecutor: {
      async execute(): Promise<string> {
        return 'file body'
      },
    },
  }))
  const elapsedMs = performance.now() - startedAt

  expect(elapsedMs).toBeLessThan(5_000)
  const secondRoundText = events
    .filter(event => event.type === 'text')
    .map(event => event.text)
    .join('')
  expect(secondRoundText).toBe(`${repeated} NEW`)
  const assistants = state.messages.filter(message => message.role === 'assistant')
  expect(assistants.map(message => message.content)).toEqual([repeated, ' NEW'])
})
