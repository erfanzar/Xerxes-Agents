// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { HookRunner } from '../src/extensions/hooks.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { classifyError } from '../src/runtime/errorClassifier.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { runTurn, StreamInactivityError } from '../src/streaming/loop.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

const READ_FILE: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ReadFile',
    description: 'Read a file.',
    parameters: { type: 'object', properties: { path: { type: 'string' } } },
  },
}

async function collect(events: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const result: StreamEvent[] = []
  for await (const event of events) result.push(event)
  return result
}

test('a retried provider attempt drops partial text, thinking, and stale tool calls', async () => {
  class MidStreamFailureClient implements LlmClient {
    calls = 0

    async *stream(): AsyncGenerator<LlmDelta> {
      this.calls += 1
      if (this.calls === 1) {
        yield { content: 'stale partial text' }
        yield { thinking: 'stale rationale' }
        yield {
          toolCalls: [{
            id: 'call-stale',
            type: 'function',
            function: { name: 'ReadFile', arguments: { path: 'stale.ts' } },
          }],
        }
        throw new Error('transient connection drop')
      }
      yield { content: 'clean response', usage: { inputTokens: 4, outputTokens: 2 } }
    }
  }

  const client = new MidStreamFailureClient()
  const state = createAgentState()
  let executions = 0
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [READ_FILE],
    userMessage: 'read then fail',
  }, {
    llm: client,
    retryDelays: [0],
    toolExecutor: {
      async execute(): Promise<string> {
        executions += 1
        return 'unreachable'
      },
    },
  }))

  expect(client.calls).toBe(2)
  expect(executions).toBe(0)
  expect(events.filter(event => event.type === 'provider_retry')).toEqual([
    expect.objectContaining({ attempt: 1, final: false }),
  ])
  const assistant = state.messages.filter(message => message.role === 'assistant')
  expect(assistant).toHaveLength(1)
  expect(assistant[0]).toMatchObject({ content: 'clean response' })
  expect(assistant[0]).not.toHaveProperty('thinking')
  expect(assistant[0]).not.toHaveProperty('tool_calls')
  expect(state.messages.some(message => message.role === 'tool')).toBe(false)
  expect(state.thinkingContent).toEqual([''])
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
})

test('auth and validation failures fail immediately while 5xx failures retry', async () => {
  class StatusClient implements LlmClient {
    calls = 0

    constructor(private readonly failures: readonly string[]) {}

    async *stream(): AsyncGenerator<LlmDelta> {
      this.calls += 1
      const failure = this.failures[this.calls - 1]
      if (failure !== undefined) {
        throw new Error(failure)
      }
      yield { content: 'recovered', usage: { inputTokens: 1, outputTokens: 1 } }
    }
  }

  const auth = new StatusClient(['stream request failed (401): invalid api key'])
  const authState = createAgentState()
  const authEvents = await collect(runTurn(
    { model: 'gpt-4o', state: authState, userMessage: 'hi' },
    { delay: async () => undefined, llm: auth, retryDelays: [0, 0] },
  ))
  expect(auth.calls).toBe(1)
  expect(authEvents.filter(event => event.type === 'provider_retry')).toEqual([
    expect.objectContaining({ attempt: 1, final: true }),
  ])
  // Terminal failures surface as an explicit error event, not as a persisted
  // assistant message that would pollute the durable transcript.
  expect(authEvents.filter(event => event.type === 'text').map(event => event.text)).toEqual([
    '[Error: stream request failed (401): invalid api key]',
  ])
  expect(authState.messages.some(message => String(message.content).includes('[Error:'))).toBe(false)
  expect(authState.messages.at(-1)).toMatchObject({ role: 'user', content: 'hi' })
  expect(authEvents.at(-1)).toMatchObject({ type: 'turn_done' })

  const validation = new StatusClient(['stream request failed (400): malformed request'])
  const validationEvents = await collect(runTurn(
    { model: 'gpt-4o', state: createAgentState(), userMessage: 'hi' },
    { delay: async () => undefined, llm: validation, retryDelays: [0, 0] },
  ))
  expect(validation.calls).toBe(1)
  expect(validationEvents.filter(event => event.type === 'provider_retry')).toEqual([
    expect.objectContaining({ attempt: 1, final: true }),
  ])

  const server = new StatusClient(['stream request failed (500): internal error'])
  const serverEvents = await collect(runTurn(
    { model: 'gpt-4o', state: createAgentState(), userMessage: 'hi' },
    { delay: async () => undefined, llm: server, retryDelays: [0, 0] },
  ))
  expect(server.calls).toBe(2)
  expect(serverEvents.filter(event => event.type === 'provider_retry')).toEqual([
    expect.objectContaining({ attempt: 1, final: false }),
  ])
  expect(serverEvents.at(-1)).toMatchObject({ type: 'turn_done' })
})

test('a provider Retry-After hint overrides the configured retry delay', async () => {
  class RateLimitedClient implements LlmClient {
    calls = 0

    async *stream(): AsyncGenerator<LlmDelta> {
      this.calls += 1
      if (this.calls === 1) {
        throw new Error('Rate limit exceeded. Retry after 2 seconds')
      }
      yield { content: 'recovered', usage: { inputTokens: 1, outputTokens: 1 } }
    }
  }

  const delays: number[] = []
  const events = await collect(runTurn(
    { model: 'gpt-4o', state: createAgentState(), userMessage: 'hi' },
    {
      delay: async milliseconds => {
        delays.push(milliseconds)
      },
      llm: new RateLimitedClient(),
      retryDelays: [0],
    },
  ))

  expect(delays).toEqual([2_000])
  expect(events.filter(event => event.type === 'provider_retry')).toEqual([
    expect.objectContaining({ attempt: 1, delay: 2_000, final: false }),
  ])
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
})

test('the inactivity watchdog aborts a stalled stream and the retry starts clean', async () => {
  class StalledThenHealthyClient implements LlmClient {
    calls = 0

    async *stream(_request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
      this.calls += 1
      if (this.calls === 1) {
        yield { content: 'stale chunk' }
        await new Promise<void>((resolve) => {
          signal?.addEventListener('abort', () => resolve(), { once: true })
        })
        throw new Error('stream aborted after stall')
      }
      yield { content: 'healthy response', usage: { inputTokens: 3, outputTokens: 2 } }
    }
  }

  const client = new StalledThenHealthyClient()
  const state = createAgentState()
  const events = await collect(runTurn(
    { model: 'gpt-4o', state, userMessage: 'wait out the stall' },
    { llm: client, retryDelays: [0], streamInactivityTimeoutMs: 25 },
  ))

  expect(client.calls).toBe(2)
  const retries = events.filter(event => event.type === 'provider_retry')
  expect(retries).toHaveLength(1)
  expect(retries[0]).toMatchObject({ attempt: 1, final: false })
  expect(String((retries[0] as { error?: string } | undefined)?.error)).toContain('stalled')
  expect(state.messages.filter(message => message.role === 'assistant')).toEqual([
    expect.objectContaining({ content: 'healthy response' }),
  ])
  expect(classifyError(new StreamInactivityError(25)).retryable).toBe(true)
})

test('a permanently stalled stream fails the round with a visible timeout error', async () => {
  class AlwaysStalledClient implements LlmClient {
    calls = 0

    async *stream(_request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
      this.calls += 1
      yield { content: 'partial ' }
      await new Promise<void>((resolve) => {
        signal?.addEventListener('abort', () => resolve(), { once: true })
      })
    }
  }

  const client = new AlwaysStalledClient()
  const state = createAgentState()
  const events = await collect(runTurn(
    { model: 'gpt-4o', state, userMessage: 'stall forever' },
    { llm: client, retryDelays: [], streamInactivityTimeoutMs: 25 },
  ))

  expect(client.calls).toBe(1)
  expect(events.filter(event => event.type === 'provider_retry')).toEqual([
    expect.objectContaining({ attempt: 1, final: true }),
  ])
  const texts = events.filter(event => event.type === 'text').map(event => event.text)
  expect(texts.at(-1)).toContain('stream inactivity timeout')
  // The partial round and the terminal error are not persisted as assistant
  // content; the turn ends cleanly with the error carried by the final
  // provider_retry and text events.
  expect(state.messages.some(message => message.role === 'assistant')).toBe(false)
  expect(state.messages.at(-1)).toMatchObject({ role: 'user', content: 'stall forever' })
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
})

test('plugin hooks fire in turn order and before_tool_call mutates tool arguments', async () => {
  class ToolThenTextClient implements LlmClient {
    async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
      if (request.messages.some(message => message.role === 'tool')) {
        yield { content: 'done', usage: { inputTokens: 1, outputTokens: 1 } }
        return
      }
      yield {
        toolCalls: [{
          id: 'call-hooks',
          type: 'function',
          function: { name: 'ReadFile', arguments: { path: 'original.ts' } },
        }],
        usage: { inputTokens: 1, outputTokens: 1 },
      }
    }
  }

  const hookRunner = new HookRunner()
  const fired: string[] = []
  hookRunner.register('on_turn_start', () => {
    fired.push('on_turn_start')
  })
  hookRunner.register('before_tool_call', payload => {
    fired.push('before_tool_call')
    return { ...(payload.arguments as Record<string, unknown>), path: 'mutated.ts' }
  })
  hookRunner.register('after_tool_call', () => {
    fired.push('after_tool_call')
  })
  hookRunner.register('tool_result_persist', () => {
    fired.push('tool_result_persist')
  })
  hookRunner.register('on_error', () => {
    fired.push('on_error')
  })
  hookRunner.register('on_turn_end', () => {
    fired.push('on_turn_end')
  })

  const executedInputs: unknown[] = []
  const state = createAgentState()
  const events = await collect(runTurn({
    agentId: 'hooks-agent',
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    sessionId: 'hooks-session',
    state,
    tools: [READ_FILE],
    userMessage: 'read the file',
  }, {
    hookRunner,
    llm: new ToolThenTextClient(),
    toolExecutor: {
      async execute(call): Promise<string> {
        executedInputs.push(call.function.arguments)
        return 'file body'
      },
    },
  }))

  expect(fired).toEqual([
    'on_turn_start',
    'before_tool_call',
    'after_tool_call',
    'tool_result_persist',
    'on_turn_end',
  ])
  expect(executedInputs).toEqual([{ path: 'mutated.ts' }])
  expect(state.messages.find(message => message.role === 'tool')).toMatchObject({
    content: 'file body',
    tool_call_id: 'call-hooks',
  })
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
})

test('on_error fires once for a terminal provider failure with the classified kind', async () => {
  class AlwaysUnauthorizedClient implements LlmClient {
    async *stream(): AsyncGenerator<LlmDelta> {
      throw new Error('stream request failed (401): unauthorized')
    }
  }

  const hookRunner = new HookRunner()
  const errors: Record<string, unknown>[] = []
  hookRunner.register('on_error', payload => {
    errors.push(payload)
  })

  await collect(runTurn(
    { model: 'gpt-4o', state: createAgentState(), userMessage: 'fail' },
    { hookRunner, llm: new AlwaysUnauthorizedClient(), retryDelays: [0, 0] },
  ))

  expect(errors).toHaveLength(1)
  expect(errors[0]).toMatchObject({ attempt: 1, kind: 'auth' })
})
