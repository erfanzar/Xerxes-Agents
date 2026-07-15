// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type { ToolExecutor } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { createDebugLlmClient, runDebugTurn } from '../src/streaming/debugLoop.js'
import type { ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

const readFileDefinition: ToolDefinition = {
  type: 'function',
  function: { name: 'ReadFile', description: 'Read a file.', parameters: {} },
}

function readFileCall(id: string): ToolCall {
  return {
    id,
    type: 'function',
    function: { name: 'ReadFile', arguments: { path: 'README.md' } },
  }
}

async function collectEvents(events: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const collected: StreamEvent[] = []
  for await (const event of events) {
    collected.push(event)
  }
  return collected
}

class TaggedTextClient implements LlmClient {
  async *stream(_request: CompletionRequest, _signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    yield { content: 'visible <thi' }
    yield { content: 'nk>private</think> tail <thinking>literal</thinking>' }
  }
}

class RepeatingToolClient implements LlmClient {
  requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest, _signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    yield { toolCalls: [readFileCall(`call_${this.requests.length}`)] }
  }
}

class ToolThenTextClient implements LlmClient {
  async *stream(request: CompletionRequest, _signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    if (request.messages.some(message => message.role === 'tool')) {
      yield { content: 'diagnostic complete' }
      return
    }
    yield { toolCalls: [readFileCall('missing_tool')] }
  }
}

class FailingClient implements LlmClient {
  calls = 0

  async *stream(_request: CompletionRequest, _signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    this.calls += 1
    throw new Error('debug provider failed')
  }
}

test('debug loop recognizes only <think> tags across split chunks', async () => {
  const state = createAgentState()
  const events = await collectEvents(runDebugTurn({
    model: 'gpt-4o',
    state,
    userMessage: 'compare parser behavior',
  }, { llm: new TaggedTextClient() }))

  expect(events.filter(event => event.type === 'text').map(event => event.text)).toEqual([
    'visible ',
    ' tail <thinking>literal</thinking>',
  ])
  expect(events.filter(event => event.type === 'thinking').map(event => event.text)).toEqual(['private'])
  expect(state.messages[1]).toMatchObject({
    role: 'assistant',
    content: 'visible  tail <thinking>literal</thinking>',
    thinking: 'private',
  })
})

test('debug loop enforces its bounded tool-turn limit', async () => {
  const client = new RepeatingToolClient()
  const executed: string[] = []
  const executor: ToolExecutor = {
    async execute(call): Promise<string> {
      executed.push(call.id)
      return `read ${call.id}`
    },
  }
  const state = createAgentState()
  const events = await collectEvents(runDebugTurn({
    maxToolTurns: 2,
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [readFileDefinition],
    userMessage: 'keep reading',
  }, { llm: client, toolExecutor: executor }))

  expect(client.requests).toHaveLength(2)
  expect(executed).toEqual(['call_1', 'call_2'])
  expect(events.filter(event => event.type === 'tool_end')).toHaveLength(2)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', toolCallsCount: 2 })
})

test('debug loop reports an unavailable tool rather than fabricating a successful result', async () => {
  const state = createAgentState()
  const events = await collectEvents(runDebugTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [readFileDefinition],
    userMessage: 'read the file',
  }, { llm: new ToolThenTextClient() }))

  const toolEnd = events.find(event => event.type === 'tool_end')
  expect(toolEnd).toMatchObject({
    type: 'tool_end',
    result: {
      result: expect.stringContaining('Tool execution failed: Function ReadFile: no tool executor is configured for the diagnostic loop'),
    },
  })
  expect(state.messages.find(message => message.role === 'tool')).toMatchObject({
    role: 'tool',
    content: expect.stringContaining('Tool execution failed:'),
  })
})

test('debug client disables Anthropic prompt cache markers', async () => {
  let payload: Record<string, unknown> | undefined
  const client = createDebugLlmClient('claude-sonnet-4-6', {}, {
    apiKey: 'debug-key',
    baseUrl: 'https://debug.example.test',
    fetchImplementation: async (_input, init): Promise<Response> => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return new Response('data: {"type":"message_stop"}\n\n', {
        headers: { 'Content-Type': 'text/event-stream' },
      })
    },
  })

  await collectDeltas(client.stream({
    model: 'claude-sonnet-4-6',
    messages: [
      { role: 'system', content: 'diagnostic system prompt' },
      { role: 'user', content: 'hello' },
    ],
    tools: [readFileDefinition],
  }))

  expect(payload).toMatchObject({ system: 'diagnostic system prompt' })
  const tools = payload?.tools
  expect(Array.isArray(tools)).toBe(true)
  if (!Array.isArray(tools)) {
    throw new Error('Anthropic diagnostic request did not include tool schemas')
  }
  expect(tools[0]).not.toHaveProperty('cache_control')
})

test('debug loop respects an already-aborted signal and emits one terminal provider error without retries', async () => {
  const cancelledClient = new TaggedTextClient()
  const cancelledState = createAgentState()
  const controller = new AbortController()
  controller.abort('stop diagnostic')

  const cancelledEvents = await collectEvents(runDebugTurn({
    model: 'gpt-4o',
    state: cancelledState,
    userMessage: 'do not start',
  }, { llm: cancelledClient }, controller.signal))

  expect(cancelledEvents).toEqual([])
  expect(cancelledState.messages).toEqual([])

  const failingClient = new FailingClient()
  const events = await collectEvents(runDebugTurn({
    model: 'gpt-4o',
    state: createAgentState(),
    userMessage: 'surface failure',
  }, { llm: failingClient }))

  expect(failingClient.calls).toBe(1)
  expect(events).toContainEqual({
    type: 'provider_retry',
    error: 'debug provider failed',
    attempt: 1,
    maxAttempts: 1,
    delay: 0,
    final: true,
  })
  expect(events.filter(event => event.type === 'text').map(event => event.text)).toEqual([
    '[Error: debug provider failed]',
  ])
})

async function collectDeltas(deltas: AsyncIterable<LlmDelta>): Promise<LlmDelta[]> {
  const collected: LlmDelta[] = []
  for await (const delta of deltas) {
    collected.push(delta)
  }
  return collected
}
