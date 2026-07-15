// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { createAgentState } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

class ToolThenTextClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    if (request.messages.some(message => message.role === 'tool')) {
      yield { content: 'All done.', usage: { inputTokens: 8, outputTokens: 3 } }
      return
    }
    yield { content: 'Checking <thi' }
    yield { content: 'nk>private rationale</think> now.' }
    yield {
      toolCalls: [{
        id: 'call_1',
        type: 'function',
        function: { name: 'ReadFile', arguments: { path: 'README.md' } },
      }],
      usage: { inputTokens: 5, outputTokens: 4 },
    }
  }
}

class RepeatedToolSentinelClient implements LlmClient {
  private calls = 0

  async *stream(): AsyncGenerator<LlmDelta> {
    this.calls += 1
    if (this.calls === 1) {
      yield { content: 'Inspecting the file.' }
      yield {
        toolCalls: [{
          id: 'call-repeat',
          type: 'function',
          function: { name: 'ReadFile', arguments: { path: 'README.md' } },
        }],
        usage: { inputTokens: 5, outputTokens: 4 },
      }
      return
    }
    // Deliberately use different delta boundaries to reproduce the live bug.
    yield { content: 'Inspecting' }
    yield { content: ' the file.', usage: { inputTokens: 8, outputTokens: 3 } }
  }
}

class OverlappingToolSentinelClient implements LlmClient {
  private calls = 0

  async *stream(): AsyncGenerator<LlmDelta> {
    this.calls += 1
    if (this.calls === 1) {
      yield { content: 'Sentinel: amber-ibis-73' }
      yield {
        toolCalls: [{
          id: 'call-overlap',
          type: 'function',
          function: { name: 'ReadFile', arguments: { path: 'README.md' } },
        }],
        usage: { inputTokens: 5, outputTokens: 4 },
      }
      return
    }
    // The final round repeats only the meaningful tail of the pre-tool text.
    yield { content: 'amber-' }
    yield { content: 'ibis-73  \nHarness Observatory', usage: { inputTokens: 8, outputTokens: 3 } }
  }
}

class ObjectiveClient implements LlmClient {
  readonly requests: CompletionRequest[] = []
  private attempts = 0

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    this.attempts += 1
    yield {
      content: this.attempts === 1 ? 'I need to investigate more.' : 'All tests pass.',
    }
  }
}

class UnverifiedObjectiveClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    yield { content: 'I need to investigate more.' }
  }
}

const readFile: ToolDefinition = {
  type: 'function',
  function: { name: 'ReadFile', description: 'Read a file.', parameters: {} },
}

test('agent loop pairs model tool calls with results and preserves thinking separation', async () => {
  const registry = new ToolRegistry()
  registry.register(readFile, inputs => `read ${inputs.path}`)
  const state = createAgentState()
  const events = []
  for await (const event of runTurn({ model: 'gpt-4o', state, userMessage: 'inspect the readme', tools: [readFile] }, {
    llm: new ToolThenTextClient(),
    toolExecutor: registry,
  })) {
    events.push(event)
  }

  expect(events.map(event => event.type)).toEqual([
    'text', 'thinking', 'text', 'tool_start', 'tool_end', 'text', 'turn_done',
  ])
  expect(state.thinkingContent).toEqual(['private rationale', ''])
  expect(state.messages.map(message => message.role)).toEqual(['user', 'assistant', 'tool', 'assistant'])
  expect(state.messages[2]).toMatchObject({ role: 'tool', tool_call_id: 'call_1', content: 'read README.md' })
  expect(state.totalInputTokens).toBe(13)
  expect(state.totalOutputTokens).toBe(7)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', apiCallsCount: 2, usageComplete: true })
})

test('agent loop emits and persists an identical cross-tool-round sentinel only once', async () => {
  const registry = new ToolRegistry()
  registry.register(readFile, () => 'read complete')
  const state = createAgentState()
  const events = []

  for await (const event of runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [readFile],
    userMessage: 'inspect the readme',
  }, {
    llm: new RepeatedToolSentinelClient(),
    toolExecutor: registry,
  })) {
    events.push(event)
  }

  expect(events.filter(event => event.type === 'text').map(event => event.text)).toEqual([
    'Inspecting the file.',
  ])
  expect(state.messages.filter(message => message.role === 'assistant').map(message => message.content)).toEqual([
    'Inspecting the file.',
    '',
  ])
  expect(state.totalApiCalls).toBe(2)
  expect(state.apiCallsComplete).toBe(true)
  expect(state.usageComplete).toBe(true)
})

test('agent loop removes a long cross-tool suffix overlap without hiding new final text', async () => {
  const registry = new ToolRegistry()
  registry.register(readFile, () => 'read complete')
  const state = createAgentState()
  const events = []

  for await (const event of runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [readFile],
    userMessage: 'inspect the readme',
  }, {
    llm: new OverlappingToolSentinelClient(),
    toolExecutor: registry,
  })) {
    events.push(event)
  }

  expect(events.filter(event => event.type === 'text').map(event => event.text).join('')).toBe(
    'Sentinel: amber-ibis-73  \nHarness Observatory',
  )
  expect(state.messages.filter(message => message.role === 'assistant').map(message => message.content)).toEqual([
    'Sentinel: amber-ibis-73',
    '  \nHarness Observatory',
  ])
})

test('objective mode feeds premature text-only stops back into the loop until verified completion', async () => {
  const client = new ObjectiveClient()
  const state = createAgentState()
  const events = []

  for await (const event of runTurn({
    interactionMode: 'objective',
    model: 'gpt-4o',
    state,
    userMessage: 'finish the task',
  }, { llm: client })) {
    events.push(event)
  }

  expect(client.requests).toHaveLength(2)
  expect(events.filter(event => event.type === 'text').map(event => event.text)).toEqual([
    'I need to investigate more.',
    '\n[Objective gate: no verified completion or concrete blocker evidence. Continuing.]',
    'All tests pass.',
  ])
  expect(client.requests[1]?.messages.some(message => (
    message.role === 'user'
      && typeof message.content === 'string'
      && message.content.includes('[Objective gate]')
  ))).toBe(true)
  expect(state.messages.map(message => message.role)).toEqual(['user', 'assistant', 'user', 'assistant'])
})

test('objective mode stops visibly after its configured retry ceiling', async () => {
  const client = new UnverifiedObjectiveClient()
  const state = createAgentState()
  const output: string[] = []

  for await (const event of runTurn({
    interactionMode: 'objective',
    model: 'gpt-4o',
    objectiveGuardMaxRetries: 1,
    state,
    userMessage: 'finish the task',
  }, { llm: client })) {
    if (event.type === 'text') output.push(event.text)
  }

  expect(client.requests).toHaveLength(2)
  expect(output.at(-1)).toContain('after 1 retries')
})
