// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { registerInteractionModeTool } from '../src/runtime/interactionModeTool.js'
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
    if (this.attempts === 1) {
      yield { content: 'I need to investigate more.' }
      return
    }
    if (this.attempts === 2) {
      yield {
        toolCalls: [{
          id: 'call-objective-tests',
          type: 'function',
          function: { name: 'exec_command', arguments: { cmd: 'bun', args: ['test'] } },
        }],
      }
      return
    }
    yield { content: 'All tests pass.' }
  }
}

class UnverifiedObjectiveClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    yield { content: 'I need to investigate more.' }
  }
}

class UnsupportedSuccessObjectiveClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    yield { content: 'All tests pass.' }
  }
}

class ModeThenObjectiveClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    if (this.requests.length === 1) {
      yield {
        toolCalls: [{
          id: 'call-mode-objective',
          type: 'function',
          function: { name: 'SetInteractionModeTool', arguments: { mode: 'objective' } },
        }],
      }
      return
    }
    if (this.requests.length === 2) {
      yield { content: 'I still need to finish.' }
      return
    }
    if (this.requests.length === 3) {
      yield {
        toolCalls: [{
          id: 'call-mode-objective-tests',
          type: 'function',
          function: { name: 'exec_command', arguments: { cmd: 'bun', args: ['test'] } },
        }],
      }
      return
    }
    yield { content: 'Verified complete: all tests pass.' }
  }
}

class FailedBlockerObjectiveClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    if (this.requests.length === 1) {
      yield {
        toolCalls: [{
          id: 'call-objective-failure',
          type: 'function',
          function: { name: 'exec_command', arguments: { cmd: 'bun', args: ['test'] } },
        }],
      }
      return
    }
    yield { content: 'BLOCKED: missing dependency. Evidence: command stderr says package not installed.' }
  }
}

class VerificationThenMutationClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    if (this.requests.length === 1) {
      yield {
        toolCalls: [{
          id: 'call-stale-tests',
          type: 'function',
          function: { name: 'exec_command', arguments: { cmd: 'bun', args: ['test'] } },
        }],
      }
      return
    }
    if (this.requests.length === 2) {
      yield {
        toolCalls: [{
          id: 'call-post-test-write',
          type: 'function',
          function: { name: 'WriteFile', arguments: { file_path: 'answer.ts', content: 'changed' } },
        }],
      }
      return
    }
    yield { content: 'All tests pass.' }
  }
}

const readFile: ToolDefinition = {
  type: 'function',
  function: { name: 'ReadFile', description: 'Read a file.', parameters: {} },
}

const execCommand: ToolDefinition = {
  type: 'function',
  function: { name: 'exec_command', description: 'Run a command.', parameters: {} },
}

const writeFile: ToolDefinition = {
  type: 'function',
  function: { name: 'WriteFile', description: 'Write a file.', parameters: {} },
}

function registerSuccessfulVerification(registry: ToolRegistry): void {
  registry.register(execCommand, () => ({
    exitCode: 0,
    stderr: '',
    stdout: '1 pass',
    timedOut: false,
  }))
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
  const registry = new ToolRegistry()
  registerSuccessfulVerification(registry)
  const state = createAgentState()
  const events = []

  for await (const event of runTurn({
    interactionMode: 'objective',
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: registry.definitions(),
    userMessage: 'finish the task',
  }, { llm: client, toolExecutor: registry })) {
    events.push(event)
  }

  expect(client.requests).toHaveLength(3)
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
  expect(state.messages.map(message => message.role)).toEqual([
    'user', 'assistant', 'user', 'assistant', 'tool', 'assistant',
  ])
})

test('a model mode transition is deferred so the next turn receives the enforced objective policy', async () => {
  const client = new ModeThenObjectiveClient()
  const registry = new ToolRegistry()
  registerInteractionModeTool(registry, {
    setMode({ mode }) {
      return { mode, planMode: mode === 'plan' }
    },
  })
  registerSuccessfulVerification(registry)
  const state = createAgentState()
  const events = []

  for await (const event of runTurn({
    interactionMode: 'code',
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: registry.definitions(),
    userMessage: 'switch to objective mode and finish',
  }, { llm: client, toolExecutor: registry })) {
    events.push(event)
  }

  expect(client.requests).toHaveLength(2)
  expect(state.metadata).toMatchObject({ pending_interaction_mode: 'objective' })
  expect(events.filter(event => event.type === 'text').map(event => event.text)).not.toContain(
    '\n[Objective gate: no verified completion or concrete blocker evidence. Continuing.]',
  )
  const modeResult = state.messages.find(message => message.role === 'tool')
  expect(String(modeResult?.content)).toContain('apply on the next user turn')

  state.metadata.interaction_mode = 'objective'
  delete state.metadata.pending_interaction_mode
  for await (const event of runTurn({
    interactionMode: 'objective',
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: registry.definitions(),
    userMessage: 'continue under objective mode',
  }, { llm: client, toolExecutor: registry })) {
    events.push(event)
  }

  expect(client.requests).toHaveLength(4)
  expect(events.filter(event => event.type === 'text').map(event => event.text)).toContain(
    'Verified complete: all tests pass.',
  )
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

test('objective mode ignores verification executions retained from an earlier turn', async () => {
  const client = new UnsupportedSuccessObjectiveClient()
  const state = createAgentState()
  state.toolExecutions.push({
    durationMs: 1,
    inputs: { cmd: 'bun', args: ['test'] },
    name: 'exec_command',
    permitted: true,
    result: JSON.stringify({ exitCode: 0, stdout: '1 pass', timedOut: false }),
    toolCallId: 'prior-turn-test',
  })
  const output: string[] = []

  for await (const event of runTurn({
    interactionMode: 'objective',
    model: 'gpt-4o',
    objectiveGuardMaxRetries: 1,
    state,
    userMessage: 'finish a new task',
  }, { llm: client })) {
    if (event.type === 'text') output.push(event.text)
  }

  expect(client.requests).toHaveLength(2)
  expect(output).toContain(
    '\n[Objective gate: unsupported success claim `all tests pass` without current-turn verification evidence. Continuing.]',
  )
  expect(output.at(-1)).toContain('after 1 retries')
})

test('objective mode accepts a blocker only when the current turn recorded a runtime failure', async () => {
  const client = new FailedBlockerObjectiveClient()
  const registry = new ToolRegistry()
  registry.register(execCommand, () => ({
    exitCode: 1,
    stderr: 'package not installed',
    stdout: '',
    timedOut: false,
  }))
  const state = createAgentState()
  const output: string[] = []

  for await (const event of runTurn({
    interactionMode: 'objective',
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: registry.definitions(),
    userMessage: 'finish or report a concrete blocker',
  }, { llm: client, toolExecutor: registry })) {
    if (event.type === 'text') output.push(event.text)
  }

  expect(client.requests).toHaveLength(2)
  expect(output).toEqual([
    'BLOCKED: missing dependency. Evidence: command stderr says package not installed.',
  ])
})

test('objective mode expires successful verification after a later mutating tool', async () => {
  const client = new VerificationThenMutationClient()
  const registry = new ToolRegistry()
  registerSuccessfulVerification(registry)
  registry.register(writeFile, () => 'Wrote answer.ts.')
  const state = createAgentState()
  const output: string[] = []

  for await (const event of runTurn({
    interactionMode: 'objective',
    model: 'gpt-4o',
    objectiveGuardMaxRetries: 1,
    permissionMode: 'accept-all',
    state,
    tools: registry.definitions(),
    userMessage: 'verify, change, and finish',
  }, { llm: client, toolExecutor: registry })) {
    if (event.type === 'text') output.push(event.text)
  }

  expect(client.requests).toHaveLength(4)
  expect(output).toContain(
    '\n[Objective gate: unsupported success claim `all tests pass` without current-turn verification evidence. Continuing.]',
  )
  expect(output.at(-1)).toContain('after 1 retries')
})
