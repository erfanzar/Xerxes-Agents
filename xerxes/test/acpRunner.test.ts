// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AcpAgentRunner } from '../src/acp/runner.js'
import { AcpServer } from '../src/acp/server.js'
import type { SubagentTurnCoordinator } from '../src/daemon/subagentCoordinator.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import type { SpawnedAgentSnapshot } from '../src/operators/subagents.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

class TextClient implements LlmClient {
  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    yield { content: 'Hello from ACP.', usage: { inputTokens: 3, outputTokens: 5 } }
  }
}

class WriteThenTextClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    if (request.messages.some(message => message.role === 'tool')) {
      yield { content: 'Write complete.', usage: { inputTokens: 7, outputTokens: 2 } }
      return
    }
    yield {
      toolCalls: [{
        id: 'call-write',
        type: 'function',
        function: { name: 'WriteFile', arguments: { path: 'note.txt', content: 'hello' } },
      }],
      usage: { inputTokens: 4, outputTokens: 1 },
    }
  }
}

class JoinedAgentClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    const context = request.messages.map(message => String(message.content)).join('\n')
    yield context.includes('[sub-agent events]')
      ? { content: 'Integrated delegated evidence.', usage: { inputTokens: 2, outputTokens: 2 } }
      : { content: 'Parent draft.', usage: { inputTokens: 1, outputTokens: 1 } }
  }
}

class RecordingSubagentCoordinator implements SubagentTurnCoordinator {
  beginCalls: string[] = []
  closeCalls = 0
  waitCalls = 0

  begin(sourceId: string) {
    this.beginCalls.push(sourceId)
    let delivered = false
    return {
      close: () => {
        this.closeCalls += 1
      },
      waitForResults: async () => {
        this.waitCalls += 1
        if (delivered) return []
        delivered = true
        return [completedAgent(sourceId)]
      },
    }
  }

  consume(_snapshots: readonly SpawnedAgentSnapshot[]): void {}

  track(_snapshots: readonly SpawnedAgentSnapshot[]): void {}

  trackedIds(_sourceId: string): readonly string[] {
    return []
  }
}

const WRITE_FILE: ToolDefinition = {
  type: 'function',
  function: {
    name: 'WriteFile',
    description: 'Write a file.',
    parameters: { type: 'object', required: ['path', 'content'] },
  },
}

test('ACP agent runner maps the portable loop into persistent ACP session events', async () => {
  const runner = new AcpAgentRunner({ llm: new TextClient(), model: 'gpt-4o' })
  const server = new AcpServer({ runner })
  const sessionId = String(server.openSession('/workspace').session_id)
  const session = server.sessions.get(sessionId)
  if (!session) {
    throw new Error('expected ACP session')
  }
  const events: Record<string, unknown>[] = []

  const first = await runner.runPrompt({ session, text: 'first turn', emit: event => { events.push(event) } })
  const second = await runner.runPrompt({ session, text: 'second turn', emit: event => { events.push(event) } })

  expect(first).toEqual({ ok: true, cancelled: false, input_tokens: 3, output_tokens: 5, tool_calls_count: 0, model: 'gpt-4o' })
  expect(second).toEqual({ ok: true, cancelled: false, input_tokens: 3, output_tokens: 5, tool_calls_count: 0, model: 'gpt-4o' })
  expect(events.filter(event => event.kind === 'text_delta').map(event => event.text)).toEqual(['Hello from ACP.', 'Hello from ACP.'])
  expect(events.filter(event => event.kind === 'turn_end')).toHaveLength(2)
  expect(runner.stateFor(sessionId)?.messages.map(message => message.role)).toEqual([
    'user', 'assistant', 'user', 'assistant',
  ])
})

test('ACP agent runner gives a session model override precedence over its fallback model', async () => {
  const client = new RecordingModelClient()
  const runner = new AcpAgentRunner({ llm: client, model: 'fallback-model' })
  const server = new AcpServer({ runner })
  const sessionId = String(server.openSession('/workspace').session_id)
  const session = server.sessions.get(sessionId)
  if (!session) {
    throw new Error('expected ACP session')
  }

  expect(server.setModel(sessionId, 'session-model')).toEqual({ ok: true })
  await runner.runPrompt({ session, text: 'use the selected model' })

  expect(client.requests).toHaveLength(1)
  expect(client.requests[0]?.model).toBe('session-model')
})

test('ACP agent runner automatically joins detached subagents and asks the model to synthesize them', async () => {
  const client = new JoinedAgentClient()
  const coordinator = new RecordingSubagentCoordinator()
  const runner = new AcpAgentRunner({
    llm: client,
    model: 'gpt-4o',
    subagentCoordinator: coordinator,
  })
  const server = new AcpServer({ runner })
  const sessionId = String(server.openSession('/workspace').session_id)
  const session = server.sessions.get(sessionId)
  if (!session) {
    throw new Error('expected ACP session')
  }
  const events: Record<string, unknown>[] = []

  const result = await runner.runPrompt({
    session,
    text: 'delegate this work',
    emit: event => { events.push(event) },
  })

  expect(result).toMatchObject({ ok: true, cancelled: false, input_tokens: 3, output_tokens: 3 })
  expect(coordinator.beginCalls).toEqual([sessionId])
  expect(coordinator.waitCalls).toBe(2)
  expect(coordinator.closeCalls).toBe(1)
  expect(client.requests).toHaveLength(2)
  expect(client.requests[1]?.messages).toEqual(expect.arrayContaining([
    expect.objectContaining({ role: 'user', content: expect.stringContaining('[sub-agent events]') }),
  ]))
  expect(client.requests[1]?.messages).toEqual(expect.arrayContaining([
    expect.objectContaining({ role: 'user', content: expect.stringContaining('Independent review evidence') }),
  ]))
  expect(events.filter(event => event.kind === 'text_delta').map(event => event.text)).toEqual([
    'Parent draft.',
    'Integrated delegated evidence.',
  ])
})

test('ACP agent runner waits for editor approval and feeds the decision back into tool execution', async () => {
  const registry = new ToolRegistry()
  registry.register(WRITE_FILE, inputs => `wrote ${inputs.path}`)
  const runner = new AcpAgentRunner({
    llm: new WriteThenTextClient(),
    model: 'gpt-4o',
    defaultPermissionMode: 'manual',
    tools: [WRITE_FILE],
    toolExecutor: registry,
  })
  const server = new AcpServer({ runner })
  const sessionId = String(server.openSession('/workspace').session_id)
  const session = server.sessions.get(sessionId)
  if (!session) {
    throw new Error('expected ACP session')
  }
  const events: Record<string, unknown>[] = []
  let resolvePermission!: (event: Record<string, unknown>) => void
  const permission = new Promise<Record<string, unknown>>(resolve => {
    resolvePermission = resolve
  })

  const turn = runner.runPrompt({
    session,
    text: 'write the note',
    emit: event => {
      events.push(event)
      if (event.kind === 'permission_request') {
        resolvePermission(event)
      }
    },
  })
  const request = await permission
  const permissionId = String(request.permission_id)
  expect(server.pendingPermissions()).toMatchObject([{ id: permissionId, session_id: sessionId, tool_name: 'WriteFile' }])
  expect(server.respondPermission(permissionId, true)).toEqual({ ok: true })

  await expect(turn).resolves.toEqual({
    ok: true,
    cancelled: false,
    input_tokens: 11,
    output_tokens: 3,
    tool_calls_count: 1,
    model: 'gpt-4o',
  })
  expect(events.map(event => event.kind)).toEqual([
    'permission_request', 'tool_call_start', 'tool_call_end', 'text_delta', 'turn_end',
  ])
  expect(runner.stateFor(sessionId)?.messages).toMatchObject([
    { role: 'user', content: 'write the note' },
    { role: 'assistant' },
    { role: 'tool', content: 'wrote note.txt', tool_call_id: 'call-write' },
    { role: 'assistant', content: 'Write complete.' },
  ])
  expect(runner.stateFor(sessionId)?.metadata.permission_mode).toBe('manual')
})

test('ACP cancellation aborts a pending approval without leaving the turn blocked', async () => {
  const registry = new ToolRegistry()
  registry.register(WRITE_FILE, inputs => `wrote ${inputs.path}`)
  const runner = new AcpAgentRunner({
    llm: new WriteThenTextClient(),
    model: 'gpt-4o',
    defaultPermissionMode: 'manual',
    tools: [WRITE_FILE],
    toolExecutor: registry,
  })
  const server = new AcpServer({ runner })
  const sessionId = String(server.openSession('/workspace').session_id)
  const session = server.sessions.get(sessionId)
  if (!session) {
    throw new Error('expected ACP session')
  }
  let resolvePermission!: () => void
  const pending = new Promise<void>(resolve => {
    resolvePermission = resolve
  })
  const turn = runner.runPrompt({
    session,
    text: 'write then cancel',
    emit: event => {
      if (event.kind === 'permission_request') {
        resolvePermission()
      }
    },
  })

  await pending
  expect(server.cancel(sessionId)).toEqual({ ok: true })
  await expect(turn).resolves.toMatchObject({ ok: true, cancelled: true })
  expect(server.pendingPermissions()).toEqual([])
})

test('ACP agent runner surfaces and resolves tool-driven input requests', async () => {
  const askTool: ToolDefinition = {
    type: 'function',
    function: { name: 'AskForColor', description: 'Ask for a color.', parameters: { type: 'object' } },
  }
  const registry = new ToolRegistry()
  const questionRunner = new AcpAgentRunner({
    llm: new AskThenTextClient(),
    model: 'gpt-4o',
    defaultPermissionMode: 'accept-all',
    tools: [askTool],
    toolExecutor: registry,
  })
  const questionServer = new AcpServer({ runner: questionRunner })
  const questionSessionId = String(questionServer.openSession('/workspace').session_id)
  const questionSession = questionServer.sessions.get(questionSessionId)
  if (!questionSession) {
    throw new Error('expected ACP session')
  }
  registry.register(askTool, async (_inputs, context) => questionRunner.askUserQuestion(context.sessionId ?? '', 'Pick a color?'))
  let resolveQuestion!: (event: Record<string, unknown>) => void
  const question = new Promise<Record<string, unknown>>(resolve => {
    resolveQuestion = resolve
  })
  const turn = questionRunner.runPrompt({
    session: questionSession,
    text: 'ask me',
    emit: event => {
      if (event.kind === 'input_request') {
        resolveQuestion(event)
      }
    },
  })

  const input = await question
  expect(questionServer.pendingQuestions()).toMatchObject([{
    input_id: input.input_id,
    session_id: questionSessionId,
    question: 'Pick a color?',
  }])
  expect(questionServer.respondQuestion(String(input.input_id), 'blue')).toEqual({ ok: true })
  await expect(turn).resolves.toMatchObject({ ok: true, cancelled: false })
})

class AskThenTextClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    if (request.messages.some(message => message.role === 'tool')) {
      yield { content: 'Thanks.', usage: { inputTokens: 2, outputTokens: 1 } }
      return
    }
    yield {
      toolCalls: [{
        id: 'call-question',
        type: 'function',
        function: { name: 'AskForColor', arguments: {} },
      }],
      usage: { inputTokens: 1, outputTokens: 1 },
    }
  }
}

class RecordingModelClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    yield { content: 'Model selected.', usage: { inputTokens: 1, outputTokens: 1 } }
  }
}

function completedAgent(sourceAgentId: string): SpawnedAgentSnapshot {
  return {
    agentId: 'reviewer',
    closed: false,
    createdAt: '2026-01-01T00:00:00.000Z',
    id: 'subagent-review',
    lastOutput: 'Independent review evidence',
    name: 'review-one',
    promptProfile: 'reviewer',
    queueSize: 0,
    sourceAgentId,
    status: 'completed',
    title: 'Independent review',
    updatedAt: '2026-01-01T00:00:01.000Z',
  }
}
