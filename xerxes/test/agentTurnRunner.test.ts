// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { DaemonInteractionBoard } from '../src/daemon/interactions.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { AgentMemory } from '../src/memory/agentMemory.js'
import { AgentSelfMemory } from '../src/memory/agentSelfMemory.js'
import { registerInteractionModeTool } from '../src/runtime/interactionModeTool.js'
import { BUILTIN_AGENTS, type AgentDefinition } from '../src/agents/definitions.js'
import { AuditEmitter, InMemoryCollector } from '../src/index.js'
import type { DaemonEvent, DaemonSession } from '../src/daemon/runtime.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

class TextClient implements LlmClient {
  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    yield { content: 'Hello from the real loop.', usage: { inputTokens: 3, outputTokens: 5 } }
  }
}

class CapturingClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    yield { content: 'configured agent reply' }
  }
}

class ModeSwitchClient implements LlmClient {
  private calls = 0

  async *stream(): AsyncGenerator<LlmDelta> {
    this.calls += 1
    if (this.calls === 1) {
      yield {
        toolCalls: [{
          id: 'mode-plan',
          type: 'function',
          function: { name: 'SetInteractionModeTool', arguments: { mode: 'plan' } },
        }],
      }
      return
    }
    yield { content: 'Plan ready.' }
  }
}

class AskUserClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    if (request.messages.some(message => message.role === 'tool')) {
      yield { content: 'Thanks for the answer.' }
      return
    }
    yield {
      toolCalls: [{
        id: 'ask-1',
        type: 'function',
        function: { name: 'AskUserQuestionTool', arguments: { question: 'Continue?' } },
      }],
    }
  }
}

class RepeatedSentinelClient implements LlmClient {
  private calls = 0

  async *stream(): AsyncGenerator<LlmDelta> {
    this.calls += 1
    if (this.calls === 1) {
      yield { content: 'Reading now.' }
      yield {
        toolCalls: [{
          id: 'read-repeat',
          type: 'function',
          function: { name: 'ReadFile', arguments: { path: 'README.md' } },
        }],
        usage: { inputTokens: 4, outputTokens: 2 },
      }
      return
    }
    yield { content: 'Reading' }
    yield { content: ' now.', usage: { inputTokens: 6, outputTokens: 2 } }
  }
}

const repeatedReadTool: ToolDefinition = {
  type: 'function',
  function: { name: 'ReadFile', description: 'Read a file.', parameters: {} },
}

test('agent turn runner maps portable loop events to daemon v35 event names', async () => {
  const runner = new AgentTurnRunner({ llm: new TextClient(), model: 'gpt-4o' })
  const session: DaemonSession = {
    activeTurnId: '',
    agentId: 'default',
    cancelRequested: false,
    cwd: process.cwd(),
    extra: {},
    id: 'session-1',
    interactionMode: 'code',
    sessionKey: 'test',
    lastActive: 0,
    messages: [],
    metadata: {},
    model: 'gpt-4o',
    planMode: false,
    status: 'working',
    thinkingContent: [],
    toolExecutions: [],
    totalInputTokens: 0,
    totalOutputTokens: 0,
    turnCount: 0,
    workspace: '/tmp/agents/default',
  }
  const controller = new AbortController()
  const events = []
  for await (const event of runner.run(session, 'say hello', controller.signal)) {
    events.push(event)
  }

  expect(events).toEqual([
    { type: 'text_part', payload: { text: 'Hello from the real loop.' } },
    {
      type: 'status_update',
      payload: {
        model: 'gpt-4o',
        usage: { inputTokens: 3, outputTokens: 5 },
        usage_complete: true,
        tool_calls: 0,
        api_calls: 1,
        calls: 1,
        total_input_tokens: 3,
        total_output_tokens: 5,
        input_tokens: 3,
        output_tokens: 5,
        total_tokens: 8,
        context_tokens: 13,
        max_context: 128_000,
        mode: 'code',
        plan_mode: false,
      },
    },
  ])
  expect(runner.stateFor('session-1')?.messages.map(message => message.role)).toEqual(['user', 'assistant'])
})

test('agent turn runner reports a model-scheduled next-turn mode in the terminal status event', async () => {
  const activeSession: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {}, id: 'mode-status',
    interactionMode: 'code', sessionKey: 'mode-status', lastActive: 0, messages: [], metadata: {}, model: 'gpt-4o',
    planMode: false, status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0,
    totalOutputTokens: 0, turnCount: 0, workspace: '/tmp/agents/default',
  }
  const registry = new ToolRegistry()
  registerInteractionModeTool(registry, {
    setMode({ mode }) {
      activeSession.interactionMode = mode
      activeSession.planMode = mode === 'plan'
      return { mode, planMode: activeSession.planMode }
    },
  })
  const runner = new AgentTurnRunner({
    llm: new ModeSwitchClient(), model: 'gpt-4o', permissionMode: 'accept-all',
    toolExecutor: registry, tools: registry.definitions(),
  })
  const events: DaemonEvent[] = []

  for await (const event of runner.run(activeSession, 'plan this', new AbortController().signal)) events.push(event)

  expect(events.at(-1)).toMatchObject({
    type: 'status_update',
    payload: { mode: 'plan', plan_mode: true },
  })
})

test('agent turn runner synchronizes persisted daemon sessions for explicit resume', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-session-'))
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: new TextClient(), model: 'gpt-4o' }), {
    currentProjectDirectory: directory,
    model: 'gpt-4o',
    sessionDirectory: join(directory, 'sessions'),
  })
  try {
    await runtime.submitTurn('tui:agent', 'persist this turn', () => {})
    const live = runtime.sessionStatus('tui:agent')
    if (!live) {
      throw new Error('expected a live session')
    }
    expect(live.messages).toMatchObject([
      { role: 'user', content: 'persist this turn' },
      { role: 'assistant', content: 'Hello from the real loop.' },
    ])
    expect(live).toMatchObject({ totalInputTokens: 3, totalOutputTokens: 5, turnCount: 1 })

    runtime.evictSession('tui:agent')
    const resumed = await runtime.openSession(live.id, undefined, { resume: true })
    expect(resumed).toMatchObject({
      id: live.id,
      sessionKey: live.id,
      totalInputTokens: 3,
      totalOutputTokens: 5,
      turnCount: 1,
    })
    expect(resumed.messages.map(message => message.role)).toEqual(['user', 'assistant'])
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('daemon session eviction releases resources owned by the evicted session id', async () => {
  const evicted: string[] = []
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: '/workspace',
    onSessionEvict: sessionId => evicted.push(sessionId),
  })
  const active = await runtime.openSession('tui:evict-owned')

  runtime.evictSession(active.sessionKey)

  expect(evicted).toEqual([active.id])
  expect(runtime.sessionStatus(active.sessionKey)).toBeUndefined()
})

test('agent turn runner keeps streamed and resumed transcripts aligned when a tool sentinel repeats', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-sentinel-'))
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({
    llm: new RepeatedSentinelClient(),
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    toolExecutor: { async execute(): Promise<string> { return 'file contents' } },
    tools: [repeatedReadTool],
  }), {
    currentProjectDirectory: directory,
    model: 'gpt-4o',
    sessionDirectory: join(directory, 'sessions'),
  })
  try {
    const events: DaemonEvent[] = []
    await runtime.submitTurn('tui:sentinel', 'inspect it', event => events.push(event))
    expect(events.filter(event => event.type === 'text_part').map(event => event.payload.text)).toEqual([
      'Reading now.',
    ])

    const live = runtime.sessionStatus('tui:sentinel')
    if (!live) throw new Error('expected a live sentinel session')
    expect(live.messages.filter(message => message.role === 'assistant').map(message => message.content)).toEqual([
      'Reading now.',
      '',
    ])
    expect(live).toMatchObject({
      apiCallsComplete: true,
      totalApiCalls: 2,
      totalInputTokens: 10,
      totalOutputTokens: 4,
      usageComplete: true,
    })

    runtime.evictSession('tui:sentinel')
    const resumed = await runtime.openSession(live.id, undefined, { resume: true })
    expect(resumed.messages.filter(message => message.role === 'assistant').map(message => message.content)).toEqual([
      'Reading now.',
      '',
    ])
    expect(resumed).toMatchObject({ apiCallsComplete: true, totalApiCalls: 2, usageComplete: true })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('agent turn runner keeps file attachments provider-facing and preserves authored transcript text', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-attachment-'))
  const client = new CapturingClient()
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: client, model: 'gpt-4o' }), {
    currentProjectDirectory: directory,
    model: 'gpt-4o',
    sessionDirectory: join(directory, 'sessions'),
  })
  try {
    await Bun.write(join(directory, 'context.md'), 'first line\nsecond line')
    const canonicalDirectory = await realpath(directory)
    const events: DaemonEvent[] = []
    const authored = 'review @context.md'
    await runtime.submitTurn('tui:attachment', authored, event => events.push(event))
    await runtime.submitTurn('tui:attachment', 'continue', () => {})

    const firstProviderUser = client.requests[0]?.messages.find(message => message.role === 'user')
    expect(firstProviderUser?.content).toContain('<attached_files>')
    expect(firstProviderUser?.content).toContain('1 | first line\n2 | second line')
    expect(events.find(event => event.type === 'turn_begin')?.payload).toMatchObject({
      text: authored,
      mentioned_files: [join(canonicalDirectory, 'context.md')],
    })

    const live = runtime.sessionStatus('tui:attachment')
    if (!live) throw new Error('expected a live attachment session')
    expect(live.messages[0]).toMatchObject({
      role: 'user',
      content: expect.stringContaining('<attached_files>'),
      text: authored,
    })

    runtime.evictSession('tui:attachment')
    const resumed = await runtime.openSession(live.id, undefined, { resume: true })
    expect(resumed.messages[0]).toMatchObject({
      role: 'user',
      content: expect.stringContaining('<attached_files>'),
      text: authored,
    })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('agent turn runner applies the selected agent prompt, model, and allowed tool surface', async () => {
  const client = new CapturingClient()
  const definition: AgentDefinition = {
    name: 'reviewer',
    description: 'review code',
    systemPrompt: 'Review safely and explain findings.',
    model: 'gpt-4.1-mini',
    tools: ['ReadFile'],
    allowedTools: ['ReadFile'],
    excludeTools: [],
    source: 'test',
    maxDepth: 3,
    isolation: '',
  }
  const runner = new AgentTurnRunner({
    agentDefinitions: new Map([[definition.name, definition]]),
    llm: client,
    model: 'gpt-4o',
    topK: 64,
    tools: [
      { type: 'function', function: { name: 'ReadFile', description: '', parameters: {} } },
      { type: 'function', function: { name: 'WriteFile', description: '', parameters: {} } },
    ],
  })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'reviewer', cancelRequested: false, cwd: process.cwd(), extra: {}, id: 'agent-spec-session',
    interactionMode: 'code', sessionKey: 'agent-spec', lastActive: 0, messages: [], metadata: {}, model: '', planMode: false,
    status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: '/tmp/agents/reviewer',
  }
  for await (const _event of runner.run(session, 'inspect this', new AbortController().signal)) {
    // The assertions inspect the normalized provider request after the stream completes.
  }
  expect(client.requests).toHaveLength(1)
  expect(client.requests[0]?.model).toBe('gpt-4.1-mini')
  expect(client.requests[0]?.topK).toBe(64)
  expect(client.requests[0]?.messages[0]).toMatchObject({ role: 'system', content: 'Review safely and explain findings.' })
  expect(client.requests[0]?.tools?.map(tool => tool.function.name)).toEqual(['ReadFile'])
})

test('resumed subagent history keeps its delegated policy and tool ceiling', async () => {
  const client = new CapturingClient()
  const definition: AgentDefinition = {
    name: 'reviewer',
    description: 'review code',
    systemPrompt: 'Review safely.',
    model: '',
    tools: [],
    allowedTools: null,
    excludeTools: [],
    source: 'test',
    maxDepth: 3,
    isolation: '',
  }
  const runner = new AgentTurnRunner({
    agentDefinitions: new Map([[definition.name, definition]]),
    llm: client,
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    tools: [
      { type: 'function', function: { name: 'ReadFile', description: '', parameters: {} } },
      { type: 'function', function: { name: 'WriteFile', description: '', parameters: {} } },
      { type: 'function', function: { name: 'NewlyRegisteredTool', description: '', parameters: {} } },
      { type: 'function', function: { name: 'SpawnAgents', description: '', parameters: {} } },
      { type: 'function', function: { name: 'SetInteractionModeTool', description: '', parameters: {} } },
    ],
  })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'reviewer', cancelRequested: false, cwd: process.cwd(), extra: {},
    id: 'child-history-session', interactionMode: 'code', sessionKey: 'child-history', lastActive: 0,
    messages: [
      { role: 'user', content: 'prior request' },
      {
        role: 'assistant',
        content: 'prior answer',
        thinking: 'signed reasoning',
        thinking_signature: 'provider-signature',
      },
    ], metadata: {
      delegated_permission_mode: 'plan',
      project_root: `${process.cwd()}/parent-project`,
      session_kind: 'subagent',
      toolsets: ['ReadFile'],
    },
    model: '', planMode: false, status: 'working', thinkingContent: [], toolExecutions: [],
    totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0, workspace: '/tmp/agents/reviewer',
  }

  for await (const _event of runner.run(session, 'continue the review', new AbortController().signal)) {
    // Consume the turn so the request and synchronized policy are final.
  }

  expect(client.requests[0]?.tools?.map(tool => tool.function.name)).toEqual(['ReadFile'])
  expect(client.requests[0]?.messages.find(message => message.role === 'assistant')).toMatchObject({
    thinking: 'signed reasoning',
    thinking_signature: 'provider-signature',
  })
  expect(session.metadata.permission_mode).toBe('plan')
  expect(session.metadata.project_root).toBe(`${process.cwd()}/parent-project`)
  expect(session.metadata.status).toBe('completed')
})

test('agent turn runner rejects an unknown selected profile before contacting the model', async () => {
  const client = new CapturingClient()
  const runner = new AgentTurnRunner({
    agentDefinitions: new Map(),
    llm: client,
    model: 'gpt-4o',
  })
  const unknownSession: DaemonSession = {
    activeTurnId: '', agentId: 'missing-profile', cancelRequested: false, cwd: process.cwd(), extra: {},
    id: 'unknown-agent-session', interactionMode: 'code', sessionKey: 'unknown-agent', lastActive: 0,
    messages: [], metadata: {}, model: '', planMode: false, status: 'working', thinkingContent: [],
    toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: '/tmp/agents/missing-profile',
  }

  const consume = async (): Promise<void> => {
    for await (const _event of runner.run(unknownSession, 'do work', new AbortController().signal)) {
      // The runner must reject before producing events or calling the provider.
    }
  }
  await expect(consume()).rejects.toThrow('is not a registered agent profile')
  expect(client.requests).toEqual([])
})

test('plan and researcher modes enforce read-only tool ceilings and a non-YOLO permission mode', async () => {
  const availableTools: ToolDefinition[] = [
    { type: 'function', function: { name: 'ReadFile', description: '', parameters: {} } },
    { type: 'function', function: { name: 'WriteFile', description: '', parameters: {} } },
    { type: 'function', function: { name: 'exec_command', description: '', parameters: {} } },
    { type: 'function', function: { name: 'SpawnAgents', description: '', parameters: {} } },
    { type: 'function', function: { name: 'SetInteractionModeTool', description: '', parameters: {} } },
  ]

  for (const mode of ['plan', 'researcher'] as const) {
    const client = new CapturingClient()
    const runner = new AgentTurnRunner({
      agentDefinitions: BUILTIN_AGENTS,
      llm: client,
      model: 'gpt-4o',
      permissionMode: 'accept-all',
      tools: availableTools,
    })
    const restrictedSession: DaemonSession = {
      activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {},
      id: `${mode}-restricted-session`, interactionMode: mode, sessionKey: `${mode}-restricted`, lastActive: 0,
      messages: [], metadata: {}, model: '', planMode: mode === 'plan', status: 'working', thinkingContent: [],
      toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
      workspace: `/tmp/agents/${mode}`,
    }

    for await (const _event of runner.run(restrictedSession, 'inspect only', new AbortController().signal)) {
      // Consume the turn so state and the provider request are final.
    }
    expect(client.requests[0]?.tools?.map(tool => tool.function.name)).toEqual(['ReadFile'])
    expect(restrictedSession.metadata.permission_mode).toBe('plan')
    const systemPrompt = String(client.requests[0]?.messages[0]?.content)
    expect(systemPrompt).toContain(mode === 'plan'
      ? 'You are an expert software architect and planner.'
      : 'You are a research assistant focused on understanding codebases.')
  }
})

test('objective mode applies its profile prompt, tool ceiling, and creator identity', async () => {
  const client = new CapturingClient()
  const availableTools: ToolDefinition[] = [
    { type: 'function', function: { name: 'ReadFile', description: '', parameters: {} } },
    { type: 'function', function: { name: 'WriteFile', description: '', parameters: {} } },
    { type: 'function', function: { name: 'AskUserQuestionTool', description: '', parameters: {} } },
    { type: 'function', function: { name: 'SkillTool', description: '', parameters: {} } },
    { type: 'function', function: { name: 'SpawnAgents', description: '', parameters: {} } },
  ]
  let bootstrapAgentId = ''
  const runner = new AgentTurnRunner({
    agentDefinitions: BUILTIN_AGENTS,
    bootstrapSystemPrompt: ({ agentId }) => {
      bootstrapAgentId = agentId
      return `Catalog for ${agentId}`
    },
    llm: client,
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    tools: availableTools,
  })
  const objectiveSession: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {},
    id: 'objective-profile-session', interactionMode: 'objective', sessionKey: 'objective-profile', lastActive: 0,
    messages: [], metadata: {}, model: '', planMode: false, status: 'working', thinkingContent: [],
    toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: '/tmp/agents/objective',
  }

  for await (const _event of runner.run(objectiveSession, 'reach the target', new AbortController().signal)) {}

  expect(bootstrapAgentId).toBe('objective')
  expect(client.requests[0]?.tools?.map(tool => tool.function.name)).toEqual([
    'ReadFile', 'WriteFile', 'SpawnAgents',
  ])
  const systemPrompt = String(client.requests[0]?.messages[0]?.content)
  expect(systemPrompt).toContain('Catalog for objective')
  expect(systemPrompt).toContain('You are an objective runner for hard engineering goals.')
  expect(systemPrompt).not.toContain('Catalog for default')
})

test('session mode changes notify the host with the durable session id', async () => {
  const changes: Array<{ id: string; mode: string }> = []
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: process.cwd(),
    onSessionModeChange: (id, mode) => changes.push({ id, mode }),
  })
  const active = await runtime.openSession('mode-callback')

  await runtime.setSessionMode('mode-callback', 'researcher')

  expect(changes).toEqual([{ id: active.id, mode: 'researcher' }])
})

test('agent turn runner caches a native bootstrap prompt only for the same workspace, model, agent, and tools', async () => {
  const client = new CapturingClient()
  let bootstrapCalls = 0
  const runner = new AgentTurnRunner({
    llm: client,
    model: 'gpt-4o',
    bootstrapSystemPrompt: ({ model, session }) => {
      bootstrapCalls += 1
      return `Bootstrap ${model} in ${session.cwd}`
    },
  })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: '/workspace/bootstrap', extra: {},
    id: 'bootstrap-session', interactionMode: 'code', sessionKey: 'bootstrap', lastActive: 0, messages: [],
    metadata: {}, model: 'gpt-4o', planMode: false, status: 'working', thinkingContent: [], toolExecutions: [],
    totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0, workspace: '/tmp/agents/default',
  }
  for await (const _event of runner.run(session, 'first', new AbortController().signal)) {
    // The provider requests below are the observable prompt boundary.
  }
  for await (const _event of runner.run(session, 'second', new AbortController().signal)) {
    // The cached prompt remains valid for the same workspace/model pair.
  }

  expect(bootstrapCalls).toBe(1)
  expect(client.requests).toHaveLength(2)
  expect(client.requests[0]?.messages[0]).toMatchObject({
    role: 'system',
    content: 'Bootstrap gpt-4o in /workspace/bootstrap',
  })
  expect(client.requests[1]?.messages[0]).toMatchObject({
    role: 'system',
    content: 'Bootstrap gpt-4o in /workspace/bootstrap',
  })

  const alternateSession = { ...session, agentId: 'reviewer', id: 'reviewer-bootstrap-session' }
  for await (const _event of runner.run(alternateSession, 'review', new AbortController().signal)) {
    // A different agent profile must not reuse the default agent's bootstrap prompt.
  }
  expect(bootstrapCalls).toBe(2)
})

test('agent turn runner invalidates its bootstrap cache when the visible tool surface changes', async () => {
  const client = new CapturingClient()
  let bootstrapCalls = 0
  const readTool = { type: 'function' as const, function: { name: 'ReadFile', description: '', parameters: {} } }
  const writeTool = { type: 'function' as const, function: { name: 'WriteFile', description: '', parameters: {} } }
  const definition = (tools: readonly string[]): AgentDefinition => ({
    name: 'default', description: 'test', systemPrompt: '', model: '', tools, allowedTools: null,
    excludeTools: [], source: 'test', maxDepth: 3, isolation: '',
  })
  const activeDefinitions = new Map<string, AgentDefinition>([['default', definition(['ReadFile'])]])
  const runner = new AgentTurnRunner({
    agentDefinitions: activeDefinitions,
    bootstrapSystemPrompt: ({ tools }) => {
      bootstrapCalls += 1
      return `Tools: ${(tools ?? []).map(tool => tool.function.name).join(',')}`
    },
    llm: client,
    model: 'gpt-4o',
    tools: [readTool, writeTool],
  })
  const activeSession: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: '/workspace/tool-cache', extra: {},
    id: 'tool-cache-session', interactionMode: 'code', sessionKey: 'tool-cache', lastActive: 0, messages: [],
    metadata: {}, model: 'gpt-4o', planMode: false, status: 'working', thinkingContent: [], toolExecutions: [],
    totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0, workspace: '/tmp/agents/default',
  }

  for await (const _event of runner.run(activeSession, 'read', new AbortController().signal)) {}
  activeDefinitions.set('default', definition(['WriteFile']))
  for await (const _event of runner.run(activeSession, 'write', new AbortController().signal)) {}

  expect(bootstrapCalls).toBe(2)
  expect(client.requests[0]?.messages[0]?.content).toContain('Tools: ReadFile')
  expect(client.requests[1]?.messages[0]?.content).toContain('Tools: WriteFile')
})

test('agent turn runner includes a trusted session system-prompt addendum', async () => {
  const client = new CapturingClient()
  const runner = new AgentTurnRunner({ llm: client, model: 'gpt-4o' })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: '/workspace/channel', extra: {},
    id: 'channel-workspace-session', interactionMode: 'code', sessionKey: 'channel-workspace', lastActive: 0,
    messages: [], metadata: {}, model: 'gpt-4o', planMode: false, status: 'working',
    systemPromptAddendum: 'Channel workspace: use the current daily notes.', thinkingContent: [], toolExecutions: [],
    totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0, workspace: '/tmp/agents/default',
  }
  for await (const _event of runner.run(session, 'recall channel context', new AbortController().signal)) {
    // The provider request is the observable prompt boundary.
  }

  const system = client.requests[0]?.messages[0]
  expect(system?.role).toBe('system')
  expect(system?.content).toBe('Channel workspace: use the current daily notes.')
})

test('agent turn runner injects project-scoped persistent memory and exposes its project root to tools', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-runner-memory-'))
  try {
    const memory = new AgentMemory({ globalDirectory: join(root, 'global'), projectRoot: root })
    await memory.write('project', 'MEMORY.md', 'The project requires Bun-native persistence.')
    const selfMemory = new AgentSelfMemory({ agentId: 'default', directory: join(root, 'self-memory'), projectRoot: root })
    await selfMemory.learn('The user prefers direct status reports', 'user_taste')
    const client = new CapturingClient()
    const runner = new AgentTurnRunner({
      agentMemory: () => memory,
      agentSelfMemory: () => selfMemory,
      llm: client,
      model: 'gpt-4o',
    })
    const session: DaemonSession = {
      activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: root, extra: {}, id: 'memory-session',
      interactionMode: 'code', sessionKey: 'memory', lastActive: 0, messages: [], metadata: {}, model: 'gpt-4o', planMode: false,
      status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
      workspace: '/tmp/agents/default',
    }
    for await (const _event of runner.run(session, 'recall it', new AbortController().signal)) {
      // The first provider request carries the generated memory context.
    }
    const initialMessage = client.requests[0]?.messages[0]
    expect(initialMessage?.role).toBe('system')
    expect(typeof initialMessage?.content === 'string' ? initialMessage.content : '').toContain(
      'The project requires Bun-native persistence.',
    )
    expect(typeof initialMessage?.content === 'string' ? initialMessage.content : '').toContain(
      'The user prefers direct status reports',
    )
    expect(runner.stateFor(session.id)?.metadata.project_root).toBe(root)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('agent turn runner captures explicit workflow instructions before building the memory prompt', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-runner-workflow-'))
  try {
    const memory = new AgentMemory({ globalDirectory: join(root, 'global'), projectRoot: root })
    const client = new CapturingClient()
    const runner = new AgentTurnRunner({ agentMemory: () => memory, llm: client, model: 'gpt-4o' })
    const session: DaemonSession = {
      activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: root, extra: {}, id: 'workflow-session',
      interactionMode: 'code', sessionKey: 'workflow', lastActive: 0, messages: [], metadata: {}, model: 'gpt-4o', planMode: false,
      status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
      workspace: '/tmp/agents/default',
    }
    const instruction = 'Remember that every release needs a Bun test run.'
    for await (const _event of runner.run(session, instruction, new AbortController().signal)) {
      // The provider request is asserted after the loop has constructed its prompt.
    }
    const system = client.requests[0]?.messages[0]
    expect(system?.role).toBe('system')
    expect(typeof system?.content === 'string' ? system.content : '').toContain(instruction)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('agent turn runner feeds canonical turn lifecycle records to the audit subsystem', async () => {
  const collector = new InMemoryCollector()
  const runner = new AgentTurnRunner({ llm: new TextClient(), model: 'gpt-4o', auditEmitter: new AuditEmitter({ collector }) })
  const session: DaemonSession = {
    activeTurnId: 'audit-turn', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {}, id: 'audit-session',
    interactionMode: 'code', sessionKey: 'audit', lastActive: 0, messages: [], metadata: {}, model: 'gpt-4o', planMode: false,
    status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: '/tmp/agents/default',
  }
  for await (const _event of runner.run(session, 'audit this turn', new AbortController().signal)) {
    // The audit collector receives lifecycle records independent of daemon presentation events.
  }
  expect(collector.getEvents().map(event => event.toRecord().event_type)).toEqual(['turn_start', 'turn_end'])
  expect(collector.getEvents()[0]?.toRecord().session_id).toBe('audit-session')
})

test('agent turn runner routes AskUserQuestionTool through the native daemon reply board', async () => {
  const board = new DaemonInteractionBoard()
  const session: DaemonSession = {
    activeTurnId: 'ask-turn', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {}, id: 'ask-session',
    interactionMode: 'code', sessionKey: 'ask', lastActive: 0, messages: [], metadata: {}, model: 'gpt-4o', planMode: false,
    status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: '/tmp/agents/default',
  }
  const questionEvents: DaemonEvent[] = []
  const release = board.bind(session.id, event => {
    questionEvents.push(event)
    if (event.type === 'question_request') {
      queueMicrotask(() => {
        board.respondQuestion(String(event.payload.id), { answer: 'yes' })
      })
    }
  })
  try {
    const runner = new AgentTurnRunner({
      interactions: board,
      llm: new AskUserClient(),
      model: 'gpt-4o',
      tools: [{
        type: 'function',
        function: { name: 'AskUserQuestionTool', description: 'ask', parameters: { type: 'object' } },
      }],
    })
    const events: DaemonEvent[] = []
    for await (const event of runner.run(session, 'need a choice', new AbortController().signal)) {
      events.push(event)
    }
    expect(questionEvents).toEqual([expect.objectContaining({
      type: 'question_request',
      payload: expect.objectContaining({ questions: [expect.objectContaining({ question: 'Continue?' })] }),
    })])
    expect(events).toEqual(expect.arrayContaining([
      expect.objectContaining({
        type: 'tool_result',
        payload: expect.objectContaining({
          name: 'AskUserQuestionTool',
          permitted: true,
          result: expect.stringContaining('"answer":"yes"'),
          return_value: expect.stringContaining('"answer":"yes"'),
          display_blocks: [],
        }),
      }),
      { type: 'text_part', payload: { text: 'Thanks for the answer.' } },
    ]))
    expect(session.messages.find(message => message.role === 'tool')?.content).toContain('"answer":"yes"')
  } finally {
    release()
  }
})

test('agent turn runner preserves external undo edits across turns', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-undo-'))
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: new TextClient(), model: 'gpt-4o' }), {
    currentProjectDirectory: directory,
    model: 'gpt-4o',
    sessionDirectory: join(directory, 'sessions'),
  })
  try {
    await runtime.submitTurn('tui:undo', 'first prompt', () => {})
    const session = runtime.sessionStatus('tui:undo')
    if (!session) throw new Error('expected a live session')
    expect(session.messages.map(message => message.role)).toEqual(['user', 'assistant'])

    // /undo mutates session.messages directly while the runner keeps state.
    session.messages.pop()
    session.messages.pop()
    session.turnCount = Math.max(0, session.turnCount - 1)

    await runtime.submitTurn('tui:undo', 'second prompt', () => {})
    expect(session.messages).toMatchObject([
      { role: 'user', content: 'second prompt' },
      { role: 'assistant', content: 'Hello from the real loop.' },
    ])
    expect(session.turnCount).toBe(1)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('agent turn runner preserves idle steering appended to the session across turns', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-steer-'))
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: new TextClient(), model: 'gpt-4o' }), {
    currentProjectDirectory: directory,
    model: 'gpt-4o',
    sessionDirectory: join(directory, 'sessions'),
  })
  try {
    await runtime.submitTurn('tui:steer', 'first prompt', () => {})
    expect(runtime.steerTurn('tui:steer', 'hold on')).toBe(true)
    const session = runtime.sessionStatus('tui:steer')
    if (!session) throw new Error('expected a live session')
    expect(session.messages.at(-1)).toMatchObject({
      role: 'user',
      content: '[steer from user]\nhold on',
    })

    await runtime.submitTurn('tui:steer', 'second prompt', () => {})
    expect(session.messages.map(message => `${message.role}:${message.content}`)).toEqual([
      'user:first prompt',
      'assistant:Hello from the real loop.',
      'user:[steer from user]\nhold on',
      'user:second prompt',
      'assistant:Hello from the real loop.',
    ])
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('agent turn runner drops cached session state when the session is evicted', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-drop-'))
  const runner = new AgentTurnRunner({ llm: new TextClient(), model: 'gpt-4o' })
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: 'gpt-4o',
    sessionDirectory: join(directory, 'sessions'),
  })
  try {
    await runtime.submitTurn('tui:drop', 'cache this state', () => {})
    const session = runtime.sessionStatus('tui:drop')
    if (!session) throw new Error('expected a live session')
    expect(runner.stateFor(session.id)).toBeDefined()

    runtime.evictSession('tui:drop')
    expect(runner.stateFor(session.id)).toBeUndefined()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('daemon session eviction aborts the in-flight turn and frees the session key', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-evict-turn-'))
  const runner = new GatedTurnRunner()
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: 'gpt-4o',
    sessionDirectory: join(directory, 'sessions'),
  })
  try {
    const firstEvents: DaemonEvent[] = []
    const first = runtime.submitTurn('tui:evict-turn', 'long work', event => firstEvents.push(event))
    await waitForCondition(() => runner.runs === 1)

    runtime.evictSession('tui:evict-turn')
    await first
    expect(firstEvents.find(event => event.type === 'turn_end')?.payload).toMatchObject({ cancelled: true })
    expect(runtime.sessionStatus('tui:evict-turn')).toBeUndefined()

    const secondEvents: DaemonEvent[] = []
    await runtime.submitTurn('tui:evict-turn', 'replacement turn', event => secondEvents.push(event))
    expect(secondEvents.filter(event => event.type === 'text_part').map(event => event.payload.text)).toEqual([
      'replacement done',
    ])
    expect(secondEvents.some(event => `${event.payload.message ?? ''}`.includes('already active'))).toBe(false)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('daemon cancellation reports false when the session has no active turn', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-cancel-idle-'))
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, 'sessions'),
  })
  try {
    await runtime.openSession('tui:idle')

    expect(runtime.cancelTurn('tui:idle')).toBe(false)
    expect(runtime.cancelTurn('missing')).toBe(false)
    expect(runtime.sessionStatus('tui:idle')?.cancelRequested).toBe(false)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

class GatedTurnRunner {
  runs = 0

  async *run(
    _session: DaemonSession,
    _text: string,
    signal: AbortSignal,
  ): AsyncGenerator<DaemonEvent> {
    this.runs += 1
    if (this.runs > 1) {
      yield { type: 'text_part', payload: { text: 'replacement done' } }
      return
    }
    yield { type: 'text_part', payload: { text: 'gated' } }
    await new Promise<void>(resolve => {
      if (signal.aborted) {
        resolve()
        return
      }
      signal.addEventListener('abort', () => resolve(), { once: true })
    })
  }
}

async function waitForCondition(predicate: () => boolean, timeout = 2_000): Promise<void> {
  const deadline = Date.now() + timeout
  while (!predicate()) {
    if (Date.now() >= deadline) {
      throw new Error('Timed out waiting for daemon runner state')
    }
    await Bun.sleep(5)
  }
}
