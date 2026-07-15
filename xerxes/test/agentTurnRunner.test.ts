// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { DaemonInteractionBoard } from '../src/daemon/interactions.js'
import { AgentMemory } from '../src/memory/agentMemory.js'
import { AgentSelfMemory } from '../src/memory/agentSelfMemory.js'
import type { AgentDefinition } from '../src/agents/definitions.js'
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
      },
    },
  ])
  expect(runner.stateFor('session-1')?.messages.map(message => message.role)).toEqual(['user', 'assistant'])
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
  expect(client.requests[0]?.messages[0]).toMatchObject({ role: 'system', content: 'Review safely and explain findings.' })
  expect(client.requests[0]?.tools?.map(tool => tool.function.name)).toEqual(['ReadFile'])
})

test('agent turn runner injects and caches a native bootstrap prompt by workspace and model', async () => {
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
