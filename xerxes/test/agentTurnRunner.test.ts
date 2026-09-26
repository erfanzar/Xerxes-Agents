// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { AgentTurnRunner, SessionPromptSnapshots } from '../src/daemon/turnRunner.js'
import { DaemonInteractionBoard } from '../src/daemon/interactions.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { AgentMemory } from '../src/memory/agentMemory.js'
import { AgentSelfMemory } from '../src/memory/agentSelfMemory.js'
import { appendContextDelta, readContextDeltas } from '../src/runtime/contextDeltas.js'
import { registerInteractionModeTool } from '../src/runtime/interactionModeTool.js'
import { createGoal, editGoal, getGoal } from '../src/runtime/goalDomain.js'
import { registerGoalTools } from '../src/runtime/goalTools.js'
import { findGoalEvidenceExecution } from '../src/runtime/goalEvidence.js'
import { BUILTIN_AGENTS, type AgentDefinition } from '../src/agents/definitions.js'
import { AuditEmitter, InMemoryCollector } from '../src/index.js'
import type { DaemonEvent, DaemonSession } from '../src/daemon/runtime.js'
import type { RawMessage, TranscriptMessageJournalAppend } from '../src/session/daemonTranscript.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

class TextClient implements LlmClient {
  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    yield { content: 'Hello from the real loop.', usage: { inputTokens: 3, outputTokens: 5 } }
  }
}

test('silent provider waits are delivered live and terminal errors are not lost in pre-content buffering', async () => {
  const release = Promise.withResolvers<void>()
  const started = Promise.withResolvers<void>()
  const runner = new AgentTurnRunner({ model: 'test-model', llm: {
    async *stream(): AsyncGenerator<LlmDelta> {
      started.resolve()
      await release.promise
      throw new Error('invalid request: test failure')
    },
  } })
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-silent-provider-'))
  const runtime = new InMemoryDaemonRuntime(runner, { sessionDirectory: join(directory, 'sessions') })
  const session = await runtime.openSession('silent-provider')
  const events: DaemonEvent[] = []
  const turn = runtime.submitTurn(session.sessionKey, 'continue', event => { events.push(event) })
  try {
    await started.promise
    expect(events.some(event => event.type === 'status_update' && event.payload.kind === 'provider_wait')).toBe(true)
  } finally {
    release.resolve()
    await turn
  }
  expect(events.some(event => event.type === 'notification' && String(event.payload.message).includes('test failure'))).toBe(true)
  expect(events.some(event => event.type === 'status_update' && event.payload.kind === 'provider_ready')).toBe(true)
})

test('proactive compaction metadata survives turn synchronization and blocks oversized inference', async () => {
  let session: DaemonSession
  let reductions = 0
  const runner = new AgentTurnRunner({
    model: 'gpt-test', llm: new TextClient(), contextLimit: 100_000, maxTokens: 1000,
    reduceContext: async messages => {
      reductions++
      expect(events.some(event => event.type === 'status_update' && event.payload.kind === 'compressing')).toBe(true)
      session.metadata.last_compaction = { reason: 'mid-turn-auto-compact', tokens_after: 10 }
      return { messages: messages.slice(-1), tokensFreed: 150_000 }
    },
  })
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-compaction-metadata-'))
  const runtime = new InMemoryDaemonRuntime(runner, { sessionDirectory: join(directory, 'sessions') })
  session = await runtime.openSession('compaction-metadata')
  session.messages = [{ role: 'user', content: 'historical output '.repeat(50_000) }]
  const events: DaemonEvent[] = []
  await runtime.submitTurn(session.sessionKey, 'continue', event => { events.push(event) })
  expect(events.filter(event => event.type === 'status_update').map(event => event.payload.kind).filter(Boolean)).toEqual(['compressing', 'compaction', 'provider_wait', 'provider_ready'])
  expect(reductions).toBe(1)
  expect(session.metadata.last_compaction).toMatchObject({ reason: 'mid-turn-auto-compact' })
  expect(session.messages.some(message => String(message.content).includes('historical output'))).toBe(false)
})

test('goal evidence resolves a real completed tool call before the next provider inference', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-goal-evidence-'))
  try {
    const registry = new ToolRegistry()
    let runtime: InMemoryDaemonRuntime
    registerGoalTools(registry, {
      sessionId: context => String(context.sessionId), metadata: context => context.metadata,
      isHumanTurn: () => true, currentRound: () => undefined,
      evidenceExecution: (context, id) => findGoalEvidenceExecution(runtime.listSessions().find(row => row.id === context.sessionId)?.toolExecutions ?? [], id),
    })
    registry.register({ type: 'function', function: { name: 'CompareArtifacts', description: 'Compare two test artifacts', parameters: { type: 'object', properties: {} } } }, () => ({ exit_code: 0 }))
    let calls = 0
    let activeSession: DaemonSession
    const runner = new AgentTurnRunner({ model: 'gpt-4o', permissionMode: 'accept-all', toolExecutor: registry, tools: registry.definitions(),
      llm: { async *stream() {
        calls++
        const goal = getGoal(activeSession.metadata, activeSession.id)!
        if (calls === 1) yield { toolCalls: [{ id: 'artifact-proof', type: 'function' as const, function: { name: 'CompareArtifacts', arguments: {} } }] }
        else if (calls === 2) yield { toolCalls: [{ id: 'record-proof', type: 'function' as const, function: { name: 'update_goal', arguments: { goal_id: goal.id, revision: goal.revision, action: 'record_evidence', criterion_id: 'equal', tool_call_id: 'artifact-proof', evidence_summary: 'Comparison exited zero' } } }] }
        else if (calls === 3) {
          expect(goal.criteria?.[0]?.evidence?.toolCallId).toBe('artifact-proof')
          yield { toolCalls: [{ id: 'complete-goal', type: 'function' as const, function: { name: 'update_goal', arguments: { goal_id: goal.id, revision: goal.revision, action: 'complete' } } }] }
        } else yield { content: 'Completed with comparison evidence.' }
      } },
    })
    runtime = new InMemoryDaemonRuntime(runner, { model: 'gpt-4o', currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
    activeSession = await runtime.openSession('evidence-flow')
    createGoal(activeSession.metadata, activeSession.id, { objective: 'Match the artifacts', criteria: [{ id: 'equal', description: 'Artifacts match' }] }, Date.now())
    await runtime.submitTurn(activeSession.sessionKey, 'Verify and finish', () => {})
    expect(activeSession.toolExecutions[1]).toMatchObject({ result: expect.stringContaining('artifact-proof') })
    expect(activeSession.toolExecutions).toHaveLength(3)
    expect(getGoal(activeSession.metadata, activeSession.id)?.phase).toBe('complete')
    expect(calls).toBe(4)
    const reloadedRuntime = new InMemoryDaemonRuntime(undefined, { model: 'gpt-4o', currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
    const reloaded = await reloadedRuntime.openSession(activeSession.id, activeSession.agentId, { resume: true, cwd: activeSession.cwd })
    expect(getGoal(reloaded.metadata, reloaded.id)?.criteria?.[0]?.evidence).toMatchObject({ toolCallId: 'artifact-proof', summary: 'Comparison exited zero' })
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('goal tools update the live session before the next inference and preserve intervening human edits on failure', async () => {
  const registry = new ToolRegistry()
  registerGoalTools(registry, {
    sessionId: context => String(context.sessionId), metadata: context => context.metadata,
    isHumanTurn: () => true, currentRound: () => undefined,
  })
  let activeSession: DaemonSession
  let calls = 0
  const runner = new AgentTurnRunner({
    model: 'gpt-4o', permissionMode: 'accept-all', toolExecutor: registry, tools: registry.definitions(),
    llm: { async *stream() {
      if (++calls === 1) {
        const goal = getGoal(activeSession.metadata, activeSession.id)!
        yield { toolCalls: [{ id: 'edit-goal', type: 'function' as const, function: {
          name: 'update_goal', arguments: { goal_id: goal.id, revision: goal.revision, action: 'edit', objective: 'Optimize TPU kernels' },
        } }] }
        return
      }
      const goal = getGoal(activeSession.metadata, activeSession.id)!
      expect(goal.objective).toBe('Optimize TPU kernels')
      editGoal(activeSession.metadata, activeSession.id, goal, { objective: 'Human revised objective' }, Date.now())
      throw new Error('provider unavailable after goal edit')
    } },
  })
  const runtime = new InMemoryDaemonRuntime(runner, { model: 'gpt-4o' })
  activeSession = await runtime.openSession('live-goal', 'default')
  createGoal(activeSession.metadata, activeSession.id, { objective: 'Old review goal' }, Date.now())
  let observed = false
  try {
    for await (const event of runner.run(activeSession, 'change the goal', new AbortController().signal)) {
      if (event.type === 'tool_result') {
        expect(getGoal(activeSession.metadata, activeSession.id)?.objective).toBe('Optimize TPU kernels')
        observed = true
      }
    }
  } catch (error) {
    expect(String(error)).toContain('provider unavailable')
  }
  expect(observed).toBe(true)
  expect(getGoal(activeSession.metadata, activeSession.id)?.objective).toBe('Human revised objective')
})

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

class GatedClient implements LlmClient {
  readonly requests: CompletionRequest[] = []
  started = false
  private releaseGate: (() => void) | undefined

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    this.started = true
    await new Promise<void>(resolve => { this.releaseGate = resolve })
    yield { content: 'released' }
  }

  release(): void {
    this.releaseGate?.()
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

test('a vendor-prefixed model id runs without being read as a routing prefix', async () => {
  // `stealth/ox-alpha` is an OpenRouter MODEL id: the part before the slash
  // is a vendor, not a provider prefix. Provider routing used to infer a
  // provider from the id and threw `unknown provider prefix 'stealth'` on
  // every turn that used one. The active profile knows the provider, so the
  // runner is given it and nothing has to guess.
  const runner = new AgentTurnRunner({
    llm: new TextClient(),
    model: 'stealth/ox-alpha',
    providerOverrides: { provider: 'openrouter' },
  })
  const session: DaemonSession = {
    activeTurnId: '',
    agentId: 'default',
    cancelRequested: false,
    cwd: process.cwd(),
    extra: {},
    id: 'session-openrouter',
    interactionMode: 'code',
    sessionKey: 'test',
    lastActive: 0,
    messages: [],
    metadata: {},
    model: 'stealth/ox-alpha',
    planMode: false,
    status: 'working',
    thinkingContent: [],
    toolExecutions: [],
    totalInputTokens: 0,
    totalOutputTokens: 0,
    turnCount: 0,
    workspace: '/tmp/agents/default',
  }
  const events = []
  for await (const event of runner.run(session, 'hi', new AbortController().signal)) {
    events.push(event)
  }

  const status = events.find(event => event.type === 'status_update')

  expect(status).toBeDefined()
  // Provider routing succeeds, but no capacity was reported for this model.
  expect((status as unknown as { payload: Record<string, unknown> }).payload).not.toHaveProperty('max_context')
})

test.each([false, true])('agent turn runner fires extension hooks with session selection=%s', async useFactory => {
  const { HookRunner } = await import('../src/extensions/hooks.js')
  const hookRunner = new HookRunner()
  const fired: string[] = []
  hookRunner.register('on_turn_start', payload => {
    fired.push(`start:${String(payload.model ?? '')}`)
  })
  hookRunner.register('on_turn_end', () => {
    fired.push('end')
  })

  const runner = new AgentTurnRunner({
    ...(useFactory ? { hookRunnerForSession: (session: DaemonSession) => { expect(session.cwd).toBe(process.cwd()); return hookRunner } } : { hookRunner }),
    llm: new TextClient(),
    model: 'hook-model',
  })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {},
    id: 'hook-session', interactionMode: 'code', sessionKey: 'hook-key', lastActive: 0,
    messages: [], metadata: {}, model: 'hook-model', planMode: false, status: 'working',
    thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0,
    turnCount: 0, workspace: '/tmp/agents/default',
  }
  for await (const _event of runner.run(session, 'hi', new AbortController().signal)) {
    void _event
  }

  expect(fired).toEqual(['start:hook-model', 'end'])
})

test('agent turn runner resolves output capacity for the session model after explicit max tokens', async () => {
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {},
    id: 'model-cap-session', interactionMode: 'code', sessionKey: 'model-cap', lastActive: 0,
    messages: [], metadata: {}, model: 'pinned-model', planMode: false, status: 'working',
    thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0,
    turnCount: 0, workspace: '/tmp/agents/default',
  }
  const catalogClient = new CapturingClient()
  const lookedUp: string[] = []
  const catalogRunner = new AgentTurnRunner({
    llm: catalogClient,
    maxOutputTokens: model => {
      lookedUp.push(model)
      return 65_536
    },
    model: 'global-model',
  })
  for await (const _event of catalogRunner.run(session, 'hi', new AbortController().signal)) {
    // drain
  }
  expect(lookedUp).toEqual(['pinned-model'])
  expect(catalogClient.requests[0]?.maxTokens).toBe(65_536)

  const explicitClient = new CapturingClient()
  const explicitRunner = new AgentTurnRunner({
    llm: explicitClient,
    maxOutputTokens: () => 65_536,
    maxTokens: 8_192,
    model: 'global-model',
  })
  for await (const _event of explicitRunner.run({ ...session, id: 'explicit-cap-session', messages: [] }, 'hi', new AbortController().signal)) {
    // drain
  }
  expect(explicitClient.requests[0]?.maxTokens).toBe(8_192)
})

test('agent turn runner maps portable loop events and supplied live capacity to daemon v35 events', async () => {
  const runner = new AgentTurnRunner({ contextLimit: 128_000, llm: new TextClient(), model: 'gpt-4o' })
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
    { type: 'status_update', payload: { kind: 'provider_wait', text: 'Waiting for model response…' } },
    { type: 'status_update', payload: { kind: 'provider_ready', text: '' } },
    // Text streams inline while the provider is still emitting; the round's
    // live token/context status_update lands once per provider round, right
    // after that round's deltas, and the terminal status_update closes the
    // turn with the complete totals.
    { type: 'text_part', payload: { text: 'Hello from the real loop.' } },
    {
      type: 'status_update',
      payload: {
        model: 'gpt-4o',
        usage: { inputTokens: 3, outputTokens: 5 },
        total_input_tokens: 3,
        total_output_tokens: 5,
        input_tokens: 3,
        output_tokens: 5,
        total_tokens: 8,
        context_tokens: 8,
        max_context: 128_000,
        llm_duration_ms: expect.any(Number),
        ttft_ms: expect.any(Number),
      },
    },
    {
      type: 'status_update',
      payload: {
        model: 'gpt-4o',
        usage: { inputTokens: 3, outputTokens: 5 },
        usage_complete: true,
        stop_reason: 'completed',
        tool_calls: 0,
        api_calls: 1,
        calls: 1,
        total_input_tokens: 3,
        total_output_tokens: 5,
        input_tokens: 3,
        output_tokens: 5,
        total_tokens: 8,
        // Includes the compaction notice: this runner has a context limit, so it compacts.
        context_tokens: 59,
        max_context: 128_000,
        mode: 'code',
        plan_mode: false,
      },
    },
  ])
  // This fixture completes in under 100ms; its event burst is not a decode-rate sample.
  expect(events[1]?.payload).not.toHaveProperty('tokens_per_second')
  expect(runner.stateFor('session-1')?.messages.map(message => message.role)).toEqual(['user', 'assistant'])
  expect(session.extra.runtime_telemetry).toMatchObject({
    cacheTelemetryKnown: false,
    llmSteps: 1,
    llmDurationMs: expect.any(Number),
    toolSteps: 0,
    toolDurationMs: 0,
    ttftSamples: 1,
    ttftTotalMs: expect.any(Number),
  })
})

test('agent turn runner keeps live context monotonic when a later round is fully cached', async () => {
  let round = 0
  const runner = new AgentTurnRunner({
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        round += 1
        if (round === 1) {
          yield {
            toolCalls: [{
              id: 'cached-read',
              type: 'function',
              function: { name: 'ReadFile', arguments: {} },
            }],
            usage: { inputTokens: 100, outputTokens: 10 },
          }
          return
        }
        yield {
          content: 'done',
          usage: { inputTokens: 5, outputTokens: 2, cacheReadTokens: 100 },
        }
      },
    },
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    toolExecutor: { execute: async () => 'file contents' },
    tools: [repeatedReadTool],
  })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {},
    id: 'cached-context', interactionMode: 'code', sessionKey: 'cached-context', lastActive: 0,
    messages: [], metadata: {}, model: 'gpt-4o', planMode: false, status: 'working', thinkingContent: [],
    toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: '/tmp/agents/default',
  }
  const events: DaemonEvent[] = []

  for await (const event of runner.run(session, 'read it', new AbortController().signal)) events.push(event)

  const liveContext = events
    .filter(event => event.type === 'status_update' && event.payload.usage_complete === undefined && event.payload.context_tokens !== undefined)
    .map(event => Number(event.payload.context_tokens))
  expect(liveContext).toEqual([110, 107])
  expect(session.extra.runtime_telemetry).toMatchObject({
    cacheTelemetryKnown: true,
    cacheReadTokens: 100,
    inputTokens: 5,
    cacheHitRate: 100 / 105,
  })
})

test('agent turn runner forwards tool arguments into capability refinement', async () => {
  let providerRound = 0
  const observed: Array<Readonly<Record<string, unknown>> | undefined> = []
  const tool: ToolDefinition = {
    type: 'function',
    function: { name: 'ReadFile', description: 'Read', parameters: {} },
  }
  const runner = new AgentTurnRunner({
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        providerRound += 1
        if (providerRound === 1) {
          yield {
            toolCalls: [{
              id: 'read-args',
              type: 'function',
              function: { name: 'ReadFile', arguments: { file_path: 'README.md' } },
            }],
          }
          return
        }
        yield { content: 'done' }
      },
    },
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    toolCapabilities: (_name, _agentId, args) => {
      observed.push(args)
      return { concurrencySafe: true, interruptBehavior: 'cancel' }
    },
    toolExecutor: { execute: async () => 'body' },
    tools: [tool],
  })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {},
    id: 'capability-args', interactionMode: 'code', sessionKey: 'capability-args', lastActive: 0,
    messages: [], metadata: {}, model: 'gpt-4o', planMode: false, status: 'working', thinkingContent: [],
    toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: '/tmp/agents/default',
  }

  for await (const _event of runner.run(session, 'read it', new AbortController().signal)) {
    // Drain the complete turn.
  }

  expect(observed).toContainEqual({ file_path: 'README.md' })
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
  expect(readContextDeltas(activeSession.metadata)).toEqual([
    expect.objectContaining({ layer: 'interaction-mode', value: 'plan' }),
  ])
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
    // The fully duplicated round persists no empty assistant message, so the
    // provider never receives empty assistant content on the next request.
    expect(live.messages.filter(message => message.role === 'assistant').map(message => message.content)).toEqual([
      'Reading now.',
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
    expect(client.requests[0]?.tools?.map(tool => tool.function.name)).toEqual(['ReadFile', 'SetInteractionModeTool'])
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

test('agent turn runner keeps generated prompts request-only while preserving caller-owned system messages', async () => {
  const client = new CapturingClient()
  const runner = new AgentTurnRunner({
    bootstrapSystemPrompt: () => 'generated daemon bootstrap',
    llm: client,
    model: 'gpt-4o',
  })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: '/workspace/request-only', extra: {},
    id: 'request-only-session', interactionMode: 'code', sessionKey: 'request-only', lastActive: 0,
    messages: [{ role: 'system', content: 'caller-owned system instruction' }], metadata: {}, model: 'gpt-4o',
    planMode: false, status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0,
    totalOutputTokens: 0, turnCount: 0, workspace: '/tmp/agents/default',
  }

  for await (const _event of runner.run(session, 'first', new AbortController().signal)) {}
  for await (const _event of runner.run(session, 'second', new AbortController().signal)) {}

  expect(client.requests).toHaveLength(2)
  for (const request of client.requests) {
    expect(request.messages.filter(message => message.role === 'system')).toEqual([
      { role: 'system', content: 'generated daemon bootstrap' },
      { role: 'system', content: 'caller-owned system instruction' },
    ])
  }
  expect(session.messages.filter(message => message.role === 'system')).toEqual([
    { role: 'system', content: 'caller-owned system instruction' },
  ])
})

test('agent turn runner preserves legacy system messages without guessing their provenance', async () => {
  const client = new CapturingClient()
  const runner = new AgentTurnRunner({
    bootstrapSystemPrompt: () => 'generated daemon bootstrap',
    llm: client,
    model: 'gpt-4o',
  })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: '/workspace/legacy-prompt', extra: {},
    id: 'legacy-prompt-session', interactionMode: 'code', sessionKey: 'legacy-prompt', lastActive: 0,
    messages: [
      { role: 'system', content: 'generated daemon bootstrap' },
      { role: 'system', content: 'caller-owned system instruction' },
      { role: 'user', content: 'old request' },
      { role: 'assistant', content: 'old response' },
    ], metadata: {}, model: 'gpt-4o', planMode: false, status: 'working', thinkingContent: [], toolExecutions: [],
    totalInputTokens: 0, totalOutputTokens: 0, turnCount: 1, workspace: '/tmp/agents/default',
  }

  for await (const _event of runner.run(session, 'continue', new AbortController().signal)) {}

  expect(client.requests[0]?.messages.filter(message => message.role === 'system')).toEqual([
    { role: 'system', content: 'generated daemon bootstrap' },
    { role: 'system', content: 'generated daemon bootstrap' },
    { role: 'system', content: 'caller-owned system instruction' },
  ])
  expect(session.messages.filter(message => message.role === 'system')).toEqual([
    { role: 'system', content: 'generated daemon bootstrap' },
    { role: 'system', content: 'caller-owned system instruction' },
  ])
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

test('agent turn runner rekeys its bootstrap cache when plan mode or the addendum changes', async () => {
  const client = new CapturingClient()
  let bootstrapCalls = 0
  const runner = new AgentTurnRunner({
    bootstrapSystemPrompt: ({ session }) => {
      bootstrapCalls += 1
      return `Bootstrap plan=${session.planMode === true} addendum=${session.systemPromptAddendum ?? ''}`
    },
    llm: client,
    model: 'gpt-4o',
  })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: '/workspace/rekey', extra: {},
    id: 'rekey-session', interactionMode: 'code', sessionKey: 'rekey', lastActive: 0, messages: [],
    metadata: {}, model: 'gpt-4o', planMode: false, status: 'working', thinkingContent: [], toolExecutions: [],
    totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0, workspace: '/tmp/agents/default',
  }
  for await (const _event of runner.run(session, 'first', new AbortController().signal)) {
    // The provider requests below are the observable prompt boundary.
  }
  for await (const _event of runner.run(session, 'second', new AbortController().signal)) {
    // An unchanged session keeps reusing the cached bootstrap prompt.
  }
  expect(bootstrapCalls).toBe(1)

  session.planMode = true
  for await (const _event of runner.run(session, 'third', new AbortController().signal)) {
    // Plan mode is session state the provider sees, so it rekeys the cache.
  }
  expect(bootstrapCalls).toBe(2)
  expect(client.requests[2]?.messages[0]?.content).toContain('plan=true')

  session.systemPromptAddendum = 'Channel workspace notes.'
  for await (const _event of runner.run(session, 'fourth', new AbortController().signal)) {
    // The trusted addendum is session state too.
  }
  expect(bootstrapCalls).toBe(3)
  expect(client.requests[3]?.messages[0]?.content).toContain('addendum=Channel workspace notes.')
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

test('the system prompt stays byte-identical across turns; memory written mid-session rides with the next message', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-runner-stable-prefix-'))
  try {
    const memory = new AgentMemory({ globalDirectory: join(root, 'global'), projectRoot: root })
    await memory.write('global', 'EXPERIENCES.md', 'Bun tests run offline.')
    const selfMemory = new AgentSelfMemory({ agentId: 'default', directory: join(root, 'self-memory'), projectRoot: root })
    await selfMemory.learn('The user prefers direct status reports', 'user_taste')
    const client = new CapturingClient()
    const runner = new AgentTurnRunner({ agentMemory: () => memory, agentSelfMemory: () => selfMemory, llm: client, model: 'gpt-4o' })
    const session: DaemonSession = {
      activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: root, extra: {}, id: 'stable-prefix-session',
      interactionMode: 'code', sessionKey: 'stable-prefix', lastActive: 0, messages: [], metadata: {}, model: 'gpt-4o', planMode: false,
      status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
      workspace: '/tmp/agents/default',
    }
    const turn = async (text: string) => { for await (const _event of runner.run(session, text, new AbortController().signal)) { /* provider requests are asserted below */ } }
    await turn('first question')
    // What the agent (or another session, through the shared global file) writes between turns.
    await memory.append('global', 'EXPERIENCES.md', 'Claude Code quotes of tool-call tags are parsed as calls.')
    await selfMemory.learn('The user wants numbers, not adjectives', 'user_taste')
    await turn('second question')
    await turn('third question')

    const [first, second, third] = client.requests
    const systemOf = (request: CompletionRequest | undefined) => request?.messages.filter(message => message.role === 'system').map(message => message.content)
    // The cached prefix survives every turn: the system prompt never moves.
    expect(systemOf(second)).toEqual(systemOf(first))
    expect(systemOf(third)).toEqual(systemOf(first))
    expect(String(systemOf(first))).toContain('Bun tests run offline.')
    expect(String(systemOf(second))).not.toContain('tool-call tags')
    // Turn 1 replays byte for byte in turn 2's request.
    const conversation = (request: CompletionRequest | undefined) => request?.messages.filter(message => message.role !== 'system') ?? []
    expect(conversation(second).slice(0, 2) as unknown[]).toEqual([...conversation(first), { role: 'assistant', content: 'configured agent reply' }].slice(0, 2))
    // The new memory reaches the model with the next message, once.
    const secondUser = String(conversation(second).at(-1)?.content)
    expect(secondUser.startsWith('<turn-context>')).toBe(true)
    expect(secondUser).toContain('Claude Code quotes of tool-call tags are parsed as calls.')
    expect(secondUser).not.toContain('Bun tests run offline.')
    expect(secondUser).toContain('The user wants numbers, not adjectives')
    expect(secondUser.endsWith('second question')).toBe(true)
    expect(String(conversation(first).at(-1)?.content)).toBe('first question')
    expect(String(conversation(third).at(-1)?.content)).toBe('third question')
    // The transcript still shows only what the user typed.
    const secondRecord = session.messages.find(message => message.role === 'user' && String(message.content).endsWith('second question')) as Record<string, unknown> | undefined
    expect(secondRecord?.displayText ?? secondRecord?.text).toBe('second question')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('session memory controls reach provider assembly without removing mandatory context', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-runner-controls-'))
  try {
    const memory = new AgentMemory({ globalDirectory: join(root, 'global'), projectRoot: root })
    await memory.write('project', 'MEMORY.md', 'EXCLUDED_PRIVATE_FACT')
    await memory.write('project', 'KNOWLEDGE.md', 'VISIBLE_KNOWLEDGE')
    const client = new CapturingClient()
    const runner = new AgentTurnRunner({ agentMemory: () => memory, llm: client, model: 'gpt-4o' })
    const session: DaemonSession = {
      activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: root, extra: {}, id: 'controls-session',
      interactionMode: 'code', sessionKey: 'controls', lastActive: 0, messages: [],
      metadata: { context_controls: { version: 1, revision: 1, pins: [{ scope: 'project', path: 'saved.md', content: 'PINNED_FACT' }], excluded: [{ scope: 'project', path: 'MEMORY.md' }] } },
      model: 'gpt-4o', planMode: false, status: 'working', thinkingContent: [], toolExecutions: [],
      totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0, workspace: '/tmp/agents/default',
      systemPromptAddendum: 'MANDATORY_CONTEXT',
    }
    for await (const _event of runner.run(session, 'recall', new AbortController().signal)) { /* observe provider below */ }
    const system = String(client.requests[0]?.messages[0]?.content)
    expect(system).toContain('PINNED_FACT')
    expect(system).toContain('VISIBLE_KNOWLEDGE')
    expect(system).toContain('MANDATORY_CONTEXT')
    expect(system).not.toContain('EXCLUDED_PRIVATE_FACT')
    expect(session.requestScaffold?.memorySources?.some(source => source.path === 'KNOWLEDGE.md')).toBe(true)
    expect(session.requestScaffold?.memorySources?.some(source => source.scope === 'project' && source.path === 'MEMORY.md')).toBe(false)
  } finally { await rm(root, { recursive: true, force: true }) }
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
    expect(session.extra.runtime_telemetry).toMatchObject({
      llmSteps: 0,
      toolSteps: 1,
      toolDurationMs: expect.any(Number),
    })
  } finally {
    release()
  }
})

test('agent turn runner preserves a mode delta appended while its turn is active', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-mode-race-'))
  const llm = new GatedClient()
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm, model: 'gpt-4o' }), {
    currentProjectDirectory: directory,
    model: 'gpt-4o',
    sessionDirectory: join(directory, 'sessions'),
  })
  try {
    const active = runtime.submitTurn('tui:mode-race', 'first prompt', () => {})
    await waitForCondition(() => llm.started)
    await runtime.setSessionMode('tui:mode-race', 'researcher')
    llm.release()
    await active

    const session = runtime.sessionStatus('tui:mode-race')
    if (!session) throw new Error('expected a live session')
    expect(readContextDeltas(session.metadata)).toEqual([
      expect.objectContaining({ layer: 'interaction-mode', value: 'researcher' }),
    ])
  } finally {
    llm.release()
    await rm(directory, { recursive: true, force: true })
  }
})

test('agent turn runner consumes a queued context delta exactly once', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-mode-once-'))
  const llm = new CapturingClient()
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm, model: 'gpt-4o' }), {
    currentProjectDirectory: directory,
    model: 'gpt-4o',
    sessionDirectory: join(directory, 'sessions'),
  })
  try {
    const opened = await runtime.openSession('tui:mode-once')
    appendContextDelta(opened.metadata, { at: 1, layer: 'interaction-mode', value: 'researcher' })

    await runtime.submitTurn('tui:mode-once', 'first prompt', () => {})
    const session = runtime.sessionStatus('tui:mode-once')
    if (!session) throw new Error('expected a live session')
    // Delivered with the message it applies to, not in the system prompt.
    const latestUser = (index: number) => String(llm.requests[index]?.messages.filter(message => message.role === 'user').at(-1)?.content)
    expect(latestUser(0)).toContain('[Context updated]\n- interaction mode: researcher')
    expect(llm.requests[0]?.messages.some(message => message.role === 'system' && String(message.content).includes('[Context updated]'))).toBe(false)
    expect(readContextDeltas(session.metadata)).toEqual([])

    await runtime.submitTurn('tui:mode-once', 'second prompt', () => {})
    // Once: the second message carries nothing; the first replays unchanged.
    expect(latestUser(1)).toBe('second prompt')
    expect(llm.requests[1]?.messages.filter(message => message.content.toString().includes('[Context updated]'))).toHaveLength(1)
    expect(readContextDeltas(session.metadata)).toEqual([])
  } finally {
    await rm(directory, { recursive: true, force: true })
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

test('agent turn runner resolves per-turn thinking from keywords, ultra mode, and session defaults', async () => {
  const sessionFor = (id: string, ultra = false): DaemonSession => ({
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {}, id,
    interactionMode: 'code', sessionKey: id, lastActive: 0, messages: [], metadata: {}, model: 'gpt-4o',
    planMode: false, status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0,
    totalOutputTokens: 0, turnCount: 0, workspace: '/tmp/agents/default',
    ...(ultra ? { ultraMode: true } : {}),
  })
  const drain = async (events: AsyncIterable<DaemonEvent>): Promise<void> => {
    for await (const _ of events) void _
  }

  const keyword = new CapturingClient()
  await drain(new AgentTurnRunner({ llm: keyword, model: 'gpt-4o' })
    .run(sessionFor('t-keyword'), 'please ultrathink this change', new AbortController().signal))
  expect(keyword.requests[0]?.thinking).toEqual({ budgetTokens: 32_000, effort: 'high' })

  const ultra = new CapturingClient()
  await drain(new AgentTurnRunner({ llm: ultra, model: 'gpt-4o' })
    .run(sessionFor('t-ultra', true), 'a plain prompt', new AbortController().signal))
  expect(ultra.requests[0]?.thinking).toEqual({ budgetTokens: 32_000, effort: 'high' })

  const defaults = new CapturingClient()
  await drain(new AgentTurnRunner({
    llm: defaults,
    model: 'gpt-4o',
    reasoningEffort: 'high',
    thinking: true,
    thinkingBudget: 24_576,
  }).run(sessionFor('t-defaults'), 'a plain prompt', new AbortController().signal))
  expect(defaults.requests[0]?.thinking).toEqual({ budgetTokens: 24_576, effort: 'high' })

  const off = new CapturingClient()
  await drain(new AgentTurnRunner({ llm: off, model: 'gpt-4o' })
    .run(sessionFor('t-off'), 'a plain prompt', new AbortController().signal))
  expect(off.requests[0]?.thinking).toBeUndefined()
})

test('agent turn runner persists per-message journal entries for crash recovery', async () => {
  const runner = new AgentTurnRunner({ llm: new TextClient(), model: 'gpt-4o' })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {},
    id: 'journal-session', interactionMode: 'code', sessionKey: 'journal-session', lastActive: 0,
    messages: [], metadata: {}, model: 'gpt-4o', planMode: false, status: 'working',
    thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: '/tmp/agents/default',
  }
  const journalEntries: Array<{ readonly index: number; readonly role: unknown }> = []
  const journal: TranscriptMessageJournalAppend = (message: RawMessage, index: number) => {
    journalEntries.push({ index, role: message.role })
  }

  for await (const _event of runner.run(session, 'journal this turn', new AbortController().signal, { journal })) {
    // Consume the turn so every append has been recorded.
  }

  expect(journalEntries).toEqual([
    { index: 0, role: 'user' },
    { index: 1, role: 'assistant' },
  ])
  expect(session.messages.map(message => message.role)).toEqual(['user', 'assistant'])
})

test('deferred tool loading sends the core surface, not every registered schema', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-deferred-tools-'))
  const tool = (name: string): ToolDefinition => ({
    type: 'function',
    function: { name, description: name, parameters: { type: 'object', properties: {} } },
  })
  // Two always-loaded core tools plus a long tail of deferrable ones — the
  // shape of the real surface, where 76 schemas shipped on every request and
  // models started borrowing one tool's arguments for another.
  const registry = new ToolRegistry({ deferredToolLoading: true })
  registry.register(tool('ReadFile'), async () => 'ok')
  registry.register(tool('ToolSearchTool'), async () => 'ok')
  for (let index = 0; index < 40; index += 1) {
    registry.register(tool(`Deferrable${index}`), async () => 'ok')
  }

  const client = new CapturingClient()
  const runtime = new InMemoryDaemonRuntime(
    new AgentTurnRunner({
      llm: client,
      model: 'gpt-4o',
      toolRegistry: registry,
      tools: registry.definitions(),
    }),
    { currentProjectDirectory: directory, model: 'gpt-4o', sessionDirectory: join(directory, 'sessions') },
  )

  try {
    await runtime.openSession('deferred')
    await runtime.submitTurn('deferred', 'hello', () => {})

    const sent = (client.requests[0]?.tools ?? []).map(entry => entry.function.name)
    expect(sent).toContain('ReadFile')
    expect(sent).toContain('ToolSearchTool')
    // The whole point: the deferrable tail is not on the wire. Before this was
    // wired, every production call site used definitions(), so the flag existed
    // but could not change what a request carried.
    expect(sent.some(name => name.startsWith('Deferrable'))).toBe(false)
    expect(sent.length).toBeLessThan(registry.definitions().length)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

class OverloadedThenOkClient implements LlmClient {
  async *stream(): AsyncGenerator<LlmDelta> {
    throw new Error('HTTP 529: provider is overloaded')
  }
}

class FallbackTextClient implements LlmClient {
  async *stream(): AsyncGenerator<LlmDelta> {
    yield { content: 'fallback model reply' }
  }
}

test('agent turn runner restarts a pre-content overload on the configured fallback model', async () => {
  const runner = new AgentTurnRunner({
    createLlmForModel: model => model === 'fallback-model' ? new FallbackTextClient() : new TextClient(),
    fallbackModel: 'fallback-model',
    llm: new OverloadedThenOkClient(),
    model: 'primary-model',
    // Fast local-route schedule; the production 5x10s cadence is asserted in
    // retryPolicies.test.ts, not waited out here.
    providerOverrides: { provider: 'ollama' },
  })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {},
    id: 'fallback-session', interactionMode: 'code', sessionKey: 'fallback-key', lastActive: 0,
    messages: [], metadata: {}, model: 'primary-model', planMode: false, status: 'working',
    thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0,
    turnCount: 0, workspace: '/tmp/agents/default',
  }
  const events: { type: string, payload: Record<string, unknown> }[] = []
  for await (const event of runner.run(session, 'hi', new AbortController().signal)) {
    events.push(event as { type: string, payload: Record<string, unknown> })
  }
  const texts = events.filter(event => event.type === 'text_part').map(event => String(event.payload.text ?? ''))
  expect(texts.join('')).toContain('fallback model reply')
  expect(texts.join('')).not.toContain('[Error:')
  const notice = events.find(event => event.type === 'notification' && String(event.payload.message ?? '').includes('fallback model'))
  expect(notice).toBeDefined()
})

test('agent turn runner does not fall back once content has streamed', async () => {
  class HalfStreamedThenOverload implements LlmClient {
    async *stream(): AsyncGenerator<LlmDelta> {
      yield { content: 'partial ' }
      throw new Error('HTTP 529: provider is overloaded')
    }
  }
  const runner = new AgentTurnRunner({
    createLlmForModel: () => new FallbackTextClient(),
    fallbackModel: 'fallback-model',
    llm: new HalfStreamedThenOverload(),
    model: 'primary-model',
    providerOverrides: { provider: 'ollama' },
  })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {},
    id: 'no-fallback-session', interactionMode: 'code', sessionKey: 'no-fallback-key', lastActive: 0,
    messages: [], metadata: {}, model: 'primary-model', planMode: false, status: 'working',
    thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0,
    turnCount: 0, workspace: '/tmp/agents/default',
  }
  const texts: string[] = []
  for await (const event of runner.run(session, 'hi', new AbortController().signal)) {
    if (event.type === 'text_part') texts.push(String((event.payload as { text?: string }).text ?? ''))
  }
  const joined = texts.join('')
  expect(joined).toContain('partial')
  expect(joined).not.toContain('fallback model reply')
})

test('background turn origins do not inherit human or previous goal-round authority', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-turn-origin-'))
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: new TextClient(), model: 'test-model' }), {
    currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions'),
  })
  try {
    const session = await runtime.openSession('origin-test')
    await runtime.submitTurn('origin-test', 'human task', () => {})
    expect(session.metadata.goal_turn_human).toBe(true)
    for (const origin of ['monitor', 'schedule'] as const) {
      session.metadata.goal_turn_round = 99
      await runtime.submitTurn('origin-test', 'background evidence', () => {}, { origin })
      expect(session.metadata.goal_turn_human).toBe(false)
      expect(session.metadata.goal_turn_round).toBeUndefined()
      expect(session.metadata.turn_origin).toBe(origin)
    }
    await expect(runtime.submitTurn('origin-test', 'invalid', () => {}, { origin: 'monitor', goalRound: 1 })).rejects.toThrow('authority')
    await runtime.submitTurn('origin-test', 'human again', () => {})
    expect(session.metadata.goal_turn_human).toBe(true)
    expect(session.metadata.turn_origin).toBe('human')
  } finally { await runtime.shutdown(); await rm(directory, { recursive: true, force: true }) }
})

test('agent runner exposes output-limit stop reason through the daemon runtime', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-stop-reason-'))
  const runner = new AgentTurnRunner({ model: 'fixture-model', llm: { async *stream() { yield { content: 'Partial', finishReason: 'length' as const } } } })
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') })
  try {
    const events: DaemonEvent[] = []
    await runtime.openSession('limited')
    await runtime.submitTurn('limited', 'Continue', event => events.push(event))
    expect(events).toContainEqual(expect.objectContaining({ type: 'status_update', payload: expect.objectContaining({ stop_reason: 'output_limit' }) }))
  } finally { await runtime.shutdown(); await rm(directory, { recursive: true, force: true }) }
})

test('runtime tool inventory reflects live registration, deferral and agent filtering without calling the provider', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-tool-inventory-'))
  const registry = new ToolRegistry({ deferredToolLoading: true })
  for (const name of ['ReadFile', 'WriteFile', 'Special']) registry.register({ type: 'function', function: { name, description: name, parameters: {} } }, async () => 'ok')
  const client = new CapturingClient()
  const definition: AgentDefinition = { name: 'reviewer', description: '', systemPrompt: '', model: '', tools: [], allowedTools: null, excludeTools: ['WriteFile'], source: 'test', maxDepth: 3, isolation: '' }
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: client, model: 'fixture', toolRegistry: registry, tools: registry.definitions(), agentDefinitions: new Map([['reviewer', definition]]) }), { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') })
  try {
    await runtime.openSession('inventory')
    const session = runtime.sessionStatus('inventory')!
    session.agentId = 'reviewer'
    expect(runtime.toolInventory('missing')).toBeUndefined()
    expect(runtime.toolInventory('inventory')).toEqual([
      expect.objectContaining({ name: 'ReadFile', exposure: 'loaded' }),
      expect.objectContaining({ name: 'WriteFile', exposure: 'filtered' }),
      expect.objectContaining({ name: 'Special', exposure: 'deferred' }),
    ])
    session.messages.push({ role: 'tool', content: '{"loaded_tool":"Special"}', tool_call_id: 'load' })
    expect(runtime.toolInventory('inventory')?.find(tool => tool.name === 'Special')?.exposure).toBe('loaded')
    registry.unregister('Special')
    expect(runtime.toolInventory('inventory')?.some(tool => tool.name === 'Special')).toBe(false)
    const staticRegistry = new ToolRegistry()
    staticRegistry.register({ type: 'function', function: { name: 'ReadFile', description: '', parameters: {} } }, async () => 'ok')
    const staticRunner = new AgentTurnRunner({ llm: client, model: 'fixture', toolRegistry: staticRegistry, tools: [] })
    expect(staticRunner.toolInventory(session)[0]?.exposure).toBe('unexposed')
    session.interactionMode = 'plan'
    expect(() => runtime.toolInventory('inventory')).toThrow('enforcement profile')
    expect(client.requests).toHaveLength(0)
  } finally { await runtime.shutdown(); await rm(directory, { recursive: true, force: true }) }
})

test('resume reconciles saved active agents against exact live worker ownership', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'agent-ownership-'))
  try {
    const options = { sessionDirectory: directory, currentProjectDirectory: directory }
    const original = new InMemoryDaemonRuntime(undefined, options)
    const session = await original.openSession('ownership')
    session.messages.push({ role: 'user', content: 'important work' })
    session.messages.push({ role: 'assistant', content: 'Working on it.' })
    session.metadata.xerxes_subagent_snapshots_v1 = [
      { id: 'dead', status: 'running', summary: 'partial work' },
      { id: 'alive', status: 'running' },
      { id: 'done', status: 'completed', summary: 'finished report' },
    ]
    await original.flushSessions()
    const restored = new InMemoryDaemonRuntime(undefined, { ...options, liveSubagentIds: () => ['alive'] })
    const resumed = await restored.openSession(session.id, undefined, { resume: true })
    expect(resumed.metadata.xerxes_subagent_snapshots_v1).toMatchObject([
      { id: 'dead', status: 'interrupted', summary: 'partial work' },
      { id: 'alive', status: 'running' },
      { id: 'done', status: 'completed', summary: 'finished report' },
    ])
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('session telemetry keeps cache writes, and a step that rewrote the cache is a miss, not a hit', async () => {
  const steps = [
    { inputTokens: 10, outputTokens: 5, cacheReadTokens: 0, cacheCreationTokens: 90_000 },
    { inputTokens: 10, outputTokens: 5, cacheReadTokens: 90_000, cacheCreationTokens: 100 },
  ]
  let call = 0
  const client: LlmClient = { async *stream() { yield { content: 'ok' }; yield { usage: steps[call++]! } } }
  const runner = new AgentTurnRunner({ llm: client, model: 'gpt-4o' })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {}, id: 'cache-telemetry',
    interactionMode: 'code', sessionKey: 'cache-telemetry', lastActive: 0, messages: [], metadata: {}, model: 'gpt-4o', planMode: false,
    status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: '/tmp/agents/default',
  }
  for await (const _event of runner.run(session, 'one', new AbortController().signal)) { /* telemetry below */ }
  // A full rewrite used to read as a 100% hit: writes were left out of the rate.
  expect(session.extra.runtime_telemetry).toMatchObject({ cacheTelemetryKnown: true, cacheReadTokens: 0, cacheWriteTokens: 90_000, cacheHitRate: 0 })
  for await (const _event of runner.run(session, 'two', new AbortController().signal)) { /* telemetry below */ }
  const telemetry = session.extra.runtime_telemetry as { cacheWriteTokens: number; cacheHitRate: number }
  expect(telemetry.cacheWriteTokens).toBe(90_100)
  expect(telemetry.cacheHitRate).toBeCloseTo(90_000 / (20 + 90_000 + 90_100), 6)
})

test('goal status travels with the message when it changes; the goal rules stay in a fixed system prompt', async () => {
  const client = new CapturingClient()
  const updateGoal = { type: 'function' as const, function: { name: 'update_goal', description: 'Update the goal.', parameters: { type: 'object', properties: {} } } }
  const runner = new AgentTurnRunner({ llm: client, model: 'gpt-4o', tools: [updateGoal] })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {}, id: 'goal-status-session',
    interactionMode: 'code', sessionKey: 'goal-status', lastActive: 0, messages: [], metadata: {}, model: 'gpt-4o', planMode: false,
    status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: '/tmp/agents/default',
  }
  const turn = async (text: string) => { for await (const _event of runner.run(session, text, new AbortController().signal)) { /* requests below */ } }
  await turn('one')
  const goal = createGoal(session.metadata, session.id, { objective: 'Ship the cache fix' }, 1_000)
  await turn('two')
  await turn('three')
  editGoal(session.metadata, session.id, goal, { objective: 'Ship the cache fix and measure it' }, 2_000)
  await turn('four')

  const system = client.requests.map(request => request.messages.filter(message => message.role === 'system').map(message => String(message.content)).join('\n'))
  expect(new Set(system).size).toBe(1)
  expect(system[0]).not.toContain('Ship the cache fix')
  const latest = client.requests.map(request => String(request.messages.filter(message => message.role === 'user').at(-1)?.content))
  // No goal is the default: nothing to announce.
  expect(latest[0]).toBe('one')
  expect(latest[1]).toContain('Current goal: "Ship the cache fix"')
  expect(latest[2]).toBe('three')
  expect(latest[3]).toContain('Ship the cache fix and measure it')
})

test('a rebuilt runner keeps each session\'s system prompt byte-identical; a new day is told, not re-rendered', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-runner-rebuild-'))
  try {
    const memory = new AgentMemory({ globalDirectory: join(root, 'global'), projectRoot: root })
    await memory.write('global', 'EXPERIENCES.md', 'First note.')
    const client = new CapturingClient()
    const snapshots = new SessionPromptSnapshots()
    let bootstraps = 0
    // A settings change (/fast, /permissions, profile switch) builds a fresh runner.
    const build = () => new AgentTurnRunner({
      agentMemory: () => memory, llm: client, model: 'gpt-4o', promptSnapshots: snapshots,
      bootstrapSystemPrompt: () => `bootstrap #${++bootstraps}`,
    })
    const session: DaemonSession = {
      activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: root, extra: {}, id: 'rebuild-session',
      interactionMode: 'code', sessionKey: 'rebuild', lastActive: 0, messages: [], metadata: {}, model: 'gpt-4o', planMode: false,
      status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
      workspace: '/tmp/agents/default',
    }
    for await (const _event of build().run(session, 'one', new AbortController().signal)) { /* requests below */ }
    await memory.append('global', 'EXPERIENCES.md', 'Written after the first turn.')
    for await (const _event of build().run(session, 'two', new AbortController().signal)) { /* requests below */ }
    // A day passes.
    snapshots.promptDay.set(session.id, '1999-01-01 Friday')
    for await (const _event of build().run(session, 'three', new AbortController().signal)) { /* requests below */ }

    const system = client.requests.map(request => request.messages.filter(message => message.role === 'system').map(message => String(message.content)).join('\n'))
    expect(new Set(system).size).toBe(1)
    expect(bootstraps).toBe(1)
    const latest = client.requests.map(request => String(request.messages.filter(message => message.role === 'user').at(-1)?.content))
    expect(latest[1]).toContain('Written after the first turn.')
    expect(latest[2]).toContain("Today's date is now")
    expect(latest[2]).not.toContain('Written after the first turn.')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('loading a deferred tool appends it to the tools array and leaves the system prompt byte-identical', async () => {
  const { TOOL_SEARCH_LOADED_KEY } = await import('../src/executors/toolRegistry.js')
  const tool = (name: string) => ({ type: 'function' as const, function: { name, description: `${name} does one thing.`, parameters: { type: 'object', properties: {} } } })
  const registry = new ToolRegistry({ deferredToolLoading: true })
  registry.register(tool('ReadFile'), async () => 'ok')
  registry.register(tool('ToolSearchTool'), async () => 'ok')
  for (const name of ['Alpha', 'Beta', 'Gamma']) registry.register(tool(name), async () => 'ok')
  registry.register(tool('WriteFile'), async () => 'ok')
  const client = new CapturingClient()
  const runner = new AgentTurnRunner({ llm: client, model: 'gpt-4o', toolRegistry: registry, tools: registry.definitions(), bootstrapSystemPrompt: ({ tools }) => `bootstrap listing ${(tools ?? []).map(entry => entry.function.name).join(',')}` })
  const session: DaemonSession = {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: process.cwd(), extra: {}, id: 'deferred-prefix',
    interactionMode: 'code', sessionKey: 'deferred-prefix', lastActive: 0, messages: [], metadata: {}, model: 'gpt-4o', planMode: false,
    status: 'working', thinkingContent: [], toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: '/tmp/agents/default',
  }
  for await (const _event of runner.run(session, 'one', new AbortController().signal)) { /* requests below */ }
  // A ToolSearchTool result loaded Gamma, then Alpha (in that order).
  session.messages.push(
    { role: 'assistant', content: '', tool_calls: [{ id: 's1', type: 'function', function: { name: 'ToolSearchTool', arguments: { query: 'Gamma Alpha' } } }] },
    { role: 'tool', tool_call_id: 's1', name: 'ToolSearchTool', content: `[{"${TOOL_SEARCH_LOADED_KEY}":"Gamma"},{"${TOOL_SEARCH_LOADED_KEY}":"Alpha"}]` },
  )
  for await (const _event of runner.run(session, 'two', new AbortController().signal)) { /* requests below */ }
  const [first, second] = client.requests
  const names = (request: CompletionRequest | undefined) => (request?.tools ?? []).map(entry => entry.function.name)
  const system = (request: CompletionRequest | undefined) => request?.messages.filter(message => message.role === 'system').map(message => String(message.content)).join('\n')
  // The earlier tools keep their positions; the loaded ones follow, in load order.
  expect(names(second).slice(0, names(first).length)).toEqual(names(first))
  expect(names(second).slice(names(first).length)).toEqual(['Gamma', 'Alpha'])
  // Catalog, bootstrap and guidance describe the fixed core: no re-render.
  expect(system(second)).toBe(system(first))
})

test('a harness-written prompt is tagged with its origin, live and in the saved transcript', async () => {
  // Goal rounds used to render in the desktop as if the user had typed
  // "Goal round 1/unlimited — …".
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-harness-origin-'))
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: new TextClient(), model: 'test-model' }), {
    currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions'),
  })
  try {
    const session = await runtime.openSession('origin-tag')
    const begins: Array<Record<string, unknown>> = []
    const onEvent = (event: DaemonEvent) => { if (event.type === 'turn_begin') begins.push(event.payload) }
    await runtime.submitTurn('origin-tag', 'human task', onEvent)
    await runtime.submitTurn('origin-tag', 'Goal round 1/unlimited — ship it', onEvent, { goalRound: 1 })
    await runtime.submitTurn('origin-tag', 'monitor saw a failure', onEvent, { origin: 'monitor' })
    expect(begins.map(payload => payload.origin)).toEqual([undefined, 'goal', 'monitor'])
    expect(begins[1]!.goal_round).toBe(1)
    const users = session.messages.filter(message => message.role === 'user') as Array<{ origin?: string }>
    expect(users.map(message => message.origin)).toEqual([undefined, 'goal', 'monitor'])
  } finally { await runtime.shutdown(); await rm(directory, { recursive: true, force: true }) }
})

test('a prompt carries its display text from the moment it is appended, not only at turn end', async () => {
  // A client that reloaded history mid-turn used to get the provider form —
  // the model-only <turn-context> block — as the user's message.
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-display-on-append-'))
  let seenMidTurn: Array<Record<string, unknown>> = []
  const llm: LlmClient = {
    async *stream(request) {
      seenMidTurn = request.messages.filter(message => message.role === 'user') as unknown as Array<Record<string, unknown>>
      yield { content: 'ok', usage: { inputTokens: 1, outputTokens: 1 } }
    },
  }
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm, model: 'test-model' }), {
    currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions'),
  })
  try {
    await runtime.openSession('display-on-append')
    // The provider reads the expanded prompt; the transcript shows what was typed.
    await runtime.submitTurn('display-on-append', 'expanded prompt with attached file contents', () => {}, { displayText: 'look at @a.ts' })
    const typed = seenMidTurn.at(-1)!
    expect(String(typed.content)).toContain('expanded prompt')
    expect(typed.displayText).toBe('look at @a.ts')
    expect(typed.origin).toBeUndefined()
    await runtime.submitTurn('display-on-append', 'Goal round 1/unlimited — ship it', () => {}, { goalRound: 1 })
    expect(seenMidTurn.at(-1)!.origin).toBe('goal')
  } finally { await runtime.shutdown(); await rm(directory, { recursive: true, force: true }) }
})
