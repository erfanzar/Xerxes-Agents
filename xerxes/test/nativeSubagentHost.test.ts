// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { BUILTIN_AGENTS, type AgentDefinition } from '../src/agents/definitions.js'
import {
  persistedSubagentSnapshotValues,
  replacePersistedSubagentSnapshots,
} from '../src/agents/subagentPersistence.js'
import {
  NativeSubagentTurnCoordinator,
  recoverSubagentSnapshots,
} from '../src/daemon/subagentCoordinator.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import {
  type SubagentConversationContext,
  SubagentConversationPersistence,
} from '../src/daemon/subagentConversations.js'
import { InMemoryDaemonRuntime, type DaemonEvent, type DaemonSession } from '../src/daemon/runtime.js'
import {
  DaemonTranscriptStore,
  INTERRUPTED_TOOL_RESULT,
} from '../src/session/daemonTranscript.js'
import { AgentTurnRunner, formatSubagentResults } from '../src/daemon/turnRunner.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { messagesToAnthropic } from '../src/llms/anthropic.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { AGENT_MEMORY_WRITE_DEFINITION } from '../src/tools/agentMemoryTools.js'
import { registerClaudeAgentTools } from '../src/tools/claudeTools/agentOps.js'
import type { SpawnedAgentManagerPort, SpawnedAgentSnapshot } from '../src/operators/subagents.js'
import type { JsonObject, ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

function toolCall(name: string, arguments_: JsonObject): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name, arguments: arguments_ } }
}

function agentDefinition(name: string): AgentDefinition {
  return {
    allowedTools: null,
    description: `${name} test agent`,
    excludeTools: [],
    isolation: '',
    maxDepth: 3,
    model: '',
    name,
    source: 'test',
    systemPrompt: `You are the ${name} test agent.`,
    tools: [],
  }
}

function creatorDefinition(...children: readonly string[]): AgentDefinition {
  return {
    ...agentDefinition('default'),
    subagents: Object.freeze(Object.fromEntries(children.map(name => [
      name,
      Object.freeze({ path: `${name}.yaml`, description: `${name} child` }),
    ]))),
  }
}

function session(id = 'parent-session'): DaemonSession {
  return {
    activeTurnId: 'parent-turn',
    agentId: 'default',
    cancelRequested: false,
    cwd: process.cwd(),
    extra: {},
    id,
    interactionMode: 'code',
    lastActive: 0,
    messages: [],
    metadata: {},
    model: 'test-model',
    planMode: false,
    sessionKey: id,
    status: 'working',
    thinkingContent: [],
    toolExecutions: [],
    totalInputTokens: 0,
    totalOutputTokens: 0,
    turnCount: 0,
    workspace: `/tmp/agents/${id}`,
  }
}

async function waitFor(predicate: () => boolean, timeoutMs = 1_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error(`condition was not met within ${timeoutMs}ms`)
    await Bun.sleep(2)
  }
}

test('subagent turn coordinator bounds a hung cohort and reports its pending snapshot', async () => {
  const snapshot: SpawnedAgentSnapshot = {
    agentId: 'coder',
    closed: false,
    createdAt: '2026-07-16T00:00:00.000Z',
    id: 'hung-child',
    lastInput: 'inspect forever',
    name: 'hung-child',
    promptProfile: 'coder',
    queueSize: 0,
    sourceAgentId: 'parent-session',
    status: 'running',
    title: 'Hung child',
    updatedAt: '2026-07-16T00:00:00.000Z',
  }
  let observedTimeout: number | undefined
  const coordinator = new NativeSubagentTurnCoordinator({
    async waitFor(predicate, options): Promise<boolean> {
      observedTimeout = options?.timeoutMs
      return predicate()
    },
  }, () => [snapshot], 25, () => 1_000)
  const cohort = coordinator.begin('parent-session')
  coordinator.track([snapshot])

  expect(coordinator.trackedIds('parent-session')).toEqual(['hung-child'])
  expect(await cohort.waitForResults()).toEqual([snapshot])
  expect(observedTimeout).toBe(25)
  expect(coordinator.trackedIds('parent-session')).toEqual([])
})

test('default subagent cohort joining has no wall-clock deadline', async () => {
  let snapshot: SpawnedAgentSnapshot = {
    agentId: 'coder',
    closed: false,
    createdAt: '2026-07-16T00:00:00.000Z',
    id: 'long-child',
    name: 'long-child',
    promptProfile: 'coder',
    queueSize: 0,
    sourceAgentId: 'parent-session',
    status: 'running',
    title: 'Long child',
    updatedAt: '2026-07-16T00:00:00.000Z',
  }
  let observedTimeout: number | undefined = -1
  const coordinator = new NativeSubagentTurnCoordinator({
    async waitFor(predicate, options): Promise<boolean> {
      observedTimeout = options?.timeoutMs
      snapshot = { ...snapshot, lastOutput: 'eventually finished', status: 'completed' }
      return predicate()
    },
  }, () => [snapshot])
  const cohort = coordinator.begin('parent-session')
  coordinator.track([snapshot])

  expect(await cohort.waitForResults()).toEqual([snapshot])
  expect(observedTimeout).toBeUndefined()
})

test('the next parent turn rejoins an interrupted child and delivers its result exactly once', async () => {
  const sourceId = 'interrupted-parent'
  let snapshot: SpawnedAgentSnapshot = {
    agentId: 'researcher',
    closed: false,
    createdAt: '2026-07-16T00:00:00.000Z',
    id: 'child-finishing-between-turns',
    lastInput: 'finish the delegated review',
    name: 'between-turns-reviewer',
    promptProfile: 'researcher',
    queueSize: 0,
    sourceAgentId: sourceId,
    status: 'running',
    title: 'Between-turns review',
    updatedAt: '2026-07-16T00:00:00.000Z',
  }
  let waitCalls = 0
  const coordinator = new NativeSubagentTurnCoordinator({
    async waitFor(predicate): Promise<boolean> {
      waitCalls += 1
      return predicate()
    },
  }, () => [snapshot])

  const interruptedTurn = coordinator.begin(sourceId)
  expect(coordinator.trackedIds(sourceId)).toEqual([snapshot.id])
  interruptedTurn.close()

  snapshot = {
    ...snapshot,
    lastOutput: 'The delegated review is complete.',
    status: 'completed',
    updatedAt: '2026-07-16T00:01:00.000Z',
  }
  const resumedTurn = coordinator.begin(sourceId)
  expect(coordinator.trackedIds(sourceId)).toEqual([snapshot.id])
  expect(await resumedTurn.waitForResults()).toEqual([snapshot])
  resumedTurn.close()

  const followingTurn = coordinator.begin(sourceId)
  expect(coordinator.trackedIds(sourceId)).toEqual([])
  expect(await followingTurn.waitForResults()).toEqual([])
  expect(waitCalls).toBe(1)
  followingTurn.close()
})

test('resumed transcript recovery and cohort tracking have no 8-agent cap', async () => {
  const sourceId = 'recovered-session'
  const archived = Array.from({ length: 1_000 }, (_, index) => ({
    agent_id: 'researcher',
    closed: false,
    created_at: `2026-07-16T00:00:${String(index % 60).padStart(2, '0')}.000Z`,
    id: `archived-${index}`,
    last_input: `inspect area ${index}`,
    name: `agent-${index}`,
    prompt_profile: 'researcher',
    queue_size: 0,
    source_agent_id: sourceId,
    status: 'running',
    title: `Area ${index}`,
    updated_at: '2026-07-16T00:01:00.000Z',
  }))

  const recovered = recoverSubagentSnapshots([{
    role: 'tool',
    name: 'SpawnAgents',
    content: JSON.stringify(archived),
  }], sourceId)

  expect(recovered).toHaveLength(1_000)
  expect(recovered.find(snapshot => snapshot.id === 'archived-0')).toMatchObject({ status: 'running' })
  expect(recovered.find(snapshot => snapshot.id === 'archived-999')).toMatchObject({ status: 'running' })

  const settled = recovered.map(snapshot => ({
    ...snapshot,
    lastOutput: `finished ${snapshot.id}`,
    status: 'completed' as const,
  }))
  const coordinator = new NativeSubagentTurnCoordinator({
    async waitFor(predicate): Promise<boolean> {
      return predicate()
    },
  }, () => settled)
  const cohort = coordinator.begin(sourceId)
  coordinator.track(recovered)
  expect(coordinator.trackedIds(sourceId)).toHaveLength(1_000)
  expect(await cohort.waitForResults()).toHaveLength(1_000)
})

test('joined subagent results stay inside one hard context budget for large cohorts', () => {
  const longOutput = 'evidence '.repeat(4_000)
  const snapshots = Array.from({ length: 1_000 }, (_, index): SpawnedAgentSnapshot => ({
    agentId: 'researcher',
    closed: false,
    createdAt: '2026-07-16T00:00:00.000Z',
    id: `large-agent-${index}`,
    lastOutput: longOutput,
    name: `large-agent-${index}`,
    promptProfile: 'researcher',
    queueSize: 0,
    sourceAgentId: 'large-parent',
    status: 'completed',
    title: `Large review ${index}`,
    updatedAt: '2026-07-16T00:01:00.000Z',
  }))

  const thirtyThree = formatSubagentResults(snapshots.slice(0, 33))
  const thousand = formatSubagentResults(snapshots)

  expect(thirtyThree.join('\n').length).toBeLessThanOrEqual(64_000)
  expect(thirtyThree).toHaveLength(33)
  expect(thirtyThree.at(-1)).toContain('large-agent-32')
  expect(thousand.join('\n').length).toBeLessThanOrEqual(64_000)
  expect(thousand).toHaveLength(65)
  expect(thousand.at(-1)).toContain('omitted count=936')
  expect(thousand.at(-1)).toContain('paged TaskListTool')
})

test('compact foreground batches and awaits retain one bounded coordinator delivery without duplicating small batches', async () => {
  const sourceId = 'compact-delivery-parent'
  let generated = 0
  let snapshots: SpawnedAgentSnapshot[] = []
  const snapshotById = (id: string): SpawnedAgentSnapshot => {
    const snapshot = snapshots.find(candidate => candidate.id === id)
    if (!snapshot) throw new Error(`missing fixture snapshot ${id}`)
    return snapshot
  }
  const manager: SpawnedAgentManagerPort = {
    close: id => {
      const current = snapshotById(id)
      const closed: SpawnedAgentSnapshot = { ...current, closed: true, status: 'closed' }
      snapshots = snapshots.map(snapshot => snapshot.id === id ? closed : snapshot)
      return { ...closed, previousStatus: current.status }
    },
    listHandles: () => [...snapshots],
    resume: id => snapshotById(id),
    sendInput: async id => snapshotById(id ?? ''),
    spawn: async options => {
      const id = options?.nickname?.trim() || `compact-${generated++}`
      const created: SpawnedAgentSnapshot = {
        agentId: options?.agent?.id ?? 'researcher',
        closed: false,
        createdAt: '2026-07-16T00:00:00.000Z',
        id,
        lastInput: options?.message ?? '',
        name: id,
        promptProfile: options?.promptProfile ?? 'researcher',
        queueSize: 0,
        ...(options?.sourceAgentId === undefined ? {} : { sourceAgentId: options.sourceAgentId }),
        status: 'running',
        title: options?.title ?? id,
        updatedAt: '2026-07-16T00:00:00.000Z',
      }
      snapshots.push(created)
      return created
    },
    wait: async targets => {
      const targetSet = new Set(targets)
      snapshots = snapshots.map(snapshot => targetSet.has(snapshot.id)
        ? {
          ...snapshot,
          lastOutput: `result from ${snapshot.id}: ${'evidence '.repeat(2_000)}`,
          status: 'completed' as const,
          updatedAt: '2026-07-16T00:01:00.000Z',
        }
        : snapshot)
      return {
        completed: targets.map(snapshotById),
        pending: [],
      }
    },
  }
  const coordinator = new NativeSubagentTurnCoordinator({
    async waitFor(predicate): Promise<boolean> {
      return predicate()
    },
  }, () => manager.listHandles())
  const registry = new ToolRegistry()
  registerClaudeAgentTools(registry, { backgroundAgents: coordinator, manager })
  const context = { agentId: 'default', metadata: {}, sessionId: sourceId }

  const largeCohort = coordinator.begin(sourceId)
  const largeReceipt = JSON.parse(await registry.execute(toolCall('SpawnAgents', {
    agents: Array.from({ length: 9 }, (_, index) => ({
      name: `large-${index}`,
      prompt: `inspect area ${index}`,
      title: `Large review ${index}`,
    })),
    wait: true,
  }), context)) as Record<string, unknown>
  expect(largeReceipt).toMatchObject({ accepted_count: 9, omitted_count: 1, shown_count: 8 })
  expect(coordinator.trackedIds(sourceId)).toHaveLength(9)

  await registry.execute(toolCall('TaskListTool', {}), context)
  const compactAwait = JSON.parse(await registry.execute(toolCall('AwaitAgents', {
    timeout_seconds: 0,
    wake_on: 'all',
  }), context)) as Record<string, unknown>
  expect(compactAwait).toMatchObject({ agent_count: 9, omitted_count: 1, shown_count: 8 })
  expect(coordinator.trackedIds(sourceId)).toHaveLength(9)

  const joined = await largeCohort.waitForResults()
  const joinedPayload = formatSubagentResults(joined)
  expect(joined).toHaveLength(9)
  expect(joinedPayload.join('\n').length).toBeLessThanOrEqual(64_000)
  expect(joinedPayload.join('\n')).toContain('result from large-8')
  expect(await largeCohort.waitForResults()).toEqual([])
  expect(coordinator.trackedIds(sourceId)).toEqual([])

  const staleAwait = JSON.parse(await registry.execute(toolCall('AwaitAgents', {
    timeout_seconds: 0,
    wake_on: 'all',
  }), context)) as { agents: unknown[] }
  expect(staleAwait.agents).toEqual([])
  expect(coordinator.trackedIds(sourceId)).toEqual([])

  const smallCohort = coordinator.begin(sourceId)
  const smallReceipt = JSON.parse(await registry.execute(toolCall('SpawnAgents', {
    agents: Array.from({ length: 8 }, (_, index) => ({
      name: `small-${index}`,
      prompt: `inspect small area ${index}`,
      title: `Small review ${index}`,
    })),
    wait: true,
  }), context)) as Array<Record<string, unknown>>
  expect(smallReceipt).toHaveLength(8)
  expect(smallReceipt.every(snapshot => typeof snapshot.last_output === 'string')).toBeTrue()
  expect(coordinator.trackedIds(sourceId)).toEqual([])
  expect(await smallCohort.waitForResults()).toEqual([])
})

test('session metadata preserves every handle omitted from a bounded provider receipt', () => {
  const sourceId = 'manifest-parent'
  const snapshots = Array.from({ length: 20 }, (_, index): SpawnedAgentSnapshot => ({
    agentId: 'researcher',
    closed: false,
    createdAt: '2026-07-16T00:00:00.000Z',
    id: `manifest-${index}`,
    ...(index === 19 ? { historySessionId: '0123456789abcdef0123456789abcdef' } : {}),
    lastInput: `inspect area ${index}`,
    name: `manifest-${index}`,
    promptProfile: 'researcher',
    queueSize: 0,
    sourceAgentId: sourceId,
    status: 'running',
    title: `Manifest ${index}`,
    updatedAt: '2026-07-16T00:00:01.000Z',
  }))
  const metadata: Record<string, unknown> = {}
  replacePersistedSubagentSnapshots(metadata, snapshots)
  const boundedReceipt = [{
    role: 'tool',
    name: 'SpawnAgents',
    content: JSON.stringify({
      accepted_count: 20,
      agents: snapshots.slice(0, 8).map(snapshot => ({
        id: snapshot.id,
        name: snapshot.name,
        title: snapshot.title,
        status: snapshot.status,
      })),
      omitted_count: 12,
    }),
  }]

  const recovered = recoverSubagentSnapshots(
    boundedReceipt,
    sourceId,
    persistedSubagentSnapshotValues(metadata),
  )

  expect(recovered).toHaveLength(20)
  expect(recovered.find(snapshot => snapshot.id === 'manifest-19')).toMatchObject({
    historySessionId: '0123456789abcdef0123456789abcdef',
    id: 'manifest-19',
    lastInput: 'inspect area 19',
  })
})

test('transcript recovery replaces reset targets by id or stable name and preserves later stops', () => {
  const sourceId = 'reset-recovery-session'
  const snapshot = (id: string, name: string, status: string) => ({
    agent_id: 'researcher',
    closed: status === 'closed',
    created_at: '2026-07-16T00:00:00.000Z',
    id,
    last_input: `inspect ${name}`,
    name,
    prompt_profile: 'researcher',
    queue_size: 0,
    source_agent_id: sourceId,
    status,
    title: `${name} title`,
    updated_at: '2026-07-16T00:01:00.000Z',
  })
  const replacementByName = snapshot('replacement-by-name', 'stable-name', 'running')
  const replacementById = snapshot('replacement-by-id', 'second-stable-name', 'running')
  const messages = [
    {
      role: 'tool',
      name: 'SpawnAgents',
      content: JSON.stringify([
        snapshot('old-by-name', 'stable-name', 'running'),
        snapshot('old-by-id', 'second-stable-name', 'running'),
      ]),
    },
    {
      role: 'tool',
      name: 'ResetAgent',
      content: JSON.stringify({ reset_target: 'stable-name', new_task: replacementByName }),
    },
    {
      role: 'tool',
      name: 'ResetAgent',
      content: JSON.stringify({ reset_target: 'old-by-id', new_task: replacementById }),
    },
    {
      role: 'tool',
      name: 'TaskStopTool',
      content: JSON.stringify({ ...replacementById, closed: true, status: 'closed' }),
    },
  ]

  const recovered = recoverSubagentSnapshots(messages, sourceId)

  expect(recovered.map(item => item.id).sort()).toEqual(['replacement-by-id', 'replacement-by-name'])
  expect(recovered.find(item => item.id === 'replacement-by-name')).toMatchObject({ status: 'running' })
  expect(recovered.find(item => item.id === 'replacement-by-id')).toMatchObject({ closed: true, status: 'closed' })
})

test('resumed transcript recovery keeps the newest handle for a reused stable name', () => {
  const sourceId = 'two-wave-recovery-session'
  const archived = (
    id: string,
    name: string,
    status: 'completed' | 'running',
    createdAt: string,
    updatedAt: string,
  ) => ({
    agent_id: 'researcher',
    closed: false,
    created_at: createdAt,
    id,
    last_output: status === 'completed' ? `finished ${id}` : null,
    name,
    prompt_profile: 'researcher',
    queue_size: 0,
    source_agent_id: sourceId,
    status,
    title: `${name} title`,
    updated_at: updatedAt,
  })
  const messages = [
    {
      role: 'tool',
      name: 'SpawnAgents',
      content: JSON.stringify([
        archived(
          'old-persistence-id',
          'eyvan-persistence-hunter',
          'running',
          '2026-07-16T01:00:00.000Z',
          '2026-07-16T01:30:00.000Z',
        ),
        archived(
          'unrelated-id',
          'eyvan-runtime-hunter',
          'completed',
          '2026-07-16T01:00:00.000Z',
          '2026-07-16T01:20:00.000Z',
        ),
      ]),
    },
    {
      role: 'tool',
      name: 'SpawnAgents',
      content: JSON.stringify([
        archived(
          'new-persistence-id',
          'eyvan-persistence-hunter',
          'completed',
          '2026-07-16T02:00:00.000Z',
          '2026-07-16T02:10:00.000Z',
        ),
      ]),
    },
  ]

  const recovered = recoverSubagentSnapshots(messages, sourceId)

  expect(recovered.map(snapshot => snapshot.id)).toEqual([
    'unrelated-id',
    'new-persistence-id',
  ])
  expect(recovered.find(snapshot => snapshot.name === 'eyvan-persistence-hunter')).toMatchObject({
    id: 'new-persistence-id',
    lastOutput: 'finished new-persistence-id',
    status: 'completed',
  })
})

test('daemon subagent event bus isolates sessions and removes unsubscribed listeners', () => {
  const bus = new DaemonSubagentEventBus()
  const first: DaemonEvent[] = []
  const second: DaemonEvent[] = []
  const releaseFirst = bus.subscribe('session-a', event => first.push(event))
  bus.subscribe('session-b', event => second.push(event))
  const event: DaemonEvent = { type: 'subagent_event', payload: { agent_id: 'child-a' } }

  bus.publish('session-a', event)
  expect(first).toEqual([event])
  expect(second).toEqual([])

  releaseFirst()
  bus.publish('session-a', { type: 'subagent_event', payload: { agent_id: 'late-child' } })
  bus.publish('session-b', { type: 'subagent_event', payload: { agent_id: 'child-b' } })
  expect(first).toEqual([event])
  expect(second).toEqual([{ type: 'subagent_event', payload: { agent_id: 'child-b' } }])
})

test('daemon subagent event bus replays events emitted between parent turns exactly once', () => {
  const bus = new DaemonSubagentEventBus()
  const event: DaemonEvent = {
    type: 'subagent_event',
    payload: { agent_id: 'background-child', event: { type: 'turn_end', payload: { status: 'completed' } } },
  }
  bus.publish('session-a', event)

  const replayed: DaemonEvent[] = []
  const release = bus.subscribe('session-a', item => replayed.push(item))
  expect(replayed).toEqual([event])
  release()

  const duplicate: DaemonEvent[] = []
  bus.subscribe('session-a', item => duplicate.push(item))
  expect(duplicate).toEqual([])
})

class ToolCapturingParentClient implements LlmClient {
  requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    yield { content: 'ready' }
  }
}

test('agent turn runner exposes registered delegation tools to the provider', async () => {
  const client = new ToolCapturingParentClient()
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: BUILTIN_AGENTS,
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'test-model',
    permissionMode: 'auto',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })
  const runner = new AgentTurnRunner({
    agentDefinitions: BUILTIN_AGENTS,
    llm: client,
    model: 'test-model',
    permissionMode: 'auto',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    for await (const _event of runner.run(session(), 'delegate if useful', new AbortController().signal)) {
      // Consume the complete turn so the captured request reflects production iteration.
    }
    const names = client.requests[0]?.tools?.map(tool => tool.function.name) ?? []
    expect(names).toContain('AgentTool')
    expect(names).toContain('SpawnAgents')
    expect(names).toContain('AwaitAgents')
    expect(names).toContain('SendMessageTool')
  } finally {
    await host.manager.shutdown()
  }
})

test('a restarted daemon exposes transcript agents as honest terminal snapshots on the next turn', async () => {
  const sourceId = '7dd01499b710'
  const client = new ToolCapturingParentClient()
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: BUILTIN_AGENTS,
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, {
    backgroundAgents: host.turnCoordinator,
    manager: host.managerPort,
  })
  const runner = new AgentTurnRunner({
    agentDefinitions: BUILTIN_AGENTS,
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    subagentCoordinator: host.turnCoordinator,
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  const resumed = session(sourceId)
  resumed.messages.push({
    role: 'tool',
    name: 'SpawnAgents',
    tool_call_id: 'spawn-before-restart',
    content: JSON.stringify([
      {
        agent_id: 'researcher',
        closed: false,
        created_at: '2026-07-16T01:42:47.302Z',
        id: 'subagent-running',
        last_input: 'inspect persistence',
        name: 'persistence-hunter',
        prompt_profile: 'researcher',
        queue_size: 0,
        source_agent_id: sourceId,
        status: 'running',
        summary: 'last observed reading repository code',
        title: 'Persistence hunt',
        updated_at: '2026-07-16T01:50:00.000Z',
      },
      {
        agent_id: 'researcher',
        closed: false,
        created_at: '2026-07-16T01:42:47.303Z',
        id: 'subagent-complete',
        last_input: 'inspect API code',
        last_output: 'Found a reproducible middleware failure.',
        name: 'api-hunter',
        prompt_profile: 'researcher',
        queue_size: 0,
        source_agent_id: sourceId,
        status: 'completed',
        title: 'API hunt',
        updated_at: '2026-07-16T01:55:00.000Z',
      },
    ]),
  })
  resumed.messages.push({
    role: 'tool',
    name: 'TaskListTool',
    tool_call_id: 'compact-list-before-restart',
    content: JSON.stringify([
      {
        id: 'subagent-running',
        name: 'persistence-hunter',
        title: 'Persistence hunt',
        status: 'running',
        has_output: false,
      },
      {
        id: 'subagent-complete',
        name: 'api-hunter',
        title: 'API hunt',
        status: 'completed',
        has_output: true,
      },
    ]),
  })

  try {
    for await (const _event of runner.run(resumed, 'continue', new AbortController().signal)) {
      // Restoring happens before the provider request for this resumed turn.
    }

    expect(client.requests).toHaveLength(2)
    const joinedContext = client.requests[1]?.messages
      .filter(message => message.role === 'user')
      .map(message => String(message.content))
      .join('\n') ?? ''
    expect(joinedContext).toContain('[sub-agent events]')
    expect(joinedContext).toContain('Persistence hunt')
    expect(joinedContext).toContain('daemon process ended')
    expect(joinedContext).toContain('API hunt')
    expect(joinedContext).toContain('Found a reproducible middleware failure.')

    const systemPrompt = client.requests[0]?.messages
      .filter(message => message.role === 'system')
      .map(message => String(message.content))
      .join('\n') ?? ''
    expect(systemPrompt).toContain('2 delegated task handle(s) were recovered')
    const context = { agentId: 'default', metadata: {}, sessionId: sourceId }
    const listed = JSON.parse(await registry.execute(
      toolCall('TaskListTool', {}),
      context,
    )) as Array<Record<string, unknown>>
    expect(listed).toHaveLength(2)
    expect(listed.find(snapshot => snapshot.id === 'subagent-running')).toMatchObject({
      status: 'interrupted',
    })
    expect(String(listed.find(snapshot => snapshot.id === 'subagent-running')?.error)).toContain('daemon process ended')
    const recoveredRunning = JSON.parse(await registry.execute(
      toolCall('TaskGetTool', { task_id: 'subagent-running' }),
      context,
    )) as Record<string, unknown>
    const recoveredComplete = JSON.parse(await registry.execute(
      toolCall('TaskGetTool', { task_id: 'subagent-complete' }),
      context,
    )) as Record<string, unknown>
    expect(recoveredRunning).toMatchObject({
      summary: 'last observed reading repository code',
      status: 'interrupted',
    })
    expect(recoveredComplete).toMatchObject({
      last_output: 'Found a reproducible middleware failure.',
      status: 'completed',
    })

    for await (const _event of runner.run(resumed, 'continue again', new AbortController().signal)) {
      // Recovered results were already delivered and must not trigger another continuation.
    }
    expect(client.requests).toHaveLength(3)

    const peeked = JSON.parse(await registry.execute(
      toolCall('PeekAgent', { target: 'persistence-hunter' }),
      context,
    )) as Record<string, unknown>
    expect(peeked).toMatchObject({ id: 'subagent-running', status: 'interrupted' })

    const awaited = JSON.parse(await registry.execute(toolCall('AwaitAgents', {
      agent_ids: ['subagent-running', 'subagent-complete'],
      timeout_seconds: 30,
      wake_on: 'all',
    }), context)) as { agents: Array<Record<string, unknown>>; wake_reason: string }
    expect(awaited.wake_reason).toBe('agents_done')
    expect(awaited.agents.map(snapshot => snapshot.status)).toEqual(['interrupted', 'completed'])

    const reset = JSON.parse(await registry.execute(toolCall('ResetAgent', {
      target: 'persistence-hunter',
      new_prompt: 'restart the persistence inspection from the recovered context',
    }), context)) as { new_task: Record<string, unknown>; reset_target: string }
    expect(reset.reset_target).toBe('persistence-hunter')
    expect(reset.new_task.id).not.toBe('subagent-running')
    expect(reset.new_task).toMatchObject({
      name: 'persistence-hunter',
      source_agent_id: sourceId,
    })
    const restarted = JSON.parse(await registry.execute(toolCall('AwaitAgents', {
      agent_ids: [String(reset.new_task.id)],
      timeout_seconds: 5,
      wake_on: 'all',
    }), context)) as { agents: Array<Record<string, unknown>>; wake_reason: string }
    expect(restarted).toMatchObject({
      wake_reason: 'agents_done',
      agents: [{ last_output: 'ready', status: 'completed' }],
    })
    const afterReset = JSON.parse(await registry.execute(
      toolCall('TaskListTool', {}),
      context,
    )) as Array<Record<string, unknown>>
    expect(afterReset.some(snapshot => snapshot.id === 'subagent-running')).toBe(false)
    expect(afterReset.some(snapshot => snapshot.id === reset.new_task.id)).toBe(true)
  } finally {
    await host.manager.shutdown()
  }
})

test('agent turn runner refreshes the model-visible mode overlay on an existing session', async () => {
  const client = new ToolCapturingParentClient()
  const activeSession = session('mode-session')
  const modeTool: ToolDefinition = {
    type: 'function',
    function: { name: 'SetInteractionModeTool', description: 'switch modes', parameters: {} },
  }
  const runner = new AgentTurnRunner({
    agentDefinitions: BUILTIN_AGENTS,
    llm: client,
    model: 'test-model',
    tools: [modeTool],
  })

  for await (const _event of runner.run(activeSession, 'inspect the implementation', new AbortController().signal)) {
    // Consume the first turn.
  }
  activeSession.interactionMode = 'researcher'
  activeSession.planMode = false
  for await (const _event of runner.run(activeSession, 'continue in research mode', new AbortController().signal)) {
    // Consume the second turn after the mode change.
  }

  const firstSystem = client.requests[0]?.messages.filter(message => message.role === 'system') ?? []
  const secondSystem = client.requests[1]?.messages.filter(message => message.role === 'system') ?? []
  expect(firstSystem).toHaveLength(1)
  expect(secondSystem).toHaveLength(1)
  expect(String(firstSystem[0]?.content)).toContain('Use code mode for normal implementation.')
  expect(String(firstSystem[0]?.content)).not.toContain('You are in researcher mode.')
  expect(String(secondSystem[0]?.content)).toContain('You are in researcher mode.')
  expect(String(secondSystem[0]?.content)).not.toContain('Use code mode for normal implementation.')
})

test('native subagent host rejects unknown profiles instead of granting a generic child surface', async () => {
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([['coder', agentDefinition('coder')]]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: new ToolCapturingParentClient(),
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    await expect(host.managerPort.spawn({
      message: 'review the change',
      promptProfile: 'review',
      title: 'Review change',
    })).rejects.toThrow('is not a registered agent profile')
    expect(host.managerPort.listHandles()).toEqual([])
  } finally {
    await host.manager.shutdown()
  }
})

test('native subagent host enforces the creator profile child catalog after alias resolution', async () => {
  const registry = new ToolRegistry()
  const researcher = agentDefinition('researcher')
  const creator: AgentDefinition = {
    ...agentDefinition('default'),
    subagents: Object.freeze({
      researcher: Object.freeze({ path: 'researcher.yaml', description: 'read-only research' }),
    }),
  }
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['coder', agentDefinition('coder')],
      ['default', creator],
      ['researcher', researcher],
    ]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: new ToolCapturingParentClient(),
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    await expect(host.managerPort.spawn({
      creatorAgentId: 'default',
      message: 'implement it',
      promptProfile: 'general-purpose',
      title: 'Implement change',
    })).rejects.toThrow("is not allowed by agent 'default'")
    expect(host.managerPort.listHandles()).toEqual([])
  } finally {
    await host.manager.shutdown()
  }
})

test('native subagent host rejects unknown creators and creators without a child catalog', async () => {
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['coder', agentDefinition('coder')],
      ['default', agentDefinition('default')],
    ]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: new ToolCapturingParentClient(),
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    await expect(host.managerPort.spawn({
      creatorAgentId: 'missing-parent',
      message: 'implement it',
      promptProfile: 'coder',
      title: 'Implement change',
    })).rejects.toThrow('is not a registered agent profile')
    await expect(host.managerPort.spawn({
      creatorAgentId: 'default',
      message: 'implement it',
      promptProfile: 'coder',
      title: 'Implement change',
    })).rejects.toThrow("allowed profiles: (none)")
    expect(host.managerPort.listHandles()).toEqual([])
  } finally {
    await host.manager.shutdown()
  }
})

class ConcurrentChildClient implements LlmClient {
  active = 0
  calls = 0
  maxActive = 0
  private readonly bothStarted = Promise.withResolvers<void>()

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.calls += 1
    this.active += 1
    this.maxActive = Math.max(this.maxActive, this.active)
    if (this.calls === 2) this.bothStarted.resolve()
    try {
      await Promise.race([
        this.bothStarted.promise,
        Bun.sleep(500).then(() => {
          throw new Error('second subagent did not start concurrently')
        }),
      ])
      const prompt = request.messages.findLast(message => message.role === 'user')?.content
      const text = typeof prompt === 'string' ? prompt : '(missing prompt)'
      yield { thinking: `checking ${text}` }
      await Bun.sleep(2)
      yield {
        content: `finished:${text}`,
        usage: { inputTokens: 11, outputTokens: 7, reasoningTokens: 3 },
      }
    } finally {
      this.active -= 1
    }
  }
}

class BoundedSwarmChildClient implements LlmClient {
  active = 0
  calls = 0
  maxActive = 0
  readonly firstWaveStarted = Promise.withResolvers<void>()
  readonly release = Promise.withResolvers<void>()

  async *stream(): AsyncGenerator<LlmDelta> {
    this.calls += 1
    this.active += 1
    this.maxActive = Math.max(this.maxActive, this.active)
    if (this.calls === 8) this.firstWaveStarted.resolve()
    try {
      await this.release.promise
      yield { content: 'bounded worker complete' }
    } finally {
      this.active -= 1
    }
  }
}

class ReloadGenerationChildClient implements LlmClient {
  readonly models: string[] = []
  readonly prompts: string[] = []
  readonly started = Promise.withResolvers<void>()
  private readonly released = Promise.withResolvers<void>()

  constructor(
    private readonly label: string,
    blocked = false,
  ) {
    if (!blocked) this.released.resolve()
  }

  release(): void {
    this.released.resolve()
  }

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    const prompt = request.messages.findLast(message => message.role === 'user')?.content
    const text = typeof prompt === 'string' ? prompt : '(missing prompt)'
    this.models.push(request.model)
    this.prompts.push(text)
    this.started.resolve()
    await this.released.promise
    yield { content: `${this.label}:${request.model}:${text}` }
  }
}

test('native child spawning honors the creator-local resolved catalog profile', async () => {
  const client = new ReloadGenerationChildClient('catalog-provider')
  const registry = new ToolRegistry()
  const globalCoder = { ...agentDefinition('coder'), model: 'global-model' }
  const boundCoder = { ...agentDefinition('coder'), model: 'bound-model' }
  const creator: AgentDefinition = {
    ...agentDefinition('default'),
    subagents: {
      coder: {
        description: 'creator-local coder',
        path: '/profiles/readonly-coder.yaml',
        resolvedProfile: '@catalog:coder:/profiles/readonly-coder.yaml',
      },
    },
  }
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['@catalog:coder:/profiles/readonly-coder.yaml', boundCoder],
      ['coder', globalCoder],
      ['default', creator],
    ]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'connection-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })

  try {
    await registry.execute(toolCall('AgentTool', {
      prompt: 'use the creator-local profile',
      subagent_type: 'coder',
      title: 'Bound catalog',
    }), { agentId: 'default', metadata: {}, sessionId: 'catalog-parent' })
    expect(client.models).toEqual(['bound-model'])
  } finally {
    await host.manager.shutdown()
  }
})

test('native child models follow explicit, profile, parent-session, then connection precedence', async () => {
  const client = new ReloadGenerationChildClient('model-provider')
  const registry = new ToolRegistry()
  const profiled: AgentDefinition = {
    ...agentDefinition('profiled'),
    model: 'profile-model',
  }
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['default', creatorDefinition('plain', 'profiled')],
      ['plain', agentDefinition('plain')],
      ['profiled', profiled],
    ]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'connection-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })
  const parentContext = {
    agentId: 'default',
    metadata: { model: 'parent-session-model', permission_mode: 'plan' },
    sessionId: 'parent-session',
  }

  try {
    await registry.execute(toolCall('AgentTool', {
      model: 'explicit-model',
      prompt: 'use the explicit model',
      subagent_type: 'profiled',
      title: 'Explicit model',
    }), parentContext)
    await registry.execute(toolCall('AgentTool', {
      prompt: 'use the profile model',
      subagent_type: 'profiled',
      title: 'Profile model',
    }), parentContext)
    await registry.execute(toolCall('AgentTool', {
      prompt: 'inherit the parent model',
      subagent_type: 'plain',
      title: 'Parent model',
    }), parentContext)
    await registry.execute(toolCall('AgentTool', {
      prompt: 'use the connection fallback',
      subagent_type: 'plain',
      title: 'Connection model',
    }), { agentId: 'default', metadata: {}, sessionId: 'fallback-session' })

    expect(client.models).toEqual([
      'explicit-model',
      'profile-model',
      'parent-session-model',
      'connection-model',
    ])
    expect(host.managerPort.listHandles().map(snapshot => snapshot.rules?.[0])).toEqual([
      'permission:plan',
      'permission:plan',
      'permission:plan',
      'permission:accept-all',
    ])
  } finally {
    await host.manager.shutdown()
  }
})

test('native subagent handles survive runtime reloads while new spawns use the latest generation', async () => {
  const oldClient = new ReloadGenerationChildClient('old-provider', true)
  const newClient = new ReloadGenerationChildClient('new-provider')
  const eventBus = new DaemonSubagentEventBus()
  const definitions = new Map([['coder', agentDefinition('coder')]])
  const oldRegistry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: process.cwd(),
    eventBus,
    llm: oldClient,
    model: 'old-model',
    permissionMode: 'manual',
    toolExecutor: oldRegistry,
    tools: oldRegistry.definitions(),
  })
  const stableManagerPort = host.managerPort

  try {
    const oldTask = await host.managerPort.spawn({
      message: 'finish the old generation',
      nickname: 'old-worker',
      promptProfile: 'coder',
      title: 'Old worker',
    })
    await oldClient.started.promise

    const newRegistry = new ToolRegistry()
    host.reconfigure({
      agentDefinitions: definitions,
      cwd: process.cwd(),
      eventBus,
      llm: newClient,
      model: 'new-model',
      permissionMode: 'accept-all',
      toolExecutor: newRegistry,
      tools: newRegistry.definitions(),
    })

    expect(host.managerPort).toBe(stableManagerPort)
    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({
      id: oldTask.id,
      model: 'old-model',
      rules: ['permission:manual', 'delegation:blocked'],
      status: 'running',
      title: 'Old worker',
    }))

    const newTask = await host.managerPort.spawn({
      message: 'run on the new generation',
      nickname: 'new-worker',
      promptProfile: 'coder',
      title: 'New worker',
    })
    const newResult = await host.managerPort.wait([newTask.id], 1_000)
    expect(newResult.completed[0]).toMatchObject({
      lastOutput: 'new-provider:new-model:run on the new generation',
      model: 'new-model',
      rules: ['permission:accept-all', 'delegation:blocked'],
      status: 'completed',
      title: 'New worker',
    })

    oldClient.release()
    const oldResult = await host.managerPort.wait([oldTask.id], 1_000)
    expect(oldResult.completed[0]).toMatchObject({
      lastOutput: 'old-provider:old-model:finish the old generation',
      model: 'old-model',
      rules: ['permission:manual', 'delegation:blocked'],
      status: 'completed',
      title: 'Old worker',
    })

    host.managerPort.resume(oldTask.id)
    const continuedTask = await host.managerPort.sendInput(oldTask.id, {
      message: 'follow up after reload',
    })
    const continuedResult = await host.managerPort.wait([continuedTask.id], 1_000)
    expect(continuedResult.completed[0]?.lastOutput).toBe('old-provider:old-model:follow up after reload')
    expect(oldClient.models).toEqual(['old-model', 'old-model'])
    expect(newClient.models).toEqual(['new-model'])
  } finally {
    await host.manager.shutdown()
  }
})

test('tightening YOLO permissions cancels active children and prevents old-policy resume', async () => {
  const client = new ReloadGenerationChildClient('yolo-provider', true)
  const eventBus = new DaemonSubagentEventBus()
  const definitions = new Map([['coder', agentDefinition('coder')]])
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'yolo-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      message: 'keep working under yolo',
      nickname: 'yolo-worker',
      promptProfile: 'coder',
      title: 'YOLO worker',
    })
    await client.started.promise

    host.reconfigure({
      agentDefinitions: definitions,
      cwd: process.cwd(),
      eventBus,
      llm: client,
      model: 'safe-model',
      permissionMode: 'auto',
      toolExecutor: registry,
      tools: registry.definitions(),
    })

    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({
      closed: true,
      id: task.id,
      status: 'cancelled',
    }))
    expect(() => host.managerPort.resume(task.id)).toThrow(
      'was invalidated when permissions were tightened',
    )
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('stale parent metadata cannot spawn a child above the current permission ceiling', async () => {
  const client = new ReloadGenerationChildClient('ceiling-provider')
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([['coder', agentDefinition('coder')]]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'safe-model',
    permissionMode: 'auto',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      message: 'continue from a stale yolo parent turn',
      permissionMode: 'accept-all',
      promptProfile: 'coder',
      title: 'Ceiling worker',
    })
    expect(task.rules).toContain('permission:auto')
    expect(task.rules).not.toContain('permission:accept-all')
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('agent catalog changes invalidate old handles before they can resume', async () => {
  const client = new ReloadGenerationChildClient('catalog-version-provider')
  const registry = new ToolRegistry()
  const eventBus = new DaemonSubagentEventBus()
  const coder = agentDefinition('coder')
  const initialDefinitions = new Map([
    ['coder', coder],
    ['default', creatorDefinition('coder')],
  ])
  const host = createNativeSubagentHost({
    agentDefinitions: initialDefinitions,
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      creatorAgentId: 'default',
      message: 'finish before catalog reload',
      promptProfile: 'coder',
      title: 'Catalog worker',
    })
    await host.managerPort.wait([task.id], 1_000)
    host.reconfigure({
      agentDefinitions: new Map([
        ['coder', coder],
        ['default', agentDefinition('default')],
      ]),
      cwd: process.cwd(),
      eventBus,
      llm: client,
      model: 'test-model',
      permissionMode: 'accept-all',
      toolExecutor: registry,
      tools: registry.definitions(),
    })
    expect(() => host.managerPort.resume(task.id)).toThrow('invalidated')
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('removing a parent session cancels and invalidates its background children', async () => {
  const client = new ReloadGenerationChildClient('background-provider', true)
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([['coder', agentDefinition('coder')]]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      message: 'keep working in the background',
      promptProfile: 'coder',
      sourceAgentId: 'session-a',
      title: 'Background worker',
    })
    await client.started.promise

    expect(host.cancelSource('session-b')).toBe(0)
    expect(host.cancelSource('session-a')).toBe(1)
    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({
      closed: true,
      id: task.id,
      status: 'cancelled',
    }))
    expect(() => host.managerPort.resume(task.id)).toThrow('invalidated')
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('tightening auto permissions to plan also cancels broader-policy children', async () => {
  const client = new ReloadGenerationChildClient('auto-provider', true)
  const eventBus = new DaemonSubagentEventBus()
  const definitions = new Map([['coder', agentDefinition('coder')]])
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'auto-model',
    permissionMode: 'auto',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      message: 'keep working under auto policy',
      promptProfile: 'coder',
      title: 'Auto worker',
    })
    await client.started.promise

    host.reconfigure({
      agentDefinitions: definitions,
      cwd: process.cwd(),
      eventBus,
      llm: client,
      model: 'plan-model',
      permissionMode: 'plan',
      toolExecutor: registry,
      tools: registry.definitions(),
    })

    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({
      closed: true,
      id: task.id,
      status: 'cancelled',
    }))
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('switching delegated permissions from plan to manual cancels children that retained automatic read access', async () => {
  const client = new ReloadGenerationChildClient('plan-provider', true)
  const eventBus = new DaemonSubagentEventBus()
  const definitions = new Map([['coder', agentDefinition('coder')]])
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'plan-model',
    permissionMode: 'plan',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      message: 'keep reading under plan policy',
      promptProfile: 'coder',
      title: 'Plan worker',
    })
    await client.started.promise

    host.reconfigure({
      agentDefinitions: definitions,
      cwd: process.cwd(),
      eventBus,
      llm: client,
      model: 'manual-model',
      permissionMode: 'manual',
      toolExecutor: registry,
      tools: registry.definitions(),
    })

    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({
      closed: true,
      id: task.id,
      status: 'cancelled',
    }))
    expect(() => host.managerPort.resume(task.id)).toThrow('invalidated')
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('invalidating a native host cancels every child and prevents later resume', async () => {
  const client = new ReloadGenerationChildClient('retired-provider', true)
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([['coder', agentDefinition('coder')]]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'retired-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      message: 'continue until the runtime disappears',
      promptProfile: 'coder',
      title: 'Retired worker',
    })
    await client.started.promise

    expect(host.invalidateAll()).toBe(1)
    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({
      closed: true,
      id: task.id,
      status: 'cancelled',
    }))
    expect(() => host.managerPort.resume(task.id)).toThrow('invalidated')
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('native SpawnAgents runs two real child turns concurrently and routes their lifecycle to the parent session', async () => {
  const client = new ConcurrentChildClient()
  const eventBus = new DaemonSubagentEventBus()
  const registry = new ToolRegistry()
  const events: DaemonEvent[] = []
  const release = eventBus.subscribe('session-a', event => events.push(event))
  const definitions = new Map([
    ['coder', agentDefinition('coder')],
    ['default', creatorDefinition('coder', 'researcher')],
    ['researcher', agentDefinition('researcher')],
  ])
  const host = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })

  try {
    const response = JSON.parse(await registry.execute(toolCall('SpawnAgents', {
      agents: [
        { name: 'implementation', prompt: 'inspect runtime', subagent_type: 'coder', title: 'Inspect runtime' },
        { name: 'research', prompt: 'inspect tests', subagent_type: 'researcher', title: 'Inspect tests' },
      ],
      timeout: 2,
      wait: true,
    }), { agentId: 'default', metadata: {}, sessionId: 'session-a' })) as Array<Record<string, unknown>>

    expect(client.calls).toBe(2)
    expect(client.maxActive).toBe(2)
    expect(response).toHaveLength(2)
    expect(response.map(entry => entry.status)).toEqual(['completed', 'completed'])
    expect(response.map(entry => entry.source_agent_id)).toEqual(['session-a', 'session-a'])
    expect(response.map(entry => entry.creator_id)).toEqual(['default', 'default'])
    expect(response.map(entry => entry.parent_id)).toEqual(['default', 'default'])
    expect(response.map(entry => entry.title)).toEqual(['Inspect runtime', 'Inspect tests'])
    expect(response.map(entry => entry.last_output)).toEqual([
      'finished:inspect runtime',
      'finished:inspect tests',
    ])

    await waitFor(() => events.filter(event => nestedEventType(event) === 'turn_end').length === 2)
    const starts = events.filter(event => nestedEventType(event) === 'turn_begin')
    const completions = events.filter(event => nestedEventType(event) === 'turn_end')
    expect(starts).toHaveLength(2)
    expect(completions).toHaveLength(2)
    expect(new Set(starts.map(event => event.payload.agent_id)).size).toBe(2)
    expect(starts.map(event => event.payload.agent_name).sort()).toEqual(['implementation', 'research'])
    expect(starts.map(event => event.payload.subagent_type).sort()).toEqual(['coder', 'researcher'])
    expect(starts.map(event => event.payload.title).sort()).toEqual(['Inspect runtime', 'Inspect tests'])
    expect(starts.every(event => event.payload.creator_id === 'default' && event.payload.parent_id === 'default')).toBeTrue()
    expect(starts.every(event => event.payload.model === 'test-model')).toBeTrue()
    expect(starts.every(event => Array.isArray(event.payload.rules))).toBeTrue()
    expect(starts.every(event => Array.isArray(event.payload.toolsets))).toBeTrue()
    expect(events.filter(event => nestedEventType(event) === 'think_part')).toHaveLength(2)
    expect(completions.map(event => event.payload.api_calls)).toEqual([1, 1])
    expect(completions.map(event => event.payload.input_tokens)).toEqual([11, 11])
    expect(completions.map(event => event.payload.output_tokens)).toEqual([7, 7])
    expect(completions.map(event => event.payload.reasoning_tokens)).toEqual([3, 3])
    expect(completions.map(event => event.payload.tool_count)).toEqual([0, 0])
    expect(completions.map(event => nestedPayload(event).summary).sort()).toEqual([
      'finished:inspect runtime',
      'finished:inspect tests',
    ])
  } finally {
    release()
    await host.manager.shutdown()
  }
})

test('native swarm execution queues work above eight without exceeding eight live model turns', async () => {
  const client = new BoundedSwarmChildClient()
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['coder', agentDefinition('coder')],
      ['default', creatorDefinition('coder')],
    ]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })

  try {
    const receipt = JSON.parse(await registry.execute(toolCall('SpawnAgents', {
      agents: Array.from({ length: 24 }, (_, index) => ({
        name: `bounded-${index}`,
        prompt: `bounded task ${index}`,
        subagent_type: 'coder',
        title: `Bounded task ${index}`,
      })),
      wait: false,
    }), { agentId: 'default', metadata: {}, sessionId: 'bounded-session' })) as Record<string, unknown>

    await client.firstWaveStarted.promise
    await Bun.sleep(10)
    expect(client.calls).toBe(8)
    expect(client.maxActive).toBe(8)
    expect(receipt).toMatchObject({ accepted_count: 24, omitted_count: 16, shown_count: 8 })

    client.release.resolve()
    const ids = host.managerPort.listHandles().map(snapshot => snapshot.id)
    const settled = await host.managerPort.wait(ids, 5_000)
    expect(settled.pending).toHaveLength(0)
    expect(settled.completed).toHaveLength(24)
    expect(client.calls).toBe(24)
    expect(client.maxActive).toBe(8)
  } finally {
    client.release.resolve()
    await host.manager.shutdown()
  }
})

class DistinctReadChildClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    const completedReads = request.messages.filter(message => message.role === 'tool').length
    if (completedReads < 30) {
      const next = completedReads + 1
      yield { toolCalls: [toolCall('ReadFile', { file_path: `src/file-${next}.ts` })] }
      return
    }
    yield { content: 'completed thirty distinct delegated reads' }
  }
}

test('native subagents execute more than twenty-five distinct tool calls', async () => {
  const registry = new ToolRegistry()
  const executedPaths: string[] = []
  const readFile: ToolDefinition = {
    type: 'function',
    function: {
      description: 'Read one test file.',
      name: 'ReadFile',
      parameters: {
        type: 'object',
        additionalProperties: false,
        required: ['file_path'],
        properties: { file_path: { type: 'string' } },
      },
    },
  }
  registry.register(readFile, inputs => {
    executedPaths.push(String(inputs.file_path))
    return 'fixture body'
  })
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['coder', agentDefinition('coder')],
      ['default', creatorDefinition('coder')],
    ]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: new DistinctReadChildClient(),
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })

  try {
    const response = JSON.parse(await registry.execute(toolCall('SpawnAgents', {
      agents: [{
        name: 'long-reader',
        prompt: 'read thirty distinct files',
        subagent_type: 'coder',
        title: 'Read fixtures',
      }],
      timeout: 5,
      wait: true,
    }), { agentId: 'default', metadata: {}, sessionId: 'long-read-session' })) as Array<Record<string, unknown>>

    expect(executedPaths).toEqual(Array.from({ length: 30 }, (_, index) => `src/file-${index + 1}.ts`))
    expect(response).toHaveLength(1)
    expect(response[0]).toMatchObject({
      last_output: 'completed thirty distinct delegated reads',
      status: 'completed',
      tool_count: 30,
    })
  } finally {
    await host.manager.shutdown()
  }
})

class ProjectMemoryChildClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    const results = request.messages.filter(message => message.role === 'tool')
    if (results.length === 0) {
      yield {
        toolCalls: [toolCall('agent_memory_write', {
          body: 'project findings',
          path: 'deepscan/findings/researcher.md',
          scope: 'project',
        })],
      }
      return
    }
    if (results.length === 1) {
      yield {
        toolCalls: [toolCall('agent_memory_write', {
          body: 'must stay denied',
          path: 'global.md',
          scope: 'global',
        })],
      }
      return
    }
    yield { content: results.map(message => String(message.content)).join('\n') }
  }
}

test('auto-mode native subagents can persist project memory without gaining global-memory writes', async () => {
  const registry = new ToolRegistry()
  const writes: JsonObject[] = []
  registry.register(AGENT_MEMORY_WRITE_DEFINITION, inputs => {
    writes.push(inputs)
    return { ok: true, path: inputs.path, scope: inputs.scope }
  })
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([['researcher', agentDefinition('researcher')]]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: new ProjectMemoryChildClient(),
    model: 'test-model',
    permissionMode: 'auto',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })

  try {
    const response = JSON.parse(await registry.execute(toolCall('SpawnAgents', {
      agents: [{
        name: 'memory-writer',
        prompt: 'persist findings',
        subagent_type: 'researcher',
        title: 'Persist findings',
      }],
      timeout: 2,
      wait: true,
    }), { metadata: {}, sessionId: 'session-memory' })) as Array<Record<string, unknown>>

    expect(writes).toEqual([{
      body: 'project findings',
      path: 'deepscan/findings/researcher.md',
      scope: 'project',
    }])
    expect(response[0]?.status).toBe('completed')
    expect(response[0]?.last_output).toContain('"ok":true')
    expect(response[0]?.last_output).toContain('Permission denied for agent_memory_write.')
  } finally {
    await host.manager.shutdown()
  }
})

class MetadataChildClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    const results = request.messages.filter(message => message.role === 'tool')
    if (results.length === 0) {
      yield {
        toolCalls: [toolCall('ReadFile', { file_path: 'src/input.ts' })],
        usage: { inputTokens: 5, outputTokens: 2, reasoningTokens: 1 },
      }
      return
    }
    if (results.length === 1) {
      yield {
        toolCalls: [toolCall('WriteFile', { content: 'done', file_path: 'src/output.ts' })],
        usage: { inputTokens: 7, outputTokens: 2, reasoningTokens: 1 },
      }
      return
    }
    yield {
      content: 'metadata task complete',
      usage: { inputTokens: 11, outputTokens: 5, reasoningTokens: 2 },
    }
  }
}

test('native completion events expose exact title, hierarchy, policy, file, tool, and usage metadata', async () => {
  const eventBus = new DaemonSubagentEventBus()
  const registry = new ToolRegistry()
  const toolDefinition = (name: string): ToolDefinition => ({
    type: 'function',
    function: {
      description: `${name} test tool`,
      name,
      parameters: { type: 'object', additionalProperties: true, properties: {} },
    },
  })
  registry.register(toolDefinition('ReadFile'), () => 'input body')
  registry.register(toolDefinition('WriteFile'), () => 'written')
  const events: DaemonEvent[] = []
  const release = eventBus.subscribe('metadata-session', event => events.push(event))
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['coder', agentDefinition('coder')],
      ['default', creatorDefinition('coder')],
    ]),
    cwd: process.cwd(),
    eventBus,
    llm: new MetadataChildClient(),
    model: 'metadata-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })

  try {
    await registry.execute(toolCall('AgentTool', {
      name: 'metadata-worker',
      prompt: 'inspect and update metadata fixtures',
      subagent_type: 'coder',
      timeout: 2,
      title: 'Inspect metadata fixtures',
      wait: true,
    }), { agentId: 'default', metadata: {}, sessionId: 'metadata-session' })

    await waitFor(() => events.some(event => nestedEventType(event) === 'turn_end'))
    const completion = events.find(event => nestedEventType(event) === 'turn_end')
    expect(completion?.payload).toMatchObject({
      agent_name: 'metadata-worker',
      api_calls: 3,
      creator_id: 'default',
      files_read: ['src/input.ts'],
      files_written: ['src/output.ts'],
      input_tokens: 23,
      model: 'metadata-model',
      output_tokens: 9,
      parent_id: 'default',
      reasoning_tokens: 4,
      rules: ['permission:accept-all', 'delegation:blocked'],
      summary: 'metadata task complete',
      title: 'Inspect metadata fixtures',
      tool_count: 2,
      toolsets: ['ReadFile', 'WriteFile'],
    })
    expect(nestedPayload(completion!).summary).toBe('metadata task complete')
  } finally {
    release()
    await host.manager.shutdown()
  }
})

class ToolCallingParentClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    if (request.messages.some(message => message.role === 'tool')) {
      yield { content: 'parent finished' }
      return
    }
    yield {
      toolCalls: [{
        id: 'delegate-1',
        type: 'function',
        function: { name: 'DelegateForTest', arguments: {} },
      }],
    }
  }
}

test('agent turn runner multiplexes a child event between parent tool start and result with turn correlation', async () => {
  const eventBus = new DaemonSubagentEventBus()
  const registry = new ToolRegistry()
  let delegatedPermissionMode: unknown
  const definition: ToolDefinition = {
    type: 'function',
    function: {
      description: 'emit one delegated lifecycle event',
      name: 'DelegateForTest',
      parameters: { type: 'object', additionalProperties: false, properties: {} },
    },
  }
  registry.register(definition, async (_inputs, context) => {
    delegatedPermissionMode = context.metadata.permission_mode
    eventBus.publish(context.sessionId ?? '', {
      type: 'subagent_event',
      payload: {
        agent_id: 'child-1',
        event: { type: 'turn_begin', payload: { status: 'running' } },
        goal: 'verify multiplexing',
      },
    })
    await Bun.sleep(5)
    return 'delegation complete'
  })
  const runner = new AgentTurnRunner({
    llm: new ToolCallingParentClient(),
    model: 'test-model',
    permissionMode: 'accept-all',
    subagentEvents: eventBus,
    toolExecutor: registry,
    tools: [definition],
  })
  const events: DaemonEvent[] = []
  for await (const event of runner.run(session(), 'delegate this', new AbortController().signal)) {
    events.push(event)
  }

  const toolCallIndex = events.findIndex(event => event.type === 'tool_call')
  const childIndex = events.findIndex(event => event.type === 'subagent_event')
  const toolResultIndex = events.findIndex(event => event.type === 'tool_result')
  expect(toolCallIndex).toBeGreaterThanOrEqual(0)
  expect(childIndex).toBeGreaterThan(toolCallIndex)
  expect(toolResultIndex).toBeGreaterThan(childIndex)
  expect(events[childIndex]).toEqual({
    type: 'subagent_event',
    payload: {
      agent_id: 'child-1',
      event: { type: 'turn_begin', payload: { status: 'running' } },
      goal: 'verify multiplexing',
      session_id: 'parent-session',
      turn_id: 'parent-turn',
    },
  })
  expect(events).toContainEqual({ type: 'text_part', payload: { text: 'parent finished' } })
  expect(delegatedPermissionMode).toBe('accept-all')
})

class PersistedHistoryChildClient implements LlmClient {
  readonly firstStarted = Promise.withResolvers<void>()
  readonly releaseFirst = Promise.withResolvers<void>()
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    if (this.requests.length === 1) {
      this.firstStarted.resolve()
      await this.releaseFirst.promise
    }
    const latestUser = request.messages.findLast(message => message.role === 'user')?.content
    yield {
      content: `answer:${typeof latestUser === 'string' ? latestUser : '(missing)'}`,
      usage: { inputTokens: 17, outputTokens: 5 },
    }
  }
}

test('persisted subagent state restores signed provider reasoning blocks', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-signed-child-history-'))
  const historySessionId = '0123456789abcdef0123456789abcdef'
  const transcripts = new DaemonTranscriptStore({
    currentProjectDirectory: process.cwd(),
    directory: join(directory, 'sessions'),
  })
  try {
    await transcripts.save({
      agentId: 'coder',
      cwd: process.cwd(),
      extra: {},
      format: 'bun-v2',
      interactionMode: 'code',
      key: historySessionId,
      messages: [{
        role: 'assistant',
        content: 'prior child result',
        thinking: 'provider reasoning',
        thinking_signature: 'signed-reasoning-block',
      }],
      metadata: { session_kind: 'subagent' },
      pendingResumeReplays: [],
      planMode: false,
      schemaVersion: undefined,
      sessionId: historySessionId,
      thinkingContent: ['provider reasoning'],
      toolExecutions: [],
      totalApiCalls: 1,
      totalInputTokens: 8,
      totalOutputTokens: 3,
      turnCount: 1,
      updatedAt: new Date().toISOString(),
      usageComplete: true,
      workspace: '',
    })
    const persistence = new SubagentConversationPersistence(transcripts)
    const state = await persistence.stateFor({
      agentId: 'coder',
      cwd: process.cwd(),
      handleId: 'subagent_signed',
      historySessionId,
      model: 'test-model',
      permissionCeiling: 'plan',
      permissionMode: 'plan',
      profile: 'coder',
      projectRoot: process.cwd(),
      rules: [],
      title: 'Signed child',
      toolsAllowed: [],
      toolsExcluded: [],
      toolsWhitelist: [],
      toolsets: [],
    })

    expect(state.messages[0]).toMatchObject({
      thinking: 'provider reasoning',
      thinking_signature: 'signed-reasoning-block',
    })
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

test('subagent stream checkpoints persist partial output without mutating live turn state', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-partial-child-history-'))
  const historySessionId = '1123456789abcdef0123456789abcdef'
  const transcripts = new DaemonTranscriptStore({
    currentProjectDirectory: process.cwd(),
    directory: join(directory, 'sessions'),
  })
  const context: SubagentConversationContext = {
    agentId: 'coder',
    cwd: process.cwd(),
    handleId: 'subagent_partial',
    historySessionId,
    model: 'test-model',
    permissionCeiling: 'plan',
    permissionMode: 'plan',
    profile: 'coder',
    projectRoot: process.cwd(),
    rules: [],
    title: 'Partial child',
    toolsAllowed: [],
    toolsExcluded: [],
    toolsWhitelist: [],
    toolsets: [],
  }
  const persistence = new SubagentConversationPersistence(transcripts)
  try {
    const state = await persistence.stateFor(context)
    state.messages.push({ role: 'user', content: 'start a long answer' })
    state.turnCount = 1

    await persistence.save(context, state, 'running', undefined, {
      content: 'partial answer',
      thinking: 'partial reasoning',
    })

    expect(state.messages).toEqual([{ role: 'user', content: 'start a long answer' }])
    expect(state.thinkingContent).toEqual([])
    const checkpoint = await transcripts.load(historySessionId, {
      currentProjectDirectory: process.cwd(),
    })
    expect(checkpoint?.messages).toEqual([
      { role: 'user', content: 'start a long answer' },
      {
        role: 'assistant',
        content: 'partial answer',
        thinking: 'partial reasoning',
        checkpoint_partial: true,
      },
    ])
    expect(checkpoint?.thinkingContent).toEqual(['partial reasoning'])

    await persistence.save(context, state, 'running', undefined, {
      content: '',
      thinking: 'thinking when the process stopped',
    })
    const resumedState = await new SubagentConversationPersistence(transcripts).stateFor(context)
    expect(resumedState.messages.at(-1)).toEqual({
      role: 'assistant',
      content: '[interrupted while reasoning]',
      thinking: 'thinking when the process stopped',
    })
    expect(messagesToAnthropic(resumedState.messages).messages.at(-1)).toEqual({
      role: 'assistant',
      content: [{ type: 'text', text: '[interrupted while reasoning]' }],
    })

    await persistence.save(context, state, 'completed')
    const completed = await transcripts.load(historySessionId, {
      currentProjectDirectory: process.cwd(),
    })
    expect(completed?.messages).toEqual([{ role: 'user', content: 'start a long answer' }])
    expect(completed?.thinkingContent).toEqual([])
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

class ToolCheckpointChildClient implements LlmClient {
  private calls = 0

  async *stream(): AsyncGenerator<LlmDelta> {
    this.calls += 1
    if (this.calls === 1) {
      yield {
        toolCalls: [{
          id: 'checkpoint-call',
          type: 'function',
          function: { name: 'CheckpointTool', arguments: { path: 'README.md' } },
        }],
        usage: { inputTokens: 9, outputTokens: 2 },
      }
      return
    }
    yield {
      content: 'completed after the checkpointed tool',
      usage: { inputTokens: 11, outputTokens: 4 },
    }
  }
}

test('native subagents checkpoint committed tool calls before long-running tools finish', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-tool-checkpoint-history-'))
  const transcripts = new DaemonTranscriptStore({
    currentProjectDirectory: process.cwd(),
    directory: join(directory, 'sessions'),
  })
  const toolStarted = Promise.withResolvers<void>()
  const releaseTool = Promise.withResolvers<void>()
  const registry = new ToolRegistry()
  const definition: ToolDefinition = {
    type: 'function',
    function: {
      name: 'CheckpointTool',
      description: 'Hold a child tool open while its history is inspected.',
      parameters: {
        type: 'object',
        additionalProperties: false,
        properties: { path: { type: 'string' } },
        required: ['path'],
      },
    },
  }
  registry.register(definition, async () => {
    toolStarted.resolve()
    await releaseTool.promise
    return 'checkpoint tool finished'
  })
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['coder', agentDefinition('coder')],
      ['default', creatorDefinition('coder')],
    ]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: new ToolCheckpointChildClient(),
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
    transcriptStore: transcripts,
  })

  try {
    const spawned = await host.managerPort.spawn({
      creatorAgentId: 'default',
      message: 'run the checkpoint tool',
      promptProfile: 'coder',
      sourceAgentId: 'feedface',
      title: 'Tool checkpoint child',
    })
    await toolStarted.promise
    const historySessionId = spawned.historySessionId
    if (!historySessionId) throw new Error('expected a persisted child history id')
    const running = await transcripts.load(historySessionId, {
      currentProjectDirectory: process.cwd(),
    })
    expect(running?.metadata.status).toBe('running')
    expect(running?.messages).toContainEqual({
      role: 'assistant',
      content: '',
      tool_calls: [{
        id: 'checkpoint-call',
        type: 'function',
        function: { name: 'CheckpointTool', arguments: { path: 'README.md' } },
      }],
    })
    expect(running?.messages).toContainEqual({
      role: 'tool',
      tool_call_id: 'checkpoint-call',
      content: INTERRUPTED_TOOL_RESULT,
    })
    expect(running?.pendingResumeReplays).toEqual([{
      arguments: JSON.stringify({ path: 'README.md' }),
      name: 'CheckpointTool',
      tool_call_id: 'checkpoint-call',
    }])

    releaseTool.resolve()
    const settled = await host.managerPort.wait([spawned.id], 5_000)
    expect(settled.completed[0]?.status).toBe('completed')
  } finally {
    releaseTool.resolve()
    await host.manager.shutdown()
    await rm(directory, { force: true, recursive: true })
  }
})

test('native subagent conversations persist as resumable histories and retain queued follow-up context', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-subagent-history-'))
  const transcripts = new DaemonTranscriptStore({
    currentProjectDirectory: process.cwd(),
    directory: join(directory, 'sessions'),
  })
  const client = new PersistedHistoryChildClient()
  const eventBus = new DaemonSubagentEventBus()
  const childEvents: DaemonEvent[] = []
  const releaseEvents = eventBus.subscribe('feedbeef', event => childEvents.push(event))
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['coder', agentDefinition('coder')],
      ['default', creatorDefinition('coder')],
    ]),
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
    transcriptStore: transcripts,
  })

  try {
    const spawned = await host.managerPort.spawn({
      creatorAgentId: 'default',
      message: 'inspect the first area',
      promptProfile: 'coder',
      sourceAgentId: 'feedbeef',
      title: 'Durable child review',
    })
    expect(spawned.historySessionId).toMatch(/^[0-9a-f]{32}$/)
    await client.firstStarted.promise

    const queued = await host.managerPort.sendInput(spawned.id, {
      message: 'now inspect the second area',
    })
    expect(queued.queueSize).toBe(1)
    client.releaseFirst.resolve()

    const settled = await host.managerPort.wait([spawned.id], 5_000)
    expect(settled.pending).toHaveLength(0)
    expect(settled.completed[0]).toMatchObject({
      historySessionId: spawned.historySessionId,
      status: 'completed',
    })
    expect(client.requests).toHaveLength(2)
    expect(client.requests[1]?.messages
      .filter(message => message.role !== 'system')
      .map(message => [message.role, message.content])).toEqual([
      ['user', 'inspect the first area'],
      ['assistant', 'answer:inspect the first area'],
      ['user', 'now inspect the second area'],
    ])

    const historySessionId = spawned.historySessionId
    if (!historySessionId) throw new Error('expected a persisted child history id')
    const transcript = await transcripts.load(historySessionId, {
      currentProjectDirectory: process.cwd(),
    })
    expect(transcript).toMatchObject({
      agentId: 'coder',
      cwd: process.cwd(),
      key: historySessionId,
      sessionId: historySessionId,
      totalApiCalls: 2,
      totalInputTokens: 34,
      totalOutputTokens: 10,
      turnCount: 2,
    })
    expect(transcript?.messages
      .filter(message => message.role !== 'system')
      .map(message => [message.role, message.content])).toEqual([
      ['user', 'inspect the first area'],
      ['assistant', 'answer:inspect the first area'],
      ['user', 'now inspect the second area'],
      ['assistant', 'answer:now inspect the second area'],
    ])
    expect(transcript?.metadata).toMatchObject({
      session_kind: 'subagent',
      parent_session_id: 'feedbeef',
      root_session_id: 'feedbeef',
      subagent_id: spawned.id,
      history_session_id: historySessionId,
      status: 'completed',
      model: 'test-model',
      permission_mode: 'accept-all',
      delegated_permission_mode: 'accept-all',
      project_root: process.cwd(),
      title: 'Durable child review',
    })
    expect(childEvents.length).toBeGreaterThan(0)
    expect(childEvents.every(event => event.payload.history_session_id === historySessionId)).toBe(true)

    const parentMetadata: Record<string, unknown> = {}
    replacePersistedSubagentSnapshots(parentMetadata, settled.completed)
    expect(persistedSubagentSnapshotValues(parentMetadata)[0]).toMatchObject({
      history_session_id: historySessionId,
    })
  } finally {
    client.releaseFirst.resolve()
    releaseEvents()
    await host.manager.shutdown()
    await rm(directory, { force: true, recursive: true })
  }
})

class CancelledHistoryChildClient implements LlmClient {
  readonly started = Promise.withResolvers<void>()

  async *stream(_request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    this.started.resolve()
    await new Promise<void>((_resolve, reject) => {
      if (!signal) {
        reject(new Error('expected a delegated cancellation signal'))
        return
      }
      const cancel = (): void => reject(signal.reason ?? new Error('cancelled'))
      if (signal.aborted) cancel()
      else signal.addEventListener('abort', cancel, { once: true })
    })
    yield { content: 'unreachable' }
  }
}

test('cancelled native subagent runs atomically retain their attempted conversation and terminal status', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-cancelled-history-'))
  const transcripts = new DaemonTranscriptStore({
    currentProjectDirectory: process.cwd(),
    directory: join(directory, 'sessions'),
  })
  const client = new CancelledHistoryChildClient()
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['coder', agentDefinition('coder')],
      ['default', creatorDefinition('coder')],
    ]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
    transcriptStore: transcripts,
  })

  try {
    const spawned = await host.managerPort.spawn({
      creatorAgentId: 'default',
      message: 'inspect until cancelled',
      promptProfile: 'coder',
      sourceAgentId: 'decafbad',
      title: 'Cancelled child review',
    })
    await client.started.promise
    host.managerPort.close(spawned.id)
    await host.manager.shutdown()

    const historySessionId = spawned.historySessionId
    if (!historySessionId) throw new Error('expected a persisted child history id')
    const transcript = await transcripts.load(historySessionId, {
      currentProjectDirectory: process.cwd(),
    })
    expect(transcript?.metadata).toMatchObject({
      session_kind: 'subagent',
      status: 'cancelled',
      parent_session_id: 'decafbad',
      subagent_id: spawned.id,
    })
    expect(transcript?.messages.some(message => (
      message.role === 'user' && message.content === 'inspect until cancelled'
    ))).toBe(true)
  } finally {
    await host.manager.shutdown()
    await rm(directory, { force: true, recursive: true })
  }
})

test('background agents keep the parent turn live and deliver one joined result continuation', async () => {
  const sessionDirectory = await mkdtemp(join(tmpdir(), 'xerxes-joined-background-'))
  const alphaRelease = Promise.withResolvers<void>()
  const betaRelease = Promise.withResolvers<void>()
  const parentWaiting = Promise.withResolvers<void>()
  const parentRequests: CompletionRequest[] = []
  const definitions = new Map<string, AgentDefinition>([
    ['default', creatorDefinition('researcher')],
    ['researcher', agentDefinition('researcher')],
  ])
  const client: LlmClient = {
    async *stream(request): AsyncGenerator<LlmDelta> {
      const userText = request.messages
        .filter(message => message.role === 'user')
        .map(message => String(message.content))
        .join('\n')
      if (userText.includes('alpha child task')) {
        await alphaRelease.promise
        yield { content: 'alpha final report' }
        return
      }
      if (userText.includes('beta child task')) {
        await betaRelease.promise
        yield { content: 'beta final report' }
        return
      }

      parentRequests.push(request)
      if (userText.includes('[sub-agent events]')) {
        yield { content: 'Integrated alpha and beta reports.' }
        return
      }
      if (request.messages.some(message => message.role === 'tool' && message.name === 'SpawnAgents')) {
        parentWaiting.resolve()
        yield { content: 'The delegated reviews are still running.' }
        return
      }
      yield {
        toolCalls: [{
          id: 'spawn-background-batch',
          type: 'function',
          function: {
            name: 'SpawnAgents',
            arguments: {
              agents: [
                { name: 'alpha', prompt: 'alpha child task', subagent_type: 'researcher', title: 'Alpha review' },
                { name: 'beta', prompt: 'beta child task', subagent_type: 'researcher', title: 'Beta review' },
              ],
              wait: false,
            },
          },
        }],
      }
    },
  }
  const eventBus = new DaemonSubagentEventBus()
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: [],
  })
  registerClaudeAgentTools(registry, {
    backgroundAgents: host.turnCoordinator,
    manager: host.managerPort,
  })
  const runner = new AgentTurnRunner({
    agentDefinitions: definitions,
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    subagentCoordinator: host.turnCoordinator,
    subagentEvents: eventBus,
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: process.cwd(),
    model: 'test-model',
    sessionDirectory,
  })
  const events: DaemonEvent[] = []

  try {
    const turn = runtime.submitTurn('joined-background', 'run both reviews', event => events.push(event))
    await parentWaiting.promise
    await waitFor(() => parentRequests.length === 2)

    expect(runtime.sessionStatus('joined-background')).toMatchObject({
      activeTurnId: expect.any(String),
      status: 'working',
    })
    expect(events.some(event => event.type === 'turn_end')).toBe(false)

    alphaRelease.resolve()
    await waitFor(() => events.some(event => (
      event.type === 'subagent_event'
      && event.payload.title === 'Alpha review'
      && nestedEventType(event) === 'turn_end'
    )))
    expect(parentRequests).toHaveLength(2)
    expect(events.some(event => event.type === 'turn_end')).toBe(false)

    betaRelease.resolve()
    await turn

    expect(parentRequests).toHaveLength(3)
    const joinedContext = parentRequests[2]?.messages
      .filter(message => message.role === 'user')
      .map(message => String(message.content))
      .join('\n') ?? ''
    expect(joinedContext).toContain('[sub-agent events]')
    expect(joinedContext).toContain('Alpha review')
    expect(joinedContext).toContain('alpha final report')
    expect(joinedContext).toContain('Beta review')
    expect(joinedContext).toContain('beta final report')
    expect(events.filter(event => event.type === 'turn_end')).toHaveLength(1)
    expect(events.filter(event => event.type === 'text_part').map(event => event.payload.text)).toEqual([
      'The delegated reviews are still running.',
      'Integrated alpha and beta reports.',
    ])
    expect(runtime.sessionStatus('joined-background')).toMatchObject({
      activeTurnId: '',
      status: 'idle',
      turnCount: 1,
    })
  } finally {
    alphaRelease.resolve()
    betaRelease.resolve()
    await host.manager.shutdown()
    await rm(sessionDirectory, { force: true, recursive: true })
  }
})

function nestedEventType(event: DaemonEvent): unknown {
  return nestedEvent(event)?.type
}

function nestedPayload(event: DaemonEvent): Record<string, unknown> {
  const payload = nestedEvent(event)?.payload
  return isRecord(payload) ? payload : {}
}

function nestedEvent(event: DaemonEvent): Record<string, unknown> | undefined {
  const nested = event.payload.event
  return isRecord(nested) ? nested : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
