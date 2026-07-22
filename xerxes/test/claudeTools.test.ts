// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  MAX_SKILL_INDEX_ENTRIES,
  SkillRegistry,
  parseSkillMarkdown,
} from '../src/extensions/skills.js'
import { persistedSubagentSnapshotValues } from '../src/agents/subagentPersistence.js'
import { MCPClient } from '../src/mcp/client.js'
import { SpawnedAgentManager } from '../src/operators/subagents.js'
import { UserPromptManager } from '../src/operators/userPrompt.js'
import { JobStore } from '../src/cron/jobs.js'
import {
  ClaudeAgentTools,
  CLAUDE_AGENT_TOOL_DEFINITIONS,
  CLAUDE_MCP_TOOL_DEFINITIONS,
  CLAUDE_WORKFLOW_TOOL_DEFINITIONS,
  MCPClientRegistry,
  NativeWorktreeManager,
  RemoteTriggerRegistry,
  WorkflowState,
  claudeCompatibilityGaps,
  editNotebookCell,
  registerClaudeAgentTools,
  registerClaudeMcpTools,
  registerClaudeRemoteTools,
  registerClaudeSearchTools,
  registerClaudeSkillTool,
  registerClaudeWorkflowTools,
  type WorktreeManager,
} from '../src/tools/claudeTools/index.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import type { SpawnedAgentManagerPort, SpawnedAgentSnapshot } from '../src/operators/subagents.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

function toolCall(name: string, arguments_: JsonObject): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name, arguments: arguments_ } }
}

async function executeJson(registry: ToolRegistry, name: string, arguments_: JsonObject): Promise<unknown> {
  return JSON.parse(await registry.execute(toolCall(name, arguments_), { metadata: {} })) as unknown
}

async function waitUntil(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) return
    await Bun.sleep(1)
  }
  throw new Error('Timed out waiting for deterministic test condition')
}

function agentSnapshot(
  id: string,
  status: SpawnedAgentSnapshot['status'] = 'running',
  sourceAgentId?: string,
): SpawnedAgentSnapshot {
  return {
    agentId: 'coder',
    closed: status === 'closed',
    createdAt: '2026-01-01T00:00:00.000Z',
    id,
    lastInput: `task:${id}`,
    name: id,
    title: `Task ${id}`,
    promptProfile: 'coder',
    queueSize: 0,
    ...(sourceAgentId === undefined ? {} : { sourceAgentId }),
    status,
    updatedAt: '2026-01-01T00:00:00.000Z',
  }
}

test('Claude compatibility schemas use provider-portable scalar type keywords', () => {
  const definitions = [
    ...CLAUDE_AGENT_TOOL_DEFINITIONS,
    ...CLAUDE_MCP_TOOL_DEFINITIONS,
    ...CLAUDE_WORKFLOW_TOOL_DEFINITIONS,
  ]
  const visit = (value: unknown): void => {
    if (Array.isArray(value)) {
      for (const item of value) visit(item)
      return
    }
    if (value === null || typeof value !== 'object') return
    for (const [key, child] of Object.entries(value)) {
      if (key === 'type') expect(typeof child).toBe('string')
      visit(child)
    }
  }
  for (const definition of definitions) visit(definition)
})

test('SkillTool searches beyond the prompt index, bounds listings, and preserves exact activation', async () => {
  const skills = new SkillRegistry()
  const targetIndex = MAX_SKILL_INDEX_ENTRIES + 1
  for (let index = 0; index <= targetIndex; index += 1) {
    const target = index === targetIndex
    skills.register(parseSkillMarkdown(
      `---\nname: skill-${index}\ndescription: ${target ? 'Rare needle workflow' : `Routine workflow ${index}`}\ntags: [${target ? 'rare' : 'common'}]\n---\n${target ? 'Run the hidden needle instructions.' : `Run workflow ${index}.`}`,
      `/virtual/skill-${index}/SKILL.md`,
    ))
  }
  const registry = new ToolRegistry()
  const definition = registerClaudeSkillTool(registry, skills)

  expect(definition.function.description).toContain('Discover installed skills')
  expect(definition.function.parameters?.required).toBeUndefined()

  const listing = await registry.execute(toolCall('SkillTool', {}), { metadata: {} })
  expect(listing.match(/^  - /gmu)).toHaveLength(20)
  expect(listing).toContain('more matching skills omitted')

  const search = await registry.execute(toolCall('SkillTool', { query: 'needle', tags: ['rare'] }), {
    metadata: {},
  })
  expect(search).toContain(`skill-${targetIndex}: Rare needle workflow`)
  expect(search).not.toContain('Run the hidden needle instructions.')

  const miss = await registry.execute(toolCall('SkillTool', { skill_name: 'needle' }), { metadata: {} })
  expect(miss).toContain('No installed skill has that exact name.')
  expect(miss).toContain(`skill-${targetIndex}: Rare needle workflow`)

  const activated = await registry.execute(toolCall('SkillTool', {
    skill_name: `skill-${targetIndex}`,
    args: 'Apply it now.',
  }), { metadata: {} })
  expect(activated).toContain(`[Skill: skill-${targetIndex}]`)
  expect(activated).toContain('Run the hidden needle instructions.')
  expect(activated).toContain('User request: Apply it now.')
})

test('Claude agent tools map task lifecycle, outputs, and mailbox events to SpawnedAgentManager', async () => {
  const manager = new SpawnedAgentManager({
    idFactory: () => 'generated-agent',
    runner: async request => ({ content: `${request.agent.id}:${request.input.toUpperCase()}` }),
  })
  const registry = new ToolRegistry()
  registerClaudeAgentTools(registry, { manager })

  const created = await executeJson(registry, 'AgentTool', {
    prompt: 'inspect code',
    title: 'Inspect code',
    name: 'inspector',
    subagent_type: 'reviewer',
    wait: true,
    timeout: 1,
  }) as { id: string; status: string; last_output: string; title: string }
  expect(created).toMatchObject({
    id: 'inspector',
    status: 'completed',
    last_output: 'inspector:INSPECT CODE',
    title: 'Inspect code',
  })
  expect(await registry.execute(toolCall('TaskOutputTool', { task_id: 'inspector' }), { metadata: {} }))
    .toBe('inspector:INSPECT CODE')

  const messages = await executeJson(registry, 'CheckAgentMessages', { peek: true }) as {
    latest_seq: number
    events: Array<{ event: string; title: string }>
  }
  expect(messages.latest_seq).toBeGreaterThan(0)
  expect(messages.events.map(event => event.event)).toContain('agent_spawned')
  expect(messages.events.map(event => event.event)).toContain('agent_output')
  expect(messages.events.every(event => event.title.length > 0)).toBeTrue()

  const batch = await executeJson(registry, 'SpawnAgents', {
    agents: [
      { name: 'one', prompt: 'first', title: 'First task' },
      { name: 'two', prompt: 'second', subagent_type: 'coder', title: 'Second task' },
    ],
    wait: true,
    timeout: 1,
  }) as Array<{ status: string; last_output: string }>
  expect(batch).toHaveLength(2)
  expect(batch.every(entry => entry.status === 'completed')).toBeTrue()
  expect(batch.map(entry => entry.last_output)).toEqual(['one:FIRST', 'two:SECOND'])
})

test('Claude agent management tools cannot inspect or mutate another session handles', async () => {
  const snapshots = [
    agentSnapshot('session-a-task', 'completed', 'session-a'),
    {
      ...agentSnapshot('session-b-task', 'running', 'session-b'),
      historySessionId: '0123456789abcdef0123456789abcdef',
    },
  ]
  const manager: SpawnedAgentManagerPort = {
    close: id => ({ ...snapshots.find(snapshot => snapshot.id === id)!, closed: true, previousStatus: 'running', status: 'closed' }),
    listHandles: () => snapshots,
    resume: id => snapshots.find(snapshot => snapshot.id === id)!,
    sendInput: async id => snapshots.find(snapshot => snapshot.id === id)!,
    spawn: async () => snapshots[0]!,
    wait: async () => ({ completed: [], pending: [] }),
  }
  const tools = new ClaudeAgentTools({ manager })
  const sessionA = { agentId: 'session-b', metadata: {}, sessionId: 'session-a' }
  const sessionBByAgentFallback = { agentId: 'session-b', metadata: {} }

  const listedA = await tools.execute('TaskListTool', {}, sessionA) as Array<{ id: string }>
  const listedB = await tools.execute('TaskListTool', {}, sessionBByAgentFallback) as Array<{
    history_session_id?: string
    id: string
  }>
  expect(listedA.map(snapshot => snapshot.id)).toEqual(['session-a-task'])
  expect(listedB.map(snapshot => snapshot.id)).toEqual(['session-b-task'])
  expect(listedB[0]?.history_session_id).toBe('0123456789abcdef0123456789abcdef')
  expect(await tools.execute('TaskGetTool', {
    task_id: 'session-b-task',
  }, sessionBByAgentFallback)).toMatchObject({
    history_session_id: '0123456789abcdef0123456789abcdef',
  })

  const crossSessionCalls: Array<{ readonly inputs: JsonObject; readonly name: string }> = [
    { name: 'SendMessageTool', inputs: { target: 'session-b-task', message: 'leak' } },
    { name: 'TaskGetTool', inputs: { task_id: 'session-b-task' } },
    { name: 'TaskOutputTool', inputs: { task_id: 'session-b-task' } },
    { name: 'TaskStopTool', inputs: { task_id: 'session-b-task' } },
    { name: 'TaskUpdateTool', inputs: { task_id: 'session-b-task', message: 'leak' } },
    { name: 'AwaitAgents', inputs: { agent_ids: ['session-b-task'], timeout_seconds: 0 } },
    { name: 'PeekAgent', inputs: { target: 'session-b-task' } },
    { name: 'ResetAgent', inputs: { target: 'session-b-task', new_prompt: 'leak' } },
  ]
  for (const call of crossSessionCalls) {
    await expect(tools.execute(call.name, call.inputs, sessionA)).rejects.toThrow('managed subagent not found')
  }

  const awaited = await tools.execute('AwaitAgents', {
    timeout_seconds: 0,
    wake_on: 'none',
  }, sessionA) as { agents: Array<{ id: string }> }
  expect(awaited.agents).toEqual([])
})

test('Claude agent message drains are filtered and cursor-isolated by session owner', async () => {
  const manager: SpawnedAgentManagerPort = {
    close: id => ({ ...agentSnapshot(id, 'closed'), previousStatus: 'running' }),
    listHandles: () => [
      agentSnapshot('session-a-task', 'running', 'session-a'),
      agentSnapshot('session-b-task', 'running', 'session-b'),
    ],
    resume: id => agentSnapshot(id),
    sendInput: async id => agentSnapshot(id ?? 'missing'),
    spawn: async () => agentSnapshot('unused'),
    wait: async () => ({ completed: [], pending: [] }),
  }
  const tools = new ClaudeAgentTools({ manager })
  const sessionA = { metadata: {}, sessionId: 'session-a' }
  const sessionB = { metadata: {}, sessionId: 'session-b' }

  const firstA = await tools.execute('CheckAgentMessages', {}, sessionA) as {
    events: Array<{ agentId: string; sourceAgentId?: string }>
    latest_seq: number
  }
  const firstB = await tools.execute('CheckAgentMessages', {}, sessionB) as {
    events: Array<{ agentId: string; sourceAgentId?: string }>
    latest_seq: number
  }
  expect(firstA.events).toMatchObject([{ agentId: 'session-a-task', sourceAgentId: 'session-a' }])
  expect(firstB.events).toMatchObject([{ agentId: 'session-b-task', sourceAgentId: 'session-b' }])
  expect(firstA.latest_seq).toBeGreaterThan(0)
  expect(firstB.latest_seq).toBeGreaterThan(firstA.latest_seq)

  expect(await tools.execute('CheckAgentMessages', {}, sessionA)).toMatchObject({ events: [] })
  expect(await tools.execute('CheckAgentMessages', {}, sessionB)).toMatchObject({ events: [] })
})

test('agent creation schemas require concise titles for single and batch delegation', async () => {
  const byName = new Map(CLAUDE_AGENT_TOOL_DEFINITIONS.map(tool => [tool.function.name, tool]))
  const required = (name: string) => byName.get(name)?.function.parameters.required as string[] | undefined
  const properties = byName.get('AgentTool')?.function.parameters.properties as Record<string, Record<string, unknown>>
  const spawnProperties = byName.get('SpawnAgents')?.function.parameters.properties as Record<string, Record<string, unknown>>
  const awaitProperties = byName.get('AwaitAgents')?.function.parameters.properties as Record<string, Record<string, unknown>>

  expect(required('AgentTool')).toContain('title')
  expect(required('TaskCreateTool')).toContain('title')
  expect(properties.title?.maxLength).toBe(48)
  expect(spawnProperties.agents?.type).toBe('array')
  expect(spawnProperties.agents?.minItems).toBe(1)
  expect(spawnProperties.agents?.maxItems).toBe(32)
  expect(properties.run_in_background?.default).toBe(false)
  expect(properties.wait?.default).toBe(true)
  expect(spawnProperties.wait?.default).toBe(true)
  expect(awaitProperties.agent_ids?.type).toBe('array')
  expect(awaitProperties.agent_ids?.items).toEqual({ type: 'string' })
  expect(awaitProperties.wake_on?.default).toBe('all')

  const registry = new ToolRegistry()
  const manager = new SpawnedAgentManager({ runner: async () => ({ content: 'done' }) })
  registerClaudeAgentTools(registry, {
    manager,
  })
  await expect(registry.execute(toolCall('AgentTool', { prompt: 'missing title' }), { metadata: {} }))
    .rejects.toThrow('title')
  await expect(registry.execute(toolCall('SpawnAgents', {
    agents: [{ prompt: 'missing batch title' }],
  }), { metadata: {} })).rejects.toThrow('title')
  const aboveLegacyLimit = Array.from({ length: 9 }, (_, index) => ({
    prompt: `task ${index}`,
    title: `Task ${index}`,
  }))
  const accepted = JSON.parse(await registry.execute(toolCall('SpawnAgents', {
    agents: aboveLegacyLimit,
    wait: false,
  }), { metadata: {} })) as Record<string, unknown>
  expect(accepted).toMatchObject({ accepted_count: 9, omitted_count: 1, shown_count: 8 })
  expect(manager.listHandles()).toHaveLength(9)

  const oversized = Array.from({ length: 33 }, (_, index) => ({
    prompt: `oversized task ${index}`,
    title: `Oversized ${index}`,
  }))
  await expect(registry.execute(toolCall('SpawnAgents', {
    agents: oversized,
    wait: false,
  }), { metadata: {} })).rejects.toThrow('at most 32')
  expect(manager.listHandles()).toHaveLength(9)
})

test('SpawnAgents runs registrations through a bounded concurrency pool while preserving batch order', async () => {
  const snapshots: SpawnedAgentSnapshot[] = []
  let activeSpawns = 0
  let peakSpawns = 0
  const manager: SpawnedAgentManagerPort = {
    close: id => ({ ...agentSnapshot(id, 'closed'), previousStatus: 'running' }),
    listHandles: () => snapshots,
    resume: id => agentSnapshot(id),
    sendInput: async id => agentSnapshot(id ?? 'missing'),
    spawn: async options => {
      activeSpawns += 1
      peakSpawns = Math.max(peakSpawns, activeSpawns)
      await Bun.sleep(5)
      const snapshot = agentSnapshot(options?.nickname ?? `agent-${snapshots.length}`)
      snapshots.push(snapshot)
      activeSpawns -= 1
      return snapshot
    },
    wait: async () => ({ completed: [], pending: snapshots }),
  }
  const tools = new ClaudeAgentTools({ manager })
  const agents = Array.from({ length: 24 }, (_, index) => ({
    name: `bounded-${index}`,
    prompt: `task ${index}`,
    title: `Bounded task ${index}`,
  }))

  const result = await tools.execute('SpawnAgents', { agents, wait: false }, { metadata: {} }) as {
    agents: Array<{ name: string }>
    omitted_count: number
  }

  expect(peakSpawns).toBe(8)
  expect(result.agents.map(snapshot => snapshot.name)).toEqual(agents.slice(0, 8).map(agent => agent.name))
  expect(result.omitted_count).toBe(16)
})

test('SpawnAgents honors a configured spawn concurrency override', async () => {
  const snapshots: SpawnedAgentSnapshot[] = []
  let activeSpawns = 0
  let peakSpawns = 0
  const manager: SpawnedAgentManagerPort = {
    close: id => ({ ...agentSnapshot(id, 'closed'), previousStatus: 'running' }),
    listHandles: () => snapshots,
    resume: id => agentSnapshot(id),
    sendInput: async id => agentSnapshot(id ?? 'missing'),
    spawn: async options => {
      activeSpawns += 1
      peakSpawns = Math.max(peakSpawns, activeSpawns)
      await Bun.sleep(5)
      const snapshot = agentSnapshot(options?.nickname ?? `agent-${snapshots.length}`)
      snapshots.push(snapshot)
      activeSpawns -= 1
      return snapshot
    },
    wait: async () => ({ completed: [], pending: snapshots }),
  }
  const tools = new ClaudeAgentTools({ manager, spawnConcurrency: 2 })
  const agents = Array.from({ length: 6 }, (_, index) => ({
    name: `pool-${index}`,
    prompt: `task ${index}`,
    title: `Pool task ${index}`,
  }))

  await tools.execute('SpawnAgents', { agents, wait: false }, { metadata: {} })

  expect(peakSpawns).toBe(2)
  expect(snapshots).toHaveLength(6)
  expect(() => new ClaudeAgentTools({ manager, spawnConcurrency: 0 })).toThrow('positive integer')
})

test('SpawnAgents accepts exactly 32 registrations at the documented batch boundary', async () => {
  let registered = 0
  const snapshots: SpawnedAgentSnapshot[] = []
  const manager: SpawnedAgentManagerPort = {
    close: id => ({ ...agentSnapshot(id, 'closed'), previousStatus: 'running' }),
    listHandles: () => snapshots,
    resume: id => agentSnapshot(id),
    sendInput: async id => agentSnapshot(id ?? 'missing'),
    spawn: async options => {
      registered += 1
      const snapshot = agentSnapshot(options?.nickname ?? `boundary-${registered}`)
      snapshots.push(snapshot)
      return snapshot
    },
    wait: async () => ({ completed: [], pending: [] }),
  }
  const tools = new ClaudeAgentTools({ manager })
  const agents = Array.from({ length: 32 }, (_, index) => ({
    name: `boundary-${index}`,
    prompt: `task ${index}`,
    title: `Boundary task ${index}`,
  }))
  const metadata: Record<string, unknown> = {}

  const result = await tools.execute('SpawnAgents', { agents, wait: false }, { metadata })

  expect(registered).toBe(32)
  expect(result).toMatchObject({
    accepted_count: 32,
    agent_count: 32,
    shown_count: 8,
    omitted_count: 24,
  })
  expect(JSON.stringify(result).length).toBeLessThan(5_000)
  expect(persistedSubagentSnapshotValues(metadata)).toHaveLength(32)

  const overBoundary = Array.from({ length: 33 }, (_, index) => ({
    name: `over-${index}`,
    prompt: `task ${index}`,
    title: `Over task ${index}`,
  }))
  await expect(tools.execute('SpawnAgents', { agents: overBoundary, wait: false }, { metadata: {} }))
    .rejects.toThrow('at most 32')
  expect(registered).toBe(32)
})

test('TaskListTool pages compact rows and large AwaitAgents results stay bounded', async () => {
  const snapshots = Array.from({ length: 120 }, (_, index) => ({
    ...agentSnapshot(`paged-${index}`, 'completed', 'paged-session'),
    lastOutput: `full output ${index} ${'x'.repeat(2_000)}`,
  }))
  const manager: SpawnedAgentManagerPort = {
    close: id => ({ ...snapshots.find(snapshot => snapshot.id === id)!, closed: true, previousStatus: 'completed', status: 'closed' }),
    listHandles: () => snapshots,
    resume: id => snapshots.find(snapshot => snapshot.id === id)!,
    sendInput: async id => snapshots.find(snapshot => snapshot.id === id)!,
    spawn: async () => snapshots[0]!,
    wait: async targets => ({
      completed: targets.map(target => snapshots.find(snapshot => snapshot.id === target)!),
      pending: [],
    }),
  }
  const tools = new ClaudeAgentTools({ manager })
  const context = { metadata: {}, sessionId: 'paged-session' }

  const first = await tools.execute('TaskListTool', {}, context) as Array<Record<string, unknown>>
  const third = await tools.execute('TaskListTool', { offset: 100, limit: 50 }, context) as Array<Record<string, unknown>>
  const awaited = await tools.execute('AwaitAgents', {
    agent_ids: snapshots.map(snapshot => snapshot.id),
    timeout_seconds: 0,
    wake_on: 'all',
  }, context) as Record<string, unknown>

  expect(first).toHaveLength(50)
  expect(first[0]).toMatchObject({ id: 'paged-0', has_output: true, status: 'completed' })
  expect(first[0]).not.toHaveProperty('last_output')
  expect(third).toHaveLength(20)
  expect(third[0]).toMatchObject({ id: 'paged-100' })
  expect(awaited).toMatchObject({ agent_count: 120, shown_count: 8, omitted_count: 112 })
  expect(JSON.stringify(awaited).length).toBeLessThan(5_000)
})

test('AwaitAgents defaults to the complete tracked cohort rather than the first finisher', async () => {
  const snapshots = [
    agentSnapshot('first', 'completed', 'await-default-session'),
    agentSnapshot('second', 'running', 'await-default-session'),
  ]
  const manager: SpawnedAgentManagerPort = {
    close: id => ({ ...snapshots.find(snapshot => snapshot.id === id)!, closed: true, previousStatus: 'running', status: 'closed' }),
    listHandles: () => snapshots,
    resume: id => snapshots.find(snapshot => snapshot.id === id)!,
    sendInput: async id => snapshots.find(snapshot => snapshot.id === id)!,
    spawn: async () => snapshots[0]!,
    wait: async () => ({ completed: [], pending: [] }),
  }
  const tools = new ClaudeAgentTools({ manager })
  const result = await tools.execute('AwaitAgents', {
    agent_ids: snapshots.map(snapshot => snapshot.id),
    timeout_seconds: 0,
  }, { metadata: {}, sessionId: 'await-default-session' }) as Record<string, unknown>

  expect(result).toMatchObject({ wake_on: 'all', wake_reason: 'timeout' })
})

test('SpawnAgents abort stops claiming new registrations and closes in-flight successes', async () => {
  const controller = new AbortController()
  const release = Promise.withResolvers<void>()
  const snapshots: SpawnedAgentSnapshot[] = []
  const started: string[] = []
  const closed: string[] = []
  const manager: SpawnedAgentManagerPort = {
    close: id => {
      closed.push(id)
      return { ...agentSnapshot(id, 'closed'), previousStatus: 'running' }
    },
    listHandles: () => snapshots,
    resume: id => agentSnapshot(id),
    sendInput: async id => agentSnapshot(id ?? 'missing'),
    spawn: async options => {
      const name = options?.nickname ?? `abort-${started.length}`
      started.push(name)
      await release.promise
      const snapshot = agentSnapshot(name)
      snapshots.push(snapshot)
      return snapshot
    },
    wait: async () => ({ completed: [], pending: [] }),
  }
  const tools = new ClaudeAgentTools({ manager })
  const agents = Array.from({ length: 16 }, (_, index) => ({
    name: `abort-${index}`,
    prompt: `task ${index}`,
    title: `Abort task ${index}`,
  }))

  const pending = tools.execute('SpawnAgents', { agents, wait: false }, { metadata: {} }, controller.signal)
  // With the bounded pool, only the in-flight registrations have started.
  await waitUntil(() => started.length === 8)
  controller.abort(new Error('stop registration'))
  release.resolve()

  await expect(pending).rejects.toThrow('stop registration')
  expect(started).toHaveLength(8)
  expect(closed.sort()).toEqual([...started].sort())
})

test('SpawnAgents first failure stops claiming new registrations and closes partial successes', async () => {
  const fail = Promise.withResolvers<void>()
  const release = Promise.withResolvers<void>()
  const snapshots: SpawnedAgentSnapshot[] = []
  const started: string[] = []
  const closed: string[] = []
  const manager: SpawnedAgentManagerPort = {
    close: id => {
      closed.push(id)
      return { ...agentSnapshot(id, 'closed'), previousStatus: 'running' }
    },
    listHandles: () => snapshots,
    resume: id => agentSnapshot(id),
    sendInput: async id => agentSnapshot(id ?? 'missing'),
    spawn: async options => {
      const name = options?.nickname ?? `failure-${started.length}`
      started.push(name)
      if (name === 'failure-0') {
        await fail.promise
        throw new Error('registration failed')
      }
      await release.promise
      const snapshot = agentSnapshot(name)
      snapshots.push(snapshot)
      return snapshot
    },
    wait: async () => ({ completed: [], pending: [] }),
  }
  const tools = new ClaudeAgentTools({ manager })
  const agents = Array.from({ length: 16 }, (_, index) => ({
    name: `failure-${index}`,
    prompt: `task ${index}`,
    title: `Failure task ${index}`,
  }))

  const pending = tools.execute('SpawnAgents', { agents, wait: false }, { metadata: {} })
  await waitUntil(() => started.length === 8)
  fail.resolve()
  await Bun.sleep(0)
  release.resolve()

  await expect(pending).rejects.toThrow('registration failed')
  expect(started).toHaveLength(8)
  expect(closed.sort()).toEqual(started.filter(name => name !== 'failure-0').sort())
})

test('stale subagent targets receive non-retry guidance after runtime attachment is lost', async () => {
  const manager: SpawnedAgentManagerPort = {
    close: id => ({ ...agentSnapshot(id, 'closed'), previousStatus: 'running' }),
    listHandles: () => [],
    resume: id => agentSnapshot(id),
    sendInput: async id => agentSnapshot(id ?? 'missing'),
    spawn: async () => agentSnapshot('unused'),
    wait: async () => ({ completed: [], pending: [] }),
  }
  const tools = new ClaudeAgentTools({ manager })
  const context = { metadata: {}, sessionId: 'resumed-session' }

  await expect(tools.execute('TaskGetTool', {
    task_id: 'eyvan_persistence_hunter',
  }, context)).rejects.toThrow(
    'Validation error for task_id: managed subagent not found; TaskListTool returned no tasks attached to the current runtime',
  )
  await expect(tools.execute('AwaitAgents', {
    agent_ids: ['eyvan_persistence_hunter', 'eyvan_core_hunter'],
    timeout_seconds: 60,
    wake_on: 'all',
  }, context)).rejects.toThrow(
    'Validation error for agent_ids: managed subagent not found; TaskListTool returned no tasks attached to the current runtime. Do not retry stale names or ids',
  )
})

test('foreground agent waits stop promptly when the parent turn is cancelled', async () => {
  const waitStarted = Promise.withResolvers<void>()
  const snapshot = agentSnapshot('interruptible')
  const closed: string[] = []
  const manager: SpawnedAgentManagerPort = {
    close: id => {
      closed.push(id)
      return { ...snapshot, closed: true, previousStatus: snapshot.status, status: 'closed' }
    },
    listHandles: () => [closed.length ? agentSnapshot('interruptible', 'closed') : snapshot],
    resume: () => snapshot,
    sendInput: async () => snapshot,
    spawn: async () => snapshot,
    wait: async () => {
      waitStarted.resolve()
      return await new Promise(() => undefined)
    },
  }
  const tools = new ClaudeAgentTools({ manager })
  const controller = new AbortController()
  const pending = tools.execute('AgentTool', {
    prompt: 'wait forever',
    title: 'Wait for cancellation',
    timeout: 60,
    wait: true,
  }, { metadata: {} }, controller.signal)
  await waitStarted.promise

  const started = performance.now()
  controller.abort(new Error('parent turn cancelled'))
  await expect(pending).rejects.toThrow('parent turn cancelled')
  expect(performance.now() - started).toBeLessThan(100)
  expect(closed).toEqual(['interruptible'])
})

test('foreground SpawnAgents cancellation closes only children spawned by that call', async () => {
  const waitStarted = Promise.withResolvers<void>()
  const snapshots = [agentSnapshot('preexisting')]
  const closed: string[] = []
  const manager: SpawnedAgentManagerPort = {
    close: id => {
      closed.push(id)
      return { ...agentSnapshot(id, 'closed'), previousStatus: 'running' }
    },
    listHandles: () => snapshots.map(snapshot => closed.includes(snapshot.id) ? agentSnapshot(snapshot.id, 'closed') : snapshot),
    resume: id => agentSnapshot(id),
    sendInput: async id => agentSnapshot(id ?? 'missing'),
    spawn: async options => {
      const snapshot = agentSnapshot(options?.nickname ?? `agent-${snapshots.length}`)
      snapshots.push(snapshot)
      return snapshot
    },
    wait: async () => {
      waitStarted.resolve()
      return await new Promise(() => undefined)
    },
  }
  const tools = new ClaudeAgentTools({ manager })
  const controller = new AbortController()
  const pending = tools.execute('SpawnAgents', {
    agents: [
      { name: 'batch-a', prompt: 'first', title: 'Batch task A' },
      { name: 'batch-b', prompt: 'second', title: 'Batch task B' },
    ],
    wait: true,
  }, { metadata: {} }, controller.signal)
  await waitStarted.promise
  controller.abort(new Error('parent turn cancelled'))

  await expect(pending).rejects.toThrow('parent turn cancelled')
  expect(closed).toEqual(['batch-a', 'batch-b'])
})

test('background AgentTool can detach but an already-aborted SpawnAgents batch registers nothing', async () => {
  const snapshots: SpawnedAgentSnapshot[] = []
  const closed: string[] = []
  const tracked: string[][] = []
  let waits = 0
  const manager: SpawnedAgentManagerPort = {
    close: id => {
      closed.push(id)
      return { ...agentSnapshot(id, 'closed'), previousStatus: 'running' }
    },
    listHandles: () => snapshots,
    resume: id => agentSnapshot(id),
    sendInput: async id => agentSnapshot(id ?? 'missing'),
    spawn: async options => {
      const snapshot = agentSnapshot(options?.nickname ?? `background-${snapshots.length + 1}`)
      snapshots.push(snapshot)
      return snapshot
    },
    wait: async () => {
      waits += 1
      return { completed: [], pending: snapshots }
    },
  }
  const tools = new ClaudeAgentTools({
    backgroundAgents: {
      consume: () => undefined,
      track: observed => tracked.push(observed.map(snapshot => snapshot.id)),
    },
    manager,
  })
  const controller = new AbortController()
  controller.abort(new Error('parent already cancelled'))

  const single = await tools.execute('AgentTool', {
    name: 'detached-one',
    prompt: 'continue independently',
    title: 'Continue independently',
    run_in_background: true,
  }, { metadata: {} }, controller.signal) as { id: string }
  await expect(tools.execute('SpawnAgents', {
    agents: [{ name: 'detached-two', prompt: 'also continue', title: 'Also continue' }],
    wait: false,
  }, { metadata: {} }, controller.signal)).rejects.toThrow('parent already cancelled')

  expect(single.id).toBe('detached-one')
  expect(waits).toBe(0)
  expect(closed).toEqual([])
  expect(snapshots.map(snapshot => snapshot.id)).toEqual(['detached-one'])
  expect(tracked).toEqual([['detached-one']])
})

test('AwaitAgents observes the exact tracked cohort even when every child already finished', async () => {
  const owner = 'live-session'
  let snapshot = agentSnapshot('fast-child', 'running', owner)
  const tracked = new Set<string>()
  const consumed: Array<{ id: string; status: string }> = []
  const manager: SpawnedAgentManagerPort = {
    close: id => ({ ...snapshot, id, closed: true, previousStatus: snapshot.status, status: 'closed' }),
    listHandles: () => [snapshot],
    resume: () => snapshot,
    sendInput: async () => snapshot,
    spawn: async options => {
      snapshot = agentSnapshot(options?.nickname ?? 'fast-child', 'running', options?.sourceAgentId)
      return snapshot
    },
    wait: async () => ({ completed: [], pending: [snapshot] }),
  }
  const tools = new ClaudeAgentTools({
    backgroundAgents: {
      consume: observed => {
        for (const item of observed) {
          consumed.push({ id: item.id, status: item.status })
          tracked.delete(item.id)
        }
      },
      track: observed => {
        for (const item of observed) tracked.add(item.id)
      },
      trackedIds: sourceAgentId => sourceAgentId === owner ? [...tracked] : [],
    },
    manager,
  })
  const context = { metadata: {}, sessionId: owner }

  await tools.execute('SpawnAgents', {
    agents: [{ name: 'fast-child', prompt: 'finish quickly', title: 'Fast child' }],
    wait: false,
  }, context)
  snapshot = { ...snapshot, lastOutput: 'finished before wait', status: 'completed' }

  const started = performance.now()
  const result = await tools.execute('AwaitAgents', {
    timeout_seconds: 5,
    wake_on: 'all',
  }, context) as { agents: Array<{ id: string; status: string }>; wake_reason: string }

  expect(performance.now() - started).toBeLessThan(100)
  expect(result).toMatchObject({
    wake_reason: 'agents_done',
    agents: [{ id: 'fast-child', status: 'completed' }],
  })
  expect(consumed).toContainEqual({ id: 'fast-child', status: 'completed' })
  expect(tracked.size).toBe(0)
})

test('ResetAgent replaces the tracked handle instead of retaining the closed task', async () => {
  const owner = 'reset-session'
  let snapshot = agentSnapshot('old-task', 'running', owner)
  const tracked = new Set<string>()
  const consumed: Array<{ id: string; status: string }> = []
  const manager: SpawnedAgentManagerPort = {
    close: id => ({ ...snapshot, id, closed: true, previousStatus: snapshot.status, status: 'closed' }),
    listHandles: () => [snapshot],
    resume: () => ({ ...snapshot, status: 'idle' }),
    sendInput: async () => {
      snapshot = agentSnapshot('new-task', 'running', owner)
      return snapshot
    },
    spawn: async () => snapshot,
    wait: async () => ({ completed: [], pending: [snapshot] }),
  }
  const tools = new ClaudeAgentTools({
    backgroundAgents: {
      consume: observed => {
        for (const item of observed) {
          consumed.push({ id: item.id, status: item.status })
          tracked.delete(item.id)
        }
      },
      track: observed => {
        for (const item of observed) tracked.add(item.id)
      },
      trackedIds: sourceAgentId => sourceAgentId === owner ? [...tracked] : [],
    },
    manager,
  })
  const context = { metadata: {}, sessionId: owner }

  await tools.execute('TaskCreateTool', {
    name: 'old-task',
    prompt: 'original task',
    title: 'Original task',
  }, context)
  await tools.execute('ResetAgent', {
    target: 'old-task',
    new_prompt: 'replacement task',
  }, context)

  expect(consumed).toContainEqual({ id: 'old-task', status: 'closed' })
  expect([...tracked]).toEqual(['new-task'])
})

test('SpawnAgents preserves request order across completed and pending partitions', async () => {
  const snapshots: SpawnedAgentSnapshot[] = []
  const tracked: string[][] = []
  const manager: SpawnedAgentManagerPort = {
    close: id => ({ ...agentSnapshot(id, 'closed'), previousStatus: 'running' }),
    listHandles: () => snapshots,
    resume: id => agentSnapshot(id),
    sendInput: async id => agentSnapshot(id ?? 'missing'),
    spawn: async options => {
      const snapshot = agentSnapshot(options?.nickname ?? `agent-${snapshots.length + 1}`)
      snapshots.push(snapshot)
      return snapshot
    },
    wait: async () => ({
      completed: [{ ...snapshots[1]!, status: 'completed' }],
      pending: [snapshots[0]!],
    }),
  }
  const tools = new ClaudeAgentTools({
    backgroundAgents: {
      consume: () => undefined,
      track: observed => tracked.push(observed.map(snapshot => snapshot.id)),
    },
    manager,
  })
  const result = await tools.execute('SpawnAgents', {
    agents: [
      { name: 'first', prompt: 'slow task', title: 'Slow task' },
      { name: 'second', prompt: 'fast task', title: 'Fast task' },
    ],
    timeout: 1,
    wait: true,
  }, { metadata: {} }) as Array<{ name: string }>

  expect(result.map(entry => entry.name)).toEqual(['first', 'second'])
  expect(tracked).toEqual([['first']])
})

test('SpawnAgents closes partial work when one parallel spawn fails', async () => {
  const manager = new SpawnedAgentManager({
    runner: async (_request, signal) => await new Promise(resolve => {
      signal.addEventListener('abort', () => resolve({ content: 'cancelled' }), { once: true })
    }),
  })
  const registry = new ToolRegistry()
  registerClaudeAgentTools(registry, { manager })

  await expect(registry.execute(toolCall('SpawnAgents', {
    agents: [
      { name: 'duplicate', prompt: 'first task', title: 'First duplicate' },
      { name: 'duplicate', prompt: 'second task', title: 'Second duplicate' },
    ],
    wait: false,
  }), { metadata: {} })).rejects.toThrow('already identifies a spawned agent')
  expect(manager.listHandles()).toMatchObject([{
    closed: true,
    name: 'duplicate',
    status: 'closed',
  }])
})

test('subagent outputs crossing the parent boundary are truncated with an explicit marker', async () => {
  const huge = 'y'.repeat(20_000)
  const snapshots = [{
    ...agentSnapshot('loud-task', 'completed', 'loud-session'),
    error: `E${huge}`,
    lastOutput: huge,
  }]
  const manager: SpawnedAgentManagerPort = {
    close: id => ({ ...snapshots.find(snapshot => snapshot.id === id)!, closed: true, previousStatus: 'completed', status: 'closed' }),
    listHandles: () => snapshots,
    resume: id => snapshots.find(snapshot => snapshot.id === id)!,
    sendInput: async id => snapshots.find(snapshot => snapshot.id === id)!,
    spawn: async () => snapshots[0]!,
    wait: async () => ({ completed: [], pending: [] }),
  }
  const tools = new ClaudeAgentTools({ manager })
  const context = { metadata: {}, sessionId: 'loud-session' }

  const output = await tools.execute('TaskOutputTool', { task_id: 'loud-task' }, context) as string
  expect(output.length).toBeLessThan(9_000)
  expect(output).toContain('[truncated 12000 chars]')

  const wire = await tools.execute('TaskGetTool', { task_id: 'loud-task' }, context) as {
    error: string
    last_output: string
  }
  expect(wire.last_output).toContain('[truncated 12000 chars]')
  expect(wire.last_output.length).toBeLessThan(9_000)
  expect(wire.error).toContain('[truncated')
  expect(wire.error.length).toBeLessThan(9_000)

  const messages = await tools.execute('CheckAgentMessages', { peek: true }, context) as {
    events: Array<{ event: string; output?: string }>
  }
  const outputEvent = messages.events.find(event => event.event === 'agent_output')
  expect(outputEvent?.output).toContain('[truncated 12000 chars]')
  expect(outputEvent?.output?.length).toBeLessThan(9_000)
})

test('Claude workflow tools preserve session todos/modes, bridge questions, and run injected plans', async () => {
  const prompts = new UserPromptManager({ idFactory: () => 'question' })
  const manager = new SpawnedAgentManager({
    runner: async request => ({ content: `finished:${request.input.match(/Step (\w+)/)?.[1] ?? 'unknown'}` }),
  })
  const worktrees: WorktreeManager = {
    create: async branch => ({ base: '/repo', branch: branch ?? 'generated', path: '/tmp/worktree' }),
    remove: async () => {},
  }
  const state = new WorkflowState()
  const registry = new ToolRegistry()
  registerClaudeWorkflowTools(registry, {
    state,
    userPromptManager: prompts,
    subagentManager: manager,
    worktreeManager: worktrees,
    planGenerator: {
      generate: async () => [
        { id: 'one', agent: 'researcher', description: 'Collect evidence', depends: [] },
        { id: 'two', agent: 'coder', description: 'Implement using evidence', depends: ['one'] },
      ],
    },
  })

  expect(await registry.execute(toolCall('TodoWriteTool', {
    todos: [{ content: 'Port tools', status: 'in_progress' }, { content: 'Verify', status: 'pending' }],
  }), { metadata: {} })).toContain('Progress: 0/2')
  expect((await executeJson(registry, 'SetInteractionModeTool', { mode: 'planner', reason: 'need design' }) as { mode: string }).mode)
    .toBe('plan')
  expect(state.isPlanMode).toBeTrue()
  expect((await executeJson(registry, 'SetInteractionModeTool', { mode: 'goal-runner' }) as { mode: string }).mode)
    .toBe('objective')
  expect(state.isPlanMode).toBeFalse()
  expect((await executeJson(registry, 'EnterWorktreeTool', { branch_name: 'tool-port' }) as { branch: string }).branch)
    .toBe('tool-port')

  const pending = registry.execute(toolCall('AskUserQuestionTool', { question: 'Proceed?' }), { metadata: {} })
  await Bun.sleep(0)
  prompts.answer('yes')
  expect(await pending).toBe('yes')

  const plan = await executeJson(registry, 'PlanTool', { objective: 'Ship tools', execute: true }) as {
    executed: boolean
    results: Array<{ id: string; status: string }>
  }
  expect(plan.executed).toBeTrue()
  expect(plan.results).toHaveLength(2)
  expect(plan.results.map(result => result.status)).toEqual(['completed', 'completed'])
  const matches = await executeJson(registry, 'ToolSearchTool', { query: 'todo' }) as Array<{ name: string }>
  expect(matches[0]?.name).toBe('TodoWriteTool')
})

test('Claude MCP tools call native MCP clients and enumerate resources', async () => {
  const client = new MCPClient({ name: 'demo', command: 'unused' })
  const calls: Array<{ readonly arguments_: JsonObject; readonly name: string }> = []
  client.callTool = async (name, arguments_) => {
    calls.push({ arguments_: arguments_ ?? {}, name })
    return { content: [{ type: 'text', text: 'called' }], structuredContent: { answer: 42 } }
  }
  client.listResources = async () => [{ name: 'guide', uri: 'memo://guide', description: 'Guide', mimeType: 'text/plain', serverName: 'demo' }]
  client.readResource = async () => ({ contents: [{ uri: 'memo://guide', text: 'Read me', mimeType: 'text/plain' }] })
  const clients = new MCPClientRegistry()
  clients.register(client)
  const registry = new ToolRegistry()
  registerClaudeMcpTools(registry, { clients })

  const tool = await executeJson(registry, 'MCPTool', { server_name: 'demo', tool_name: 'answer', arguments: { q: 1 } }) as {
    content: Array<{ type: string; text: string }>
    structured_content: { answer: number }
  }
  expect(tool.content).toEqual([{ type: 'text', text: 'called' }])
  expect(tool.structured_content).toEqual({ answer: 42 })
  await executeJson(registry, 'MCPTool', {
    server_name: 'demo',
    tool_name: 'answer',
    arguments: { q: 2 },
  })
  await executeJson(registry, 'MCPTool', { server_name: 'demo', tool_name: 'answer' })
  expect(calls).toEqual([
    { name: 'answer', arguments_: { q: 1 } },
    { name: 'answer', arguments_: { q: 2 } },
    { name: 'answer', arguments_: {} },
  ])
  expect(await executeJson(registry, 'ListMcpResourcesTool', {})).toEqual([
    { server_name: 'demo', uri: 'memo://guide', name: 'guide', description: 'Guide', mime_type: 'text/plain' },
  ])
  expect(await executeJson(registry, 'ReadMcpResourceTool', { server_name: 'demo', uri: 'memo://guide' })).toEqual({
    server_name: 'demo',
    uri: 'memo://guide',
    contents: [{ uri: 'memo://guide', mime_type: 'text/plain', text: 'Read me' }],
  })
})

test('Claude MCP tools bound oversized content and resource listings with truncation markers', async () => {
  const client = new MCPClient({ name: 'demo', command: 'unused' })
  const huge = 'z'.repeat(40_000)
  client.callTool = async () => ({
    content: [{ type: 'text', text: huge }],
    structuredContent: { blob: huge },
  })
  client.listResources = async () => Array.from({ length: 600 }, (_, index) => ({
    description: `Resource ${index}`,
    mimeType: 'text/plain',
    name: `res-${index}`,
    serverName: 'demo',
    uri: `memo://res-${index}`,
  }))
  client.readResource = async () => ({ contents: [{ mimeType: 'text/plain', text: huge, uri: 'memo://big' }] })
  const clients = new MCPClientRegistry()
  clients.register(client)
  const registry = new ToolRegistry()
  registerClaudeMcpTools(registry, { clients })

  const tool = await executeJson(registry, 'MCPTool', { server_name: 'demo', tool_name: 'big' }) as {
    content: Array<{ text: string }>
    structured_content: Record<string, unknown>
  }
  expect(tool.content[0]?.text).toContain('[truncated 8000 chars]')
  expect(tool.content[0]?.text.length).toBeLessThan(33_000)
  expect(tool.structured_content).toMatchObject({ truncated: true })

  const listing = await executeJson(registry, 'ListMcpResourcesTool', {}) as Array<Record<string, unknown>>
  expect(listing).toHaveLength(501)
  expect(listing.at(-1)).toMatchObject({ omitted_count: 100, truncated: true })

  const read = await executeJson(registry, 'ReadMcpResourceTool', { server_name: 'demo', uri: 'memo://big' }) as {
    contents: Array<{ text: string }>
  }
  expect(read.contents[0]?.text).toContain('[truncated 8000 chars]')
  expect(read.contents[0]?.text.length).toBeLessThan(33_000)
})

test('Claude remote tools enforce configured endpoints and persist cron jobs', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-claude-tools-'))
  try {
    const payloads: string[] = []
    const triggers = new RemoteTriggerRegistry({
      fetcher: async (_url, init) => {
        payloads.push(String(init.body ?? ''))
        return { ok: true, status: 202, text: async () => 'accepted' }
      },
    })
    triggers.register({ name: 'notify', url: 'https://example.com/webhook' })
    const registry = new ToolRegistry()
    registerClaudeRemoteTools(registry, { remoteTriggers: triggers, cronStore: new JobStore(join(directory, 'jobs.json')) })
    expect(await executeJson(registry, 'RemoteTriggerTool', { trigger_name: 'notify', payload: 'done' }))
      .toEqual({ name: 'notify', status: 202, ok: true, response: 'accepted' })
    expect(await executeJson(registry, 'RemoteTriggerTool', { trigger_name: 'notify' }))
      .toEqual({ name: 'notify', status: 202, ok: true, response: 'accepted' })
    expect(payloads).toEqual(['done', ''])
    await expect(executeJson(registry, 'RemoteTriggerTool', { trigger_name: 'missing' }))
      .rejects.toThrow('is not configured')
    const cronStore = new JobStore(join(directory, 'cron-jobs.json'))
    const cronRegistry = new ToolRegistry()
    registerClaudeRemoteTools(cronRegistry, { cronStore })
    const job = await executeJson(cronRegistry, 'ScheduleCronTool', { schedule: '0 9 * * *', prompt: 'daily report', name: 'daily' }) as {
      id: string
      name: string | null
      next_run_at: string
    }
    // Job ids are minted server-side; the caller name is only a label.
    expect(job.id).not.toBe('daily')
    expect(job.id).toMatch(/^[0-9a-f]{12}$/)
    expect(job.name).toBe('daily')
    expect(job.next_run_at).toContain('T')

    // A second schedule with the same name creates a new job instead of
    // overwriting the first one through a caller-chosen id.
    const second = await executeJson(cronRegistry, 'ScheduleCronTool', { schedule: '0 10 * * *', prompt: 'other report', name: 'daily' }) as {
      id: string
    }
    expect(second.id).not.toBe(job.id)
    expect(cronStore.listJobs()).toHaveLength(2)

    // Over-limit prompts are rejected before anything is persisted.
    await expect(executeJson(cronRegistry, 'ScheduleCronTool', {
      schedule: '0 9 * * *',
      prompt: 'x'.repeat(4_001),
    })).rejects.toThrow('at most 4000 characters')
    expect(cronStore.listJobs()).toHaveLength(2)

    const unconfigured = new ToolRegistry()
    registerClaudeRemoteTools(unconfigured, { cronStore: new JobStore(join(directory, 'unconfigured-jobs.json')) })
    await expect(executeJson(unconfigured, 'RemoteTriggerTool', { trigger_name: 'notify' }))
      .rejects.toThrow('requires an attached RemoteTriggerRegistry')
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

test('LSP delegates through a workspace-safe host adapter and notebook edits stay contained', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-claude-notebook-'))
  try {
    const resolver = new WorkspacePathResolver(directory)
    await Bun.write(join(directory, 'sample.ipynb'), JSON.stringify({ cells: [{ cell_type: 'code', source: ['old\n'] }] }))
    const registry = new ToolRegistry()
    registerClaudeSearchTools(registry, {
      paths: resolver,
      lspAdapter: { execute: async request => ({ ...request, source: 'fake' }) },
    })
    const lsp = await executeJson(registry, 'LSPTool', { action: 'definition', file_path: 'sample.ipynb', line: 1, character: 2 }) as {
      filePath: string
      source: string
    }
    expect(lsp.filePath).toBe(await resolver.resolve('sample.ipynb'))
    expect(lsp.source).toBe('fake')
    expect(await editNotebookCell({
      notebook_path: 'sample.ipynb',
      cell_index: 0,
      new_source: 'print(1)\n',
      cell_type: 'code',
    }, resolver)).toContain('Updated cell 0')
    expect(JSON.parse(await Bun.file(join(directory, 'sample.ipynb')).text())).toMatchObject({
      cells: [{ cell_type: 'code', source: ['print(1)\n'] }],
    })
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

test('compatibility gaps make unsupported legacy behavior explicit', () => {
  const gaps = claudeCompatibilityGaps()
  expect(gaps).toContain('LSPTool requires a host-provided language-server adapter.')
  expect(gaps.some(gap => gap.includes('fuzzy-whitespace'))).toBeTrue()
  expect(NativeWorktreeManager).toBeDefined()
  expect(ClaudeAgentTools).toBeDefined()
})

test('NativeWorktreeManager sweeps worktrees this process created but never exited', async () => {
  const git = async (cwd: string, arguments_: readonly string[]): Promise<void> => {
    const child = Bun.spawn(['git', ...arguments_], { cwd, stdin: 'ignore', stderr: 'pipe', stdout: 'pipe' })
    const [code, stderr] = await Promise.all([child.exited, new Response(child.stderr).text()])
    if (code !== 0) throw new Error(stderr)
  }
  const repository = await mkdtemp(join(tmpdir(), 'xerxes-worktree-sweep-'))
  try {
    await git(repository, ['init'])
    await git(repository, ['config', 'user.email', 'test@example.invalid'])
    await git(repository, ['config', 'user.name', 'Xerxes Test'])
    await Bun.write(join(repository, 'seed.txt'), 'seed\n')
    await git(repository, ['add', 'seed.txt'])
    await git(repository, ['commit', '-m', 'seed'])

    const manager = new NativeWorktreeManager(repository)
    const created = await manager.create('sweep-me')
    expect((await stat(created.path)).isDirectory()).toBeTrue()

    manager.sweepCreated()
    await expect(stat(created.path)).rejects.toMatchObject({ code: 'ENOENT' })
    expect((await stat(repository)).isDirectory()).toBeTrue()

    // A second sweep is a no-op, and the exit-time sweep ignores already-removed paths.
    manager.sweepCreated()
  } finally {
    await rm(repository, { force: true, recursive: true })
  }
})
