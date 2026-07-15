// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { MCPClient } from '../src/mcp/client.js'
import { SpawnedAgentManager } from '../src/operators/subagents.js'
import { UserPromptManager } from '../src/operators/userPrompt.js'
import { JobStore } from '../src/cron/jobs.js'
import {
  ClaudeAgentTools,
  CLAUDE_AGENT_TOOL_DEFINITIONS,
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
    agents: JSON.stringify([
      { name: 'one', prompt: 'first', title: 'First task' },
      { name: 'two', prompt: 'second', subagent_type: 'coder', title: 'Second task' },
    ]),
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
    agentSnapshot('session-b-task', 'running', 'session-b'),
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
  const listedB = await tools.execute('TaskListTool', {}, sessionBByAgentFallback) as Array<{ id: string }>
  expect(listedA.map(snapshot => snapshot.id)).toEqual(['session-a-task'])
  expect(listedB.map(snapshot => snapshot.id)).toEqual(['session-b-task'])

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

  expect(required('AgentTool')).toContain('title')
  expect(required('TaskCreateTool')).toContain('title')
  expect(properties.title?.maxLength).toBe(48)

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
  const oversized = Array.from({ length: 9 }, (_, index) => ({
    prompt: `task ${index}`,
    title: `Task ${index}`,
  }))
  await expect(registry.execute(toolCall('SpawnAgents', { agents: oversized }), { metadata: {} }))
    .rejects.toThrow('at most 8 agents')
  expect(manager.listHandles()).toEqual([])
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

test('explicit background agent calls survive an already-cancelled parent signal', async () => {
  const snapshots: SpawnedAgentSnapshot[] = []
  const closed: string[] = []
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
  const tools = new ClaudeAgentTools({ manager })
  const controller = new AbortController()
  controller.abort(new Error('parent already cancelled'))

  const single = await tools.execute('AgentTool', {
    name: 'detached-one',
    prompt: 'continue independently',
    title: 'Continue independently',
    run_in_background: true,
  }, { metadata: {} }, controller.signal) as { id: string }
  const batch = await tools.execute('SpawnAgents', {
    agents: [{ name: 'detached-two', prompt: 'also continue', title: 'Also continue' }],
    wait: false,
  }, { metadata: {} }, controller.signal) as Array<{ id: string }>

  expect(single.id).toBe('detached-one')
  expect(batch.map(item => item.id)).toEqual(['detached-two'])
  expect(waits).toBe(0)
  expect(closed).toEqual([])
})

test('SpawnAgents preserves request order across completed and pending partitions', async () => {
  const snapshots: SpawnedAgentSnapshot[] = []
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
    wait: async () => ({ completed: [snapshots[1]!], pending: [snapshots[0]!] }),
  }
  const tools = new ClaudeAgentTools({ manager })
  const result = await tools.execute('SpawnAgents', {
    agents: [
      { name: 'first', prompt: 'slow task', title: 'Slow task' },
      { name: 'second', prompt: 'fast task', title: 'Fast task' },
    ],
    timeout: 1,
    wait: true,
  }, { metadata: {} }) as Array<{ name: string }>

  expect(result.map(entry => entry.name)).toEqual(['first', 'second'])
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

  const tool = await executeJson(registry, 'MCPTool', { server_name: 'demo', tool_name: 'answer', arguments: '{"q":1}' }) as {
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
    const job = await executeJson(registry, 'ScheduleCronTool', { schedule: '0 9 * * *', prompt: 'daily report', name: 'daily' }) as {
      id: string
      next_run_at: string
    }
    expect(job.id).toBe('daily')
    expect(job.next_run_at).toContain('T')

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
