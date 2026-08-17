// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { loadAgentSpec } from '../src/agents/agentSpec.js'
import { loadAgentDefinitions, type AgentDefinition } from '../src/agents/definitions.js'
import { AgentOrchestrator } from '../src/agents/orchestrator.js'
import {
  filterSubagentTools,
  SubAgentManager,
  type SubagentWorktreePort,
} from '../src/agents/subagentManager.js'
import { SpawnedAgentManager } from '../src/operators/subagents.js'

function deferred(): { readonly promise: Promise<void>; readonly resolve: () => void } {
  let resolve: (() => void) | undefined
  const promise = new Promise<void>(res => {
    resolve = res
  })
  return { promise, resolve: () => resolve?.() }
}

test('unset YAML max_depth falls back to the manager ceiling and finite definition values only tighten it', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-hardening-depth-'))
  try {
    await writeFile(join(root, 'unset.yaml'), `version: 1
agent:
  name: unset-depth
  system_prompt: worker without a depth limit
`, 'utf8')
    const unsetSpec = loadAgentSpec(join(root, 'unset.yaml'))
    expect(unsetSpec.maxDepth).toBe(Number.POSITIVE_INFINITY)

    const unsetDefinition: AgentDefinition = {
      allowedTools: null,
      description: '',
      excludeTools: [],
      isolation: '',
      maxDepth: unsetSpec.maxDepth,
      model: '',
      name: unsetSpec.name,
      source: 'yaml',
      systemPrompt: unsetSpec.systemPrompt,
      tools: [],
    }
    const wideDefinition: AgentDefinition = { ...unsetDefinition, name: 'wide', maxDepth: 10 }

    const manager = new SubAgentManager({ maxDepth: 2, runner: () => ({ content: 'ok' }) })

    // Infinity (unset) must not disable the finite manager ceiling.
    const unsetBlocked = await manager.spawn({ prompt: 'too deep', depth: 2, agentDefinition: unsetDefinition })
    expect(unsetBlocked.status).toBe('failed')
    expect(unsetBlocked.error).toBe('Max depth (2, from manager default) exceeded: cannot spawn at depth 2')

    // A definition may never widen the manager ceiling either.
    const wideBlocked = await manager.spawn({ prompt: 'too deep', depth: 2, agentDefinition: wideDefinition })
    expect(wideBlocked.status).toBe('failed')
    expect(wideBlocked.error).toBe('Max depth (2, from manager default) exceeded: cannot spawn at depth 2')

    // A definition tighter than the manager still wins.
    const tight: AgentDefinition = { ...unsetDefinition, name: 'tight', maxDepth: 1 }
    const tightBlocked = await manager.spawn({ prompt: 'too deep', depth: 1, agentDefinition: tight })
    expect(tightBlocked.status).toBe('failed')
    expect(tightBlocked.error).toBe("Max depth (1, from agent definition 'tight') exceeded: cannot spawn at depth 1")

    const shallow = await manager.spawn({ prompt: 'shallow', depth: 1, agentDefinition: unsetDefinition })
    await manager.wait(shallow.id, 1_000)
    expect(shallow.status).toBe('completed')
    await manager.close()
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('agent max_depth rejects negative and unsafe integers in YAML and Markdown', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-hardening-invalid-depth-'))
  try {
    for (const [name, value] of [['negative', '-1'], ['unsafe', '9007199254740992']] as const) {
      const yamlPath = join(root, `${name}.yaml`)
      await writeFile(yamlPath, `version: 1\nagent:\n  name: ${name}\n  system_prompt: invalid\n  max_depth: ${value}\n`, 'utf8')
      expect(() => loadAgentSpec(yamlPath)).toThrow(/max_depth.*non-negative safe integer/u)

      const markdownPath = join(root, `${name}.md`)
      await writeFile(markdownPath, `---\nmax_depth: ${value}\n---\ninvalid\n`, 'utf8')
      expect(() => loadAgentDefinitions({
        builtinDefinitions: new Map(),
        cwd: root,
        userDirectory: join(root, 'none'),
        projectDirectory: root,
      })).not.toThrow()
      expect(loadAgentDefinitions({
        builtinDefinitions: new Map(),
        cwd: root,
        userDirectory: join(root, 'none'),
        projectDirectory: root,
      }).has(name)).toBeFalse()
    }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('interrupt replacement does not overlap the interrupted spawned-agent run', async () => {
  const firstGate = Promise.withResolvers<void>()
  const inputs: string[] = []
  let active = 0
  let maxActive = 0
  const manager = new SpawnedAgentManager({
    idFactory: () => 'worker',
    runner: async request => {
      inputs.push(request.input)
      active += 1
      maxActive = Math.max(maxActive, active)
      try {
        if (inputs.length === 1) await firstGate.promise
        return request.input
      } finally {
        active -= 1
      }
    },
  })

  const task = await manager.spawn({ message: 'original' })
  await Bun.sleep(1)
  const replacing = manager.sendInput(task.id, { message: 'replacement', interrupt: true })
  await Bun.sleep(20)
  expect(inputs).toEqual(['original'])
  firstGate.resolve()
  await replacing
  await manager.wait([task.id], 1_000)
  expect(inputs).toEqual(['original', 'replacement'])
  expect(maxActive).toBe(1)
})

test('reset waits for an interrupted run before starting replacement work', async () => {
  const firstGate = Promise.withResolvers<void>()
  const runs: string[] = []
  let active = 0
  let maxActive = 0
  const manager = new SubAgentManager({
    runner: async request => {
      runs.push(request.prompt)
      active += 1
      maxActive = Math.max(maxActive, active)
      try {
        if (runs.length === 1) await firstGate.promise
        return request.prompt
      } finally {
        active -= 1
      }
    },
  })
  try {
    const original = await manager.spawn({ prompt: 'original', name: 'worker' })
    await manager.waitFor(() => original.status === 'running', { timeoutMs: 1_000 })
    const resetting = manager.reset(original.id, 'replacement')
    const duplicate = manager.reset(original.id, 'ignored duplicate')
    await Bun.sleep(20)
    expect(runs).toEqual(['original'])
    firstGate.resolve()
    const [replacement, duplicateReplacement] = await Promise.all([resetting, duplicate])
    expect(duplicateReplacement).toBe(replacement)
    expect(replacement).toBeDefined()
    if (replacement) await manager.wait(replacement.id, 1_000)
    expect(runs).toEqual(['original', 'replacement'])
    expect(maxActive).toBe(1)
  } finally {
    firstGate.resolve()
    await manager.close()
  }
})

test('retry recreates required worktree isolation instead of downgrading to shared workspace', async () => {
  const created: string[] = []
  const removed: string[] = []
  const worktree: SubagentWorktreePort = {
    create: request => {
      const path = `/worktrees/${request.taskId}-${created.length + 1}`
      created.push(path)
      return Promise.resolve({ branch: `agent/${created.length}`, path })
    },
    isClean: () => Promise.resolve(true),
    remove: tree => { removed.push(tree.path); return Promise.resolve() },
  }
  let runs = 0
  const seenWorktrees: string[] = []
  const manager = new SubAgentManager({
    worktree,
    runner: request => {
      runs += 1
      if (request.worktree) seenWorktrees.push(request.worktree.path)
      if (runs === 1) throw new Error('first failed')
      return 'recovered'
    },
  })
  try {
    const task = await manager.spawn({ prompt: 'isolated', name: 'worker', isolation: 'worktree' })
    await manager.wait(task.id, 1_000)
    await manager.waitFor(() => removed.length === 1, { timeoutMs: 1_000 })
    const retried = await manager.retry(task.id)
    expect(retried).toBeDefined()
    if (retried) await manager.wait(retried.id, 1_000)
    expect(created).toHaveLength(2)
    expect(seenWorktrees).toEqual(created)
    expect(retried?.worktreePath).toBe(created[1])
  } finally {
    await manager.close()
  }
})

test('definition tool policy intersects with caller restrictions instead of replacing them', async () => {
  const configs: Array<Readonly<Record<string, unknown>>> = []
  const definition: AgentDefinition = {
    allowedTools: ['ReadFile', 'WriteFile'],
    description: '',
    excludeTools: ['exec_command'],
    isolation: '',
    maxDepth: 5,
    model: '',
    name: 'writer',
    source: 'test',
    systemPrompt: '',
    tools: ['ReadFile', 'WriteFile'],
  }
  const manager = new SubAgentManager({
    runner: request => {
      configs.push(request.config)
      return { content: 'ok' }
    },
  })

  // A read-only parent spawning a writable definition child keeps its own restriction.
  const restricted = await manager.spawn({
    prompt: 'review only',
    agentDefinition: definition,
    config: { _toolsWhitelist: ['ReadFile'], _toolsAllowed: ['ReadFile'], _toolsExcluded: ['GrepTool'] },
  })
  await manager.wait(restricted.id, 1_000)
  expect(restricted.status).toBe('completed')
  expect(configs[0]).toMatchObject({
    _toolsWhitelist: ['ReadFile'],
    _toolsAllowed: ['ReadFile'],
  })
  expect(configs[0]?._toolsExcluded).toEqual(expect.arrayContaining(['GrepTool', 'exec_command']))

  // Disjoint caller/definition allow-lists fail closed rather than falling back to "unrestricted".
  const disjoint = await manager.spawn({
    prompt: 'no overlap',
    agentDefinition: definition,
    config: { _toolsAllowed: ['GrepTool'] },
  })
  await manager.wait(disjoint.id, 1_000)
  const disjointAllowed = configs[1]?._toolsAllowed
  expect(Array.isArray(disjointAllowed)).toBeTrue()
  expect(disjointAllowed).not.toEqual(expect.arrayContaining(['GrepTool', 'ReadFile', 'WriteFile']))

  const filtered = filterSubagentTools<{ name: string }>({
    config: configs[1] ?? {},
    isSubagent: true,
    toolSchemas: [{ name: 'ReadFile' }, { name: 'GrepTool' }],
    toolExecutor: () => 'ok',
  })
  expect(filtered.toolSchemas).toEqual([])
  expect(await filtered.execute?.('ReadFile', {})).toBe("Error: tool 'ReadFile' is not allowed for this agent.")
  await manager.close()
})

test('spawn depth derives from the tracked parent task and the spawned-agent budget is enforced', async () => {
  const depths: number[] = []
  const manager = new SubAgentManager({
    maxDepth: 2,
    runner: request => {
      depths.push(request.depth)
      return { content: 'ok' }
    },
  })

  const parent = await manager.spawn({ prompt: 'root', name: 'root' })
  expect(parent.depth).toBe(0)
  // No depth supplied: the manager derives it from the tracked parent instead of resetting to 0.
  const child = await manager.spawn({ prompt: 'child', parentId: parent.id })
  expect(child.depth).toBe(1)
  const grandchild = await manager.spawn({ prompt: 'grandchild', parentId: child.id })
  expect(grandchild.depth).toBe(2)
  expect(grandchild.status).toBe('failed')
  expect(grandchild.error).toContain('Max depth (2')
  await manager.wait(child.id, 1_000)
  expect(depths).toEqual([1, 2])
  await manager.close()

  const gate = deferred()
  const bounded = new SubAgentManager({
    maxSpawnedAgents: 2,
    runner: async () => {
      await gate.promise
      return { content: 'ok' }
    },
  })
  const first = await bounded.spawn({ prompt: 'one' })
  const second = await bounded.spawn({ prompt: 'two' })
  expect(bounded.listTasks()).toHaveLength(2)
  await expect(bounded.spawn({ prompt: 'three' })).rejects.toThrow('Spawned-agent budget (2) reached')
  gate.resolve()
  await bounded.wait(first.id, 1_000)
  await bounded.wait(second.id, 1_000)
  await bounded.close()
})

test('terminal history does not permanently exhaust the live spawned-agent budget', async () => {
  const manager = new SubAgentManager({
    maxRetainedTerminalTasks: 10,
    maxSpawnedAgents: 2,
    runner: () => ({ content: 'ok' }),
  })

  const first = await manager.spawn({ prompt: 'first generation' })
  const second = await manager.spawn({ prompt: 'second generation' })
  await manager.waitAll([first.id, second.id], 1_000)
  expect(manager.listTasks()).toHaveLength(2)

  // Retained terminal history is inspectable, but only pending/running tasks
  // consume the live spawn budget.
  const third = await manager.spawn({ prompt: 'third generation' })
  const fourth = await manager.spawn({ prompt: 'fourth generation' })
  await manager.waitAll([third.id, fourth.id], 1_000)
  expect(third.status).toBe('completed')
  expect(fourth.status).toBe('completed')
  expect(manager.listTasks()).toHaveLength(4)
  await manager.close()
})

test('terminal tasks are evicted beyond the retention bound while live tasks survive', async () => {
  const manager = new SubAgentManager({
    maxRetainedTerminalTasks: 2,
    maxSpawnedAgents: 10,
    runner: () => ({ content: 'ok' }),
  })
  const tasks = []
  for (let index = 0; index < 5; index += 1) {
    tasks.push(await manager.spawn({ prompt: `task ${index}` }))
  }
  await manager.waitAll()
  for (const task of tasks) expect(task.status).toBe('completed')
  await manager.waitFor(() => manager.listTasks().length <= 2, { timeoutMs: 1_000 })
  expect(manager.listTasks().length).toBeLessThanOrEqual(2)
  expect(manager.getResult(tasks[0]!.id)).toBeUndefined()
  expect(manager.getResult(tasks[4]!.id)).toBe('ok')
  await manager.close()
})

test('cancel waits for the in-flight run before done events and worktree cleanup', async () => {
  const gate = deferred()
  const removed: string[] = []
  const worktree: SubagentWorktreePort = {
    create: request => Promise.resolve({ branch: `agent/${request.taskId}`, path: `/worktrees/${request.taskId}` }),
    isClean: () => Promise.resolve(true),
    remove: tree => {
      removed.push(tree.path)
      return Promise.resolve()
    },
  }
  const manager = new SubAgentManager({
    idFactory: () => 'cancel-race-task',
    worktree,
    runner: async request => {
      await gate.promise
      // Late reports from a cooperatively aborted runner must precede `done`.
      request.report.text('late output')
      return { content: 'finished after cancel' }
    },
  })

  const task = await manager.spawn({ prompt: 'isolated work', isolation: 'worktree' })
  await manager.waitFor(() => task.status === 'running', { timeoutMs: 1_000 })
  expect(manager.cancel(task.id)).toBeTrue()
  await Bun.sleep(25)
  expect(removed).toEqual([])
  expect(manager.peekMailbox().some(event => event.taskId === task.id && event.type === 'done')).toBeFalse()

  gate.resolve()
  await manager.waitFor(
    () => manager.peekMailbox().some(event => event.taskId === task.id && event.type === 'done'),
    { timeoutMs: 1_000 },
  )
  expect(removed).toEqual([`/worktrees/${task.id}`])
  const types = manager.peekMailbox().filter(event => event.taskId === task.id).map(event => event.type)
  expect(types.indexOf('done')).toBeGreaterThan(types.lastIndexOf('text_burst'))
  expect(types.indexOf('done')).toBeGreaterThan(types.indexOf('cancelled'))
  await manager.close()
})

test('filterSubagentTools fails closed when tool schemas are unavailable', async () => {
  const calls: string[] = []
  const filtered = filterSubagentTools({
    isSubagent: true,
    config: { _toolsAllowed: ['ReadFile'], _toolsExcluded: ['WriteFile'] },
    toolExecutor: toolName => {
      calls.push(toolName)
      return 'ok'
    },
  })

  expect(filtered.toolSchemas).toEqual([])
  expect(await filtered.execute?.('SpawnAgents', {})).toBe("Error: tool 'SpawnAgents' is not allowed for this agent.")
  expect(await filtered.execute?.('WriteFile', {})).toBe("Error: tool 'WriteFile' is not allowed for this agent.")
  expect(await filtered.execute?.('GrepTool', {})).toBe("Error: tool 'GrepTool' is not allowed for this agent.")
  expect(await filtered.execute?.('ReadFile', {})).toBe('ok')
  expect(calls).toEqual(['ReadFile'])

  const delegationAllowed = filterSubagentTools({
    isSubagent: true,
    config: { _allowSubagentDelegation: true },
    toolExecutor: () => 'ok',
  })
  expect(await delegationAllowed.execute?.('SpawnAgents', {})).toBe('ok')
})

test('referenced-only profiles bind through catalog keys and never claim the plain alias', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-hardening-catalog-'))
  const nested = join(root, 'nested')
  try {
    await mkdir(nested, { recursive: true })
    await writeFile(join(nested, 'alpha.yaml'), `version: 1
agent:
  name: alpha-worker
  system_prompt: alpha profile
`, 'utf8')
    await writeFile(join(nested, 'beta.yaml'), `version: 1
agent:
  name: beta-worker
  system_prompt: beta profile
`, 'utf8')
    await writeFile(join(root, 'agents.yaml'), `version: 1
agents:
  first:
    system_prompt: first creator
    subagents:
      helper:
        path: ./nested/alpha.yaml
        description: alpha helper
  second:
    system_prompt: second creator
    subagents:
      helper:
        path: ./nested/beta.yaml
        description: beta helper
`, 'utf8')

    const definitions = loadAgentDefinitions({
      builtinDefinitions: new Map(),
      cwd: root,
      userDirectory: join(root, 'user'),
      projectDirectory: join(root, '.xerxes', 'agents'),
    })

    expect(definitions.get('helper')).toBeUndefined()
    const firstKey = definitions.get('first')?.subagents?.helper?.resolvedProfile ?? ''
    const secondKey = definitions.get('second')?.subagents?.helper?.resolvedProfile ?? ''
    expect(firstKey).toStartWith('@catalog:helper:')
    expect(secondKey).toStartWith('@catalog:helper:')
    expect(firstKey).not.toBe(secondKey)
    expect(definitions.get(firstKey)?.systemPrompt).toBe('alpha profile')
    expect(definitions.get(secondKey)?.systemPrompt).toBe('beta profile')

    // Loading again resolves the same catalog keys deterministically.
    const reloaded = loadAgentDefinitions({
      builtinDefinitions: new Map(),
      cwd: root,
      userDirectory: join(root, 'user'),
      projectDirectory: join(root, '.xerxes', 'agents'),
    })
    expect(reloaded.get('first')?.subagents?.helper?.resolvedProfile).toBe(firstKey)
    expect(reloaded.get('second')?.subagents?.helper?.resolvedProfile).toBe(secondKey)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('orchestrator clones registered agents, keeps auto-ids collision-free, and freezes history', () => {
  const orchestrator = new AgentOrchestrator({ now: () => new Date('2026-07-21T00:00:00.000Z') })
  orchestrator.registerAgent({ id: 'agent_1' })
  const anonymous: { id?: string } = {}
  const firstAutoId = orchestrator.registerAgent(anonymous)
  const secondAutoId = orchestrator.registerAgent({})
  expect(anonymous.id).toBeUndefined()
  expect(firstAutoId).toBe('agent_0')
  expect(secondAutoId).toBe('agent_2')
  expect(new Set([...orchestrator.agents.keys()])).toEqual(new Set(['agent_1', 'agent_0', 'agent_2']))

  orchestrator.switchAgent('agent_0', 'handoff')
  const history = orchestrator.executionHistory
  expect(Object.isFrozen(history)).toBeTrue()
  expect(Object.isFrozen(history[0])).toBeTrue()
  expect(history).toHaveLength(1)
  orchestrator.switchAgent('agent_1', 'second handoff')
  expect(history).toHaveLength(1)
  expect(orchestrator.executionHistory).toHaveLength(2)
})
