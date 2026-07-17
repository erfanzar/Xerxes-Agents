// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  AgentSelfMemory,
  clearAgentSelfMemoryCache,
  getAgentSelfMemory,
  listAgentSelfMemories,
} from '../src/memory/agentSelfMemory.js'
import { registerAgentMemoryTools } from '../src/tools/agentMemoryTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

async function inTemporaryDirectory(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-self-memory-'))
  try {
    await run(directory)
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
}

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

test('agent self-memory keeps isolated defaults, learning records, and prompt context', async () => {
  await inTemporaryDirectory(async directory => {
    const memory = new AgentSelfMemory({
      agentId: 'planner',
      directory: join(directory, 'memories', 'planner'),
      projectRoot: directory,
    })
    await memory.ensure()

    expect(await memory.read('user_taste')).toContain('# User Taste Profile')
    expect(await memory.learn('User prefers terse status updates', 'user_taste')).toBe(
      'User taste updated: User prefers terse status updates',
    )
    expect(await memory.learn('ReadFile followed by FileEditTool works well', 'tool_pattern')).toBe(
      'Tool pattern recorded: ReadFile followed by FileEditTool works well',
    )
    expect(await memory.learn('Ship checklist. Always verify a clean build', 'skill_proposal')).toBe(
      'Skill proposed: Ship checklist',
    )
    await memory.markSkillImplemented('Ship checklist')

    expect(await memory.read('user_taste')).toContain('User prefers terse status updates')
    expect(await memory.read('tool_usage_patterns')).toContain('ReadFile followed by FileEditTool works well')
    expect(await memory.read('skill_journal')).toContain('Status: implemented')
    expect(await memory.systemPromptAddendum()).toContain('[User Taste Profile]')
  })
})

test('agent self-memory syncs project instructions from the supplied project root', async () => {
  await inTemporaryDirectory(async directory => {
    const project = join(directory, 'project')
    await mkdir(project, { recursive: true })
    await Bun.write(join(project, 'AGENTS.md'), 'Use Bun for every command.')
    await Bun.write(join(project, 'SOUL.md'), 'Be candid about incomplete work.')
    const memory = new AgentSelfMemory({
      agentId: 'reviewer',
      directory: join(directory, 'memories', 'reviewer'),
      projectRoot: project,
    })

    await memory.syncProjectContext()
    const context = await memory.read('project_context')
    expect(context).toContain('## AGENTS.md')
    expect(context).toContain('Use Bun for every command.')
    expect(context).toContain('## SOUL.md')
    expect(context).toContain('Be candid about incomplete work.')
    expect(await listAgentSelfMemories(join(directory, 'memories'))).toEqual(['reviewer'])
  })
})

test('agent-memory learn and sync tools use the injected self-memory without requiring scoped memory', async () => {
  await inTemporaryDirectory(async directory => {
    const project = join(directory, 'project')
    await mkdir(project, { recursive: true })
    await Bun.write(join(project, 'XERXES.md'), 'Runtime: Bun.')
    const selfMemory = new AgentSelfMemory({
      agentId: 'operator',
      directory: join(directory, 'memories', 'operator'),
      projectRoot: project,
    })
    const registry = new ToolRegistry()
    registerAgentMemoryTools(registry, { selfMemory })
    const context = { agentId: 'operator', metadata: { project_root: project } }

    expect(await registry.execute(call('agent_memory_learn', {
      observation: 'Use the configured workspace resolver',
      category: 'self_reflection',
      importance: 'high',
    }), context)).toBe('Self-reflection recorded: Use the configured workspace resolver')
    expect(await registry.execute(call('agent_memory_sync_context', {}), context)).toBe(
      'Project context synced to agent memory.',
    )
    expect(await selfMemory.read('self_reflection')).toContain('Use the configured workspace resolver')
    expect(await selfMemory.read('project_context')).toContain('Runtime: Bun.')
  })
})

test('agent self-memory serializes concurrent patches and taste updates without lost updates', async () => {
  await inTemporaryDirectory(async directory => {
    const memory = new AgentSelfMemory({
      agentId: 'concurrent',
      directory: join(directory, 'memories', 'concurrent'),
      projectRoot: directory,
    })
    await memory.ensure()

    await Promise.all([
      memory.patch('self_reflection', '## What Worked', '## What Worked\n- first patch'),
      memory.patch('self_reflection', '## What Worked', '## What Worked\n- second patch'),
      memory.updateUserTaste('prefers bun'),
      memory.updateUserTaste('prefers terse output'),
    ])

    const reflection = await memory.read('self_reflection')
    expect(reflection).toContain('- first patch')
    expect(reflection).toContain('- second patch')
    const taste = await memory.read('user_taste')
    expect(taste).toContain('prefers bun')
    expect(taste).toContain('prefers terse output')
  })
})

test('process-wide self-memory cache stays bounded like a simple LRU', () => {
  clearAgentSelfMemoryCache()
  const evicted = getAgentSelfMemory('cache-agent-0')
  for (let index = 1; index < 300; index += 1) getAgentSelfMemory(`cache-agent-${index}`)
  const retained = getAgentSelfMemory('cache-agent-299')

  expect(getAgentSelfMemory('cache-agent-299')).toBe(retained)
  // The oldest entry was evicted past the 256-entry bound and is re-created.
  expect(getAgentSelfMemory('cache-agent-0')).not.toBe(evicted)
  clearAgentSelfMemoryCache()
})
