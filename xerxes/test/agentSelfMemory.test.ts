// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'

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

test('agent self-memory serializes same-key mutations across instances sharing a directory', async () => {
  await inTemporaryDirectory(async directory => {
    const shared = join(directory, 'memories', 'shared')
    const first = new AgentSelfMemory({ agentId: 'shared', directory: shared, projectRoot: directory })
    const second = new AgentSelfMemory({ agentId: 'shared', directory: shared, projectRoot: directory })
    await first.ensure()

    await Promise.all(Array.from({ length: 40 }, (_, index) =>
      (index % 2 === 0 ? first : second).append('self_reflection', `- shared-${index}`),
    ))

    const reflection = await first.read('self_reflection')
    for (let index = 0; index < 40; index += 1) expect(reflection).toContain(`- shared-${index}`)
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

test('concurrent processes appending to one self-memory directory keep every entry', async () => {
  await inTemporaryDirectory(async directory => {
    const shared = join(directory, 'memories', 'procs')
    const sourceUrl = pathToFileURL(join(import.meta.dir, '../src/memory/agentSelfMemory.ts')).href
    const workerPath = join(directory, 'append-worker.ts')
    // Separate processes share no promise chain, so only O_APPEND append
    // semantics can keep both writers' entries from clobbering each other.
    await Bun.write(workerPath, `
      const { AgentSelfMemory } = await import(${JSON.stringify(sourceUrl)})
      const [shared, label] = process.argv.slice(2)
      const memory = new AgentSelfMemory({ agentId: 'proc-' + label, directory: shared, projectRoot: shared })
      for (let index = 0; index < 20; index += 1) {
        await memory.append('tool_usage_patterns', '- ' + label + '-' + index)
      }
    `)
    const workers = ['alpha', 'beta'].map(label => Bun.spawn({
      cmd: [process.execPath, workerPath, shared, label],
      stdout: 'pipe',
      stderr: 'pipe',
    }))
    const results = await Promise.all(workers.map(async worker => ({
      exitCode: await worker.exited,
      stderr: await new Response(worker.stderr).text(),
    })))
    expect(results).toEqual([{ exitCode: 0, stderr: '' }, { exitCode: 0, stderr: '' }])

    const reader = new AgentSelfMemory({ agentId: 'reader', directory: shared, projectRoot: directory })
    const patterns = await reader.read('tool_usage_patterns')
    for (const label of ['alpha', 'beta']) {
      for (let index = 0; index < 20; index += 1) expect(patterns).toContain(`- ${label}-${index}`)
    }
  })
})

test('read-modify-write taste updates wait behind an OS-level lock holder', async () => {
  await inTemporaryDirectory(async directory => {
    const memory = new AgentSelfMemory({
      agentId: 'rmw',
      directory: join(directory, 'memories', 'rmw'),
      projectRoot: directory,
    })
    await memory.ensure()

    // Emulate another process mid-rewrite by holding the per-file mutation
    // lock with a live owner record.
    const lockPath = join(directory, 'memories', 'rmw', 'user_taste.md.lock')
    await Bun.write(lockPath, `${JSON.stringify({ pid: process.pid, token: 'held-elsewhere', createdAt: Date.now() })}\n`)

    let settled = false
    const pending = memory.updateUserTaste('late preference').then(() => {
      settled = true
    })
    await Bun.sleep(150)
    expect(settled).toBeFalse()

    await rm(lockPath, { force: true })
    await pending
    expect(await memory.read('user_taste')).toContain('- late preference')
  })
})

test('concurrent cross-process appends and rewrites keep every appended entry', async () => {
  await inTemporaryDirectory(async directory => {
    const shared = join(directory, 'memories', 'race')
    const sourceUrl = pathToFileURL(join(import.meta.dir, '../src/memory/agentSelfMemory.ts')).href
    const workerPath = join(directory, 'append-vs-rmw.ts')
    // One process plays the daemon auto-appending reflections; the other
    // plays model-driven self-memory writes whose read-modify-write cycles
    // publish via atomic rename. Before appends shared the per-file OS lock,
    // an append landing between a rewrite's read and rename was erased.
    await Bun.write(workerPath, `
      const { AgentSelfMemory } = await import(${JSON.stringify(sourceUrl)})
      const [shared, mode] = process.argv.slice(2)
      const memory = new AgentSelfMemory({ agentId: 'race-' + mode, directory: shared, projectRoot: shared })
      if (mode === 'append') {
        for (let index = 0; index < 40; index += 1) {
          await memory.append('self_reflection', '- raced-' + index)
          await Bun.sleep(8)
        }
      } else {
        const deadline = Date.now() + 4000
        let round = 0
        while (Date.now() < deadline) {
          await memory.updateUserTaste('preference ' + round, round % 2 === 0 ? 'notes' : 'communication style')
          await memory.patch('user_taste', '- preference ' + round, '- preference ' + round + ' confirmed')
          round += 1
        }
      }
    `)
    const workers = ['append', 'rewrite'].map(mode => Bun.spawn({
      cmd: [process.execPath, workerPath, shared, mode],
      stdout: 'pipe',
      stderr: 'pipe',
    }))
    const results = await Promise.all(workers.map(async worker => ({
      exitCode: await worker.exited,
      stderr: await new Response(worker.stderr).text(),
    })))
    expect(results).toEqual([{ exitCode: 0, stderr: '' }, { exitCode: 0, stderr: '' }])

    const reader = new AgentSelfMemory({ agentId: 'reader', directory: shared, projectRoot: directory })
    const reflection = await reader.read('self_reflection')
    for (let index = 0; index < 40; index += 1) expect(reflection).toContain('- raced-' + index)
  })
})

test('ensure seeds templates in one exclusive step and heals crashed creators', async () => {
  await inTemporaryDirectory(async directory => {
    const memory = new AgentSelfMemory({
      agentId: 'templates',
      directory: join(directory, 'memories', 'templates'),
      projectRoot: directory,
    })

    // A foreign-created file is respected: no template is ever laid over it.
    const foreign = join(memory.directory, 'project_context.md')
    await mkdir(memory.directory, { recursive: true })
    await Bun.write(foreign, 'CUSTOM CONTEXT')
    await memory.ensure()
    expect(await Bun.file(foreign).text()).toBe('CUSTOM CONTEXT')

    // An empty leftover from a creator that crashed between exclusive
    // creation and its template write heals on the next pass.
    const reflectionPath = join(memory.directory, 'self_reflection.md')
    await Bun.write(reflectionPath, '')
    await memory.ensure()
    expect(await Bun.file(reflectionPath).text()).toContain('# Self Reflection')

    // Appends racing creation converge with the template ABOVE the entries:
    // the winner lays down the complete template before any append lands.
    await rm(reflectionPath, { force: true })
    await Promise.all([memory.ensure(), memory.append('self_reflection', '- raced entry')])
    const healed = await memory.read('self_reflection')
    expect(healed.indexOf('# Self Reflection')).toBeLessThan(healed.indexOf('- raced entry'))

    // No temporary template files survive a normal pass.
    const leftovers = (await readdir(memory.directory)).filter(name => name.includes('.tmp'))
    expect(leftovers).toEqual([])
  })
})
