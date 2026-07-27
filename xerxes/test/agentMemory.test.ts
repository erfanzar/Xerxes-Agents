// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import {
  AgentMemory,
  MEMORY_STALENESS_RULE,
  journalDayLabel,
  projectMemoryDirectoryFor,
} from '../src/memory/agentMemory.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerAgentMemoryTools } from '../src/tools/agentMemoryTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

async function withDirectories(run: (globalDirectory: string, projectDirectory: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-memory-'))
  try {
    await run(join(root, 'global'), join(root, 'project'))
  } finally {
    await rm(root, { force: true, recursive: true })
  }
}

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

test('agent memory persists scoped writes, concurrent appends, search, journals, and prompt context', async () => {
  await withDirectories(async (globalDirectory, projectDirectory) => {
    const memory = new AgentMemory({ globalDirectory, projectDirectory, projectRoot: projectDirectory })
    await memory.ensure()
    await memory.write('project', 'MEMORY.md', 'Bun is the runtime.')
    await Promise.all(
      Array.from({ length: 12 }, (_, index) => memory.append('project', 'EXPERIENCES.md', 'entry ' + index, {
        timestamp: false,
      })),
    )
    await memory.journal('project', 'checkpoint reached', new Date('2026-07-13T10:11:12.000Z'))

    const loaded = await memory.read('project', 'MEMORY.md')
    expect(loaded).toBe('Bun is the runtime.')
    const experience = await memory.read('project', 'EXPERIENCES.md')
    for (let index = 0; index < 12; index += 1) expect(experience).toContain('entry ' + index)

    const files = await memory.listFiles('project')
    expect(files.map(file => file.path)).toEqual(expect.arrayContaining(['MEMORY.md', 'journal/2026-07-13.md']))
    expect(await memory.search('bun')).toEqual([
      expect.objectContaining({ scope: 'project', path: 'MEMORY.md', snippet: expect.stringContaining('Bun') }),
    ])
    const prompt = await memory.toPromptSection({ maxBytesPerFile: 500 })
    expect(prompt).toContain('Bun is the runtime.')
    expect(prompt).toContain('Do not write memory for routine questions, arithmetic, transient test prompts')
    expect(prompt).toContain('Otherwise do not call a memory-writing tool.')
    // Journal bodies are no longer concatenated into every turn: a recent day
    // is listed in the manifest, an old one is outside the window entirely.
    const today = new Date()
    await memory.journal('project', 'today checkpoint', today)
    const withToday = await memory.toPromptSection({ maxBytesPerFile: 500 })
    expect(withToday).not.toContain('today checkpoint')
    expect(withToday).toContain(`  - [project] journal/${today.toISOString().slice(0, 10)}.md:`)
    expect(withToday).not.toContain('journal/2026-07-13.md')
    expect(await memory.status()).toMatchObject({ totalFiles: expect.any(Number), filesByScope: { global: expect.any(Number) } })
  })
})

test('agent memory blocks lexical and existing symlink escapes', async () => {
  await withDirectories(async (globalDirectory, projectDirectory) => {
    const memory = new AgentMemory({ globalDirectory, projectDirectory })
    await memory.ensure()
    const outside = await mkdtemp(join(tmpdir(), 'xerxes-agent-memory-outside-'))
    try {
      await symlink(outside, join(globalDirectory, 'escape'))
      await expect(memory.write('global', '../outside.md', 'no')).rejects.toThrow('escapes')
      await expect(memory.write('global', 'escape/outside.md', 'no')).rejects.toThrow('outside')
    } finally {
      await rm(outside, { force: true, recursive: true })
    }
  })
})

test('agent-memory tools expose Python-compatible success and unavailable result shapes', async () => {
  await withDirectories(async (globalDirectory, projectDirectory) => {
    const registry = new ToolRegistry()
    const context = { metadata: {} }
    registerAgentMemoryTools(registry, {})
    expect(JSON.parse(await registry.execute(call('agent_memory_status', {}), context))).toEqual({ ok: true, available: false })

    const memory = new AgentMemory({ globalDirectory, projectDirectory })
    const configured = new ToolRegistry()
    registerAgentMemoryTools(configured, { memory })
    const write = JSON.parse(await configured.execute(
      call('agent_memory_write', { scope: 'project', path: 'KNOWLEDGE.md', body: 'typed persistence' }),
      context,
    )) as { bytes: number; ok: boolean; path: string; scope: string }
    expect(write).toMatchObject({ ok: true, scope: 'project', path: 'KNOWLEDGE.md', bytes: 17 })

    const read = JSON.parse(await configured.execute(
      call('agent_memory_read', { scope: 'project', path: 'KNOWLEDGE.md' }),
      context,
    )) as { body: string; ok: boolean }
    expect(read).toEqual(expect.objectContaining({ ok: true, body: 'typed persistence' }))
    const escaped = JSON.parse(await configured.execute(
      call('agent_memory_write', { scope: 'global', path: '../escape.md', body: 'bad' }),
      context,
    )) as { error: string; ok: boolean }
    expect(escaped.ok).toBeFalse()
    expect(escaped.error).toContain('escapes')
  })
})

test('project memory location is stable for the same root and salt', () => {
  expect(projectMemoryDirectoryFor('/tmp/example', 'salt-a')).toBe(projectMemoryDirectoryFor('/tmp/example', 'salt-a'))
  expect(projectMemoryDirectoryFor('/tmp/example', 'salt-a')).not.toBe(projectMemoryDirectoryFor('/tmp/example', 'salt-b'))
})

test('only dated journal entries carry a relative-day label, and the prompt warns that memory ages', async () => {
  const now = new Date('2026-07-27T09:00:00.000Z')
  expect(journalDayLabel('journal/2026-07-27.md', now)).toBe('today')
  expect(journalDayLabel('journal/2026-07-26.md', now)).toBe('yesterday')
  expect(journalDayLabel('journal/2026-07-24.md', now)).toBe('3 days ago')
  // A clock disagreement must not render "-1 days ago".
  expect(journalDayLabel('journal/2026-07-28.md', now)).toBeUndefined()
  // Everything else is dated by mtime, which says when the file was edited and
  // not when its claims were true; labelling those is worse than not labelling.
  expect(journalDayLabel('topics/deploy.md', now)).toBeUndefined()
  expect(journalDayLabel('MEMORY.md', now)).toBeUndefined()
  expect(journalDayLabel('journal/notes.md', now)).toBeUndefined()

  await withDirectories(async (globalDirectory, projectDirectory) => {
    const memory = new AgentMemory({ globalDirectory, projectDirectory, projectRoot: projectDirectory })
    const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000)
    await memory.journal('project', 'ran the migration', yesterday)
    await memory.write('project', 'topics/deploy.md', '---\nname: deploy\ndescription: How releases ship\n---\n\nbody')

    const prompt = await memory.toPromptSection({ maxBytesPerFile: 500 })
    expect(prompt).toContain(`  - [project] journal/${yesterday.toISOString().slice(0, 10)}.md:`)
    expect(prompt).toContain('(yesterday)')
    expect(prompt).toContain('  - [project] topics/deploy.md: How releases ship')
    expect(prompt).not.toContain('topics/deploy.md: How releases ship (')
    expect(prompt).toContain(MEMORY_STALENESS_RULE)
    expect(prompt).toContain('confirm it with a read or a search before acting on it')
  })
})
