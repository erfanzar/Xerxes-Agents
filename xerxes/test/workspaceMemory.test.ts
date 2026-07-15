// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerWorkspaceMemoryTools, WorkspaceMemoryStore } from '../src/tools/workspaceMemory.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

function call(name: string, arguments_: JsonObject): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name, arguments: arguments_ } }
}

test('workspace line memory preserves Python-compatible CRUD results and serializes concurrent additions', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-workspace-memory-'))
  try {
    const store = new WorkspaceMemoryStore({ workspaceRoot: root })
    const added = await Promise.all(Array.from({ length: 10 }, (_, index) => store.add('memory', 'entry ' + index)))
    expect(added.every(result => result.ok)).toBeTrue()
    const listed = await store.list('memory')
    expect(listed.items).toHaveLength(10)
    expect(await store.replace('memory', 2, 'updated')).toEqual({ ok: true, id: 2, content: 'updated' })
    expect(await store.remove('memory', 2)).toEqual({ ok: true, id: 2, content: 'updated' })
    expect(await store.list('memory', 3)).toMatchObject({ ok: true, items: [{ id: 7 }, { id: 8 }, { id: 9 }] })
    expect(await store.add('user', 'terse answers')).toEqual({ ok: true, id: 1, content: 'terse answers' })
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('workspace line-memory tools return structured failures and preserve workspace containment', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-workspace-memory-'))
  const outside = await mkdtemp(join(tmpdir(), 'xerxes-workspace-memory-outside-'))
  try {
    const registry = new ToolRegistry()
    registerWorkspaceMemoryTools(registry, { workspaceRoot: root })
    const context = { metadata: {} }
    expect(JSON.parse(await registry.execute(call('memory_add', { content: 'fact' }), context))).toEqual({
      ok: true,
      id: 1,
      content: 'fact',
    })
    expect(JSON.parse(await registry.execute(call('memory_remove', { entry_id: 99 }), context))).toEqual({
      ok: false,
      error: 'id 99 not found',
    })
    await rm(join(root, 'MEMORY.md'))
    await symlink(outside, join(root, 'MEMORY.md'))
    const blocked = new WorkspaceMemoryStore({ workspaceRoot: root })
    await expect(blocked.list('memory')).rejects.toThrow('outside workspace root')
  } finally {
    await rm(root, { force: true, recursive: true })
    await rm(outside, { force: true, recursive: true })
  }
})
