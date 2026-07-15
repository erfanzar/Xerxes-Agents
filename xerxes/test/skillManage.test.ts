// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { AgentAuthoredSkillStore, registerSkillManageTool } from '../src/tools/skillManage.js'

test('agent-authored skill store supports the complete managed Markdown lifecycle', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-authored-skills-'))
  try {
    const store = new AgentAuthoredSkillStore({ authoredDirectory: directory })
    expect(await store.manage('list')).toEqual({ ok: true, intent: 'list', skills: [] })
    expect(await store.manage('create', {
      name: 'deploy',
      description: 'Deploy safely',
      body: '# Deploy\n\nVerify first.',
    })).toMatchObject({ ok: true, intent: 'create', name: 'deploy' })
    expect(await store.manage('create', { name: 'deploy', body: 'duplicate' })).toEqual({
      ok: false,
      intent: 'create',
      name: 'deploy',
      error: 'skill already exists; use intent=edit',
    })
    const viewed = await store.manage('view', { name: 'deploy' })
    expect(viewed).toMatchObject({
      ok: true,
      intent: 'view',
      name: 'deploy',
      body: expect.stringContaining('Verify first.'),
    })
    expect(await store.manage('edit', { name: 'deploy', body: 'Updated body.' })).toMatchObject({ ok: true, intent: 'edit' })
    expect(await store.manage('delete', { name: 'deploy' })).toMatchObject({ ok: true, intent: 'delete' })
    expect(await store.manage('view', { name: 'deploy' })).toEqual({ ok: false, intent: 'view', name: 'deploy', error: 'not found' })
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

test('skill_manage tool rejects traversal and routes tool calls through its configured authored directory', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-authored-skills-'))
  try {
    const tools = new ToolRegistry()
    registerSkillManageTool(tools, { authoredDirectory: directory })
    const invalid = JSON.parse(await tools.execute({
      id: 'bad', type: 'function', function: { name: 'skill_manage', arguments: { intent: 'create', name: '../bad', body: 'x' } },
    }, { metadata: {} }))
    expect(invalid.ok).toBeFalse()
    const created = JSON.parse(await tools.execute({
      id: 'create', type: 'function', function: { name: 'skill_manage', arguments: { intent: 'create', name: 'notes', body: 'keep this' } },
    }, { metadata: {} }))
    expect(created).toMatchObject({ ok: true, name: 'notes' })
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})
