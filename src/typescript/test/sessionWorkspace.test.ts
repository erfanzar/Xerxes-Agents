// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { WorkspaceIdentity, WorkspaceManager } from '../src/session/index.js'

test('workspace identities serialize Python-compatible records without leaking metadata mutations', () => {
  const metadata = { channel: 'telegram', labels: ['ops'], nested: { chat_id: 7 } }
  const identity = new WorkspaceIdentity({
    workspaceId: 'workspace-a',
    name: 'Operations chat',
    rootPath: null,
    createdAt: '2026-07-13T10:00:00.000Z',
    metadata,
  })
  metadata.channel = 'mutated'
  metadata.labels.push('later')
  metadata.nested.chat_id = 8

  const record = identity.toRecord()
  expect(record).toEqual({
    workspace_id: 'workspace-a',
    name: 'Operations chat',
    root_path: null,
    created_at: '2026-07-13T10:00:00.000Z',
    metadata: { channel: 'telegram', labels: ['ops'], nested: { chat_id: 7 } },
  })
  expect(WorkspaceIdentity.fromRecord(record).toRecord()).toEqual(record)
  expect(JSON.parse(JSON.stringify(identity))).toEqual(record)
  expect(Object.isFrozen(identity)).toBeTrue()
  expect(Object.isFrozen(record)).toBeTrue()
  expect(Object.isFrozen(record.metadata)).toBeTrue()
  expect(Object.isFrozen(record.metadata.nested as object)).toBeTrue()
})

test('workspace manager uses injected IDs and clocks while returning defensive immutable snapshots', () => {
  const generatedIds = ['workspace-generated']
  const metadata = { channel: 'telegram', nested: { chat_id: 7 } }
  const manager = new WorkspaceManager({
    idFactory: () => generatedIds.shift() ?? 'unexpected',
    clock: () => new Date('2026-07-13T11:00:00.000Z'),
  })

  const created = manager.createWorkspace({ name: 'Project', rootPath: '/repo', metadata })
  const explicit = manager.createWorkspace({ name: 'Chat', workspaceId: 'chat-a' })
  metadata.channel = 'mutated after registration'
  metadata.nested.chat_id = 8

  expect(created.toRecord()).toEqual({
    workspace_id: 'workspace-generated',
    name: 'Project',
    root_path: '/repo',
    created_at: '2026-07-13T11:00:00.000Z',
    metadata: { channel: 'telegram', nested: { chat_id: 7 } },
  })
  expect(explicit.toRecord()).toMatchObject({
    workspace_id: 'chat-a',
    root_path: null,
    created_at: '2026-07-13T11:00:00.000Z',
  })

  const fetched = manager.getWorkspace('workspace-generated')
  const listed = manager.listWorkspaces()
  expect(fetched).toBeDefined()
  expect(fetched).not.toBe(created)
  expect(listed).toHaveLength(2)
  expect(listed.map(workspace => workspace.workspaceId)).toEqual(['workspace-generated', 'chat-a'])
  expect(listed[0]).not.toBe(created)
  expect(Object.isFrozen(created)).toBeTrue()
  expect(Object.isFrozen(fetched)).toBeTrue()
  expect(Object.isFrozen(listed)).toBeTrue()
  expect(manager.getWorkspace('missing')).toBeUndefined()
})

test('workspace identity rejects non-serializable metadata and invalid manager clocks', () => {
  expect(() => WorkspaceIdentity.fromRecord({
    workspace_id: 'workspace-a', name: 'Project', metadata: { missing: undefined },
  })).toThrow('JSON-serializable')
  const manager = new WorkspaceManager({ clock: () => new Date('invalid') })
  expect(() => manager.createWorkspace({ name: 'Project' })).toThrow('valid Date')
})
