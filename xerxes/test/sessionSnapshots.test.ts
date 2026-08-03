// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { SnapshotManager } from '../src/session/snapshots.js'

function workspaceFixture(prefix: string): { directory: string; shadow: string; workspace: string } {
  const directory = mkdtempSync(join(tmpdir(), prefix))
  const workspace = join(directory, 'workspace')
  mkdirSync(workspace)
  return { directory, shadow: join(directory, 'shadow'), workspace }
}

test('snapshot records carry the session and turn they precede', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-link-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'turn zero', 'utf8')
    await snapshots.snapshot('turn-0', { sessionId: 'a1b2c3d4', turnIndex: 0 })
    writeFileSync(join(workspace, 'a.txt'), 'turn one', 'utf8')
    const second = await snapshots.snapshot('turn-1', { sessionId: 'a1b2c3d4', turnIndex: 1 })
    await snapshots.snapshot('turn-0', { sessionId: 'ffffffff', turnIndex: 0 })
    const manual = await snapshots.snapshot('manual')

    expect(manual.sessionId).toBeUndefined()
    expect(manual.turnIndex).toBeUndefined()
    expect(snapshots.listForSession('a1b2c3d4').map(record => record.turnIndex)).toEqual([0, 1])
    expect(snapshots.getForTurn('a1b2c3d4', 1)?.id).toBe(second.id)
    expect(snapshots.getForTurn('a1b2c3d4', 9)).toBeUndefined()
    expect(snapshots.getForTurn('', 0)).toBeUndefined()

    // "Take me back to before turn 1" restores the tree as the user saw it then.
    writeFileSync(join(workspace, 'a.txt'), 'agent damage', 'utf8')
    await snapshots.rollback(second.id)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('turn one')
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('concurrent managers serialize snapshots through one shadow repository', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-concurrent-')
  try {
    writeFileSync(join(workspace, 'a.txt'), 'shared content', 'utf8')
    const managers = Array.from({ length: 8 }, () => new SnapshotManager(workspace, { shadowRoot: shadow }))

    const records = await Promise.all(managers.map((manager, index) => manager.snapshot(`parallel-${index}`)))

    expect(new SnapshotManager(workspace, { shadowRoot: shadow }).list()).toHaveLength(records.length)
    expect(new Set(records.map(record => record.id)).size).toBe(records.length)
    expect(new Set(records.map(record => record.commitSha)).size).toBe(records.length)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('reset waits behind an in-flight repository operation', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-reset-race-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'content', 'utf8')
    const snapshotting = snapshots.snapshot('in-flight')
    const resetting = snapshots.reset()

    await Promise.all([snapshotting, resetting])

    expect(snapshots.list()).toEqual([])
    expect(existsSync(snapshots.shadowDirectory)).toBeFalse()
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('a retried turn resolves to its newest capture', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-retry-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'first attempt', 'utf8')
    await snapshots.snapshot('turn-3', { sessionId: 'a1b2c3d4', turnIndex: 3 })
    writeFileSync(join(workspace, 'a.txt'), 'second attempt', 'utf8')
    const retry = await snapshots.snapshot('turn-3', { sessionId: 'a1b2c3d4', turnIndex: 3 })

    expect(snapshots.getForTurn('a1b2c3d4', 3)?.id).toBe(retry.id)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('records written before the session link stay readable and roll back', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-legacy-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'legacy content', 'utf8')
    const record = await snapshots.snapshot('legacy')

    // Rewrite the log in the five-field shape earlier versions produced.
    const recordsPath = join(snapshots.shadowDirectory, '_records.txt')
    writeFileSync(
      recordsPath,
      `${[record.id, record.label, record.commitSha, record.createdAt, record.workspaceDir].join('\t')}\n`,
      'utf8',
    )

    const [restored] = snapshots.list()
    expect(restored?.id).toBe(record.id)
    expect(restored?.sessionId).toBeUndefined()
    expect(restored?.turnIndex).toBeUndefined()

    writeFileSync(join(workspace, 'a.txt'), 'changed', 'utf8')
    await snapshots.rollback(record.id)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('legacy content')
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('pruning keeps the session link on the rewritten records', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-prune-link-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'one', 'utf8')
    await snapshots.snapshot('turn-0', { sessionId: 'a1b2c3d4', turnIndex: 0 })
    writeFileSync(join(workspace, 'a.txt'), 'two', 'utf8')
    const second = await snapshots.snapshot('turn-1', { sessionId: 'a1b2c3d4', turnIndex: 1 })

    expect(await snapshots.prune({ keep: 1 })).toBe(1)
    const retained = snapshots.list()
    expect(retained).toHaveLength(1)
    expect(retained[0]).toMatchObject({ id: second.id, sessionId: 'a1b2c3d4', turnIndex: 1 })
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('single-file restore leaves unrelated edits alone and can be undone', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-restore-')
  try {
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    mkdirSync(join(workspace, 'src'))
    writeFileSync(join(workspace, 'src', 'damaged.txt'), 'good version', 'utf8')
    writeFileSync(join(workspace, 'keep.txt'), 'original', 'utf8')
    const snapshot = await snapshots.snapshot('base', { sessionId: 'a1b2c3d4', turnIndex: 0 })

    writeFileSync(join(workspace, 'src', 'damaged.txt'), 'agent damage', 'utf8')
    writeFileSync(join(workspace, 'keep.txt'), 'deliberate later edit', 'utf8')
    writeFileSync(join(workspace, 'added.txt'), 'new work', 'utf8')

    const restored = await snapshots.restoreFile(snapshot.id, 'src/damaged.txt')

    expect(restored.path).toBe('src/damaged.txt')
    expect(readFileSync(join(workspace, 'src', 'damaged.txt'), 'utf8')).toBe('good version')
    // A single-file restore is not a rollback: everything else survives.
    expect(readFileSync(join(workspace, 'keep.txt'), 'utf8')).toBe('deliberate later edit')
    expect(existsSync(join(workspace, 'added.txt'))).toBe(true)

    // The pre-restore capture makes a mistaken restore reversible.
    await snapshots.rollback(restored.previous.id)
    expect(readFileSync(join(workspace, 'src', 'damaged.txt'), 'utf8')).toBe('agent damage')
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('single-file restore refuses untracked files and paths outside the workspace', async () => {
  if (!Bun.which('git')) return
  const { directory, shadow, workspace } = workspaceFixture('xerxes-snapshot-restore-guard-')
  try {
    writeFileSync(join(directory, 'outside.txt'), 'do not touch', 'utf8')
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'content', 'utf8')
    const snapshot = await snapshots.snapshot('base')

    await expect(snapshots.restoreFile(snapshot.id, '../outside.txt')).rejects.toThrow(
      /escapes the snapshot workspace/,
    )
    await expect(snapshots.restoreFile(snapshot.id, 'never-existed.txt')).rejects.toThrow(
      /does not track never-existed.txt/,
    )
    await expect(snapshots.restoreFile('missing-ref', 'a.txt')).rejects.toThrow(/snapshot not found/)
    await expect(snapshots.restoreFile(snapshot.id, '   ')).rejects.toThrow(/file path is required/)
    // A refused restore never takes a pre-restore capture either.
    expect(snapshots.list()).toHaveLength(1)
    expect(readFileSync(join(directory, 'outside.txt'), 'utf8')).toBe('do not touch')
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})
