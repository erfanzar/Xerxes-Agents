// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  DEFAULT_MAX_RETAINED_TERMINAL_HANDLES,
  SpawnedAgentManager,
} from '../src/operators/subagents.js'

test('terminal handles are evicted oldest-first beyond the retention bound', async () => {
  expect(DEFAULT_MAX_RETAINED_TERMINAL_HANDLES).toBe(128)
  let tick = 0
  const releases: Array<() => void> = []
  const manager = new SpawnedAgentManager({
    maxRetainedTerminalHandles: 8,
    idFactory: () => `generated-${tick++}`,
    now: () => new Date(Date.UTC(2026, 6, 13, 10, 0, tick++)),
    // Hold every run open so nothing becomes evictable before the controlled release.
    runner: async request => {
      await new Promise<void>(resolve => {
        releases.push(resolve)
      })
      return `done:${request.input}`
    },
  })

  const ids: string[] = []
  for (let index = 0; index < 10; index += 1) {
    const nickname = `worker-${index}`
    ids.push(nickname)
    await manager.spawn({ nickname, message: `task ${index}` })
  }
  expect(manager.listHandles()).toHaveLength(10)
  for (const release of releases) release()
  await manager.wait(ids, 5_000)

  // The next mutation compacts down to the bound, dropping the oldest settled handles.
  // Compaction runs synchronously inside spawn, so the blocked overflow run does
  // not need to settle first.
  await manager.spawn({ nickname: 'overflow', message: 'overflow' })

  const remainingIds = manager.listHandles().map(snapshot => snapshot.id).sort()
  expect(remainingIds).toHaveLength(8)
  expect(remainingIds).not.toContain('worker-0')
  expect(remainingIds).not.toContain('worker-1')
  expect(remainingIds).not.toContain('worker-2')
  for (const kept of ['worker-3', 'worker-4', 'worker-5', 'worker-6', 'worker-7', 'worker-8', 'worker-9', 'overflow']) {
    expect(remainingIds).toContain(kept)
  }
})

test('compaction never evicts a handle that is still running or queued', async () => {
  let release!: () => void
  const gate = new Promise<void>(resolve => {
    release = resolve
  })
  const manager = new SpawnedAgentManager({
    maxRetainedTerminalHandles: 2,
    runner: async request => {
      if (request.input === 'block') await gate
      return `done:${request.input}`
    },
  })

  await manager.spawn({ nickname: 'busy', message: 'block' })
  for (let index = 0; index < 5; index += 1) {
    await manager.spawn({ nickname: `finished-${index}`, message: `task ${index}` })
    await manager.wait([`finished-${index}`], 5_000)
  }

  const ids = manager.listHandles().map(snapshot => snapshot.id)
  expect(ids).toContain('busy')
  release()
  const settled = await manager.wait(['busy'], 5_000)
  expect(settled.completed[0]?.lastOutput).toBe('done:block')
})

test('an evicted identity can be re-spawned under the same id so retry keeps working', async () => {
  const manager = new SpawnedAgentManager({
    maxRetainedTerminalHandles: 1,
    now: () => new Date(),
    runner: async request => ({ content: `done:${request.input}` }),
  })

  await manager.spawn({ nickname: 'ephemeral', message: 'first' })
  await manager.wait(['ephemeral'], 5_000)
  // A second spawn both settles past the bound and triggers compaction.
  await manager.spawn({ nickname: 'push', message: 'second' })
  await manager.wait(['push'], 5_000)
  expect(manager.listHandles().map(snapshot => snapshot.id)).not.toContain('ephemeral')

  // SubAgentManager.retry() performs exactly this when its handle lookup misses.
  const respawned = await manager.spawn({ nickname: 'ephemeral', message: 'retry input' })
  expect(respawned.id).toBe('ephemeral')
  await manager.wait(['ephemeral'], 5_000)
  expect(manager.listHandles().find(snapshot => snapshot.id === 'ephemeral')?.lastOutput).toBe(
    'done:retry input',
  )
})

test('the retention bound must be a positive safe integer', () => {
  expect(() => new SpawnedAgentManager({ runner: async () => 'x', maxRetainedTerminalHandles: 0 })).toThrow(
    'maxRetainedTerminalHandles',
  )
  expect(() => new SpawnedAgentManager({ runner: async () => 'x', maxRetainedTerminalHandles: -3 })).toThrow(
    'maxRetainedTerminalHandles',
  )
})
