// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { join } from 'node:path'
import { HubEventStore } from '../src/hub/eventStore.js'
import { createHubSupervisor } from '../src/hub/supervisor/index.js'
import type { SupervisorModelRequest } from '../src/hub/supervisor/runner.js'
import { SupervisorThreads } from '../src/hub/supervisor/threads.js'
import { disposeAll, registerCleanup, tempDir } from './hubHarness.js'
import { makeWorld, toolCall } from './hubSupervisorHarness.js'

afterEach(disposeAll)
const target = { workspaceId: 'w1', sessionId: 's2' }

test('live ownership revocation overrides a captured owner principal', () => {
  const world = makeWorld(); let owner = true
  const threads = new SupervisorThreads({ db: world.db, identity: world.identity, clock: world.clock, isHubOwner: () => owner })
  const principal = world.user('owner'); const thread = threads.privateFor(principal)
  const reader = threads.scopeReader(thread, principal)
  expect(reader.canRead(target)).toBe(true)
  owner = false
  expect(reader.canRead(target)).toBe(false)
})

test('an existing reader immediately observes grant removal', () => {
  const world = makeWorld(); const reader = world.reader('alice')
  expect(reader.canRead(target)).toBe(true)
  world.identity.grantsFor = () => []
  expect(reader.canRead(target)).toBe(false)
})

test('an existing reader evaluates expiry against current time, not creation time', () => {
  const world = makeWorld(); const grants = world.identity.grantsFor('alice')
  world.identity.grantsFor = () => grants.map(grant => ({ ...grant, expiresAt: new Date(world.time.now + 10).toISOString() }))
  const expiring = world.identity.grantsFor('alice')
  world.identity.grantsFor = () => expiring
  const reader = world.reader('alice'); expect(reader.canRead(target)).toBe(true)
  world.time.now += 1000
  expect(reader.canRead(target)).toBe(false)
})

test('new shared participants narrow a reader that was created before they joined', () => {
  const world = makeWorld(); const reader = world.reader('alice')
  expect(reader.canRead(target)).toBe(true)
  world.threads.access('shared', world.user('guest'))
  expect(reader.canRead(target)).toBe(false)
  expect(reader.canRead({ workspaceId: 'w1', sessionId: 's1' })).toBe(true)
})

test('revocation during model completion prevents retained task contents reaching the next request', async () => {
  const world = makeWorld(); const dir = await tempDir('superagent-live-scope-')
  const journal = new HubEventStore(':memory:'); registerCleanup(() => journal.close())
  journal.append({ workspace_id: 'w1', session_id: 's2', type: 'tool_result', payload: { result: 'private-retained-marker' } })
  const requests: SupervisorModelRequest[] = []
  const supervisor = createHubSupervisor({ databasePath: join(dir, 'hub.sqlite'), identity: world.identity, overlay: world.overlay,
    workspaces: world.workspaces, backend: world.backend, journal, clock: world.clock, persona: 'Offline fixture',
    isHubOwner: id => id === 'owner', autopilot: { checkInMs: 0 }, execute: async () => { throw new Error('read must not mutate') },
    model: { id: 'offline-scope', reportsCost: false, async complete(request) {
      requests.push(request)
      if (requests.length > 1) return { text: 'finished', toolCalls: [] }
      world.identity.grantsFor = () => []
      return { text: '', toolCalls: [toolCall('task_records', { workspace_id: 'w1', task_id: 's2' })] }
    } } })
  registerCleanup(() => supervisor.close())
  const alice = world.user('alice'); const thread = supervisor.threads.privateFor(alice)
  await supervisor.runner.enqueue({ threadId: thread.id, requester: alice })
  expect(JSON.stringify(requests)).not.toContain('private-retained-marker')
  expect(world.backendCalls).toHaveLength(0)
})
