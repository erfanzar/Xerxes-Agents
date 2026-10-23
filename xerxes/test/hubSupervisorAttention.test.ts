// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { emptyProjection, markProjectionStale, replaceSessionSummaries, type WorkspaceProjection } from '../src/hub/projections.js'
import { AttentionQueue, type AttentionItem, type AttentionPush } from '../src/hub/supervisor/attention.js'
import { T0 } from './hubSupervisorHarness.js'

function setup(options: { failPush?: boolean } = {}) {
  const time = { now: T0 }
  let tick = 0
  const state = { projection: emptyProjection('w1') as WorkspaceProjection }
  const set = (rows: readonly Record<string, unknown>[]): void => {
    tick += 1
    state.projection = replaceSessionSummaries(state.projection, rows, new Date(T0 + tick * 1000).toISOString())
  }
  const pushes: { userId: string; message: AttentionPush }[] = []
  const changes: AttentionItem[] = []
  const errors: string[] = []
  const queue = new AttentionQueue({
    workspaces: { list: () => [{ id: 'w1', label: 'One' }], projection: () => state.projection },
    clock: () => time.now,
    notifier: {
      recipients: () => ['alice', 'owner'],
      push: async (userId, message) => {
        if (options.failPush) throw new Error('push down')
        pushes.push({ userId, message })
      },
    },
    onChange: item => { changes.push(item) },
    onError: (_error, context) => { errors.push(context) },
  })
  return { queue, state, set, pushes, changes, errors, time }
}

describe('attention queue', () => {
  test('blocked, failed and waiting-on-human tasks open items with severity', async () => {
    const t = setup()
    t.set([
      { id: 'a', title: 'Waiting task', status: 'working', pending_interaction_count: 1 },
      { id: 'b', title: 'Broken task', status: 'failed' },
      { id: 'c', title: 'Fine task', status: 'idle' },
    ])
    const opened = await t.queue.scan()
    expect(opened.map(item => item.kind).sort()).toEqual(['human_waiting', 'task_failed'])
    expect(t.queue.list()[0]).toMatchObject({ kind: 'task_failed', severity: 'critical', sessionId: 'b' })
    expect(t.queue.list().find(item => item.kind === 'human_waiting')).toMatchObject({ severity: 'attention', sessionId: 'a' })
  })

  test('a blocked goal comes from a status event', async () => {
    const t = setup()
    t.set([{ id: 'a', title: 'Goal task', status: 'working' }])
    t.queue.observeEvents([{ workspace_id: 'w1', session_id: 'a', type: 'status_update', payload: { goal_phase: 'blocked' } }])
    expect((await t.queue.scan()).map(item => item.kind)).toEqual(['goal_blocked'])
    t.queue.observeEvents([{ workspace_id: 'w1', session_id: 'a', type: 'status_update', payload: { goal_phase: 'active' } }])
    await t.queue.scan()
    expect(t.queue.list()).toHaveLength(0)
  })

  test('items are deduplicated and subscribers are told once, with minimal text', async () => {
    const t = setup()
    t.set([{ id: 'b', title: 'Secret customer migration', status: 'failed' }])
    await t.queue.scan()
    await t.queue.scan()
    await t.queue.scan()
    expect(t.queue.list()).toHaveLength(1)
    expect(t.pushes.map(push => push.userId)).toEqual(['alice', 'owner'])
    for (const { message } of t.pushes) {
      expect(JSON.stringify(message)).not.toContain('Secret customer migration')
      expect(message.body).toBe('A task failed.')
      expect(message.url).toBe('/w/w1/s/b')
    }
  })

  test('acknowledgement is kept per person and stops further noise; the item stays until the cause clears', async () => {
    const t = setup()
    t.set([{ id: 'a', title: 'Waiting task', status: 'working', pending_interaction_count: 1 }])
    const [item] = await t.queue.scan()
    const acknowledged = t.queue.acknowledge(item!.id, 'alice')!
    expect(acknowledged).toMatchObject({ state: 'acknowledged', acknowledgedBy: ['alice'] })
    t.queue.acknowledge(item!.id, 'alice')
    expect(t.queue.list()[0]!.acknowledgedBy).toEqual(['alice'])
    expect(await t.queue.scan()).toHaveLength(0)
    expect(t.queue.acknowledge('nope', 'alice')).toBeUndefined()
    t.set([{ id: 'a', title: 'Waiting task', status: 'idle', pending_interaction_count: 0 }])
    await t.queue.scan()
    expect(t.queue.list()).toHaveLength(0)
    expect(t.queue.list({ includeResolved: true })[0]).toMatchObject({ state: 'resolved' })
  })

  test('a resolved cause that returns opens a new item and notifies again', async () => {
    const t = setup()
    t.set([{ id: 'b', title: 'Broken task', status: 'failed' }])
    await t.queue.scan()
    t.set([{ id: 'b', title: 'Broken task', status: 'idle' }])
    await t.queue.scan()
    t.set([{ id: 'b', title: 'Broken task', status: 'failed' }])
    const again = await t.queue.scan()
    expect(again).toHaveLength(1)
    expect(t.pushes).toHaveLength(4)
    expect(t.changes.map(item => item.state)).toEqual(['open', 'resolved', 'open'])
  })

  test('losing contact neither raises nor clears items', async () => {
    const t = setup()
    t.set([{ id: 'a', title: 'Waiting task', status: 'working', pending_interaction_count: 1 }])
    await t.queue.scan()
    t.set([{ id: 'a', title: 'Waiting task', status: 'idle', pending_interaction_count: 0 }])
    t.state.projection = markProjectionStale(t.state.projection)
    await t.queue.scan()
    expect(t.queue.list()).toHaveLength(1)
    expect(t.queue.list()[0]!.state).toBe('open')
  })

  test('a failing push is reported and does not break the scan', async () => {
    const t = setup({ failPush: true })
    t.set([{ id: 'b', title: 'Broken task', status: 'failed' }])
    expect(await t.queue.scan()).toHaveLength(1)
    expect(t.errors.filter(context => context === 'attention push')).toHaveLength(2)
  })

  test('it is read-only: the queue has nothing to resolve an interaction with', () => {
    const t = setup()
    const keys = Object.keys(t.queue)
    expect(keys.some(key => /backend|pool|router|call/i.test(key))).toBe(false)
  })
})
