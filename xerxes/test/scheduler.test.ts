// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { Scheduler } from '../src/runtime/scheduler.js'

function taskPayload(id: string): Parameters<Scheduler['createTrigger']>[0]['payload'] {
  return { id, objective: `scheduled task ${id}`, creatorId: 'scheduler', dependencies: [] }
}

test('scheduler creates, disables, enables, and removes triggers durably', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-scheduler-'))
  let now = 1_000
  try {
    const scheduler = new Scheduler({ directory, now: () => now })
    const created = await scheduler.createTrigger({
      id: 't1',
      owner: 'user',
      schedule: { kind: 'interval', intervalSeconds: 60 },
      payload: taskPayload('t1-run'),
    })
    expect(created.enabled).toBeTrue()

    await scheduler.disableTrigger('t1')
    let state = await scheduler.load()
    expect(state.triggers.get('t1')?.enabled).toBeFalse()

    await scheduler.enableTrigger('t1')
    state = await scheduler.load()
    expect(state.triggers.get('t1')?.enabled).toBeTrue()

    await scheduler.removeTrigger('t1')
    state = await scheduler.load()
    expect(state.triggers.has('t1')).toBeFalse()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('scheduler evaluates interval triggers and records delivery idempotency', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-scheduler-'))
  let now = 1_000
  try {
    const scheduler = new Scheduler({ directory, now: () => now })
    await scheduler.createTrigger({
      id: 'every-minute',
      owner: 'user',
      schedule: { kind: 'interval', intervalSeconds: 60 },
      payload: taskPayload('every-minute-run'),
    })

    let due = await scheduler.evaluate(now)
    expect(due).toHaveLength(1)

    await scheduler.fire('every-minute', 'delivery-1')

    const repeated = await scheduler.fire('every-minute', 'delivery-1')
    expect(repeated.fired).toBeFalse()

    now += 61_000
    due = await scheduler.evaluate(now)
    expect(due).toHaveLength(1)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('scheduler matches cron patterns against current time', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-scheduler-cron-'))
  try {
    const scheduler = new Scheduler({ directory, now: () => new Date('2026-08-26T14:05:00.000Z').getTime() })
    await scheduler.createTrigger({
      id: 'top-of-hour',
      owner: 'user',
      schedule: { kind: 'cron', minute: '5', hour: '14' },
      payload: taskPayload('cron-run'),
    })
    const due = await scheduler.evaluate()
    expect(due).toHaveLength(1)

    const recovered = new Scheduler({ directory, now: () => new Date('2026-08-26T14:06:00.000Z').getTime() })
    expect((await recovered.evaluate())).toHaveLength(0)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('scheduler only fires webhook and event triggers', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-scheduler-fire-'))
  try {
    const scheduler = new Scheduler({ directory })
    await scheduler.createTrigger({
      id: 'web',
      owner: 'user',
      schedule: { kind: 'webhook', path: '/webhook' },
      payload: taskPayload('web-run'),
    })
    await scheduler.createTrigger({
      id: 'interval',
      owner: 'user',
      schedule: { kind: 'interval', intervalSeconds: 1 },
      payload: taskPayload('interval-run'),
    })

    const webResult = await scheduler.fire('web', 'd1')
    expect(webResult.fired).toBeTrue()

    const intervalResult = await scheduler.fire('interval', 'd2')
    expect(intervalResult.fired).toBeFalse()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('a time-based trigger stops being due once it is recorded as fired', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-scheduler-interval-'))
  try {
    let clock = 1_000_000
    const scheduler = new Scheduler({ directory, now: () => clock })
    await scheduler.createTrigger({
      id: 'hourly',
      owner: 'user',
      schedule: { kind: 'interval', intervalSeconds: 3600 },
      payload: taskPayload('hourly-run'),
    })

    // `lastFiredAt` was only written by delivery_recorded, and fire() refuses
    // anything that is not event or webhook — so nothing could ever set it and
    // an interval trigger was due on every single poll.
    expect((await scheduler.evaluate()).map(t => t.id)).toEqual(['hourly'])
    expect(await scheduler.markFired('hourly')).toBe(true)
    expect(await scheduler.evaluate()).toEqual([])

    clock += 3599 * 1000
    expect(await scheduler.evaluate()).toEqual([])
    clock += 2 * 1000
    expect((await scheduler.evaluate()).map(t => t.id)).toEqual(['hourly'])
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('concurrent fires of one delivery id produce exactly one task', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-scheduler-race-'))
  try {
    const scheduler = new Scheduler({ directory })
    await scheduler.createTrigger({
      id: 'hook',
      owner: 'user',
      schedule: { kind: 'webhook', path: '/incoming' },
      payload: taskPayload('hook-run'),
    })

    // Read-then-append with nothing between: both callers loaded before either
    // appended, both missed the dedupe set, and a webhook delivered twice
    // concurrently produced two tasks.
    const results = await Promise.all([
      scheduler.fire('hook', 'delivery-1'),
      scheduler.fire('hook', 'delivery-1'),
      scheduler.fire('hook', 'delivery-1'),
    ])
    expect(results.filter(result => result.fired)).toHaveLength(1)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
