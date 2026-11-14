// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { Database } from 'bun:sqlite'

import { subscriptionEvent } from '../src/hub/supervisor/abilities/events.js'
import { SupervisorWakes } from '../src/hub/supervisor/abilities/wakes.js'
import type { HubEvent } from '../src/hub/protocol.js'

const NOW = Date.parse('2026-11-14T19:30:00Z')
const status = (sequence: number, payload: Record<string, unknown>, occurred = '2026-11-14T19:29:59Z'): HubEvent => ({
  event_id: `e${sequence}`, sequence, cursor: `c${sequence}`, workspace_id: 'w', session_id: 's1', type: 'status_update', payload, occurred_at: occurred,
})

function setup(clock: { now: number }) {
  const db = new Database(':memory:')
  const delivered: string[] = []
  const wakes = new SupervisorWakes(db, { clock: () => clock.now, paused: () => false, authorized: () => true,
    deliver: async wake => { delivered.push(wake.reason) } })
  wakes.subscribe({ owner: 'o', thread: 't', workspace: 'w' }, { kind: 'goal_blocked' })
  return { db, wakes, delivered }
}

test('a resync re-posting the same blocked goal wakes once, not once per replay', async () => {
  const clock = { now: NOW }
  const { db, wakes, delivered } = setup(clock)
  try {
    const blocked = { goal: 'ship it', goal_phase: 'blocked' }
    for (let sequence = 1; sequence <= 5; sequence += 1) await wakes.observe(subscriptionEvent(status(sequence, blocked))!)
    await wakes.drain()
    for (let sequence = 6; sequence <= 10; sequence += 1) await wakes.observe(subscriptionEvent(status(sequence, blocked))!)
    await wakes.drain()
    expect(delivered).toHaveLength(1)
    // A different goal revision is a real change of state.
    await wakes.observe(subscriptionEvent(status(11, { ...blocked, goal_revision: 7 }))!)
    await wakes.drain()
    expect(delivered).toHaveLength(2)
  } finally { wakes.dispose(); db.close() }
})

test('leaving blocked clears the state, so a real re-block wakes again', async () => {
  const clock = { now: NOW }
  const { db, wakes, delivered } = setup(clock)
  try {
    const blocked = { goal: 'ship it', goal_phase: 'blocked' }
    await wakes.observe(subscriptionEvent(status(1, blocked))!); await wakes.drain()
    await wakes.observe(subscriptionEvent(status(2, { goal: 'ship it', goal_phase: 'active' }))!)
    await wakes.observe(subscriptionEvent(status(3, blocked))!); await wakes.drain()
    expect(delivered).toHaveLength(2)
  } finally { wakes.dispose(); db.close() }
})

test('a replayed old fact never wakes; a burst merges into the waiting wake', async () => {
  const clock = { now: NOW }
  const { db, wakes, delivered } = setup(clock)
  try {
    wakes.subscribe({ owner: 'o', thread: 't', workspace: 'w' }, { kind: 'task_idle' })
    const old = { ...status(1, {}, '2026-11-13T10:00:00Z'), type: 'turn_end' }
    await wakes.observe(subscriptionEvent(old)!)
    await wakes.drain()
    expect(delivered).toHaveLength(0)
    // Three different fresh turn_end events before the drain: one queued wake.
    for (let sequence = 2; sequence <= 4; sequence += 1) await wakes.observe(subscriptionEvent({ ...status(sequence, {}), type: 'turn_end' })!)
    await wakes.drain()
    expect(delivered).toHaveLength(1)
  } finally { wakes.dispose(); db.close() }
})

test('a stuck claim expires, is re-queued once, then marked expired', async () => {
  const clock = { now: NOW }
  const db = new Database(':memory:')
  let hang = true
  const delivered: string[] = []
  const wakes = new SupervisorWakes(db, { clock: () => clock.now, paused: () => false, authorized: () => true, claimTtlMs: 60_000,
    deliver: async wake => { if (hang) await new Promise(() => {}); delivered.push(wake.id) } })
  try {
    const wake = wakes.schedule({ owner: 'o', thread: 't', workspace: 'w' }, { after: 0, reason: 'check' })
    void wakes.drain()
    await Bun.sleep(5)
    const state = () => db.query<{ state: string }, [string]>('SELECT state FROM hub_superagent_wakes WHERE id=?').get(wake.id)!.state
    expect(state()).toBe('claimed')
    clock.now += 61_000
    expect(wakes.recoverClaims()).toBe(1)
    expect(state()).toBe('pending')
    // Second claim also hangs: after the TTL it is given up, not looped.
    db.query("UPDATE hub_superagent_wakes SET state='claimed' WHERE id=?").run(wake.id)
    db.query('UPDATE hub_superagent_wake_claims SET attempts=2, claimed_at=? WHERE id=?').run(clock.now, wake.id)
    clock.now += 61_000
    wakes.recoverClaims()
    expect(state()).toBe('expired')
    hang = false
  } finally { wakes.dispose(); db.close() }
})

test('a legacy claimed wake with no claim record is expired, never replayed', async () => {
  const clock = { now: NOW }
  const db = new Database(':memory:')
  const delivered: string[] = []
  const wakes = new SupervisorWakes(db, { clock: () => clock.now, paused: () => false, authorized: () => true, claimTtlMs: 60_000,
    deliver: async wake => { delivered.push(wake.id) } })
  try {
    db.query("INSERT INTO hub_superagent_wakes VALUES ('legacy','o','t','w',?,'old','claimed')").run(clock.now - 3 * 86_400_000)
    // A recent legacy claim (inside the TTL) is left alone for now.
    db.query("INSERT INTO hub_superagent_wakes VALUES ('fresh','o','t','w',?,'new','claimed')").run(clock.now - 1000)
    await wakes.drain()
    const state = (id: string) => db.query<{ state: string }, [string]>('SELECT state FROM hub_superagent_wakes WHERE id=?').get(id)!.state
    expect(state('legacy')).toBe('expired')
    expect(state('fresh')).toBe('claimed')
    expect(delivered).toEqual([])
  } finally { wakes.dispose(); db.close() }
})

test('a timed claim is redelivered exactly once, marked as a retry', async () => {
  const clock = { now: NOW }
  const db = new Database(':memory:')
  const deliveries: { id: string; retry: boolean }[] = []
  let hang = true
  const wakes = new SupervisorWakes(db, { clock: () => clock.now, paused: () => false, authorized: () => true, claimTtlMs: 60_000,
    deliver: async wake => { deliveries.push({ id: wake.id, retry: wake.retry === true }); if (hang) await new Promise(() => {}) } })
  try {
    const wake = wakes.schedule({ owner: 'o', thread: 't', workspace: 'w' }, { after: 0, reason: 'check' })
    void wakes.drain()
    await Bun.sleep(5)
    // The first delivery hangs; the drain never settles, so recover directly and drain on a fresh instance.
    clock.now += 61_000
    const second = new SupervisorWakes(db, { clock: () => clock.now, paused: () => false, authorized: () => true, claimTtlMs: 60_000,
      deliver: async w => { deliveries.push({ id: w.id, retry: w.retry === true }); if (hang) await new Promise(() => {}) } })
    void second.drain()
    await Bun.sleep(5)
    expect(deliveries).toEqual([{ id: wake.id, retry: false }, { id: wake.id, retry: true }])
    // The retry hangs too: after the TTL it is expired, never a third delivery.
    clock.now += 61_000
    hang = false
    const third = new SupervisorWakes(db, { clock: () => clock.now, paused: () => false, authorized: () => true, claimTtlMs: 60_000,
      deliver: async w => { deliveries.push({ id: w.id, retry: w.retry === true }) } })
    await third.drain()
    expect(deliveries).toHaveLength(2)
    expect(db.query<{ state: string }, [string]>('SELECT state FROM hub_superagent_wakes WHERE id=?').get(wake.id)!.state).toBe('expired')
    second.dispose(); third.dispose()
  } finally { wakes.dispose(); db.close() }
})

test('a replayed old unblocked status does not clear the live blocked receipt', async () => {
  const clock = { now: NOW }
  const { db, wakes, delivered } = setup(clock)
  try {
    const blocked = { goal: 'ship it', goal_phase: 'blocked' }
    await wakes.observe(subscriptionEvent(status(1, blocked))!); await wakes.drain()
    expect(delivered).toHaveLength(1)
    // A resync replays yesterday's active status...
    await wakes.observe(subscriptionEvent(status(2, { goal: 'ship it', goal_phase: 'active' }, '2026-11-13T10:00:00Z'))!)
    // ...then the live status is still blocked: no extra wake.
    await wakes.observe(subscriptionEvent(status(3, blocked))!); await wakes.drain()
    expect(delivered).toHaveLength(1)
  } finally { wakes.dispose(); db.close() }
})

test('the goal_blocked id ignores a non-string goal with changing fields', () => {
  const a = subscriptionEvent(status(1, { goal_phase: 'blocked', goal: { objective: 'x', round: 3, updated_at: 'a' } }))!
  const b = subscriptionEvent(status(2, { goal_phase: 'blocked', goal: { objective: 'x', round: 4, updated_at: 'b' } }))!
  expect(a.id).toBe(b.id)
  const c = subscriptionEvent(status(3, { goal_phase: 'blocked', goal: 'ship it' }))!
  const d = subscriptionEvent(status(4, { goal_phase: 'blocked', goal: 'ship it', goal_revision: 2 }))!
  expect(c.id).not.toBe(d.id)
})
