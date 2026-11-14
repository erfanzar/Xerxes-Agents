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
    // A legacy claim with no claim record recovers from its due time.
    hang = false
    db.query("INSERT INTO hub_superagent_wakes VALUES ('legacy','o','t','w',?,'old','claimed')").run(clock.now - 120_000)
    expect(wakes.recoverClaims()).toBe(1)
  } finally { wakes.dispose(); db.close() }
})
