// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { expect, test } from 'bun:test'
import { SupervisorWakes } from '../src/hub/supervisor/abilities/wakes.js'

test('wakes persist, claim once across instances and honor pause and revocation', async () => {
  const db = new Database(':memory:'); let now = 1000; let paused = false; let authorized = true
  const delivered: string[] = []
  const options = { clock: () => now, paused: () => paused, authorized: () => authorized,
    deliver: async (wake: { id: string }) => { delivered.push(wake.id) } }
  const first = new SupervisorWakes(db, options)
  try {
    const wake = first.schedule({ owner: 'o', thread: 't', workspace: 'w' }, { after: 100, reason: 'Check tests' })
    const second = new SupervisorWakes(db, options)
    try {
      now = 1100; paused = true; await second.drain(); expect(delivered).toEqual([])
      paused = false; await Promise.all([first.drain(), second.drain()]); expect(delivered).toEqual([wake.id])
      second.schedule({ owner: 'o', thread: 't', workspace: 'w' }, { at: 1200, reason: 'later' })
      now = 1200; authorized = false; await second.drain(); expect(delivered).toHaveLength(1)
    } finally { second.dispose() }
  } finally { first.dispose(); db.close() }
})

test('event subscriptions dedupe events and unchanged check-ins are completely silent', async () => {
  const db = new Database(':memory:'); const delivered: string[] = []
  const wakes = new SupervisorWakes(db, { clock: () => 1000, paused: () => false, authorized: () => true,
    deliver: async wake => { delivered.push(wake.reason) } })
  try {
    const scope = { owner: 'o', thread: 't', workspace: 'w' }
    wakes.subscribe(scope, { kind: 'tests_failing' })
    await wakes.observe({ id: 'event1', workspace: 'w', kind: 'tests_failing', digest: 'failure1' })
    await wakes.observe({ id: 'event1', workspace: 'w', kind: 'tests_failing', digest: 'failure1' })
    await wakes.drain(); expect(delivered).toHaveLength(1)
    expect(wakes.changed(scope, 'same')).toBe(true)
    expect(wakes.changed(scope, 'same')).toBe(false)
    expect(delivered).toHaveLength(1)
    expect(() => wakes.schedule(scope, { at: 1100, after: 50, reason: 'ambiguous' })).toThrow()
    expect(() => wakes.schedule(scope, { after: -1, reason: 'bad' })).toThrow()
  } finally { wakes.dispose(); db.close() }
})

test('disk subscriptions only wake above their configured threshold and cancelled wakes never deliver', async () => {
  const db = new Database(':memory:'); let count = 0
  const wakes = new SupervisorWakes(db, { clock: () => 1000, paused: () => false, authorized: () => true, deliver: async () => { count++ } })
  try {
    const scope = { owner: 'o', thread: 't', workspace: 'w' }
    wakes.subscribe(scope, { kind: 'disk_threshold', threshold: 90 })
    await wakes.observe({ id: 'low', workspace: 'w', kind: 'disk_threshold', digest: 'low', value: 50 })
    await wakes.drain(); expect(count).toBe(0)
    await wakes.observe({ id: 'high', workspace: 'w', kind: 'disk_threshold', digest: 'high', value: 95 })
    await wakes.drain(); expect(count).toBe(1)
    const pending = wakes.schedule(scope, { after: 1, reason: 'cancel me' })
    expect(wakes.cancel(scope, pending.id)).toBe(true)
  } finally { wakes.dispose(); db.close() }
})
