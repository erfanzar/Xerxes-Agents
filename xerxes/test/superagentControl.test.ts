// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { SuperagentControl } from '../src/hub/supervisor/abilities/control.js'
import { makeWorld, scriptedModel } from './hubSupervisorHarness.js'
import { disposeAll } from './hubHarness.js'

afterEach(disposeAll)
function fixture() {
  const world = makeWorld()
  const events: string[] = []
  const options = { db: world.db, clock: world.clock, isOwner: (id: string) => id === world.user('owner').id,
    audit: async (event: { kind: string }) => { events.push(event.kind) } }
  return { ...world, options, control: new SuperagentControl(options), events }
}

test('only the live owner can configure durable pause/kill and bounded daily limits', async () => {
  const world = fixture()
  await expect(world.control.configure('alice', { revision: 0, paused: true })).rejects.toThrow()
  await expect(world.control.configure('owner', { revision: 0, limits: { actions: -1 } })).rejects.toThrow()
  await expect(world.control.configure('owner', { revision: 0, owner_approved: true })).rejects.toThrow()
  const first = await world.control.configure('owner', { revision: 0, limits: { actions: 2, tokens: 50, usd: 1 } })
  expect(first.revision).toBe(1)
  await expect(world.control.configure('owner', { revision: 0, paused: true })).rejects.toThrow()
  const reservation = await world.control.reserve('command', 'cmd-1')
  expect(reservation.id).toBe('cmd-1')
  await world.control.configure('owner', { revision: 1, killed: true })
  expect(world.control.signal.aborted).toBe(true)
  await expect(world.control.reserve('command', 'cmd-2')).rejects.toThrow()
  const reopened = new SuperagentControl(world.options)
  expect(reopened.status().killed).toBe(true)
  expect(reopened.signal.aborted).toBe(true)
  await expect(reopened.reserve('command', 'cmd-2')).rejects.toThrow()
})

test('daily budget reservations are atomic, persistent and never replay external work', async () => {
  const world = fixture()
  await world.control.configure('owner', { revision: 0, limits: { actions: 1 } })
  const outcomes = await Promise.allSettled([world.control.reserve('command', 'a'), world.control.reserve('command', 'b')])
  expect(outcomes.filter(result => result.status === 'fulfilled')).toHaveLength(1)
  const reopened = new SuperagentControl(world.options)
  await expect(reopened.reserve('command', 'c')).rejects.toThrow()
  const winner = outcomes[0]?.status === 'fulfilled' ? 'a' : 'b'
  await expect(reopened.reserve('command', winner)).rejects.toThrow()
  world.time.now += 24 * 3600_000
  expect((await reopened.reserve('command', 'next-day')).id).toBe('next-day')
})

test('model monetary/token caps refuse unknown quotes rather than counting unknown as zero', async () => {
  const world = fixture()
  await world.control.configure('owner', { revision: 0, limits: { tokens: 100, usd: 1 } })
  await expect(world.control.reserve('model', 'unknown')).rejects.toThrow()
  await expect(world.control.reserve('model', 'unknown-usd', { tokens: 10 })).rejects.toThrow()
  await expect(world.control.reserve('model', 'negative', { tokens: 10, usd: -1 })).rejects.toThrow()
  await world.control.reserve('model', 'known', { tokens: 80, usd: 0.8 })
  await expect(world.control.reserve('model', 'over', { tokens: 30, usd: 0.3 })).rejects.toThrow()
  world.control.settle('known', { tokens: 50, usd: 0.5 })
  await world.control.reserve('model', 'within', { tokens: 40, usd: 0.4 })
  expect(world.control.usage().tokens).toBe(90)
  expect(world.control.usage().usd).toBeCloseTo(0.9)
})

test('a pause while admission audit is pending prevents admission after the await', async () => {
  const world = fixture()
  let finish = (): void => {}
  const blocker = new Promise<void>(resolve => { finish = resolve })
  const control = new SuperagentControl({ ...world.options, audit: async event => { if (event.kind === 'reserve') await blocker } })
  const pending = control.reserve('command', 'pending')
  await world.control.configure('owner', { revision: 0, paused: true })
  finish()
  await expect(pending).rejects.toThrow()
})

test('usage exceeding a promised bound pauses further work and retains actual charges', async () => {
  const world = fixture()
  await world.control.configure('owner', { revision: 0, limits: { tokens: 100 } })
  await world.control.reserve('model', 'underestimated', { tokens: 10, usd: 0.01 })
  expect(() => world.control.settle('underestimated', { tokens: 20, usd: 0.02 })).toThrow()
  expect(world.control.status().paused).toBe(true)
  expect(world.control.usage().tokens).toBe(20)
  await expect(world.control.reserve('command', 'blocked')).rejects.toThrow()
})

test('controlled model calls reserve daily budget before invoking the real port', async () => {
  const world = fixture()
  const fake = scriptedModel([{ text: 'ok', toolCalls: [], usage: { inputTokens: 3, outputTokens: 2, costUsd: 0.1 } }])
  const controlled = world.control.model({ ...fake.model, quote: () => ({ tokens: 30, usd: 0.2 }) })
  await world.control.configure('owner', { revision: 0, limits: { modelCalls: 1, tokens: 100, usd: 1 } })
  const request = { system: '', messages: [], tools: [], maxTokens: 10 }
  expect((await controlled.complete(request)).text).toBe('ok')
  expect(world.control.usage()).toMatchObject({ modelCalls: 1, tokens: 5, usd: 0.1 })
  await expect(controlled.complete(request)).rejects.toThrow()
  expect(fake.requests).toHaveLength(1)
})

test('a monetary cap without a trustworthy model quote makes zero provider calls', async () => {
  const world = fixture()
  const fake = scriptedModel([{ text: '', toolCalls: [] }])
  await world.control.configure('owner', { revision: 0, limits: { usd: 1 } })
  await expect(world.control.model(fake.model).complete({ system: '', messages: [], tools: [], maxTokens: 10 })).rejects.toThrow()
  expect(fake.requests).toHaveLength(0)
})

test('kill and shutdown fence a late model response even when the model ignores abort', async () => {
  for (const shutdown of [false, true]) {
    const world = fixture()
    let finish = (): void => {}; let started = (): void => {}
    const blocking = new Promise<void>(resolve => { finish = resolve })
    const ready = new Promise<void>(resolve => { started = resolve })
    const controlled = world.control.model({ id: 'late', reportsCost: false, async complete() { started(); await blocking; return { text: 'too late', toolCalls: [] } } })
    const pending = controlled.complete({ system: '', messages: [], tools: [], maxTokens: 10 })
    await ready
    if (shutdown) { world.control.dispose(); world.db.close() }
    else await world.control.configure('owner', { revision: 0, killed: true })
    finish()
    await expect(pending).rejects.toThrow(/killed|disposed/i)
  }
})
