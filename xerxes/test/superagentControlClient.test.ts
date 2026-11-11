// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createSuperagentControlClient, parseSuperagentControl, parseDailyLimits } from '../src/client/superagentControl.js'

const reply = { control: { revision: 2, paused: false, killed: true, limits: { actions: 0, usd: 0.5 } },
  usage: { actions: 0, wakes: 2, modelCalls: 3, tokens: null, usd: null } }

test('control snapshots preserve zero caps and unknown usage without inventing values', () => {
  expect(parseSuperagentControl(reply)).toMatchObject({ revision: 2, killed: true, limits: { actions: 0 }, usage: { tokens: null, usd: null } })
})

test('malformed control snapshots fail closed instead of enabling guessed controls', () => {
  for (const control of [{}, { ...reply.control, revision: -1 }, { ...reply.control, paused: 'false' }, { ...reply.control, limits: { tokens: -1 } }]) {
    expect(() => parseSuperagentControl({ ...reply, control })).toThrow()
  }
  expect(() => parseSuperagentControl({ ...reply, usage: { ...reply.usage, usd: Number.NaN } })).toThrow()
})

test('owner control client binds the current revision and serializes a mutation without retries', async () => {
  const calls: { method: string; params: Readonly<Record<string, unknown>> }[] = []
  const client = createSuperagentControlClient({ allowed: () => true, request: async (method, params) => {
    calls.push({ method, params }); return reply
  } })
  await client.load()
  expect(await client.save({ paused: true })).toBe(true)
  expect(calls).toEqual([{ method: 'supervisor.control.get', params: {} }, { method: 'supervisor.control.set', params: { revision: 2, paused: true } }])
})

test('a second save cannot overlap an unresolved first save', async () => {
  let writes = 0; let finish = (_value: unknown): void => {}
  const pending = new Promise<unknown>(resolve => { finish = resolve })
  const client = createSuperagentControlClient({ allowed: () => true, request: async method => {
    if (method.endsWith('.set')) { writes += 1; return pending }
    return reply
  } })
  await client.load(); const first = client.save({ paused: true })
  expect(await client.save({ paused: false })).toBe(false); expect(writes).toBe(1)
  finish(reply); expect(await first).toBe(true)
})

test('refused or unknown saves lock further mutations until an explicit reload', async () => {
  let writes = 0
  const client = createSuperagentControlClient({ allowed: () => true, request: async method => {
    if (method.endsWith('.set')) { writes += 1; throw new Error('unknown outcome') }
    return reply
  } })
  await client.load(); expect(await client.save({ killed: true })).toBe(false)
  expect(await client.save({ killed: true })).toBe(false)
  expect(writes).toBe(1); expect(client.getSnapshot().phase).toBe('error')
  await client.load(); await client.save({ killed: true }); expect(writes).toBe(2)
})

test('offline or non-owner control clients make no requests', async () => {
  let calls = 0
  const client = createSuperagentControlClient({ allowed: () => false, request: async () => { calls += 1; return reply } })
  await client.load(); expect(await client.save({ paused: true })).toBe(false)
  expect(calls).toBe(0)
})

test('late responses after disposal or revoked ownership cannot publish a saved state', async () => {
  let finish = (_value: unknown): void => {}; let allowed = true
  const pending = new Promise<unknown>(resolve => { finish = resolve })
  const client = createSuperagentControlClient({ allowed: () => allowed, request: async () => pending })
  const loading = client.load(); allowed = false; finish(reply); await loading
  expect(client.getSnapshot().data).toBeUndefined()
  const disposed = createSuperagentControlClient({ allowed: () => true, request: async () => reply })
  disposed.dispose(); await disposed.load(); expect(disposed.getSnapshot().data).toBeUndefined()
})

test('daily cap input distinguishes no cap, explicit zero, integer counters and decimal USD', () => {
  expect(parseDailyLimits({ actions: '0', wakes: '', modelCalls: '5', tokens: '', usd: '0.25' })).toEqual({ actions: 0, modelCalls: 5, usd: 0.25 })
  expect(() => parseDailyLimits({ actions: '1.5', wakes: '', modelCalls: '', tokens: '', usd: '' })).toThrow('actions')
  expect(() => parseDailyLimits({ actions: '', wakes: '', modelCalls: '', tokens: '-1', usd: '' })).toThrow('tokens')
})
