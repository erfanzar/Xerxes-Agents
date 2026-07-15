// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  AsyncSessionContext,
  getActiveSession,
  runWithActiveSession,
  withActiveSession,
} from '../src/runtime/sessionContext.js'

test('active sessions propagate through asynchronous work and nested scopes restore their parent', async () => {
  const outer = { id: 'outer' }
  const inner = { id: 'inner' }
  await runWithActiveSession(outer, async () => {
    await Promise.resolve()
    expect(getActiveSession<typeof outer>()).toBe(outer)
    await runWithActiveSession(inner, async () => {
      await Promise.resolve()
      expect(getActiveSession<typeof inner>()).toBe(inner)
    })
    expect(getActiveSession<typeof outer>()).toBe(outer)
  })
  expect(getActiveSession()).toBeUndefined()
})

test('async iterator scopes remain isolated across interleaved live turns', async () => {
  const first = { id: 'first' }
  const second = { id: 'second' }
  const observed: string[] = []
  async function* source(expected: string): AsyncGenerator<string> {
    observed.push(getActiveSession<{ id: string }>()?.id ?? 'missing')
    await Promise.resolve()
    observed.push(getActiveSession<{ id: string }>()?.id ?? 'missing')
    yield expected
    await Promise.resolve()
    observed.push(getActiveSession<{ id: string }>()?.id ?? 'missing')
    yield expected + '-done'
  }

  const consume = async (session: { id: string }) => {
    const values: string[] = []
    for await (const value of withActiveSession(session, source(session.id))) values.push(value)
    return values
  }
  await expect(Promise.all([consume(first), consume(second)])).resolves.toEqual([
    ['first', 'first-done'],
    ['second', 'second-done'],
  ])
  expect(observed).toEqual(expect.arrayContaining(['first', 'second']))
  expect(observed.filter(value => value === 'missing')).toEqual([])
})

test('caller-owned session contexts do not share the default carrier', () => {
  const context = new AsyncSessionContext<{ id: string }>()
  context.run({ id: 'private' }, () => {
    expect(context.current()).toEqual({ id: 'private' })
    expect(getActiveSession()).toBeUndefined()
  })
})
