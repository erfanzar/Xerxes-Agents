// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  emitRuntimeEvent,
  getActiveConfig,
  getConfig,
  getInheritableConfig,
  runWithActiveConfig,
  setConfig,
  setEventCallback,
} from '../src/runtime/configContext.js'

test('runtime config snapshots expose only explicitly inheritable values', () => {
  setConfig({
    model: 'gpt-test',
    api_key: 'secret',
    empty: '',
    ignored: 'not-inherited',
    temperature: 0.2,
    top_p: null,
  })

  expect(getInheritableConfig()).toEqual({
    model: 'gpt-test',
    api_key: 'secret',
    temperature: 0.2,
  })
  const snapshot = getConfig()
  snapshot.model = 'changed'
  expect(getConfig().model).toBe('gpt-test')
})

test('active config is isolated by async scope and never leaks mutations', async () => {
  const outer = getActiveConfig()
  const result = await runWithActiveConfig({ model: 'scope-model' }, async () => {
    const active = getActiveConfig()
    if (active === undefined) throw new Error('missing active config')
    active.model = 'mutated-copy'
    await Promise.resolve()
    return getActiveConfig()
  })

  expect(result).toEqual({ model: 'scope-model' })
  expect(getActiveConfig()).toEqual(outer)
})

test('runtime event callbacks receive copies and cannot interrupt execution', () => {
  const received: Array<{ readonly data: Record<string, unknown>; readonly type: string }> = []
  setEventCallback((type, data) => {
    const captured = { ...data, changed: true }
    received.push({ type, data: captured })
  })
  emitRuntimeEvent('agent_started', { id: 'a1' })
  expect(received).toEqual([{ type: 'agent_started', data: { id: 'a1', changed: true } }])

  setEventCallback(() => {
    throw new Error('observer failure')
  })
  expect(() => emitRuntimeEvent('agent_finished', { id: 'a1' })).not.toThrow()
  setEventCallback(undefined)
})
