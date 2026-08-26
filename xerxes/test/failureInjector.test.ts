// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { FailureInjector } from '../src/runtime/failureInjector.js'

test('failure injector throws when a matching rule fires', async () => {
  const injector = new FailureInjector({ rules: [{ mode: 'error', probability: 1, errorMessage: 'injected' }] })
  await expect(injector.inject('write', async () => 'ok')).rejects.toThrow('injected')
})

test('failure injector passes when probability misses', async () => {
  const injector = new FailureInjector({
    rules: [{ mode: 'error', probability: 0 }],
    random: () => 0.5,
  })
  const result = await injector.inject('read', async () => 'ok')
  expect(result).toBe('ok')
})

test('failure injector matches operations and adds latency', async () => {
  const injector = new FailureInjector({
    rules: [{ mode: 'latency', probability: 1, latencyMs: 2, match: op => op === 'slow' }],
  })
  let ran = false
  const result = await injector.inject('slow', async () => {
    ran = true
    return 'done'
  })
  expect(ran).toBeTrue()
  expect(result).toBe('done')
})
