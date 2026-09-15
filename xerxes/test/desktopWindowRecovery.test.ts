// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { windowRecovery } from '../src/desktop/main/windowRecovery.js'

test('native recovery coalesces failures and reloads only after explicit retry', async () => {
  let decide!: (retry: boolean) => void
  const details: string[] = []
  let reloads = 0
  const recover = windowRecovery({ closed: () => false,
    prompt: detail => { details.push(detail); return new Promise(resolve => { decide = resolve }) },
    reload: async () => { reloads++ }, report: error => { throw error },
  })
  const first = recover('crashed')
  await recover('load failed')
  expect(details).toEqual(['crashed'])
  expect(reloads).toBe(0)
  decide(false)
  await first
  expect(reloads).toBe(0)
  const second = recover('crashed again')
  decide(true)
  await second
  expect(reloads).toBe(1)
})

test('native recovery does not reload a closed window and reports load errors', async () => {
  let closed = false
  let reloads = 0
  const errors: unknown[] = []
  const failure = new Error('missing renderer')
  const recover = windowRecovery({ closed: () => closed,
    prompt: async () => true,
    reload: async () => { reloads++; throw failure }, report: error => errors.push(error),
  })
  await recover('load failed')
  expect(errors).toEqual([failure])
  closed = true
  await recover('crashed')
  expect(reloads).toBe(1)
  const closing = windowRecovery({ closed: () => closed,
    prompt: async () => { closed = true; return true }, reload: async () => { reloads++ }, report: error => errors.push(error) })
  closed = false
  await closing('crashed')
  expect(reloads).toBe(1)
})
