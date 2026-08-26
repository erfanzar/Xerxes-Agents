// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { Benchmark } from '../src/runtime/benchmark.js'

test('benchmark records success distribution and percentile statistics', async () => {
  let now = 0
  const benchmark = new Benchmark({ name: 'identity', iterations: 10, now: () => now })
  const summary = await benchmark.run(async () => {
    now += 5
  })
  expect(summary.name).toBe('identity')
  expect(summary.iterations).toBe(10)
  expect(summary.successes).toBe(10)
  expect(summary.failures).toBe(0)
  expect(summary.meanMs).toBe(5)
  expect(summary.p50Ms).toBe(5)
  expect(summary.p95Ms).toBe(5)
})

test('benchmark records failures without throwing', async () => {
  let now = 0
  const benchmark = new Benchmark({ name: 'failing', iterations: 4, now: () => now })
  let counter = 0
  const summary = await benchmark.run(async () => {
    counter += 1
    if (counter % 2 === 0) throw new Error('boom')
    now += 1
  })
  expect(summary.successes).toBe(2)
  expect(summary.failures).toBe(2)
  expect(summary.meanMs).toBe(0.5)
})
