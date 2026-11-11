// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { DiskThresholdMonitor } from '../src/hub/supervisor/abilities/disk.js'

const idle = { intervalMs: 3_600_000, paused: () => false }

test('disk monitor coalesces overlapping reads and quietly aborts in-flight work on shutdown', async () => {
  const completion = Promise.withResolvers<unknown>()
  const errors: unknown[] = []
  let reads = 0, observations = 0
  let readSignal: AbortSignal | undefined
  const monitor = new DiskThresholdMonitor({ ...idle, workspaces: () => ['w', 'w'],
    read: async (_workspace, signal) => { reads++; readSignal = signal; return completion.promise },
    observe: async () => { observations++ }, onError: error => { errors.push(error) } })
  const first = monitor.tick(), second = monitor.tick()
  expect(reads).toBe(1)
  monitor.dispose()
  expect(readSignal?.aborted).toBe(true)
  completion.reject(new Error('shutdown cancellation'))
  await Promise.all([first, second])
  expect(observations).toBe(0); expect(errors).toEqual([])
})

test('paused disk monitors do not query subscriptions and malformed observations remain visible errors', async () => {
  let paused = true, queries = 0, reads = 0, observations = 0
  const errors: unknown[] = []
  const monitor = new DiskThresholdMonitor({ ...idle, paused: () => paused,
    workspaces: () => { queries++; return ['w'] }, read: async () => { reads++; return { ok: true, observation: { disk: { available_bytes: -1, total_bytes: 100 } } } },
    observe: async () => { observations++ }, onError: error => { errors.push(error) } })
  try {
    await monitor.tick(); expect(queries).toBe(0); expect(reads).toBe(0)
    paused = false
    await monitor.tick(); expect(reads).toBe(1); expect(observations).toBe(0)
    expect(errors).toHaveLength(1)
    expect(errors[0]).toBeInstanceOf(Error)
  } finally { monitor.dispose() }
})

test('disk observations carry only measured bounded facts and no task instructions', async () => {
  const events: unknown[] = []
  const monitor = new DiskThresholdMonitor({ ...idle, workspaces: () => ['w'],
    read: async () => ({ ok: true, observation: { disk: { available_bytes: 10, total_bytes: 100 }, instruction: 'execute forbidden command' } }),
    observe: async event => { events.push(event) } })
  try {
    await monitor.tick()
    expect(events).toEqual([expect.objectContaining({ workspace: 'w', kind: 'disk_threshold', value: 90, digest: 'disk used 90.0%' })])
    expect(JSON.stringify(events)).not.toContain('forbidden command')
  } finally { monitor.dispose() }
})
