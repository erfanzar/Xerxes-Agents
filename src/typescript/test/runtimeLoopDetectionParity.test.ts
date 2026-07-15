// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { LoopDetector, ToolLoopError } from '../src/runtime/loopDetector.js'

test('loop detector keeps distinct arguments clear while enforcing the per-turn maximum', () => {
  const detector = new LoopDetector({
    maxToolCallsPerTurn: 4,
    sameCallWarning: 2,
    sameCallCritical: 3,
  })

  expect(detector.callCount).toBe(0)
  expect(detector.recordCall('search', { q: 'hello' }).severity).toBe('ok')
  expect(detector.recordCall('search', { q: 'world' }).severity).toBe('ok')
  expect(detector.recordCall('search', { q: 'again' }).severity).toBe('ok')
  expect(detector.callCount).toBe(3)

  expect(detector.recordCall('ListDir', {})).toMatchObject({
    severity: 'critical',
    pattern: 'max_calls',
    callCount: 4,
  })
})

test('disabled loop detection leaves calls clear without accumulating per-turn state', () => {
  const detector = new LoopDetector({ enabled: false })

  for (let index = 0; index < 10; index += 1) {
    expect(detector.recordCall('same', { index }).severity).toBe('ok')
  }
  expect(detector.callCount).toBe(0)
})

test('loop detector notifies listeners, supports removal, resets state, and exposes critical errors', () => {
  const detector = new LoopDetector({
    sameCallWarning: 2,
    sameCallCritical: 3,
  })
  const events: unknown[] = []
  const removeListener = detector.addListener((event) => events.push(event))

  detector.recordCall('ReadFile', { path: 'README.md' })
  const warning = detector.recordCall('ReadFile', { path: 'README.md' })
  expect(warning).toMatchObject({ severity: 'warning', pattern: 'same_call' })
  expect(events).toEqual([warning])

  removeListener()
  const critical = detector.recordCall('ReadFile', { path: 'README.md' })
  expect(critical.severity).toBe('critical')
  expect(events).toEqual([warning])

  const error = new ToolLoopError(critical)
  expect(error.message).toContain('same_call')
  expect(error.event).toBe(critical)

  detector.reset()
  expect(detector.callCount).toBe(0)
  expect(detector.recordCall('ReadFile', { path: 'README.md' }).severity).toBe(
    'ok',
  )
})
