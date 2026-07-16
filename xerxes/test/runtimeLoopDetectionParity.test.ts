// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { LoopDetector, ToolLoopError, type LoopEvent } from '../src/runtime/loopDetector.js'

test('loop detector leaves distinct tool calls unbounded by default', () => {
  const detector = new LoopDetector()
  let latest: LoopEvent | undefined

  for (let index = 0; index < 10_000; index += 1) {
    latest = detector.recordCall('ReadFile', { path: `file-${index}.ts` })
  }
  expect(latest?.severity).toBe('ok')
  expect(detector.callCount).toBe(10_000)
})

test('loop detector enforces an explicitly configured per-turn maximum', () => {
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

  expect(detector.recordCall('ListDir', {}).severity).toBe('ok')
  expect(detector.recordCall('ReadFile', {})).toMatchObject({
    severity: 'critical',
    pattern: 'max_calls',
    callCount: 5,
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

test('loop detector still stops alternating tool ping-pong without an absolute call cap', () => {
  const detector = new LoopDetector()

  for (let index = 0; index < 10_000; index += 1) {
    detector.recordCall('ReadFile', { path: `prefix-${index}.ts` })
  }
  const alternating = ['GlobTool', 'GrepTool', 'GlobTool', 'GrepTool', 'GlobTool', 'GrepTool', 'GlobTool']
    .map((toolName, index) => detector.recordCall(toolName, { index }))

  expect(alternating[3]).toMatchObject({ severity: 'warning', pattern: 'pingpong', callCount: 4 })
  expect(alternating[5]).toMatchObject({ severity: 'critical', pattern: 'pingpong', callCount: 6 })
})
