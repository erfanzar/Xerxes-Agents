// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { LoopDetector } from '../src/runtime/loopDetector.js'

test('detector warns then fails identical repeated calls', () => {
  const detector = new LoopDetector({ sameCallWarning: 2, sameCallCritical: 3 })
  expect(detector.recordCall('ReadFile', { path: 'a' }).severity).toBe('ok')
  expect(detector.recordCall('ReadFile', { path: 'a' }).severity).toBe('warning')
  expect(detector.recordCall('ReadFile', { path: 'a' }).severity).toBe('critical')
})

test('detector recognizes alternating tool loops', () => {
  const detector = new LoopDetector({ pingPongWarning: 3, pingPongCritical: 5 })
  detector.recordCall('ReadFile')
  detector.recordCall('GrepTool')
  detector.recordCall('ReadFile')
  expect(detector.recordCall('GrepTool').pattern).toBe('pingpong')
})
