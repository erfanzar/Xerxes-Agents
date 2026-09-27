// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { durationOf, elapsedOf } from '../src/desktop/renderer/duration.js'

test('elapsed time climbs seconds → minutes → hours → days, never thousands of seconds', () => {
  expect(elapsedOf(0)).toBe('0s')
  expect(elapsedOf(43)).toBe('43s')
  expect(elapsedOf(60)).toBe('1m')
  expect(elapsedOf(255)).toBe('4m 15s')
  expect(elapsedOf(2_487)).toBe('41m 27s')
  expect(elapsedOf(7_500)).toBe('2h 5m')
  expect(elapsedOf(90_000)).toBe('1d 1h')
  expect(elapsedOf(-5)).toBe('0s')
  expect(elapsedOf(Number.NaN)).toBe('0s')
})

test('compact keeps only the largest unit, for the session list', () => {
  expect(elapsedOf(43, true)).toBe('43s')
  expect(elapsedOf(2_487, true)).toBe('41m')
  expect(elapsedOf(7_500, true)).toBe('2h')
  expect(elapsedOf(200_000, true)).toBe('2d')
  expect(durationOf(255_400)).toBe('4m 15s')
})
