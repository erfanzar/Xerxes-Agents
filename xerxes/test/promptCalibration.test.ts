// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { CONTEXT_CALIBRATION_METADATA_KEY, contextCalibrationRatio, promptCalibration } from '../src/context/windowUsage.js'

test('an estimate is scaled by the provider count of the request it estimated', () => {
  const calibration = promptCalibration()
  // Nothing measured yet: the heuristic stands.
  expect(calibration.project(500_000)).toBe(500_000)
  // The provider counted that request at 900k: every later estimate runs 1.8x.
  expect(calibration.project(550_000, 900_000)).toBe(990_000)
  expect(calibration.ratio).toBeCloseTo(1.8)
  // No new measurement keeps the last ratio.
  expect(calibration.project(100_000)).toBe(180_000)
})

test('a measured ratio is clamped and garbage counts are ignored', () => {
  const calibration = promptCalibration()
  calibration.project(1_000)
  expect(calibration.project(1_000, 50_000)).toBe(4_000)
  expect(calibration.project(1_000, 0)).toBe(4_000)
  expect(calibration.project(1_000, 1)).toBe(500)
})

test('a persisted ratio applies only to the model it was measured on', () => {
  const metadata = { [CONTEXT_CALIBRATION_METADATA_KEY]: { model: 'claude-code/opus', ratio: 1.8 } }
  expect(contextCalibrationRatio(metadata, 'claude-code/opus')).toBe(1.8)
  expect(contextCalibrationRatio(metadata, 'claude-code/sonnet')).toBe(1)
  expect(contextCalibrationRatio({}, 'claude-code/opus')).toBe(1)
  expect(contextCalibrationRatio({ [CONTEXT_CALIBRATION_METADATA_KEY]: { model: 'm', ratio: 'x' } }, 'm')).toBe(1)
  expect(contextCalibrationRatio({ [CONTEXT_CALIBRATION_METADATA_KEY]: { model: 'm', ratio: 99 } }, 'm')).toBe(4)
  expect(promptCalibration(1.8).project(100)).toBe(180)
})
