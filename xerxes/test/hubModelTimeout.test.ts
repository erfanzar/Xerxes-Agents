// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { HUB_MODEL_MAX_TIMEOUT_MS, HUB_MODEL_MIN_TIMEOUT_MS, HUB_MODEL_TIMEOUT_MS, hubModelTimeoutMs } from '../src/daemon/hubModels.js'

test('the Hub model bound is long enough for a reasoning turn and can be raised', () => {
  // Two minutes cut off slow reasoning models mid-thought; XERXES reported
  // "the model did not answer within 120 seconds" and stopped.
  expect(HUB_MODEL_TIMEOUT_MS).toBe(600_000)
  expect(hubModelTimeoutMs({})).toBe(HUB_MODEL_TIMEOUT_MS)
  expect(hubModelTimeoutMs({ XERXES_HUB_MODEL_TIMEOUT_MS: '900000' })).toBe(900_000)
})

test('an out-of-range or malformed override falls back instead of disabling the bound', () => {
  expect(hubModelTimeoutMs({ XERXES_HUB_MODEL_TIMEOUT_MS: '1000' })).toBe(HUB_MODEL_MIN_TIMEOUT_MS)
  expect(hubModelTimeoutMs({ XERXES_HUB_MODEL_TIMEOUT_MS: '99999999' })).toBe(HUB_MODEL_MAX_TIMEOUT_MS)
  expect(hubModelTimeoutMs({ XERXES_HUB_MODEL_TIMEOUT_MS: 'soon' })).toBe(HUB_MODEL_TIMEOUT_MS)
  expect(hubModelTimeoutMs({ XERXES_HUB_MODEL_TIMEOUT_MS: '-5' })).toBe(HUB_MODEL_TIMEOUT_MS)
})