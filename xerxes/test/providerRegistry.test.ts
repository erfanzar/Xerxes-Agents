// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  calcCost,
  detectProvider,
  getContextLimit,
  listAllModels,
  providerDefaultHeaders,
  providerModel,
  resolveProvider,
} from '../src/llms/providerRegistry.js'

test('provider routing preserves explicit prefixes and Kimi Code overrides', () => {
  expect(detectProvider('anthropic/claude-sonnet-4-6')).toBe('anthropic')
  expect(detectProvider('kimi-for-coding')).toBe('kimi-code')
  expect(resolveProvider('kimi/kimi-for-coding')).toBe('kimi-code')
  expect(resolveProvider('gpt-4o', { base_url: 'https://api.kimi.com/coding/v1' })).toBe('kimi-code')
  expect(providerModel('openrouter/anthropic/claude-sonnet-4.5', 'openrouter')).toBe('anthropic/claude-sonnet-4.5')
})

test('costs, context windows, model catalog, and Kimi headers match registry behavior', () => {
  expect(calcCost('gpt-4o', 1_000_000, 1_000_000)).toBe(12.5)
  expect(getContextLimit('claude-opus-4-6')).toBe(1_000_000)
  expect(getContextLimit('moonshot-v1-8k')).toBe(8_192)
  expect(listAllModels().lmstudio).toBeUndefined()
  expect(providerDefaultHeaders('kimi-code')).toMatchObject({ 'User-Agent': 'claude-code/1.0.0' })
})
