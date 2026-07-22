// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ConfigurationError } from '../src/core/errors.js'
import {
  calcCost,
  detectProvider,
  getContextLimit,
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

test('unrecognized explicit prefixes are rejected instead of silently routed to OpenAI', () => {
  expect(() => detectProvider('missing-provider/gpt-4o')).toThrow(ConfigurationError)
  expect(() => detectProvider('missing-provider/gpt-4o')).toThrow("unknown provider prefix 'missing-provider'")
  expect(() => resolveProvider('missing-provider/gpt-4o')).toThrow(ConfigurationError)
  expect(() => getContextLimit('missing-provider/gpt-4o')).toThrow(ConfigurationError)
  // A recognized explicit provider override still routes before prefix checks.
  expect(resolveProvider('missing-provider/gpt-4o', { provider: 'openrouter' })).toBe('openrouter')
  // Aliased and bare models keep their existing behavior.
  expect(detectProvider('claude_code/sonnet')).toBe('claude-code')
  expect(detectProvider('unprefixed-model')).toBe('openai')
})

test('explicit provider overrides win over model-prefix routing for every known provider', () => {
  expect(resolveProvider('anthropic/claude-sonnet-4.5', { provider: 'openrouter' })).toBe('openrouter')
  expect(resolveProvider('gpt-4o', { provider: 'anthropic' })).toBe('anthropic')
  expect(resolveProvider('claude-sonnet-4-6', { provider: 'openai' })).toBe('openai')
  expect(resolveProvider('llama3.3', { provider_type: 'openrouter' })).toBe('openrouter')
  expect(resolveProvider('sonnet', { provider: 'claude_code' })).toBe('claude-code')
  expect(resolveProvider('gpt-4o', { provider: 'not-a-provider' })).toBe('openai')
})

test('costs, context windows, and Kimi headers match registry behavior', () => {
  expect(calcCost('gpt-4o', 1_000_000, 1_000_000)).toBe(12.5)
  expect(getContextLimit('claude-opus-4-6')).toBe(1_000_000)
  expect(getContextLimit('moonshot-v1-8k')).toBe(8_192)
  expect(getContextLimit('k3', { provider: 'kimi-code' })).toBe(262_144)
  expect(getContextLimit('unknown-model', { provider: 'custom' })).toBe(0)
  expect(providerDefaultHeaders('kimi-code')).toMatchObject({ 'User-Agent': 'claude-code/1.0.0' })
})
