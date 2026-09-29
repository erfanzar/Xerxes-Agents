// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { renderDelegationModels } from '../src/agents/delegationModels.js'
import type { ProviderProfile } from '../src/bridge/profiles.js'

const profile = (name: string, model: string, capabilities: ProviderProfile['model_capabilities'] = {}): ProviderProfile => ({
  name, provider: name, base_url: 'https://example.test', api_key: 'k', model, model_capabilities: capabilities, sampling: {},
})

test('delegated-agent models come from what each provider reported, with published prices', () => {
  const text = renderDelegationModels([
    profile('claude-code', 'claude-code/opus', { 'claude-code/opus': {}, 'claude-code/sonnet': {}, 'claude-code/haiku': {} }),
    profile('cheap', 'tiny-1', { 'tiny-1': { cost: { input: 0.05, output: 0.4 } }, 'big-2': { cost: { input: 3, output: 15 } } }),
    profile('plain', 'only-model'),
  ], 'claude-code')
  expect(text).toStartWith('# Models for delegated agents')
  expect(text).toContain('- claude-code (this conversation): claude-code/opus, claude-code/sonnet, claude-code/haiku')
  expect(text).toContain('- cheap: tiny-1 ($0.05/$0.4), big-2 ($3/$15)')
  expect(text).toContain('- plain: only-model')
})

test('a large catalog is summarized and pointed at list_available_models', () => {
  const many = Object.fromEntries(Array.from({ length: 460 }, (_, i) => [`vendor/model-${i}`, {}]))
  const text = renderDelegationModels([profile('openrouter', 'vendor/model-3', many)])
  expect(text).toContain('- openrouter: 460 models; saved model vendor/model-3. Search them with list_available_models.')
  expect(text).not.toContain('vendor/model-100')
  expect(renderDelegationModels([profile('empty', '')])).toBe('')
})
