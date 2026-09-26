// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, expect, test } from 'bun:test'

import { reportedReasoning } from '../src/daemon/modelDiscovery.js'
import { reasoningFromModelsDev } from '../src/llms/modelsDev.js'
import {
  catalogReasoningLevels,
  fallbackReasoningLevels,
  liveReasoningLevels,
  providerReasoningLevels,
  resolveEffort,
  selectableEfforts,
} from '../src/llms/reasoningLevels.js'
import { clearModelsDev, seedModelsDev } from './fixtures/modelsDev.js'

beforeEach(() => seedModelsDev())
afterEach(() => clearModelsDev())

test('levels are whatever models.dev reports for the model — never a built-in ladder', () => {
  const gpt5 = catalogReasoningLevels('openai/gpt-5', 'openai')!
  expect(selectableEfforts(gpt5)).toEqual(['minimal', 'low', 'medium', 'high'])
  // Efforts without `none` or a toggle: thinking is always on, so no off row.
  expect(gpt5.canDisable).toBe(false)
  // `none` among the values is the off switch.
  const gpt51 = catalogReasoningLevels('gpt-5.1', 'openai')!
  expect(selectableEfforts(gpt51)).toEqual(['off', 'low', 'medium', 'high'])
  // A token-budget model (Claude) reasons without levels: a plain switch.
  expect(selectableEfforts(catalogReasoningLevels('claude-haiku-4-5', 'anthropic')!)).toEqual(['off', 'on'])
  // A model that does not reason offers nothing.
  expect(selectableEfforts(catalogReasoningLevels('gpt-4o', 'openai')!)).toEqual([])
  // Not described anywhere: nothing, not a guess.
  expect(catalogReasoningLevels('openai/definitely-not-a-model', 'openai')).toBeUndefined()
  expect(catalogReasoningLevels('', 'openai')).toBeUndefined()
})

test('without models.dev the answer is "unreported", which offers nothing', () => {
  clearModelsDev()
  expect(catalogReasoningLevels('gpt-5', 'openai')).toBeUndefined()
  const none = fallbackReasoningLevels('openai')
  expect(selectableEfforts(none)).toEqual([])
  expect(none.provenance).toBe('provider_fallback')
})

test('Kimi Code\'s rules: effort values, `none` = off, toggle/budget = switchable, efforts alone = always on', () => {
  expect(reasoningFromModelsDev(true, [{ type: 'effort', values: ['low', 'high', 'max'] }])).toEqual({ supported: true, canDisable: false, efforts: ['low', 'high', 'max'] })
  expect(reasoningFromModelsDev(true, [{ type: 'toggle' }, { type: 'effort', values: ['low', 'high'] }])).toMatchObject({ canDisable: true, efforts: ['low', 'high'] })
  expect(reasoningFromModelsDev(true, [{ type: 'effort', values: ['none', 'low'] }])).toMatchObject({ canDisable: true, efforts: ['low'], offEffort: 'none' })
  expect(reasoningFromModelsDev(true, [{ type: 'budget_tokens', min: 1024 }])).toMatchObject({ canDisable: true, efforts: [] })
  expect(reasoningFromModelsDev(false, undefined)).toEqual({ supported: false, canDisable: false, efforts: [] })
  expect(reasoningFromModelsDev(undefined, undefined)).toBeUndefined()
})

test('a provider\'s own /models fields win: Kimi\'s think_efforts and supports_thinking_type', () => {
  // Captured from api.kimi.com/coding/v1/models on 2026-09-24.
  const k3 = reportedReasoning({ id: 'k3', context_length: 1048576, supports_reasoning: true, supports_thinking_type: 'only', think_efforts: { support: true, valid_efforts: ['low', 'high', 'max'], default_effort: 'max' } })
  expect(k3).toEqual({ supported: true, canDisable: false, efforts: ['low', 'high', 'max'], defaultEffort: 'max' })
  const set = liveReasoningLevels(k3, 'provider')!
  expect(selectableEfforts(set)).toEqual(['low', 'high', 'max'])
  expect(set.defaultEffort).toBe('max')
  expect(set.provenance).toBe('provider_reported')
  expect(reportedReasoning({ id: 'x', supports_thinking_type: 'both' })).toMatchObject({ supported: true, canDisable: true, efforts: [] })
  expect(reportedReasoning({ id: 'x', supports_thinking_type: 'no', supports_reasoning: true })).toEqual({ supported: false, canDisable: false, efforts: [] })
  // OpenRouter states support through its supported parameters.
  expect(reportedReasoning({ id: 'x', supported_parameters: ['tools', 'reasoning'] })).toMatchObject({ supported: true, canDisable: true })
  // A bare id says nothing.
  expect(reportedReasoning({ id: 'glm-4.6', owned_by: 'z-ai' })).toBeUndefined()
})

test('resolveEffort refuses only against reported levels; an unreported model takes what was set', () => {
  const gpt5 = catalogReasoningLevels('openai/gpt-5', 'openai')!
  expect(resolveEffort(gpt5, 'HIGH')).toBe('high')
  expect(resolveEffort(gpt5, 'xhigh')).toBeUndefined()
  expect(resolveEffort(gpt5, 'off')).toBeUndefined()
  expect(resolveEffort(catalogReasoningLevels('gpt-5.1', 'openai')!, 'off')).toBe('off')
  expect(resolveEffort(fallbackReasoningLevels('custom'), 'high')).toBe('high')
})

test('provenance says where the levels came from', () => {
  expect(catalogReasoningLevels('gpt-5', 'openai')?.provenance).toBe('bundled_catalog')
  expect(fallbackReasoningLevels('kimi').provenance).toBe('provider_fallback')
  expect(providerReasoningLevels([{ effort: 'high' }], 'high').provenance).toBe('provider_reported')
})
