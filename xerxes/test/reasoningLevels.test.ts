// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  catalogReasoningLevels,
  clampEffort,
  fallbackReasoningLevels,
  resolveEffort,
  selectableEfforts,
} from '../src/llms/reasoningLevels.js'

test('catalog levels mirror pi-ai getSupportedThinkingLevels per model', () => {
  // gpt-5: full ladder minimal..high, xhigh/max explicitly disabled.
  const gpt5 = catalogReasoningLevels('openai/gpt-5', 'openai')
  expect(gpt5?.shape).toBe('effort')
  expect(gpt5?.source).toBe('provider')
  expect(gpt5?.levels.map(level => level.effort)).toEqual(['minimal', 'low', 'medium', 'high'])
  expect(gpt5?.defaultEffort).toBe('medium')

  // gpt-5-pro: only high — the null rungs are honored, not just xhigh/max.
  const pro = catalogReasoningLevels('openai/gpt-5-pro', 'openai')
  expect(pro?.levels.map(level => level.effort)).toEqual(['high'])
  expect(pro?.defaultEffort).toBe('high')

  // gpt-5.6-luna: max exists only because the map names it.
  const luna = catalogReasoningLevels('openai/gpt-5.6-luna', 'openai')
  expect(luna?.levels.map(level => level.effort)).toEqual(['low', 'medium', 'high', 'xhigh', 'max'])
})

test('non-reasoning models offer nothing and unknown models defer to the fallback', () => {
  const chat = catalogReasoningLevels('openai/gpt-5-chat-latest', 'openai')
  // reasoning: false in the catalog → nothing selectable (pi returns ["off"]).
  expect(chat?.shape).toBe('inherent')
  expect(selectableEfforts(chat!)).toEqual([])

  expect(catalogReasoningLevels('openai/definitely-not-a-model', 'openai')).toBeUndefined()
  expect(catalogReasoningLevels('', 'openai')).toBeUndefined()
})

test('clampEffort lands on the nearest offered rung, upward first', () => {
  const gpt5 = catalogReasoningLevels('openai/gpt-5', 'openai')!
  // xhigh is not offered; clamp rises to it, fails, then falls to high.
  expect(clampEffort(gpt5, 'xhigh')).toBe('high')
  expect(clampEffort(gpt5, 'high')).toBe('high')
  // On gpt-5-pro (high only), low clamps upward to high.
  const pro = catalogReasoningLevels('openai/gpt-5-pro', 'openai')!
  expect(clampEffort(pro, 'low')).toBe('high')
  expect(clampEffort(pro, 'minimal')).toBe('high')
  // Unknown words stay usage errors.
  expect(clampEffort(gpt5, 'galaxy-brain')).toBeUndefined()
  // Inherent and fallback-table sets never clamp.
  expect(clampEffort(catalogReasoningLevels('openai/gpt-5-chat-latest', 'openai')!, 'low')).toBeUndefined()
  expect(clampEffort(fallbackReasoningLevels('zhipu'), 'low')).toBeUndefined()
})

test('resolveEffort stays strict; clamping is the explicit second step', () => {
  const gpt5 = catalogReasoningLevels('openai/gpt-5', 'openai')!
  expect(resolveEffort(gpt5, 'xhigh')).toBeUndefined()
  expect(resolveEffort(gpt5, 'HIGH')).toBe('high')
  // gpt-5's map marks off: null — the model cannot disable thinking, so off
  // neither resolves nor appears among the selectable efforts.
  expect(gpt5.canDisable).toBe(false)
  expect(selectableEfforts(gpt5)).toEqual(['minimal', 'low', 'medium', 'high'])
  expect(resolveEffort(gpt5, 'off')).toBeUndefined()
  // gpt-5.1 maps off to 'none': disabling is a real choice there.
  const gpt51 = catalogReasoningLevels('openai/gpt-5.1', 'openai')!
  expect(gpt51.canDisable).toBe(true)
  expect(selectableEfforts(gpt51)[0]).toBe('off')
  expect(resolveEffort(gpt51, 'off')).toBe('off')
})
