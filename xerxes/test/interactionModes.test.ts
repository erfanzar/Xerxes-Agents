// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  INTERACTION_MODES,
  MODE_ALIASES,
  agentNameForMode,
  modeSwitchHint,
  normalizeInteractionMode,
  resolveInteractionMode,
} from '../src/runtime/interactionModes.js'

test('interaction modes normalize every supported alias and retain plan-mode precedence', () => {
  expect(INTERACTION_MODES).toEqual(['code', 'researcher', 'plan', 'objective'])
  expect(MODE_ALIASES['goal-runner']).toBe('objective')
  expect(normalizeInteractionMode('coding')).toBe('code')
  expect(normalizeInteractionMode('research')).toBe('researcher')
  expect(normalizeInteractionMode('planner')).toBe('plan')
  expect(normalizeInteractionMode(' goals ')).toBe('objective')
  expect(normalizeInteractionMode('anything')).toBe('code')
  expect(normalizeInteractionMode('objective', true)).toBe('plan')
  expect(resolveInteractionMode('objective')).toBe('objective')
  expect(resolveInteractionMode('anything')).toBeUndefined()
})

test('interaction modes resolve to matching built-in agents', () => {
  expect(agentNameForMode('code')).toBe('coder')
  expect(agentNameForMode('research')).toBe('researcher')
  expect(agentNameForMode('planner')).toBe('planner')
  expect(agentNameForMode('goal')).toBe('objective')
})

test('model-facing hints retain objective completion guidance and safe transitions', () => {
  const objectiveHint = modeSwitchHint('objective')

  expect(objectiveHint).toContain('acceptance criteria')
  expect(objectiveHint).toContain('Do not final-answer')
  expect(objectiveHint).toContain('SetInteractionModeTool(mode="code")')
  expect(modeSwitchHint('plan')).toContain('Produce a plan only')
  expect(modeSwitchHint('research')).toContain('Gather evidence and answer with citations')
  expect(modeSwitchHint('unknown')).toContain('Use code mode for normal implementation')
})
