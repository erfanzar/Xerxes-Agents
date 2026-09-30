// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { parseAgentIntelligenceConfig } from '../src/agents/intelligence.js'
import { draftOf, draftProblem, settingsOf } from '../src/desktop/renderer/AgentIntelligenceCard.js'

test('stored tiers become an editable draft, bare model names on the active provider', () => {
  const draft = draftOf({ default: 'balanced', light: 'gpt-6-luna', balanced: { model: 'sonnet', provider_profile: 'claude-code', reasoning_effort: 'high' } })
  expect(draft.default).toBe('balanced')
  expect(draft.tiers.light).toEqual({ provider_profile: '', model: 'gpt-6-luna', reasoning_effort: '' })
  expect(draft.tiers.balanced).toEqual({ provider_profile: 'claude-code', model: 'sonnet', reasoning_effort: 'high' })
  expect(draft.tiers.smart.model).toBe('')
  expect(draftOf(undefined).default).toBe('inherit')
})

test('a draft saves in the shape the daemon validates: any provider, model and effort per tier', () => {
  const draft = draftOf({})
  const edited = {
    default: 'light' as const,
    tiers: {
      ...draft.tiers,
      light: { provider_profile: 'codex', model: ' gpt-6-astra ', reasoning_effort: 'low' },
      smart: { provider_profile: 'claude-code', model: 'opus', reasoning_effort: '' },
    },
  }
  const wire = settingsOf(edited)
  expect(wire).toEqual({
    default: 'light',
    light: { model: 'gpt-6-astra', provider_profile: 'codex', reasoning_effort: 'low' },
    smart: { model: 'opus', provider_profile: 'claude-code' },
  })
  // The daemon's own parser accepts exactly what the card sends.
  expect(parseAgentIntelligenceConfig(wire)).toMatchObject({ default: 'light', light: { model: 'gpt-6-astra' } })
})

test('a default tier without a model is caught before saving', () => {
  const draft = { ...draftOf({}), default: 'smart' as const }
  expect(draftProblem(draft)).toContain('Set a model for smart')
  expect(draftProblem(draftOf({}))).toBeUndefined()
})
