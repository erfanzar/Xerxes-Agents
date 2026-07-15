// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import type { ClarifyReq } from '../types.js'
import {
  isProviderPrompt,
  providerPromptCancelAnswer,
  providerPromptChoices,
  providerPromptIsSecret,
  providerPromptTitle
} from './providerPrompt.js'

const providerPrompt: ClarifyReq = {
  allowFreeform: false,
  choices: ['local (kimi @ localhost)', '+ Add new profile…', 'Cancel'],
  question: 'Provider profiles — pick a profile to switch, or choose an action:',
  questionId: 'action',
  requestId: 'provider-request:action',
  source: 'provider'
}

describe('provider prompt presentation', () => {
  it('keeps cancellation on Escape instead of presenting it as a profile row', () => {
    expect(providerPromptChoices(providerPrompt)).toEqual(['local (kimi @ localhost)', '+ Add new profile…'])
    expect(providerPromptCancelAnswer(providerPrompt)).toBe('Cancel')
  })

  it('labels provider stages without exposing the daemon clarify primitive', () => {
    expect(providerPromptTitle('action')).toBe('Provider profiles')
    expect(providerPromptTitle('model')).toBe('Add provider')
    expect(providerPromptTitle('field')).toBe('Edit provider')
    expect(providerPromptTitle('confirm')).toBe('Remove provider')
  })

  it('distinguishes provider prompts and credential entry', () => {
    expect(isProviderPrompt(providerPrompt)).toBe(true)
    expect(providerPromptIsSecret({ ...providerPrompt, questionId: 'api_key' })).toBe(true)
    expect(isProviderPrompt({ ...providerPrompt, source: 'agent' })).toBe(false)
  })
})
