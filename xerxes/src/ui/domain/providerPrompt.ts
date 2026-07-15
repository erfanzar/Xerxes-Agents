// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ClarifyReq } from '../types.js'

const CANCEL = 'cancel'

export type ProviderPrompt = ClarifyReq & { source: 'provider' }

export const providerPromptCancelAnswer = (prompt: ClarifyReq): string =>
  prompt.choices?.find(choice => choice.trim().toLowerCase() === CANCEL) ?? ''

export const providerPromptChoices = (prompt: ClarifyReq): string[] =>
  (prompt.choices ?? []).filter(choice => choice.trim().toLowerCase() !== CANCEL)

export const providerPromptTitle = (questionId?: string): string => {
  switch (questionId) {
    case 'action':
      return 'Provider profiles'
    case 'field':
    case 'profile':
    case 'value':
      return 'Edit provider'
    case 'confirm':
      return 'Remove provider'
    default:
      return 'Add provider'
  }
}

export const isProviderPrompt = (prompt: ClarifyReq | null): prompt is ProviderPrompt => prompt?.source === 'provider'

export const providerPromptIsSecret = (prompt: ClarifyReq): boolean => prompt.questionId === 'api_key'
