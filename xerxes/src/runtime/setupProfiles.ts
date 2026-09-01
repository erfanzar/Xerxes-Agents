// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type SetupProfileName = 'developer' | 'personal' | 'operator' | 'framework' | 'minimal'

export interface SetupProfile {
  readonly name: SetupProfileName
  readonly label: string
  readonly answers: Readonly<Record<string, unknown>>
}

export const SETUP_PROFILES: Readonly<Record<SetupProfileName, SetupProfile>> = Object.freeze({
  developer: {
    name: 'developer',
    label: 'iterative coding assistant with manual approvals and tool auditing',
    answers: {
      provider: 'anthropic',
      model: 'claude-sonnet-4',
      permission_mode: 'manual',
      enable_voice: 'n',
      messaging_platform: 'none',
    },
  },
  personal: {
    name: 'personal',
    label: 'accept-all local helper with voice and chat history',
    answers: {
      provider: 'openai',
      model: 'gpt-4o',
      permission_mode: 'accept-all',
      enable_voice: 'y',
      messaging_platform: 'none',
    },
  },
  operator: {
    name: 'operator',
    label: 'automation daemon with durable scheduler and strict audit',
    answers: {
      provider: 'anthropic',
      model: 'claude-opus-4-6',
      permission_mode: 'manual',
      enable_voice: 'n',
      messaging_platform: 'telegram',
    },
  },
  framework: {
    name: 'framework',
    label: 'minimal provider wiring for embedding in another application',
    answers: {
      provider: 'openai',
      model: 'gpt-4o-mini',
      permission_mode: 'manual',
      enable_voice: 'n',
      messaging_platform: 'none',
    },
  },
  minimal: {
    name: 'minimal',
    label: 'provider and permissions only; everything else disabled',
    answers: {
      provider: 'anthropic',
      model: 'claude-haiku-4',
      permission_mode: 'manual',
      enable_voice: 'n',
      messaging_platform: 'none',
    },
  },
})

export function isSetupProfileName(value: unknown): value is SetupProfileName {
  return typeof value === 'string' && value in SETUP_PROFILES
}
