// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type SetupProfileName = 'developer' | 'personal' | 'operator' | 'framework' | 'minimal'

export interface SetupProfile {
  readonly name: SetupProfileName
  readonly label: string
  readonly answers: Readonly<Record<string, unknown>>
}

/**
 * Behaviour presets: permissions, voice, messaging. They name no provider or
 * model — those are the user's, and the model comes from what the provider
 * reports (see runSetupCommand).
 */
export const SETUP_PROFILES: Readonly<Record<SetupProfileName, SetupProfile>> = Object.freeze({
  developer: {
    name: 'developer',
    label: 'iterative coding assistant with manual approvals and tool auditing',
    answers: {
      permission_mode: 'manual',
      enable_voice: 'n',
      messaging_platform: 'none',
    },
  },
  personal: {
    name: 'personal',
    label: 'accept-all local helper with voice and chat history',
    answers: {
      permission_mode: 'accept-all',
      enable_voice: 'y',
      messaging_platform: 'none',
    },
  },
  operator: {
    name: 'operator',
    label: 'automation daemon with durable scheduler and strict audit',
    answers: {
      permission_mode: 'manual',
      enable_voice: 'n',
      messaging_platform: 'telegram',
    },
  },
  framework: {
    name: 'framework',
    label: 'minimal provider wiring for embedding in another application',
    answers: {
      permission_mode: 'manual',
      enable_voice: 'n',
      messaging_platform: 'none',
    },
  },
  minimal: {
    name: 'minimal',
    label: 'permissions only; everything else disabled',
    answers: {
      permission_mode: 'manual',
      enable_voice: 'n',
      messaging_platform: 'none',
    },
  },
})

export function isSetupProfileName(value: unknown): value is SetupProfileName {
  return typeof value === 'string' && value in SETUP_PROFILES
}
