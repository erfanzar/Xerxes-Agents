// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Named prompt-prefix verbosity levels used by native runtime composition. */
export const PromptProfile = {
  FULL: 'full',
  COMPACT: 'compact',
  MINIMAL: 'minimal',
  NONE: 'none',
} as const

export type PromptProfile = (typeof PromptProfile)[keyof typeof PromptProfile]

/** Immutable inclusion gates and size limits for a prompt-prefix profile. */
export interface PromptProfileConfig {
  readonly includeBootstrap: boolean
  readonly includeEnabledSkills: boolean
  readonly includeGitInfo: boolean
  readonly includeGuardrails: boolean
  readonly includeRelevantMemories: boolean
  readonly includeRepoMap: boolean
  readonly includeRuntimeInfo: boolean
  readonly includeSandboxInfo: boolean
  readonly includeSkillsIndex: boolean
  readonly includeToolsList: boolean
  readonly includeUserProfile: boolean
  readonly includeWorkspaceInfo: boolean
  readonly maxMemoriesInjected: number
  readonly maxSkillInstructionsLength: number | undefined
  readonly maxToolsListed: number | undefined
  readonly profile: PromptProfile
}

/** Resolve one canonical, immutable configuration for a named profile. */
export function getPromptProfileConfig(profile: PromptProfile | string): PromptProfileConfig {
  switch (normalizeProfile(profile)) {
    case PromptProfile.FULL:
      return fullConfig()
    case PromptProfile.COMPACT:
      return compactConfig()
    case PromptProfile.MINIMAL:
      return minimalConfig()
    case PromptProfile.NONE:
      return noneConfig()
  }
}

/**
 * Normalize an optional named profile or retain an explicit configuration.
 *
 * Missing profile selection deliberately means FULL, matching the native
 * runtime default rather than silently applying a compact subagent policy.
 */
export function resolvePromptProfileConfig(
  profile: PromptProfile | PromptProfileConfig | string | undefined,
): PromptProfileConfig {
  if (profile === undefined) return getPromptProfileConfig(PromptProfile.FULL)
  if (typeof profile === 'string') return getPromptProfileConfig(profile)
  return freezeConfig(profile)
}

function fullConfig(): PromptProfileConfig {
  return freezeConfig({
    profile: PromptProfile.FULL,
    includeRuntimeInfo: true,
    includeWorkspaceInfo: true,
    includeSandboxInfo: true,
    includeSkillsIndex: true,
    includeEnabledSkills: true,
    includeToolsList: true,
    includeGuardrails: true,
    includeBootstrap: true,
    includeRelevantMemories: true,
    includeUserProfile: true,
    includeRepoMap: true,
    includeGitInfo: true,
    maxSkillInstructionsLength: undefined,
    maxToolsListed: undefined,
    maxMemoriesInjected: 5,
  })
}

function compactConfig(): PromptProfileConfig {
  return freezeConfig({
    ...fullConfig(),
    profile: PromptProfile.COMPACT,
    includeWorkspaceInfo: false,
    includeBootstrap: false,
    includeRepoMap: false,
    maxSkillInstructionsLength: 500,
    maxToolsListed: 20,
  })
}

function minimalConfig(): PromptProfileConfig {
  return freezeConfig({
    ...fullConfig(),
    profile: PromptProfile.MINIMAL,
    includeRuntimeInfo: false,
    includeWorkspaceInfo: false,
    includeSkillsIndex: false,
    includeEnabledSkills: false,
    includeBootstrap: false,
    includeRelevantMemories: false,
    includeUserProfile: false,
    includeRepoMap: false,
    maxToolsListed: 10,
  })
}

function noneConfig(): PromptProfileConfig {
  return freezeConfig({
    ...fullConfig(),
    profile: PromptProfile.NONE,
    includeRuntimeInfo: false,
    includeWorkspaceInfo: false,
    includeSandboxInfo: false,
    includeSkillsIndex: false,
    includeEnabledSkills: false,
    includeToolsList: false,
    includeGuardrails: false,
    includeBootstrap: false,
    includeRelevantMemories: false,
    includeUserProfile: false,
    includeRepoMap: false,
  })
}

function normalizeProfile(profile: PromptProfile | string): PromptProfile {
  const normalized = profile.trim().toLowerCase()
  if (Object.values(PromptProfile).includes(normalized as PromptProfile)) {
    return normalized as PromptProfile
  }
  throw new RangeError('Unknown prompt profile: ' + profile)
}

function freezeConfig(config: PromptProfileConfig): PromptProfileConfig {
  const profile = normalizeProfile(config.profile)
  if (!Number.isSafeInteger(config.maxMemoriesInjected) || config.maxMemoriesInjected < 0) {
    throw new RangeError('maxMemoriesInjected must be a non-negative safe integer')
  }
  if (
    config.maxSkillInstructionsLength !== undefined
    && (!Number.isSafeInteger(config.maxSkillInstructionsLength) || config.maxSkillInstructionsLength < 0)
  ) {
    throw new RangeError('maxSkillInstructionsLength must be a non-negative safe integer when specified')
  }
  if (
    config.maxToolsListed !== undefined
    && (!Number.isSafeInteger(config.maxToolsListed) || config.maxToolsListed < 0)
  ) {
    throw new RangeError('maxToolsListed must be a non-negative safe integer when specified')
  }
  return Object.freeze({ ...config, profile })
}
