// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'

export const AGENT_INTELLIGENCE_LEVELS = ['light', 'balanced', 'smart'] as const
export type AgentIntelligence = typeof AGENT_INTELLIGENCE_LEVELS[number]
export interface AgentTierSettings {
  readonly model: string
  readonly provider_profile?: string
  readonly reasoning_effort?: string
}
export interface AgentIntelligenceConfig {
  readonly default?: AgentIntelligence | 'inherit'
  readonly light?: string | AgentTierSettings
  readonly balanced?: string | AgentTierSettings
  readonly smart?: string | AgentTierSettings
}

export function parseAgentIntelligence(value: unknown): AgentIntelligence | undefined {
  if (value === undefined) return undefined
  if (AGENT_INTELLIGENCE_LEVELS.some(level => level === value)) return value as AgentIntelligence
  throw new ValidationError('intelligence', 'must be light, balanced, or smart', value)
}

/** Validate user-owned mappings without guessing provider availability or model quality. */
export function parseAgentIntelligenceConfig(value: unknown): AgentIntelligenceConfig {
  if (value === undefined) return Object.freeze({})
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ValidationError('agent_intelligence', 'must be an object with default, light, balanced, or smart', value)
  }
  const config: { default?: AgentIntelligence | 'inherit'; light?: string | AgentTierSettings; balanced?: string | AgentTierSettings; smart?: string | AgentTierSettings } = {}
  for (const [key, entry] of Object.entries(value)) {
    if (key === 'default') {
      const level = entry === 'inherit' ? 'inherit' : parseAgentIntelligence(entry)
      if (level === undefined) throw new ValidationError('agent_intelligence.default', 'must be inherit, light, balanced, or smart', entry)
      config.default = level
    } else if (AGENT_INTELLIGENCE_LEVELS.some(level => level === key)) {
      if (typeof entry === 'string' && entry.trim()) config[key as AgentIntelligence] = entry.trim()
      else if (entry && typeof entry === 'object' && !Array.isArray(entry)) {
        const fields = entry as Record<string, unknown>
        if (Object.keys(fields).some(key => !['model', 'provider_profile', 'reasoning_effort'].includes(key)) || typeof fields.model !== 'string' || !fields.model.trim()) throw new ValidationError(`agent_intelligence.${key}`, 'requires model and optional provider_profile/reasoning_effort', entry)
        for (const field of ['provider_profile', 'reasoning_effort']) if (fields[field] !== undefined && (typeof fields[field] !== 'string' || !(fields[field] as string).trim())) throw new ValidationError(`agent_intelligence.${key}.${field}`, 'must be a non-empty string', fields[field])
        config[key as AgentIntelligence] = Object.freeze({ model: fields.model.trim(),
          ...(typeof fields.provider_profile === 'string' ? { provider_profile: fields.provider_profile.trim() } : {}),
          ...(typeof fields.reasoning_effort === 'string' ? { reasoning_effort: fields.reasoning_effort.trim() } : {}),
        })
      } else throw new ValidationError(`agent_intelligence.${key}`, 'must be a model name or tier settings', entry)
    } else {
      throw new ValidationError(`agent_intelligence.${key}`, 'unknown setting', entry)
    }
  }
  if (config.default && config.default !== 'inherit' && !config[config.default]) {
    throw new ValidationError('agent_intelligence.default', 'requires a model mapping for its tier', config.default)
  }
  return Object.freeze(config)
}

export function resolveAgentIntelligenceModel(config: AgentIntelligenceConfig, intelligence?: AgentIntelligence, model?: string): string | undefined {
  return resolveAgentIntelligenceSettings(config, intelligence, model)?.model
}

/**
 * The tier a request falls under, ignoring any explicit model: the named tier,
 * else the configured default. Undefined for `inherit` or an unmapped tier.
 * The data boundary uses it to stop an explicit cloud model from overriding a
 * local tier; routing itself still lets an explicit model win.
 */
export function resolveAgentIntelligenceTier(config: AgentIntelligenceConfig, intelligence?: AgentIntelligence): AgentTierSettings | undefined {
  const level = intelligence ?? config.default
  if (!level || level === 'inherit') return undefined
  const selected = config[level]
  if (!selected) return undefined
  return typeof selected === 'string' ? { model: selected } : selected
}

export function resolveAgentIntelligenceSettings(config: AgentIntelligenceConfig, intelligence?: AgentIntelligence, model?: string): AgentTierSettings | undefined {
  if (model?.trim()) return { model: model.trim() }
  const level = intelligence ?? config.default
  if (!level || level === 'inherit') return undefined
  const selected = config[level]
  if (!selected) throw new ValidationError('intelligence', `configure runtime.agent_intelligence.${level} with a model name, or pass model explicitly`, level)
  return typeof selected === 'string' ? { model: selected } : selected
}
