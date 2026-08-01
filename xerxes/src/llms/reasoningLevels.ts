// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Which reasoning efforts a given model actually accepts.
 *
 * There is no single answer: on the Codex backend alone the set ranges from
 * four efforts to six depending on the model, and the default varies between
 * `low`, `medium`, and `high`. So the levels are asked for — live, per model —
 * wherever a provider can answer, and only fall back to a per-provider table
 * where no such endpoint exists.
 */

import type { ProviderName } from './providerRegistry.js'

/** One selectable effort, with the provider's own description when it gives one. */
export interface ReasoningLevel {
  readonly description?: string
  readonly effort: string
}

export interface ReasoningLevelSet {
  /** Effort applied when the user has not chosen one. */
  readonly defaultEffort: string | undefined
  readonly levels: readonly ReasoningLevel[]
  /** `provider` when the model itself reported these, `fallback` otherwise. */
  readonly source: 'fallback' | 'provider'
}

/**
 * Turning reasoning off is a Xerxes-side choice, not a provider effort: it
 * suppresses the request field entirely rather than sending `effort: 'off'`.
 */
export const REASONING_OFF = 'off'

/**
 * Efforts to offer for providers that publish no capability endpoint.
 *
 * Deliberately per provider rather than one global list — Anthropic's
 * budget-based thinking and OpenAI's effort scale are different vocabularies,
 * and collapsing them into a single set is what produced a fixed four-item
 * menu that was wrong for nearly every model.
 */
const FALLBACK_LEVELS: Partial<Record<ProviderName, readonly ReasoningLevel[]>> = {
  anthropic: [
    { effort: 'low', description: 'Brief thinking budget' },
    { effort: 'medium', description: 'Balanced thinking budget' },
    { effort: 'high', description: 'Extended thinking budget' },
  ],
  'claude-code': [
    { effort: 'low', description: 'Brief thinking budget' },
    { effort: 'medium', description: 'Balanced thinking budget' },
    { effort: 'high', description: 'Extended thinking budget' },
  ],
  openai: [
    { effort: 'low', description: 'Fast responses with lighter reasoning' },
    { effort: 'medium', description: 'Balances speed and reasoning depth' },
    { effort: 'high', description: 'Greater reasoning depth for complex problems' },
  ],
}

const GENERIC_FALLBACK: readonly ReasoningLevel[] = [
  { effort: 'low', description: 'Fast responses with lighter reasoning' },
  { effort: 'medium', description: 'Balances speed and reasoning depth' },
  { effort: 'high', description: 'Greater reasoning depth for complex problems' },
]

/** Levels to offer when the provider cannot be asked. */
export function fallbackReasoningLevels(providerName: ProviderName | undefined): ReasoningLevelSet {
  const levels = (providerName ? FALLBACK_LEVELS[providerName] : undefined) ?? GENERIC_FALLBACK
  return { defaultEffort: 'medium', levels, source: 'fallback' }
}

/** Wrap a provider-reported list, preserving its order and descriptions. */
export function providerReasoningLevels(
  levels: readonly ReasoningLevel[],
  defaultEffort: string | undefined,
): ReasoningLevelSet {
  return { defaultEffort, levels, source: 'provider' }
}

/** Every value the user may select, including the Xerxes-side off switch. */
export function selectableEfforts(set: ReasoningLevelSet): readonly string[] {
  return [REASONING_OFF, ...set.levels.map(level => level.effort)]
}

/**
 * Validate a requested effort against what the model accepts.
 *
 * Case-insensitive, and returns the provider's own spelling so a request
 * carries the exact token the backend published rather than the user's casing.
 */
export function resolveEffort(set: ReasoningLevelSet, requested: string): string | undefined {
  const clean = requested.trim().toLowerCase()
  if (!clean) return undefined
  if (clean === REASONING_OFF) return REASONING_OFF
  return set.levels.find(level => level.effort.toLowerCase() === clean)?.effort
}
