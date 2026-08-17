// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Which reasoning controls a given model actually exposes.
 *
 * There is no single answer, and the differences are not just which words are
 * accepted — they are differences in kind. Some providers take a graded
 * effort, some take a plain on/off switch, some decide entirely on their own
 * based on which model you picked. Offering a graded `low|medium|high` menu
 * for a provider that only has a switch is the same mistake as hardcoding
 * four levels: it invites a choice that cannot be honored.
 *
 * So levels are asked for — live, per model — wherever a provider publishes a
 * capability endpoint, and otherwise come from a per-provider table describing
 * the shape that provider documents.
 */

import type { ProviderName } from './providerRegistry.js'

/**
 * How a provider lets a caller influence reasoning.
 *
 * - `effort` — a graded scale the caller selects from.
 * - `toggle` — extended thinking is on or off; there are no gradations.
 * - `inherent` — the model decides; picking a different model is the only lever.
 */
export type ReasoningShape = 'effort' | 'inherent' | 'toggle'

/** One selectable effort, with the provider's own description when it gives one. */
export interface ReasoningLevel {
  readonly description?: string
  readonly effort: string
}

export interface ReasoningLevelSet {
  /** Effort applied when the user has not chosen one. */
  readonly defaultEffort: string | undefined
  readonly levels: readonly ReasoningLevel[]
  readonly shape: ReasoningShape
  /** `provider` when the model itself reported these, `fallback` otherwise. */
  readonly source: 'fallback' | 'provider'
}

/**
 * Turning reasoning off is a Xerxes-side choice, not a provider effort: it
 * suppresses the request field entirely rather than sending `effort: 'off'`.
 */
export const REASONING_OFF = 'off'

/**
 * Marker for a toggle-shaped provider's "on" state.
 *
 * It is deliberately not sent as an effort value — {@link isGradedEffort} is
 * what keeps `reasoning_effort: 'on'` off the wire, since no provider
 * documents that as a level.
 */
export const REASONING_ON = 'on'

const GRADED: readonly ReasoningLevel[] = [
  { effort: 'low', description: 'Fast responses with lighter reasoning' },
  { effort: 'medium', description: 'Balances speed and reasoning depth' },
  { effort: 'high', description: 'Greater reasoning depth for complex problems' },
]

const BUDGETED: readonly ReasoningLevel[] = [
  { effort: 'low', description: 'Brief thinking budget' },
  { effort: 'medium', description: 'Balanced thinking budget' },
  { effort: 'high', description: 'Extended thinking budget' },
]

const TOGGLE: readonly ReasoningLevel[] = [
  { effort: REASONING_ON, description: 'Enable extended thinking' },
]

interface FallbackEntry {
  readonly defaultEffort: string | undefined
  readonly levels: readonly ReasoningLevel[]
  readonly shape: ReasoningShape
}

const EFFORT_FALLBACK: FallbackEntry = { defaultEffort: 'medium', levels: GRADED, shape: 'effort' }
const BUDGET_FALLBACK: FallbackEntry = { defaultEffort: 'medium', levels: BUDGETED, shape: 'effort' }
const TOGGLE_FALLBACK: FallbackEntry = { defaultEffort: REASONING_ON, levels: TOGGLE, shape: 'toggle' }
const INHERENT_FALLBACK: FallbackEntry = { defaultEffort: undefined, levels: [], shape: 'inherent' }

/**
 * Reasoning control each provider documents, for hosts that publish no
 * capability endpoint to ask.
 *
 * These entries describe documented behavior rather than behavior Xerxes has
 * measured — unlike the Codex catalog, none of these providers can be probed
 * without a key. Sets built from this table are reported with
 * `source: 'fallback'` so a caller can tell the difference.
 */
const FALLBACK_LEVELS: Partial<Record<ProviderName, FallbackEntry>> = {
  // Budget-based extended thinking; Xerxes maps the rungs onto token budgets.
  anthropic: BUDGET_FALLBACK,
  'claude-code': BUDGET_FALLBACK,
  // Documented effort scales.
  openai: EFFORT_FALLBACK,
  openrouter: EFFORT_FALLBACK,
  // Thinking budget rather than an effort word, mapped from the same rungs.
  gemini: BUDGET_FALLBACK,
  // Switch-shaped: thinking is enabled or disabled, with no gradations.
  zhipu: TOGGLE_FALLBACK,
  qwen: TOGGLE_FALLBACK,
  kimi: TOGGLE_FALLBACK,
  'kimi-code': TOGGLE_FALLBACK,
  minimax: TOGGLE_FALLBACK,
  // Reasoning follows from the chosen model rather than a request field.
  deepseek: INHERENT_FALLBACK,
}

/**
 * Locally hosted and custom endpoints serve whatever model the user loaded, so
 * the control is genuinely unknown. A toggle is the safe assumption: it never
 * offers a gradation the backend cannot honor.
 */
const UNKNOWN_FALLBACK: FallbackEntry = TOGGLE_FALLBACK

/** Levels to offer when the provider cannot be asked. */
export function fallbackReasoningLevels(providerName: ProviderName | undefined): ReasoningLevelSet {
  const entry = (providerName ? FALLBACK_LEVELS[providerName] : undefined) ?? UNKNOWN_FALLBACK
  return {
    defaultEffort: entry.defaultEffort,
    levels: entry.levels,
    shape: entry.shape,
    source: 'fallback',
  }
}

/** Wrap a provider-reported list, preserving its order and descriptions. */
export function providerReasoningLevels(
  levels: readonly ReasoningLevel[],
  defaultEffort: string | undefined,
): ReasoningLevelSet {
  return { defaultEffort, levels, shape: 'effort', source: 'provider' }
}

/** Every value the user may select, including the Xerxes-side off switch. */
export function selectableEfforts(set: ReasoningLevelSet): readonly string[] {
  // An `inherent` provider offers nothing to select: presenting `off` alone
  // would imply reasoning can be disabled, which it cannot.
  if (set.shape === 'inherent') {
    return []
  }
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
  if (set.shape === 'inherent') return undefined
  if (clean === REASONING_OFF) return REASONING_OFF
  return set.levels.find(level => level.effort.toLowerCase() === clean)?.effort
}

/**
 * True when a value is a real effort word worth putting on the wire.
 *
 * `off` and `on` are Xerxes-side switch positions, not levels any provider
 * documents; sending either as `reasoning_effort` would be a field the backend
 * has to ignore at best.
 */
export function isGradedEffort(effort: string | undefined): boolean {
  if (!effort) return false
  const clean = effort.trim().toLowerCase()
  return clean !== REASONING_OFF && clean !== REASONING_ON
}

/** Human-readable note describing how a provider exposes reasoning. */
export function reasoningShapeNote(set: ReasoningLevelSet): string {
  if (set.shape === 'inherent') {
    return 'this provider selects reasoning by model; there is nothing to set'
  }
  if (set.shape === 'toggle') {
    return 'this provider only switches thinking on or off'
  }
  return set.source === 'provider'
    ? 'reported by the provider for this model'
    : 'provider publishes no level list; documented defaults shown'
}
