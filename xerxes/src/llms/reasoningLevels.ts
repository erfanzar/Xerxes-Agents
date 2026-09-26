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
 * So levels are only ever what something reported, live, per model: the
 * provider itself (its /models entry, Codex's catalog, Claude Code's model
 * list), else models.dev (Kimi Code's approach). Nothing here is a list.
 * When neither says anything, nothing is offered — never a guessed ladder.
 */

import { modelsDev, type LiveReasoning } from './modelsDev.js'
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
  /** Precise origin; source is retained for existing UI/protocol consumers. */
  readonly provenance?: 'provider_reported' | 'bundled_catalog' | 'provider_fallback'
  /**
   * Whether `off` is a real choice. Models whose thinking-level map marks
   * `off: null` (always-on adaptive thinking, e.g. Kimi K3) cannot disable
   * reasoning, and offering the switch would lie (pi-ai getSupportedThinkingLevels
   * filters the rung the same way).
   */
  readonly canDisable?: boolean
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

/**
 * A picker set from reported reasoning: effort levels as given, `off` only
 * where it can be switched off, a plain switch when it reasons without
 * levels, nothing when it does not reason (or cannot be told apart).
 */
export function liveReasoningLevels(reasoning: LiveReasoning | undefined, source: 'provider' | 'catalog'): ReasoningLevelSet | undefined {
  if (!reasoning) return undefined
  const provenance = source === 'provider' ? 'provider_reported' as const : 'bundled_catalog' as const
  if (!reasoning.supported) return { defaultEffort: undefined, levels: [], shape: 'inherent', source: 'provider', provenance, canDisable: false }
  if (reasoning.efforts.length) {
    return {
      defaultEffort: reasoning.defaultEffort && reasoning.efforts.includes(reasoning.defaultEffort) ? reasoning.defaultEffort : undefined,
      levels: reasoning.efforts.map(effort => ({ effort })),
      shape: 'effort',
      source: 'provider',
      provenance,
      canDisable: reasoning.canDisable,
    }
  }
  // Reasons, no levels: on/off where it can be switched off, else always on.
  return reasoning.canDisable
    ? { defaultEffort: undefined, levels: [{ effort: REASONING_ON }], shape: 'toggle', source: 'provider', provenance, canDisable: true }
    : { defaultEffort: undefined, levels: [], shape: 'inherent', source: 'provider', provenance, canDisable: false }
}

/** True when nothing reported the model's reasoning controls (see fallbackReasoningLevels). */
export function reasoningUnreported(set: ReasoningLevelSet): boolean {
  return set.source === 'fallback' && set.provenance === 'provider_fallback'
}

/**
 * Nothing reported this model's reasoning controls: nothing is offered. The
 * provider's own default applies (no field is sent).
 */
export function fallbackReasoningLevels(_providerName?: ProviderName): ReasoningLevelSet {
  return { defaultEffort: undefined, levels: [], shape: 'inherent', source: 'fallback', provenance: 'provider_fallback', canDisable: false }
}

/** Wrap a provider-reported list, preserving its order and descriptions. */
export function providerReasoningLevels(
  levels: readonly ReasoningLevel[],
  defaultEffort: string | undefined,
): ReasoningLevelSet {
  return { defaultEffort, levels, shape: 'effort', source: 'provider', provenance: 'provider_reported' }
}

/**
 * What models.dev says about a model's reasoning, for providers that do not
 * describe their own models (Kimi Code's approach). `baseUrl` matches the
 * profile to the right models.dev provider.
 */
export function catalogReasoningLevels(
  model: string,
  providerName: ProviderName | undefined,
  baseUrl?: string,
): ReasoningLevelSet | undefined {
  if (!model.trim()) return undefined
  const found = modelsDev.find({ model, ...(providerName ? { provider: providerName } : {}), ...(baseUrl ? { baseUrl } : {}) })
  return liveReasoningLevels(found?.reasoning, 'catalog')
}

/** Every value the user may select, including the Xerxes-side off switch. */
export function selectableEfforts(set: ReasoningLevelSet): readonly string[] {
  // An `inherent` provider offers nothing to select: presenting `off` alone
  // would imply reasoning can be disabled, which it cannot.
  if (set.shape === 'inherent') {
    return []
  }
  // Always-on models (thinking map marks off: null) get no off row either.
  if (set.canDisable === false) {
    return set.levels.map(level => level.effort)
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
  // Nothing reported this model's levels: an effort someone set explicitly
  // goes to the provider as written (it validates); only reported data is
  // grounds to refuse one.
  if (reasoningUnreported(set)) return requested.trim()
  if (set.shape === 'inherent') return undefined
  if (clean === REASONING_OFF) return set.canDisable === false ? undefined : REASONING_OFF
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
    if (set.source !== 'provider') return 'nothing reports reasoning controls for this model; the provider decides'
    return set.levels.length ? 'this provider selects reasoning by model; there is nothing to set' : 'this model decides its own reasoning; there is nothing to set'
  }
  if (set.shape === 'toggle') {
    return 'this provider only switches thinking on or off'
  }
  if (set.source !== 'provider') return 'nothing reports reasoning controls for this model; the provider decides'
  return set.provenance === 'bundled_catalog' ? 'from models.dev for this model' : 'reported by the provider for this model'
}
