// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Claude-Code-style thinking escalation ladder.
 *
 * WHY this exists: users need a lightweight, inline way to ask for deeper
 * reasoning on one hard prompt without touching session configuration. A
 * keyword embedded anywhere in the user prompt escalates the thinking budget
 * for that turn only: `think` < `think hard` (alias `megathink`) <
 * `think harder` < `ultrathink`.
 *
 * WHY whole-word, longest-phrase-first detection: a bare substring match
 * would fire on ordinary words like "rethinking" or "something", and testing
 * `think` before `ultrathink` would shadow the strongest level. Anchoring
 * each keyword with `\b` word boundaries keeps prose inert, and scanning the
 * ladder strongest-first guarantees the highest escalation wins when several
 * keywords appear in one prompt. Matching is case-insensitive because these
 * keywords are conversational triggers, not syntax.
 */
export type ThinkingLevel = 'think' | 'think_hard' | 'think_harder' | 'ultrathink'

export interface ThinkingDirective {
  /** Token budget requested from providers that accept one. */
  readonly budgetTokens: number
  /** Provider-neutral effort hint for effort-based reasoning APIs. */
  readonly effort: 'high' | 'low' | 'medium'
  readonly level: ThinkingLevel
  /** The exact keyword that matched, for status surfaces. */
  readonly matchedKeyword: string
}

/**
 * The ladder, ordered strongest-first so a linear scan resolves precedence.
 * Budgets roughly double per rung: the bottom rung is a cheap nudge while the
 * top approaches common provider thinking caps. Each rung also carries an
 * effort hint so effort-based reasoning APIs (which accept no token budget)
 * follow the same escalation shape.
 */
const LEVELS: readonly (ThinkingDirective & { readonly keywords: readonly string[] })[] = [
  { budgetTokens: 32_000, effort: 'high', keywords: ['ultrathink'], level: 'ultrathink', matchedKeyword: 'ultrathink' },
  { budgetTokens: 20_000, effort: 'high', keywords: ['think harder'], level: 'think_harder', matchedKeyword: 'think harder' },
  { budgetTokens: 10_000, effort: 'medium', keywords: ['think hard', 'megathink'], level: 'think_hard', matchedKeyword: 'think hard' },
  { budgetTokens: 4_000, effort: 'medium', keywords: ['think'], level: 'think', matchedKeyword: 'think' },
]

/**
 * Build the whole-word matcher for one keyword. Regex metacharacters are
 * escaped so keywords stay literal text, and whitespace runs collapse to
 * `\s+` so "think   hard" still matches the "think hard" rung.
 */
function keywordPattern(keyword: string): RegExp {
  const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s+')
  return new RegExp(`\\b${escaped}\\b`, 'i')
}

/**
 * Pre-compiled matchers flattened from LEVELS. flatMap preserves LEVELS
 * order, so the first pattern that matches is always the strongest level
 * present in the prompt. Each entry pins `matchedKeyword` to the specific
 * alias being scanned (e.g. `megathink` rather than `think hard`).
 */
const PATTERNS: readonly { readonly directive: ThinkingDirective; readonly pattern: RegExp }[] = LEVELS.flatMap(
  ({ keywords, ...directive }) =>
    keywords.map(keyword => ({
      directive: { ...directive, matchedKeyword: keyword },
      pattern: keywordPattern(keyword),
    })),
)

/**
 * Detect the strongest thinking escalation keyword in a prompt, if any. The
 * early return is safe because PATTERNS is pre-sorted strongest-first: the
 * first hit is already the winning rung, so weaker keywords never shadow a
 * stronger one. A fresh directive copy is returned so callers cannot mutate
 * the shared ladder entries.
 */
export function detectThinkingDirective(prompt: string): ThinkingDirective | undefined {
  for (const { directive, pattern } of PATTERNS) {
    if (pattern.test(prompt)) return { ...directive }
  }
  return undefined
}

/**
 * Ultra mode pins every turn to the top of the ladder, regardless of prompt
 * content. Frozen because it is a shared singleton: a caller mutating the
 * returned directive must never corrupt the canonical ultra preset.
 */
export const ULTRA_THINKING_DIRECTIVE: ThinkingDirective = Object.freeze({
  budgetTokens: 32_000,
  effort: 'high',
  level: 'ultrathink',
  matchedKeyword: 'ultra mode',
})

export interface SessionThinkingDefaults {
  readonly budgetTokens?: number
  readonly effort?: string
  /**
   * Explicit session toggle. A bare budget or effort implies enabled on its
   * own, so a profile can switch thinking on with a single field; only
   * `enabled: false` force-disables thinking even when the others are set.
   */
  readonly enabled?: boolean
}

/**
 * Resolve one turn's effective thinking directive.
 *
 * Precedence: ultra mode > keyword in this turn's prompt > session defaults
 * (runtime settings over profile sampling). WHY this order: an explicit,
 * deliberate per-turn signal must outrank ambient session configuration, and
 * between the two per-turn signals the session-wide ultra toggle outranks an
 * inline keyword because the user opted every turn into maximum reasoning.
 * Keywords are re-detected per prompt, so an escalation in one turn never
 * leaks into later turns. No enabled flag, budget, or effort means thinking
 * stays off for the turn.
 */
export function resolveTurnThinking(options: {
  readonly defaults?: SessionThinkingDefaults
  readonly prompt: string
  readonly ultraMode: boolean
}): ThinkingDirective | undefined {
  if (options.ultraMode) return { ...ULTRA_THINKING_DIRECTIVE }
  const keyword = detectThinkingDirective(options.prompt)
  if (keyword) return keyword
  const defaults = options.defaults
  if (defaults?.enabled !== true && defaults?.budgetTokens === undefined && !defaults?.effort) return undefined
  if (defaults?.enabled === false) return undefined
  // An explicit off-value effort disables even when a budget is configured;
  // a configured 'low' effort is preserved instead of silently upgraded.
  const configuredEffort = defaults?.effort?.trim().toLowerCase()
  if (configuredEffort === 'off' || configuredEffort === 'none' || configuredEffort === 'disabled') return undefined
  // Session defaults land on the think_hard rung (10k / medium) when a field
  // is omitted, so a bare `enabled: true` toggle picks a mid-ladder budget
  // rather than an extreme.
  return {
    budgetTokens: defaults.budgetTokens ?? 10_000,
    effort: configuredEffort === 'high' ? 'high' : configuredEffort === 'low' ? 'low' : 'medium',
    level: 'think_hard',
    matchedKeyword: 'session default',
  }
}
