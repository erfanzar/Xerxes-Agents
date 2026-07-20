// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Claude-Code-style thinking escalation ladder.
 *
 * A keyword embedded anywhere in the user prompt escalates the thinking
 * budget for that turn only: `think` < `think hard` (alias `megathink`) <
 * `think harder` < `ultrathink`. Detection is whole-word, case-insensitive,
 * and longest-phrase-first so "ultrathink" beats "think" and "rethinking"
 * matches nothing.
 */
export type ThinkingLevel = 'think' | 'think_hard' | 'think_harder' | 'ultrathink'

export interface ThinkingDirective {
  /** Token budget requested from providers that accept one. */
  readonly budgetTokens: number
  /** Provider-neutral effort hint for effort-based reasoning APIs. */
  readonly effort: 'high' | 'medium'
  readonly level: ThinkingLevel
  /** The exact keyword that matched, for status surfaces. */
  readonly matchedKeyword: string
}

const LEVELS: readonly (ThinkingDirective & { readonly keywords: readonly string[] })[] = [
  { budgetTokens: 32_000, effort: 'high', keywords: ['ultrathink'], level: 'ultrathink', matchedKeyword: 'ultrathink' },
  { budgetTokens: 20_000, effort: 'high', keywords: ['think harder'], level: 'think_harder', matchedKeyword: 'think harder' },
  { budgetTokens: 10_000, effort: 'medium', keywords: ['think hard', 'megathink'], level: 'think_hard', matchedKeyword: 'think hard' },
  { budgetTokens: 4_000, effort: 'medium', keywords: ['think'], level: 'think', matchedKeyword: 'think' },
]

function keywordPattern(keyword: string): RegExp {
  const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s+')
  return new RegExp(`\\b${escaped}\\b`, 'i')
}

const PATTERNS: readonly { readonly directive: ThinkingDirective; readonly pattern: RegExp }[] = LEVELS.flatMap(
  ({ keywords, ...directive }) =>
    keywords.map(keyword => ({
      directive: { ...directive, matchedKeyword: keyword },
      pattern: keywordPattern(keyword),
    })),
)

/** Detect the strongest thinking escalation keyword in a prompt, if any. */
export function detectThinkingDirective(prompt: string): ThinkingDirective | undefined {
  for (const { directive, pattern } of PATTERNS) {
    if (pattern.test(prompt)) return { ...directive }
  }
  return undefined
}

/** Ultra mode pins every turn to the top of the ladder. */
export const ULTRA_THINKING_DIRECTIVE: ThinkingDirective = Object.freeze({
  budgetTokens: 32_000,
  effort: 'high',
  level: 'ultrathink',
  matchedKeyword: 'ultra mode',
})

export interface SessionThinkingDefaults {
  readonly budgetTokens?: number
  readonly effort?: string
  /** Explicit session toggle; budget or effort imply enabled on their own. */
  readonly enabled?: boolean
}

/**
 * Resolve one turn's effective thinking directive.
 *
 * Precedence: ultra mode > keyword in this turn's prompt > session defaults
 * (runtime settings over profile sampling). No enabled flag, budget, or
 * effort means thinking stays off for the turn.
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
  return {
    budgetTokens: defaults.budgetTokens ?? 10_000,
    effort: defaults.effort === 'high' ? 'high' : 'medium',
    level: 'think_hard',
    matchedKeyword: 'session default',
  }
}
