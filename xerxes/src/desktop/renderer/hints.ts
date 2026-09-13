// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Pure helpers for the composer's live completion hints. The daemon owns the
 * completion content (slash prefixes and `/skill` references via the same
 * `complete` RPC the TUI uses); these helpers only decide WHEN hints show and
 * HOW a pick rewrites the draft.
 */

export interface HintItem {
  readonly kind?: 'command' | 'skill'
  readonly value: string
  readonly label: string
  readonly meta: string
}

/**
 * Typing that should surface live hints: a single `/token` being typed, or a
 * `/skill <reference>` in progress. Prose, and multi-word commands with their
 * arguments already typed, never hint.
 */
export function wantsHints(draft: string): boolean {
  if (/\s$/.test(draft) && draft.trim() !== '/skill') return false
  const text = draft.trim()
  if (!text.startsWith('/')) return false
  if (!/\s/.test(text)) return true
  return /^\/skill\s+\S*$/.test(text)
}

/** A picked hint becomes the draft; values that lack a trailing space get one. */
export function applyCompletion(value: string): string {
  return /\S$/.test(value) ? `${value} ` : value
}
