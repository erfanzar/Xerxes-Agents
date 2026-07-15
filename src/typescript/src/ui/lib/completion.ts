// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Pure completion helpers shared by the useCompletion hook. The daemon's
// `complete` RPC returns the candidates; this decides WHEN to ask and HOW to
// apply a chosen value over the active token.

export interface Completion {
  value: string
  label: string
  meta?: string
}

/** True when the draft's active token is worth requesting completions for. */
export function shouldRequestCompletion(draft: string): boolean {
  if (!draft.trim()) {
    return false
  }
  // Slash command name (still typing it, no space yet).
  if (draft.startsWith('/') && !draft.includes(' ')) {
    return true
  }
  // Path-like last token: starts with @ / . ~ or contains a slash.
  const last = draft.split(/\s+/).at(-1) ?? ''
  return /^@?([./~]|[^\s]*\/)/.test(last) && last.length > 0
}

/** The token the completion menu is operating on (last whitespace token). */
export function activeToken(draft: string): string {
  if (draft.startsWith('/') && !draft.includes(' ')) {
    return draft
  }
  return draft.split(/\s+/).at(-1) ?? ''
}

/** Replace the active token in `draft` with `value`. */
export function applyCompletion(draft: string, value: string): string {
  // Slash command: the value is the whole new draft.
  if (draft.startsWith('/') && !draft.includes(' ')) {
    return value
  }
  // Otherwise replace the trailing non-space run (the active token).
  const idx = draft.search(/\S+$/)
  if (idx < 0) {
    return draft + value
  }
  return draft.slice(0, idx) + value
}

/** Cycle a selection index within [0, len) with wraparound. */
export function cycleIndex(current: number, len: number, delta: number): number {
  if (len <= 0) {
    return 0
  }
  return (((current + delta) % len) + len) % len
}
