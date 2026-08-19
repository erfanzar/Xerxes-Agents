// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Pure status-bar formatting logic shared by both the compatibility view and
// OpenTUI. Keep this file free of framework imports so status formatting never
// pulls renderer state into the native OpenTUI bundle.
// Keep this file free of framework imports; only Theme/Usage types.
import type { Theme } from '../theme.js'
import { fmtK } from '../lib/text.js'
import type { Usage } from '../types.js'

export function ctxBarColor(pct: number | undefined, t: Theme) {
  if (pct == null) {
    return t.color.muted
  }

  if (pct >= 95) {
    return t.color.statusCritical
  }

  if (pct > 80) {
    return t.color.statusBad
  }

  if (pct >= 50) {
    return t.color.statusWarn
  }

  return t.color.statusGood
}

// Shared context-budget read-out so the persistent top breadcrumb and the
// composer-adjacent status rule never drift on what "used"/"max" mean.
export function usageCounts(usage: Usage): { max: number; used: number } {
  return { max: usage.context_max ?? 0, used: usage.context_used ?? usage.total ?? 0 }
}

const effortLabel = (effort?: string) => {
  const value = String(effort ?? '')
    .trim()
    .toLowerCase()

  return value && value !== 'medium' && value !== 'normal' && value !== 'default' ? value : ''
}

const shortModelLabel = (model: string) =>
  model
    .split('/')
    .pop()!
    .replace(/^claude[-_]/, '')
    .replace(/^anthropic[-_]/, '')
    .replace(/[-_]/g, ' ')
    .replace(/\b(\d+)\s+(\d+)\b/g, '$1.$2')
    .trim()

export const modelLabel = (model: string, effort?: string, fast?: boolean) =>
  [shortModelLabel(model), effortLabel(effort), fast ? 'fast' : ''].filter(Boolean).join(' ')

/**
 * The status identity keeps model and mode first because they are the
 * persistent, keyboard-relevant session facts. Workspace and activity trail
 * afterward so narrow terminals reveal the important information first.
 */
export const statusIdentity = (model: string, mode?: string, effort?: string, fast?: boolean) =>
  `${modelLabel(model, effort, fast) || 'model unset'} · ${mode || 'code'}`

/** Show only an explicit or model-generated title, never conversation text. */
export function sessionDisplayTitle(sessionTitle?: null | string, max = 72): string {
  const value = String(sessionTitle ?? '')
    .replace(/\s+/g, ' ')
    .trim()
  const title = value && !/^tui:[0-9a-f]+$/i.test(value) ? value : 'Untitled chat'

  return title.length > max ? `${title.slice(0, Math.max(1, max - 1)).trimEnd()}…` : title
}

/**
 * Compact `in/out/cached` token row for the footer.
 *
 * Cached is shown separately rather than folded into input because they are
 * priced and cached differently, and because a reader watching a long turn
 * wants to see the cache doing its job. Zero-valued parts are omitted so an
 * uncached provider does not carry a permanent `0c`.
 */
export function tokenBreakdown(usage: Usage): string {
  const parts: string[] = []
  if (usage.input) parts.push(`${fmtK(usage.input)} in`)
  if (usage.output) parts.push(`${fmtK(usage.output)} out`)
  if (usage.cache_read) parts.push(`${fmtK(usage.cache_read)} cached`)
  return parts.join(" · ")
}
