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

/** Cell count of the compact block-glyph context meter (mockup 02/11). */
export const CTX_METER_CELLS = 5
/** Filled cell of the block-glyph meter. */
export const CTX_METER_FILLED = '▰'
/** Empty cell of the block-glyph meter; deliberately a lighter twin of ▰. */
export const CTX_METER_EMPTY = '▱'

/**
 * Render the small block-glyph bar that precedes the textual context read-out.
 *
 * Unknown pressure renders as an all-empty bar rather than guessing, and any
 * non-zero usage earns at least one filled cell so a nearly-fresh window never
 * reads as broken next to its own percentage.
 */
export function ctxMeterBar(pct: number | undefined, cells = CTX_METER_CELLS): string {
  const safeCells = Math.trunc(cells)

  if (!Number.isFinite(safeCells) || safeCells <= 0) {
    return ''
  }

  if (pct == null || !Number.isFinite(pct)) {
    return CTX_METER_EMPTY.repeat(safeCells)
  }

  const clamped = Math.max(0, Math.min(100, pct))
  const filled =
    clamped <= 0
      ? 0
      : clamped >= 100
        ? safeCells
        : Math.min(safeCells, Math.max(1, Math.round((clamped / 100) * safeCells)))

  return CTX_METER_FILLED.repeat(filled) + CTX_METER_EMPTY.repeat(safeCells - filled)
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
 * What the next ⏎ is allowed to do to your files.
 *
 * The composer's second row states mode, model and write policy together
 * because those three decide it between them — knowing the model without
 * knowing whether it may write is not knowing what will happen.
 */
export const writePolicyLabel = (permissionMode?: string): string => {
  switch (permissionMode) {
    case 'plan':
      return 'plan only, no writes'
    case 'manual':
      return 'writes need approval'
    case 'auto':
      return 'writes approved in repo'
    default:
      // 'accept-all' is the shipped default, so this is the common case and
      // has to say so plainly rather than hiding behind the word "auto".
      return 'writes apply without asking'
  }
}

/**
 * The status identity keeps model and mode first because they are the
 * persistent, keyboard-relevant session facts. Workspace and activity trail
 * afterward so narrow terminals reveal the important information first.
 */
export const statusIdentity = (model: string, mode?: string, effort?: string, fast?: boolean) =>
  `${modelLabel(model, effort, fast) || 'model unset'} · ${mode || 'code'}`

/**
 * Show only an explicit or model-generated title, never conversation text.
 *
 * Returns '' for a session that has not been named yet. That blank is load
 * bearing: `SessionHeader` renders just the mode label for it, which is
 * honest. Substituting a placeholder here made every chat read as "Untitled
 * chat" and made the header's empty-title branch unreachable.
 */
export function sessionDisplayTitle(sessionTitle?: null | string, max = 72): string {
  const value = String(sessionTitle ?? '')
    .replace(/\s+/g, ' ')
    .trim()
  const title = value && !/^tui:[0-9a-f]+$/i.test(value) ? value : ''

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

/**
 * Wall-clock format matching the desktop stats bar exactly: minutes and
 * zero-padded seconds, with minutes running past 60 ("885m11s") so the TUI
 * and desktop never disagree on the same session.
 */
export function telemetryDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.round(ms / 1000))
  const minutes = Math.trunc(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes > 0) return `${minutes}m${String(seconds).padStart(2, '0')}s`
  return `${seconds}s`
}

/**
 * One-line cumulative session telemetry, the same counters the desktop stats
 * bar shows: turns, steps, LLM/tool wall time, TTFT, throughput, cache hit
 * rate, token totals.
 *
 * Returns '' for a fresh session (no turns yet) so the row can hide instead
 * of displaying a bar of zeros. Cache rate renders only when the provider
 * actually reported cache telemetry — an absent value is never faked as 0%.
 */
export function sessionTelemetryLine(usage: Usage): string {
  const turns = usage.turns ?? 0
  if (turns <= 0) return ''

  const parts: string[] = [`${turns} turn${turns === 1 ? '' : 's'}`]
  const steps = (usage.llm_steps ?? 0) + (usage.tool_steps ?? 0)
  if (steps > 0) parts.push(`${fmtK(steps)} steps`)
  if (usage.llm_ms) parts.push(`LLM ${telemetryDuration(usage.llm_ms)}`)
  if (usage.tool_ms) parts.push(`tools ${telemetryDuration(usage.tool_ms)}`)
  if (usage.ttft_avg_ms) parts.push(`TTFT ${(usage.ttft_avg_ms / 1000).toFixed(1)}s`)
  if (usage.tok_per_sec) parts.push(`${Math.round(usage.tok_per_sec)} tok/s`)
  if (usage.cache_hit_rate !== undefined) parts.push(`cache ${Math.round(usage.cache_hit_rate * 100)}%`)
  if (usage.input) parts.push(`${fmtK(usage.input)} in`)
  if (usage.output) parts.push(`${fmtK(usage.output)} out`)
  return parts.join(' · ')
}
