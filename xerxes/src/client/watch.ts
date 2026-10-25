// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { plural } from './format.js'
import type { WatchAnomalyKind, WatchRow, WatchStats } from './hubTypes.js'

/**
 * Shared Watch wording and choices for the desktop and the phone, so both say
 * the same thing. A watcher on the Hub reads every step of a watched run at no
 * model cost; XERXES wakes only on an anomaly or at the review interval.
 * Everything that came from a run (titles, test names, anomaly details, the
 * summary line) is untrusted text: callers render it as text, never as HTML.
 */

export const WATCH_EXPLANATION =
  'A watcher on the Hub reads every step of a watched run as it happens, at no model cost. XERXES only wakes when something looks wrong (a loop, a stall, tests failing again and again, a burst of errors, a nearly full context or a blocked goal) and, if you choose, at the review interval. In a workspace with Autopilot on, XERXES may steer or stop the run itself; otherwise it asks you to confirm.'

export const WATCH_STRIP_NOTE = 'Read live from every step of the run, at no model cost.'

export const WATCH_READ_ONLY_NOTE = 'Only the Hub owner can change Watch. This is how it is set now.'

export const WATCH_EMPTY_NOTE = 'No task is being watched. Pick a task below to start.'

export const WATCH_NO_TASKS_NOTE = 'No task to watch yet. Tasks from your workspaces show up here.'

// ── review interval ───────────────────────────────────────────────────

/** Review intervals offered, in minutes. The Hub accepts any whole number from 5 to 120. */
export const WATCH_REVIEW_PRESETS: readonly number[] = [5, 15, 30, 60]

export interface WatchReviewOption {
  readonly value: string
  readonly label: string
}

/** `off` or the minutes as a string; this is what a select holds. */
export function reviewValueOf(minutes: number | null): string {
  return minutes === null ? 'off' : String(minutes)
}

/** The `review_interval_minutes` argument for a select value: null clears, a number sets. */
export function reviewMinutesOf(value: string): number | null {
  if (value === 'off') return null
  const minutes = Number(value)
  return Number.isInteger(minutes) ? minutes : null
}

/** Off, then the presets; a saved interval outside the presets is offered too so it is never lost silently. */
export function watchReviewOptions(current: number | null): readonly WatchReviewOption[] {
  const minutes = [...WATCH_REVIEW_PRESETS]
  if (current !== null && !minutes.includes(current)) minutes.push(current)
  minutes.sort((a, b) => a - b)
  return [{ value: 'off', label: 'Off' }, ...minutes.map(m => ({ value: String(m), label: `${m} min` }))]
}

// ── wording of one card ───────────────────────────────────────────────

const ANOMALY_LABELS: Readonly<Record<WatchAnomalyKind, string>> = {
  loop: 'Looping',
  stall: 'Stalled',
  test_fail_repeat: 'Tests keep failing',
  error_burst: 'Many errors',
  context_high: 'Context almost full',
  goal_blocked: 'Goal blocked',
}

export function anomalyLabel(kind: WatchAnomalyKind): string {
  return ANOMALY_LABELS[kind]
}

/** "Watching", or why it is off. `active` is the Hub's verdict. */
export function watchStateText(row: Pick<WatchRow, 'active' | 'enabled'>): string {
  if (row.active) return 'Watching'
  return row.enabled ? 'Off (expired or the owner changed)' : 'Off'
}

/** "Watching · 2 tasks", or undefined when no watch is active. */
export function watchChipText(rows: readonly WatchRow[]): string | undefined {
  const active = rows.filter(row => row.active).length
  if (active === 0) return undefined
  return `Watching · ${plural(active, 'task')}`
}

/** Context use in whole percent, or null when the window size is not known. */
export function watchContextPercent(stats: Pick<WatchStats, 'contextTokens' | 'maxContext'>): number | null {
  if (stats.contextTokens === null || stats.maxContext === null || stats.maxContext <= 0) return null
  return Math.min(100, Math.round((stats.contextTokens / stats.maxContext) * 100))
}

/** "just now", "12 s ago", "3 min ago", "2 h ago", "4 d ago". Finer than `relativeTime`, because a watched run moves in seconds. */
export function activityAgo(iso: string | null, nowMs: number): string {
  if (iso === null) return 'no activity yet'
  const then = Date.parse(iso)
  if (!Number.isFinite(then)) return 'unknown'
  const seconds = Math.max(0, Math.round((nowMs - then) / 1000))
  if (seconds < 5) return 'just now'
  if (seconds < 60) return `${seconds} s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours} h ago`
  return `${Math.floor(hours / 24)} d ago`
}

export interface WatchFact {
  readonly key: string
  readonly text: string
  /** `bad` facts are highlighted (failures, repeated calls). */
  readonly tone: 'plain' | 'bad'
}

/** The counters of one run, in reading order. Only what is non-zero or informative is listed. */
export function watchFacts(stats: WatchStats): readonly WatchFact[] {
  const facts: WatchFact[] = [
    { key: 'edits', text: plural(stats.edits, 'edit'), tone: 'plain' },
    { key: 'tools', text: plural(stats.toolCalls, 'tool call'), tone: 'plain' },
  ]
  if (stats.toolFailures > 0) facts.push({ key: 'failures', text: `${stats.toolFailures} failed`, tone: 'bad' })
  if (stats.errors > 0) facts.push({ key: 'errors', text: plural(stats.errors, 'error'), tone: 'bad' })
  if (stats.tests.runs > 0) {
    facts.push({
      key: 'tests',
      text: `tests: ${plural(stats.tests.runs, 'run')}, ${stats.tests.passed} passed, ${stats.tests.failed} failed`,
      tone: stats.tests.last === 'fail' ? 'bad' : 'plain',
    })
  }
  if (stats.agents.total > 0) facts.push({ key: 'agents', text: `${stats.agents.running}/${stats.agents.total} agents running`, tone: 'plain' })
  const context = watchContextPercent(stats)
  if (context !== null) facts.push({ key: 'context', text: `context ${context}%`, tone: context >= 90 ? 'bad' : 'plain' })
  if (stats.repeatedCall !== null) facts.push({ key: 'repeated', text: `repeated ${stats.repeatedCall.name} ×${stats.repeatedCall.count}`, tone: 'bad' })
  return facts
}

/** "Every 15 min", or "No reviews". */
export function reviewText(minutes: number | null): string {
  return minutes === null ? 'No reviews' : `Review every ${minutes} min`
}

/**
 * Cards to show: every enabled watch, runs with an open problem first, then the
 * ones being watched, then the rest by title. Disabled rows are not cards.
 */
export function watchCardRows(rows: readonly WatchRow[]): readonly WatchRow[] {
  const rank = (row: WatchRow): number => ((row.stats?.anomalies.length ?? 0) > 0 ? 0 : row.active ? 1 : 2)
  return rows
    .filter(row => row.enabled)
    .sort((a, b) => rank(a) - rank(b) || a.title.localeCompare(b.title) || a.taskId.localeCompare(b.taskId))
}
