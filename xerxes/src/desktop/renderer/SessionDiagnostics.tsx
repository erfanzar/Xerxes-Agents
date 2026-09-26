// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Per-session measurements: turns, steps, where the time went, throughput.
 *
 * These answer "how did the model perform", which is a question you have
 * after the fact — not while an agent is working. They live in the rail's
 * diagnostics drawer; the status head above it carries the numbers you
 * want mid-turn (elapsed, steps remaining, context left, cost).
 *
 * Extracted from App.tsx so the drawer, which lives in DesktopPanels, can
 * import it without the two modules importing each other.
 */

import type { ReactElement } from 'react'

import type { Snapshot } from './store.js'

function metricDurationOf(milliseconds: number): string {
  const seconds = Math.max(0, Math.round(milliseconds / 1_000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return `${minutes}m${remainder ? `${remainder}s` : ''}`
}

function ttftOf(milliseconds: number): string {
  return milliseconds < 1_000 ? `${Math.round(milliseconds)}ms` : `${(milliseconds / 1_000).toFixed(1)}s`
}

function compactTokensOf(tokens: number): string {
  if (tokens < 1_000) return String(Math.round(tokens))
  if (tokens < 1_000_000) return `${(tokens / 1_000).toFixed(tokens < 10_000 ? 1 : 0)}K`
  return `${(tokens / 1_000_000).toFixed(1)}M`
}

/** Label/value pairs for the session's measurements, live while a turn runs. */
export function sessionMetrics(snap: Snapshot): [string, string][] {
  const livePhaseMs = snap.turnActive && snap.metricPhaseStartedAt != null ? Math.max(0, Date.now() - snap.metricPhaseStartedAt) : 0
  // A row reading "Unavailable" is worse than no row: it spends a line to
  // say nothing. Metrics the runtime has not reported are simply omitted.
  const metrics: [string, string][] = [
    ['Turns', String(snap.turnCount)],
    ['Steps', String(snap.llmSteps + snap.toolSteps)],
    ['Model time', metricDurationOf(snap.llmDurationMs + (snap.metricPhase === 'llm' ? livePhaseMs : 0))],
    ['Tool time', metricDurationOf(snap.toolDurationMs + (snap.metricPhase === 'tool' ? livePhaseMs : 0))],
  ]
  if (snap.ttftMs != null) metrics.push(['First response', ttftOf(snap.ttftMs)])
  if (snap.tokensPerSecond != null) metrics.push(['Generation', snap.tokensPerSecond.toFixed(1) + ' tokens/s'])
  // From the counts when they are known: a stored rate can predate cache
  // writes being counted and read 100% beside millions of tokens written.
  const cachePrompt = (snap.cacheReadTokens ?? 0) + (snap.cacheWriteTokens ?? 0) + snap.inputTokens
  const cacheHit = snap.cacheReadTokens != null && cachePrompt > 0 ? snap.cacheReadTokens / cachePrompt : snap.cacheHitRate
  if (cacheHit != null) metrics.push(['Cache hit', Math.round(cacheHit * 100) + '%'])
  if (snap.cacheReadTokens != null && snap.cacheReadTokens > 0) metrics.push(['Cache read', compactTokensOf(snap.cacheReadTokens)])
  if (snap.cacheWriteTokens != null && snap.cacheWriteTokens > 0) metrics.push(['Cache written', compactTokensOf(snap.cacheWriteTokens)])
  if (snap.inputTokens > 0) metrics.push(['Uncached input', compactTokensOf(snap.inputTokens)])
  if (snap.costUsd != null && snap.costUsd > 0) metrics.push(['Cost', '$' + snap.costUsd.toFixed(snap.costUsd < 0.01 ? 4 : 2)])
  if (snap.model) metrics.push(['Model', snap.model])
  if (snap.branch) metrics.push(['Branch', snap.branch])
  return metrics
}

/** This task's numbers, always shown, in the same rows as the context meter. */
export function SessionDiagnostics({ snap }: { snap: Snapshot }): ReactElement {
  const metrics = sessionMetrics(snap)
  return <dl className="session-diagnostics session-diagnostics__values" aria-label="Session statistics">
    {metrics.map(([label, value]) => <div key={label} data-wide={label === 'Model' || label === 'Branch' || undefined}><dt>{label}</dt><dd>{value}</dd></div>)}
  </dl>
}
