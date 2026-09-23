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

import { useState, type ReactElement } from 'react'

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

export function SessionDiagnostics({ snap }: { snap: Snapshot }): ReactElement {
  const [expanded, setExpanded] = useState(false)
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
  if (snap.cacheHitRate != null) metrics.push(['Cache hit', Math.round(snap.cacheHitRate * 100) + '%'])
  if (snap.inputTokens > 0) metrics.push(['Input tokens', compactTokensOf(snap.inputTokens)])
  if (snap.costUsd != null && snap.costUsd > 0) metrics.push(['Cost', '$' + snap.costUsd.toFixed(snap.costUsd < 0.01 ? 4 : 2)])
  if (snap.model) metrics.push(['Model', snap.model])
  if (snap.branch) metrics.push(['Branch', snap.branch])
  return <details className="session-diagnostics" open={expanded} onToggle={event => setExpanded(event.currentTarget.open)}><summary>Session statistics</summary>
    <dl className="session-diagnostics__values">{metrics.map(([label, value]) => <div key={label} data-wide={label === 'Model' || label === 'Branch' || undefined}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
  </details>
}
