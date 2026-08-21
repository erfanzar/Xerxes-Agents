// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SubagentProgress } from '../types.js'

export function subagentElapsedSeconds(agent: SubagentProgress, now = Date.now()): number | null {
  if (typeof agent.durationSeconds === 'number') {
    return Math.max(0, agent.durationSeconds)
  }
  if ((agent.status === 'running' || agent.status === 'queued') && agent.startedAt) {
    return Math.max(0, (now - agent.startedAt) / 1000)
  }
  return null
}

/** Compact token count: `12k`, `1.2k`, `542`. */
export function fmtTokens(n: number): string {
  if (!Number.isFinite(n) || n <= 0) {
    return '0'
  }

  if (n < 1000) {
    return String(Math.round(n))
  }

  if (n < 10_000) {
    return `${(n / 1000).toFixed(1)}k`
  }

  return `${Math.round(n / 1000)}k`
}

/**
 * `Ns` / `Nm` / `Nm Ss` formatter for seconds, shared by the agent panel
 * cards and inspector so every subagent duration speaks the same dialect.
 */
export function fmtDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.max(0, Math.round(seconds))}s`
  }

  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds - m * 60)

  return s === 0 ? `${m}m` : `${m}m ${s}s`
}
