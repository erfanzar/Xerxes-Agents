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
 * `Ns` / `Nm` / `Nm Ss` / `Hh Mm` / `Dd Hh` formatter for seconds, shared by
 * the agent panel cards and inspector so every subagent duration speaks the
 * same dialect.
 */
export function fmtDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.max(0, Math.round(seconds))}s`
  }

  const total = Math.max(0, Math.round(seconds))
  const days = Math.floor(total / 86_400)
  const hours = Math.floor((total % 86_400) / 3_600)
  const minutes = Math.floor((total % 3_600) / 60)
  const secs = total % 60

  if (days > 0) {
    return `${days}d ${hours}h`
  }
  if (hours > 0) {
    return minutes === 0 ? `${hours}h` : `${hours}h ${minutes}m`
  }
  return secs === 0 ? `${minutes}m` : `${minutes}m ${secs}s`
}
