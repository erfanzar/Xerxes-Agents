// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** Pure per-turn status data and formatting for TUI footer surfaces. */

export interface StatusSnapshot {
  activeSkill: string
  cacheReadTokens: number
  cacheWriteTokens: number
  contextWindow: number
  costUsd: number
  durationSeconds: number
  inputTokens: number
  model: string
  outputTokens: number
  permissionMode: string
  queueDepth: number
}

export interface StatusSnapshotRecord extends StatusSnapshot {
  cacheHitRate: number
  contextPercent: number
  contextUsed: number
}

export const DEFAULT_STATUS_SNAPSHOT: Readonly<StatusSnapshot> = {
  activeSkill: '',
  cacheReadTokens: 0,
  cacheWriteTokens: 0,
  contextWindow: 200_000,
  costUsd: 0,
  durationSeconds: 0,
  inputTokens: 0,
  model: '',
  outputTokens: 0,
  permissionMode: 'accept-all',
  queueDepth: 0
}

/** Materializes a complete status snapshot from a partial runtime update. */
export const createStatusSnapshot = (values: Partial<StatusSnapshot> = {}): StatusSnapshot => ({
  activeSkill: values.activeSkill ?? DEFAULT_STATUS_SNAPSHOT.activeSkill,
  cacheReadTokens: values.cacheReadTokens ?? DEFAULT_STATUS_SNAPSHOT.cacheReadTokens,
  cacheWriteTokens: values.cacheWriteTokens ?? DEFAULT_STATUS_SNAPSHOT.cacheWriteTokens,
  contextWindow: values.contextWindow ?? DEFAULT_STATUS_SNAPSHOT.contextWindow,
  costUsd: values.costUsd ?? DEFAULT_STATUS_SNAPSHOT.costUsd,
  durationSeconds: values.durationSeconds ?? DEFAULT_STATUS_SNAPSHOT.durationSeconds,
  inputTokens: values.inputTokens ?? DEFAULT_STATUS_SNAPSHOT.inputTokens,
  model: values.model ?? DEFAULT_STATUS_SNAPSHOT.model,
  outputTokens: values.outputTokens ?? DEFAULT_STATUS_SNAPSHOT.outputTokens,
  permissionMode: values.permissionMode ?? DEFAULT_STATUS_SNAPSHOT.permissionMode,
  queueDepth: values.queueDepth ?? DEFAULT_STATUS_SNAPSHOT.queueDepth
})

/** The absent value represents the shipped default, which is YOLO/accept-all. */
export const isYoloEnabled = (permissionMode?: string): boolean => (permissionMode || 'accept-all') === 'accept-all'

/** Fresh prompt tokens plus provider cache reads count against context. */
export const statusContextUsed = (snapshot: StatusSnapshot): number => snapshot.inputTokens + snapshot.cacheReadTokens

/** Percentage of the model's context window in use, capped at 100. */
export const statusContextPercent = (snapshot: StatusSnapshot): number => {
  if (snapshot.contextWindow <= 0) {
    return 0
  }

  return Math.min(100, (100 * statusContextUsed(snapshot)) / snapshot.contextWindow)
}

/** Fraction of prompt tokens obtained through the provider cache. */
export const statusCacheHitRate = (snapshot: StatusSnapshot): number => {
  const total = snapshot.inputTokens + snapshot.cacheReadTokens

  if (total <= 0) {
    return 0
  }

  return snapshot.cacheReadTokens / total
}

/** A serializable status record that includes its derived utilization values. */
export const statusSnapshotRecord = (snapshot: StatusSnapshot): StatusSnapshotRecord => ({
  ...snapshot,
  cacheHitRate: statusCacheHitRate(snapshot),
  contextPercent: statusContextPercent(snapshot),
  contextUsed: statusContextUsed(snapshot)
})

/** Formats token counts with stable K/M suffixes for a one-line status row. */
export const compactStatusNumber = (value: number): string => {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`
  }

  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`
  }

  return String(value)
}

/** Formats a non-negative duration as a zero-padded MM:SS clock. */
export const formatStatusDuration = (seconds: number): string => {
  const total = Math.max(0, Math.trunc(seconds))
  const minutes = Math.trunc(total / 60)
  const remainingSeconds = total % 60

  return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`
}

/**
 * Renders the compact plain-text status form used by non-React status
 * consumers, logs, and narrow terminal fallbacks.
 */
export const formatStatus = (snapshot: StatusSnapshot): string => {
  const cachePart =
    snapshot.cacheReadTokens || snapshot.cacheWriteTokens
      ? `/${compactStatusNumber(snapshot.cacheReadTokens)}c/${compactStatusNumber(snapshot.cacheWriteTokens)}cw`
      : ''
  const extras: string[] = []

  if (snapshot.queueDepth) {
    extras.push(`queued=${snapshot.queueDepth}`)
  }

  if (snapshot.activeSkill) {
    extras.push(`skill=${snapshot.activeSkill}`)
  }

  if (isYoloEnabled(snapshot.permissionMode)) {
    extras.push('YOLO ON')
  } else if (snapshot.permissionMode && snapshot.permissionMode !== 'auto') {
    extras.push(snapshot.permissionMode)
  }

  const extrasPart = extras.length > 0 ? ` | ${extras.join(' ')}` : ''

  return `${snapshot.model || '(no model)'} | ${compactStatusNumber(snapshot.inputTokens)}in/${compactStatusNumber(
    snapshot.outputTokens
  )}out${cachePart} | $${snapshot.costUsd.toFixed(4)} | ${statusContextPercent(snapshot).toFixed(0)}% ctx | ${formatStatusDuration(
    snapshot.durationSeconds
  )}${extrasPart}`
}
