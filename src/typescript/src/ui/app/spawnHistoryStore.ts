// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { atom } from 'nanostores'

import {
  mergeSubagentProgress,
  subagentProgressId,
  type SubagentProgressPatch
} from '../domain/subagentProgress.js'
import type { SubagentEventPayload } from '../gatewayTypes.js'
import type { SubagentProgress } from '../types.js'

export interface SpawnSnapshot {
  finishedAt: number
  id: string
  label: string
  sessionId: null | string
  startedAt: number
  subagents: SubagentProgress[]
}

export interface SpawnDiffPair {
  baseline: SpawnSnapshot
  candidate: SpawnSnapshot
}

const HISTORY_LIMIT = 10

export const $spawnHistory = atom<SpawnSnapshot[]>([])
export const $spawnDiff = atom<null | SpawnDiffPair>(null)

export const getSpawnHistory = () => $spawnHistory.get()
export const getSpawnDiff = () => $spawnDiff.get()

export const clearSpawnHistory = () => $spawnHistory.set([])
export const clearDiffPair = () => $spawnDiff.set(null)
export const setDiffPair = (pair: SpawnDiffPair) => $spawnDiff.set(pair)

/** Reconcile a background agent event into its archived spawn snapshot without changing history order. */
export const reconcileSpawnHistorySubagent = (
  payload: SubagentEventPayload,
  patch: SubagentProgressPatch
) => {
  const id = subagentProgressId(payload)
  let updated = false
  const snapshots = $spawnHistory.get().map(snapshot => {
    if (updated) return snapshot
    const index = snapshot.subagents.findIndex(agent => agent.id === id)
    const current = snapshot.subagents[index]
    if (index < 0 || !current) return snapshot
    const subagents = [...snapshot.subagents]
    subagents[index] = mergeSubagentProgress(current, payload, patch)
    updated = true
    return { ...snapshot, subagents }
  })
  if (updated) $spawnHistory.set(snapshots)
}

/**
 * Commit a finished turn's spawn tree to history.  Keeps the last 10
 * non-empty snapshots — empty turns (no subagents) are dropped.
 *
 * Why in-memory?  The primary investigation loop is "I just ran a fan-out,
 * it misbehaved, let me look at what happened" — same-session debugging.
 * Disk persistence across process restarts is a natural extension but
 * adds RPC surface for a less-common path.
 */
export const pushSnapshot = (
  subagents: readonly SubagentProgress[],
  meta: { sessionId?: null | string; startedAt?: null | number }
) => {
  if (!subagents.length) {
    return
  }

  const now = Date.now()
  const started = meta.startedAt ?? Math.min(...subagents.map(s => s.startedAt ?? now))

  const snap: SpawnSnapshot = {
    finishedAt: now,
    id: `snap-${now.toString(36)}`,
    label: summarizeLabel(subagents),
    sessionId: meta.sessionId ?? null,
    startedAt: Number.isFinite(started) ? started : now,
    subagents: subagents.map(item => ({ ...item }))
  }

  const next = [snap, ...$spawnHistory.get()].slice(0, HISTORY_LIMIT)
  $spawnHistory.set(next)
}

function summarizeLabel(subagents: readonly SubagentProgress[]): string {
  const top = subagents
    .filter(s => s.parentId == null || subagents.every(o => o.id !== s.parentId))
    .slice(0, 2)
    .map(s => s.goal || 'subagent')
    .join(' · ')

  return top || `${subagents.length} agent${subagents.length === 1 ? '' : 's'}`
}
