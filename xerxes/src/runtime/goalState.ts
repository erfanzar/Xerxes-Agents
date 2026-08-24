// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Durable objective-mode goal ledger with compare-and-set semantics.
 *
 * Objective mode used to be a prompt contract only: nothing durable recorded
 * *what* the goal was or how many guarded rounds it had consumed, so a restart
 * lost the objective's identity entirely and the guard's retry budget lived
 * only in one turn's locals. The ledger stores the goal statement, its phase,
 * a monotonically increasing revision, and the round count under one bounded
 * metadata key.
 *
 * Every mutation is compare-and-set on `revision`: writers that observed stale
 * state (a second connection resuming mid-turn, a retried assembly) get an
 * explicit conflict instead of silently clobbering newer progress. The turn
 * runner is the single writer per session today; CAS keeps that property
 * enforceable rather than assumed.
 */

export type GoalPhase = 'active' | 'blocked' | 'verified'

/** Durable record of one session's hard-goal objective. */
export interface GoalLedger {
  /** Acceptance criteria as stated alongside the goal; empty when none were given. */
  readonly criteria: readonly string[]
  readonly phase: GoalPhase
  readonly revision: number
  /** Guard-supervised rounds consumed while pursuing this goal. */
  readonly roundsStarted: number
  readonly text: string
  readonly updatedAt: number
}

const GOAL_LEDGER_KEY = 'goal_ledger'
export const MAX_GOAL_CRITERIA = 16

/** Read the current ledger without mutating anything. */
export function readGoalLedger(metadata: Readonly<Record<string, unknown>>): GoalLedger | undefined {
  const raw = metadata[GOAL_LEDGER_KEY]
  return isGoalLedger(raw) ? raw : undefined
}

/**
 * Create the ledger for a newly stated goal.
 *
 * Fails (with the incumbent) when a goal is already active: re-entering
 * objective mode must not quietly reset progress. Callers that intend a fresh
 * objective clear the old one explicitly via {@link clearGoalLedger}.
 */
export function startGoalLedger(
  metadata: Record<string, unknown>,
  goal: { readonly criteria?: readonly string[]; readonly now: number; readonly text: string },
): { readonly created: GoalLedger } | { readonly existing: GoalLedger } {
  const current = readGoalLedger(metadata)
  if (current) return { existing: current }
  const created: GoalLedger = Object.freeze({
    criteria: Object.freeze((goal.criteria ?? []).slice(0, MAX_GOAL_CRITERIA)),
    phase: 'active',
    revision: 1,
    roundsStarted: 0,
    text: goal.text,
    updatedAt: goal.now,
  })
  metadata[GOAL_LEDGER_KEY] = created
  return { created }
}

/** Remove the ledger — used when the operator leaves objective mode or resets the goal. */
export function clearGoalLedger(metadata: Record<string, unknown>): boolean {
  if (!readGoalLedger(metadata)) return false
  delete metadata[GOAL_LEDGER_KEY]
  return true
}

/** Requested changes applied through {@link updateGoalLedger}, compared against `revision`. */
export interface GoalLedgerPatch {
  readonly criteria?: readonly string[]
  readonly phase?: GoalPhase
  readonly roundDelta?: number
  readonly text?: string
}

export type GoalUpdateOutcome =
  | { readonly ledger: GoalLedger; readonly ok: true }
  | { /** The incumbent when one exists; absent when the ledger was cleared under us. */
    readonly conflictWith?: GoalLedger
    /** `stale`: another writer already advanced past `expectedRevision`. `missing`: the ledger no longer exists. */
    readonly reason: 'missing' | 'stale'
    readonly ok: false }

/**
 * Compare-and-set update. Succeeds only when the caller's `expectedRevision`
 * is still current; every success bumps the revision and refreshes `updatedAt`.
 */
export function updateGoalLedger(
  metadata: Record<string, unknown>,
  expectedRevision: number,
  patch: GoalLedgerPatch,
  now: number,
): GoalUpdateOutcome {
  const current = readGoalLedger(metadata)
  if (!current) {
    return { ok: false, reason: 'missing' }
  }
  if (current.revision !== expectedRevision) {
    return { conflictWith: current, ok: false, reason: 'stale' }
  }
  const updated: GoalLedger = Object.freeze({
    ...current,
    ...(patch.text === undefined ? {} : { text: patch.text }),
    ...(patch.criteria === undefined ? {} : { criteria: Object.freeze(patch.criteria.slice(0, MAX_GOAL_CRITERIA)) }),
    ...(patch.phase === undefined ? {} : { phase: patch.phase }),
    roundsStarted: Math.max(0, current.roundsStarted + (patch.roundDelta ?? 0)),
    revision: current.revision + 1,
    updatedAt: now,
  })
  metadata[GOAL_LEDGER_KEY] = updated
  return { ledger: updated, ok: true }
}

function isGoalLedger(value: unknown): value is GoalLedger {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return typeof candidate.text === 'string'
    && typeof candidate.revision === 'number'
    && typeof candidate.roundsStarted === 'number'
    && typeof candidate.updatedAt === 'number'
    && Array.isArray(candidate.criteria)
    && (candidate.phase === 'active' || candidate.phase === 'blocked' || candidate.phase === 'verified')
}
