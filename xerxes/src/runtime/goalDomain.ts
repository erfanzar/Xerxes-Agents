// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Event-sourced same-session goals.
 *
 * The design — phases, compare-and-set refs, whole-snapshot change events,
 * process-local activation, round attribution — follows the goal subsystem of
 * DeepSeek Harness (github.com/deepseek-ai/deepseek-harness, MIT), which
 * solves this problem better than the marker-matching guard this replaces.
 * The implementation is written against Xerxes's own session metadata rather
 * than copied: their service is a Cordis plugin over a session-log service and
 * shares no runtime with ours. No DeepSeek source is reproduced here.
 *
 * What the old ledger got wrong, and this fixes:
 *
 *   - Completion was inferred by grepping the model's prose for English
 *     phrases ("objective met", "all tests pass"). A model answering in
 *     another language, or phrasing success differently, could never stop.
 *     Lifecycle is now a typed transition the model requests explicitly.
 *   - There was no `paused`, so there was no way to hold a goal without
 *     abandoning it.
 *   - Activation was implied by the durable phase, so a resumed session
 *     silently resumed autonomous work. Activation is now process-local and
 *     starts disarmed on every load; only a human-authorised resume rearms it.
 *
 * Every mutation is compare-and-set on `revision` and appends a whole-value
 * change event. State is a strict fold over those events, so replay after a
 * restart reconstructs exactly what happened rather than trusting a mutable
 * blob.
 */

/** Identifies one goal across its durable revisions. */
export type GoalId = string

/** Compare-and-set identity for one exact goal revision. */
export interface GoalRef {
  readonly id: GoalId
  /** Positive; every durable mutation increments it. */
  readonly revision: number
}

/** Durable continuation phase. Activation is process-local and separate. */
export type GoalPhase = 'active' | 'paused' | 'blocked' | 'complete'

/** Whether this process may automatically continue an active goal. */
export type GoalActivation = 'armed' | 'disarmed'

/** Machine-routable and human-readable explanation for a blocked goal. */
export interface GoalBlockReason {
  /** Stable lower-kebab-case classification chosen by the blocking policy. */
  readonly code: string
  /** Non-empty explanation shown to humans and models. */
  readonly message: string
}

/** Full durable state written by every non-clear mutation. */
export interface GoalSnapshot extends GoalRef {
  readonly objective: string
  readonly phase: GoalPhase
  /** Present exactly while `phase` is `blocked`. */
  readonly blockedReason?: GoalBlockReason
  /** Total admitted goal-round cap. */
  readonly maxGoalRounds: number
}

/** Current projection, including values derived from the change log. */
export interface GoalView extends GoalSnapshot {
  /** Highest admitted round number for this goal. */
  readonly roundsStarted: number
  readonly createdAt: number
  readonly updatedAt: number
  /** Process-local continuation eligibility; never persisted. */
  readonly activation: GoalActivation
}

/** State-changing verbs recorded in the durable change. */
/**
 * `round` is a first-class operation rather than an edit that happens to move a
 * counter: it is the only mutation that spends the budget, and the strict fold
 * cannot check budget accounting if it is indistinguishable from an objective
 * edit in the log.
 */
export type GoalOperation =
  | 'create'
  | 'edit'
  | 'pause'
  | 'resume'
  | 'complete'
  | 'block'
  | 'round'
  | 'clear'

/** Whole-snapshot mutation committed by a durable change event. */
export interface GoalSnapshotChange {
  readonly kind: 'goal/change'
  readonly version: 1
  readonly operation: Exclude<GoalOperation, 'clear'>
  readonly goal: GoalSnapshot
  readonly roundsStarted: number
  readonly createdAt: number
  readonly updatedAt: number
}

/** Tombstone retained when the current goal is cleared. */
export interface GoalClearChange {
  readonly kind: 'goal/change'
  readonly version: 1
  readonly operation: 'clear'
  readonly cleared: GoalRef
  readonly clearedAt: number
}

export type GoalChange = GoalSnapshotChange | GoalClearChange

/** Attribution carried by an admitted continuation round's prompt. */
export interface GoalMessageSource {
  readonly kind: 'goal'
  readonly goalId: GoalId
  readonly revision: number
  /** Positive admitted continuation round. */
  readonly round: number
}

/** Stable codes for rejected reads and mutations. */
export type GoalErrorCode =
  | 'GOAL_NOT_FOUND'
  | 'GOAL_ALREADY_EXISTS'
  | 'GOAL_STALE_REVISION'
  | 'GOAL_INVALID_OBJECTIVE'
  | 'GOAL_INVALID_MAX_ROUNDS'
  | 'GOAL_INVALID_BLOCK_REASON'
  | 'GOAL_INVALID_EDIT'
  | 'GOAL_INVALID_TRANSITION'
  | 'GOAL_ROUNDS_EXHAUSTED'

export class GoalError extends Error {
  constructor(message: string, readonly code: GoalErrorCode) {
    super(message)
    this.name = 'GoalError'
  }
}

/** Default cap when a create omits one. Deliberately finite. */
export const DEFAULT_MAX_GOAL_ROUNDS = 24
/** Objectives longer than this are truncated before they reach the log. */
export const MAX_OBJECTIVE_CHARS = 4_000
/** Bounded history: a session cannot grow its metadata without limit. */
export const MAX_GOAL_CHANGES = 256

const GOAL_CHANGES_KEY = 'goal_changes'

/**
 * Process-local activation, keyed by session.
 *
 * Never persisted, and never inherited by a fresh process: a resumed or forked
 * session comes back disarmed so it cannot silently continue autonomous work
 * that a human has not re-authorised. This is the single most important
 * difference from the phase, which IS durable.
 */
const activations = new Map<string, GoalActivation>()

/** Read the change log from session metadata. */
export function readGoalChanges(metadata: Readonly<Record<string, unknown>>): readonly GoalChange[] {
  const raw = metadata[GOAL_CHANGES_KEY]
  if (!Array.isArray(raw)) return []
  return raw.filter(isGoalChange)
}

/** Pure replay fold of durable goal facts. */
export interface FoldedGoal {
  readonly goal?: GoalSnapshot
  readonly roundsStarted: number
  readonly createdAt?: number
  readonly updatedAt?: number
  readonly lastRef?: GoalRef
}

/**
 * Fold the change log into current state, strictly.
 *
 * Last-wins over whole values: every non-clear change carries the complete
 * post-mutation snapshot, so a partial or reordered write cannot produce a
 * half-applied goal the way a field-by-field patch log could.
 *
 * Strict, because this log lives in session metadata — a file on disk that
 * survives crashes, gets copied between machines, and can be hand-edited. A
 * permissive fold would happily accept a log whose revisions skip, whose round
 * counter jumps past the cap, or which admits a round against a completed goal,
 * and the result is autonomous work running on state nobody can account for.
 * Every rejection below describes a log that could not have been produced by
 * this module's own mutations.
 */
export function foldGoalChanges(changes: readonly GoalChange[]): FoldedGoal {
  let folded: FoldedGoal = { roundsStarted: 0 }
  for (const [index, change] of changes.entries()) {
    // Compaction (see `append`) replaces an overlong prefix with the single
    // snapshot it folded to, so the log may legitimately open on a mid-life
    // change rather than a create. Only the first entry may do so.
    if (index === 0 && change.operation !== 'create' && change.operation !== 'clear') {
      folded = {
        goal: change.goal,
        roundsStarted: change.roundsStarted,
        createdAt: change.createdAt,
        updatedAt: change.updatedAt,
        lastRef: { id: change.goal.id, revision: change.goal.revision },
      }
      continue
    }
    if (change.operation === 'clear') {
      if (!folded.goal) rejectLog('clear with no current goal')
      if (change.cleared.id !== folded.goal.id) rejectLog('clear of a different goal')
      if (change.cleared.revision !== folded.goal.revision + 1) {
        rejectLog('clear must advance the revision by exactly one')
      }
      folded = { roundsStarted: 0, lastRef: change.cleared }
      continue
    }

    const { goal } = change
    if (change.operation === 'create') {
      if (folded.goal && folded.goal.phase !== 'complete') {
        rejectLog(`create over a goal in phase "${folded.goal.phase}"`)
      }
      if (goal.revision !== 1) rejectLog('create must start at revision 1')
      if (change.roundsStarted !== 0) rejectLog('create must start with zero rounds')
    } else {
      if (!folded.goal) rejectLog(`${change.operation} with no current goal`)
      if (goal.id !== folded.goal.id) rejectLog(`${change.operation} of a different goal`)
      if (goal.revision !== folded.goal.revision + 1) {
        rejectLog(`${change.operation} must advance the revision by exactly one`)
      }
      if (change.operation === 'round') {
        // The one operation that is not idempotent under replay: it is what
        // spends the budget, so its accounting is checked hardest.
        if (folded.goal.phase !== 'active') rejectLog('round admitted against a non-active goal')
        if (change.roundsStarted !== folded.roundsStarted + 1) rejectLog('round numbers must be consecutive')
        if (change.roundsStarted > goal.maxGoalRounds) rejectLog('round admitted past the cap')
      } else if (change.roundsStarted !== folded.roundsStarted) {
        rejectLog(`${change.operation} must not change the round count`)
      }
    }
    if (change.roundsStarted < 0) rejectLog('round count must not be negative')

    folded = {
      goal,
      roundsStarted: change.roundsStarted,
      createdAt: change.operation === 'create' ? change.createdAt : folded.createdAt ?? change.createdAt,
      updatedAt: change.updatedAt,
      lastRef: { id: goal.id, revision: goal.revision },
    }
  }
  return folded
}

/**
 * Refuse a change log this module could not have written.
 *
 * Thrown rather than repaired: a goal is the authority for unattended work, and
 * silently continuing from a best-guess reconstruction of it is strictly worse
 * than stopping and saying the state is not trustworthy.
 */
function rejectLog(detail: string): never {
  throw new GoalError(`goal change log is inconsistent: ${detail}`, 'GOAL_INVALID_TRANSITION')
}

/** Current goal for a session, or undefined before the first create / after a clear. */
export function getGoal(
  metadata: Readonly<Record<string, unknown>>,
  sessionId: string,
): GoalView | undefined {
  const folded = foldGoalChanges(readGoalChanges(metadata))
  if (!folded.goal) return undefined
  return Object.freeze({
    ...folded.goal,
    roundsStarted: folded.roundsStarted,
    createdAt: folded.createdAt ?? 0,
    updatedAt: folded.updatedAt ?? 0,
    activation: activations.get(sessionId) ?? 'disarmed',
  })
}

export interface CreateGoalRequest {
  readonly objective: string
  readonly maxGoalRounds?: number
}

export interface EditGoalRequest {
  readonly objective?: string
  readonly maxGoalRounds?: number
}

/**
 * Create and arm a goal.
 *
 * A completed goal may be replaced; every other phase must be cleared or
 * resumed instead, so a second create cannot silently discard work in flight.
 */
export function createGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  request: CreateGoalRequest,
  now: number,
): GoalView {
  const current = getGoal(metadata, sessionId)
  if (current && current.phase !== 'complete') {
    throw new GoalError(
      `goal "${current.id}" already exists with phase "${current.phase}"`,
      'GOAL_ALREADY_EXISTS',
    )
  }
  const objective = requireObjective(request.objective)
  const maxGoalRounds = requireMaxRounds(request.maxGoalRounds ?? DEFAULT_MAX_GOAL_ROUNDS)
  const snapshot: GoalSnapshot = {
    id: `goal_${now.toString(36)}${Math.trunc(now % 1_000).toString(36)}`,
    revision: 1,
    objective,
    phase: 'active',
    maxGoalRounds,
  }
  append(metadata, {
    kind: 'goal/change',
    version: 1,
    operation: 'create',
    goal: snapshotOf(snapshot),
    roundsStarted: 0,
    createdAt: now,
    updatedAt: now,
  })
  activations.set(sessionId, 'armed')
  return getGoal(metadata, sessionId)!
}

/** Edit objective and/or cap without changing phase. At least one field is required. */
export function editGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  request: EditGoalRequest,
  now: number,
): GoalView {
  const current = expectCurrent(metadata, sessionId, ref)
  if (request.objective === undefined && request.maxGoalRounds === undefined) {
    throw new GoalError('edit requires an objective or a max_goal_rounds', 'GOAL_INVALID_EDIT')
  }
  const objective = request.objective === undefined ? current.objective : requireObjective(request.objective)
  const maxGoalRounds = request.maxGoalRounds === undefined
    ? current.maxGoalRounds
    : requireMaxRounds(request.maxGoalRounds)
  return commit(metadata, sessionId, 'edit', { ...current, objective, maxGoalRounds }, undefined, now)
}

/** Hold an active goal without abandoning it. */
export function pauseGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  now: number,
): GoalView {
  const current = expectCurrent(metadata, sessionId, ref)
  assertPhase(current, 'pause', ['active'])
  return commit(metadata, sessionId, 'pause', withPhase(current, 'paused'), 'disarmed', now)
}

/**
 * Rearm a goal after a pause, a block, or a session resume.
 *
 * The rounds check is here rather than at continuation time so an exhausted
 * goal fails where a human can read the reason and raise the cap, instead of
 * silently never continuing.
 */
export function resumeGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  now: number,
): GoalView {
  const current = expectCurrent(metadata, sessionId, ref)
  assertPhase(current, 'resume', ['active', 'paused', 'blocked'])
  const folded = foldGoalChanges(readGoalChanges(metadata))
  if (current.phase === 'active' && (activations.get(sessionId) ?? 'disarmed') === 'armed') {
    throw new GoalError(`goal "${current.id}" is already active and armed`, 'GOAL_INVALID_TRANSITION')
  }
  if (folded.roundsStarted >= current.maxGoalRounds) {
    throw new GoalError(
      `goal "${current.id}" exhausted ${current.maxGoalRounds} goal rounds; raise max_goal_rounds before resuming`,
      'GOAL_ROUNDS_EXHAUSTED',
    )
  }
  return commit(metadata, sessionId, 'resume', withPhase(current, 'active'), 'armed', now)
}

/** Mark a goal complete and disarm it. */
export function completeGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  now: number,
): GoalView {
  const current = expectCurrent(metadata, sessionId, ref)
  assertPhase(current, 'complete', ['active', 'paused', 'blocked'])
  return commit(metadata, sessionId, 'complete', withPhase(current, 'complete'), 'disarmed', now)
}

/** Mark an active goal blocked, with a durable reason, and disarm it. */
export function blockGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  reason: GoalBlockReason,
  now: number,
): GoalView {
  const current = expectCurrent(metadata, sessionId, ref)
  assertPhase(current, 'block', ['active'])
  const blockedReason = requireBlockReason(reason)
  return commit(
    metadata,
    sessionId,
    'block',
    { ...withPhase(current, 'blocked'), blockedReason },
    'disarmed',
    now,
  )
}

/** Clear the current goal, retaining a tombstone so history stays readable. */
export function clearGoal(
  metadata: Record<string, unknown>,
  sessionId: string,
  ref: GoalRef,
  now: number,
): GoalRef {
  const current = expectCurrent(metadata, sessionId, ref)
  const cleared: GoalRef = { id: current.id, revision: current.revision + 1 }
  append(metadata, { kind: 'goal/change', version: 1, operation: 'clear', cleared, clearedAt: now })
  activations.delete(sessionId)
  return cleared
}

/**
 * Drop continuation authority without touching durable phase or revision.
 *
 * Used on session load, resume and fork: the goal keeps saying what it is,
 * while this process is no longer allowed to act on it unattended.
 */
export function disarmGoal(sessionId: string): void {
  if (activations.has(sessionId)) activations.set(sessionId, 'disarmed')
}

/** Test seam: forget every process-local activation. */
export function resetGoalActivations(): void {
  activations.clear()
}

/**
 * Admit the next continuation round.
 *
 * Returns the attribution the round's prompt carries, or undefined when the
 * goal is not eligible — not active, not armed, or out of capacity. Human
 * turns never call this, which is what keeps them from consuming the cap.
 */
export function admitGoalRound(
  metadata: Record<string, unknown>,
  sessionId: string,
  now: number,
): GoalMessageSource | undefined {
  const current = getGoal(metadata, sessionId)
  if (!current) return undefined
  if (current.phase !== 'active' || current.activation !== 'armed') return undefined
  if (current.roundsStarted >= current.maxGoalRounds) return undefined

  const round = current.roundsStarted + 1
  const changes = readGoalChanges(metadata)
  const folded = foldGoalChanges(changes)
  append(metadata, {
    kind: 'goal/change',
    version: 1,
    // A round re-commits the same phase under the next revision; the round
    // counter is what actually moves.
    operation: 'round',
    goal: snapshotOf({ ...current, revision: current.revision + 1 }),
    roundsStarted: round,
    createdAt: folded.createdAt ?? now,
    updatedAt: now,
  })
  return { kind: 'goal', goalId: current.id, revision: current.revision + 1, round }
}

// ── internals ──────────────────────────────────────────────────────────

/**
 * Append one change, compacting rather than truncating when the log is full.
 *
 * Dropping the oldest entries would be wrong here even though every entry
 * carries a whole snapshot: the strict fold verifies a revision chain, and a
 * log whose head has been cut is indistinguishable from one that was tampered
 * with. Because each entry IS a complete snapshot, the correct compaction is to
 * collapse the surviving prefix into the one snapshot it folds to and keep
 * appending from there — bounded metadata, intact chain, no lost current state.
 */
function append(metadata: Record<string, unknown>, change: GoalChange): void {
  const existing = readGoalChanges(metadata)
  const next = [...existing, change]
  if (next.length <= MAX_GOAL_CHANGES) {
    metadata[GOAL_CHANGES_KEY] = next
    return
  }
  const keep = next.slice(-(MAX_GOAL_CHANGES - 1))
  const baseline = baselineFor(next.slice(0, next.length - keep.length))
  metadata[GOAL_CHANGES_KEY] = baseline ? [baseline, ...keep] : keep
}

/** The single snapshot change that a compacted prefix folds to, if any. */
function baselineFor(prefix: readonly GoalChange[]): GoalSnapshotChange | undefined {
  const folded = foldGoalChanges(prefix)
  if (!folded.goal) return undefined
  return {
    kind: 'goal/change',
    version: 1,
    operation: 'edit',
    goal: folded.goal,
    roundsStarted: folded.roundsStarted,
    createdAt: folded.createdAt ?? 0,
    updatedAt: folded.updatedAt ?? 0,
  }
}

/**
 * Reduce a live view back to exactly the fields the durable log may carry.
 *
 * `GoalView` extends the snapshot with derived and process-local values —
 * `roundsStarted`, timestamps, and above all `activation`, which exists
 * precisely because it must NOT survive a restart. Spreading a view into a
 * change event writes all of them into the transcript, and a later fold would
 * then read a persisted activation as though a human had authorised it.
 */
function snapshotOf(goal: GoalSnapshot): GoalSnapshot {
  return {
    id: goal.id,
    revision: goal.revision,
    objective: goal.objective,
    phase: goal.phase,
    ...(goal.blockedReason ? { blockedReason: goal.blockedReason } : {}),
    maxGoalRounds: goal.maxGoalRounds,
  }
}

function commit(
  metadata: Record<string, unknown>,
  sessionId: string,
  operation: Exclude<GoalOperation, 'create' | 'clear'>,
  goal: GoalSnapshot,
  activation: GoalActivation | undefined,
  now: number,
): GoalView {
  const folded = foldGoalChanges(readGoalChanges(metadata))
  append(metadata, {
    kind: 'goal/change',
    version: 1,
    operation,
    goal: snapshotOf({ ...goal, revision: goal.revision + 1 }),
    roundsStarted: folded.roundsStarted,
    createdAt: folded.createdAt ?? now,
    updatedAt: now,
  })
  if (activation) activations.set(sessionId, activation)
  return getGoal(metadata, sessionId)!
}

function expectCurrent(
  metadata: Readonly<Record<string, unknown>>,
  sessionId: string,
  ref: GoalRef,
): GoalView {
  const current = getGoal(metadata, sessionId)
  if (!current) throw new GoalError('no current goal', 'GOAL_NOT_FOUND')
  if (current.id !== ref.id) {
    throw new GoalError(`goal "${ref.id}" is not the current goal`, 'GOAL_NOT_FOUND')
  }
  if (current.revision !== ref.revision) {
    throw new GoalError(
      `stale revision ${ref.revision}; current is ${current.revision}`,
      'GOAL_STALE_REVISION',
    )
  }
  return current
}

function assertPhase(
  current: GoalSnapshot,
  operation: GoalOperation,
  allowed: readonly GoalPhase[],
): void {
  if (allowed.includes(current.phase)) return
  throw new GoalError(
    `cannot ${operation} goal "${current.id}" from phase "${current.phase}"; expected ${allowed.join(' or ')}`,
    'GOAL_INVALID_TRANSITION',
  )
}

const withPhase = (goal: GoalSnapshot, phase: GoalPhase): GoalSnapshot => {
  // A phase that is no longer blocked must not keep carrying its reason.
  const { blockedReason: _dropped, ...rest } = goal
  return { ...rest, phase }
}

function requireObjective(value: string): string {
  const objective = value.trim()
  if (!objective) throw new GoalError('objective must be a non-empty string', 'GOAL_INVALID_OBJECTIVE')
  return objective.length > MAX_OBJECTIVE_CHARS
    ? `${objective.slice(0, MAX_OBJECTIVE_CHARS - 1)}…`
    : objective
}

function requireMaxRounds(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new GoalError('max_goal_rounds must be a positive integer', 'GOAL_INVALID_MAX_ROUNDS')
  }
  return value
}

function requireBlockReason(reason: GoalBlockReason): GoalBlockReason {
  const message = reason.message?.trim() ?? ''
  if (!message) throw new GoalError('blocked_reason must explain the blocker', 'GOAL_INVALID_BLOCK_REASON')
  const code = reason.code?.trim() || 'model-reported'
  return { code, message }
}

function isGoalChange(value: unknown): value is GoalChange {
  if (value === null || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  if (record.kind !== 'goal/change' || record.version !== 1) return false
  if (record.operation === 'clear') return isRef(record.cleared)
  return isSnapshot(record.goal) && typeof record.roundsStarted === 'number'
}

function isRef(value: unknown): value is GoalRef {
  if (value === null || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return typeof record.id === 'string' && typeof record.revision === 'number'
}

function isSnapshot(value: unknown): value is GoalSnapshot {
  if (!isRef(value)) return false
  const record = value as unknown as Record<string, unknown>
  return typeof record.objective === 'string'
    && typeof record.phase === 'string'
    && typeof record.maxGoalRounds === 'number'
}
