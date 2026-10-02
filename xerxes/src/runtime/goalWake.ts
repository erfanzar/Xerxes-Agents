// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Durable, single-slot wake state for a goal continuation. */

export const GOAL_WAKE_KEY = 'goal_wake'
export const GOAL_WAKE_VERSION = 1
/** Session marker an update restart leaves on each armed goal for the next process. */
export const GOAL_REARM_AFTER_RESTART_KEY = 'goal_rearm_after_restart'
const MAX_ID_CHARS = 200
const MAX_REASON_CHARS = 2_000

export type GoalWakeState = 'queued' | 'running' | 'settled' | 'interrupted' | 'cancelled'

export interface GoalWake {
  readonly version: 1
  readonly id: string
  readonly sessionId: string
  readonly goalId: string
  readonly revision: number
  readonly state: GoalWakeState
  readonly queuedAt: number
  readonly startedAt?: number
  readonly settledAt?: number
  readonly ownerId?: string
  readonly round?: number
  readonly reason?: string
}

export type GoalWakeErrorCode =
  | 'GOAL_WAKE_INVALID'
  | 'GOAL_WAKE_NOT_FOUND'
  | 'GOAL_WAKE_STALE'
  | 'GOAL_WAKE_TRANSITION'

export class GoalWakeError extends Error {
  constructor(message: string, readonly code: GoalWakeErrorCode) {
    super(message)
    this.name = 'GoalWakeError'
  }
}

export function readGoalWake(
  metadata: Readonly<Record<string, unknown>>,
  sessionId: string,
): GoalWake | undefined {
  const raw = metadata[GOAL_WAKE_KEY]
  if (raw === undefined) return undefined
  const wake = validateWake(raw)
  if (wake.sessionId !== requireIdentity(sessionId, 'sessionId')) {
    throw invalid('wake sessionId does not match the requested session')
  }
  return copyWake(wake)
}

export function queueGoalWake(
  metadata: Record<string, unknown>,
  sessionId: string,
  goalId: string,
  revision: number,
  now: number,
): GoalWake {
  const session = requireIdentity(sessionId, 'sessionId')
  const goal = requireIdentity(goalId, 'goalId')
  const nextRevision = requirePositiveInteger(revision, 'revision')
  const queuedAt = requireTime(now, 'queuedAt')
  const current = readGoalWake(metadata, session)
  if (current?.state === 'queued') {
    if (current.goalId === goal && current.revision === nextRevision) return current
    throw transition('cannot replace a queued wake')
  }
  if (current?.state === 'running') throw transition('cannot replace a running wake')
  const next: GoalWake = {
    version: GOAL_WAKE_VERSION,
    id: crypto.randomUUID(),
    sessionId: session,
    goalId: goal,
    revision: nextRevision,
    state: 'queued',
    queuedAt,
  }
  metadata[GOAL_WAKE_KEY] = next
  return copyWake(next)
}

export function claimGoalWake(
  metadata: Record<string, unknown>,
  sessionId: string,
  id: string,
  ownerId: string,
  round: number,
  now: number,
): GoalWake {
  const current = requireCurrent(metadata, sessionId, id)
  if (current.state !== 'queued') throw transition(`cannot claim wake in state "${current.state}"`)
  const owner = requireIdentity(ownerId, 'ownerId')
  const claimedRound = requirePositiveInteger(round, 'round')
  const startedAt = requireTime(now, 'startedAt')
  const next: GoalWake = { ...current, state: 'running', ownerId: owner, round: claimedRound, startedAt }
  metadata[GOAL_WAKE_KEY] = next
  return copyWake(next)
}

export function finishGoalWake(
  metadata: Record<string, unknown>,
  sessionId: string,
  id: string,
  ownerId: string,
  state: 'settled' | 'interrupted',
  reason: string | undefined,
  now: number,
): GoalWake {
  const current = requireCurrent(metadata, sessionId, id)
  if (current.state !== 'running') throw transition(`cannot finish wake in state "${current.state}"`)
  if (current.ownerId !== requireIdentity(ownerId, 'ownerId')) throw stale('wake owner does not match')
  const settledAt = requireTime(now, 'settledAt')
  const next: GoalWake = {
    ...current,
    state,
    settledAt,
    ...(reason === undefined ? {} : { reason: requireReason(reason) }),
  }
  metadata[GOAL_WAKE_KEY] = next
  return copyWake(next)
}

export function cancelGoalWake(
  metadata: Record<string, unknown>,
  sessionId: string,
  id: string,
  reason: string | undefined,
  now: number,
): GoalWake {
  const current = requireCurrent(metadata, sessionId, id)
  if (current.state !== 'queued') throw transition(`cannot cancel wake in state "${current.state}"`)
  const settledAt = requireTime(now, 'settledAt')
  const next: GoalWake = {
    ...current,
    state: 'cancelled',
    settledAt,
    ...(reason === undefined ? {} : { reason: requireReason(reason) }),
  }
  metadata[GOAL_WAKE_KEY] = next
  return copyWake(next)
}

export function recoverGoalWake(
  metadata: Record<string, unknown>,
  sessionId: string,
  ownerId: string,
  now: number,
): GoalWake | undefined {
  const current = readGoalWake(metadata, sessionId)
  if (!current || current.state !== 'running') return current
  const owner = requireIdentity(ownerId, 'ownerId')
  if (current.ownerId === owner) return current
  const next: GoalWake = {
    ...current,
    state: 'interrupted',
    settledAt: requireTime(now, 'settledAt'),
    reason: 'wake owner was not present after restart',
  }
  metadata[GOAL_WAKE_KEY] = next
  return copyWake(next)
}

function requireCurrent(metadata: Record<string, unknown>, sessionId: string, id: string): GoalWake {
  const current = readGoalWake(metadata, sessionId)
  if (!current) throw new GoalWakeError('no goal wake is queued', 'GOAL_WAKE_NOT_FOUND')
  if (current.id !== requireIdentity(id, 'id')) throw stale('wake id does not match')
  return current
}

function validateWake(value: unknown): GoalWake {
  if (value === null || typeof value !== 'object') throw invalid('goal_wake must be an object')
  const record = value as Record<string, unknown>
  const keys = new Set(['version', 'id', 'sessionId', 'goalId', 'revision', 'state', 'queuedAt', 'startedAt', 'settledAt', 'ownerId', 'round', 'reason'])
  if (Object.keys(record).some(key => !keys.has(key))) throw invalid('goal_wake contains unknown fields')
  if (record.version !== GOAL_WAKE_VERSION) throw invalid('goal_wake version is unsupported')
  const id = requireWakeIdValue(record.id)
  const sessionId = requireStoredIdentity(record.sessionId, 'sessionId')
  const goalId = requireStoredIdentity(record.goalId, 'goalId')
  const revision = requirePositiveIntegerValue(record.revision, 'revision')
  const state = record.state
  if (state !== 'queued' && state !== 'running' && state !== 'settled' && state !== 'interrupted' && state !== 'cancelled') {
    throw invalid('goal_wake state is unsupported')
  }
  const queuedAt = requireTimeValue(record.queuedAt, 'queuedAt')
  const startedAt = optionalTime(record.startedAt, 'startedAt')
  const settledAt = optionalTime(record.settledAt, 'settledAt')
  const ownerId = optionalIdentity(record.ownerId, 'ownerId')
  const round = optionalPositiveInteger(record.round, 'round')
  const reason = optionalReason(record.reason)
  if (state === 'queued' && (startedAt !== undefined || settledAt !== undefined || ownerId !== undefined || round !== undefined || reason !== undefined)) throw invalid('queued wake has terminal or running fields')
  if (state === 'running' && (startedAt === undefined || ownerId === undefined || round === undefined || settledAt !== undefined || reason !== undefined)) throw invalid('running wake fields are incomplete')
  if ((state === 'settled' || state === 'interrupted') && (startedAt === undefined || settledAt === undefined || ownerId === undefined || round === undefined)) throw invalid('finished wake fields are incomplete')
  if (state === 'cancelled' && (settledAt === undefined || startedAt !== undefined || ownerId !== undefined || round !== undefined)) throw invalid('cancelled wake fields are invalid')
  return { version: 1, id, sessionId, goalId, revision, state, queuedAt, ...(startedAt === undefined ? {} : { startedAt }), ...(settledAt === undefined ? {} : { settledAt }), ...(ownerId === undefined ? {} : { ownerId }), ...(round === undefined ? {} : { round }), ...(reason === undefined ? {} : { reason }) }
}

function copyWake(wake: GoalWake): GoalWake { return { ...wake } }
function requireIdentity(value: unknown, field: string): string { return requireIdentityValue(value, field) }
function requireIdentityValue(value: unknown, field: string): string { if (typeof value !== 'string' || !value.trim() || value.trim().length > MAX_ID_CHARS) throw invalid(`${field} must be a non-empty bounded string`); return value.trim() }
function requireStoredIdentity(value: unknown, field: string): string { if (typeof value !== 'string' || value !== value.trim() || !value || value.length > MAX_ID_CHARS) throw invalid(`${field} must be a non-empty bounded string`); return value }
function requireWakeIdValue(value: unknown): string { if (typeof value !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value)) throw invalid('id must be a UUID'); return value }
function requirePositiveInteger(value: unknown, field: string): number { return requirePositiveIntegerValue(value, field) }
function requirePositiveIntegerValue(value: unknown, field: string): number { if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) throw invalid(`${field} must be a positive safe integer`); return value }
function requireTime(value: unknown, field: string): number { return requireTimeValue(value, field) }
function requireTimeValue(value: unknown, field: string): number { if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) throw invalid(`${field} must be a non-negative safe integer`); return value }
function optionalTime(value: unknown, field: string): number | undefined { if (value === undefined) return undefined; return requireTimeValue(value, field) }
function optionalIdentity(value: unknown, field: string): string | undefined { if (value === undefined) return undefined; return requireIdentityValue(value, field) }
function optionalPositiveInteger(value: unknown, field: string): number | undefined { if (value === undefined) return undefined; return requirePositiveIntegerValue(value, field) }
function requireReason(value: string): string { return optionalReason(value) ?? (() => { throw invalid('reason must be non-empty and bounded') })() }
function optionalReason(value: unknown): string | undefined { if (value === undefined) return undefined; if (typeof value !== 'string' || !value.trim() || value.trim().length > MAX_REASON_CHARS) throw invalid('reason must be non-empty and bounded'); return value.trim() }
function invalid(message: string): GoalWakeError { return new GoalWakeError(message, 'GOAL_WAKE_INVALID') }
function stale(message: string): GoalWakeError { return new GoalWakeError(message, 'GOAL_WAKE_STALE') }
function transition(message: string): GoalWakeError { return new GoalWakeError(message, 'GOAL_WAKE_TRANSITION') }
