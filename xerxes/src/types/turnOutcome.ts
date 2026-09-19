// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const TURN_STOP_REASONS = ['aborted', 'completed', 'context_overflow', 'objective_guard_exhausted',
  'objective_verified', 'output_limit', 'provider_failed', 'tool_budget_exhausted', 'turn_failed', 'unconfigured_tools'] as const
export type TurnOutcomeReason = typeof TURN_STOP_REASONS[number]
export interface TurnOutcome { readonly version: 1; readonly reason: TurnOutcomeReason; readonly turn_id?: string }

/** Presentation metadata travels beside native messages, never in their JSON
 * model payload. The daemon explicitly projects it into stored/wire records. */
const outcomes = new WeakMap<object, TurnOutcome>()
export function turnOutcomeReason(value: unknown): TurnOutcomeReason | undefined {
  return typeof value === 'string' && (TURN_STOP_REASONS as readonly string[]).includes(value) ? value as TurnOutcomeReason : undefined
}
export function readTurnOutcome(value: unknown): TurnOutcome | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return
  const raw = value as Record<string, unknown>
  const reason = turnOutcomeReason(raw.reason)
  const id = raw.turn_id
  if (id !== undefined && (typeof id !== 'string' || !/^[a-f0-9]{8,64}$/.test(id))) return
  if (raw.version === 1 && reason && Object.keys(raw).every(key => ['version', 'reason', 'turn_id'].includes(key))) return { version: 1, reason, ...(typeof id === 'string' ? { turn_id: id } : {}) }
}
export function setTurnOutcome(message: object, reason: TurnOutcomeReason, turnId?: string): void {
  outcomes.set(message, { version: 1, reason, ...(turnId && /^[a-f0-9]{8,64}$/.test(turnId) ? { turn_id: turnId } : {}) })
}
export function getTurnOutcome(message: object): TurnOutcome | undefined { return outcomes.get(message) }
export function copyTurnOutcome<T extends object>(source: object, target: T): T {
  const outcome = getTurnOutcome(source)
  if (outcome) outcomes.set(target, outcome)
  return target
}
export function restoreTurnOutcome<T extends object>(source: Record<string, unknown>, message: T): T {
  const outcome = readTurnOutcome(source.turn_outcome)
  if (outcome) outcomes.set(message, outcome)
  return message
}
export function turnOutcomeLabel(reason?: TurnOutcomeReason): string {
  switch (reason) {
    case 'completed': case 'objective_verified': return 'completed'
    case 'aborted': return 'interrupted'
    case 'provider_failed': return 'failed · provider request'
    case 'turn_failed': return 'failed'
    case 'context_overflow': return 'stopped · context full'
    case 'output_limit': return 'stopped · output limit'
    case 'tool_budget_exhausted': return 'stopped · tool budget'
    case 'objective_guard_exhausted': return 'stopped · objective needs attention'
    case 'unconfigured_tools': return 'stopped · tool unavailable'
    default: return 'ended · outcome unknown'
  }
}
