// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { normalizeInteractionMode } from './interactionModes.js'

export { normalizeInteractionMode } from './interactionModes.js'

const DEFAULT_RETRY_LIMIT = 6

const VERIFIED_SUCCESS_MARKERS = [
  'acceptance criteria pass',
  'all acceptance criteria pass',
  'objective met',
  'verified complete',
  'verified completion',
  'all tests pass',
  'all benchmarks pass',
  'all checks pass',
] as const

const UNRESOLVED_MARKERS = [
  '❌',
  'not met',
  'unmet',
  'not yet',
  'still fails',
  'still failing',
  'still fail',
  'still loses',
  'still losing',
  'still lose',
  'cannot beat',
  "can't beat",
  'does not pass',
  'do not pass',
  'failing benchmark',
  'failing test',
  'remaining failure',
  'remaining issue',
  'losses',
  'loses by',
] as const

const RUNAWAY_FINAL_MARKERS = [
  'want me to',
  'should i',
  'would you like',
  'do you want me',
  'honest final',
  'final state',
  'where we stand',
  'where we are',
  'path forward is',
  'next step is',
] as const

const BLOCKER_MARKERS = [
  'blocked:',
  'concrete blocker',
  'externally blocked',
  'cannot proceed because',
] as const

const BLOCKER_EVIDENCE_MARKERS = [
  'evidence:',
  'command:',
  'stderr',
  'traceback',
  'permission denied',
  'not installed',
  'missing dependency',
  'requires user',
] as const

export interface ObjectiveGuardDecision {
  readonly reason: string
  readonly reminder: string
  readonly shouldContinue: boolean
}

export interface ObjectiveGuardRetryOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
}

/** Return the configured objective retry ceiling with injectable environment lookup. */
export function objectiveGuardRetryLimit(
  config: Readonly<Record<string, unknown>>,
  options: ObjectiveGuardRetryOptions = {},
): number {
  const environment = options.environment ?? process.env
  const configured = config.objective_guard_max_retries
  const raw = configured ? configured : environment.XERXES_OBJECTIVE_GUARD_MAX_RETRIES
  const parsed = integerValue(raw)
  return parsed === undefined ? DEFAULT_RETRY_LIMIT : Math.max(1, parsed)
}

/**
 * Decide whether a no-tool answer must continue objective-mode execution.
 *
 * Objective mode may end only with an uncontradicted verified-success marker,
 * or a concrete blocker accompanied by observable evidence.
 */
export function inspectObjectiveResponse(
  text: string,
  options: { readonly mode: unknown; readonly planMode?: boolean },
): ObjectiveGuardDecision {
  if (normalizeInteractionMode(options.mode, options.planMode ?? false) !== 'objective') return allowStop()
  const stripped = text.trim()
  if (!stripped) return allowStop()

  const lowered = stripped.toLowerCase()
  const success = firstMarker(lowered, VERIFIED_SUCCESS_MARKERS)
  const unresolved = firstMarker(lowered, UNRESOLVED_MARKERS)
  if (success && !unresolved) return allowStop()

  const blocker = firstMarker(lowered, BLOCKER_MARKERS)
  const evidence = firstMarker(lowered, BLOCKER_EVIDENCE_MARKERS)
  if (blocker && evidence) return allowStop()

  const runaway = firstMarker(lowered, RUNAWAY_FINAL_MARKERS)
  const reason = unresolved
    ? 'unresolved acceptance marker ' + quote(unresolved)
    : runaway
      ? 'premature stopping marker ' + quote(runaway)
      : 'no verified completion or concrete blocker evidence'
  return Object.freeze({
    shouldContinue: true,
    reason,
    reminder: objectiveReminder(reason),
  })
}

function allowStop(): ObjectiveGuardDecision {
  return Object.freeze({ shouldContinue: false, reason: '', reminder: '' })
}

function firstMarker(text: string, markers: readonly string[]): string {
  for (const marker of markers) {
    if (text.includes(marker)) return marker
  }
  return ''
}

function integerValue(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isSafeInteger(value)) return value
  if (typeof value === 'string' && /^[-+]?\d+$/.test(value.trim())) {
    const parsed = Number(value)
    return Number.isSafeInteger(parsed) ? parsed : undefined
  }
  return undefined
}

function quote(value: string): string {
  return String.fromCharCode(96) + value + String.fromCharCode(96)
}

function objectiveReminder(reason: string): string {
  return '[Objective gate]\n'
    + 'The previous assistant response tried to stop, but objective mode is still active: ' + reason + '.\n'
    + 'Continue the hard-goal loop. Do not final-answer with a narrative status. Update the ledger, '
    + 'choose the next concrete hypothesis, use tools to edit or verify, and only end after all acceptance '
    + 'criteria pass or after you report BLOCKED: with exact evidence.'
}
