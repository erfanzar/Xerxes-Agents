// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { DenialRecord } from './denialBudget.js'

/**
 * Typed catalog of the turn loop's model-facing interventions.
 *
 * These strings steer behavior under failure: they stop refusal loops, resume
 * after truncation without an apology lap, and hold objective mode to its
 * verification contract. Wording is a behavioral contract — the daemon filters
 * on prefixes like `[Objective gate]` (`daemon/server.ts`) and the TUI adapter
 * classifies rows by the same markers (`ui/gatewayAdapter.ts`) — so every
 * rendering is centralized here and pinned byte-for-byte by
 * `test/runtimeInterventions.test.ts`. Change a rendered string only together
 * with those prefix consumers.
 */

/** Terminal guards that end a turn instead of letting a failing pattern loop. */
export type StopGuardVariant =
  | 'context-overflow'
  | 'objective-guard-exhausted'
  | 'output-limit-escalated'
  | 'unconfigured-tools-loop'

export type InterventionKind =
  | 'denial-budget'
  | 'objective-reminder'
  | 'resume-directive'
  | 'steer-note'
  | 'stop-guard'

export type Intervention =
  | {
    readonly kind: 'denial-budget'
    /** Most recent refusal, rendered when present so the stop names the rule that refused what. */
    readonly lastDenial?: DenialRecord
    readonly used: number
  }
  | {
    readonly kind: 'objective-reminder'
    /** Grounds from the objective guard; interpolated into the reminder verbatim. */
    readonly reason: string
  }
  | {
    readonly directive: 'output-limit'
    readonly kind: 'resume-directive'
  }
  | {
    readonly content: string
    readonly kind: 'steer-note'
  }
  | {
    readonly kind: 'stop-guard'
    /**
     * Round or retry count for the loop-shaped guards
     * (`output-limit-escalated`, `unconfigured-tools-loop`,
     * `objective-guard-exhausted`).
     */
    readonly attempts?: number
    /** Grounds reported by the guarding subsystem (`objective-guard-exhausted`). */
    readonly reason?: string
    readonly variant: StopGuardVariant
  }

/** Resume directive pushed after a truncation that a larger window did not fix. */
export function renderOutputLimitResumeDirective(): string {
  return '[Output limit]\nOutput token limit hit. Resume directly — no apology, no recap.'
}

/**
 * Terminal wording for a context overflow no reducer could relieve. The
 * provider's own string names a token count the user cannot act on, so echoing
 * it leaves the session repeating an identical failure; these three commands
 * are the actual remedies.
 */
export function renderContextOverflowStopGuard(): string {
  return '[Stopped: the conversation no longer fits in this model\'s context window. '
    + 'Run /compact to summarize it, /clear to start over, or /branch to keep this '
    + 'history and continue in a fresh session.]'
}

/** Render one intervention exactly as the prior inline call sites produced it. */
export function renderIntervention(intervention: Intervention): string {
  switch (intervention.kind) {
    case 'resume-directive':
      return renderOutputLimitResumeDirective()
    case 'objective-reminder':
      return renderObjectiveReminder(intervention.reason)
    case 'steer-note':
      return `\n[Steer saved for next turn: ${intervention.content}]`
    case 'stop-guard':
      return renderStopGuard(intervention)
    case 'denial-budget':
      return renderDenialBudgetStop(intervention.used, intervention.lastDenial)
  }
}

function renderObjectiveReminder(reason: string): string {
  return '[Objective gate]\n'
    + 'The previous assistant response tried to stop, but objective mode is still active: ' + reason + '.\n'
    + 'Continue the hard-goal loop. Do not final-answer with a narrative status. Update the ledger, '
    + 'choose the next concrete hypothesis, use tools to edit or verify, and only end after all acceptance '
    + 'criteria pass or after you report BLOCKED: with exact evidence.'
}

function renderDenialBudgetStop(used: number, lastDenial: DenialRecord | undefined): string {
  const detail = lastDenial === undefined
    ? ''
    : ` The last refusal was ${describeDenialKind(lastDenial.kind)} on ${lastDenial.toolName}.`
  return (
    `\n[Stopped: ${used} consecutive tool calls were refused with no successful tool `
    + `execution in between; ending the turn instead of retrying a refusal loop.${detail}]`
  )
}

function renderStopGuard(
  intervention: Extract<Intervention, { readonly kind: 'stop-guard' }>,
): string {
  switch (intervention.variant) {
    case 'context-overflow':
      return renderContextOverflowStopGuard()
    case 'output-limit-escalated':
      return `\n[Stopped: the model hit the output token limit in `
        + `${intervention.attempts ?? 0} consecutive rounds; ending the turn instead of `
        + `resuming again.]`
    case 'unconfigured-tools-loop':
      return `\n[Stopped: the model requested only unconfigured tools in `
        + `${intervention.attempts ?? 0} consecutive rounds; ending the turn `
        + `instead of looping on provider calls.]`
    case 'objective-guard-exhausted':
      return '\n[Stopped: objective guard could not get a verified completion or concrete blocker after '
        + `${intervention.attempts ?? 0} retries. The last issue was: ${intervention.reason ?? ''}.]`
  }
}

function describeDenialKind(kind: DenialRecord['kind']): string {
  switch (kind) {
    case 'cancelled':
      return 'a cancellation'
    case 'permission_rejected':
      return 'a rejected permission prompt'
    case 'policy_denied':
      return 'a policy denial'
  }
}
