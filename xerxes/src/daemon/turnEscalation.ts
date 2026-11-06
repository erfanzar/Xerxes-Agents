// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Escalation by rule (M1.1) for the daemon turn runner: the cascade.
 *
 * After a turn's result checks run, the session's EscalationController decides
 * what happens next from the findings, never from the model's opinion of its
 * own work: retry locally, consult the cloud advisor, or stop. The runner acts
 * on the step inside the same user turn: it re-runs the model with the
 * feedback (and, for `ask_advisor`, consults the advisor itself and passes on
 * the answer), then checks again. The ladder lives per session, so retries are
 * counted across turns, and a clean check resets it.
 *
 * `runtime.escalation.mode` or XERXES_ESCALATION turns it on or off; unset,
 * the host's `defaultMode` decides per session (on for local models).
 *
 * `change_model` is not offered here: switching models mid-task is the
 * router's decision, so the ladder goes retry → advisor → stop.
 */

import type { CheckFinding } from '../runtime/resultChecks.js'
import { renderFindings } from '../runtime/resultChecks.js'
import { EscalationController, parseEscalationConfig } from '../runtime/escalationPolicy.js'
import type { DataBoundaryMode } from '../security/boundaryTypes.js'

export interface EscalationNotice {
  readonly action: 'retry_local' | 'change_model' | 'ask_advisor' | 'stop'
  readonly level: 'info' | 'warning'
  /** Shown to the user. */
  readonly message: string
  /** Delivered to the model as a harness message; absent for `stop`. */
  readonly feedback?: string
  /** For `ask_advisor`: what the runner sends the advisor itself. */
  readonly advisorRequest?: { readonly question: string; readonly context: string }
}

export interface EscalationFactoryOptions<S extends { readonly id: string; readonly cwd: string }> {
  readonly runtime: () => Record<string, unknown>
  readonly env?: Record<string, string | undefined>
  readonly boundaryMode: (session: S) => DataBoundaryMode
  /** The session's context holds private data (boundary taint). */
  readonly touchesPrivate?: (session: S) => boolean
  /** True when AskAdvisor is registered (an advisor profile is configured). */
  readonly advisorAvailable: () => boolean
  /** The mode when neither runtime.escalation.mode nor XERXES_ESCALATION is set. Default off. */
  readonly defaultMode?: (session: S) => 'off' | 'on'
}

const MAX_TRACKED_SESSIONS = 500

/**
 * Build the runner's `escalationForSession`. Returns undefined when escalation
 * is off, when nothing failed and no ladder was running, or after a clean
 * check (which also resets that session's ladder).
 */
export function createEscalationFactory<S extends { readonly id: string; readonly cwd: string }>(
  options: EscalationFactoryOptions<S>,
): (session: S, findings: readonly CheckFinding[]) => EscalationNotice | undefined {
  const controllers = new Map<string, EscalationController>()
  return (session, findings) => {
    const config = parseEscalationConfig(options.runtime(), options.env ?? process.env, options.defaultMode?.(session) ?? 'off')
    if (config.mode === 'off') {
      controllers.delete(session.id)
      return undefined
    }
    const errors = findings.filter(finding => finding.severity === 'error')
    if (!errors.length) {
      // Clean (or warn-only) result: the chain is accepted and starts over.
      controllers.delete(session.id)
      return undefined
    }
    let controller = controllers.get(session.id)
    if (controller === undefined) {
      controller = new EscalationController({
        config,
        boundaryMode: options.boundaryMode(session),
        touchesPrivate: options.touchesPrivate?.(session) ?? false,
        advisorAvailable: options.advisorAvailable(),
        alternateModelAvailable: false,
      })
      controllers.set(session.id, controller)
      while (controllers.size > MAX_TRACKED_SESSIONS) {
        const oldest = controllers.keys().next().value
        if (oldest === undefined) break
        controllers.delete(oldest)
      }
    }
    const step = controller.next({ findings })
    const summary = controller.summary()
    switch (step.action) {
      case 'accept':
        controllers.delete(session.id)
        return undefined
      case 'retry_local':
        return {
          action: 'retry_local',
          level: 'info',
          message: `Escalation: checks failed, retrying locally (${summary.localRetries} of ${config.maxLocalRetries}).`,
          feedback: `${step.feedback}\nFix these and run a check that proves the fix before you report the task as done.`,
        }
      case 'change_model':
        return {
          action: 'change_model',
          level: 'info',
          message: 'Escalation: checks keep failing; a stronger model is suggested (switch with /model).',
          feedback: step.feedback,
        }
      case 'ask_advisor':
        return {
          action: 'ask_advisor',
          level: 'info',
          message: `Escalation: checks keep failing after local retries; consulting the advisor (${summary.advisorCalls} of ${config.maxAdvisorCalls}).`,
          // Used only when the runner cannot consult the advisor itself.
          feedback: [
            'Local retries did not fix the failed checks. Call AskAdvisor once, with exactly this question and context, then apply its answer:',
            `question: ${step.question}`,
            `context: ${step.context}`,
          ].join('\n'),
          advisorRequest: { question: step.question, context: step.context },
        }
      case 'stop':
        controllers.delete(session.id)
        return {
          action: 'stop',
          level: 'warning',
          message: `Escalation stopped: ${step.reason}. ${renderFindings(step.findings)}`,
        }
    }
  }
}
