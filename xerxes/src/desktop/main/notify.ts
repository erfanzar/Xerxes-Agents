// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Which daemon events deserve a native ping, and what it should say.
 * Pure logic — no Electron imports — so the decision table is testable
 * headlessly while the display side stays in main.ts.
 */

import { turnOutcomeLabel, turnOutcomeReason } from '../../types/turnOutcome.js'

export interface NotifyEvent {
  readonly type: string
  readonly payload: Record<string, unknown>
}

/** Where a click on the notification takes the user. */
export interface NotifyTarget {
  readonly sessionId?: string
  readonly panel?: 'docs'
  readonly artifactId?: string
}

export interface NotifyDecision {
  readonly title: string
  readonly body: string
  readonly target?: NotifyTarget
}

const ARTIFACT_RESULT = /^(Created|Updated) artifact "(.*)" \(id ([A-Za-z0-9_-]+), version (\d+)\)/s

/** The click target for an event: its session, plus anything more specific. */
function targetOf(payload: Record<string, unknown>, extra: Omit<NotifyTarget, 'sessionId'> = {}): NotifyTarget {
  const sessionId = text(payload.session_id)
  return { ...(sessionId ? { sessionId } : {}), ...extra }
}

const text = (value: unknown): string => (typeof value === 'string' ? value.trim() : '')

const truncate = (value: string, max = 140): string =>
  value.length > max ? `${value.slice(0, max - 1)}…` : value

/**
 * The notification a daemon event earns, or null when it should stay quiet.
 * Needs-input events (approval, question) and turn completion are the two
 * moments a user minimizes the app for.
 */
export function notificationFor(event: NotifyEvent): NotifyDecision | null {
  const payload = event.payload ?? {}
  switch (event.type) {
    case 'turn_end': {
      // Mirrors the TUI bell policy: an interrupted or never-started turn is
      // not a completion, and pinging success for one (a goal-mode turn the
      // time guard cut off, an unstarted submission) tells the user the
      // opposite of what happened.
      if (payload.unstarted === true) return null
      const reason = turnOutcomeReason(payload.stop_reason)
      if (payload.cancelled === true || reason === 'aborted') {
        return { title: 'Task stopped', body: 'The current task was interrupted before it finished.', target: targetOf(payload) }
      }
      if (reason === 'provider_failed' || reason === 'turn_failed' || payload.failed === true || text(payload.error)) {
        return { title: 'Task failed', body: truncate(text(payload.error) || 'The current task ended with an error.'), target: targetOf(payload) }
      }
      if (reason && reason !== 'completed' && reason !== 'objective_verified') {
        return { title: 'Task stopped', body: turnOutcomeLabel(reason), target: targetOf(payload) }
      }
      return { title: 'Task finished', body: 'The current task completed.', target: targetOf(payload) }
    }
    case 'tool_result': {
      // A saved artifact: clicking opens it, so nobody has to go looking for it.
      if (text(payload.name) !== 'ArtifactTool' || payload.permitted === false) return null
      const match = ARTIFACT_RESULT.exec(text(payload.result))
      if (!match) return null
      const updated = match[1] === 'Updated'
      return {
        title: updated ? 'Artifact updated' : 'New artifact',
        body: truncate(`${match[2]}${updated ? ` (v${match[4]})` : ''} — click to read it`),
        target: targetOf(payload, { panel: 'docs', artifactId: match[3]! }),
      }
    }
    case 'approval_request': {
      // Leaving plan mode is the plan review, not a tool permission.
      if (/^ExitPlanMode(Tool)?$/.test(text(payload.tool_name) || text(payload.name) || text(payload.action))) {
        return { title: 'Plan ready', body: 'Continue with the plan or refine it.', target: targetOf(payload) }
      }
      const description = text(payload.description) ||
        `${text(payload.name)} ${text(payload.tool_name)}`.trim()
      if (!description) return null
      return { title: 'Approval needed', body: truncate(description), target: targetOf(payload) }
    }
    case 'question_request': {
      const questions = Array.isArray(payload.questions) ? payload.questions : []
      const first = questions.find(
        (item): item is Record<string, unknown> =>
          item !== null && typeof item === 'object' && text((item as Record<string, unknown>).question) !== '',
      )
      if (!first) return null
      return { title: 'Xerxes has a question', body: truncate(text(first.question)), target: targetOf(payload) }
    }
    default:
      return null
  }
}

/**
 * A goal reaching complete or blocked. `status_update` repeats the phase on
 * every turn, so only a change from the previous phase seen for that session counts.
 */
export function goalPhaseNotification(previous: string | undefined, payload: Record<string, unknown>): NotifyDecision | null {
  const phase = text(payload.goal_phase)
  if (!phase || phase === previous) return null
  const goal = truncate(text(payload.goal) || 'The goal')
  if (phase === 'complete') return { title: 'Goal complete', body: goal, target: targetOf(payload) }
  if (phase === 'blocked') return { title: 'Goal blocked — needs you', body: goal, target: targetOf(payload) }
  return null
}

/** The gate around the display: preference first, then focus. */
export function shouldNotify(
  event: NotifyEvent,
  state: { readonly enabled: boolean; readonly anyWindowFocused: boolean },
  previousGoalPhase?: string,
): NotifyDecision | null {
  if (!state.enabled) return null
  // Task outcomes and requests for attention matter even while working in the app.
  // Artifact saves, by contrast, already appear in the workspace.
  if (state.anyWindowFocused && event.type === 'tool_result') return null
  return event.type === 'status_update'
    ? goalPhaseNotification(previousGoalPhase, event.payload)
    : notificationFor(event)
}
