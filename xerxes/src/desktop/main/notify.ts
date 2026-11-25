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

export type NotifyKind = 'artifact' | 'attention' | 'finished' | 'stopped' | 'failed'

export interface NotifyDecision {
  readonly title: string
  readonly body: string
  readonly kind?: NotifyKind
  readonly urgent?: boolean
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

function withSessionPrefix(sessionTitle: string, detail: string): string {
  if (!sessionTitle) return truncate(detail)
  if (detail.toLowerCase().includes(sessionTitle.toLowerCase())) return truncate(detail)
  return truncate(`${sessionTitle} — ${detail}`)
}

function friendlyArtifactLabel(artifactId: string, payload: Record<string, unknown>): string {
  const explicit = text(payload.artifact_title)
  if (explicit) return explicit
  const raw = artifactId.startsWith('agy:') ? artifactId.slice(4) : artifactId
  const base = raw.split('/').filter(Boolean).at(-1) || raw
  return base.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim() || base
}

/** Unwrap a `session_attention` observer broadcast into its inner event + merged session context. */
export function unwrapAttentionEvent(event: NotifyEvent): NotifyEvent {
  if (event.type !== 'session_attention') return event
  const outer = event.payload ?? {}
  const innerType = text(outer.event_type)
  const rawInner = outer.payload
  const innerPayload: Record<string, unknown> =
    rawInner && typeof rawInner === 'object' && !Array.isArray(rawInner)
      ? { ...(rawInner as Record<string, unknown>) }
      : {}
  if (!text(innerPayload.session_id) && text(outer.session_id)) {
    innerPayload.session_id = text(outer.session_id)
  }
  if (!text(innerPayload.session_title) && text(outer.session_title)) {
    innerPayload.session_title = text(outer.session_title)
  }
  return { type: innerType || event.type, payload: innerPayload }
}

/**
 * The notification a daemon event earns, or null when it should stay quiet.
 * Needs-input events (approval, question), saved artifacts, and turn completion /
 * stop / failure are the moments a user relies on alerts for.
 */
export function notificationFor(rawEvent: NotifyEvent): NotifyDecision | null {
  const event = unwrapAttentionEvent(rawEvent)
  const payload = event.payload ?? {}
  const sessionTitle = text(payload.session_title)
  switch (event.type) {
    case 'turn_end': {
      // Mirrors the TUI bell policy: an interrupted or never-started turn is
      // not a completion, and pinging success for one (a goal-mode turn the
      // time guard cut off, an unstarted submission) tells the user the
      // opposite of what happened.
      if (payload.unstarted === true) return null
      const reason = turnOutcomeReason(payload.stop_reason)
      if (payload.cancelled === true || reason === 'aborted') {
        return {
          title: 'Task stopped',
          body: sessionTitle
            ? withSessionPrefix(sessionTitle, 'Interrupted before finishing.')
            : 'The current task was interrupted before it finished.',
          kind: 'stopped',
          target: targetOf(payload),
        }
      }
      if (reason === 'provider_failed' || reason === 'turn_failed' || payload.failed === true || text(payload.error)) {
        const detail = text(payload.error) || 'The current task ended with an error.'
        return {
          title: 'Task failed',
          body: withSessionPrefix(sessionTitle, detail),
          kind: 'failed',
          urgent: true,
          target: targetOf(payload),
        }
      }
      if (reason && reason !== 'completed' && reason !== 'objective_verified') {
        return {
          title: 'Task stopped',
          body: withSessionPrefix(sessionTitle, turnOutcomeLabel(reason)),
          kind: 'stopped',
          target: targetOf(payload),
        }
      }
      return {
        title: 'Task finished',
        body: sessionTitle ? withSessionPrefix(sessionTitle, 'Completed.') : 'The current task completed.',
        kind: 'finished',
        target: targetOf(payload),
      }
    }
    case 'tool_result': {
      if (payload.permitted === false || text(payload.error)) return null
      // 1. Native ArtifactTool saves (live daemon emits both `return_value` and `result`;
      //    some bridges emit only `return_value`).
      if (text(payload.name) === 'ArtifactTool') {
        const output = text(payload.result) || text(payload.return_value)
        const match = ARTIFACT_RESULT.exec(output)
        if (!match) return null
        const updated = match[1] === 'Updated'
        return {
          title: updated ? 'Artifact updated' : 'New artifact',
          body: truncate(`${match[2]}${updated ? ` (v${match[4]})` : ''} — click to read it`),
          kind: 'artifact',
          target: targetOf(payload, { panel: 'docs', artifactId: match[3]! }),
        }
      }
      // 2. External ACP / Antigravity artifacts surfaced via `artifact_id` (e.g. `agy:plan.md`).
      const artifactId = text(payload.artifact_id)
      if (!artifactId) return null
      const toolLabel = `${text(payload.name)} ${text(payload.title)}`.toLowerCase()
      const updated = /\b(?:edit|update|replace|modify|patch)\b/.test(toolLabel)
      const label = friendlyArtifactLabel(artifactId, payload)
      return {
        title: updated ? 'Artifact updated' : 'New artifact',
        body: truncate(`${label} — click to read it`),
        kind: 'artifact',
        target: targetOf(payload, { panel: 'docs', artifactId }),
      }
    }
    case 'approval_request': {
      // Leaving plan mode is the plan review, not a tool permission.
      if (/^ExitPlanMode(Tool)?$/.test(text(payload.tool_name) || text(payload.name) || text(payload.action))) {
        return {
          title: 'Plan ready',
          body: sessionTitle
            ? withSessionPrefix(sessionTitle, 'Continue with the plan or refine it.')
            : 'Continue with the plan or refine it.',
          kind: 'attention',
          urgent: true,
          target: targetOf(payload),
        }
      }
      const description = text(payload.description) ||
        text(payload.title) ||
        text(payload.summary) ||
        `${text(payload.name)} ${text(payload.tool_name)}`.trim() ||
        text(payload.action)
      if (!description) return null
      return {
        title: 'Approval needed',
        body: withSessionPrefix(sessionTitle, description),
        kind: 'attention',
        urgent: true,
        target: targetOf(payload),
      }
    }
    case 'question_request': {
      const questions = Array.isArray(payload.questions) ? payload.questions : []
      let prompt = ''
      for (const item of questions) {
        if (item === null || typeof item !== 'object') continue
        const rec = item as Record<string, unknown>
        const candidate = text(rec.question) || text(rec.prompt) || text(rec.label) || text(rec.title)
        if (candidate) {
          prompt = candidate
          break
        }
      }
      if (!prompt) {
        prompt = text(payload.prompt) || text(payload.question) || text(payload.message) || text(payload.title)
      }
      if (!prompt) return null
      return {
        title: 'Xerxes has a question',
        body: withSessionPrefix(sessionTitle, prompt),
        kind: 'attention',
        urgent: true,
        target: targetOf(payload),
      }
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
  if (phase === 'complete') return { title: 'Goal complete', body: goal, kind: 'finished', target: targetOf(payload) }
  if (phase === 'blocked') return { title: 'Goal blocked — needs you', body: goal, kind: 'attention', urgent: true, target: targetOf(payload) }
  return null
}

/** Deduplication key for suppressing duplicate alerts across multiple retained views on one daemon. */
export function notificationDedupKey(rawEvent: NotifyEvent, decision: NotifyDecision): string {
  const event = unwrapAttentionEvent(rawEvent)
  const payload = event.payload ?? {}
  const sessionId = decision.target?.sessionId ?? text(payload.session_id)
  const requestId = text(payload.request_id) || text(payload.id) || text(payload.tool_call_id)
  const artifactId = decision.target?.artifactId ?? text(payload.artifact_id)
  if (event.type === 'approval_request' || event.type === 'question_request') {
    return `${event.type}:${sessionId}:${requestId || decision.body}`
  }
  if (event.type === 'tool_result' && artifactId) {
    return `artifact:${sessionId}:${artifactId}:${requestId || decision.title}`
  }
  return `${event.type}:${sessionId}:${decision.title}:${decision.body}`
}

/** The gate around the display: preference first, then event rules. */
export function shouldNotify(
  rawEvent: NotifyEvent,
  state: { readonly enabled: boolean; readonly anyWindowFocused: boolean },
  previousGoalPhase?: string,
): NotifyDecision | null {
  if (!state.enabled) return null
  const event = unwrapAttentionEvent(rawEvent)
  return event.type === 'status_update'
    ? goalPhaseNotification(previousGoalPhase, event.payload)
    : notificationFor(event)
}
