// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Which daemon events deserve a native ping, and what it should say.
 * Pure logic — no Electron imports — so the decision table is testable
 * headlessly while the display side stays in main.ts.
 */

export interface NotifyEvent {
  readonly type: string
  readonly payload: Record<string, unknown>
}

export interface NotifyDecision {
  readonly title: string
  readonly body: string
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
      return { title: 'Task finished', body: 'The current task completed.' }
    }
    case 'approval_request': {
      const description = text(payload.description) ||
        `${text(payload.name)} ${text(payload.tool_name)}`.trim()
      if (!description) return null
      return { title: 'Approval needed', body: truncate(description) }
    }
    case 'question_request': {
      const questions = Array.isArray(payload.questions) ? payload.questions : []
      const first = questions.find(
        (item): item is Record<string, unknown> =>
          item !== null && typeof item === 'object' && text((item as Record<string, unknown>).question) !== '',
      )
      if (!first) return null
      return { title: 'Xerxes has a question', body: truncate(text(first.question)) }
    }
    default:
      return null
  }
}

/** The gate around the display: preference first, then focus. */
export function shouldNotify(
  event: NotifyEvent,
  state: { readonly enabled: boolean; readonly anyWindowFocused: boolean },
): NotifyDecision | null {
  if (!state.enabled) return null
  if (state.anyWindowFocused) return null
  return notificationFor(event)
}
