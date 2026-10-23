// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ConfigField, SelectedSession } from './hubStore.js'
import type { TaskStatus } from './taskTypes.js'

/** Pure view helpers for the Task tab. */

export type ComposerMode = 'send' | 'steer'

/** A task that is working or waiting on a person takes steers; an idle one takes a new message. */
export function composerMode(status: TaskStatus | undefined): ComposerMode {
  return status === 'working' || status === 'needs_you' ? 'steer' : 'send'
}

export function isRunning(status: TaskStatus | undefined): boolean {
  return status === 'working' || status === 'needs_you'
}

/**
 * What to tell the person about a setting they changed. A request is never
 * shown as applied until the session itself reports the value.
 */
export function configNote(field: ConfigField): string | undefined {
  switch (field.phase) {
    case 'idle': return undefined
    case 'requested': return `Requested ${field.requested ?? 'a change'}. Waiting for the Hub.`
    case 'accepted': return `The Hub accepted ${field.requested ?? 'the change'}. It shows as applied once the task reports it.`
    case 'rejected': return field.message ?? 'The Hub did not accept that change.'
    case 'unknown': return field.message ?? 'Could not confirm whether this was applied.'
  }
}

export interface ResolutionNotice {
  readonly id: string
  readonly text: string
}

const VERBS: Readonly<Record<string, string>> = {
  approve: 'Approved',
  approve_for_session: 'Approved for this session',
  reject: 'Denied',
}

/** Lines such as "Already resolved by Alice" for interactions that are no longer pending. */
export function resolutionNotices(selected: SelectedSession, limit = 3): readonly ResolutionNotice[] {
  const pending = new Set(selected.tail.pending.map(item => item.id))
  const ids = [...new Set([...Object.keys(selected.tail.resolved).slice(-limit), ...Object.keys(selected.decisions)])].filter(id => !pending.has(id))
  const notices: ResolutionNotice[] = []
  for (const id of ids) {
    const mine = selected.decisions[id]
    if (mine !== undefined && mine.phase !== 'resolved') continue
    const resolution = selected.tail.resolved[id]
    const by = mine?.by ?? resolution?.by
    if (mine?.already === true) {
      notices.push({ id, text: by ? `Already resolved by ${by}.` : mine.message ?? 'Already resolved by someone else.' })
      continue
    }
    const verb = resolution?.decision ? VERBS[resolution.decision] : undefined
    if (verb) notices.push({ id, text: `${verb}${by ? ` by ${by}` : ''}.` })
    else if (mine !== undefined) notices.push({ id, text: 'Decision sent.' })
    else if (by) notices.push({ id, text: `Resolved by ${by}.` })
  }
  return notices.slice(-limit)
}
