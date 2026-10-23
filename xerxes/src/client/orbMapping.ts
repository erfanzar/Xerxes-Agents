// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { WorkspaceHealth } from '../hub/protocol.js'
import type { ConnectionStatus } from './bridge.js'
import { toolOrbState, toolPhrase, type TaskActivity, type TaskStatus, type ToolOrbState } from './taskTypes.js'

/** The nine thinking-orbs states (kept local so this module stays framework-free). */
export type OrbStateName = ToolOrbState | 'listening' | 'weaving' | 'composing' | 'breathing'

export type StatusTone = 'busy' | 'attention' | 'calm' | 'problem'

export type StatusView =
  | {
      readonly kind: 'orb'
      readonly state: OrbStateName
      /** False draws one still frame. */
      readonly live: boolean
      readonly label: string
      readonly tone: StatusTone
      readonly outdated?: boolean
    }
  | {
      readonly kind: 'mark'
      readonly mark: 'failed' | 'offline' | 'unknown'
      readonly label: string
      readonly detail?: string
      readonly tone: StatusTone
      readonly outdated?: boolean
    }

export interface StatusEvidence {
  readonly connection: ConnectionStatus['state']
  readonly workspaceHealth?: WorkspaceHealth
  readonly status: TaskStatus
  readonly activity: TaskActivity
  readonly pendingCount: number
  readonly stale: boolean
  readonly interrupted?: boolean
}

const orb = (state: OrbStateName, live: boolean, label: string, tone: StatusTone): StatusView => ({ kind: 'orb', state, live, label, tone })
const mark = (kind: 'failed' | 'offline' | 'unknown', label: string, tone: StatusTone, detail?: string): StatusView =>
  ({ kind: 'mark', mark: kind, label, tone, ...(detail === undefined ? {} : { detail }) })

/**
 * Orb mapping. Connection health and task activity are kept apart: an
 * unreachable link marks the evidence outdated rather than inventing activity.
 * With no evidence a working task reads "Working" (or "Waiting for update"
 * when stale), never an invented "Thinking" and never a percentage.
 */
export function statusView(e: StatusEvidence): StatusView {
  if (e.workspaceHealth === 'offline') return mark('offline', 'Unavailable', 'problem', 'This workspace is offline')
  if (e.workspaceHealth === 'needs_operator') return mark('offline', 'Unavailable', 'problem', 'This workspace needs attention on the hub')
  if (e.connection === 'connecting' || e.connection === 'replaying') return orb('connecting', true, 'Connecting', 'busy')
  if (e.workspaceHealth === 'connecting') return orb('connecting', true, 'Connecting to workspace', 'busy')
  const outdated = e.connection !== 'live' || e.stale || e.workspaceHealth === 'stale'
  const view = fromEvidence(e, outdated)
  return outdated ? { ...view, outdated: true } : view
}

function fromEvidence(e: StatusEvidence, outdated: boolean): StatusView {
  if (e.pendingCount > 0 || e.status === 'needs_you') return orb('breathing', true, 'Needs you', 'attention')
  if (e.status === 'failed') return mark('failed', 'Failed', 'problem', 'Open the task to see what went wrong')
  if (e.interrupted === true) return mark('unknown', 'Interrupted', 'problem', 'The workspace restarted. Check the task before continuing')
  if (e.status === 'working') {
    if (outdated && e.activity.kind === 'none') return orb('working', true, 'Waiting for update', 'busy')
    switch (e.activity.kind) {
      case 'thinking': return orb('solving', true, 'Thinking', 'busy')
      case 'tool': return orb(toolOrbState(e.activity.name), true, toolPhrase(e.activity.name), 'busy')
      case 'agents': return orb('connecting', true, e.activity.waiting > 0 ? `Waiting on ${e.activity.waiting} agent${e.activity.waiting === 1 ? '' : 's'}` : 'Waiting on agents', 'busy')
      case 'none': return orb('working', true, 'Working', 'busy')
    }
  }
  if (e.status === 'idle') return orb('breathing', false, 'Idle', 'calm')
  return mark('unknown', 'Status unknown', 'calm')
}
