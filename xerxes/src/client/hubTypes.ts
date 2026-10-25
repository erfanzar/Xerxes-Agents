// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { WorkspaceHealth } from '../hub/protocol.js'
import type { TaskActivity, TaskStatus } from './taskTypes.js'

export interface ModelChoice {
  readonly profile: string
  readonly model: string
  readonly label: string
}

export interface WorkspaceView {
  readonly id: string
  readonly label: string
  readonly health: WorkspaceHealth
  /** The workspace's runtime cannot verify signed Hub commands: view only. */
  readonly readOnly?: boolean
  /** Runtime-published choices. Empty means the workspace did not publish any. */
  readonly models: readonly ModelChoice[]
  readonly efforts: readonly string[]
}

export interface TaskSummary {
  readonly workspaceId: string
  readonly id: string
  readonly title: string
  readonly status: TaskStatus
  readonly model: string
  readonly goalSummary: string
  readonly pendingCount: number
  readonly interactionIds: readonly string[]
  readonly agents: { readonly total: number; readonly active: number }
  readonly updatedAt: string
  readonly stale: boolean
  readonly interrupted: boolean
  readonly activity: TaskActivity
}

export interface PresenceChip {
  readonly userId: string
  readonly name: string
  readonly state: 'watching' | 'typing'
  /** Several devices of one person collapse into one chip. */
  readonly devices: number
}

/** One machine the Hub reaches. Its workspaces are discovered, not typed in. */
export interface MachineView {
  readonly id: string
  readonly label: string
  readonly health: WorkspaceHealth
  /** The machine's daemon cannot verify signed Hub commands: view only. */
  readonly readOnly: boolean
  /** The machine can run XERXES's model calls (`hub.model.complete`). */
  readonly modelsSupported: boolean
  readonly workspaceIds: readonly string[]
}

export type SupervisorConfigStatus = 'configured' | 'unconfigured' | 'machine_offline' | 'invalid'

/** The model XERXES the supervisor runs on. Never carries a credential or a base URL. */
export interface SupervisorChoice {
  readonly machineId: string
  readonly machineLabel: string
  readonly profile: string
  readonly model: string
  /** null: no explicit thinking effort, the provider default applies. */
  readonly effort: string | null
}

export interface SupervisorConfigView {
  readonly status: SupervisorConfigStatus
  readonly config?: SupervisorChoice
  /** Whether this person may change the choice (the Hub owner only). */
  readonly canEdit: boolean
  readonly message?: string
}

export interface ProfileOption {
  readonly name: string
  readonly label: string
  readonly provider: string
  readonly defaultModel: string
}

export interface ModelOption {
  readonly id: string
  readonly label: string
}

export interface EffortOption {
  readonly effort: string
  readonly description?: string
}

/** What the chosen machine offers: provider profiles, then the models and efforts of one profile. */
export interface SupervisorOptions {
  readonly machineId: string
  readonly profiles: readonly ProfileOption[]
  readonly profile?: string
  readonly models: readonly ModelOption[]
  readonly model?: string
  readonly efforts: readonly EffortOption[]
  readonly defaultEffort: string | null
}

/** Who turned Autopilot on. */
export interface AutopilotPerson {
  readonly id: string
  readonly name: string
}

/**
 * One workspace's Autopilot setting (`supervisor.autopilot.get` row). `active`
 * is the Hub's verdict: enabled, not expired, and the person who enabled it is
 * still the Hub owner. Apps show `active`, never infer it from `enabled`.
 */
export interface AutopilotRow {
  readonly workspaceId: string
  readonly label: string
  readonly enabled: boolean
  /** The owner's standing instruction; empty when none. */
  readonly instruction: string
  /** ISO time Autopilot stops by itself, or null for no expiry. */
  readonly expiresAt: string | null
  readonly enabledBy: AutopilotPerson | null
  readonly updatedAt: string | null
  readonly active: boolean
}

/** The shape of `supervisor.autopilot.get`. */
export interface AutopilotView {
  /** Whether this person may change Autopilot (the Hub owner only). */
  readonly canEdit: boolean
  readonly workspaces: readonly AutopilotRow[]
}

// ── watch ─────────────────────────────────────────────────────────────

/** The fixed set of things the Hub's watcher flags. A kind outside it makes the row unreadable. */
export type WatchAnomalyKind = 'loop' | 'stall' | 'test_fail_repeat' | 'error_burst' | 'context_high' | 'goal_blocked'

/** One open problem on a watched run. `detail` came from the run: untrusted text. */
export interface WatchAnomaly {
  readonly kind: WatchAnomalyKind
  /** ISO time the problem opened. */
  readonly since: string
  readonly detail: string
}

export interface WatchTests {
  readonly runs: number
  readonly passed: number
  readonly failed: number
  readonly last: 'pass' | 'fail' | null
  /** Names of the last failing tests (untrusted text, at most a handful). */
  readonly lastFailing: readonly string[]
}

/** What the Hub's watcher has counted since it began reading the run. All numbers are counts. */
export interface WatchStats {
  readonly since: string
  readonly lastActivityAt: string | null
  /** The run's status as the Hub's projection reports it (`working`, `idle`, …). */
  readonly status: string
  readonly turns: number
  readonly toolCalls: number
  readonly toolFailures: number
  readonly errors: number
  readonly edits: number
  readonly filesTouched: number
  readonly tests: WatchTests
  readonly repeatedCall: { readonly name: string; readonly count: number } | null
  readonly contextTokens: number | null
  readonly maxContext: number | null
  readonly totalTokens: number | null
  readonly agents: { readonly running: number; readonly total: number }
  readonly anomalies: readonly WatchAnomaly[]
  readonly reviews: { readonly wakesThisHour: number; readonly lastWakeAt: string | null }
  /** Hub-rendered one-liner. Untrusted text. */
  readonly summaryLine: string
}

/**
 * One watched task (`supervisor.watch.list` row, `supervisor.watch.set` answer
 * or `supervisor.watch_update` event). `active` is the Hub's verdict: enabled,
 * not expired, and the person who enabled it is still the Hub owner. `title`,
 * `workspaceLabel` and everything in `stats` came from a task: plain text only.
 */
export interface WatchRow {
  readonly workspaceId: string
  readonly taskId: string
  readonly title: string
  readonly workspaceLabel: string
  readonly enabled: boolean
  readonly active: boolean
  /** Minutes between XERXES reviews of the run, or null for none. */
  readonly reviewIntervalMinutes: number | null
  readonly expiresAt: string | null
  readonly enabledBy: AutopilotPerson | null
  readonly updatedAt: string | null
  readonly stats: WatchStats | null
}

/** The shape of `supervisor.watch.list`. */
export interface WatchView {
  /** Whether this person may change Watch (the Hub owner only). */
  readonly canEdit: boolean
  readonly watches: readonly WatchRow[]
}

export function taskKey(workspaceId: string, sessionId: string): string {
  return `${workspaceId}\u001f${sessionId}`
}
