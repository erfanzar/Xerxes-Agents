// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isRecord, type HubEvent, type WorkspaceHealth } from '../hub/protocol.js'
import { asActor } from './hubConnection.js'
import type {
  AutopilotPerson, AutopilotRow, AutopilotView, EffortOption, MachineView, ModelChoice, ModelOption, PresenceChip, ProfileOption, SupervisorChoice, SupervisorConfigStatus, SupervisorConfigView,
  SupervisorOptions, TaskSummary, WorkspaceView,
} from './hubTypes.js'
import { arr, clip, isoOr, num, rec, str } from './parse.js'
import { toolSummary, userInputOf, type Block, type FoldInput } from './sessionFold.js'
import { NO_ACTIVITY, activityAfter, taskStatusOf, type TaskActivity } from './taskTypes.js'

/**
 * Tolerant readers for Hub payloads. Every reader drops what it cannot
 * understand instead of guessing; none of them throws. Browser-safe.
 */

const HEALTH: ReadonlySet<string> = new Set<WorkspaceHealth>(['online', 'connecting', 'stale', 'offline', 'needs_operator'])

export function parseWorkspace(raw: unknown): WorkspaceView | undefined {
  const r = rec(raw)
  const id = str(r.workspace_id) ?? str(r.id)
  if (!id) return undefined
  const health = typeof r.health === 'string' && HEALTH.has(r.health) ? (r.health as WorkspaceHealth) : 'stale'
  const caps = rec(r.capabilities)
  const models: ModelChoice[] = []
  for (const item of [...arr(r.models), ...arr(caps.models)]) {
    const m = rec(item)
    const model = str(m.model)
    const profile = str(m.profile) ?? str(m.provider_profile)
    if (model && profile) models.push({ profile, model, label: str(m.label) ?? model })
  }
  const efforts = [...arr(r.efforts), ...arr(caps.efforts)].filter((e): e is string => typeof e === 'string')
  return { id, label: str(r.label) ?? str(r.name) ?? id, health, ...(r.read_only === true ? { readOnly: true } : {}), models, efforts: [...new Set(efforts)] }
}

export function parseWorkspaces(result: Readonly<Record<string, unknown>>): readonly WorkspaceView[] {
  return arr(result.workspaces).flatMap(item => {
    const w = parseWorkspace(item)
    return w ? [w] : []
  })
}

function agentCounts(row: Readonly<Record<string, unknown>>): { readonly total: number; readonly active: number } {
  const counts = rec(row.agent_counts)
  if (num(counts.total) !== undefined) return { total: num(counts.total) ?? 0, active: num(counts.active) ?? 0 }
  const list = arr(row.subagent_snapshots ?? row.subagents)
  return { total: list.length, active: list.filter(a => ['running', 'queued', 'working'].includes(String(rec(a).status))).length }
}

function goalOf(value: unknown): string {
  const g = rec(value)
  return str(g.objective) ?? str(g.summary) ?? (typeof value === 'string' ? value : '')
}

/** One session summary row from a snapshot. */
export function parseSummary(workspaceId: string, raw: unknown, now: string, previous?: TaskSummary): TaskSummary | undefined {
  const row = rec(raw)
  const id = str(row.id) ?? str(row.session_id)
  if (!id) return undefined
  const pending = arr(row.pending_interactions)
  const interactionIds = pending.flatMap(item => {
    const p = rec(item)
    const pid = str(p.request_id) ?? str(p.id)
    return pid ? [pid] : []
  })
  const count = num(row.pending_interaction_count) ?? Math.max(pending.length, interactionIds.length)
  const status = count > 0 ? 'needs_you' : taskStatusOf(row.status)
  return {
    workspaceId,
    id,
    title: str(row.title) ?? str(row.preview) ?? previous?.title ?? 'Untitled task',
    status,
    model: str(row.model) ?? '',
    goalSummary: goalOf(row.goal ?? row.goal_summary),
    pendingCount: count,
    interactionIds,
    agents: agentCounts(row),
    updatedAt: isoOr(row.updated_at, now),
    stale: row.stale === true,
    interrupted: row.interrupted === true,
    activity: previous && previous.status === 'working' && status === 'working' ? previous.activity : NO_ACTIVITY,
  }
}

export function parseSummaries(workspaceId: string, rows: readonly unknown[], now: string, previous: Readonly<Record<string, TaskSummary>> = {}): readonly TaskSummary[] {
  return rows.flatMap(row => {
    const id = str(rec(row).id) ?? str(rec(row).session_id)
    const s = parseSummary(workspaceId, row, now, id === undefined ? undefined : previous[`${workspaceId}\u001f${id}`])
    return s ? [s] : []
  })
}

const RESOLVING: ReadonlySet<string> = new Set([
  'permission_resolved', 'question_resolved', 'interaction_resolved', 'permission_response', 'approval_response', 'question_response',
])

/** Fold one daemon-vocabulary event into a task summary. Idempotent per interaction id. */
export function foldSummaryEvent(workspaceId: string, previous: TaskSummary | undefined, event: HubEvent): TaskSummary | undefined {
  const sessionId = event.session_id ?? str(event.payload.session_id)
  if (!sessionId) return previous
  const p = event.payload
  const base: TaskSummary = previous ?? {
    workspaceId, id: sessionId, title: 'Untitled task', status: 'unknown', model: '', goalSummary: '', pendingCount: 0,
    interactionIds: [], agents: { total: 0, active: 0 }, updatedAt: event.occurred_at, stale: false, interrupted: false, activity: NO_ACTIVITY,
  }
  if (previous && Date.parse(event.occurred_at) < Date.parse(previous.updatedAt)) return previous
  let next: TaskSummary = { ...base, updatedAt: event.occurred_at, stale: false, activity: activityAfter(event.type, p, base.activity, base.agents.active) }
  switch (event.type) {
    case 'turn_begin': return { ...next, status: 'working', interrupted: false }
    case 'turn_end': return { ...next, status: p.error !== undefined && p.error !== null && p.error !== false ? 'failed' : 'idle', pendingCount: 0, interactionIds: [], activity: NO_ACTIVITY }
    case 'turn_error': case 'error': return { ...next, status: 'failed', activity: NO_ACTIVITY }
    case 'session_title': return str(p.title) ? { ...next, title: str(p.title)! } : next
    case 'session_removed': case 'session_deleted': return undefined
    case 'approval_request': case 'permission_request': case 'question_request': {
      const rid = str(p.request_id) ?? str(p.id) ?? event.event_id
      if (next.interactionIds.includes(rid)) return next
      const ids = [...next.interactionIds, rid]
      return { ...next, interactionIds: ids, pendingCount: Math.max(next.pendingCount, ids.length), status: 'needs_you' }
    }
    case 'status_update': case 'session_updated': case 'session_summary': {
      const s = event.type === 'status_update' ? p : rec(p.session ?? p)
      const model = str(s.model)
      const goal = s.goal === undefined ? undefined : goalOf(s.goal)
      const title = str(s.title)
      next = {
        ...next,
        ...(model ? { model } : {}),
        ...(goal !== undefined ? { goalSummary: goal } : {}),
        ...(title ? { title } : {}),
        ...(s.subagent_snapshots !== undefined || s.agent_counts !== undefined ? { agents: agentCounts(s) } : {}),
        ...(s.status !== undefined && event.type !== 'status_update' ? { status: next.pendingCount > 0 ? 'needs_you' : taskStatusOf(s.status) } : {}),
      }
      return next
    }
    default:
      break
  }
  if (RESOLVING.has(event.type)) {
    const rid = str(p.request_id) ?? str(p.id)
    if (!rid || !next.interactionIds.includes(rid)) return next
    const ids = next.interactionIds.filter(i => i !== rid)
    return { ...next, interactionIds: ids, pendingCount: ids.length, status: ids.length > 0 ? 'needs_you' : 'working' }
  }
  return next
}

// ── presence ──────────────────────────────────────────────────────────

export interface PresenceEntry {
  readonly userId: string
  readonly name: string
  readonly deviceId: string
  readonly state: 'watching' | 'typing'
}

export function parsePresence(payload: Readonly<Record<string, unknown>>): { readonly workspaceId: string; readonly sessionId?: string; readonly entries: readonly PresenceEntry[] } | undefined {
  const workspaceId = str(payload.workspace_id)
  if (!workspaceId) return undefined
  const sessionId = str(payload.session_id)
  const entries = arr(payload.entries ?? payload.participants).flatMap((raw): PresenceEntry[] => {
    const e = rec(raw)
    const userId = str(e.user_id) ?? str(e.principal_id)
    if (!userId) return []
    return [{ userId, name: str(e.display_name) ?? str(e.name) ?? 'Someone', deviceId: str(e.device_id) ?? '', state: e.state === 'typing' ? 'typing' : 'watching' }]
  })
  return { workspaceId, ...(sessionId ? { sessionId } : {}), entries }
}

/** Several devices of one person collapse into one chip; the person's own entry is dropped. */
export function presenceChips(entries: readonly PresenceEntry[], selfUserId?: string): readonly PresenceChip[] {
  const byUser = new Map<string, { name: string; state: 'watching' | 'typing'; devices: Set<string> }>()
  for (const e of entries) {
    if (e.userId === selfUserId) continue
    const found = byUser.get(e.userId) ?? { name: e.name, state: 'watching' as const, devices: new Set<string>() }
    found.devices.add(e.deviceId)
    if (e.state === 'typing') found.state = 'typing'
    byUser.set(e.userId, found)
  }
  return [...byUser.entries()].map(([userId, v]) => ({ userId, name: v.name, state: v.state, devices: v.devices.size }))
}

// ── session snapshots and history ─────────────────────────────────────

export function asFoldInput(raw: unknown, index: number): FoldInput | undefined {
  const e = rec(raw)
  const type = str(e.type)
  if (!type) return undefined
  const actor = asActor(e.actor)
  return { id: str(e.event_id) ?? str(e.id) ?? `h${index}`, type, payload: isRecord(e.payload) ? e.payload : {}, ...(actor ? { actor } : {}) }
}

export function parseFoldInputs(value: unknown): readonly FoldInput[] {
  return arr(value).flatMap((raw, index) => {
    const input = asFoldInput(raw, index)
    return input ? [input] : []
  })
}

/** Session rows of one workspace out of a `hub.snapshot` result (`workspaces[].sessions[]`). */
export function snapshotSessions(result: Readonly<Record<string, unknown>>, workspaceId: string): readonly unknown[] {
  for (const raw of arr(result.workspaces)) {
    const w = rec(raw)
    if (str(w.id) === workspaceId) return arr(w.sessions)
  }
  return []
}

function partsText(content: unknown): { readonly text: string; readonly images: number } {
  return userInputOf(content)
}

/**
 * One page of `hub.history`: the daemon's chronological "actions"
 * (`{id, messages, executions, thinking}`) turned into display blocks. A
 * malformed action is skipped, never guessed at.
 */
export function parseHistoryPage(result: Readonly<Record<string, unknown>>): { readonly blocks: readonly Block[]; readonly cursor?: string | number; readonly hasMore: boolean } {
  const history = rec(result.history)
  const blocks: Block[] = []
  for (const [index, raw] of arr(history.actions).entries()) {
    const action = rec(raw)
    const base = str(action.id) ?? `a${index}`
    let n = 0
    const nextId = (): string => `${base}:${n++}`
    for (const item of arr(action.thinking)) {
      const text = typeof item === 'string' ? item : str(rec(item).text) ?? str(rec(item).think)
      if (text && text.trim()) blocks.push({ kind: 'thinking', id: nextId(), text: clip(text, 4000), streaming: false })
    }
    for (const rawMessage of arr(action.messages)) {
      const message = rec(rawMessage)
      const { text, images } = partsText(message.content)
      if (message.role === 'user') blocks.push({ kind: 'user', id: nextId(), text, images, steer: false })
      else if (message.role === 'assistant' && text) blocks.push({ kind: 'assistant', id: nextId(), text, streaming: false })
    }
    for (const rawExec of arr(action.executions)) {
      const exec = rec(rawExec)
      const name = str(exec.name) ?? str(exec.toolName) ?? str(exec.tool_name) ?? 'tool'
      const callId = str(exec.toolCallId) ?? str(exec.tool_call_id) ?? nextId()
      const failed = exec.is_error === true || exec.error !== undefined || exec.status === 'error'
      const out = exec.result ?? exec.output ?? exec.error
      const output = typeof out === 'string' ? clip(out, 600) : out === undefined ? undefined : clip(JSON.stringify(out) ?? '', 600)
      blocks.push({ kind: 'tool', id: nextId(), callId, name, summary: toolSummary(exec.arguments ?? exec.args ?? exec.input), state: failed ? 'error' : 'done', ...(output ? { output } : {}) })
    }
  }
  const before = history.before
  const cursor = typeof before === 'string' && before.length > 0 ? before : typeof before === 'number' && Number.isFinite(before) ? before : undefined
  return { blocks, ...(cursor === undefined ? {} : { cursor }), hasMore: history.has_more === true && cursor !== undefined }
}

// ── supervisor ────────────────────────────────────────────────────────

export interface SupervisorMessage {
  readonly id: string
  readonly role: 'user' | 'xerxes' | 'system'
  readonly text: string
  readonly at: string
  readonly author?: string
  readonly source: 'text' | 'voice'
}

export type ConfirmationState = 'pending' | 'confirming' | 'confirmed' | 'cancelled' | 'expired' | 'failed'

export interface SupervisorConfirmation {
  readonly id: string
  readonly tool: string
  readonly summary: string
  readonly targets: readonly string[]
  readonly requestedBy?: string
  readonly expiresAt?: string
  readonly state: ConfirmationState
  readonly message?: string
  /** Created by Autopilot: it runs without a Confirm tap, so the card has no buttons. */
  readonly auto?: boolean
  /** The person Autopilot acted for (`confirmed_by` on the wire). */
  readonly confirmedBy?: string
}

export function parseSupervisorMessage(raw: unknown): SupervisorMessage | undefined {
  const m = rec(raw)
  const id = str(m.id) ?? str(m.message_id)
  const text = typeof m.text === 'string' ? m.text : undefined
  if (!id || text === undefined) return undefined
  const role = m.role === 'user' || m.role === 'system' ? m.role : 'xerxes'
  const actor = asActor(m.actor)
  return { id, role, text: clip(text, 8000), at: isoOr(m.at ?? m.occurred_at, new Date(0).toISOString()), source: m.source === 'voice' ? 'voice' : 'text', ...(actor ? { author: actor.display_name } : str(m.author) ? { author: str(m.author)! } : {}) }
}

const CONFIRMATION_STATES: ReadonlySet<string> = new Set(['pending', 'confirming', 'confirmed', 'cancelled', 'expired', 'failed'])

export function parseConfirmation(raw: unknown): SupervisorConfirmation | undefined {
  const c = rec(raw)
  const id = str(c.id) ?? str(c.confirmation_id)
  if (!id) return undefined
  const state = typeof c.state === 'string' && CONFIRMATION_STATES.has(c.state) ? (c.state as ConfirmationState) : 'pending'
  const requestedBy = str(c.requested_by) ?? str(rec(c.actor).requested_by)
  const expiresAt = str(c.expires_at)
  const message = str(c.message)
  const confirmedBy = str(c.confirmed_by) ?? str(rec(c.actor).confirmed_by)
  return {
    id,
    ...(c.auto === true ? { auto: true } : {}),
    ...(confirmedBy ? { confirmedBy } : {}),
    tool: str(c.tool) ?? str(c.action) ?? 'action',
    summary: str(c.summary) ?? str(c.description) ?? 'XERXES proposes an action.',
    targets: arr(c.targets).flatMap(t => (typeof t === 'string' ? [t] : str(rec(t).label) ? [str(rec(t).label)!] : [])),
    state,
    ...(requestedBy ? { requestedBy } : {}),
    ...(expiresAt ? { expiresAt } : {}),
    ...(message ? { message } : {}),
  }
}

// ── autopilot ─────────────────────────────────────────────────────────

/** An ISO timestamp string that parses, or null for an explicit JSON null. Anything else is malformed (undefined). */
function isoOrNull(value: unknown): string | null | undefined {
  if (value === null) return null
  return typeof value === 'string' && value.length > 0 && Number.isFinite(Date.parse(value)) ? value : undefined
}

function parseAutopilotPerson(value: unknown): AutopilotPerson | null | undefined {
  if (value === null) return null
  if (typeof value !== 'object' || Array.isArray(value)) return undefined
  const person = rec(value)
  const id = str(person.id)
  const name = str(person.name)
  return id && name ? { id, name } : undefined
}

/**
 * One row of `supervisor.autopilot.get`, `supervisor.autopilot.set` or the
 * `supervisor.autopilot_changed` event. Strict: a row with a missing or wrongly
 * typed field is rejected whole, because a guessed `active` could show
 * Autopilot as on when the Hub has it off.
 */
export function parseAutopilotRow(raw: unknown): AutopilotRow | undefined {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return undefined
  const r = rec(raw)
  const workspaceId = str(r.workspace_id)
  if (!workspaceId || typeof r.label !== 'string') return undefined
  if (typeof r.enabled !== 'boolean' || typeof r.active !== 'boolean' || typeof r.instruction !== 'string') return undefined
  const expiresAt = isoOrNull(r.expires_at)
  const updatedAt = isoOrNull(r.updated_at)
  const enabledBy = parseAutopilotPerson(r.enabled_by)
  if (expiresAt === undefined || updatedAt === undefined || enabledBy === undefined) return undefined
  return {
    workspaceId,
    label: r.label === '' ? workspaceId : r.label,
    enabled: r.enabled,
    instruction: r.instruction,
    expiresAt,
    enabledBy,
    updatedAt,
    active: r.active,
  }
}

/** `supervisor.autopilot.get`. Any malformed row or a missing `can_edit` rejects the whole answer; a repeated workspace is malformed. */
export function parseAutopilotView(raw: unknown): AutopilotView | undefined {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return undefined
  const r = rec(raw)
  if (typeof r.can_edit !== 'boolean' || !Array.isArray(r.workspaces)) return undefined
  const rows: AutopilotRow[] = []
  for (const item of r.workspaces) {
    const row = parseAutopilotRow(item)
    if (!row || rows.some(known => known.workspaceId === row.workspaceId)) return undefined
    rows.push(row)
  }
  return { canEdit: r.can_edit, workspaces: rows }
}

// ── machines and the supervisor's model ───────────────────────────────

const CONFIG_STATUSES: ReadonlySet<string> = new Set<SupervisorConfigStatus>(['configured', 'unconfigured', 'machine_offline', 'invalid'])

export function parseMachine(raw: unknown): MachineView | undefined {
  const m = rec(raw)
  const id = str(m.id) ?? str(m.machine_id)
  if (!id) return undefined
  const health = typeof m.health === 'string' && HEALTH.has(m.health) ? (m.health as WorkspaceHealth) : 'stale'
  return {
    id,
    label: str(m.label) ?? id,
    health,
    readOnly: m.read_only === true,
    modelsSupported: m.models_supported === true,
    workspaceIds: arr(m.workspace_ids).filter((w): w is string => typeof w === 'string' && w.length > 0),
  }
}

/** Rows that cannot be read are dropped; a repeated id keeps its first row. */
export function parseMachines(result: Readonly<Record<string, unknown>>): readonly MachineView[] {
  const seen = new Set<string>()
  const out: MachineView[] = []
  for (const item of arr(result.machines)) {
    const machine = parseMachine(item)
    if (!machine || seen.has(machine.id)) continue
    seen.add(machine.id)
    out.push(machine)
  }
  return out
}

function parseChoice(raw: unknown): SupervisorChoice | undefined {
  const c = rec(raw)
  const machineId = str(c.machine_id)
  const profile = str(c.profile)
  const model = str(c.model)
  if (!machineId || !profile || !model) return undefined
  return { machineId, machineLabel: str(c.machine_label) ?? machineId, profile, model, effort: str(c.effort) ?? null }
}

/**
 * The shape of `supervisor.config.get`, `supervisor.config.set` and the
 * `supervisor.config_changed` event. An unknown status is `invalid`; editing
 * is allowed only when the Hub said so explicitly.
 */
export function parseSupervisorConfig(raw: unknown): SupervisorConfigView {
  const r = rec(raw)
  const status = typeof r.status === 'string' && CONFIG_STATUSES.has(r.status) ? (r.status as SupervisorConfigStatus) : 'invalid'
  const config = parseChoice(r.config)
  const message = str(r.message)
  return {
    status,
    canEdit: r.can_edit === true,
    ...(config ? { config } : {}),
    ...(message ? { message } : {}),
  }
}

export function parseSupervisorOptions(result: Readonly<Record<string, unknown>>): SupervisorOptions | undefined {
  const machineId = str(result.machine_id)
  if (!machineId) return undefined
  const profiles: ProfileOption[] = []
  for (const item of arr(result.profiles)) {
    const p = rec(item)
    const name = str(p.name)
    if (!name || profiles.some(known => known.name === name)) continue
    profiles.push({ name, label: str(p.label) ?? name, provider: str(p.provider) ?? '', defaultModel: str(p.default_model) ?? '' })
  }
  const models: ModelOption[] = []
  for (const item of arr(result.models)) {
    const m = rec(item)
    const id = str(m.id)
    if (!id || models.some(known => known.id === id)) continue
    models.push({ id, label: str(m.label) ?? id })
  }
  const efforts: EffortOption[] = []
  for (const item of arr(result.efforts)) {
    const e = typeof item === 'string' ? { effort: item } : rec(item)
    const effort = str(e.effort)
    if (!effort || efforts.some(known => known.effort === effort)) continue
    const description = str((e as Readonly<Record<string, unknown>>).description)
    efforts.push({ effort, ...(description ? { description } : {}) })
  }
  const profile = str(result.profile)
  const model = str(result.model)
  return { machineId, profiles, models, efforts, defaultEffort: str(result.default_effort) ?? null, ...(profile ? { profile } : {}), ...(model ? { model } : {}) }
}

export type { TaskActivity }
