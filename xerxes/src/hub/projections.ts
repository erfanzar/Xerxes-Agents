// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { costLabel } from '../auth/usageView.js'
import { isRecord } from './protocol.js'
import type { UpstreamEvent } from './ports.js'

export type SessionProjectionStatus = 'idle' | 'working' | 'needs_you' | 'failed' | 'unknown'
export interface SessionProjection {
  readonly id: string
  readonly title: string
  readonly status: SessionProjectionStatus
  readonly model: string
  readonly goal_summary: string
  readonly pending_interaction_count: number
  readonly agent_counts: { readonly total: number; readonly active: number }
  readonly updated_at: string
  readonly source_timestamp: string
  readonly stale: boolean
  readonly interrupted?: boolean
}
export interface WorkspaceProjection {
  readonly workspace_id: string
  readonly sessions: Readonly<Record<string, SessionProjection>>
  /** Bounded replay memory; durable dedupe belongs to the event sink. */
  readonly recent_event_ids: readonly string[]
  readonly interaction_ids: Readonly<Record<string, readonly string[]>>
}
export function emptyProjection(workspaceId: string): WorkspaceProjection {
  return { workspace_id: workspaceId, sessions: {}, recent_event_ids: [], interaction_ids: {} }
}
const text = (value: unknown, fallback = ''): string => typeof value === 'string' ? value : fallback
const num = (value: unknown): number => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : 0
function sourceTime(row: Record<string, unknown>, fallback: string): string {
  if (typeof row.updated_at === 'string' && Number.isFinite(Date.parse(row.updated_at))) return row.updated_at
  if (typeof row.last_active === 'number' && Number.isFinite(row.last_active)) return new Date(row.last_active * 1000).toISOString()
  return fallback
}
function statusOf(value: unknown): SessionProjectionStatus {
  if (value === 'idle' || value === 'failed' || value === 'needs_you' || value === 'unknown') return value
  if (value === 'working' || value === 'running' || value === 'busy' || value === 'streaming') return 'working'
  if (value === 'waiting' || value === 'waiting_permission' || value === 'waiting_question') return 'needs_you'
  if (value === 'ready' || value === 'done' || value === 'complete' || value === 'completed' || value === 'stopped' || value === 'cancelled') return 'idle'
  if (value === 'error') return 'failed'
  return 'unknown'
}
/** A summary row with no status is idle unless a turn is running: an empty status never means "unknown". */
function rowStatus(row: Record<string, unknown>): SessionProjectionStatus {
  if (row.status === undefined || row.status === null || (typeof row.status === 'string' && row.status.trim() === '')) {
    return typeof row.active_turn_id === 'string' && row.active_turn_id ? 'working' : 'idle'
  }
  return statusOf(row.status)
}

/** Full summary replacement; never retains transcript/tool/provider fields. */
export function replaceSessionSummaries(state: WorkspaceProjection, summaries: readonly unknown[], observedAt: string): WorkspaceProjection {
  const sessions: Record<string, SessionProjection> = {}
  const interactions: Record<string, readonly string[]> = {}
  for (const row of summaries) {
    if (!isRecord(row) || typeof row.id !== 'string' || !row.id) continue
    const previous = state.sessions[row.id]
    const source = sourceTime(row, observedAt)
    if (previous && Date.parse(source) < Date.parse(previous.source_timestamp)) {
      sessions[row.id] = previous
      interactions[row.id] = state.interaction_ids[row.id] ?? []
      continue
    }
    const pending = Array.isArray(row.pending_interactions) ? row.pending_interactions : []
    const retainInteractions = row.pending_interactions === undefined && row.pending_interaction_count === undefined
      && (rowStatus(row) === 'working' || rowStatus(row) === 'needs_you')
    const count = row.pending_interaction_count === undefined ? (retainInteractions ? previous?.pending_interaction_count ?? 0 : pending.length) : num(row.pending_interaction_count)
    const agents = Array.isArray(row.subagent_snapshots) ? row.subagent_snapshots : Array.isArray(row.subagents) ? row.subagents : []
    const goal = isRecord(row.goal) ? text(row.goal.objective ?? row.goal.summary) : text(row.goal)
    // An empty title is common for a new chat: fall back to the first user message (the daemon's
    // `preview`), then the last one, then a label.
    const title = text(row.title).trim() || text(row.first_user_message).trim() || text(row.preview).trim() || text(row.last_user_message).trim() || 'Untitled task'
    sessions[row.id] = { id: row.id, title, status: count ? 'needs_you' : rowStatus(row),
      model: text(row.model), goal_summary: goal, pending_interaction_count: count,
      agent_counts: { total: agents.length, active: agents.filter(agent => isRecord(agent) && (agent.status === 'running' || agent.status === 'queued')).length },
      updated_at: source, source_timestamp: source, stale: false, ...(previous?.interrupted ? { interrupted: true } : {}) }
    interactions[row.id] = retainInteractions ? state.interaction_ids[row.id] ?? [] : pending.flatMap(item => {
      if (!isRecord(item)) return []
      const id = item.request_id ?? item.id
      return typeof id === 'string' ? [id] : []
    })
  }
  return { ...state, sessions, interaction_ids: interactions }
}

export function markProjectionStale(state: WorkspaceProjection, stale = true): WorkspaceProjection {
  return { ...state, sessions: Object.fromEntries(Object.entries(state.sessions).map(([id, row]) => [id, { ...row, stale }])) }
}
export function markProjectionInterrupted(state: WorkspaceProjection): WorkspaceProjection {
  return { ...markProjectionStale(state), sessions: Object.fromEntries(Object.entries(state.sessions).map(([id, row]) => [id,
    { ...row, stale: true, ...(row.status === 'working' || row.status === 'needs_you' ? { status: 'unknown' as const, interrupted: true } : {}) }])) }
}

/** Immutable summary-event reducer. Conversation text and provider frames are not projected. */
export function reduceProjectionEvent(state: WorkspaceProjection, event: UpstreamEvent, projectDir?: string): WorkspaceProjection {
  if ((projectDir !== undefined && event.project_dir !== projectDir) || state.recent_event_ids.includes(event.event_id)) return state
  const next = { ...state, recent_event_ids: [...state.recent_event_ids, event.event_id].slice(-2048) }
  const id = event.session_id ?? text(event.payload.session_id)
  if (!id) return next
  if (event.type === 'session_removed' || event.type === 'session_deleted') {
    const sessions = { ...state.sessions }; delete sessions[id]
    const interactions = { ...state.interaction_ids }; delete interactions[id]
    return { ...next, sessions, interaction_ids: interactions }
  }
  const row = state.sessions[id] ?? { id, title: 'Untitled task', status: 'unknown' as const, model: '', goal_summary: '', pending_interaction_count: 0, agent_counts: { total: 0, active: 0 }, updated_at: event.occurred_at, source_timestamp: event.occurred_at, stale: false }
  if (Date.parse(event.occurred_at) < Date.parse(row.source_timestamp)) return next
  const payload = event.payload
  let updated: SessionProjection = { ...row, updated_at: event.occurred_at, source_timestamp: event.occurred_at, stale: false }
  let interactions = state.interaction_ids[id] ?? []
  if (event.type === 'turn_begin') updated = { ...updated, status: 'working', interrupted: false }
  if (event.type === 'turn_end') updated = { ...updated, status: payload.error || payload.failed === true ? 'failed' : 'idle', pending_interaction_count: 0 }
  if (event.type === 'error' || event.type === 'turn_error') updated = { ...updated, status: 'failed' }
  // The daemon raises `approval_request` (turnRunner); `permission_request` is the older name.
  if (event.type === 'approval_request' || event.type === 'permission_request' || event.type === 'question_request') {
    const requestId = text(payload.request_id ?? payload.id, event.event_id)
    if (!interactions.includes(requestId)) interactions = [...interactions, requestId]
    updated = { ...updated, status: 'needs_you', pending_interaction_count: Math.max(row.pending_interaction_count, interactions.length) }
  }
  if (['approval_response', 'permission_resolved', 'question_resolved', 'interaction_resolved', 'permission_response', 'question_response'].includes(event.type)) {
    const id = text(payload.request_id ?? payload.id)
    if (interactions.includes(id) || (interactions.length === 0 && row.pending_interaction_count > 0)) {
      interactions = interactions.filter(item => item !== id)
      updated = { ...updated, pending_interaction_count: Math.max(0, row.pending_interaction_count - 1), status: row.pending_interaction_count > 1 ? 'needs_you' : 'working' }
    }
  }
  if (event.type === 'turn_end') interactions = []
  if (event.type === 'status_update' || event.type === 'session_updated' || event.type === 'session_summary') {
    const summary = isRecord(payload.session) ? payload.session : payload
    const agents = Array.isArray(summary.subagent_snapshots) ? summary.subagent_snapshots : undefined
    const goal = isRecord(summary.goal) ? text(summary.goal.objective ?? summary.goal.summary) : typeof summary.goal === 'string' ? summary.goal : undefined
    updated = { ...updated, ...(summary.status === undefined ? {} : { status: updated.pending_interaction_count > 0 ? 'needs_you' : statusOf(summary.status) }),
      ...(typeof summary.title === 'string' ? { title: summary.title } : {}), ...(typeof summary.model === 'string' ? { model: summary.model } : {}),
      ...(goal === undefined ? {} : { goal_summary: goal }), ...(agents === undefined ? {} : { agent_counts: { total: agents.length,
        active: agents.filter(agent => isRecord(agent) && (agent.status === 'running' || agent.status === 'queued')).length } }) }
  }
  return { ...next, sessions: { ...state.sessions, [id]: updated }, interaction_ids: { ...state.interaction_ids, [id]: interactions } }
}

export interface UsageAggregate {
  readonly input: number
  readonly output: number
  readonly billable_cost_usd?: number
  readonly subscription_price_estimate_usd?: number
  readonly subscription_label?: string
}
/** Session counters are own-use unless explicitly marked includes_children. Never add rolled-up and child use. */
export function aggregateUsage(summaries: readonly Readonly<Record<string, unknown>>[]): UsageAggregate {
  const rows = new Map<string, Readonly<Record<string, unknown>>>()
  for (const row of summaries) if (typeof row.id === 'string') rows.set(row.id, row)
  let input = 0, output = 0, billable: number | undefined, subscription: number | undefined
  let hasSubscription = false
  for (const [id, row] of rows) {
    let parent = text(row.parent_session_id), covered = false
    const seen = new Set([id])
    while (parent && !seen.has(parent)) {
      seen.add(parent)
      const ancestor = rows.get(parent)
      if (!ancestor) break
      if (ancestor.includes_children === true && !isRecord(ancestor.own_usage)) { covered = true; break }
      parent = text(ancestor.parent_session_id)
    }
    if (covered) continue
    const own = isRecord(row.own_usage) ? row.own_usage : row
    input += num(own.input_tokens ?? own.input)
    output += num(own.output_tokens ?? own.output)
    const cost = typeof own.cost_usd === 'number' && Number.isFinite(own.cost_usd) && own.cost_usd >= 0 ? own.cost_usd : undefined
    if ((own.cost_basis ?? row.cost_basis) === 'subscription') {
      hasSubscription = true
      if (cost !== undefined) subscription = (subscription ?? 0) + cost
    } else if (cost !== undefined) billable = (billable ?? 0) + cost
  }
  return { input, output, ...(billable === undefined ? {} : { billable_cost_usd: billable }),
    ...(subscription === undefined ? {} : { subscription_price_estimate_usd: subscription }),
    ...(hasSubscription ? { subscription_label: costLabel(subscription, 'subscription') } : {}) }
}
