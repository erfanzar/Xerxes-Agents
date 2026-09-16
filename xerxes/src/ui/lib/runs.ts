// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { GatewayRpc } from '../app/interfaces.js'

export interface RunSummary {
  id: string
  ownerSessionId: string
  title: string
  kind: string
  state: string
  revision: number
  unread: boolean
  startedAt: number
  workspace: string
  sourceId: string
  endedAt?: number
  terminalKind?: string
  exitCode?: number
}
export interface ReactionHealthView { usage?: { inputTokens: number; outputTokens: number; complete: boolean }; state: string; attempts: number; maxReactions: number; pendingEvents: number; lastError: string | null }
export interface RunDetail extends RunSummary { tokenUsage?: { inputTokens: number; outputTokens: number; complete: boolean }; cancelLabel?: string; reactionHealth?: ReactionHealthView; output: string; outputTruncated: boolean; error: string | null }
export interface UpcomingRun { id: string; title: string; nextRunAt: string; timezone: string; executionState: string }
export interface RunAttention { id: string; kind: 'approval' | 'question'; title: string }
interface Response { attention?: unknown; attention_total?: unknown; upcoming?: unknown; upcoming_total?: unknown; has_more?: boolean; ok?: boolean; error?: string; runs?: unknown; run?: unknown }
function row(value: unknown): RunSummary | null {
  if (!value || typeof value !== 'object') return null
  const v = value as Record<string, unknown>
  if (typeof v.id !== 'string' || !v.id || typeof v.title !== 'string' || typeof v.kind !== 'string' ||
    typeof v.state !== 'string' || !['running', 'succeeded', 'failed', 'cancelled', 'interrupted'].includes(v.state) ||
    typeof v.revision !== 'number' || !Number.isSafeInteger(v.revision) || v.revision < 1) return null
  return { id: v.id, ownerSessionId: typeof v.ownerSessionId === 'string' ? v.ownerSessionId : '', title: v.title, kind: v.kind, state: v.state, revision: v.revision, unread: v.unread === true,
    startedAt: typeof v.startedAt === 'number' ? v.startedAt : 0,
    ...(typeof v.endedAt === 'number' && Number.isFinite(v.endedAt) ? { endedAt: v.endedAt } : {}),
    ...(typeof v.terminalKind === 'string' ? { terminalKind: v.terminalKind } : {}),
    ...(typeof v.exitCode === 'number' ? { exitCode: v.exitCode } : {}),
    workspace: typeof v.workspace === 'string' ? v.workspace : '', sourceId: typeof v.sourceId === 'string' ? v.sourceId : '' }
}
export async function listRunPage(rpc: GatewayRpc, unreadOnly = false, scope: 'session' | 'workspace' = 'session', scheduleId?: string, before?: { startedAt: number; id: string }, filters?: { kind?: string; state?: string }): Promise<{ runs: RunSummary[]; hasMore: boolean; upcoming: UpcomingRun[]; upcomingTotal: number; attention: RunAttention[]; attentionTotal: number }> {
  const response = await rpc<Response>('run.list', { unread_only: unreadOnly, scope, ...filters, ...(before ? { before_started_at: before.startedAt, before_id: before.id } : {}), ...(scheduleId ? { source_id: scheduleId, kind: 'schedule' } : {}) })
  if (!response?.ok || !Array.isArray(response.runs)) throw new Error(response?.error || 'Run history unavailable')
  const upcoming: UpcomingRun[] = Array.isArray(response.upcoming) ? response.upcoming.flatMap(value => {
    if (!value || typeof value !== 'object') return []
    const v = value as Record<string, unknown>
    if (typeof v.id !== 'string' || !v.id || typeof v.title !== 'string' || typeof v.next_run_at !== 'string' || !Number.isFinite(Date.parse(v.next_run_at))) return []
    return [{ id: v.id, title: v.title, nextRunAt: v.next_run_at, timezone: typeof v.timezone === 'string' ? v.timezone : 'UTC', executionState: typeof v.execution_state === 'string' ? v.execution_state : 'unknown' }]
  }) : []
  const attention: RunAttention[] = Array.isArray(response.attention) ? response.attention.flatMap(value => {
    if (!value || typeof value !== 'object') return []
    const v = value as Record<string, unknown>
    return typeof v.id === 'string' && typeof v.title === 'string' && (v.kind === 'approval' || v.kind === 'question')
      ? [{ id: v.id, title: v.title, kind: v.kind }] : []
  }) : []
  return { attention, attentionTotal: typeof response.attention_total === 'number' && Number.isSafeInteger(response.attention_total) && response.attention_total >= attention.length ? response.attention_total : attention.length, upcoming, upcomingTotal: typeof response.upcoming_total === 'number' && Number.isSafeInteger(response.upcoming_total) && response.upcoming_total >= upcoming.length ? response.upcoming_total : upcoming.length, runs: response.runs.map(row).filter((value): value is RunSummary => value !== null), hasMore: response.has_more ?? response.runs.length === 100 }
}
export async function inspectRun(rpc: GatewayRpc, id: string, scope: 'session' | 'workspace' = 'session'): Promise<RunDetail> {
  const response = await rpc<Response>('run.inspect', { run_id: id, scope })
  const summary = row(response?.run)
  if (!response?.ok || !summary || summary.id !== id) throw new Error(response?.error || 'Run details unavailable')
  const detail = response.run as Record<string, unknown>
  const tokens = detail.tokenUsage && typeof detail.tokenUsage === 'object' ? detail.tokenUsage as Record<string, unknown> : undefined
  const tokenUsage = tokens && typeof tokens.input_tokens === 'number' && Number.isSafeInteger(tokens.input_tokens) && tokens.input_tokens >= 0
    && typeof tokens.output_tokens === 'number' && Number.isSafeInteger(tokens.output_tokens) && tokens.output_tokens >= 0 && typeof tokens.complete === 'boolean'
    ? { inputTokens: tokens.input_tokens, outputTokens: tokens.output_tokens, complete: tokens.complete } : undefined
  const raw = detail.reaction_health
  let reactionHealth: ReactionHealthView | undefined
  if (raw && typeof raw === 'object') {
    const value = raw as Record<string, unknown>
    if (typeof value.state === 'string' && ['waiting', 'queued', 'running', 'cancelling', 'awaiting-cleanup', 'cancelled', 'expired', 'exhausted'].includes(value.state)
      && typeof value.attempts === 'number' && Number.isSafeInteger(value.attempts) && value.attempts >= 0
      && typeof value.maxReactions === 'number' && Number.isSafeInteger(value.maxReactions) && value.maxReactions > 0
      && typeof value.pendingEvents === 'number' && Number.isSafeInteger(value.pendingEvents) && value.pendingEvents >= 0) {
      const usage = value.usage && typeof value.usage === 'object' ? value.usage as Record<string, unknown> : undefined
      const parsedUsage = usage && typeof usage.inputTokens === 'number' && Number.isSafeInteger(usage.inputTokens) && usage.inputTokens >= 0
        && typeof usage.outputTokens === 'number' && Number.isSafeInteger(usage.outputTokens) && usage.outputTokens >= 0 && typeof usage.complete === 'boolean'
        ? { inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, complete: usage.complete } : undefined
      reactionHealth = { ...(parsedUsage ? { usage: parsedUsage } : {}), state: value.state, attempts: value.attempts, maxReactions: value.maxReactions, pendingEvents: value.pendingEvents, lastError: typeof value.lastError === 'string' ? value.lastError : null }
    }
  }
  return { ...summary, ...(tokenUsage ? { tokenUsage } : {}), ...(typeof detail.cancel_label === 'string' && detail.cancel_label ? { cancelLabel: detail.cancel_label } : {}), ...(reactionHealth ? { reactionHealth } : {}), output: typeof detail.output === 'string' ? detail.output : '', outputTruncated: detail.outputTruncated === true,
    error: typeof detail.error === 'string' ? detail.error : null }
}
export async function acknowledgeRun(rpc: GatewayRpc, run: RunSummary, scope: 'session' | 'workspace' = 'session'): Promise<void> {
  const response = await rpc<Response>('run.acknowledge', { run_id: run.id, revision: run.revision, scope })
  if (!response?.ok) throw new Error(response?.error || 'Could not acknowledge run')
}

export async function listRuns(rpc: GatewayRpc, unreadOnly = false, scope: 'session' | 'workspace' = 'session', scheduleId?: string): Promise<RunSummary[]> {
  return (await listRunPage(rpc, unreadOnly, scope, scheduleId)).runs
}

export async function cancelRun(rpc: GatewayRpc, run: RunDetail, scope: 'session' | 'workspace'): Promise<void> {
  const response = await rpc<Response>('run.cancel', { run_id: run.id, revision: run.revision, scope })
  if (!response?.ok) throw new Error(response?.error || 'Run cancellation failed')
}
