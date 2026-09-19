// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { useEffect, useState } from 'react'
import { useOptionalGateway } from '../app/gatewayContext.js'
import { patchOverlayState } from '../app/overlayStore.js'
import type { Theme } from '../theme.js'
import { Box, Text } from './primitives.js'
import { parseScheduleTokens, scheduleTokensLabel, type ScheduleTokens } from '../lib/scheduleTokens.js'

interface Followup {
  id: string
  prompt: string
  paused: boolean
  execution: 'idle' | 'running' | 'cancelling'
  used: number
  maximum: number
  expires: number
  next: number | null
  last: number | null
  recovery: boolean
  outcome: string | null
  completed: boolean
  tokens?: ScheduleTokens
}
const record = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid follow-up response')
  return value as Record<string, unknown>
}
const time = (value: unknown): number | null => {
  if (value == null) return null
  if (typeof value !== 'string' || !Number.isFinite(Date.parse(value))) throw new Error('Invalid follow-up time')
  return Date.parse(value)
}
function parse(value: unknown, owner: string): Followup[] {
  const response = record(value)
  if (response.ok !== true) throw new Error(typeof response.error === 'string' ? response.error : 'Follow-ups unavailable')
  if (response.owner_session_id !== owner) throw new Error('Conversation changed; refreshing follow-ups')
  if (!Array.isArray(response.jobs)) throw new Error('Invalid follow-up list')
  return response.jobs.map(value => {
    const row = record(value)
    const execution = row.execution_state
    if (typeof row.id !== 'string' || typeof row.prompt !== 'string' || typeof row.paused !== 'boolean'
      || row.target_session_id !== owner || !['idle', 'running', 'cancelling'].includes(String(execution))
      || typeof row.runs_started !== 'number' || !Number.isSafeInteger(row.runs_started) || row.runs_started < 0
      || typeof row.max_runs !== 'number' || !Number.isSafeInteger(row.max_runs) || row.max_runs < 1) {
      throw new Error('Invalid bounded follow-up')
    }
    const expires = time(row.expires_at)
    if (expires === null) throw new Error('Follow-up expiry missing')
    const outcome = row.latest_attempt == null ? null : record(row.latest_attempt).state
    if (outcome !== null && !['running', 'succeeded', 'failed', 'cancelled', 'interrupted'].includes(String(outcome))) throw new Error('Invalid follow-up outcome')
    return { id: row.id, tokens: parseScheduleTokens(row.token_budget), prompt: row.prompt, paused: row.paused, execution: execution as Followup['execution'],
      used: row.runs_started, maximum: row.max_runs, expires, next: time(row.next_run_at), last: time(row.last_run_at), outcome: outcome as string | null,
      completed: row.metadata != null && record(row.metadata).followup_completion != null,
      recovery: row.metadata != null && record(row.metadata).execution_recovery_required === true }
  })
}
function status(job: Followup, now: number): string {
  if (job.execution === 'cancelling') return 'Cancelling'
  if (job.execution === 'running') return 'Running or queued for conversation'
  if (job.completed) return 'Condition met · model reported'
  if (job.recovery) return 'Needs review after interruption'
  if (job.expires <= now) return 'Expired'
  if (job.used >= job.maximum) return 'Attempt limit reached'
  if (job.tokens?.blocked) return job.tokens.complete ? 'Token threshold reached' : 'Token usage incomplete · needs review'
  if (job.paused) return 'Paused'
  if (job.next === null) return 'No next wake scheduled'
  if (job.next <= now) return 'Due · waiting for scheduler'
  return `Next ${new Date(job.next).toLocaleString()}`
}
export function openSessionFollowups(): void {
  patchOverlayState({ goal: false, schedules: false, loops: true })
}

/** Read-only polling of deterministic scheduler state; never invokes a model. */
export function SessionFollowups({ t, sessionId, expanded = false }: { t: Theme; sessionId?: string | null; expanded?: boolean }) {
  const gateway = useOptionalGateway()
  const [snapshot, setSnapshot] = useState<{ owner: string; jobs: Followup[]; now: number; error: string } | null>(null)
  useEffect(() => {
    if (!gateway || !sessionId) return
    let alive = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const load = async () => {
      try {
        // This poll owns its inline error display. A late failure from a
        // previous conversation must never become current transcript content.
        const result = await gateway.rpc<{ ok: boolean; owner_session_id?: string; jobs?: unknown; error?: string }>('schedule.list', { scope: 'session', owner_session_id: sessionId, summary: true }, { reportError: false })
        const jobs = parse(result, sessionId)
        if (alive) setSnapshot({ owner: sessionId, jobs, now: Date.now(), error: '' })
      } catch (error) {
        if (alive) setSnapshot({ owner: sessionId, jobs: [], now: Date.now(), error: String(error) })
      } finally {
        if (alive) timer = setTimeout(() => { void load() }, 5000)
      }
    }
    void load()
    return () => { alive = false; if (timer) clearTimeout(timer) }
  }, [gateway, sessionId])
  if (!gateway || !sessionId) return null
  const current = snapshot?.owner === sessionId ? snapshot : null
  if (!current) return expanded ? <Text color={t.ds.secondary}>Loading follow-ups…</Text> : null
  if (current.error) return <Text color={t.color.warn} wrap={expanded ? 'wrap' : 'truncate-end'}>{expanded ? current.error : 'Follow-ups unavailable · /loop to retry'}</Text>
  if (!current.jobs.length) return expanded ? <Text color={t.ds.secondary}>No follow-ups · L to create</Text> : null
  // Surface active and eligible jobs first; retain paused/expired jobs in the inspector.
  const priority = (job: Followup) => job.execution !== 'idle' ? 0 : job.recovery || job.tokens?.blocked ? 1 : !job.paused && job.expires > current.now && job.used < job.maximum ? 2 : 3
  const jobs = [...current.jobs].sort((a, b) => priority(a) - priority(b) || (a.next ?? Infinity) - (b.next ?? Infinity) || a.id.localeCompare(b.id))
  const visible = expanded ? jobs : jobs.slice(0, 1)
  return <Box flexDirection="column" flexShrink={0} onClick={openSessionFollowups}>
    {visible.map(job => <Box key={job.id} flexDirection="column" flexShrink={0} marginTop={expanded ? 1 : 0}>
      <Text color={t.color.accent} wrap={expanded ? 'wrap' : 'truncate-end'}>{expanded ? job.prompt : 'Follow-ups'} · {status(job, current.now)}</Text>
      <Text color={t.ds.secondary} wrap={expanded ? 'wrap' : 'truncate-end'}>{job.used}/{job.maximum} attempts · {Math.max(0, job.maximum - job.used)} remaining{job.outcome ? ` · Latest: ${job.outcome}` : ''}{expanded ? '' : ` · ${jobs.length} total · /loop controls`}</Text>
      {expanded ? <Text color={t.ds.secondary} wrap="wrap">Last run: {job.last === null ? 'none recorded' : new Date(job.last).toLocaleString()} · L for output history and pause/cancel</Text> : null}
      {expanded && job.tokens ? <Text color={job.tokens.blocked ? t.color.warn : t.ds.secondary} wrap="wrap">{scheduleTokensLabel(job.tokens)}</Text> : null}
    </Box>)}
  </Box>
}
