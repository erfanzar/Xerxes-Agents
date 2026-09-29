// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Keeping the in-chat agents card in step with the fleet (daemon snapshots
 * plus live subagent events). Pure so it can be tested headlessly and fast
 * enough for workflows with thousands of agents: every lookup is by map,
 * never a scan of all members per row.
 */

import { toolPhrase } from './activityPhrase.js'
import type { AgentMember, SessionRow, ToolItem } from './types.js'

/** Snapshot statuses fold onto the card's display vocabulary. */
export function agentStatusOf(status: string): string {
  const s = status.toLowerCase()
  if (s === 'working' || s === 'running' || s === 'starting' || s === 'waiting') return 'working'
  if (s === 'completed' || s === 'done' || s === 'closed') return 'completed'
  if (s === 'error' || s === 'failed') return 'failed'
  if (s === 'cancelled' || s === 'interrupted') return 'cancelled'
  return s || 'working'
}

/**
 * Fold fleet rows into the card's members, in place. A child is matched by
 * id once known; before that, by title — only to a row still waiting for its
 * agent, newest first (a failed spawn and its retry share a title, and the
 * live child must not revive the failed row). Unseen rows join only while
 * live, so the previous turn's terminal snapshots never open a stale card.
 * Returns whether anything changed.
 */
export function syncAgentMembers(members: Map<string, AgentMember>, fleet: readonly SessionRow[]): boolean {
  if (fleet.length === 0) return false
  const byRuntime = new Map<string, string>()
  const waiting = new Map<string, string[]>()
  for (const member of members.values()) {
    if (member.runtimeId) byRuntime.set(member.runtimeId, member.key)
    else if (member.status === 'working') {
      const keys = waiting.get(member.title) ?? []
      keys.push(member.key)
      waiting.set(member.title, keys)
    }
  }
  let touched = false
  for (const row of fleet) {
    const status = agentStatusOf(row.status)
    let key = byRuntime.get(row.id)
    if (!key) {
      const keys = waiting.get(row.title)
      key = keys?.pop()
      if (key) byRuntime.set(row.id, key)
    }
    const existing = key ? members.get(key) : members.get(row.id)
    if (!existing) {
      if (status !== 'working') continue
      const created = enrich({ key: row.id, runtimeId: row.id, title: row.title, status }, row, status)
      members.set(row.id, created)
      byRuntime.set(row.id, row.id)
      touched = true
      continue
    }
    const next = enrich({ ...existing, runtimeId: row.id, status }, row, status)
    if (!sameMember(existing, next)) {
      members.set(existing.key, next)
      touched = true
    }
  }
  return touched
}

function enrich(member: AgentMember, row: SessionRow, status: string): AgentMember {
  const details = row.agentDetails
  if (!details) return member
  const tokens = (details.inputTokens ?? 0) + (details.outputTokens ?? 0)
  const working = status === 'working'
  const activity = working ? agentActivity(details) : undefined
  const summary = !working ? firstLine(details.summary) : undefined
  const finishedAt = working ? undefined : member.finishedAt ?? details.lastEventAt
  const calls = details.toolCalls ?? []
  const toolUses = Math.max(details.toolCount ?? 0, calls.length)
  const recentTools = calls.slice(-RECENT_TOOLS).map(call => toolPhrase(readableCall(call)))
  const { activity: _activity, summary: _summary, finishedAt: _finished, recentTools: _recent, toolUses: _uses, ...rest } = member
  return {
    ...rest,
    // The run's cost arrives on the member from the Workflow result; events
    // only carry the id and phase, so merge rather than replace.
    ...(details.group ? { group: { ...member.group, ...details.group } } : {}),
    ...(details.model && !member.model ? { model: details.model } : {}),
    ...(details.providerProfile && !member.providerProfile ? { providerProfile: details.providerProfile } : {}),
    ...(details.reasoningEffort && !member.reasoningEffort ? { reasoningEffort: details.reasoningEffort } : {}),
    ...(details.baseAgent && !member.baseAgent ? { baseAgent: details.baseAgent } : {}),
    ...(details.error && status === 'failed' ? { error: details.error } : {}),
    ...(details.startedAt !== undefined ? { startedAt: member.startedAt ?? details.startedAt } : {}),
    ...(finishedAt !== undefined ? { finishedAt } : {}),
    ...(tokens > 0 ? { tokens } : {}),
    ...(activity ? { activity } : {}),
    ...(summary ? { summary } : {}),
    ...(toolUses ? { toolUses } : {}),
    ...(recentTools.length ? { recentTools } : {}),
  }
}

function sameMember(left: AgentMember, right: AgentMember): boolean {
  return left.status === right.status && left.runtimeId === right.runtimeId && left.activity === right.activity
    && left.tokens === right.tokens && left.summary === right.summary && left.error === right.error
    && left.model === right.model && left.startedAt === right.startedAt && left.finishedAt === right.finishedAt
    && left.group?.id === right.group?.id && left.group?.phase === right.group?.phase && left.group?.label === right.group?.label
    && left.providerProfile === right.providerProfile && left.baseAgent === right.baseAgent
    && left.toolUses === right.toolUses && (left.recentTools ?? []).join('\n') === (right.recentTools ?? []).join('\n')
}

/** What a working agent is doing now: its running tool, else what it last wrote or thought. */
export function agentActivity(details: NonNullable<SessionRow['agentDetails']>): string | undefined {
  const running = details.toolCalls?.findLast(call => call.state === 'working')
  if (running) return toolPhrase(readableCall(running))
  if (details.notes?.length) return 'Writing'
  if (details.thinking?.length) return 'Thinking'
  return undefined
}

/** The last few tool uses a row can expand to, like Claude Code's "+N more tool uses". */
const RECENT_TOOLS = 5

/**
 * A child's tool arguments arrive as the runtime's bounded preview —
 * `cmd=env, args=A=1,B=2, workdir=.` — or, from older runtimes, as JSON.
 * Either way, pull out the part a person reads.
 */
export function previewArguments(raw: string): Record<string, unknown> {
  const text = raw.trim()
  if (!text) return {}
  if (text.startsWith('{')) {
    try {
      const parsed: unknown = JSON.parse(text)
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed as Record<string, unknown>
    } catch { /* a truncated JSON preview falls through to the key=value form */ }
  }
  const args: Record<string, unknown> = {}
  for (const piece of text.split(/, (?=[A-Za-z_][\w]*=)/)) {
    const equals = piece.indexOf('=')
    if (equals <= 0) continue
    args[piece.slice(0, equals)] = piece.slice(equals + 1)
  }
  return args
}

function readableCall(call: ToolItem): ToolItem {
  const args = previewArguments(call.input || call.arg || '')
  if (!Object.keys(args).length) return call
  const pick = (...keys: string[]) => {
    for (const key of keys) {
      const value = args[key]
      if (typeof value === 'string' && value.trim()) return value.trim()
      if (Array.isArray(value) && value.every(item => typeof item === 'string') && value.length) return value.join(' ')
    }
    return ''
  }
  const path = pick('file_path', 'path', 'notebook_path')
  const command = [pick('cmd', 'command'), pick('args')].filter(Boolean).join(' ')
  const arg = path || command || pick('pattern', 'query', 'url', 'prompt')
  return { ...call, arg, ...(path ? { path } : {}) }
}

function firstLine(text: string | undefined): string | undefined {
  const line = text?.split('\n').map(value => value.trim()).find(Boolean)
  return line ? (line.length > 160 ? `${line.slice(0, 159)}…` : line) : undefined
}
