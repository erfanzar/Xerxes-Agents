// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SessionRow, ToolItem } from './types.js'
const text = (value: unknown): string => typeof value === 'string' ? value : ''
const record = (value: unknown): Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
const count = (value: unknown): number | undefined => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
const strings = (value: unknown, fallback: readonly string[]): readonly string[] => Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : fallback
const append = (values: readonly string[] = [], value: string): readonly string[] => value ? [...values, value].slice(-40) : values

/** Decode the existing recursive v35 subagent_event envelope; never mix child output into main chat. */
export function foldAgentEvent(rows: readonly SessionRow[], envelope: Record<string, unknown>, now = Date.now(), depth = 0): readonly SessionRow[] {
  if (depth > 32) return rows
  const event = record(envelope.event)
  const payload = record(event.payload)
  if (event.type === 'subagent_event') return foldAgentEvent(rows, payload, now, depth + 1)
  if (!['turn_begin','turn_end','think_part','text_part','tool_call','tool_result'].includes(text(event.type))) return rows
  const id = text(envelope.agent_id)
  if (!id) return rows
  const existing = rows.find(row => row.id === id)
  const old = existing?.agentDetails
  const kind = text(event.type)
  let status = existing?.status || 'running'
  if (kind === 'turn_begin') status = text(payload.status) || 'running'
  if (kind === 'turn_end') status = text(payload.status) || (payload.cancelled === true ? 'interrupted' : payload.error ? 'failed' : 'completed')
  let calls = kind === 'turn_begin' ? [] : old?.toolCalls ?? []
  if (kind === 'tool_call') {
    const callId = text(payload.tool_call_id) || text(payload.id) || `${id}:${calls.length}`
    if (!calls.some(call => call.id === callId)) calls = [...calls, { id: callId, name: text(payload.name) || 'tool', verb: text(payload.name) || 'tool', arg: text(payload.arguments), input: text(payload.arguments), output: '', dur: '', state: 'working' } satisfies ToolItem].slice(-100)
  }
  if (kind === 'tool_result') {
    const callId = text(payload.tool_call_id) || text(payload.id)
    const match = callId ? calls.findIndex(call => call.id === callId) : calls.findLastIndex(call => call.state === 'working' && call.name === text(payload.name))
    const failed = payload.permitted === false || payload.ok === false || Boolean(payload.error)
    const result = text(payload.return_value) || text(payload.output)
    const duration = count(payload.duration_ms)
    const base = match >= 0 ? calls[match]! : { id: callId || `${id}:result:${calls.length}`, name: text(payload.name) || 'tool', verb: text(payload.name) || 'tool', input: '', arg: '', output: '', dur: '', state: 'working' as const }
    const next: ToolItem = { ...base, state: failed ? 'failed' : 'done', output: result, dur: duration === undefined ? '' : `${(duration / 1000).toFixed(1)}s`, ...(failed ? { error: text(payload.error) || result || 'Tool denied or failed' } : {}) }
    calls = match >= 0 ? calls.map((call,index) => index === match ? next : call) : [...calls,next].slice(-100)
  }
  if (kind === 'turn_end') calls = calls.map(call => call.state === 'working' ? { ...call, state: 'failed', error: 'Agent ended before a tool result was received' } : call)
  const durationSeconds = count(payload.duration_seconds) ?? old?.durationSeconds
  const next: SessionRow = {
    ...(existing ?? { id, key:id, age:'', current:false, kind:'subagent', turns:0, messages:0, cwd:'', untitled:false }),
    title: text(envelope.title) || text(envelope.agent_name) || existing?.title || id,
    status,
    agentDetails: {
      ...old,
      summary: text(envelope.summary) || text(payload.summary) || (kind === 'turn_begin' ? '' : old?.summary) || '',
      error: text(payload.error) || (kind === 'turn_begin' ? '' : old?.error) || '', model: text(envelope.model) || old?.model || '',
      goal: text(envelope.goal) || old?.goal || '', parentId: text(envelope.parent_id) || old?.parentId || '',
      toolCount: count(envelope.tool_count) ?? old?.toolCount,
      inputTokens: count(envelope.input_tokens) ?? old?.inputTokens,
      outputTokens: count(envelope.output_tokens) ?? old?.outputTokens,
      filesRead: strings(envelope.files_read,old?.filesRead ?? []), filesWritten: strings(envelope.files_written,old?.filesWritten ?? []),
      thinking: kind === 'turn_begin' ? [] : kind === 'think_part' ? append(old?.thinking,text(payload.think)) : old?.thinking ?? [],
      notes: kind === 'turn_begin' ? [] : kind === 'text_part' ? append(old?.notes,text(payload.text)) : old?.notes ?? [],
      toolCalls: calls,
      startedAt: kind === 'turn_begin' ? now : old?.startedAt ?? now,
      lastEventAt: now,
      ...(durationSeconds === undefined ? {} : { durationSeconds }),
    },
  }
  return existing ? rows.map(row=>row.id===id?next:row) : [...rows,next]
}
