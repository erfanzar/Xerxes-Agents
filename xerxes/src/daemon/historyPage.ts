// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { ValidationError } from '../core/errors.js'
import type { DaemonSession } from './runtime.js'
type RawMessage = DaemonSession['messages'][number]

export interface HistoryAction {
  id: string
  messages: RawMessage[]
  executions: unknown[]
  thinking: unknown[]
}
const record = (v: unknown): Record<string, unknown> => v && typeof v === 'object' && !Array.isArray(v) ? v as Record<string, unknown> : {}

/** One message/reasoning entry or one complete tool call/result pair per action. */
export function sessionHistoryActions(session: Pick<DaemonSession, 'messages' | 'toolExecutions' | 'thinkingContent'>): HistoryAction[] {
  const results = new Map<string, unknown>()
  const executions = new Map<string, unknown>()
  const calls = new Set<string>()
  const callRecords = (message: RawMessage): Record<string, unknown>[] => [
    ...(Array.isArray(message.tool_calls) ? message.tool_calls.map(record) : []),
    ...(Array.isArray(message.content) ? message.content.filter(p => record(p).type === 'tool_use').map(p => {
      const part = record(p); return { id: part.id, function: { name: part.name, arguments: part.input ?? part.arguments } }
    }) : []),
  ]
  for (const raw of session.toolExecutions) {
    const e = record(raw); const id = e.toolCallId ?? e.tool_call_id
    if (typeof id === 'string') executions.set(id, raw)
  }
  for (const m of session.messages) {
    if (m.role === 'tool' && typeof m.tool_call_id === 'string') results.set(m.tool_call_id, m.content)
    for (const p of Array.isArray(m.content) ? m.content : []) {
      const part = record(p)
      if (part.type === 'tool_result' && typeof part.tool_use_id === 'string') results.set(part.tool_use_id, part.content)
    }
    for (const call of callRecords(m)) if (typeof call.id === 'string') calls.add(call.id)
  }
  const actions: HistoryAction[] = []
  let assistant = 0
  const tool = (id: string, call: Record<string, unknown>, index: string): void => {
    const execution = record(executions.get(id))
    const fn = record(call.function)
    const result = results.get(id) ?? execution.result
    actions.push({ id: index, messages: [
      { role: 'assistant', content: '', tool_calls: [{ id, type: 'function', function: { name: String(fn.name ?? execution.name ?? 'Tool'), arguments: fn.arguments ?? execution.inputs ?? {} } }] },
      ...(result !== undefined ? [{ role: 'tool', tool_call_id: id, content: result } as RawMessage] : []),
    ], executions: executions.has(id) ? [execution] : [], thinking: [] })
  }
  session.messages.forEach((m, index) => {
    if (m.role === 'system') return
    if (m.role === 'tool') {
      const id = String(m.tool_call_id ?? `legacy-${index}`)
      if (!calls.has(id)) tool(id, {}, `${index}:tool`)
      return
    }
    const content = Array.isArray(m.content) ? m.content.filter(p => !['tool_use', 'tool_result'].includes(String(record(p).type))) : m.content
    const thinking = m.role === 'assistant' ? m.thinking ?? session.thinkingContent[assistant] : undefined
    if (m.role === 'assistant') assistant++
    if ((typeof content === 'string' ? content.trim().length > 0 : Array.isArray(content) && content.length > 0) || thinking) {
      actions.push({ id: `${index}:message`, messages: [{ ...m, content, tool_calls: undefined }], executions: [], thinking: thinking ? [thinking] : [] })
    }
    callRecords(m).forEach((call, offset) => { if (typeof call.id === 'string') tool(call.id, call, `${index}:tool:${offset}`) })
  })
  return actions
}

export function historyLimit(value: unknown): number | undefined {
  if (value === undefined) return undefined
  if (!Number.isInteger(value) || (value as number) < 0 || (value as number) > 100) throw new ValidationError('history_limit', 'must be an integer from 0 to 100', value)
  return value as number
}

export function sessionHistoryPage(session: Pick<DaemonSession, 'id' | 'messages' | 'toolExecutions' | 'thinkingContent'>, limit: number, before?: unknown) {
  const actions = sessionHistoryActions(session)
  const anchor = (index: number) => createHash('sha256').update(JSON.stringify(actions[index])).digest('hex')
  let end = actions.length
  if (before !== undefined && before !== null) {
    if (typeof before !== 'string' || before.length > 1024) throw new ValidationError('before', 'invalid history cursor', before)
    let cursor: Record<string, unknown>
    try { cursor = record(JSON.parse(Buffer.from(before, 'base64url').toString())) } catch { throw new ValidationError('before', 'invalid history cursor', before) }
    if (cursor.session !== session.id || !Number.isInteger(cursor.index) || (cursor.index as number) < 0 || (cursor.index as number) >= actions.length || cursor.anchor !== anchor(cursor.index as number)) {
      throw new ValidationError('before', 'History changed. Reopen this session to load its current history.', before)
    }
    end = cursor.index as number
  }
  const start = Math.max(0, end - limit)
  return { actions: actions.slice(start, end), has_more: start > 0, before: start > 0 ? Buffer.from(JSON.stringify({ session: session.id, index: start, anchor: anchor(start) })).toString('base64url') : null, total_actions: actions.length }
}
