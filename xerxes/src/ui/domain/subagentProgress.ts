// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SubagentEventPayload, SubagentSnapshotPayload } from '../gatewayTypes.js'
import type { Msg, SubagentProgress, SubagentStatus } from '../types.js'

export type SubagentProgressPatch = (current: SubagentProgress) => Partial<SubagentProgress>

export const subagentProgressId = (payload: SubagentEventPayload): string =>
  payload.subagent_id || `sa:${payload.task_index}:${payload.goal || 'subagent'}`

export function mergeSubagentProgress(
  base: SubagentProgress,
  payload: SubagentEventPayload,
  patch: SubagentProgressPatch
): SubagentProgress {
  const outputTail = payload.output_tail
    ? payload.output_tail.map(entry => ({
        isError: Boolean(entry.is_error),
        preview: String(entry.preview ?? ''),
        tool: String(entry.tool ?? 'tool')
      }))
    : base.outputTail

  return {
    ...base,
    agentType: payload.agent_type ?? base.agentType,
    name: payload.agent_name ?? base.name,
    title: payload.title ?? base.title,
    apiCalls: payload.api_calls ?? base.apiCalls,
    costUsd: payload.cost_usd ?? base.costUsd,
    creatorId: payload.creator_id ?? base.creatorId,
    depth: payload.depth ?? base.depth,
    durationSeconds: payload.duration_seconds ?? base.durationSeconds,
    filesRead: payload.files_read ?? base.filesRead,
    filesWritten: payload.files_written ?? base.filesWritten,
    goal: payload.goal || base.goal,
    inputTokens: payload.input_tokens ?? base.inputTokens,
    iteration: payload.iteration ?? base.iteration,
    model: payload.model ?? base.model,
    outputTail,
    outputTokens: payload.output_tokens ?? base.outputTokens,
    parentId: payload.parent_id ?? base.parentId,
    cacheReadTokens: payload.cache_read_tokens ?? base.cacheReadTokens,
    cacheCreationTokens: payload.cache_creation_tokens ?? base.cacheCreationTokens,
    reasoningTokens: payload.reasoning_tokens ?? base.reasoningTokens,
    rules: payload.rules ?? base.rules,
    summary: payload.summary ?? base.summary,
    taskCount: payload.task_count ?? base.taskCount,
    toolCount: payload.tool_count ?? base.toolCount,
    toolsets: payload.toolsets ?? base.toolsets,
    ...patch(base)
  }
}

/** Terminal/visible status for a persisted snapshot row the daemon reports. */
const snapshotStatus = (row: SubagentSnapshotPayload): SubagentStatus => {
  const status = (row.status || '').toLowerCase()
  if (
    status === 'completed' ||
    status === 'error' ||
    status === 'failed' ||
    status === 'interrupted' ||
    status === 'queued' ||
    status === 'running' ||
    status === 'timeout'
  ) {
    return status
  }
  if (status === 'cancelled' || status === 'canceled') {
    return 'interrupted'
  }
  if (status === 'done' || status === 'success') {
    return 'completed'
  }

  // Unknown spelling: a closed child finished, anything else went away
  // without a terminal event and must not pose as a success.
  return row.closed === true ? 'completed' : 'interrupted'
}

/**
 * Rehydrate one persisted subagent manifest row (daemon subagent_snapshots)
 * into the progress shape trail cards render, so a reattached transcript
 * shows the session's spawned agents instead of dropping them visually.
 */
export const subagentProgressFromSnapshot = (row: SubagentSnapshotPayload, index: number): SubagentProgress => {
  const startedAt = Date.parse(row.created_at ?? '')

  return {
    ...(row.agent_id ? { agentType: row.agent_id } : {}),
    ...(row.api_calls === undefined ? {} : { apiCalls: row.api_calls }),
    creatorId: row.creator_id ?? row.parent_id ?? null,
    depth: 0,
    ...(row.files_read ? { filesRead: row.files_read } : {}),
    ...(row.files_written ? { filesWritten: row.files_written } : {}),
    goal: row.title || row.name || row.agent_id || 'subagent',
    id: row.id,
    index,
    ...(row.input_tokens === undefined ? {} : { inputTokens: row.input_tokens }),
    ...(row.model ? { model: row.model } : {}),
    ...(row.name ? { name: row.name } : {}),
    notes: [],
    ...(row.output_tokens === undefined ? {} : { outputTokens: row.output_tokens }),
    parentId: row.parent_id ?? null,
    ...(row.reasoning_tokens === undefined ? {} : { reasoningTokens: row.reasoning_tokens }),
    ...(row.rules ? { rules: row.rules } : {}),
    ...(Number.isFinite(startedAt) ? { startedAt } : {}),
    status: snapshotStatus(row),
    ...(row.summary || row.error ? { summary: row.summary || row.error } : {}),
    taskCount: 1,
    thinking: [],
    ...(row.title ? { title: row.title } : {}),
    toolCount: row.tool_count ?? 0,
    tools: [],
    ...(row.toolsets ? { toolsets: row.toolsets } : {})
  }
}

/** Update the most recent archived row for a background subagent without adding a new transcript row. */
export function reconcileArchivedSubagent(
  messages: Msg[],
  payload: SubagentEventPayload,
  patch: SubagentProgressPatch
): Msg[] {
  const id = subagentProgressId(payload)
  for (let messageIndex = messages.length - 1; messageIndex >= 0; messageIndex -= 1) {
    const message = messages[messageIndex]
    const agentIndex = message?.subagents?.findIndex(agent => agent.id === id) ?? -1
    if (!message?.subagents || agentIndex < 0) continue
    const current = message.subagents[agentIndex]
    if (!current) continue

    const subagents = [...message.subagents]
    subagents[agentIndex] = mergeSubagentProgress(current, payload, patch)
    const next = [...messages]
    next[messageIndex] = { ...message, subagents }
    return next
  }
  return messages
}
