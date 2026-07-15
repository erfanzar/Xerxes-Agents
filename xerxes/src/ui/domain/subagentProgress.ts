// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SubagentEventPayload } from '../gatewayTypes.js'
import type { Msg, SubagentProgress } from '../types.js'

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
    reasoningTokens: payload.reasoning_tokens ?? base.reasoningTokens,
    rules: payload.rules ?? base.rules,
    summary: payload.summary ?? base.summary,
    taskCount: payload.task_count ?? base.taskCount,
    toolCount: payload.tool_count ?? base.toolCount,
    toolsets: payload.toolsets ?? base.toolsets,
    ...patch(base)
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
