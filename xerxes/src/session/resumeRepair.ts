// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { stripAssistantToolCallMarkers } from '../streaming/toolMarkers.js'
import { isJsonObject, type JsonObject, type JsonValue } from '../types/toolCalls.js'

/** Explicit placeholder retained until a host elects to replay an interrupted tool call. */
export const RESUME_REPLAY_SENTINEL = '[interrupted: pending replay]'

/** One transcript message accepted by the resume-repair boundary. */
export type ResumeMessage = Record<string, unknown>

/** Python-compatible descriptor for an interrupted tool call that may be replayed later. */
export interface PendingResumeReplay {
  readonly arguments: string
  readonly name: string
  readonly tool_call_id: string
}

export interface ResumeRepairResult {
  readonly messages: readonly ResumeMessage[]
  readonly pendingReplays: readonly PendingResumeReplay[]
}

/** Explicit host-owned executor used only when replay is deliberately requested. */
export interface ResumeReplayExecutor {
  execute(name: string, arguments_: JsonObject): Promise<string | null | undefined> | string | null | undefined
}

export interface ResumeReplayOptions {
  readonly executor?: ResumeReplayExecutor
}

/**
 * Repair a resumed transcript so every assistant tool call has one matching tool reply.
 *
 * Assistant provider markers are removed from visible content. Missing replies receive
 * {@link RESUME_REPLAY_SENTINEL} and a descriptor; stale or orphaned tool replies are
 * dropped because no preceding assistant call can make them valid for a provider.
 */
export function repairResumedTranscript(messages: readonly ResumeMessage[]): ResumeRepairResult {
  const outstanding = new Map<string, PendingResumeReplay>()
  const repaired: ResumeMessage[] = []
  const pendingReplays: PendingResumeReplay[] = []

  const flushOutstanding = (): void => {
    for (const [toolCallId, replay] of outstanding) {
      repaired.push({
        role: 'tool',
        tool_call_id: toolCallId,
        content: RESUME_REPLAY_SENTINEL,
      })
      pendingReplays.push(replay)
    }
    outstanding.clear()
  }

  for (const source of messages) {
    const role = stringValue(source.role)
    if (role === 'assistant') {
      if (outstanding.size) flushOutstanding()
      const message = copyAssistantMessage(source)
      repaired.push(message)
      for (const replay of toolCallReplays(message.tool_calls)) {
        outstanding.set(replay.tool_call_id, replay)
      }
      continue
    }

    if (role === 'tool') {
      const toolCallId = stringValue(source.tool_call_id)
      if (toolCallId && outstanding.delete(toolCallId)) {
        repaired.push({ ...source })
      }
      continue
    }

    if (outstanding.size) flushOutstanding()
    repaired.push({ ...source })
  }

  if (outstanding.size) flushOutstanding()
  return { messages: repaired, pendingReplays }
}

/**
 * Optionally replace pending replay sentinels through a caller-injected executor.
 *
 * Without an executor, repaired messages and descriptors are returned intact. With
 * one, every pending descriptor is consumed exactly once, like the Python resume
 * path; per-call parse/execution failures become explicit transcript content rather
 * than preventing the remainder of a session from resuming.
 */
export async function replayPendingToolCalls(
  repair: ResumeRepairResult,
  options: ResumeReplayOptions = {},
): Promise<ResumeRepairResult> {
  const messages = repair.messages.map(message => ({ ...message }))
  const pendingReplays = repair.pendingReplays.map(replay => ({ ...replay }))
  const executor = options.executor
  if (executor === undefined) {
    return { messages, pendingReplays }
  }

  const byToolCallId = new Map(pendingReplays.map(replay => [replay.tool_call_id, replay]))
  for (const message of messages) {
    if (stringValue(message.role) !== 'tool' || message.content !== RESUME_REPLAY_SENTINEL) continue
    const toolCallId = stringValue(message.tool_call_id)
    const replay = byToolCallId.get(toolCallId)
    if (replay === undefined) continue
    try {
      const arguments_ = replayArguments(replay.arguments)
      try {
        message.content = await executor.execute(replay.name, arguments_) ?? ''
      } catch (error) {
        message.content = `[replay error: ${errorMessage(error)}]`
      }
    } catch (error) {
      message.content = `[replay error: invalid arguments — ${errorMessage(error)}]`
    }
  }

  return { messages, pendingReplays: [] }
}

function copyAssistantMessage(source: ResumeMessage): ResumeMessage {
  const message = { ...source }
  if (typeof message.content === 'string') {
    message.content = stripAssistantToolCallMarkers(message.content)
  }
  return message
}

function toolCallReplays(value: unknown): PendingResumeReplay[] {
  if (!Array.isArray(value)) return []
  const replays: PendingResumeReplay[] = []
  for (const call of value) {
    if (!isPlainRecord(call)) continue
    const toolCallId = stringValue(call.id)
    if (!toolCallId) continue
    const functionValue = isPlainRecord(call.function) ? call.function : undefined
    const name = stringValue(functionValue?.name) || stringValue(call.name)
    replays.push({
      tool_call_id: toolCallId,
      name,
      arguments: rawArguments(functionValue?.arguments, call.input),
    })
  }
  return replays
}

function rawArguments(functionArguments: unknown, input: unknown): string {
  if (typeof functionArguments === 'string' && functionArguments) return functionArguments
  if (isJsonObject(functionArguments)) {
    try {
      return JSON.stringify(functionArguments)
    } catch {
      return ''
    }
  }
  if (typeof input === 'string') return input
  if (!isJsonObject(input)) return ''
  try {
    return JSON.stringify(input)
  } catch {
    return ''
  }
}

function replayArguments(rawArguments: string): JsonObject {
  if (!rawArguments) return {}
  const parsed: unknown = JSON.parse(rawArguments)
  if (isJsonObject(parsed)) return parsed
  if (!isJsonValue(parsed)) {
    throw new TypeError('tool arguments must decode to JSON')
  }
  return { value: parsed }
}

function isJsonValue(value: unknown): value is JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') return true
  if (Array.isArray(value)) return value.every(isJsonValue)
  return isJsonObject(value) && Object.values(value).every(isJsonValue)
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
