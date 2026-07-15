// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject } from '../types/toolCalls.js'

const COMPACTION_UNAVAILABLE_RESULT = '[Tool result unavailable after context compaction]'

/**
 * Make a compaction window safe for providers: discard orphan tool results and
 * append synthetic errors for assistant calls whose paired result was removed.
 */
export function repairToolMessageSequence<T extends Record<string, unknown>>(messages: readonly T[]): T[] {
  const output: T[] = []
  let pending = new Map<string, string>()
  for (const message of messages) {
    if (message.role === 'assistant') {
      completePending(output, pending)
      output.push(message)
      pending = callsFor(message)
      continue
    }
    if (message.role === 'tool') {
      const callId = typeof message.tool_call_id === 'string' ? message.tool_call_id : ''
      if (callId && pending.has(callId)) {
        output.push(message)
        pending.delete(callId)
      }
      continue
    }
    completePending(output, pending)
    output.push(message)
  }
  completePending(output, pending)
  return output
}

function callsFor(message: Record<string, unknown>): Map<string, string> {
  const pending = new Map<string, string>()
  if (!Array.isArray(message.tool_calls)) {
    return pending
  }
  for (const call of message.tool_calls) {
    if (!isRecord(call) || typeof call.id !== 'string' || !call.id) {
      continue
    }
    const directName = typeof call.name === 'string' ? call.name : ''
    const functionName = isRecord(call.function) && typeof call.function.name === 'string' ? call.function.name : ''
    pending.set(call.id, directName || functionName)
  }
  return pending
}

function completePending<T extends Record<string, unknown>>(output: T[], pending: Map<string, string>): void {
  for (const [toolCallId, name] of pending) {
    output.push({
      role: 'tool',
      tool_call_id: toolCallId,
      name,
      content: COMPACTION_UNAVAILABLE_RESULT,
      is_error: true,
    } as unknown as T)
  }
  pending.clear()
}

function isRecord(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
