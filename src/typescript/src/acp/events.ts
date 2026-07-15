// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { StreamEvent } from '../streaming/events.js'
import type { AcpWireEvent } from './types.js'

/** Event kinds exposed by Xerxes' Agent Client Protocol adapter. */
export enum AcpEventKind {
  TEXT_DELTA = 'text_delta',
  THINKING_DELTA = 'thinking_delta',
  TOOL_CALL_START = 'tool_call_start',
  TOOL_CALL_END = 'tool_call_end',
  PERMISSION_REQUEST = 'permission_request',
  TURN_END = 'turn_end',
  SKILL_SUGGESTION = 'skill_suggestion',
  UNKNOWN = 'unknown',
}

/** A tagged ACP event with its payload separated for convenient in-process use. */
export class AcpEvent {
  readonly kind: AcpEventKind
  readonly payload: Readonly<Record<string, unknown>>

  constructor(kind: AcpEventKind, payload: Readonly<Record<string, unknown>> = {}) {
    this.kind = kind
    this.payload = payload
  }

  /** Flatten the discriminator and payload into the JSON wire shape. */
  toWire(): AcpWireEvent {
    return { kind: this.kind, ...this.payload }
  }
}

/** Convert a portable streaming-loop event into the ACP event vocabulary. */
export function toAcpEvent(event: StreamEvent | unknown): AcpEvent {
  if (!isStreamEvent(event)) {
    return new AcpEvent(AcpEventKind.UNKNOWN, { repr: describeUnknown(event) })
  }

  switch (event.type) {
    case 'text':
      return new AcpEvent(AcpEventKind.TEXT_DELTA, { text: event.text })
    case 'thinking':
      return new AcpEvent(AcpEventKind.THINKING_DELTA, { text: event.text })
    case 'tool_start':
      return new AcpEvent(AcpEventKind.TOOL_CALL_START, {
        name: event.call.function.name,
        inputs: event.call.function.arguments,
        tool_call_id: event.call.id,
      })
    case 'tool_end':
      return new AcpEvent(AcpEventKind.TOOL_CALL_END, {
        name: event.result.name,
        result: event.result.result,
        permitted: event.result.permitted,
        tool_call_id: event.result.toolCallId,
        duration_ms: event.result.durationMs,
      })
    case 'permission_request':
      return new AcpEvent(AcpEventKind.PERMISSION_REQUEST, {
        tool_name: event.request.toolCall.function.name,
        description: event.request.description,
        inputs: event.request.inputs,
      })
    case 'turn_done':
      return new AcpEvent(AcpEventKind.TURN_END, {
        input_tokens: event.usage.inputTokens,
        output_tokens: event.usage.outputTokens,
        tool_calls_count: event.toolCallsCount,
        model: event.model,
        cache_read_tokens: event.usage.cacheReadTokens ?? 0,
        cache_creation_tokens: event.usage.cacheCreationTokens ?? 0,
      })
    case 'skill_suggestion':
      return new AcpEvent(AcpEventKind.SKILL_SUGGESTION, {
        skill_name: event.skillName,
        description: event.description,
      })
    default:
      return new AcpEvent(AcpEventKind.UNKNOWN, { repr: describeUnknown(event), type: event.type })
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isStreamEvent(value: unknown): value is StreamEvent {
  return isRecord(value) && typeof value.type === 'string'
}

function describeUnknown(value: unknown): string {
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}
