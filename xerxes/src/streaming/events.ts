// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject, ToolCall } from '../types/toolCalls.js'
import type { TokenUsage } from '../llms/client.js'

export interface AgentState {
  /** False when an imported history predates exact cumulative API-call accounting. */
  apiCallsComplete: boolean
  readonly messages: import('../types/messages.js').ChatMessage[]
  metadata: Record<string, unknown>
  thinkingContent: string[]
  toolExecutions: ToolExecutionRecord[]
  totalApiCalls: number
  totalCacheCreationTokens: number
  totalCacheReadTokens: number
  totalInputTokens: number
  totalOutputTokens: number
  turnCount: number
  /** False when any provider round omitted usage or failed after it may have consumed tokens. */
  usageComplete: boolean
}

export interface ToolExecutionRecord {
  readonly durationMs: number
  readonly inputs: JsonObject
  readonly name: string
  readonly permitted: boolean
  readonly result: string
  readonly toolCallId: string
}

export interface PermissionRequest {
  readonly description: string
  readonly inputs: JsonObject
  readonly requestId: string
  readonly toolCall: ToolCall
}

export interface ToolResult {
  readonly durationMs: number
  readonly name: string
  readonly permitted: boolean
  readonly result: string
  readonly toolCallId: string
}

export type StreamEvent =
  | { readonly text: string; readonly type: 'text' }
  | { readonly text: string; readonly type: 'thinking' }
  | { readonly attempt: number; readonly delay: number; readonly error: string; readonly final: boolean; readonly maxAttempts: number; readonly type: 'provider_retry' }
  | { readonly call: ToolCall; readonly type: 'tool_start' }
  | { readonly request: PermissionRequest; readonly type: 'permission_request' }
  | { readonly result: ToolResult; readonly type: 'tool_end' }
  | {
    readonly apiCallsCount?: number
    readonly model: string
    readonly toolCallsCount: number
    readonly type: 'turn_done'
    readonly usage: TokenUsage
    /** True only when every provider round supplied exact token usage. */
    readonly usageComplete?: boolean
  }
  | { readonly description: string; readonly skillName: string; readonly sourcePath: string; readonly toolCount: number; readonly type: 'skill_suggestion'; readonly uniqueTools: readonly string[]; readonly version: string }

export function createAgentState(messages: import('../types/messages.js').ChatMessage[] = []): AgentState {
  return {
    apiCallsComplete: true,
    messages,
    metadata: {},
    thinkingContent: [],
    toolExecutions: [],
    totalApiCalls: 0,
    totalCacheCreationTokens: 0,
    totalCacheReadTokens: 0,
    totalInputTokens: 0,
    totalOutputTokens: 0,
    turnCount: 0,
    usageComplete: true,
  }
}
