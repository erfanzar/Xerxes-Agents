// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { OpenAiCustomToolCall, OpenAiToolCall, ToolCall } from './toolCalls.js'
import { toolCallToOpenAi, toolCallToOpenAiCustom } from './toolCalls.js'

export interface TextContentPart {
  readonly type: 'text'
  readonly text: string
}

export interface ImageUrlContentPart {
  readonly type: 'image_url'
  readonly image_url: {
    readonly url: string
    readonly detail?: 'auto' | 'high' | 'low'
  }
}

export type ContentPart = ImageUrlContentPart | TextContentPart
export type MessageContent = ContentPart[] | string
export type MessageRole = 'assistant' | 'system' | 'tool' | 'user'

export interface SystemMessage {
  readonly role: 'system'
  readonly content: MessageContent
}

/** Who wrote a user-role message that the human did not type. `agent` is a Hub agent principal (the XERXES supervisor). */
export type HarnessOrigin = 'goal' | 'monitor' | 'schedule' | 'harness' | 'agent'

export function isHarnessOrigin(value: unknown): value is HarnessOrigin {
  return value === 'goal' || value === 'monitor' || value === 'schedule' || value === 'harness' || value === 'agent'
}

export type XerxesActionKind = 'submit' | 'steer' | 'continuation'

/**
 * Provider-omitted authorship of a user message. Written only from a verified
 * Hub identity context (never from request parameters) and never sent to a model.
 */
export interface XerxesActor {
  readonly principal_kind: 'user' | 'agent'
  readonly principal_id: string
  readonly user_id?: string
  readonly device_id?: string
  readonly display_name: string
  readonly action_id: string
  readonly submission_id?: string
  readonly interaction_id?: string
  readonly action_kind: XerxesActionKind | 'permission' | 'question'
  readonly recorded_at: string
  /** Agent principals only: the humans who asked and confirmed. Attribution, never authority. */
  readonly requested_by?: string
  readonly confirmed_by?: string
}

export function isXerxesActor(value: unknown): value is XerxesActor {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  return (record.principal_kind === 'user' || record.principal_kind === 'agent')
    && typeof record.principal_id === 'string' && typeof record.display_name === 'string'
    && typeof record.action_id === 'string' && typeof record.action_kind === 'string'
    && typeof record.recorded_at === 'string'
}

/**
 * Provider-omitted record of who decided one permission or question, stored on the
 * tool message the decision unblocked. Free-text answers are never copied here.
 */
export interface XerxesInteractionRecord {
  readonly interaction_id: string
  readonly kind: 'permission' | 'question'
  readonly decision: string
  readonly decided_by: XerxesActor
  readonly revision: number
  readonly recorded_at: string
  readonly tool_name?: string
  readonly tool_call_id?: string
}

/** Transcript-annotation fields that must never reach a provider request. */
export const PROVIDER_OMITTED_MESSAGE_FIELDS = ['xerxes_actor', 'xerxes_interaction'] as const

/** The same array (and message objects) when nothing needs stripping, so legacy requests stay byte-identical. */
export function withoutXerxesAnnotations<T extends object>(messages: readonly T[]): readonly T[] {
  if (!messages.some(message => PROVIDER_OMITTED_MESSAGE_FIELDS.some(field => field in message))) return messages
  return messages.map(message => {
    if (!PROVIDER_OMITTED_MESSAGE_FIELDS.some(field => field in message)) return message
    const copy = { ...message } as Record<string, unknown>
    for (const field of PROVIDER_OMITTED_MESSAGE_FIELDS) delete copy[field]
    return copy as unknown as T
  })
}

export interface UserMessage {
  readonly role: 'user'
  readonly content: MessageContent
  /** Provider-omitted text used when an attachment-expanded prompt is rendered. */
  readonly displayText?: string
  /**
   * Provider-omitted author of a prompt the human did not type: a goal
   * round, a monitor reaction, a schedule, or a reminder the harness adds
   * mid-turn. Absent means the human typed it. Clients never show these.
   */
  readonly origin?: HarnessOrigin
  /**
   * Provider-omitted mark of a compaction summary (the compressor's
   * COMPACTION_SUMMARY_MARKER). Clients render the summary by it, so every
   * rebuild of a stored message must carry it through.
   */
  readonly xerxes_compaction_summary?: true
  /** Provider-omitted verified author (Hub attribution). Absent for legacy and local-human messages. */
  readonly xerxes_actor?: XerxesActor
  readonly xerxes_interaction?: readonly XerxesInteractionRecord[]
}

export interface AssistantMessage {
  readonly role: 'assistant'
  readonly content: MessageContent
  readonly thinking?: string
  readonly thinking_signature?: string
  readonly tool_calls?: readonly ToolCall[]
}

export interface ToolMessage {
  /**
   * Tool names this result made available (pi-ai `addedToolNames`): set when a
   * tool-search result loaded schemas. Provider adapters with native deferred
   * tool loading use it as the load point for their wire protocol.
   */
  readonly added_tool_names?: readonly string[]
  readonly role: 'tool'
  readonly content: string
  readonly name?: string
  readonly is_error?: boolean
  readonly tool_call_id: string
  /** Provider-omitted record of the human decision that unblocked this call. */
  readonly xerxes_interaction?: readonly XerxesInteractionRecord[]
}

export type ChatMessage = AssistantMessage | SystemMessage | ToolMessage | UserMessage

export interface OpenAiChatMessage {
  readonly role: MessageRole
  readonly content: MessageContent
  readonly name?: string
  readonly tool_call_id?: string
  readonly tool_calls?: readonly (OpenAiCustomToolCall | OpenAiToolCall)[]
  readonly reasoning_content?: string
}

export function messageText(message: ChatMessage): string {
  if (typeof message.content === 'string') {
    return message.content
  }
  return message.content.filter((part): part is TextContentPart => part.type === 'text').map(part => part.text).join('')
}

/** Convert once at the OpenAI-compatible provider boundary. */
export function messageToOpenAi(
  message: ChatMessage,
  grammarProperties: ReadonlyMap<string, string> = new Map(),
): OpenAiChatMessage {
  const base: { content: MessageContent; role: MessageRole } = {
    role: message.role,
    content: message.content,
  }

  if (message.role === 'assistant') {
    return {
      ...base,
      ...(message.tool_calls?.length
        ? {
          tool_calls: message.tool_calls.map((call) => {
            const grammarProperty = grammarProperties.get(call.function.name)
            return grammarProperty === undefined
              ? toolCallToOpenAi(call)
              : toolCallToOpenAiCustom(call, grammarProperty)
          }),
        }
        : {}),
      ...(message.thinking ? { reasoning_content: message.thinking } : {}),
    }
  }
  if (message.role === 'tool') {
    return {
      ...base,
      ...(message.name ? { name: message.name } : {}),
      tool_call_id: message.tool_call_id,
    }
  }
  return base
}

export function messagesToOpenAi(
  messages: readonly ChatMessage[],
  grammarProperties: ReadonlyMap<string, string> = new Map(),
): OpenAiChatMessage[] {
  return messages.map(message => messageToOpenAi(message, grammarProperties))
}
