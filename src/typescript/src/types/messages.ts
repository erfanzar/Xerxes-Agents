// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { OpenAiToolCall, ToolCall } from './toolCalls.js'
import { toolCallToOpenAi } from './toolCalls.js'

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

export interface UserMessage {
  readonly role: 'user'
  readonly content: MessageContent
  /** Provider-omitted text used when an attachment-expanded prompt is rendered. */
  readonly displayText?: string
}

export interface AssistantMessage {
  readonly role: 'assistant'
  readonly content: MessageContent
  readonly thinking?: string
  readonly thinking_signature?: string
  readonly tool_calls?: readonly ToolCall[]
}

export interface ToolMessage {
  readonly role: 'tool'
  readonly content: string
  readonly name?: string
  readonly is_error?: boolean
  readonly tool_call_id: string
}

export type ChatMessage = AssistantMessage | SystemMessage | ToolMessage | UserMessage

export interface OpenAiChatMessage {
  readonly role: MessageRole
  readonly content: MessageContent
  readonly name?: string
  readonly tool_call_id?: string
  readonly tool_calls?: readonly OpenAiToolCall[]
  readonly reasoning_content?: string
}

export function messageText(message: ChatMessage): string {
  if (typeof message.content === 'string') {
    return message.content
  }
  return message.content.filter((part): part is TextContentPart => part.type === 'text').map(part => part.text).join('')
}

/** Convert once at the OpenAI-compatible provider boundary. */
export function messageToOpenAi(message: ChatMessage): OpenAiChatMessage {
  const base: { content: MessageContent; role: MessageRole } = {
    role: message.role,
    content: message.content,
  }

  if (message.role === 'assistant') {
    return {
      ...base,
      ...(message.tool_calls?.length ? { tool_calls: message.tool_calls.map(toolCallToOpenAi) } : {}),
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

export function messagesToOpenAi(messages: readonly ChatMessage[]): OpenAiChatMessage[] {
  return messages.map(messageToOpenAi)
}
