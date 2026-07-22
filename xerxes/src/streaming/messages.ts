// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import type { ChatMessage, ContentPart, MessageContent } from '../types/messages.js'
import {
  isJsonObject,
  toolCallToOpenAi,
  type JsonObject,
  type OpenAiToolCall,
  type ToolCall,
} from '../types/toolCalls.js'

const DATA_IMAGE_URL_PATTERN = /^data:(image\/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)$/i

/** The provider-neutral conversation shape used by the Bun streaming loop. */
export type NeutralMessage = ChatMessage

/** OpenAI chat-completions message shape after neutral-message conversion. */
export interface OpenAiWireMessage {
  readonly content: MessageContent | null
  readonly name?: string
  readonly reasoning_content?: string
  readonly role: 'assistant' | 'system' | 'tool' | 'user'
  readonly tool_call_id?: string
  readonly tool_calls?: readonly OpenAiToolCall[]
}

/** OpenAI tool-call shape accepted at the provider-to-neutral boundary. */
export interface OpenAiMessageInputToolCall {
  readonly function: {
    readonly arguments?: unknown
    readonly name: string
  }
  readonly id: string
  readonly type: 'function'
}

/** OpenAI message shape accepted at the provider-to-neutral boundary. */
export interface OpenAiMessageInput {
  readonly content?: MessageContent | null
  readonly name?: string
  readonly reasoning_content?: string | null
  readonly role: 'assistant' | 'system' | 'tool' | 'user'
  readonly tool_call_id?: string
  readonly tool_calls?: readonly OpenAiMessageInputToolCall[]
}

/** Select whether decoded OpenAI system messages retain their neutral system role. */
export interface OpenAiMessageDecodeOptions {
  /** Python-compatible decoding maps system messages to user messages by default. */
  readonly preserveSystemRole?: boolean
}

export interface AnthropicTextBlock {
  readonly text: string
  readonly type: 'text'
}

export interface AnthropicImageBlock {
  readonly source: {
    readonly data: string
    readonly media_type: string
    readonly type: 'base64'
  }
  readonly type: 'image'
}

export interface AnthropicThinkingBlock {
  readonly signature: string
  readonly thinking: string
  readonly type: 'thinking'
}

export interface AnthropicToolUseBlock {
  readonly id: string
  readonly input: JsonObject
  readonly name: string
  readonly type: 'tool_use'
}

export interface AnthropicToolResultBlock {
  readonly content: string | readonly AnthropicUserContentBlock[]
  readonly is_error?: boolean
  readonly name?: string
  readonly tool_use_id: string
  readonly type: 'tool_result'
}

export type AnthropicUserContentBlock = AnthropicImageBlock | AnthropicTextBlock
export type AnthropicWireContentBlock =
  | AnthropicImageBlock
  | AnthropicTextBlock
  | AnthropicThinkingBlock
  | AnthropicToolResultBlock
  | AnthropicToolUseBlock

/** Anthropic Messages-API conversation message after neutral-message conversion. */
export interface AnthropicWireMessage {
  readonly content: string | readonly AnthropicWireContentBlock[]
  readonly role: 'assistant' | 'user'
}

/** Provider payload with system text separated from Anthropic conversation messages. */
export interface AnthropicWirePayload {
  readonly messages: readonly AnthropicWireMessage[]
  readonly system?: string
}

/**
 * Convert neutral messages to canonical OpenAI chat-completions messages.
 *
 * Args:
 *   messages: Provider-neutral streaming-loop history.
 *   system: Optional system prompt prepended before the neutral history.
 *
 * Returns:
 *   OpenAI wire messages with JSON-encoded function arguments.
 */
export function messagesToOpenAi(messages: readonly NeutralMessage[], system?: string): OpenAiWireMessage[] {
  const result: OpenAiWireMessage[] = []
  if (system) {
    result.push({ role: 'system', content: system })
  }

  for (const message of messages) {
    if (message.role === 'user' || message.role === 'system') {
      result.push({ role: message.role, content: copyContent(message.content) })
      continue
    }
    if (message.role === 'assistant') {
      result.push({
        role: 'assistant',
        content: openAiAssistantContent(message.content),
        ...(message.tool_calls?.length ? { tool_calls: message.tool_calls.map(toolCallToOpenAi) } : {}),
        ...(message.thinking ? { reasoning_content: message.thinking } : {}),
      })
      continue
    }
    result.push({
      role: 'tool',
      content: message.content,
      tool_call_id: message.tool_call_id,
      ...(message.name === undefined ? {} : { name: message.name }),
    })
  }
  return result
}

/**
 * Convert OpenAI chat-completions messages back to neutral streaming messages.
 *
 * Tool-call arguments accept the provider's usual JSON string, a pre-parsed
 * object, or an omitted/null value. System messages map to user messages by
 * default to match the Python streaming converter, while Bun callers that
 * retain system messages in their state can opt in to preserving that role.
 */
export function messagesFromOpenAi(
  messages: readonly OpenAiMessageInput[],
  options: OpenAiMessageDecodeOptions = {},
): ChatMessage[] {
  const result: ChatMessage[] = []
  const preserveSystemRole = options.preserveSystemRole ?? false
  for (const message of messages) {
    if (message.role === 'system') {
      const content = neutralContent(message.content)
      result.push(preserveSystemRole ? { role: 'system', content } : { role: 'user', content })
      continue
    }
    if (message.role === 'user') {
      result.push({ role: 'user', content: neutralContent(message.content) })
      continue
    }
    if (message.role === 'assistant') {
      const toolCalls = (message.tool_calls ?? []).map(openAiToolCallToNeutral)
      result.push({
        role: 'assistant',
        content: neutralContent(message.content),
        ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
        ...(message.reasoning_content ? { thinking: message.reasoning_content } : {}),
      })
      continue
    }
    if (!message.tool_call_id) {
      throw new ValidationError('tool_call_id', 'must be provided for OpenAI tool messages', message)
    }
    result.push({
      role: 'tool',
      tool_call_id: message.tool_call_id,
      content: contentAsText(message.content),
      ...(message.name === undefined ? {} : { name: message.name }),
    })
  }
  return result
}

/**
 * Normalize an OpenAI function-call arguments field to its canonical object form.
 *
 * Missing, null, blank, and non-string/non-object values normalize to an empty
 * object, as in the Python converter. JSON strings must decode to an object so
 * the result remains usable as a typed Bun tool call.
 */
export function normalizeOpenAiToolArguments(value: unknown): JsonObject {
  if (value === undefined || value === null) {
    return {}
  }
  if (isJsonObject(value)) {
    return value
  }
  if (typeof value !== 'string') {
    return {}
  }
  const text = value.trim()
  if (!text) {
    return {}
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(text) as unknown
  } catch {
    throw new ValidationError('tool_call.arguments', 'must be valid JSON', value)
  }
  if (!isJsonObject(parsed)) {
    throw new ValidationError('tool_call.arguments', 'must decode to a JSON object', value)
  }
  return parsed
}

/**
 * Convert neutral messages to Anthropic content blocks.
 *
 * Consecutive tool results are coalesced into one user message. System messages
 * are excluded because Anthropic transports them in a separate payload field;
 * use {@link messagesToAnthropicPayload} when that field is needed.
 */
export function messagesToAnthropic(messages: readonly NeutralMessage[]): AnthropicWireMessage[] {
  const result: AnthropicWireMessage[] = []
  let index = 0
  while (index < messages.length) {
    const message = messages[index]
    if (!message) {
      break
    }
    if (message.role === 'system') {
      index += 1
      continue
    }
    if (message.role === 'user') {
      result.push({ role: 'user', content: anthropicUserContent(message.content) })
      index += 1
      continue
    }
    if (message.role === 'assistant') {
      const blocks: AnthropicWireContentBlock[] = []
      if (message.thinking && message.thinking_signature) {
        blocks.push({ type: 'thinking', thinking: message.thinking, signature: message.thinking_signature })
      }
      const text = contentAsText(message.content)
      if (text) {
        blocks.push({ type: 'text', text })
      }
      for (const call of message.tool_calls ?? []) {
        blocks.push({ type: 'tool_use', id: call.id, name: call.function.name, input: call.function.arguments })
      }
      result.push({ role: 'assistant', content: blocks })
      index += 1
      continue
    }

    const toolResults: AnthropicToolResultBlock[] = []
    while (index < messages.length) {
      const toolMessage = messages[index]
      if (!toolMessage || toolMessage.role !== 'tool') {
        break
      }
      toolResults.push({
        type: 'tool_result',
        tool_use_id: toolMessage.tool_call_id,
        content: toolMessage.content,
        ...(toolMessage.is_error ? { is_error: true } : {}),
      })
      index += 1
    }
    if (toolResults.length) {
      result.push({ role: 'user', content: toolResults })
    }
  }
  return result
}

/**
 * Build an Anthropic payload while extracting all neutral system messages.
 *
 * This mirrors the live provider's separate Anthropic system field without
 * coupling this normalization module to a client implementation.
 */
export function messagesToAnthropicPayload(messages: readonly NeutralMessage[]): AnthropicWirePayload {
  const system = messages
    .filter((message): message is Extract<NeutralMessage, { role: 'system' }> => message.role === 'system')
    .map(message => contentAsText(message.content))
    .filter(Boolean)
    .join('\n\n')
  return {
    messages: messagesToAnthropic(messages),
    ...(system ? { system } : {}),
  }
}

/**
 * Convert Anthropic content-block messages back to neutral streaming messages.
 *
 * Text blocks become text content, base64 image blocks become OpenAI-compatible
 * data URLs, tool-use blocks become canonical function calls, and tool-result
 * blocks become separate tool messages.
 */
export function messagesFromAnthropic(messages: readonly AnthropicWireMessage[]): ChatMessage[] {
  const result: ChatMessage[] = []
  for (const message of messages) {
    if (typeof message.content === 'string') {
      result.push({ role: message.role, content: message.content })
      continue
    }

    const contentParts: ContentPart[] = []
    const toolCalls: ToolCall[] = []
    const toolResults: ChatMessage[] = []
    let thinking = ''
    let thinkingSignature: string | undefined

    for (const block of message.content) {
      if (block.type === 'text') {
        contentParts.push({ type: 'text', text: block.text })
        continue
      }
      if (block.type === 'image') {
        contentParts.push({ type: 'image_url', image_url: { url: imageDataUrl(block) } })
        continue
      }
      if (block.type === 'thinking') {
        // A message may carry several thinking blocks; keep all reasoning and
        // retain the latest signature for replay.
        thinking = thinking ? `${thinking}\n${block.thinking}` : block.thinking
        thinkingSignature = block.signature || thinkingSignature
        continue
      }
      if (block.type === 'tool_use') {
        toolCalls.push({
          id: block.id,
          type: 'function',
          function: { name: block.name, arguments: block.input },
        })
        continue
      }
      toolResults.push({
        role: 'tool',
        tool_call_id: block.tool_use_id,
        content: anthropicToolResultText(block.content),
        ...(block.name === undefined ? {} : { name: block.name }),
        ...(block.is_error ? { is_error: true } : {}),
      })
    }

    if (toolCalls.length) {
      result.push({
        role: 'assistant',
        content: contentFromParts(contentParts),
        tool_calls: toolCalls,
        ...(thinking ? { thinking } : {}),
        ...(thinking && thinkingSignature ? { thinking_signature: thinkingSignature } : {}),
      })
      continue
    }
    if (toolResults.length) {
      result.push(...toolResults)
      continue
    }
    result.push({ role: message.role, content: contentFromParts(contentParts) })
  }
  return result
}

function openAiToolCallToNeutral(call: OpenAiMessageInputToolCall): ToolCall {
  if (!call.id || !call.function?.name) {
    throw new ValidationError('tool_call', 'must include id and function name', call)
  }
  return {
    id: call.id,
    type: 'function',
    function: {
      name: call.function.name,
      arguments: normalizeOpenAiToolArguments(call.function.arguments),
    },
  }
}

function openAiAssistantContent(content: MessageContent): MessageContent | null {
  if (typeof content === 'string') {
    return content || null
  }
  return content.length ? copyContent(content) : null
}

function neutralContent(content: MessageContent | null | undefined): MessageContent {
  return content === null || content === undefined ? '' : copyContent(content)
}

function copyContent(content: MessageContent): MessageContent {
  if (typeof content === 'string') {
    return content
  }
  return content.map(part => {
    if (part.type === 'text') {
      return { type: 'text', text: part.text }
    }
    return {
      type: 'image_url',
      image_url: {
        url: part.image_url.url,
        ...(part.image_url.detail === undefined ? {} : { detail: part.image_url.detail }),
      },
    }
  })
}

function contentAsText(content: MessageContent | null | undefined): string {
  if (content === null || content === undefined) {
    return ''
  }
  if (typeof content === 'string') {
    return content
  }
  return content.map(contentPartText).join('')
}

function contentPartText(part: ContentPart): string {
  return part.type === 'text' ? part.text : `[Image: ${part.image_url.url}]`
}

function anthropicUserContent(content: MessageContent): string | AnthropicUserContentBlock[] {
  if (typeof content === 'string') {
    return content
  }
  return content.map(part => {
    if (part.type === 'text') {
      return { type: 'text', text: part.text }
    }
    return imageBlockFromUrl(part.image_url.url) ?? { type: 'text', text: contentPartText(part) }
  })
}

function imageBlockFromUrl(url: string): AnthropicImageBlock | undefined {
  const match = DATA_IMAGE_URL_PATTERN.exec(url)
  if (!match?.[1] || !match[2]) {
    return undefined
  }
  return {
    type: 'image',
    source: {
      type: 'base64',
      media_type: match[1].toLowerCase(),
      data: match[2].replace(/\s/g, ''),
    },
  }
}

function imageDataUrl(block: AnthropicImageBlock): string {
  return `data:${block.source.media_type};base64,${block.source.data}`
}

function anthropicToolResultText(content: AnthropicToolResultBlock['content']): string {
  if (typeof content === 'string') {
    return content
  }
  return content.map(block => block.type === 'text' ? block.text : `[Image: ${imageDataUrl(block)}]`).join('\n')
}

function contentFromParts(parts: readonly ContentPart[]): MessageContent {
  if (!parts.length) {
    return ''
  }
  if (parts.every(part => part.type === 'text')) {
    return parts.map(part => part.type === 'text' ? part.text : '').join('\n')
  }
  return copyContent([...parts])
}
