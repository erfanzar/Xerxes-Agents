// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { CompletionRequest, TokenUsage } from '../llms/client.js'
import type { ChatMessage, ContentPart, MessageContent } from '../types/messages.js'
import {
  isJsonObject,
  parseToolArguments,
  toolCallToOpenAi,
  type JsonObject,
  type ToolCall,
  type ToolChoice,
  type ToolDefinition,
} from '../types/toolCalls.js'

export interface ParsedChatCompletionRequest {
  readonly completion: CompletionRequest
  /** Extension payload preserved for a model-specific backend to interpret. */
  readonly metadata?: unknown
  readonly stream: boolean
}

export interface OpenAiUsage {
  readonly completion_tokens: number
  readonly prompt_tokens: number
  readonly total_tokens: number
}

/** Validation error that is safe to translate into OpenAI's error envelope. */
export class ApiRequestError extends Error {
  constructor(
    message: string,
    readonly parameter: string | null,
  ) {
    super(message)
    this.name = 'ApiRequestError'
  }
}

/** Parse the OpenAI chat-completions subset supported by the portable LLM client. */
export function parseChatCompletionRequest(value: unknown): ParsedChatCompletionRequest {
  const body = record(value, 'body')
  const model = requiredString(body.model, 'model')
  const messagesValue = body.messages
  if (!Array.isArray(messagesValue)) {
    fail('messages must be an array.', 'messages')
  }
  const messages = messagesValue.map((message, index) => parseMessage(message, index))
  const stream = optionalBoolean(body.stream, 'stream') ?? false
  validateChoiceCount(body.n)

  const options: {
    maxTokens?: number
    stop?: readonly string[]
    temperature?: number
    toolChoice?: ToolChoice
    tools?: readonly ToolDefinition[]
    topP?: number
  } = {}
  const maxTokens = optionalInteger(body.max_tokens, 'max_tokens', 1)
  if (maxTokens !== undefined) {
    options.maxTokens = maxTokens
  }
  const temperature = optionalNumber(body.temperature, 'temperature')
  if (temperature !== undefined) {
    options.temperature = temperature
  }
  const topP = optionalNumber(body.top_p, 'top_p')
  if (topP !== undefined) {
    options.topP = topP
  }
  const stop = parseStop(body.stop)
  if (stop !== undefined) {
    options.stop = stop
  }
  const tools = parseTools(body.tools)
  if (tools !== undefined) {
    options.tools = tools
  }
  const toolChoice = parseToolChoice(body.tool_choice)
  if (toolChoice !== undefined) {
    options.toolChoice = toolChoice
  }

  return {
    completion: { model, messages, ...options },
    stream,
    ...(body.metadata === undefined ? {} : { metadata: body.metadata }),
  }
}

export function toOpenAiUsage(usage: TokenUsage | undefined): OpenAiUsage {
  const promptTokens = usage?.inputTokens ?? 0
  const completionTokens = usage?.outputTokens ?? 0
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  }
}

export function toOpenAiToolCalls(toolCalls: readonly ToolCall[]): ReturnType<typeof toolCallToOpenAi>[] {
  return toolCalls.map(toolCallToOpenAi)
}

function parseMessage(value: unknown, index: number): ChatMessage {
  const parameter = `messages[${index}]`
  const message = record(value, parameter)
  const role = requiredString(message.role, `${parameter}.role`)
  if (role === 'system' || role === 'user') {
    return { role, content: parseContent(message.content, `${parameter}.content`, false) }
  }
  if (role === 'assistant') {
    const content = parseContent(message.content, `${parameter}.content`, true)
    const toolCalls = parseToolCalls(message.tool_calls, `${parameter}.tool_calls`)
    return toolCalls.length ? { role, content, tool_calls: toolCalls } : { role, content }
  }
  if (role === 'tool') {
    const content = parseContent(message.content, `${parameter}.content`, false)
    if (typeof content !== 'string') {
      fail('tool message content must be a string.', `${parameter}.content`)
    }
    const toolCallId = requiredString(message.tool_call_id, `${parameter}.tool_call_id`)
    const name = optionalString(message.name, `${parameter}.name`)
    return {
      role,
      content,
      tool_call_id: toolCallId,
      ...(name ? { name } : {}),
    }
  }
  fail('role must be one of system, user, assistant, or tool.', `${parameter}.role`)
}

function parseContent(value: unknown, parameter: string, allowsEmpty: boolean): MessageContent {
  if ((value === undefined || value === null) && allowsEmpty) {
    return ''
  }
  if (typeof value === 'string') {
    return value
  }
  if (!Array.isArray(value)) {
    fail('content must be a string or an array of supported content parts.', parameter)
  }
  return value.map((part, index) => parseContentPart(part, `${parameter}[${index}]`))
}

function parseContentPart(value: unknown, parameter: string): ContentPart {
  const part = record(value, parameter)
  const type = requiredString(part.type, `${parameter}.type`)
  if (type === 'text') {
    return { type, text: requiredString(part.text, `${parameter}.text`) }
  }
  if (type !== 'image_url') {
    fail('only text and image_url content parts are supported.', `${parameter}.type`)
  }
  const imageUrl = record(part.image_url, `${parameter}.image_url`)
  const detail = optionalString(imageUrl.detail, `${parameter}.image_url.detail`)
  if (detail !== undefined && detail !== 'auto' && detail !== 'high' && detail !== 'low') {
    fail('detail must be auto, high, or low.', `${parameter}.image_url.detail`)
  }
  return {
    type,
    image_url: {
      url: requiredString(imageUrl.url, `${parameter}.image_url.url`),
      ...(detail ? { detail } : {}),
    },
  }
}

function parseToolCalls(value: unknown, parameter: string): ToolCall[] {
  if (value === undefined) {
    return []
  }
  if (!Array.isArray(value)) {
    fail('tool_calls must be an array.', parameter)
  }
  return value.map((item, index) => {
    const raw = record(item, `${parameter}[${index}]`)
    const type = raw.type
    if (type !== undefined && type !== 'function') {
      fail('tool call type must be function.', `${parameter}[${index}].type`)
    }
    const functionValue = record(raw.function, `${parameter}[${index}].function`)
    const argumentsValue = functionValue.arguments
    let argumentsObject: JsonObject
    try {
      argumentsObject = isJsonObject(argumentsValue)
        ? argumentsValue
        : parseToolArguments(requiredString(argumentsValue, `${parameter}[${index}].function.arguments`))
    } catch {
      fail(
        'tool call arguments must be a JSON object or JSON-encoded object.',
        `${parameter}[${index}].function.arguments`,
      )
    }
    return {
      id: requiredString(raw.id, `${parameter}[${index}].id`),
      type: 'function',
      function: {
        name: requiredString(functionValue.name, `${parameter}[${index}].function.name`),
        arguments: argumentsObject,
      },
    }
  })
}

function parseTools(value: unknown): ToolDefinition[] | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (!Array.isArray(value)) {
    fail('tools must be an array.', 'tools')
  }
  return value.map((item, index) => {
    const parameter = `tools[${index}]`
    const tool = record(item, parameter)
    if (tool.type !== 'function') {
      fail('tool type must be function.', `${parameter}.type`)
    }
    const functionValue = record(tool.function, `${parameter}.function`)
    const description = functionValue.description
    if (description !== undefined && description !== null && typeof description !== 'string') {
      fail('description must be a string.', `${parameter}.function.description`)
    }
    const parameters = functionValue.parameters ?? {}
    if (!isRecord(parameters)) {
      fail('parameters must be a JSON object.', `${parameter}.function.parameters`)
    }
    return {
      type: 'function',
      function: {
        name: requiredString(functionValue.name, `${parameter}.function.name`),
        description: typeof description === 'string' ? description : '',
        parameters,
      },
    }
  })
}

function parseToolChoice(value: unknown): ToolChoice | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (value === 'auto' || value === 'none') {
    return value
  }
  if (value === 'required') {
    return 'any'
  }
  fail('tool_choice must be auto, none, or required.', 'tool_choice')
}

function parseStop(value: unknown): readonly string[] | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (typeof value === 'string') {
    return [value]
  }
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string')) {
    fail('stop must be a string or an array of strings.', 'stop')
  }
  return value
}

function validateChoiceCount(value: unknown): void {
  if (value === undefined || value === null || value === 1) {
    return
  }
  fail('only n=1 is supported.', 'n')
}

function optionalBoolean(value: unknown, parameter: string): boolean | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (typeof value !== 'boolean') {
    fail('must be a boolean.', parameter)
  }
  return value
}

function optionalInteger(value: unknown, parameter: string, minimum: number): number | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum) {
    fail(`must be an integer greater than or equal to ${minimum}.`, parameter)
  }
  return value
}

function optionalNumber(value: unknown, parameter: string): number | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    fail('must be a finite number.', parameter)
  }
  return value
}

function optionalString(value: unknown, parameter: string): string | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (typeof value !== 'string') {
    fail('must be a string.', parameter)
  }
  return value
}

function requiredString(value: unknown, parameter: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    fail('must be a non-empty string.', parameter)
  }
  return value
}

function record(value: unknown, parameter: string): Record<string, unknown> {
  if (!isRecord(value)) {
    fail('must be an object.', parameter)
  }
  return value
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function fail(message: string, parameter: string): never {
  throw new ApiRequestError(message, parameter)
}
