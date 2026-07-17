// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { CompletionRequest, TokenUsage } from '../llms/client.js'
import { DEFAULT_TEMPERATURE, DEFAULT_TOP_K } from '../llms/samplingDefaults.js'
import {
  messageToOpenAi,
  type ChatMessage,
  type ContentPart,
  type MessageContent,
} from './messages.js'
import {
  toolCallFromOpenAi,
  toolCallToOpenAi,
  type JsonSchema,
  type OpenAiToolCall,
  type ToolCall,
  type ToolChoice,
  type ToolDefinition,
} from './toolCalls.js'

/** Largest request `max_tokens` value accepted by the portable OpenAI surface. */
export const MAX_OPENAI_TOKENS = 1_000_000

const CHAT_COMPLETION_FIELDS = new Set([
  'model',
  'messages',
  'max_tokens',
  'presence_penalty',
  'frequency_penalty',
  'repetition_penalty',
  'temperature',
  'top_p',
  'top_k',
  'min_p',
  'suppress_tokens',
  'functions',
  'function_call',
  'tools',
  'tool_choice',
  'n',
  'stream',
  'stop',
  'logit_bias',
  'user',
  'chat_template_kwargs',
])

const COMPLETION_FIELDS = new Set([
  'model',
  'prompt',
  'max_tokens',
  'presence_penalty',
  'frequency_penalty',
  'repetition_penalty',
  'temperature',
  'top_p',
  'top_k',
  'min_p',
  'suppress_tokens',
  'n',
  'stream',
  'stop',
  'logit_bias',
  'user',
])

const MESSAGE_FIELDS = new Set(['role', 'content', 'name', 'tool_call_id', 'tool_calls', 'function_call'])

/** Explicit boundary failure for OpenAI-compatible payloads. */
export class OpenAiProtocolValidationError extends Error {
  constructor(
    readonly field: string,
    message: string,
    readonly value: unknown,
  ) {
    super(`${field}: ${message}`)
    this.name = 'OpenAiProtocolValidationError'
  }
}

/** Wire-level content parts retain native text/image parts while allowing provider extensions. */
export type OpenAiContentPart = ContentPart | Readonly<Record<string, unknown>>

export type OpenAiMessageContent = string | readonly OpenAiContentPart[] | null
export type OpenAiMessageRole = 'assistant' | 'function' | 'system' | 'tool' | 'user'

/** Legacy function-call payload used before OpenAI's `tool_calls` array. */
export interface OpenAiLegacyFunctionCall {
  readonly arguments?: string
  readonly name?: string
}

/** OpenAI chat message before conversion into Xerxes's canonical message types. */
export interface OpenAiProtocolMessage {
  readonly content?: OpenAiMessageContent
  readonly extensions?: Readonly<Record<string, unknown>>
  readonly function_call?: OpenAiLegacyFunctionCall
  readonly name?: string
  readonly role: OpenAiMessageRole
  readonly tool_call_id?: string
  readonly tool_calls?: readonly OpenAiToolCall[]
}

export interface OpenAiFunctionCallDelta {
  readonly arguments?: string
  readonly name?: string
}

/** One indexed partial tool call in a streamed chat-completion delta. */
export interface OpenAiToolCallDelta {
  readonly function?: OpenAiFunctionCallDelta
  readonly id?: string
  readonly index: number
  readonly type?: 'function'
}

export interface OpenAiMessageDelta {
  readonly content?: string | readonly OpenAiContentPart[]
  readonly function_call?: OpenAiLegacyFunctionCall
  readonly role?: OpenAiMessageRole
  readonly tool_calls?: readonly OpenAiToolCallDelta[]
}

/** OpenAI wire definition. Canonical runtime tools require a normalized description. */
export interface OpenAiFunctionDefinition {
  readonly description?: string
  readonly name: string
  readonly parameters: JsonSchema
}

export interface OpenAiToolDefinition {
  readonly function: OpenAiFunctionDefinition
  readonly type: 'function'
}

export type OpenAiFunctionSelection = string | Readonly<Record<string, unknown>>

export interface OpenAiUsageInfo {
  readonly completion_tokens: number | null
  readonly processing_time: number
  readonly prompt_tokens: number
  readonly tokens_per_second: number
  readonly total_tokens: number
}

/** Fully normalized request body for `POST /v1/chat/completions`. */
export interface OpenAiChatCompletionRequest {
  readonly chat_template_kwargs?: Readonly<Record<string, boolean | number | string>>
  readonly extensions: Readonly<Record<string, unknown>>
  readonly frequency_penalty: number
  readonly function_call?: OpenAiFunctionSelection
  readonly functions?: readonly OpenAiFunctionDefinition[]
  readonly logit_bias?: Readonly<Record<string, number>>
  readonly max_tokens: number
  readonly messages: readonly OpenAiProtocolMessage[]
  readonly min_p: number
  readonly model: string
  readonly n: number
  readonly presence_penalty: number
  readonly repetition_penalty: number
  readonly stop?: readonly string[]
  readonly stream: boolean
  readonly suppress_tokens: readonly number[]
  readonly temperature: number
  readonly tool_choice?: OpenAiFunctionSelection
  readonly tools?: readonly OpenAiToolDefinition[]
  readonly top_k: number
  readonly top_p: number
  readonly user?: string
}

/** Fully normalized request body for `POST /v1/completions`. */
export interface OpenAiCompletionRequest {
  readonly extensions: Readonly<Record<string, unknown>>
  readonly frequency_penalty: number
  readonly logit_bias?: Readonly<Record<string, number>>
  readonly max_tokens: number
  readonly min_p: number
  readonly model: string
  readonly n: number
  readonly presence_penalty: number
  readonly prompt: readonly string[] | string
  readonly repetition_penalty: number
  readonly stop?: readonly string[]
  readonly stream: boolean
  readonly suppress_tokens: readonly number[]
  readonly temperature: number
  readonly top_k: number
  readonly top_p: number
  readonly user?: string
}

export interface OpenAiCountTokenRequest {
  readonly conversation: readonly OpenAiProtocolMessage[] | string
  readonly model: string
}

export type OpenAiChatFinishReason = 'abort' | 'function_call' | 'length' | 'stop' | 'tool_calls' | null
export type OpenAiCompletionFinishReason = 'function_call' | 'length' | 'stop' | null

export interface OpenAiChatCompletionChoice {
  readonly finish_reason: OpenAiChatFinishReason
  readonly index: number
  readonly message: OpenAiProtocolMessage
}

export interface OpenAiChatCompletionStreamChoice {
  readonly delta: OpenAiMessageDelta
  readonly finish_reason: OpenAiChatFinishReason
  readonly index: number
}

export interface OpenAiChatCompletionResponse {
  readonly choices: readonly OpenAiChatCompletionChoice[]
  readonly created: number
  readonly id: string
  readonly model: string
  readonly object: 'chat.completion'
  readonly usage: OpenAiUsageInfo
}

export interface OpenAiChatCompletionStreamResponse {
  readonly choices: readonly OpenAiChatCompletionStreamChoice[]
  readonly created: number
  readonly id: string
  readonly model: string
  readonly object: 'chat.completion.chunk'
  readonly usage: OpenAiUsageInfo
}

export interface OpenAiCompletionLogprobs {
  readonly text_offset?: readonly number[]
  readonly token_logprobs: readonly number[]
  readonly tokens: readonly string[]
  readonly top_logprobs?: readonly Readonly<Record<string, number>>[]
}

export interface OpenAiCompletionChoice {
  readonly finish_reason: OpenAiCompletionFinishReason
  readonly index: number
  readonly logprobs?: OpenAiCompletionLogprobs
  readonly text: string
}

export interface OpenAiCompletionStreamChoice {
  readonly finish_reason: OpenAiCompletionFinishReason
  readonly index: number
  readonly logprobs?: OpenAiCompletionLogprobs
  readonly text: string
}

export interface OpenAiCompletionResponse {
  readonly choices: readonly OpenAiCompletionChoice[]
  readonly created: number
  readonly id: string
  readonly model: string
  readonly object: 'text_completion'
  readonly usage: OpenAiUsageInfo
}

export interface OpenAiCompletionStreamResponse {
  readonly choices: readonly OpenAiCompletionStreamChoice[]
  readonly created: number
  readonly id: string
  readonly model: string
  readonly object: 'text_completion.chunk'
  readonly usage?: OpenAiUsageInfo
}

export const OpenAiFunctionCallFormat = Object.freeze({
  OPENAI: 'openai',
  JSON_SCHEMA: 'json_schema',
  XML_TAG: 'xml_tag',
  GORILLA: 'gorilla',
  QWEN: 'qwen',
} as const)

export type OpenAiFunctionCallFormat = (typeof OpenAiFunctionCallFormat)[keyof typeof OpenAiFunctionCallFormat]

export interface OpenAiExtractedToolCallInformation {
  readonly content?: string
  readonly tool_calls: readonly OpenAiToolCall[]
  readonly tools_called: boolean
}

export interface OpenAiMessageConversionOptions {
  /** Supplies an id when a legacy `function_call` must become a canonical tool call. */
  readonly legacyToolCallId?: (functionName: string) => string
}

export interface OpenAiUsageConversionOptions {
  readonly processingTime?: number
  readonly tokensPerSecond?: number
}

export interface OpenAiChatCompletionResponseInput {
  readonly choices: readonly OpenAiChatCompletionChoice[]
  readonly created?: number
  readonly id?: string
  readonly model: string
  readonly usage: OpenAiUsageInfo
}

export interface OpenAiChatCompletionStreamResponseInput {
  readonly choices: readonly OpenAiChatCompletionStreamChoice[]
  readonly created?: number
  readonly id?: string
  readonly model: string
  readonly usage: OpenAiUsageInfo
}

export interface OpenAiCompletionResponseInput {
  readonly choices: readonly OpenAiCompletionChoice[]
  readonly created?: number
  readonly id?: string
  readonly model: string
  readonly usage: OpenAiUsageInfo
}

export interface OpenAiCompletionStreamResponseInput {
  readonly choices: readonly OpenAiCompletionStreamChoice[]
  readonly created?: number
  readonly id?: string
  readonly model: string
  readonly usage?: OpenAiUsageInfo
}

/** Validate and normalize an OpenAI-compatible chat-completions request. */
export function parseOpenAiChatCompletionRequest(value: unknown): OpenAiChatCompletionRequest {
  const body = record(value, 'body')
  const functions = optionalArray(body.functions, 'functions', parseOpenAiFunctionDefinition)
  const tools = optionalArray(body.tools, 'tools', parseOpenAiToolDefinition)
  const functionCall = optionalFunctionSelection(body.function_call, 'function_call')
  const toolChoice = optionalFunctionSelection(body.tool_choice, 'tool_choice')
  const stop = optionalStop(body.stop, 'stop')
  const logitBias = optionalLogitBias(body.logit_bias, 'logit_bias')
  const user = optionalText(body.user, 'user')
  const templateArguments = optionalTemplateArguments(body.chat_template_kwargs, 'chat_template_kwargs')

  return {
    model: requiredText(body.model, 'model'),
    messages: requiredArray(body.messages, 'messages', parseOpenAiProtocolMessage),
    max_tokens: integerInRange(body.max_tokens, 'max_tokens', 128, 1, MAX_OPENAI_TOKENS),
    presence_penalty: finiteNumber(body.presence_penalty, 'presence_penalty', 0),
    frequency_penalty: finiteNumber(body.frequency_penalty, 'frequency_penalty', 0),
    repetition_penalty: finiteNumber(body.repetition_penalty, 'repetition_penalty', 1),
    temperature: finiteNumber(body.temperature, 'temperature', DEFAULT_TEMPERATURE),
    top_p: finiteNumber(body.top_p, 'top_p', 0.95),
    top_k: integerInRange(body.top_k, 'top_k', DEFAULT_TOP_K, 0),
    min_p: finiteNumber(body.min_p, 'min_p', 0),
    suppress_tokens: integerArray(body.suppress_tokens, 'suppress_tokens'),
    n: integerInRange(body.n, 'n', 1, 1),
    stream: booleanValue(body.stream, 'stream', false),
    extensions: extensions(body, CHAT_COMPLETION_FIELDS),
    ...(functions === undefined ? {} : { functions }),
    ...(functionCall === undefined ? {} : { function_call: functionCall }),
    ...(tools === undefined ? {} : { tools }),
    ...(toolChoice === undefined ? {} : { tool_choice: toolChoice }),
    ...(stop === undefined ? {} : { stop }),
    ...(logitBias === undefined ? {} : { logit_bias: logitBias }),
    ...(user === undefined ? {} : { user }),
    ...(templateArguments === undefined ? {} : { chat_template_kwargs: templateArguments }),
  }
}

/** Validate and normalize an OpenAI-compatible legacy text-completions request. */
export function parseOpenAiCompletionRequest(value: unknown): OpenAiCompletionRequest {
  const body = record(value, 'body')
  const prompt = parsePrompt(body.prompt, 'prompt')
  const stop = optionalStop(body.stop, 'stop')
  const logitBias = optionalLogitBias(body.logit_bias, 'logit_bias')
  const user = optionalText(body.user, 'user')

  return {
    model: requiredText(body.model, 'model'),
    prompt,
    max_tokens: integerInRange(body.max_tokens, 'max_tokens', 128, 1, MAX_OPENAI_TOKENS),
    presence_penalty: finiteNumber(body.presence_penalty, 'presence_penalty', 0),
    frequency_penalty: finiteNumber(body.frequency_penalty, 'frequency_penalty', 0),
    repetition_penalty: finiteNumber(body.repetition_penalty, 'repetition_penalty', 1),
    temperature: finiteNumber(body.temperature, 'temperature', DEFAULT_TEMPERATURE),
    top_p: finiteNumber(body.top_p, 'top_p', 0.95),
    top_k: integerInRange(body.top_k, 'top_k', DEFAULT_TOP_K, 0),
    min_p: finiteNumber(body.min_p, 'min_p', 0),
    suppress_tokens: integerArray(body.suppress_tokens, 'suppress_tokens'),
    n: integerInRange(body.n, 'n', 1, 1),
    stream: booleanValue(body.stream, 'stream', false),
    extensions: extensions(body, COMPLETION_FIELDS),
    ...(stop === undefined ? {} : { stop }),
    ...(logitBias === undefined ? {} : { logit_bias: logitBias }),
    ...(user === undefined ? {} : { user }),
  }
}

/** Validate the small model-and-conversation shape used by token-count endpoints. */
export function parseOpenAiCountTokenRequest(value: unknown): OpenAiCountTokenRequest {
  const body = record(value, 'body')
  const conversationValue = body.conversation
  if (typeof conversationValue === 'string') {
    return { model: requiredText(body.model, 'model'), conversation: conversationValue }
  }
  return {
    model: requiredText(body.model, 'model'),
    conversation: requiredArray(conversationValue, 'conversation', parseOpenAiProtocolMessage),
  }
}

/** Parse one OpenAI-compatible message without collapsing it into the neutral runtime representation. */
export function parseOpenAiProtocolMessage(value: unknown): OpenAiProtocolMessage {
  const message = record(value, 'message')
  const role = requiredText(message.role, 'message.role')
  if (!isOpenAiMessageRole(role)) {
    throw validation('message.role', 'must be one of assistant, function, system, tool, or user', role)
  }
  const content = parseMessageContent(message.content, 'message.content')
  const name = optionalText(message.name, 'message.name')
  const toolCallId = optionalText(message.tool_call_id, 'message.tool_call_id')
  const functionCall = optionalLegacyFunctionCall(message.function_call, 'message.function_call')
  const toolCalls = optionalArray(message.tool_calls, 'message.tool_calls', parseOpenAiToolCall)
  const extra = extensions(message, MESSAGE_FIELDS)

  return {
    role,
    ...(content === null ? {} : { content }),
    ...(name === undefined ? {} : { name }),
    ...(toolCallId === undefined ? {} : { tool_call_id: toolCallId }),
    ...(functionCall === undefined ? {} : { function_call: functionCall }),
    ...(toolCalls === undefined ? {} : { tool_calls: toolCalls }),
    ...(Object.keys(extra).length === 0 ? {} : { extensions: extra }),
  }
}

/** Convert an OpenAI wire message to the existing neutral `ChatMessage` vocabulary. */
export function chatMessageFromOpenAi(
  message: OpenAiProtocolMessage,
  options: OpenAiMessageConversionOptions = {},
): ChatMessage {
  const content = message.content ?? null
  if (message.role === 'system' || message.role === 'user') {
    return { role: message.role, content: canonicalContent(content, 'message.content', false) }
  }
  if (message.role === 'tool') {
    const toolCallId = requiredText(message.tool_call_id, 'message.tool_call_id')
    if (typeof content !== 'string') {
      throw validation('message.content', 'tool messages require string content', content)
    }
    return {
      role: 'tool',
      content,
      tool_call_id: toolCallId,
      ...(message.name === undefined ? {} : { name: message.name }),
    }
  }
  if (message.role === 'function') {
    throw validation(
      'message.role',
      'legacy function result messages require an explicit tool_call_id before they can enter the neutral runtime',
      message,
    )
  }

  const toolCalls = (message.tool_calls ?? []).map(toolCallFromOpenAi)
  const legacyCall = message.function_call
  if (legacyCall !== undefined) {
    const functionName = requiredText(legacyCall.name, 'message.function_call.name')
    const id = requiredText(
      options.legacyToolCallId?.(functionName) ?? `call_legacy_${crypto.randomUUID().replaceAll('-', '')}`,
      'legacyToolCallId',
    )
    toolCalls.push(toolCallFromOpenAi({
      id,
      type: 'function',
      function: { name: functionName, arguments: legacyCall.arguments ?? '' },
    }))
  }
  return {
    role: 'assistant',
    content: canonicalContent(content, 'message.content', true),
    ...(toolCalls.length === 0 ? {} : { tool_calls: toolCalls }),
  }
}

/** Convert a canonical Xerxes message to its OpenAI wire counterpart. */
export function chatMessageToOpenAi(message: ChatMessage): OpenAiProtocolMessage {
  const openAi = messageToOpenAi(message)
  return {
    role: openAi.role,
    content: openAi.content,
    ...(openAi.name === undefined ? {} : { name: openAi.name }),
    ...(openAi.tool_call_id === undefined ? {} : { tool_call_id: openAi.tool_call_id }),
    ...(openAi.tool_calls === undefined ? {} : { tool_calls: openAi.tool_calls }),
  }
}

export function chatMessagesFromOpenAi(
  messages: readonly OpenAiProtocolMessage[],
  options: OpenAiMessageConversionOptions = {},
): ChatMessage[] {
  return messages.map(message => chatMessageFromOpenAi(message, options))
}

export function chatMessagesToOpenAi(messages: readonly ChatMessage[]): OpenAiProtocolMessage[] {
  return messages.map(chatMessageToOpenAi)
}

/** Convert an OpenAI tool definition to the canonical definition consumed by LLM adapters. */
export function toolDefinitionFromOpenAi(value: OpenAiToolDefinition): ToolDefinition {
  return {
    type: 'function',
    function: {
      name: value.function.name,
      description: value.function.description ?? '',
      parameters: value.function.parameters,
    },
  }
}

/** Convert a canonical tool definition to OpenAI's optional-description wire form. */
export function toolDefinitionToOpenAi(value: ToolDefinition): OpenAiToolDefinition {
  return {
    type: 'function',
    function: {
      name: value.function.name,
      ...(value.function.description ? { description: value.function.description } : {}),
      parameters: value.function.parameters,
    },
  }
}

/** Delegate tool-call conversion to the existing canonical `toolCalls.ts` helpers. */
export function toolCallFromOpenAiProtocol(value: OpenAiToolCall): ToolCall {
  return toolCallFromOpenAi(value)
}

/** Delegate tool-call serialization to the existing canonical `toolCalls.ts` helpers. */
export function toolCallToOpenAiProtocol(value: ToolCall): OpenAiToolCall {
  return toolCallToOpenAi(value)
}

/**
 * Project a normalized OpenAI chat request onto the portable LLM client.
 *
 * The native client intentionally has no representation for a named forced
 * function. Callers that require it should retain the parsed wire request and
 * apply that policy outside the generic provider adapter.
 */
export function completionRequestFromOpenAi(
  request: OpenAiChatCompletionRequest,
  options: OpenAiMessageConversionOptions = {},
): CompletionRequest {
  const modernTools = request.tools?.map(toolDefinitionFromOpenAi)
  const legacyTools = request.functions?.map(functionValue => toolDefinitionFromOpenAi({
    type: 'function',
    function: functionValue,
  }))
  const tools = modernTools ?? legacyTools
  const choice = nativeToolChoice(request.tool_choice ?? request.function_call)

  return {
    model: request.model,
    messages: chatMessagesFromOpenAi(request.messages, options),
    maxTokens: request.max_tokens,
    temperature: request.temperature,
    topK: request.top_k,
    topP: request.top_p,
    ...(request.stop === undefined ? {} : { stop: request.stop }),
    ...(tools === undefined ? {} : { tools }),
    ...(choice === undefined ? {} : { toolChoice: choice }),
  }
}

/** Convert portable token accounting to the complete OpenAI usage wire shape. */
export function usageInfoFromTokenUsage(
  usage: TokenUsage | undefined,
  options: OpenAiUsageConversionOptions = {},
): OpenAiUsageInfo {
  const promptTokens = usage?.inputTokens ?? 0
  const completionTokens = usage?.outputTokens ?? 0
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
    tokens_per_second: finiteOption(options.tokensPerSecond, 'tokensPerSecond', 0),
    processing_time: finiteOption(options.processingTime, 'processingTime', 0),
  }
}

/** Build a complete non-streaming chat-completions response with safe protocol defaults. */
export function createOpenAiChatCompletionResponse(
  input: OpenAiChatCompletionResponseInput,
): OpenAiChatCompletionResponse {
  return {
    id: responseId(input.id, 'chat-'),
    object: 'chat.completion',
    created: responseCreated(input.created),
    model: requiredText(input.model, 'model'),
    choices: [...input.choices],
    usage: input.usage,
  }
}

/** Build one streaming chat-completion response frame. */
export function createOpenAiChatCompletionStreamResponse(
  input: OpenAiChatCompletionStreamResponseInput,
): OpenAiChatCompletionStreamResponse {
  return {
    id: responseId(input.id, 'chat-'),
    object: 'chat.completion.chunk',
    created: responseCreated(input.created),
    model: requiredText(input.model, 'model'),
    choices: [...input.choices],
    usage: input.usage,
  }
}

/** Build a complete non-streaming legacy text-completions response. */
export function createOpenAiCompletionResponse(input: OpenAiCompletionResponseInput): OpenAiCompletionResponse {
  return {
    id: responseId(input.id, 'cmpl-'),
    object: 'text_completion',
    created: responseCreated(input.created),
    model: requiredText(input.model, 'model'),
    choices: [...input.choices],
    usage: input.usage,
  }
}

/** Build one streaming legacy text-completion response frame. */
export function createOpenAiCompletionStreamResponse(
  input: OpenAiCompletionStreamResponseInput,
): OpenAiCompletionStreamResponse {
  return {
    id: responseId(input.id, 'cmpl-'),
    object: 'text_completion.chunk',
    created: responseCreated(input.created),
    model: requiredText(input.model, 'model'),
    choices: [...input.choices],
    ...(input.usage === undefined ? {} : { usage: input.usage }),
  }
}

function parseOpenAiFunctionDefinition(value: unknown): OpenAiFunctionDefinition {
  const functionValue = record(value, 'function')
  const description = optionalText(functionValue.description, 'function.description')
  const parameters = functionValue.parameters ?? {}
  if (!isRecord(parameters)) {
    throw validation('function.parameters', 'must be an object', parameters)
  }
  return {
    name: requiredText(functionValue.name, 'function.name'),
    parameters: { ...parameters },
    ...(description === undefined ? {} : { description }),
  }
}

function parseOpenAiToolDefinition(value: unknown): OpenAiToolDefinition {
  const tool = record(value, 'tool')
  if (tool.type !== 'function') {
    throw validation('tool.type', 'must equal "function"', tool.type)
  }
  return { type: 'function', function: parseOpenAiFunctionDefinition(tool.function) }
}

function parseOpenAiToolCall(value: unknown): OpenAiToolCall {
  const call = record(value, 'tool_call')
  if (call.type !== undefined && call.type !== 'function') {
    throw validation('tool_call.type', 'must equal "function"', call.type)
  }
  const functionValue = record(call.function, 'tool_call.function')
  return {
    id: requiredText(call.id, 'tool_call.id'),
    type: 'function',
    function: {
      name: requiredText(functionValue.name, 'tool_call.function.name'),
      arguments: serializedToolArguments(functionValue.arguments, 'tool_call.function.arguments'),
    },
  }
}

function parseMessageContent(value: unknown, field: string): OpenAiMessageContent {
  if (value === undefined || value === null) return null
  if (typeof value === 'string') return value
  if (!Array.isArray(value)) {
    throw validation(field, 'must be a string, an array of objects, or null', value)
  }
  return value.map((part, index) => ({ ...record(part, `${field}[${index}]`) }))
}

function optionalLegacyFunctionCall(value: unknown, field: string): OpenAiLegacyFunctionCall | undefined {
  if (value === undefined || value === null) return undefined
  const call = record(value, field)
  const name = optionalText(call.name, `${field}.name`)
  const argumentsValue = call.arguments
  if (argumentsValue !== undefined && typeof argumentsValue !== 'string') {
    throw validation(`${field}.arguments`, 'must be a string', argumentsValue)
  }
  return {
    ...(name === undefined ? {} : { name }),
    ...(argumentsValue === undefined ? {} : { arguments: argumentsValue }),
  }
}

function canonicalContent(
  content: OpenAiMessageContent,
  field: string,
  emptyWhenNull: boolean,
): MessageContent {
  if (content === null) {
    if (emptyWhenNull) return ''
    throw validation(field, 'must be provided', content)
  }
  if (typeof content === 'string') return content
  return content.map((part, index) => canonicalContentPart(part, `${field}[${index}]`))
}

function canonicalContentPart(part: OpenAiContentPart, field: string): ContentPart {
  const rawPart = record(part, field)
  const type = requiredText(rawPart.type, `${field}.type`)
  if (type === 'text') {
    return { type, text: requiredText(rawPart.text, `${field}.text`) }
  }
  if (type !== 'image_url') {
    throw validation(`${field}.type`, 'must equal "text" or "image_url" for the neutral runtime', type)
  }
  const imageUrl = record(rawPart.image_url, `${field}.image_url`)
  const detail = optionalText(imageUrl.detail, `${field}.image_url.detail`)
  if (detail !== undefined && detail !== 'auto' && detail !== 'high' && detail !== 'low') {
    throw validation(`${field}.image_url.detail`, 'must be auto, high, or low', detail)
  }
  return {
    type,
    image_url: {
      url: requiredText(imageUrl.url, `${field}.image_url.url`),
      ...(detail === undefined ? {} : { detail }),
    },
  }
}

function nativeToolChoice(value: OpenAiFunctionSelection | undefined): ToolChoice | undefined {
  if (value === undefined) return undefined
  if (value === 'auto' || value === 'none') return value
  if (value === 'required') return 'any'
  throw validation(
    'tool_choice',
    'named function selection is not representable by the provider-neutral ToolChoice; apply it at the caller boundary',
    value,
  )
}

function parsePrompt(value: unknown, field: string): readonly string[] | string {
  if (typeof value === 'string') return value
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string')) {
    throw validation(field, 'must be a string or an array of strings', value)
  }
  return [...value]
}

function optionalFunctionSelection(value: unknown, field: string): OpenAiFunctionSelection | undefined {
  if (value === undefined || value === null) return undefined
  if (typeof value === 'string') return value
  if (!isRecord(value)) throw validation(field, 'must be a string or an object', value)
  return { ...value }
}

function optionalStop(value: unknown, field: string): readonly string[] | undefined {
  if (value === undefined || value === null) return undefined
  if (typeof value === 'string') return [value]
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string')) {
    throw validation(field, 'must be a string or an array of strings', value)
  }
  return [...value]
}

function optionalLogitBias(value: unknown, field: string): Readonly<Record<string, number>> | undefined {
  if (value === undefined || value === null) return undefined
  const bias = record(value, field)
  const result: Record<string, number> = {}
  for (const [key, entry] of Object.entries(bias)) {
    if (typeof entry !== 'number' || !Number.isFinite(entry)) {
      throw validation(`${field}.${key}`, 'must be a finite number', entry)
    }
    result[key] = entry
  }
  return result
}

function optionalTemplateArguments(
  value: unknown,
  field: string,
): Readonly<Record<string, boolean | number | string>> | undefined {
  if (value === undefined || value === null) return undefined
  const argumentsValue = record(value, field)
  const result: Record<string, boolean | number | string> = {}
  for (const [key, entry] of Object.entries(argumentsValue)) {
    if (typeof entry !== 'boolean' && typeof entry !== 'number' && typeof entry !== 'string') {
      throw validation(`${field}.${key}`, 'must be a boolean, number, or string', entry)
    }
    result[key] = entry
  }
  return result
}

function requiredArray<T>(
  value: unknown,
  field: string,
  parse: (item: unknown) => T,
): T[] {
  if (!Array.isArray(value)) throw validation(field, 'must be an array', value)
  return value.map(parse)
}

function optionalArray<T>(
  value: unknown,
  field: string,
  parse: (item: unknown) => T,
): T[] | undefined {
  if (value === undefined || value === null) return undefined
  return requiredArray(value, field, parse)
}

function integerArray(value: unknown, field: string): number[] {
  if (value === undefined || value === null) return []
  if (!Array.isArray(value) || value.some(item => typeof item !== 'number' || !Number.isInteger(item))) {
    throw validation(field, 'must be an array of integers', value)
  }
  return [...value]
}

function integerInRange(
  value: unknown,
  field: string,
  fallback: number,
  minimum: number,
  maximum = Number.MAX_SAFE_INTEGER,
): number {
  if (value === undefined || value === null) return fallback
  if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum || value > maximum) {
    throw validation(field, `must be an integer between ${minimum} and ${maximum}`, value)
  }
  return value
}

function finiteNumber(value: unknown, field: string, fallback: number): number {
  if (value === undefined || value === null) return fallback
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw validation(field, 'must be a finite number', value)
  }
  return value
}

function finiteOption(value: number | undefined, field: string, fallback: number): number {
  if (value === undefined) return fallback
  if (!Number.isFinite(value)) throw validation(field, 'must be a finite number', value)
  return value
}

function booleanValue(value: unknown, field: string, fallback: boolean): boolean {
  if (value === undefined || value === null) return fallback
  if (typeof value !== 'boolean') throw validation(field, 'must be a boolean', value)
  return value
}

function requiredText(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw validation(field, 'must be a non-empty string', value)
  }
  return value
}

function serializedToolArguments(value: unknown, field: string): string {
  if (typeof value === 'string') return value
  if (!isRecord(value)) throw validation(field, 'must be a JSON string or object', value)
  try {
    return JSON.stringify(value)
  } catch {
    throw validation(field, 'must be JSON serializable', value)
  }
}

function optionalText(value: unknown, field: string): string | undefined {
  if (value === undefined || value === null) return undefined
  if (typeof value !== 'string') throw validation(field, 'must be a string', value)
  return value
}

function responseId(value: string | undefined, prefix: string): string {
  if (value !== undefined) return requiredText(value, 'id')
  return `${prefix}${crypto.randomUUID().replaceAll('-', '')}`
}

function responseCreated(value: number | undefined): number {
  if (value === undefined) return Math.floor(Date.now() / 1_000)
  if (!Number.isInteger(value) || value < 0) throw validation('created', 'must be a non-negative integer', value)
  return value
}

function extensions(recordValue: Readonly<Record<string, unknown>>, fields: ReadonlySet<string>): Readonly<Record<string, unknown>> {
  const result: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(recordValue)) {
    if (!fields.has(key)) result[key] = value
  }
  return result
}

function record(value: unknown, field: string): Readonly<Record<string, unknown>> {
  if (!isRecord(value)) throw validation(field, 'must be an object', value)
  return value
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isOpenAiMessageRole(value: string): value is OpenAiMessageRole {
  return value === 'assistant' || value === 'function' || value === 'system' || value === 'tool' || value === 'user'
}

function validation(field: string, message: string, value: unknown): OpenAiProtocolValidationError {
  return new OpenAiProtocolValidationError(field, message, value)
}
