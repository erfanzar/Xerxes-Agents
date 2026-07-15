// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError, ProviderError } from '../core/errors.js'
import { deterministicToolCallId } from '../streaming/toolCallIds.js'
import type { ChatMessage, ContentPart, MessageContent } from '../types/messages.js'
import type { JsonObject, JsonSchema, ToolCall, ToolDefinition } from '../types/toolCalls.js'
import { isJsonObject } from '../types/toolCalls.js'
import type { CompletionRequest, FetchImplementation, LlmClient, LlmCompletion, LlmDelta, TokenUsage } from './client.js'
import { internalSseData } from './client.js'
import { bareModel, getApiKey } from './providerRegistry.js'

/** Root REST endpoint for Gemini's native Generate Content API. */
export const DEFAULT_GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'

/** One native Gemini safety override supplied with every request. */
export interface GeminiSafetySetting {
  readonly category: string
  readonly threshold: string
}

/** Supported static generation settings beyond the neutral completion contract. */
export interface GeminiGenerationConfig {
  readonly maxOutputTokens?: number
  readonly responseMimeType?: string
  readonly responseSchema?: JsonSchema
  readonly stopSequences?: readonly string[]
  readonly temperature?: number
  readonly topK?: number
  readonly topP?: number
}

export interface GeminiClientOptions {
  /** Gemini API key. Falls back to GEMINI_API_KEY, then GOOGLE_API_KEY. */
  readonly apiKey?: string
  /** Native Generative Language API root, normally ending in `/v1beta`. */
  readonly baseUrl?: string
  readonly fetchImplementation?: FetchImplementation
  /** Static generation settings, overridden by per-turn neutral request values. */
  readonly generationConfig?: GeminiGenerationConfig
  /** Static safety settings for every request made by this client. */
  readonly safetySettings?: readonly GeminiSafetySetting[]
}

export interface GeminiInlineData {
  readonly data: string
  readonly mimeType: string
}

export interface GeminiFunctionCall {
  readonly args?: JsonObject
  readonly id?: string
  readonly name: string
}

export interface GeminiFunctionResponse {
  readonly id?: string
  readonly name: string
  readonly response: JsonObject
}

/** A supported native Gemini content part. */
export type GeminiPart =
  | { readonly inlineData: GeminiInlineData }
  | { readonly functionCall: GeminiFunctionCall; readonly thoughtSignature?: string }
  | { readonly functionResponse: GeminiFunctionResponse }
  | { readonly text: string; readonly thought?: boolean; readonly thoughtSignature?: string }

export interface GeminiContent {
  readonly parts: readonly GeminiPart[]
  readonly role: 'model' | 'user'
}

export interface GeminiMessagePayload {
  readonly contents: readonly GeminiContent[]
  readonly systemInstruction?: {
    readonly parts: readonly GeminiPart[]
  }
}

/**
 * Convert the neutral transcript to Gemini's native Generate Content shape.
 *
 * System messages become the separate `systemInstruction`. Consecutive tool
 * replies become one user content object because Gemini expects its function
 * responses to be sent as a user turn. Remote image URLs remain visible text
 * rather than causing hidden network downloads; data URLs are native
 * `inlineData` parts.
 */
export function messagesToGemini(messages: readonly ChatMessage[]): GeminiMessagePayload {
  const contents: GeminiContent[] = []
  const systemParts: GeminiPart[] = []
  const toolNames = new Map<string, string>()
  let index = 0

  while (index < messages.length) {
    const message = messages[index]
    if (message === undefined) {
      break
    }
    if (message.role === 'system') {
      systemParts.push(...systemContentParts(message.content))
      index += 1
      continue
    }
    if (message.role === 'user') {
      appendContent(contents, 'user', contentToGeminiParts(message.content))
      index += 1
      continue
    }
    if (message.role === 'assistant') {
      const parts = assistantContentParts(message)
      for (const call of message.tool_calls ?? []) {
        toolNames.set(call.id, call.function.name)
        parts.push({
          functionCall: {
            id: call.id,
            name: call.function.name,
            args: call.function.arguments,
          },
        })
      }
      attachAssistantSignature(parts, message.thinking_signature)
      appendContent(contents, 'model', parts)
      index += 1
      continue
    }

    const toolParts: GeminiPart[] = []
    while (index < messages.length) {
      const toolMessage = messages[index]
      if (toolMessage === undefined || toolMessage.role !== 'tool') {
        break
      }
      const name = toolMessage.name ?? toolNames.get(toolMessage.tool_call_id)
      if (!name) {
        throw new ConfigurationError(
          'messages',
          `Gemini tool response ${toolMessage.tool_call_id} is missing its function name`,
        )
      }
      toolParts.push({
        functionResponse: {
          id: toolMessage.tool_call_id,
          name,
          response: toolMessage.is_error ? { error: toolMessage.content } : { result: toolMessage.content },
        },
      })
      index += 1
    }
    appendContent(contents, 'user', toolParts)
  }

  return {
    contents,
    ...(systemParts.length ? { systemInstruction: { parts: systemParts } } : {}),
  }
}

/**
 * Native direct REST client for Gemini's streamed Generate Content endpoint.
 *
 * This adapter deliberately does not go through Gemini's OpenAI compatibility
 * endpoint or an SDK. It emits the shared LlmDelta vocabulary and leaves
 * registry/factory selection to the caller that elects to integrate it.
 */
export class GeminiClient implements LlmClient {
  private readonly apiKey: string
  private readonly baseUrl: string
  private readonly fetchImplementation: FetchImplementation
  private readonly generationConfig: GeminiGenerationConfig
  private readonly safetySettings: readonly GeminiSafetySetting[]

  constructor(options: GeminiClientOptions = {}) {
    const configuredKey = options.apiKey ?? getApiKey('gemini')
    this.apiKey = configuredKey || process.env.GOOGLE_API_KEY || ''
    this.baseUrl = validBaseUrl(options.baseUrl ?? DEFAULT_GEMINI_BASE_URL)
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.generationConfig = copyGenerationConfig(options.generationConfig ?? {})
    this.safetySettings = copySafetySettings(options.safetySettings ?? [])

    if (!this.apiKey.trim()) {
      throw new ConfigurationError('GEMINI_API_KEY', 'Gemini API key not provided')
    }
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    const converted = messagesToGemini(request.messages)
    if (!converted.contents.length) {
      throw new ConfigurationError('messages', 'Gemini requires at least one user, assistant, or tool message')
    }

    let response: Response
    try {
      response = await this.fetchImplementation(geminiCompletionEndpoint(this.baseUrl, request.model), {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
          'x-goog-api-key': this.apiKey,
        },
        body: JSON.stringify(geminiRequestPayload(request, converted, this.generationConfig, this.safetySettings)),
        ...(signal === undefined ? {} : { signal }),
      })
    } catch (error) {
      throw new ProviderError('gemini', 'completion request failed before receiving a response', error)
    }
    if (!response.ok) {
      const body = await response.text()
      const status = response.statusText ? ` ${response.statusText}` : ''
      throw new ProviderError(
        'gemini',
        `completion request failed (${response.status}${status}): ${body.slice(0, 4_096)}`,
      )
    }

    const completion = parseGeminiEvent(await response.text())
    throwGeminiApiError(completion)
    const candidate = geminiCandidates(completion.candidates)[0]
    const usage = geminiUsage(completion.usageMetadata)
    if (candidate === undefined) {
      const promptBlock = promptBlockReason(completion.promptFeedback)
      return {
        content: '',
        toolCalls: [],
        ...(promptBlock === undefined ? {} : { finishReason: normalizeFinishReason(promptBlock) }),
        ...(usage === undefined ? {} : { usage }),
      }
    }

    const parsed = parseGeminiCandidate(candidate)
    return {
      content: parsed.content ?? '',
      toolCalls: parsed.toolCalls,
      ...(parsed.finishReason === undefined ? {} : { finishReason: parsed.finishReason }),
      ...(parsed.thinking === undefined ? {} : { thinking: parsed.thinking }),
      ...(parsed.thinkingSignature === undefined ? {} : { thinkingSignature: parsed.thinkingSignature }),
      ...(usage === undefined ? {} : { usage }),
    }
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const converted = messagesToGemini(request.messages)
    if (!converted.contents.length) {
      throw new ConfigurationError('messages', 'Gemini requires at least one user, assistant, or tool message')
    }

    const endpoint = geminiStreamEndpoint(this.baseUrl, request.model)
    const payload = geminiRequestPayload(request, converted, this.generationConfig, this.safetySettings)
    let response: Response
    try {
      response = await this.fetchImplementation(endpoint, {
        method: 'POST',
        headers: {
          Accept: 'text/event-stream',
          'Content-Type': 'application/json',
          'x-goog-api-key': this.apiKey,
        },
        body: JSON.stringify(payload),
        ...(signal === undefined ? {} : { signal }),
      })
    } catch (error) {
      throw new ProviderError('gemini', 'stream request failed before receiving a response', error)
    }
    if (!response.ok) {
      const body = await response.text()
      const status = response.statusText ? ` ${response.statusText}` : ''
      throw new ProviderError(
        'gemini',
        `stream request failed (${response.status}${status}): ${body.slice(0, 4_096)}`,
      )
    }
    if (!response.body) {
      throw new ProviderError('gemini', 'stream request returned no response body')
    }

    const pendingToolCalls = new Map<string, ToolCall>()
    let emittedToolCalls = false
    for await (const data of internalSseData(response.body)) {
      if (data === '[DONE]') {
        break
      }
      if (!data.trim()) {
        continue
      }
      const chunk = parseGeminiEvent(data)
      throwGeminiApiError(chunk)
      const usage = geminiUsage(chunk.usageMetadata)
      const candidates = geminiCandidates(chunk.candidates)
      const candidate = candidates[0]
      const delta: {
        content?: string
        finishReason?: string
        thinking?: string
        thinkingSignature?: string
        toolCalls?: readonly ToolCall[]
        usage?: TokenUsage
      } = {}

      if (candidate !== undefined) {
        const parsed = parseGeminiCandidate(candidate)
        for (const call of parsed.toolCalls) {
          pendingToolCalls.set(call.id, call)
        }
        if (parsed.content) {
          delta.content = parsed.content
        }
        if (parsed.thinking) {
          delta.thinking = parsed.thinking
        }
        if (parsed.thinkingSignature) {
          delta.thinkingSignature = parsed.thinkingSignature
        }
        if (parsed.finishReason) {
          delta.finishReason = parsed.finishReason
          if (pendingToolCalls.size) {
            delta.toolCalls = [...pendingToolCalls.values()]
            emittedToolCalls = true
          }
        }
      } else {
        const promptBlock = promptBlockReason(chunk.promptFeedback)
        if (promptBlock) {
          delta.finishReason = normalizeFinishReason(promptBlock)
        }
      }
      if (usage) {
        delta.usage = usage
      }
      if (Object.keys(delta).length) {
        yield delta
      }
    }
    if (!emittedToolCalls && pendingToolCalls.size) {
      yield { toolCalls: [...pendingToolCalls.values()] }
    }
  }
}

function appendContent(contents: GeminiContent[], role: GeminiContent['role'], parts: readonly GeminiPart[]): void {
  if (parts.length) {
    contents.push({ role, parts })
  }
}

function assistantContentParts(message: Extract<ChatMessage, { role: 'assistant' }>): GeminiPart[] {
  const parts = contentToGeminiParts(message.content)
  if (message.thinking) {
    parts.unshift({
      text: message.thinking,
      thought: true,
      ...(message.thinking_signature ? { thoughtSignature: message.thinking_signature } : {}),
    })
  }
  return parts
}

function attachAssistantSignature(parts: GeminiPart[], signature: string | undefined): void {
  if (!signature || parts.some(part => 'thoughtSignature' in part && part.thoughtSignature === signature)) {
    return
  }
  const functionCallIndex = parts.findIndex(part => 'functionCall' in part)
  const targetIndex = functionCallIndex >= 0 ? functionCallIndex : 0
  const target = parts[targetIndex]
  if (target === undefined) {
    throw new ConfigurationError('messages', 'Gemini assistant signature has no content part to preserve')
  }
  if ('text' in target) {
    parts[targetIndex] = { ...target, thoughtSignature: signature }
    return
  }
  if ('functionCall' in target) {
    parts[targetIndex] = { ...target, thoughtSignature: signature }
    return
  }
  throw new ConfigurationError('messages', 'Gemini assistant signature cannot be attached to an inline data part')
}

function contentToGeminiParts(content: MessageContent): GeminiPart[] {
  if (typeof content === 'string') {
    return content ? [{ text: content }] : []
  }
  return content.flatMap(part => geminiContentPart(part))
}

function systemContentParts(content: MessageContent): GeminiPart[] {
  if (typeof content === 'string') {
    return content ? [{ text: content }] : []
  }
  return content.map(part => part.type === 'text' ? { text: part.text } : { text: `[Image: ${part.image_url.url}]` })
}

function geminiContentPart(part: ContentPart): GeminiPart[] {
  if (part.type === 'text') {
    return part.text ? [{ text: part.text }] : []
  }
  const inlineData = inlineDataFromUrl(part.image_url.url)
  return inlineData === undefined ? [{ text: `[Image: ${part.image_url.url}]` }] : [{ inlineData }]
}

function inlineDataFromUrl(value: string): GeminiInlineData | undefined {
  const match = /^data:([^;,]+);base64,([\s\S]+)$/i.exec(value)
  if (!match) {
    return undefined
  }
  const mimeType = match[1]
  const data = match[2]?.replace(/\s/g, '')
  if (!mimeType || !data) {
    return undefined
  }
  return { mimeType, data }
}

function geminiRequestPayload(
  request: CompletionRequest,
  converted: GeminiMessagePayload,
  defaults: GeminiGenerationConfig,
  safetySettings: readonly GeminiSafetySetting[],
): Record<string, unknown> {
  const payload: Record<string, unknown> = { contents: converted.contents }
  if (converted.systemInstruction) {
    payload.systemInstruction = converted.systemInstruction
  }
  const generationConfig = requestGenerationConfig(request, defaults)
  if (Object.keys(generationConfig).length) {
    payload.generationConfig = generationConfig
  }
  if (safetySettings.length) {
    payload.safetySettings = safetySettings
  }
  if (request.tools?.length) {
    payload.tools = [{ functionDeclarations: request.tools.map(toolToGemini) }]
    const toolConfig = geminiToolConfig(request.toolChoice)
    if (toolConfig) {
      payload.toolConfig = toolConfig
    }
  }
  return payload
}

function requestGenerationConfig(
  request: CompletionRequest,
  defaults: GeminiGenerationConfig,
): Record<string, unknown> {
  const generationConfig: Record<string, unknown> = {}
  if (defaults.maxOutputTokens !== undefined) generationConfig.maxOutputTokens = defaults.maxOutputTokens
  if (defaults.responseMimeType !== undefined) generationConfig.responseMimeType = defaults.responseMimeType
  if (defaults.responseSchema !== undefined) generationConfig.responseSchema = defaults.responseSchema
  if (defaults.stopSequences !== undefined) generationConfig.stopSequences = [...defaults.stopSequences]
  if (defaults.temperature !== undefined) generationConfig.temperature = defaults.temperature
  if (defaults.topK !== undefined) generationConfig.topK = defaults.topK
  if (defaults.topP !== undefined) generationConfig.topP = defaults.topP

  if (request.maxTokens !== undefined) generationConfig.maxOutputTokens = request.maxTokens
  if (request.temperature !== undefined) generationConfig.temperature = request.temperature
  if (request.topP !== undefined) generationConfig.topP = request.topP
  if (request.stop?.length) generationConfig.stopSequences = [...request.stop]
  return generationConfig
}

function toolToGemini(tool: ToolDefinition): Record<string, unknown> {
  return {
    name: tool.function.name,
    description: tool.function.description,
    parameters: tool.function.parameters,
  }
}

function geminiToolConfig(choice: CompletionRequest['toolChoice']): Record<string, unknown> | undefined {
  if (choice === undefined) {
    return undefined
  }
  const mode = choice === 'any' ? 'ANY' : choice === 'auto' ? 'AUTO' : 'NONE'
  return { functionCallingConfig: { mode } }
}

function geminiStreamEndpoint(baseUrl: string, model: string): URL {
  const name = bareModel(model).replace(/^models\//, '')
  if (!name) {
    throw new ConfigurationError('model', 'Gemini model must not be empty')
  }
  const endpoint = new URL(`models/${encodeURIComponent(name)}:streamGenerateContent`, withTrailingSlash(baseUrl))
  endpoint.searchParams.set('alt', 'sse')
  return endpoint
}

function geminiCompletionEndpoint(baseUrl: string, model: string): URL {
  const name = bareModel(model).replace(/^models\//, '')
  if (!name) {
    throw new ConfigurationError('model', 'Gemini model must not be empty')
  }
  return new URL(`models/${encodeURIComponent(name)}:generateContent`, withTrailingSlash(baseUrl))
}

function validBaseUrl(value: string): string {
  let parsed: URL
  try {
    parsed = new URL(value)
  } catch (error) {
    throw new ConfigurationError('baseUrl', `must be a valid HTTP URL: ${String(error)}`)
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new ConfigurationError('baseUrl', `must use http:// or https://, got ${parsed.protocol}`)
  }
  return parsed.toString()
}

function withTrailingSlash(value: string): string {
  return value.endsWith('/') ? value : `${value}/`
}

function copyGenerationConfig(value: GeminiGenerationConfig): GeminiGenerationConfig {
  if (
    value.maxOutputTokens !== undefined
    && (!Number.isSafeInteger(value.maxOutputTokens) || value.maxOutputTokens <= 0)
  ) {
    throw new ConfigurationError('generationConfig.maxOutputTokens', 'must be a positive integer')
  }
  if (value.topK !== undefined && (!Number.isSafeInteger(value.topK) || value.topK < 0)) {
    throw new ConfigurationError('generationConfig.topK', 'must be a non-negative integer')
  }
  if (value.temperature !== undefined && !Number.isFinite(value.temperature)) {
    throw new ConfigurationError('generationConfig.temperature', 'must be finite')
  }
  if (value.topP !== undefined && (!Number.isFinite(value.topP) || value.topP < 0 || value.topP > 1)) {
    throw new ConfigurationError('generationConfig.topP', 'must be between 0 and 1')
  }
  if (value.stopSequences !== undefined && value.stopSequences.some(stop => !stop)) {
    throw new ConfigurationError('generationConfig.stopSequences', 'must not contain empty values')
  }
  return {
    ...(value.maxOutputTokens === undefined ? {} : { maxOutputTokens: value.maxOutputTokens }),
    ...(value.responseMimeType === undefined ? {} : { responseMimeType: value.responseMimeType }),
    ...(value.responseSchema === undefined ? {} : { responseSchema: value.responseSchema }),
    ...(value.stopSequences === undefined ? {} : { stopSequences: [...value.stopSequences] }),
    ...(value.temperature === undefined ? {} : { temperature: value.temperature }),
    ...(value.topK === undefined ? {} : { topK: value.topK }),
    ...(value.topP === undefined ? {} : { topP: value.topP }),
  }
}

function copySafetySettings(values: readonly GeminiSafetySetting[]): readonly GeminiSafetySetting[] {
  return values.map((value, index) => {
    if (!value.category.trim()) {
      throw new ConfigurationError(`safetySettings.${index}.category`, 'must not be empty')
    }
    if (!value.threshold.trim()) {
      throw new ConfigurationError(`safetySettings.${index}.threshold`, 'must not be empty')
    }
    return { category: value.category, threshold: value.threshold }
  })
}

interface ParsedGeminiCandidate {
  readonly content?: string
  readonly finishReason?: string
  readonly thinking?: string
  readonly thinkingSignature?: string
  readonly toolCalls: readonly ToolCall[]
}

function parseGeminiEvent(data: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(data)
    if (!isRecord(parsed)) {
      throw new Error('a JSON object was expected')
    }
    return parsed
  } catch (error) {
    throw new ProviderError('gemini', `invalid Gemini SSE JSON: ${data.slice(0, 200)}`, error)
  }
}

function throwGeminiApiError(chunk: Record<string, unknown>): void {
  if (chunk.error === undefined) {
    return
  }
  if (!isRecord(chunk.error)) {
    throw new ProviderError('gemini', 'stream returned a malformed API error payload')
  }
  const code = numberValue(chunk.error.code)
  const status = stringValue(chunk.error.status)
  const message = stringValue(chunk.error.message)
  const suffix = [code === undefined ? '' : String(code), status].filter(Boolean).join(' ')
  const description = `stream returned API error${suffix ? ` (${suffix})` : ''}: ${message || 'unknown error'}`
  throw new ProviderError('gemini', description)
}

function geminiCandidates(value: unknown): Record<string, unknown>[] {
  if (value === undefined) {
    return []
  }
  if (!Array.isArray(value)) {
    throw new ProviderError('gemini', 'Gemini SSE candidates must be an array')
  }
  return value.map((candidate, index) => {
    if (!isRecord(candidate)) {
      throw new ProviderError('gemini', `Gemini SSE candidate ${index} must be an object`)
    }
    return candidate
  })
}

function parseGeminiCandidate(candidate: Record<string, unknown>): ParsedGeminiCandidate {
  const textParts: string[] = []
  const thinkingParts: string[] = []
  const toolCalls: ToolCall[] = []
  let thinkingSignature: string | undefined
  if (candidate.content !== undefined) {
    if (!isRecord(candidate.content)) {
      throw new ProviderError('gemini', 'Gemini SSE candidate content must be an object')
    }
    const parts = candidate.content.parts
    if (parts !== undefined && !Array.isArray(parts)) {
      throw new ProviderError('gemini', 'Gemini SSE candidate content parts must be an array')
    }
    for (const [index, part] of (parts ?? []).entries()) {
      if (!isRecord(part)) {
        throw new ProviderError('gemini', `Gemini SSE content part ${index} must be an object`)
      }
      const text = part.text
      if (text !== undefined) {
        if (typeof text !== 'string') {
          throw new ProviderError('gemini', `Gemini SSE text part ${index} must be a string`)
        }
        if (part.thought === true) {
          thinkingParts.push(text)
        } else {
          textParts.push(text)
        }
      }
      if (part.thought !== undefined && typeof part.thought !== 'boolean') {
        throw new ProviderError('gemini', `Gemini SSE thought flag ${index} must be a boolean`)
      }
      if (part.thoughtSignature !== undefined) {
        if (typeof part.thoughtSignature !== 'string') {
          throw new ProviderError('gemini', `Gemini SSE thought signature ${index} must be a string`)
        }
        if (part.thoughtSignature) {
          thinkingSignature = part.thoughtSignature
        }
      }
      if (part.functionCall !== undefined) {
        toolCalls.push(parseGeminiFunctionCall(part.functionCall, index))
      }
    }
  }
  const rawFinishReason = candidate.finishReason
  if (rawFinishReason !== undefined && typeof rawFinishReason !== 'string') {
    throw new ProviderError('gemini', 'Gemini SSE finishReason must be a string')
  }
  return {
    ...(textParts.length ? { content: textParts.join('') } : {}),
    ...(thinkingParts.length ? { thinking: thinkingParts.join('') } : {}),
    ...(thinkingSignature ? { thinkingSignature } : {}),
    ...(rawFinishReason ? { finishReason: normalizeFinishReason(rawFinishReason) } : {}),
    toolCalls,
  }
}

function parseGeminiFunctionCall(value: unknown, index: number): ToolCall {
  if (!isRecord(value)) {
    throw new ProviderError('gemini', `Gemini SSE functionCall ${index} must be an object`)
  }
  const name = stringValue(value.name)
  if (!name) {
    throw new ProviderError('gemini', `Gemini SSE functionCall ${index} is missing a name`)
  }
  if (value.args !== undefined && !isJsonObject(value.args)) {
    throw new ProviderError('gemini', `Gemini SSE functionCall ${index} args must be an object`)
  }
  const arguments_ = value.args === undefined ? {} : value.args
  const id = stringValue(value.id) || deterministicToolCallId(name, arguments_)
  return {
    id,
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

function geminiUsage(value: unknown): TokenUsage | undefined {
  if (value === undefined) {
    return undefined
  }
  if (!isRecord(value)) {
    throw new ProviderError('gemini', 'Gemini SSE usageMetadata must be an object')
  }
  const inputTokens = tokenCount(value, 'promptTokenCount')
  const outputTokens = tokenCount(value, 'candidatesTokenCount')
  const cacheReadTokens = tokenCount(value, 'cachedContentTokenCount')
  const reasoningTokens = tokenCount(value, 'thoughtsTokenCount')
  if (
    inputTokens === undefined
    && outputTokens === undefined
    && cacheReadTokens === undefined
    && reasoningTokens === undefined
  ) {
    return undefined
  }
  return {
    inputTokens: inputTokens ?? 0,
    outputTokens: outputTokens ?? 0,
    ...(cacheReadTokens === undefined ? {} : { cacheReadTokens }),
    ...(reasoningTokens === undefined ? {} : { reasoningTokens }),
  }
}

function tokenCount(value: Record<string, unknown>, key: string): number | undefined {
  const raw = value[key]
  if (raw === undefined) {
    return undefined
  }
  if (typeof raw !== 'number' || !Number.isSafeInteger(raw) || raw < 0) {
    throw new ProviderError('gemini', `Gemini SSE ${key} must be a non-negative integer`)
  }
  return raw
}

function promptBlockReason(value: unknown): string | undefined {
  if (value === undefined) {
    return undefined
  }
  if (!isRecord(value)) {
    throw new ProviderError('gemini', 'Gemini SSE promptFeedback must be an object')
  }
  const blockReason = value.blockReason
  if (blockReason === undefined) {
    return undefined
  }
  if (typeof blockReason !== 'string') {
    throw new ProviderError('gemini', 'Gemini SSE promptFeedback.blockReason must be a string')
  }
  return blockReason || undefined
}

function normalizeFinishReason(value: string): string {
  const normalized = value.toLowerCase()
  if (normalized === 'max_tokens') {
    return 'length'
  }
  return normalized === 'finish_reason_unspecified' ? 'stop' : normalized
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}
