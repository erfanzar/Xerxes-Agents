// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Mistral native chat-completions transport (pi-ai `mistral-conversations`).
 *
 * Distinct from the generic OpenAI-compatible client: Mistral's wire dialect
 * carries `thinking` content chunks, `tool_calls` with stream indexes,
 * `prompt_mode`/`reasoning_effort` reasoning controls, `prompt_cache_key`
 * routing plus the `x-affinity` affinity header, and tool-call ids restricted
 * to nine alphanumeric characters.
 */

import { ConfigurationError, ProviderError } from '../core/errors.js'
import type { ChatMessage } from '../types/messages.js'
import { messageText } from '../types/messages.js'
import { parseToolArguments, type ToolCall, type ToolDefinition } from '../types/toolCalls.js'
import type { CompletionRequest, FetchImplementation, LlmClient, LlmCompletion, LlmDelta, TokenUsage } from './client.js'
import { internalSseData } from './client.js'
import type { PiModelCapabilities } from './piModelCatalog.js'
import { piCatalogModelCapabilities } from './piModelCatalog.js'
import { bareModel } from './providerRegistry.js'

const MISTRAL_TOOL_CALL_ID_LENGTH = 9
const MAX_ERROR_BODY_CHARS = 4_000
/** Models that take reasoning strength through `reasoning_effort` (pi-ai usesReasoningEffort). */
const REASONING_EFFORT_MODELS = new Set(['mistral-small-2603', 'mistral-small-latest', 'mistral-medium-3.5'])

export interface MistralClientOptions {
  readonly apiKey?: string
  readonly baseUrl?: string
  /**
   * Enable Mistral's prompt-cache affinity: `x-affinity: <sessionId>` header
   * plus `prompt_cache_key` in the payload (pi-ai when cacheRetention allows).
   */
  readonly promptCaching?: boolean
  readonly fetchImplementation?: FetchImplementation
}

/** pi-ai shortHash: fast deterministic 36-base pair hash used to shorten ids. */
function shortHash(value: string): string {
  let h1 = 0xdeadbeef
  let h2 = 0x41c6ce57
  for (let index = 0; index < value.length; index++) {
    const ch = value.charCodeAt(index)
    h1 = Math.imul(h1 ^ ch, 2654435761)
    h2 = Math.imul(h2 ^ ch, 1597334677)
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909)
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909)
  return (h2 >>> 0).toString(36) + (h1 >>> 0).toString(36)
}

function deriveMistralToolCallId(id: string, attempt: number): string {
  const normalized = id.replace(/[^a-zA-Z0-9]/g, '')
  if (attempt === 0 && normalized.length === MISTRAL_TOOL_CALL_ID_LENGTH) return normalized
  const seedBase = normalized || id
  const seed = attempt === 0 ? seedBase : `${seedBase}:${attempt}`
  return shortHash(seed).replace(/[^a-zA-Z0-9]/g, '').slice(0, MISTRAL_TOOL_CALL_ID_LENGTH)
}

/**
 * Collision-safe nine-character id normalizer for replayed tool calls
 * (pi-ai createMistralToolCallIdNormalizer): distinct source ids always map
 * to distinct wire ids within one request.
 */
export function createMistralToolCallIdNormalizer(): (id: string) => string {
  const idMap = new Map<string, string>()
  const reverseMap = new Map<string, string>()
  return (id: string) => {
    const existing = idMap.get(id)
    if (existing) return existing
    let attempt = 0
    for (;;) {
      const candidate = deriveMistralToolCallId(id, attempt)
      const owner = reverseMap.get(candidate)
      if (owner === undefined || owner === id) {
        idMap.set(id, candidate)
        reverseMap.set(candidate, id)
        return candidate
      }
      attempt += 1
    }
  }
}

function sanitizeSurrogates(value: string): string {
  return value.replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, '')
}

type MistralWireChunk =
  | { thinking: { text: string; type: "text" }[]; type: "thinking" }
  | { text: string; type: "text" }
  | { imageUrl: string; type: "image_url" }

interface MistralWireMessage {
  content: MistralWireChunk[] | string
  name?: string
  prefix?: boolean
  role: 'assistant' | 'system' | 'tool' | 'user'
  tool_call_id?: string
  tool_calls?: {
    function: { arguments: string; name: string }
    id: string
    index: number
    type: 'function'
  }[]
}

function toolResultText(text: string, isError: boolean): string {
  const trimmed = text.trim()
  return `${isError ? '[tool error] ' : ''}${trimmed || '(no tool output)'}`
}

/**
 * pi-ai toChatMessages: Mistral dialect replay including thinking chunks and
 * tool calls with the nine-character id normalization.
 */
export function mistralMessages(
  request: CompletionRequest,
  normalizeToolCallId: (id: string) => string,
): MistralWireMessage[] {
  const result: MistralWireMessage[] = []
  const systemParts = request.messages
    .filter(message => message.role === 'system')
    .map(message => sanitizeSurrogates(messageText(message)))
    .filter(Boolean)
  if (systemParts.length) {
    result.push({ content: systemParts.join('\n\n'), role: 'system' })
  }

  for (const message of request.messages) {
    if (message.role === 'system') continue
    if (message.role === 'user') {
      if (typeof message.content === 'string') {
        result.push({ content: sanitizeSurrogates(message.content), role: 'user' })
        continue
      }
      const chunks: MistralWireChunk[] = []
      for (const part of message.content) {
        if (part.type === 'text') {
          const text = sanitizeSurrogates(part.text)
          if (text) chunks.push({ text, type: 'text' })
          continue
        }
        const match = /^data:([^;,]+);base64,([A-Za-z0-9+/=\r\n]+)$/.exec(part.image_url.url)
        if (match?.[1] && match?.[2]) {
          chunks.push({ imageUrl: `data:${match[1]};base64,${match[2].replaceAll(/\s/g, '')}`, type: 'image_url' })
        }
      }
      if (chunks.length) result.push({ content: chunks, role: 'user' })
      continue
    }
    if (message.role === 'assistant') {
      const content: (
        | { text: string; type: 'text' }
        | { thinking: { text: string; type: 'text' }[]; type: 'thinking' }
      )[] = []
      const text = sanitizeSurrogates(messageText(message))
      if (text.trim()) content.push({ text, type: 'text' })
      if (message.thinking?.trim()) {
        content.push({ thinking: [{ text: sanitizeSurrogates(message.thinking), type: 'text' }], type: 'thinking' })
      }
      const toolCalls = (message.tool_calls ?? []).map(call => ({
        function: { arguments: JSON.stringify(call.function.arguments), name: call.function.name },
        id: normalizeToolCallId(call.id),
        index: 0,
        type: 'function' as const,
      }))
      if (content.length === 0 && toolCalls.length === 0) continue
      result.push({
        content,
        prefix: false,
        role: 'assistant',
        ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
      })
      continue
    }

    // Tool result: text joined with error prefix (Mistral takes text chunks).
    const text = toolResultText(sanitizeSurrogates(message.content), message.is_error === true)
    result.push({
      content: [{ text, type: 'text' }],
      ...(message.name ? { name: message.name } : {}),
      role: 'tool',
      tool_call_id: normalizeToolCallId(message.tool_call_id),
    })
  }
  return result
}

function mistralTools(tools: readonly ToolDefinition[]): Record<string, unknown>[] {
  return tools.map(tool => ({
    type: 'function',
    function: {
      name: tool.function.name,
      description: tool.function.description,
      parameters: tool.function.parameters,
      strict: false,
    },
  }))
}

function mapToolChoice(choice: CompletionRequest['toolChoice']): string | undefined {
  switch (choice) {
    case 'any':
    case 'auto':
    case 'none':
      return choice
    default:
      return undefined
  }
}

/** pi-ai buildChatPayload + toMistralWirePayload. */
export function mistralPayload(
  request: CompletionRequest,
  options: { promptCaching: boolean },
): Record<string, unknown> {
  const modelId = bareModel(request.model)
  const capabilities: PiModelCapabilities | undefined = piCatalogModelCapabilities(modelId, 'mistral')
  const normalizeToolCallId = createMistralToolCallIdNormalizer()
  const payload: Record<string, unknown> = {
    model: modelId,
    stream: true,
    messages: mistralMessages(request, normalizeToolCallId),
  }
  if (request.tools?.length) {
    payload.tools = mistralTools(request.tools)
    const choice = mapToolChoice(request.toolChoice)
    if (choice) payload.tool_choice = choice
  }
  if (request.temperature !== undefined) payload.temperature = request.temperature
  if (request.maxTokens !== undefined) payload.max_tokens = request.maxTokens
  if (request.topP !== undefined) payload.top_p = request.topP
  const reasoning = request.thinking?.effort !== undefined && request.thinking.effort !== 'off'
    ? request.thinking.effort
    : undefined
  if (reasoning && capabilities?.reasoning !== false) {
    if (REASONING_EFFORT_MODELS.has(modelId)) {
      const mapped = capabilities?.thinkingLevelMap?.[reasoning]
      payload.reasoning_effort = typeof mapped === 'string' && mapped ? mapped : 'high'
    } else {
      payload.prompt_mode = 'reasoning'
    }
  }
  if (options.promptCaching && request.sessionId) {
    payload.prompt_cache_key = request.sessionId
  }
  return payload
}

interface MistralStreamEvent {
  choices?: {
    delta?: {
      content?: string | Record<string, unknown>[]
      tool_calls?: {
        function?: { arguments?: string | Record<string, unknown>; name?: string }
        id?: string
        index?: number
      }[]
    }
    finish_reason?: string | null
  }[]
  id?: string
  usage?: {
    completion_tokens?: number
    prompt_tokens?: number
    prompt_tokens_details?: { cached_tokens?: number }
    total_tokens?: number
  }
}

function mistralCachedTokens(usage: {
  completion_tokens?: number | undefined
  prompt_tokens?: number | undefined
  prompt_tokens_details?: { cached_tokens?: number | undefined } | undefined
}): number {
  const cached = usage.prompt_tokens_details?.cached_tokens ?? 0
  const promptTokens = usage.prompt_tokens ?? 0
  return Math.min(promptTokens, Math.max(0, typeof cached === 'number' && Number.isFinite(cached) ? cached : 0))
}

function mistralUsage(usage: {
  completion_tokens?: number | undefined
  prompt_tokens?: number | undefined
  prompt_tokens_details?: { cached_tokens?: number | undefined } | undefined
  total_tokens?: number | undefined
}): TokenUsage {
  const cached = mistralCachedTokens(usage)
  const input = Math.max(0, (usage.prompt_tokens ?? 0) - cached)
  const output = usage.completion_tokens ?? 0
  return {
    inputTokens: input,
    outputTokens: output,
    ...(cached ? { cacheReadTokens: cached } : {}),
  }
}

function mapFinishReason(reason: string | null | undefined): string | undefined {
  if (reason === null || reason === undefined) return undefined
  switch (reason) {
    case 'stop':
      return 'stop'
    case 'length':
    case 'model_length':
      return 'length'
    case 'tool_calls':
      return 'tool_calls'
    case 'error':
      throw new ProviderError('mistral', 'Provider stopped with: error')
    default:
      throw new ProviderError('mistral', `Provider stopped with: ${reason}`)
  }
}

function contentDeltaText(item: unknown): { thinking?: string; text?: string } {
  if (typeof item === 'string') return { text: item }
  if (typeof item !== 'object' || item === null) return {}
  const record = item as Record<string, unknown>
  if (record.type === 'thinking') {
    const thinking = record.thinking
    if (!Array.isArray(thinking)) return {}
    const joined = thinking
      .map(part => (typeof part === 'object' && part !== null && typeof (part as Record<string, unknown>).text === 'string'
        ? (part as Record<string, unknown>).text as string
        : ''))
      .filter(text => text.length > 0)
      .join('')
    return { thinking: joined }
  }
  const text = typeof record.text === 'string' ? record.text : ''
  return text ? { text } : {}
}

export class MistralClient implements LlmClient {
  private readonly apiKey: string
  private readonly baseUrl: string
  private readonly fetchImplementation: FetchImplementation
  private readonly promptCaching: boolean

  constructor(options: MistralClientOptions = {}) {
    this.apiKey = options.apiKey ?? process.env.MISTRAL_API_KEY ?? ''
    this.baseUrl = trimSlash(options.baseUrl ?? 'https://api.mistral.ai')
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.promptCaching = options.promptCaching ?? true
    if (!this.apiKey) {
      throw new ConfigurationError('MISTRAL_API_KEY', 'Mistral API key not provided')
    }
    if (!this.baseUrl) {
      throw new ConfigurationError('base_url', 'No base URL is configured for mistral')
    }
  }

  private headers(accept: string, request: CompletionRequest): Record<string, string> {
    const headers: Record<string, string> = {
      Accept: accept,
      Authorization: `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
      'User-Agent': 'xerxes-agents/0.3.0',
    }
    if (this.promptCaching && request.sessionId) {
      headers['x-affinity'] = request.sessionId
    }
    return headers
  }

  private endpoint(): string {
    return `${this.baseUrl}/v1/chat/completions`
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    const payload = { ...mistralPayload(request, { promptCaching: this.promptCaching }), stream: false }
    const response = await this.fetchImplementation(this.endpoint(), {
      method: 'POST',
      headers: this.headers('application/json', request),
      body: JSON.stringify(payload),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      throw await mistralHttpError(response)
    }
    const body = await response.json() as {
      choices?: {
        finish_reason?: string | null
        message?: {
          content?: string
          tool_calls?: { function?: { arguments?: string }; id?: string; name?: string }[]
        }
      }[]
      usage?: { completion_tokens?: number; prompt_tokens?: number; prompt_tokens_details?: { cached_tokens?: number } }
    }
    const choice = body.choices?.[0]
    if (!choice?.message) {
      throw new ProviderError('mistral', 'completion response did not include a choice')
    }
    const normalizeToolCallId = createMistralToolCallIdNormalizer()
    const content = typeof choice.message.content === 'string' ? choice.message.content : ''
    const toolCalls: ToolCall[] = (choice.message.tool_calls ?? []).map(call => ({
      id: normalizeToolCallId(call.id ?? ""),
      type: "function" as const,
      function: {
        name: call.name ?? "",
        arguments: parseToolArguments(
          typeof call.function?.arguments === "string" ? call.function.arguments : "{}",
        ),
      },
    }))
    const finishReason = mapFinishReason(choice.finish_reason)
    const usage = body.usage
      ? mistralUsage({
        completion_tokens: body.usage.completion_tokens,
        prompt_tokens: body.usage.prompt_tokens,
        prompt_tokens_details: body.usage.prompt_tokens_details,
      })
      : undefined
    return {
      content,
      toolCalls,
      ...(finishReason ? { finishReason } : {}),
      ...(usage ? { usage } : {}),
    }
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const response = await this.fetchImplementation(this.endpoint(), {
      method: 'POST',
      headers: this.headers('text/event-stream', request),
      body: JSON.stringify(mistralPayload(request, { promptCaching: this.promptCaching })),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      throw await mistralHttpError(response)
    }
    if (!response.body) {
      throw new ProviderError('mistral', 'stream response returned no body')
    }

    const pendingToolCalls = new Map<number, { arguments: string; id: string; name: string }>()
    let usage: TokenUsage | undefined
    let finishReason: string | undefined
    for await (const data of internalSseData(response.body)) {
      if (data === '[DONE]') break
      let event: MistralStreamEvent
      try {
        event = JSON.parse(data) as MistralStreamEvent
      } catch {
        throw new ProviderError('mistral', 'stream produced a malformed SSE event')
      }
      if (!Array.isArray(event.choices)) {
        throw new ProviderError('mistral', 'Invalid Mistral streaming event')
      }
      if (event.usage) usage = mistralUsage(event.usage)
      const choice = event.choices[0]
      if (!choice) continue
      if (choice.finish_reason !== undefined && choice.finish_reason !== null) {
        finishReason = mapFinishReason(choice.finish_reason)
      }
      const delta = choice.delta
      if (!delta) continue

      const contentItems = delta.content === undefined || delta.content === null
        ? []
        : typeof delta.content === 'string'
          ? [delta.content]
          : delta.content
      for (const item of contentItems) {
        const { text, thinking } = contentDeltaText(item)
        if (thinking) yield { thinking }
        if (text) yield { content: text }
      }

      for (const [position, call] of (delta.tool_calls ?? []).entries()) {
        const index = call.index ?? position
        const existing = pendingToolCalls.get(index)
        const argumentsDelta = typeof call.function?.arguments === 'string'
          ? call.function.arguments
          : JSON.stringify(call.function?.arguments ?? {})
        if (existing) {
          existing.arguments += argumentsDelta
        } else {
          pendingToolCalls.set(index, {
            id: call.id && call.id !== 'null' ? call.id : deriveMistralToolCallId(`toolcall:${index}`, 0),
            name: call.function?.name ?? '',
            arguments: argumentsDelta,
          })
        }
      }
    }

    if (pendingToolCalls.size) {
      yield {
        toolCalls: [...pendingToolCalls.values()].map(call => ({
          id: call.id,
          type: "function" as const,
          function: { name: call.name, arguments: parseToolArguments(call.arguments) },
        })),
      }
    }
    if (usage) yield { usage }
    if (finishReason) yield { finishReason }
  }
}

async function mistralHttpError(response: Response): Promise<ProviderError> {
  const body = (await response.text()).trim()
  const detail = body
    ? `${body.slice(0, MAX_ERROR_BODY_CHARS)}${body.length > MAX_ERROR_BODY_CHARS ? `... [truncated ${body.length - MAX_ERROR_BODY_CHARS} chars]` : ''}`
    : response.statusText
  return new ProviderError('mistral', `Mistral API error (${response.status}): ${detail}`)
}

function trimSlash(value: string): string {
  return value.replace(/\/+$/, '')
}
