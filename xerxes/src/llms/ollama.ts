// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError, ProviderError } from '../core/errors.js'
import { deterministicToolCallId } from '../streaming/toolCallIds.js'
import type { ChatMessage, MessageContent } from '../types/messages.js'
import type { JsonObject, ToolCall } from '../types/toolCalls.js'
import { isJsonObject, parseToolArguments } from '../types/toolCalls.js'
import type { CompletionRequest, FetchImplementation, LlmClient, LlmCompletion, LlmDelta, TokenUsage } from './client.js'
import { bareModel } from './providerRegistry.js'

/** Default root endpoint for a locally running Ollama daemon. */
export const DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434'

/** Bound a single NDJSON record before parsing it into a provider response. */
export const MAX_OLLAMA_NDJSON_LINE_BYTES = 10 * 1024 * 1024

export interface OllamaClientOptions {
  /** Ollama daemon root. A `/v1` path is tolerated because `/api/chat` is root-relative. */
  readonly baseUrl?: string
  readonly fetchImplementation?: FetchImplementation
  readonly maxLineBytes?: number
  /** Default `top_k` sampling option for every request made by this client. */
  readonly topK?: number
}

export interface OllamaMessage {
  readonly content: string
  readonly role: ChatMessage['role']
  readonly thinking?: string
  readonly tool_calls?: readonly OllamaToolCall[]
}

export interface OllamaToolCall {
  readonly function: {
    readonly arguments: JsonObject
    readonly name: string
  }
}

/** Convert Xerxes's provider-neutral transcript to Ollama's chat message shape. */
export function messagesToOllama(messages: readonly ChatMessage[]): OllamaMessage[] {
  return messages.map(message => {
    const base = { role: message.role, content: ollamaContent(message.content) }
    if (message.role !== 'assistant') {
      return base
    }
    return {
      ...base,
      ...(message.thinking ? { thinking: message.thinking } : {}),
      ...(message.tool_calls?.length ? {
        tool_calls: message.tool_calls.map(call => ({
          function: {
            name: call.function.name,
            arguments: call.function.arguments,
          },
        })),
      } : {}),
    }
  })
}

/**
 * Native direct adapter for Ollama's `/api/chat` NDJSON endpoint.
 *
 * Ollama's direct endpoint does not expose an OpenAI-style `tool_choice`
 * switch. Supplying request tools makes them available to the model; the
 * neutral delta stream still reports completed tool calls for the runtime.
 */
export class OllamaClient implements LlmClient {
  private readonly baseUrl: string
  private readonly fetchImplementation: FetchImplementation
  private readonly maxLineBytes: number
  private readonly topK: number | undefined

  constructor(options: OllamaClientOptions = {}) {
    this.baseUrl = validBaseUrl(options.baseUrl ?? DEFAULT_OLLAMA_BASE_URL)
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.maxLineBytes = options.maxLineBytes ?? MAX_OLLAMA_NDJSON_LINE_BYTES
    this.topK = options.topK

    if (!Number.isSafeInteger(this.maxLineBytes) || this.maxLineBytes <= 0) {
      throw new ConfigurationError('maxLineBytes', 'must be a positive integer')
    }
    if (this.topK !== undefined && (!Number.isSafeInteger(this.topK) || this.topK < 0)) {
      throw new ConfigurationError('topK', 'must be a non-negative integer')
    }
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    const response = await this.fetchImplementation(new URL('/api/chat', this.baseUrl), {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(ollamaRequestPayload(request, this.topK, false)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      const status = response.statusText ? ` ${response.statusText}` : ''
      throw new ProviderError('ollama', `chat completion request failed (${response.status}${status}): ${body.slice(0, 4_096)}`)
    }

    const chunk = parseChunk(await response.text())
    const message = asRecord(chunk.message)
    const content = stringAt(message, 'content') ?? ''
    const thinking = stringAt(message, 'thinking')
    const rawToolCalls = message.tool_calls ?? chunk.tool_calls
    const toolCalls = rawToolCalls === undefined ? [] : parseToolCalls(rawToolCalls) ?? []
    const finishReason = stringAt(chunk, 'done_reason') || (chunk.done === true ? 'stop' : undefined)
    const usage = ollamaUsage(chunk)

    return {
      content,
      toolCalls,
      ...(finishReason === undefined ? {} : { finishReason }),
      ...(thinking === undefined ? {} : { thinking }),
      ...(usage === undefined ? {} : { usage }),
    }
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const response = await this.fetchImplementation(new URL('/api/chat', this.baseUrl), {
      method: 'POST',
      headers: {
        Accept: 'application/x-ndjson',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(ollamaRequestPayload(request, this.topK, true)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      const status = response.statusText ? ` ${response.statusText}` : ''
      const message = `chat stream request failed (${response.status}${status}): ${body.slice(0, 4_096)}`
      throw new ProviderError('ollama', message)
    }
    if (!response.body) {
      throw new ProviderError('ollama', 'chat stream returned no response body')
    }

    for await (const line of ndjsonLines(response.body, this.maxLineBytes)) {
      const chunk = parseChunk(line)
      const message = asRecord(chunk.message)
      const content = stringAt(message, 'content')
      const thinking = stringAt(message, 'thinking')
      const rawToolCalls = message.tool_calls ?? chunk.tool_calls
      const toolCalls = rawToolCalls === undefined ? undefined : parseToolCalls(rawToolCalls)
      const finishReason = chunk.done === true ? stringAt(chunk, 'done_reason') || 'stop' : undefined
      const usage = ollamaUsage(chunk)

      const delta: {
        content?: string
        finishReason?: string
        thinking?: string
        toolCalls?: readonly ToolCall[]
        usage?: TokenUsage
      } = {}
      if (content) {
        delta.content = content
      }
      if (thinking) {
        delta.thinking = thinking
      }
      if (toolCalls) {
        delta.toolCalls = toolCalls
      }
      if (finishReason) {
        delta.finishReason = finishReason
      }
      if (usage) {
        delta.usage = usage
      }
      if (Object.keys(delta).length) {
        yield delta
      }
    }
  }
}

function ollamaRequestPayload(
  request: CompletionRequest,
  topK: number | undefined,
  stream: boolean,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    model: bareModel(request.model),
    messages: messagesToOllama(request.messages),
    stream,
  }
  const options = samplingOptions(request, topK)
  if (Object.keys(options).length) {
    payload.options = options
  }
  if (request.tools?.length) {
    payload.tools = request.tools
  }
  return payload
}

function ollamaContent(content: MessageContent): string {
  if (typeof content === 'string') {
    return content
  }
  return content.map(part => part.type === 'text' ? part.text : `[Image: ${part.image_url.url}]`).join('')
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

function samplingOptions(request: CompletionRequest, topK: number | undefined): Record<string, unknown> {
  const options: Record<string, unknown> = {}
  if (request.temperature !== undefined) {
    options.temperature = request.temperature
  }
  if (request.topP !== undefined) {
    options.top_p = request.topP
  }
  if (request.maxTokens !== undefined) {
    options.num_predict = request.maxTokens
  }
  if (request.stop?.length) {
    options.stop = request.stop
  }
  if (topK !== undefined) {
    options.top_k = topK
  }
  return options
}

function parseChunk(line: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(line)
    if (!isRecord(parsed)) {
      throw new Error('record expected')
    }
    return parsed
  } catch (error) {
    throw new ProviderError('ollama', `invalid NDJSON JSON: ${line.slice(0, 200)}`, error)
  }
}

function parseToolCalls(value: unknown): ToolCall[] | undefined {
  if (value === null) {
    return undefined
  }
  if (!Array.isArray(value)) {
    throw new ProviderError('ollama', 'message.tool_calls must be an array')
  }
  if (!value.length) {
    return undefined
  }
  return value.map((rawCall, index) => {
    const call = asRecord(rawCall)
    const function_ = asRecord(call.function)
    const name = stringAt(function_, 'name')
    if (!name) {
      throw new ProviderError('ollama', `tool call ${index} is missing a function name`)
    }
    const rawArguments = function_.arguments
    if (rawArguments !== undefined && typeof rawArguments !== 'string' && !isJsonObject(rawArguments)) {
      throw new ProviderError('ollama', `tool call ${index} has invalid function arguments`)
    }
    const arguments_ = parseToolArguments(rawArguments as string | JsonObject | undefined)
    return {
      id: stringAt(call, 'id') || deterministicToolCallId(name, arguments_),
      type: 'function' as const,
      function: { name, arguments: arguments_ },
    }
  })
}

function ollamaUsage(chunk: Record<string, unknown>): TokenUsage | undefined {
  const inputTokens = numberAt(chunk, 'prompt_eval_count')
  const outputTokens = numberAt(chunk, 'eval_count')
  if (inputTokens === undefined && outputTokens === undefined) {
    return undefined
  }
  return { inputTokens: inputTokens ?? 0, outputTokens: outputTokens ?? 0 }
}

function asRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {}
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function numberAt(value: Record<string, unknown>, key: string): number | undefined {
  const item = value[key]
  return typeof item === 'number' && Number.isFinite(item) ? item : undefined
}

function stringAt(value: Record<string, unknown>, key: string): string | undefined {
  const item = value[key]
  return typeof item === 'string' ? item : undefined
}

async function* ndjsonLines(body: ReadableStream<Uint8Array>, maxLineBytes: number): AsyncGenerator<string> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffered: Uint8Array<ArrayBufferLike> = new Uint8Array()
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }
      if (!value?.byteLength) {
        continue
      }
      buffered = appendBytes(buffered, value)
      let lineStart = 0
      for (let index = 0; index < buffered.byteLength; index += 1) {
        if (buffered[index] !== 10) {
          continue
        }
        const lineEnd = index > lineStart && buffered[index - 1] === 13 ? index - 1 : index
        ensureLineSize(lineEnd - lineStart, maxLineBytes)
        if (lineEnd > lineStart) {
          yield decoder.decode(buffered.subarray(lineStart, lineEnd))
        }
        lineStart = index + 1
      }
      buffered = buffered.slice(lineStart)
      const trailingCarriageReturn = buffered.byteLength > 0 && buffered[buffered.byteLength - 1] === 13 ? 1 : 0
      ensureLineSize(buffered.byteLength - trailingCarriageReturn, maxLineBytes)
    }
    if (buffered.byteLength) {
      const lineEnd = buffered[buffered.byteLength - 1] === 13 ? buffered.byteLength - 1 : buffered.byteLength
      ensureLineSize(lineEnd, maxLineBytes)
      if (lineEnd) {
        yield decoder.decode(buffered.subarray(0, lineEnd))
      }
    }
  } finally {
    reader.releaseLock()
  }
}

function appendBytes(
  left: Uint8Array<ArrayBufferLike>,
  right: Uint8Array<ArrayBufferLike>,
): Uint8Array<ArrayBufferLike> {
  const joined = new Uint8Array(left.byteLength + right.byteLength)
  joined.set(left)
  joined.set(right, left.byteLength)
  return joined
}

function ensureLineSize(size: number, maxLineBytes: number): void {
  if (size > maxLineBytes) {
    throw new ProviderError('ollama', `NDJSON line exceeded maximum size of ${maxLineBytes} bytes`)
  }
}
