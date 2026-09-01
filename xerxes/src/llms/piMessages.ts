// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * pi-messages API client (pi-ai parity, dist/api/pi-messages.ts).
 *
 * Streams pi's own message protocol directly to a backend: the request is a
 * single POST of `{ model, context, options }` to `<baseUrl>/messages`, the
 * response is an SSE stream of serialized assistant-message events plus a
 * terminal `done`/`error` event. This is the wire protocol spoken by the
 * Radius gateway (see ./radiusGateway.ts), but any backend implementing it
 * can be used, e.g. via a models.json custom provider with
 * `"api": "pi-messages"`.
 */

import { parseStreamingJson } from '@earendil-works/pi-ai'

import { ConfigurationError, ProviderError } from '../core/errors.js'
import type { ChatMessage, ContentPart, ImageUrlContentPart } from '../types/messages.js'
import { messageText } from '../types/messages.js'
import type { JsonObject, ToolChoice, ToolDefinition } from '../types/toolCalls.js'
import type { CompletionRequest, FetchImplementation, LlmClient, LlmCompletion, LlmDelta, TokenUsage } from './client.js'
import { collectLlmCompletion } from './client.js'

/** Pi thinking levels the wire accepts; Xerxes effort hints are a subset. */
const PI_THINKING_LEVELS = new Set(['minimal', 'low', 'medium', 'high', 'xhigh', 'max'])

/** Impact summary of a server-side message rewrite (e.g. a gateway policy). */
export interface PiMessagesRewriteImpact {
  changed: boolean
  messageCountChange: number
  policyId: string
  policyVersion: number
  systemPromptChanged: boolean
  tokenCountChange: number
}

/** Serialized assistant-message event as sent by a pi-messages backend. */
export type PiMessagesEvent =
  | { type: 'start' }
  | { type: 'text_start'; contentIndex: number }
  | { type: 'text_delta'; contentIndex: number; delta: string }
  | { content: string; contentIndex: number; contentSignature?: string; type: 'text_end' }
  | { contentIndex: number; type: 'thinking_start' }
  | { contentIndex: number; delta: string; type: 'thinking_delta' }
  | {
    content: string
    contentIndex: number
    contentSignature?: string
    redacted?: boolean
    type: 'thinking_end'
  }
  | { contentIndex: number; id: string; toolName: string; type: 'toolcall_start' }
  | { contentIndex: number; delta: string; type: 'toolcall_delta' }
  | { contentIndex: number; toolCall: PiWireToolCall; type: 'toolcall_end' }
  | {
    reason: 'stop' | 'length' | 'toolUse'
    responseId?: string
    rewrite?: PiMessagesRewriteImpact
    type: 'done'
    usage: PiWireUsage
  }
  | {
    errorMessage?: string
    reason: 'aborted' | 'error'
    responseId?: string
    rewrite?: PiMessagesRewriteImpact
    type: 'error'
    usage: PiWireUsage
  }

/** Tool call as serialized by a pi-messages backend (Pi `ToolCall`). */
export interface PiWireToolCall {
  arguments: JsonObject
  id: string
  name: string
  namespace?: string
  thoughtSignature?: string
  type: 'toolCall'
}

/** Usage as serialized by a pi-messages backend (Pi `Usage`). */
export interface PiWireUsage {
  cacheRead: number
  cacheWrite: number
  cacheWrite1h?: number
  cost: { cacheRead: number; cacheWrite: number; input: number; output: number; total: number }
  input: number
  output: number
  reasoning?: number
  totalTokens: number
}

interface PiMessagesErrorBody {
  error?: { [key: string]: unknown; code?: unknown; details?: unknown; message?: unknown }
}

/** Typed response failure carrying the backend error code (Pi parity). */
export class PiMessagesResponseError extends ProviderError {
  code: string | undefined
  readonly diagnosticDetails: Record<string, unknown>

  constructor(
    provider: string,
    message: string,
    code: string | undefined,
    diagnosticDetails: Record<string, unknown>,
  ) {
    super(provider, message)
    this.name = 'PiMessagesResponseError'
    this.code = code
    this.diagnosticDetails = diagnosticDetails
  }
}

function parseErrorBody(body: string): PiMessagesErrorBody | undefined {
  try {
    const parsed: unknown = JSON.parse(body)
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return undefined
    const error = (parsed as PiMessagesErrorBody).error
    return typeof error === 'object' && error !== null && !Array.isArray(error)
      ? (parsed as PiMessagesErrorBody)
      : undefined
  } catch {
    return undefined
  }
}

function truncateDiagnosticString(value: string): string {
  const maxLength = 8_192
  return value.length > maxLength ? `${value.slice(0, maxLength)}…` : value
}

function createResponseError(
  url: URL,
  response: Response,
  body: string,
  providerId: string,
  modelId: string,
): PiMessagesResponseError {
  const errorBody = parseErrorBody(body)
  const message = typeof errorBody?.error?.message === 'string' ? errorBody.error.message : undefined
  const code = typeof errorBody?.error?.code === 'string' ? errorBody.error.code : undefined
  const suffix = message ?? body
  const codeSuffix = code ? ` (${code})` : ''
  return new PiMessagesResponseError(
    providerId,
    `${response.status} ${response.statusText}: ${suffix}${codeSuffix}`,
    code,
    {
      body: errorBody ? undefined : truncateDiagnosticString(body),
      error: errorBody?.error,
      model: modelId,
      provider: providerId,
      status: response.status,
      statusText: response.statusText,
      timestampMs: Date.now(),
      url: url.toString(),
      version: 1,
    },
  )
}

/** Pi `Usage` → neutral {@link TokenUsage}; cost is computed by Xerxes' own tracker. */
function toTokenUsage(usage: PiWireUsage): TokenUsage {
  return {
    ...(typeof usage.cacheWrite === 'number' ? { cacheCreationTokens: usage.cacheWrite } : {}),
    ...(typeof usage.cacheRead === 'number' ? { cacheReadTokens: usage.cacheRead } : {}),
    inputTokens: usage.input,
    outputTokens: usage.output,
    ...(typeof usage.reasoning === 'number' ? { reasoningTokens: usage.reasoning } : {}),
  }
}

/** Pi stop reason on a successful terminal event → Xerxes finish reason. */
function toFinishReason(reason: 'stop' | 'length' | 'toolUse'): string {
  if (reason === 'toolUse') return 'tool_calls'
  return reason
}

/**
 * Incremental delta conversion for one response (Pi's createEventConverter,
 * reshaped onto Xerxes' delta vocabulary). Text and thinking `*_end` events
 * are authoritative: their content replaces what the deltas produced, so a
 * compliant backend can skip deltas entirely and stream only endings.
 */
class PiMessagesEventConverter {
  private readonly thinkingParts = new Map<number, string>()
  private readonly textParts = new Map<number, string>()
  private readonly toolJson = new Map<number, { id: string; json: string; name: string }>()

  convert(event: PiMessagesEvent): LlmDelta | undefined {
    switch (event.type) {
      case 'start':
      case 'text_start':
      case 'thinking_start':
        return undefined
      case 'text_delta': {
        this.textParts.set(event.contentIndex, (this.textParts.get(event.contentIndex) ?? '') + event.delta)
        return { content: event.delta }
      }
      case 'text_end': {
        // A signature has no neutral channel; the content is what matters.
        return this.authoritativeDelta(this.textParts, event.contentIndex, event.content, 'text_end')
      }
      case 'thinking_delta': {
        this.thinkingParts.set(event.contentIndex, (this.thinkingParts.get(event.contentIndex) ?? '') + event.delta)
        return { thinking: event.delta }
      }
      case 'thinking_end': {
        const delta = this.authoritativeDelta(this.thinkingParts, event.contentIndex, event.content, 'thinking_end')
        if (event.contentSignature === undefined) return delta
        return { ...delta, thinkingSignature: event.contentSignature }
      }
      case 'toolcall_start': {
        this.toolJson.set(event.contentIndex, { id: event.id, json: '', name: event.toolName })
        return undefined
      }
      case 'toolcall_delta': {
        const pending = this.toolJson.get(event.contentIndex)
        if (pending) pending.json += event.delta
        return undefined
      }
      case 'toolcall_end': {
        // The terminal toolCall is authoritative; arguments arrive as a parsed
        // object, so pending delta text is only an intermediate view.
        this.toolJson.delete(event.contentIndex)
        return { toolCalls: [wireToolCallToNeutral(event.toolCall)] }
      }
      case 'done':
        return { finishReason: toFinishReason(event.reason), usage: toTokenUsage(event.usage) }
      case 'error':
        throw new ProviderError(
          'pi-messages',
          event.errorMessage
            ?? `pi-messages stream failed (${event.reason})`,
        )
    }
  }

  /** Emit only what the deltas have not already produced; throw on divergence. */
  private authoritativeDelta(
    parts: Map<number, string>,
    contentIndex: number,
    content: string,
    eventType: 'text_end' | 'thinking_end',
  ): LlmDelta | undefined {
    const accumulated = parts.get(contentIndex) ?? ''
    parts.set(contentIndex, content)
    if (content === accumulated) return undefined
    if (!content.startsWith(accumulated)) {
      // The emitted prefix can never be taken back, so an unrelated rewrite
      // would silently corrupt the transcript if it were appended.
      throw new ProviderError(
        'pi-messages',
        `${eventType} content diverges from the streamed deltas; the backend rewrote text that was already emitted`,
      )
    }
    const remainder = content.slice(accumulated.length)
    return eventType === 'text_end' ? { content: remainder } : { thinking: remainder }
  }
}

/** Pi serialized tool call → neutral {@link ToolCall}-shaped delta entry. */
function wireToolCallToNeutral(toolCall: PiWireToolCall): {
  function: { arguments: JsonObject; name: string }
  id: string
  type: 'function'
} {
  return {
    id: toolCall.id,
    type: 'function',
    function: {
      name: toolCall.name,
      arguments: toolCall.arguments,
    },
  }
}

/**
 * Read pi-messages SSE frames: blocks separated by a blank line, payload on
 * the first `data:` line, `[DONE]` ignored (pi-ai parity). A trailing block
 * without a separator still parses.
 */
export async function* readPiMessagesEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<PiMessagesEvent> {
  const decoder = new TextDecoder()
  const reader = stream.getReader()
  let buffer = ''
  try {
    for (;;) {
      const { done, value } = await reader.read()
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true })
      buffer = buffer.replaceAll('\r\n', '\n')

      let split = buffer.indexOf('\n\n')
      while (split !== -1) {
        const event = parsePiMessagesEvent(buffer.slice(0, split))
        if (event !== undefined) yield event
        buffer = buffer.slice(split + 2)
        split = buffer.indexOf('\n\n')
      }
      if (done) break
    }
    if (buffer.trim()) {
      const event = parsePiMessagesEvent(buffer)
      if (event !== undefined) yield event
    }
  } finally {
    reader.releaseLock()
  }
}

function parsePiMessagesEvent(raw: string): PiMessagesEvent | undefined {
  const data = raw
    .split('\n')
    .find(line => line.startsWith('data:'))
    ?.slice(5)
    .trim()
  if (!data || data === '[DONE]') return undefined
  return JSON.parse(data) as PiMessagesEvent
}

/** Tool schemas in the Pi `Tool` wire shape. */
function piWireTool(tool: ToolDefinition): Record<string, unknown> {
  return {
    name: tool.function.name,
    description: tool.function.description,
    parameters: tool.function.parameters,
    ...(tool.constrainedSampling ? { constrainedSampling: tool.constrainedSampling } : {}),
  }
}

function piImagePart(part: ImageUrlContentPart): Record<string, unknown> {
  const match = /^data:([^;,]+);base64,([A-Za-z0-9+/=\r\n]+)$/.exec(part.image_url.url)
  if (match?.[1] && match?.[2]) {
    return { type: 'image', data: match[2].replaceAll(/\s/g, ''), mimeType: match[1] }
  }
  // Pi's image content carries base64 data only; a remote URL has no
  // representation, so it degrades to the same placeholder text other
  // Xerxes adapters use rather than being dropped silently.
  return { type: 'text', text: `[Image: ${part.image_url.url}]` }
}

function zeroedUsage(): PiWireUsage {
  return {
    cacheRead: 0,
    cacheWrite: 0,
    cost: { cacheRead: 0, cacheWrite: 0, input: 0, output: 0, total: 0 },
    input: 0,
    output: 0,
    totalTokens: 0,
  }
}

/** Serialized context messages for one request (Pi `Context.messages`). */
function piWireMessages(
  messages: readonly ChatMessage[],
  wire: { modelId: string; now: () => number; providerId: string },
): Record<string, unknown>[] {
  const toolNames = new Map<string, string>()
  for (const message of messages) {
    if (message.role !== 'assistant') continue
    for (const call of message.tool_calls ?? []) toolNames.set(call.id, call.function.name)
  }

  const output: Record<string, unknown>[] = []
  for (const message of messages) {
    if (message.role === 'system') continue
    if (message.role === 'user') {
      output.push({
        role: 'user',
        content: typeof message.content === 'string'
          ? message.content
          : message.content.map(part => part.type === 'text'
            ? { type: 'text', text: part.text }
            : piImagePart(part)),
        timestamp: wire.now(),
      })
      continue
    }
    if (message.role === 'assistant') {
      const content: Record<string, unknown>[] = []
      if (message.thinking) {
        content.push({
          type: 'thinking',
          thinking: message.thinking,
          ...(message.thinking_signature ? { thinkingSignature: message.thinking_signature } : {}),
        })
      }
      const text = messageText(message)
      if (text) content.push({ type: 'text', text })
      for (const call of message.tool_calls ?? []) {
        content.push({ type: 'toolCall', id: call.id, name: call.function.name, arguments: call.function.arguments })
      }
      output.push({
        role: 'assistant',
        api: 'pi-messages',
        provider: wire.providerId,
        model: wire.modelId,
        content,
        usage: zeroedUsage(),
        stopReason: message.tool_calls?.length ? 'toolUse' : 'stop',
        timestamp: wire.now(),
      })
      continue
    }
    output.push({
      role: 'toolResult',
      toolCallId: message.tool_call_id,
      toolName: message.name ?? toolNames.get(message.tool_call_id) ?? '',
      content: [{ type: 'text', text: message.content }],
      ...(message.added_tool_names?.length ? { addedToolNames: [...message.added_tool_names] } : {}),
      isError: message.is_error ?? false,
      timestamp: wire.now(),
    })
  }
  return output
}

export interface PiMessagesContext {
  messages: Record<string, unknown>[]
  systemPrompt?: string
  tools?: Record<string, unknown>[]
}

/**
 * Serialize a neutral request into the Pi `Context` wire shape: system
 * messages fold into `systemPrompt`, tools into the Pi `Tool` form, and
 * prior turns replay as Pi user/assistant/toolResult messages.
 */
export function piMessagesContext(
  request: Pick<CompletionRequest, 'messages' | 'tools'>,
  wire: { modelId: string; now?: () => number; providerId: string },
): PiMessagesContext {
  const now = wire.now ?? (() => Date.now())
  const systemPrompt = request.messages
    .filter(message => message.role === 'system')
    .map(messageText)
    .filter(Boolean)
    .join('\n\n')
  const visible = request.messages.filter(message => message.role !== 'system')
  return {
    messages: piWireMessages(visible, { modelId: wire.modelId, now, providerId: wire.providerId }),
    ...(systemPrompt ? { systemPrompt } : {}),
    ...(request.tools?.length ? { tools: request.tools.map(piWireTool) } : {}),
  }
}

/** Map the neutral tool choice onto Pi's (`any` → `required`). */
function piToolChoice(toolChoice: ToolChoice): 'auto' | 'none' | 'required' {
  return toolChoice === 'any' ? 'required' : toolChoice
}

export interface PiMessagesClientOptions {
  readonly apiKey?: string
  /** Append `?debug=1` so the backend attaches routing metadata (Pi parity). */
  readonly debug?: boolean
  readonly baseUrl?: string
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly fetchImplementation?: FetchImplementation
  /** Extra static headers (gateway-specific); sent after the protocol headers. */
  readonly headers?: Readonly<Record<string, string>>
  /** Free-form provider label used in error tagging; not a registry key. */
  readonly providerName?: string
}

/**
 * Client for backends speaking the pi-messages protocol (Radius gateway and
 * compatible custom providers). The provider label is intentionally a plain
 * string: the registry entry that routes to this client lives in
 * providerRegistry.ts, and the client itself must not depend on it.
 */
export class PiMessagesClient implements LlmClient {
  private readonly apiKey: string
  private readonly baseUrl: string
  private readonly debug: boolean
  private readonly environment: Readonly<Record<string, string | undefined>>
  private readonly fetchImplementation: FetchImplementation
  private readonly providerLabel: string
  private readonly staticHeaders: Readonly<Record<string, string>> | undefined

  constructor(model: string, options: PiMessagesClientOptions = {}) {
    this.apiKey = options.apiKey?.trim() ?? ''
    this.baseUrl = options.baseUrl?.replace(/\/+$/g, '') ?? ''
    this.debug = options.debug ?? false
    this.environment = options.environment ?? process.env
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.providerLabel = options.providerName ?? 'pi-messages'
    this.staticHeaders = options.headers
    if (!this.baseUrl) {
      throw new ConfigurationError('base_url', `No base URL is configured for ${this.providerLabel}`)
    }
    if (!this.apiKey) {
      throw new ConfigurationError(
        'api_key',
        `No API key is configured for ${this.providerLabel}; it authorizes with a bearer token`,
      )
    }
    this.model = model
  }

  private readonly model: string

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    return collectLlmCompletion(this.stream(request, signal))
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const bare = request.model.includes('/') ? request.model.slice(request.model.indexOf('/') + 1) : request.model
    const url = new URL(`${this.baseUrl}/messages`)
    if (this.debug) url.searchParams.set('debug', '1')

    const context = piMessagesContext(request, { modelId: bare, providerId: this.providerLabel })
    const thinking = request.thinking
    const cacheRetention = this.environment['PI_CACHE_RETENTION'] === 'long' ? 'long' : undefined
    const options: Record<string, unknown> = {
      ...(request.temperature !== undefined ? { temperature: request.temperature } : {}),
      ...(request.maxTokens !== undefined ? { maxTokens: request.maxTokens } : {}),
      ...(thinking?.effort !== undefined && PI_THINKING_LEVELS.has(thinking.effort)
        ? { reasoning: thinking.effort }
        : {}),
      ...(cacheRetention !== undefined ? { cacheRetention } : {}),
      ...(request.sessionId !== undefined ? { sessionId: request.sessionId } : {}),
      ...(request.toolChoice !== undefined ? { toolChoice: piToolChoice(request.toolChoice) } : {}),
    }
    const payload: Record<string, unknown> = { model: bare, context, options }

    const response = await this.fetchImplementation(url, {
      method: 'POST',
      headers: {
        authorization: `Bearer ${this.apiKey}`,
        accept: 'text/event-stream',
        'content-type': 'application/json',
        ...(this.staticHeaders ?? {}),
      },
      body: JSON.stringify(payload),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      throw createResponseError(url, response, await response.text(), this.providerLabel, bare)
    }
    if (!response.body) {
      throw new ProviderError(this.providerLabel, 'response has no body')
    }

    const converter = new PiMessagesEventConverter()
    let terminal = false
    for await (const event of readPiMessagesEvents(response.body)) {
      const delta = converter.convert(event)
      if (delta !== undefined) yield delta
      if (event.type === 'done' || event.type === 'error') {
        terminal = true
        break
      }
    }
    if (!terminal) {
      throw new ProviderError(this.providerLabel, 'stream ended without a terminal event')
    }
  }
}
