// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError, ProviderError } from '../core/errors.js'
import { wrapSystemWithCache, wrapToolsWithCache } from '../streaming/promptCaching.js'
import { deterministicToolCallId } from '../streaming/toolCallIds.js'
import type { ChatMessage, ContentPart, MessageContent } from '../types/messages.js'
import { messageText } from '../types/messages.js'
import type { JsonObject, ToolCall, ToolChoice, ToolDefinition } from '../types/toolCalls.js'
import { parseToolArguments } from '../types/toolCalls.js'
import type { CompletionRequest, FetchImplementation, LlmClient, LlmCompletion, LlmDelta, TokenUsage } from './client.js'
import { internalSseData } from './client.js'
import { bareModel, getApiKey } from './providerRegistry.js'

export interface AnthropicClientOptions {
  readonly apiKey?: string
  readonly baseUrl?: string
  readonly fetchImplementation?: FetchImplementation
  /** Add Anthropic's ephemeral cache breakpoints to stable system/tool prefixes. */
  readonly promptCaching?: boolean
  readonly version?: string
}

export interface AnthropicMessage {
  readonly content: AnthropicContent
  readonly role: 'assistant' | 'user'
}

export type AnthropicContent = string | readonly AnthropicContentBlock[]

export type AnthropicContentBlock =
  | { readonly text: string; readonly type: 'text' }
  | { readonly input: JsonObject; readonly id: string; readonly name: string; readonly type: 'tool_use' }
  | { readonly content: string; readonly is_error?: boolean; readonly tool_use_id: string; readonly type: 'tool_result' }
  | { readonly signature: string; readonly thinking: string; readonly type: 'thinking' }

export interface AnthropicMessagePayload {
  readonly messages: readonly AnthropicMessage[]
  readonly system?: string
}

/** Convert neutral Xerxes messages to Anthropic's content-block protocol. */
export function messagesToAnthropic(messages: readonly ChatMessage[]): AnthropicMessagePayload {
  const converted: AnthropicMessage[] = []
  const systems: string[] = []
  let index = 0
  while (index < messages.length) {
    const message = messages[index]
    if (!message) {
      break
    }
    if (message.role === 'system') {
      systems.push(messageText(message))
      index += 1
      continue
    }
    if (message.role === 'user') {
      converted.push({ role: 'user', content: anthropicUserContent(message.content) })
      index += 1
      continue
    }
    if (message.role === 'assistant') {
      const blocks: AnthropicContentBlock[] = []
      if (message.thinking && message.thinking_signature) {
        blocks.push({ type: 'thinking', thinking: message.thinking, signature: message.thinking_signature })
      }
      if (messageText(message)) {
        blocks.push({ type: 'text', text: messageText(message) })
      }
      for (const call of message.tool_calls ?? []) {
        blocks.push({ type: 'tool_use', id: call.id, name: call.function.name, input: call.function.arguments })
      }
      converted.push({ role: 'assistant', content: blocks })
      index += 1
      continue
    }

    const toolResults: AnthropicContentBlock[] = []
    while (index < messages.length && messages[index]?.role === 'tool') {
      const toolMessage = messages[index]
      if (toolMessage?.role === 'tool') {
        toolResults.push({
          type: 'tool_result',
          tool_use_id: toolMessage.tool_call_id,
          content: toolMessage.content,
          ...(toolMessage.is_error ? { is_error: true } : {}),
        })
      }
      index += 1
    }
    if (toolResults.length) {
      converted.push({ role: 'user', content: toolResults })
    }
  }
  return {
    messages: converted,
    ...(systems.filter(Boolean).length ? { system: systems.filter(Boolean).join('\n\n') } : {}),
  }
}

/** Native-fetch adapter for Anthropic's Messages API and SSE stream. */
export class AnthropicMessagesClient implements LlmClient {
  private readonly apiKey: string
  private readonly baseUrl: string
  private readonly fetchImplementation: FetchImplementation
  private readonly promptCaching: boolean
  private readonly version: string

  constructor(options: AnthropicClientOptions = {}) {
    this.apiKey = options.apiKey ?? getApiKey('anthropic')
    this.baseUrl = options.baseUrl ?? 'https://api.anthropic.com'
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.promptCaching = options.promptCaching ?? true
    this.version = options.version ?? '2023-06-01'
    if (!this.apiKey) {
      throw new ConfigurationError('ANTHROPIC_API_KEY', 'Anthropic API key not provided')
    }
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    const converted = messagesToAnthropic(request.messages)
    if (!converted.messages.length) {
      throw new ConfigurationError('messages', 'Anthropic requires at least one user or assistant message')
    }

    const response = await this.fetchImplementation(new URL('v1/messages', withTrailingSlash(this.baseUrl)), {
      method: 'POST',
      headers: anthropicHeaders(this.apiKey, this.version, 'application/json'),
      body: JSON.stringify(anthropicRequestPayload(request, converted, this.promptCaching, false)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw new ProviderError('anthropic', `completion request failed (${response.status}): ${body.slice(0, 4_096)}`)
    }

    const completion = parseEvent(await response.text())
    const rawContent = completion.content
    if (!Array.isArray(rawContent)) {
      throw new ProviderError('anthropic', 'completion response content must be an array')
    }

    const content: string[] = []
    const thinking: string[] = []
    const toolCalls = new Map<number, PendingToolCall>()
    let thinkingSignature: string | undefined
    for (const [index, rawBlock] of rawContent.entries()) {
      const block = asRecord(rawBlock)
      const type = stringAt(block, 'type')
      if (type === 'text') {
        const text = stringAt(block, 'text')
        if (text) content.push(text)
        continue
      }
      if (type === 'thinking') {
        const value = stringAt(block, 'thinking')
        if (value) thinking.push(value)
        const signature = stringAt(block, 'signature')
        if (signature) thinkingSignature = signature
        continue
      }
      if (type === 'tool_use') {
        const input = block.input
        if (input !== undefined && !isJsonObject(input)) {
          throw new ProviderError('anthropic', `tool_use block ${index} input must be an object`)
        }
        toolCalls.set(index, {
          id: stringAt(block, 'id') || undefined,
          name: stringAt(block, 'name'),
          arguments: JSON.stringify(input ?? {}),
        })
      }
    }

    const finishReason = stringAt(completion, 'stop_reason') || undefined
    const usage = anthropicUsage(completion)
    return {
      content: content.join(''),
      toolCalls: completeToolCalls(toolCalls),
      ...(finishReason === undefined ? {} : { finishReason }),
      ...(thinking.length ? { thinking: thinking.join('') } : {}),
      ...(thinkingSignature === undefined ? {} : { thinkingSignature }),
      ...(usage === undefined ? {} : { usage }),
    }
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const converted = messagesToAnthropic(request.messages)
    if (!converted.messages.length) {
      throw new ConfigurationError('messages', 'Anthropic requires at least one user or assistant message')
    }

    const response = await this.fetchImplementation(new URL('v1/messages', withTrailingSlash(this.baseUrl)), {
      method: 'POST',
      headers: anthropicHeaders(this.apiKey, this.version, 'text/event-stream'),
      body: JSON.stringify(anthropicRequestPayload(request, converted, this.promptCaching, true)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw new ProviderError('anthropic', `stream request failed (${response.status}): ${body.slice(0, 4_096)}`)
    }
    if (!response.body) {
      throw new ProviderError('anthropic', 'stream request returned no response body')
    }

    const pendingToolCalls = new Map<number, PendingToolCall>()
    let emittedToolCalls = false
    for await (const data of internalSseData(response.body)) {
      if (data === '[DONE]') {
        break
      }
      const event = parseEvent(data)
      const type = stringAt(event, 'type')
      const eventUsage = anthropicUsage(event)
      if (type === 'error') {
        const error = asRecord(event.error)
        const errorType = stringAt(error, 'type')
        const message = stringAt(error, 'message')
        throw new ProviderError(
          'anthropic',
          `stream returned API error${errorType ? ` (${errorType})` : ''}: ${message || 'unknown error'}`,
        )
      }
      if (type === 'message_start') {
        if (eventUsage) {
          yield { usage: eventUsage }
        }
        continue
      }
      if (type === 'content_block_start') {
        const block = asRecord(event.content_block)
        const blockType = stringAt(block, 'type')
        const index = numberAt(event, 'index')
        if (blockType === 'tool_use' && index !== undefined) {
          pendingToolCalls.set(index, {
            id: stringAt(block, 'id') || undefined,
            name: stringAt(block, 'name'),
            arguments: isJsonObject(block.input) && Object.keys(block.input).length ? JSON.stringify(block.input) : '',
          })
        }
        if (blockType === 'thinking') {
          const signature = stringAt(block, 'signature')
          if (signature) {
            yield { thinkingSignature: signature }
          }
        }
        if (eventUsage) {
          yield { usage: eventUsage }
        }
        continue
      }
      if (type === 'content_block_delta') {
        const delta = asRecord(event.delta)
        const deltaType = stringAt(delta, 'type')
        if (deltaType === 'text_delta' && stringAt(delta, 'text')) {
          yield { content: stringAt(delta, 'text') }
        } else if (deltaType === 'thinking_delta' && stringAt(delta, 'thinking')) {
          yield { thinking: stringAt(delta, 'thinking') }
        } else if (deltaType === 'signature_delta' && stringAt(delta, 'signature')) {
          yield { thinkingSignature: stringAt(delta, 'signature') }
        } else if (deltaType === 'input_json_delta') {
          const index = numberAt(event, 'index')
          const current = index === undefined ? undefined : pendingToolCalls.get(index)
          const partial = stringAt(delta, 'partial_json')
          if (current && partial) {
            current.arguments += partial
          }
        }
        if (eventUsage) {
          yield { usage: eventUsage }
        }
        continue
      }
      if (type === 'message_delta') {
        const stopReason = anthropicFinishReason(stringAt(asRecord(event.delta), 'stop_reason'))
        if (stopReason || eventUsage) {
          yield {
            ...(stopReason ? { finishReason: stopReason } : {}),
            ...(eventUsage ? { usage: eventUsage } : {}),
          }
        }
        continue
      }
      if (type === 'message_stop') {
        if (pendingToolCalls.size) {
          yield { toolCalls: completeToolCalls(pendingToolCalls) }
          emittedToolCalls = true
        }
        if (eventUsage) {
          yield { usage: eventUsage }
        }
      }
    }
    if (!emittedToolCalls && pendingToolCalls.size) {
      yield { toolCalls: completeToolCalls(pendingToolCalls) }
    }
  }
}

interface PendingToolCall {
  arguments: string
  readonly id: string | undefined
  readonly name: string
}

function anthropicRequestPayload(
  request: CompletionRequest,
  converted: AnthropicMessagePayload,
  promptCaching: boolean,
  stream: boolean,
): Record<string, unknown> {
  const thinkingBudget = request.thinking === undefined
    ? undefined
    : request.thinking.budgetTokens ?? 10_000
  const payload: Record<string, unknown> = {
    model: bareModel(request.model),
    // Extended thinking rejects budget_tokens >= max_tokens. An unconfigured
    // or undersized max_tokens is raised to budget + reply headroom so a
    // thinking escalation can never produce an invalid request.
    max_tokens: thinkingBudget === undefined
      ? request.maxTokens ?? 2048
      : Math.max(request.maxTokens ?? 0, thinkingBudget + 4_096),
    messages: converted.messages,
    stream,
  }
  if (converted.system) {
    payload.system = promptCaching ? wrapSystemWithCache(converted.system) : converted.system
  }
  if (request.temperature !== undefined && (thinkingBudget === undefined || request.temperature === 1)) {
    // Extended thinking requires temperature exactly 1; any other value is a
    // provider-side rejection, so it is omitted rather than sent.
    payload.temperature = request.temperature
  }
  if (request.topP !== undefined && thinkingBudget === undefined) {
    // top_p sampling is likewise incompatible with extended thinking.
    payload.top_p = request.topP
  }
  if (request.stop?.length) {
    payload.stop_sequences = request.stop
  }
  if (thinkingBudget !== undefined) {
    // WHY budget_tokens: Anthropic extended thinking is budget-based, not
    // effort-based — the wire contract is { type: 'enabled', budget_tokens },
    // so the neutral ThinkingRequest's effort hint has no Anthropic meaning
    // and is intentionally not translated. The 10_000 fallback mirrors the
    // session-default budget in runtime/thinkingLevels.ts so an effort-only
    // directive still produces a valid budget.
    payload.thinking = {
      type: 'enabled',
      budget_tokens: thinkingBudget,
    }
  }
  if (request.tools?.length) {
    const tools = request.tools.map(toolToAnthropic)
    payload.tools = promptCaching ? wrapToolsWithCache(tools) : tools
    const choice = anthropicToolChoice(request.toolChoice)
    if (choice) {
      payload.tool_choice = choice
    }
  }
  return payload
}

function anthropicHeaders(apiKey: string, version: string, accept: string): Record<string, string> {
  return {
    Accept: accept,
    'Content-Type': 'application/json',
    'anthropic-version': version,
    'x-api-key': apiKey,
  }
}

function anthropicUserContent(content: MessageContent): AnthropicContent {
  if (typeof content === 'string') {
    return content
  }
  const blocks = content.flatMap(part => anthropicContentPart(part))
  return blocks
}

function anthropicContentPart(part: ContentPart): AnthropicContentBlock[] {
  if (part.type === 'text') {
    return [{ type: 'text', text: part.text }]
  }
  // URL images need a provider download step; preserve user-visible context rather
  // than issuing an invalid Anthropic block until media transport is ported.
  return [{ type: 'text', text: `[Image: ${part.image_url.url}]` }]
}

function toolToAnthropic(tool: ToolDefinition): Record<string, unknown> {
  return {
    name: tool.function.name,
    description: tool.function.description,
    input_schema: tool.function.parameters,
  }
}

function anthropicToolChoice(choice: ToolChoice | undefined): Record<string, string> | undefined {
  if (choice === 'any') {
    return { type: 'any' }
  }
  if (choice === 'auto') {
    return { type: 'auto' }
  }
  if (choice === 'none') {
    return { type: 'none' }
  }
  return undefined
}

/** Map Anthropic stop reasons onto the neutral OpenAI-style finish vocabulary. */
function anthropicFinishReason(stopReason: string): string {
  if (stopReason === 'end_turn' || stopReason === 'stop_sequence') {
    return 'stop'
  }
  if (stopReason === 'max_tokens') {
    return 'length'
  }
  if (stopReason === 'tool_use') {
    return 'tool_calls'
  }
  return stopReason
}

function completeToolCalls(calls: Map<number, PendingToolCall>): ToolCall[] {
  return [...calls.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, call]) => {
      if (!call.name) {
        throw new ProviderError('anthropic', 'tool_use block missing a name')
      }
      const arguments_ = parseToolArguments(call.arguments)
      return {
        id: call.id ?? deterministicToolCallId(call.name, arguments_),
        type: 'function' as const,
        function: { name: call.name, arguments: arguments_ },
      }
    })
}

function anthropicUsage(event: Record<string, unknown>): TokenUsage | undefined {
  const messageUsage = asRecord(asRecord(event.message).usage)
  const deltaUsage = asRecord(event.usage)
  const inputTokens = numberAt(messageUsage, 'input_tokens') ?? numberAt(deltaUsage, 'input_tokens')
  const outputTokens = numberAt(messageUsage, 'output_tokens') ?? numberAt(deltaUsage, 'output_tokens')
  if (inputTokens === undefined && outputTokens === undefined) {
    return undefined
  }
  const cacheReadTokens = numberAt(messageUsage, 'cache_read_input_tokens') ?? numberAt(deltaUsage, 'cache_read_input_tokens')
  const cacheCreationTokens = numberAt(messageUsage, 'cache_creation_input_tokens')
    ?? numberAt(deltaUsage, 'cache_creation_input_tokens')
  return {
    inputTokens: inputTokens ?? 0,
    outputTokens: outputTokens ?? 0,
    ...(cacheReadTokens === undefined ? {} : { cacheReadTokens }),
    ...(cacheCreationTokens === undefined ? {} : { cacheCreationTokens }),
  }
}

function parseEvent(data: string): Record<string, unknown> {
  try {
    return asRecord(JSON.parse(data) as unknown)
  } catch (error) {
    throw new ProviderError('anthropic', `invalid SSE JSON: ${data.slice(0, 200)}`, error)
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function numberAt(value: Record<string, unknown>, key: string): number | undefined {
  const item = value[key]
  return typeof item === 'number' && Number.isFinite(item) ? item : undefined
}

function stringAt(value: Record<string, unknown>, key: string): string {
  const item = value[key]
  return typeof item === 'string' ? item : ''
}

function withTrailingSlash(value: string): string {
  return value.endsWith('/') ? value : `${value}/`
}
