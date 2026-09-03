// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { parseStreamingJson } from '@earendil-works/pi-ai'

import {
  ANTHROPIC_OAUTH_IDENTITY_PROMPT,
  anthropicOAuthHeaders,
  isAnthropicOAuthToken,
  toClaudeCodeToolName,
} from '../auth/anthropicOAuth.js'
import { ConfigurationError, ProviderError } from '../core/errors.js'
import {
  cacheableSystemPrompt,
  markLastMessageForCache,
  wrapSystemSegmentsWithCache,
  wrapToolsWithCache,
} from '../streaming/promptCaching.js'
import { deterministicToolCallId } from '../streaming/toolCallIds.js'
import type { ChatMessage, ContentPart, MessageContent } from '../types/messages.js'
import { messageText } from '../types/messages.js'
import type { JsonObject, ToolCall, ToolChoice, ToolDefinition } from '../types/toolCalls.js'
import { parseToolArguments } from '../types/toolCalls.js'
import type { CompletionRequest, FetchImplementation, LlmClient, LlmCompletion, LlmDelta, TokenUsage } from './client.js'
import { internalSseData } from './client.js'
import {
  anthropicSupportsToolReferences,
  splitDeferredTools,
  type DeferredToolSplit,
} from './deferredTools.js'
import { piCatalogModelCapabilities } from './piModelCatalog.js'
import { bareModel, getApiKey } from './providerRegistry.js'

export interface AnthropicClientOptions {
  readonly apiKey?: string
  readonly baseUrl?: string
  readonly fetchImplementation?: FetchImplementation
  /** Add Anthropic's ephemeral cache breakpoints to stable system/tool prefixes. */
  readonly promptCaching?: boolean
  readonly version?: string
  /**
   * The Xerxes provider this client serves ('anthropic', 'kimi-code',
   * 'minimax', …). Catalog lookups (thinking maps, tool-reference support)
   * are keyed by it — hardcoding 'anthropic' would miss every
   * anthropic-protocol satellite provider.
   */
  readonly providerName?: string
  /**
   * Resolve a subscription OAuth bearer token per request (Claude Pro/Max).
   * A token carrying `sk-ant-oat` switches the request to the OAuth surface:
   * `Authorization: Bearer` instead of x-api-key, the Claude Code beta flags
   * and identity, and Claude Code tool naming. Returning `undefined` keeps
   * the ordinary API-key request.
   */
  readonly resolveOAuthToken?: (signal?: AbortSignal) => Promise<string | undefined>
}

export interface AnthropicMessage {
  readonly content: AnthropicContent
  readonly role: 'assistant' | 'user'
}

export type AnthropicContent = string | readonly AnthropicContentBlock[]

export type AnthropicContentBlock =
  | { readonly text: string; readonly type: 'text' }
  | { readonly source: { readonly data: string; readonly media_type: string; readonly type: 'base64' }; readonly type: 'image' }
  | { readonly input: JsonObject; readonly id: string; readonly name: string; readonly type: 'tool_use' }
  | { readonly content: string; readonly is_error?: boolean; readonly tool_use_id: string; readonly type: 'tool_result' }
  | { readonly content: readonly { readonly tool_name: string; readonly type: 'tool_reference' }[]; readonly is_error?: boolean; readonly tool_use_id: string; readonly type: 'tool_result' }
  | { readonly data: string; readonly type: 'redacted_thinking' }
  | { readonly signature: string; readonly thinking: string; readonly type: 'thinking' }

export interface AnthropicMessagePayload {
  readonly messages: readonly AnthropicMessage[]
  readonly system?: string
}

/** Options controlling how neutral messages are converted for one Anthropic request. */
export interface AnthropicConversionOptions {
  /**
   * Names of tools still deferred for this request. A tool result that loaded
   * one of them replays as `tool_reference` blocks with its content displaced
   * to sibling text blocks, per pi-ai's tool-reference mode.
   */
  readonly deferredToolNames?: ReadonlySet<string>
  /**
   * Claude Code stealth mode (pi-ai): rewrite tool names to their canonical
   * Claude Code casing on `tool_use`, `tool_reference`, and tool definitions.
   * Identity by default; the OAuth subscription surface sets this.
   */
  readonly normalizeToolName?: (name: string) => string
  /**
   * Re-emit signed thinking blocks only when the current request enables
   * extended thinking; replaying them otherwise is a provider-side rejection.
   */
  readonly thinkingEnabled?: boolean
}

/** Convert neutral Xerxes messages to Anthropic's content-block protocol. */
export function messagesToAnthropic(
  messages: readonly ChatMessage[],
  options: AnthropicConversionOptions = {},
): AnthropicMessagePayload {
  const thinkingEnabled = options.thinkingEnabled ?? false
  const deferredToolNames = options.deferredToolNames ?? new Set<string>()
  const normalizeToolName = options.normalizeToolName ?? ((name: string): string => name)
  // Each deferred schema is introduced once, at the result that loaded it.
  const loadedToolNames = new Set<string>()
  const converted: AnthropicMessage[] = []
  const systems: string[] = []
  const unresolvedToolCalls = new Set<string>()
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
    if (message.role !== 'tool' && unresolvedToolCalls.size) {
      converted.push({
        role: 'user',
        content: [...unresolvedToolCalls].map(toolUseId => ({
          type: 'tool_result' as const,
          tool_use_id: toolUseId,
          content: 'No result provided',
          is_error: true,
        })),
      })
      unresolvedToolCalls.clear()
    }
    if (message.role === 'user') {
      const content = anthropicUserContent(message.content)
      if ((typeof content === 'string' && content.trim()) || (Array.isArray(content) && content.length)) {
        converted.push({ role: 'user', content })
      }
      index += 1
      continue
    }
    if (message.role === 'assistant') {
      const blocks: AnthropicContentBlock[] = []
      const redactedThinking = anthropicRedactedThinking(message.thinking_signature)
      if (thinkingEnabled && redactedThinking) {
        blocks.push(redactedThinking)
      } else if (thinkingEnabled && message.thinking && message.thinking_signature) {
        blocks.push({ type: 'thinking', thinking: message.thinking, signature: message.thinking_signature })
      }
      if (messageText(message)) {
        blocks.push({ type: 'text', text: messageText(message) })
      }
      for (const call of message.tool_calls ?? []) {
        blocks.push({ type: 'tool_use', id: call.id, name: normalizeToolName(call.function.name), input: call.function.arguments })
        unresolvedToolCalls.add(call.id)
      }
      // Anthropic rejects an assistant turn with an empty content array, so a
      // message whose only block was a non-replayable thinking trace is
      // dropped rather than sent as `content: []`.
      if (blocks.length) {
        converted.push({ role: 'assistant', content: blocks })
      }
      index += 1
      continue
    }

    const toolResults: AnthropicContentBlock[] = []
    // Anthropic rejects tool references mixed with ordinary tool-result
    // content, so reference-bearing results displace their text into sibling
    // blocks that follow every tool_result block of the same user message.
    const displacedContent: AnthropicContentBlock[] = []
    while (index < messages.length && messages[index]?.role === 'tool') {
      const toolMessage = messages[index]
      if (toolMessage?.role === 'tool') {
        const references = (toolMessage.added_tool_names ?? []).filter(
          name => deferredToolNames.has(normalizeToolName(name)) && !loadedToolNames.has(normalizeToolName(name)),
        )
        for (const name of references) loadedToolNames.add(normalizeToolName(name))
        const content = toolMessage.content || '(no tool output)'
        toolResults.push(references.length
          ? {
            type: 'tool_result',
            tool_use_id: toolMessage.tool_call_id,
            content: references.map(name => ({ type: 'tool_reference' as const, tool_name: normalizeToolName(name) })),
            ...(toolMessage.is_error ? { is_error: true } : {}),
          }
          : {
            type: 'tool_result',
            tool_use_id: toolMessage.tool_call_id,
            content,
            ...(toolMessage.is_error ? { is_error: true } : {}),
          })
        if (references.length) displacedContent.push({ type: 'text', text: content })
        unresolvedToolCalls.delete(toolMessage.tool_call_id)
      }
      index += 1
    }
    for (const toolUseId of unresolvedToolCalls) {
      toolResults.push({ type: 'tool_result', tool_use_id: toolUseId, content: 'No result provided', is_error: true })
    }
    unresolvedToolCalls.clear()
    if (toolResults.length || displacedContent.length) {
      converted.push({ role: 'user', content: [...toolResults, ...displacedContent] })
    }
  }
  if (unresolvedToolCalls.size) {
    converted.push({
      role: 'user',
      content: [...unresolvedToolCalls].map(toolUseId => ({
        type: 'tool_result' as const,
        tool_use_id: toolUseId,
        content: 'No result provided',
        is_error: true,
      })),
    })
  }
  return {
    messages: converted,
    ...(systems.filter(Boolean).length ? { system: systems.filter(Boolean).join('\n\n') } : {}),
  }
}

function anthropicRedactedThinking(signature: string | undefined): AnthropicContentBlock | undefined {
  if (!signature) return undefined
  try {
    const parsed: unknown = JSON.parse(signature)
    const record = typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : {}
    return record.type === 'redacted_thinking' && typeof record.data === 'string'
      ? { type: 'redacted_thinking', data: record.data }
      : undefined
  } catch {
    return undefined
  }
}

/** Native-fetch adapter for Anthropic's Messages API and SSE stream. */
export class AnthropicMessagesClient implements LlmClient {
  private readonly apiKey: string
  private readonly baseUrl: string
  private readonly fetchImplementation: FetchImplementation
  private readonly promptCaching: boolean
  private readonly version: string
  private readonly providerName: string
  private readonly resolveOAuthToken:
    | ((signal?: AbortSignal) => Promise<string | undefined>)
    | undefined

  constructor(options: AnthropicClientOptions = {}) {
    this.apiKey = options.apiKey ?? getApiKey('anthropic')
    this.baseUrl = options.baseUrl ?? 'https://api.anthropic.com'
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.promptCaching = options.promptCaching ?? true
    this.version = options.version ?? '2023-06-01'
    this.providerName = options.providerName ?? 'anthropic'
    this.resolveOAuthToken = options.resolveOAuthToken
    // A subscription session can authorize requests without an API key, so
    // the missing-key error waits until a request actually lacks both.
    if (!this.apiKey && !this.resolveOAuthToken) {
      throw new ConfigurationError('ANTHROPIC_API_KEY', 'Anthropic API key not provided')
    }
  }

  /**
   * Per-request auth context: a resolved subscription token and whether it
   * takes the OAuth surface (`sk-ant-oat`, pi-ai's `isOAuthToken`).
   */
  private async resolveOAuthContext(signal?: AbortSignal): Promise<{
    readonly isOAuthToken: boolean
    readonly token?: string
  }> {
    const token = (await this.resolveOAuthToken?.(signal))?.trim()
    if (!token) return { isOAuthToken: false }
    return { isOAuthToken: isAnthropicOAuthToken(token), ...(token ? { token } : {}) }
  }

  /** Request headers for the resolved auth context. */
  private requestHeaders(accept: string, oauth: { readonly isOAuthToken: boolean; readonly token?: string }): Record<string, string> {
    if (oauth.token) {
      return {
        Accept: accept,
        'Content-Type': 'application/json',
        'anthropic-version': this.version,
        ...anthropicOAuthHeaders(oauth.token, oauth.isOAuthToken),
      }
    }
    if (!this.apiKey) {
      throw new ConfigurationError(
        'ANTHROPIC_API_KEY',
        "Anthropic API key not provided and no subscription session found ('xerxes auth login anthropic')",
      )
    }
    return anthropicHeaders(this.apiKey, this.version, accept)
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    const oauth = await this.resolveOAuthContext(signal)
    const placement = anthropicDeferredPlacement(request, this.providerName)
    const converted = messagesToAnthropic(request.messages, {
      thinkingEnabled: planAnthropicThinking(request, this.providerName).requested,
      ...(oauth.isOAuthToken ? { normalizeToolName: toClaudeCodeToolName } : {}),
      ...(placement.enabled && placement.split.deferred.size
        ? { deferredToolNames: new Set(placement.split.deferred.keys()) }
        : {}),
    })
    if (!converted.messages.length) {
      throw new ConfigurationError('messages', 'Anthropic requires at least one user or assistant message')
    }

    const response = await this.fetchImplementation(new URL('v1/messages', withTrailingSlash(this.baseUrl)), {
      method: 'POST',
      headers: this.requestHeaders('application/json', oauth),
      body: JSON.stringify(anthropicRequestPayload(request, converted, this.promptCaching, false, placement, oauth.isOAuthToken, this.providerName)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw anthropicHttpError(`completion request failed (${response.status}): ${body.slice(0, 4_096)}`, response)
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
      if (type === 'redacted_thinking') {
        const data = stringAt(block, 'data')
        if (data) thinkingSignature = JSON.stringify({ type: 'redacted_thinking', data })
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

    const finishReason = anthropicFinishReason(
      stringAt(completion, 'stop_reason'),
      stringAt(asRecord(completion.stop_details), 'explanation'),
    ) || undefined
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
    const oauth = await this.resolveOAuthContext(signal)
    const placement = anthropicDeferredPlacement(request, this.providerName)
    const converted = messagesToAnthropic(request.messages, {
      thinkingEnabled: planAnthropicThinking(request, this.providerName).requested,
      ...(oauth.isOAuthToken ? { normalizeToolName: toClaudeCodeToolName } : {}),
      ...(placement.enabled && placement.split.deferred.size
        ? { deferredToolNames: new Set(placement.split.deferred.keys()) }
        : {}),
    })
    if (!converted.messages.length) {
      throw new ConfigurationError('messages', 'Anthropic requires at least one user or assistant message')
    }

    const response = await this.fetchImplementation(new URL('v1/messages', withTrailingSlash(this.baseUrl)), {
      method: 'POST',
      headers: this.requestHeaders('text/event-stream', oauth),
      body: JSON.stringify(anthropicRequestPayload(request, converted, this.promptCaching, true, placement, oauth.isOAuthToken, this.providerName)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw anthropicHttpError(`stream request failed (${response.status}): ${body.slice(0, 4_096)}`, response)
    }
    if (!response.body) {
      throw new ProviderError('anthropic', 'stream request returned no response body')
    }

    const pendingToolCalls = new Map<number, PendingToolCall>()
    const streamUsage: AnthropicStreamUsage = {}
    let receivedMessageStop = false
    for await (const data of internalSseData(response.body)) {
      if (data === '[DONE]') {
        break
      }
      const event = parseEvent(data)
      const type = stringAt(event, 'type')
      const eventUsage = trackAnthropicUsage(streamUsage, event)
      if (type === 'error') {
        const error = asRecord(event.error)
        const errorType = stringAt(error, 'type')
        const message = stringAt(error, 'message')
        throw new ProviderError(
          'anthropic',
          `stream returned API error${errorType ? ` (${errorType})` : ''}: ${message || 'unknown error'}`,
        )
      }
      if (receivedMessageStop) {
        // Retain accounting metadata, but never accept text, thinking, finish
        // changes, or tool mutations after the provider's terminal event.
        if (eventUsage) yield { usage: eventUsage }
        continue
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
          const snapshot = isJsonObject(block.input) && Object.keys(block.input).length
            ? JSON.stringify(block.input)
            : ''
          pendingToolCalls.set(index, {
            id: stringAt(block, 'id') || undefined,
            name: stringAt(block, 'name'),
            arguments: snapshot,
            // The start block's `input` snapshot is only a fallback for
            // providers that never stream `input_json_delta` partials. Once a
            // partial arrives it carries the full argument text itself, so the
            // snapshot seed must be discarded rather than appended to.
            seededFromSnapshot: snapshot !== '',
          })
        }
        if (blockType === 'thinking') {
          const signature = stringAt(block, 'signature')
          if (signature) yield { thinkingSignature: signature }
        } else if (blockType === 'redacted_thinking') {
          const data = stringAt(block, 'data')
          if (data) yield { thinkingSignature: JSON.stringify({ type: 'redacted_thinking', data }) }
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
            if (current.seededFromSnapshot) {
              current.arguments = partial
              current.seededFromSnapshot = false
            } else {
              current.arguments += partial
            }
          }
        }
        if (eventUsage) {
          yield { usage: eventUsage }
        }
        continue
      }
      if (type === 'message_delta') {
        const messageDelta = asRecord(event.delta)
        const stopReason = anthropicFinishReason(
          stringAt(messageDelta, 'stop_reason'),
          stringAt(asRecord(messageDelta.stop_details), 'explanation'),
        )
        if (stopReason || eventUsage) {
          yield {
            ...(stopReason ? { finishReason: stopReason } : {}),
            ...(eventUsage ? { usage: eventUsage } : {}),
          }
        }
        continue
      }
      if (type === 'message_stop') {
        receivedMessageStop = true
        if (pendingToolCalls.size) {
          yield { toolCalls: completeToolCalls(pendingToolCalls) }
        }
        if (eventUsage) {
          yield { usage: eventUsage }
        }
      }
    }
    if (!receivedMessageStop) {
      throw new ProviderError('anthropic', 'stream ended before message_stop')
    }
  }
}

/** Preserve HTTP retry metadata without exposing provider headers in the error message. */
function anthropicHttpError(message: string, response: Response): ProviderError {
  const retryAfterMilliseconds = parseRetryAfterHeader(response.headers.get('retry-after-ms'))
  const retryAfterSeconds = retryAfterMilliseconds === undefined
    ? parseRetryAfterHeader(response.headers.get('retry-after'))
    : retryAfterMilliseconds / 1_000
  return new ProviderError('anthropic', message, undefined, {
    status: response.status,
    ...(retryAfterSeconds === undefined ? {} : { retryAfterSeconds }),
  })
}

/** Parse Retry-After delta-seconds or an HTTP date into a non-negative delay. */
function parseRetryAfterHeader(value: string | null, now = Date.now()): number | undefined {
  if (value === null) return undefined
  const normalized = value.trim()
  if (/^\d+(?:\.\d+)?$/.test(normalized)) {
    const seconds = Number(normalized)
    return Number.isFinite(seconds) ? seconds : undefined
  }
  const retryAt = Date.parse(normalized)
  if (!Number.isFinite(retryAt)) return undefined
  return Math.max(0, Math.ceil((retryAt - now) / 1_000))
}

interface PendingToolCall {
  arguments: string
  readonly id: string | undefined
  readonly name: string
  /** True while `arguments` holds the start-block input snapshot instead of streamed partials. */
  seededFromSnapshot?: boolean
}

interface AnthropicDeferredPlacement {
  readonly enabled: boolean
  readonly split: DeferredToolSplit
}

/**
 * pi-ai's tool-reference placement: first-party Claude 4.5+ (non-Haiku) splits
 * transcript-loaded tools out of the immediate surface.
 */
function anthropicDeferredPlacement(request: CompletionRequest, provider: string): AnthropicDeferredPlacement {
  const enabled = anthropicSupportsToolReferences(request.model, provider)
  return { enabled, split: splitDeferredTools(request.tools, request.messages, enabled) }
}

/**
 * How thinking lands on the wire for this request (pi-ai anthropic-messages
 * parity).
 *
 * - `requested` — the caller asked for thinking. `off`/`none` count as *not*
 *   requested: before this, any defined `thinking` object was treated as on,
 *   which is how an explicit "off" still produced a thinking model.
 * - adaptive models (compat.forceAdaptiveThinking, e.g. Kimi K3) take
 *   `{ type: 'adaptive' }` plus `output_config.effort`; budget models take
 *   `{ type: 'enabled', budget_tokens }`.
 * - A model whose thinking map marks `off: null` cannot disable thinking;
 *   sending `{ type: 'disabled' }` would be a provider-side rejection, so it
 *   is omitted (pi-ai guards the same way).
 */
interface AnthropicThinkingPlan {
  readonly adaptive: boolean
  readonly canDisable: boolean
  readonly effort: string | undefined
  readonly reasoningModel: boolean
  readonly requested: boolean
  readonly budgetTokens: number | undefined
}

function planAnthropicThinking(request: CompletionRequest, provider: string): AnthropicThinkingPlan {
  const effortRaw = request.thinking?.effort?.trim().toLowerCase()
  const requested = request.thinking !== undefined && effortRaw !== 'off' && effortRaw !== 'none'
  const capabilities = piCatalogModelCapabilities(request.model, provider)
  const map = capabilities?.thinkingLevelMap
  const adaptive = capabilities?.compat?.forceAdaptiveThinking === true
  const mapped = effortRaw ? map?.[effortRaw] : undefined
  // pi mapThinkingLevelToEffort: catalog mapping wins, unknowns land on high.
  const effort = !adaptive
    ? undefined
    : typeof mapped === 'string'
      ? mapped
      : effortRaw === 'minimal' || effortRaw === 'low'
        ? 'low'
        : effortRaw === 'medium'
          ? 'medium'
          : 'high'
  return {
    adaptive,
    canDisable: map?.off !== null,
    effort,
    reasoningModel: capabilities?.reasoning ?? true,
    requested,
    budgetTokens: request.thinking === undefined
      ? undefined
      : request.thinking.budgetTokens ?? 10_000,
  }
}

function anthropicRequestPayload(
  request: CompletionRequest,
  converted: AnthropicMessagePayload,
  promptCaching: boolean,
  stream: boolean,
  placement: AnthropicDeferredPlacement,
  isOAuthToken = false,
  provider = 'anthropic',
): Record<string, unknown> {
  const plan = planAnthropicThinking(request, provider)
  // Budget-based thinking raises max_tokens past the budget; adaptive models
  // carry no budget, and an off request must not inflate the cap either.
  const thinkingBudget = plan.requested && !plan.adaptive ? plan.budgetTokens : undefined
  const payload: Record<string, unknown> = {
    model: bareModel(request.model),
    // Extended thinking rejects budget_tokens >= max_tokens. An unconfigured
    // or undersized max_tokens is raised to budget + reply headroom so a
    // thinking escalation can never produce an invalid request.
    max_tokens: thinkingBudget === undefined
      ? request.maxTokens ?? 2048
      : Math.max(request.maxTokens ?? 0, thinkingBudget + 4_096),
    // Cache the transcript too, not just the prelude. The system prompt and
    // tool schemas are the part that does not grow; the conversation is the
    // part that does, and it was being re-sent at full price every request.
    messages: promptCaching
      ? markLastMessageForCache(converted.messages as never) as never
      : converted.messages,
    stream,
  }
  if (converted.system) {
    // Prefer the segmented form so the cache breakpoint lands after the stable
    // sources rather than after the memory section, which the agent rewrites on
    // most substantive turns and which would otherwise invalidate the whole
    // prefix every request.
    //
    // Only when the segments reproduce the converted system text exactly: the
    // conversion also folds in any other system message in the transcript, and
    // silently dropping one to gain a cache hit would be a bad trade.
    const segments = request.systemSegments
    const segmentText = segments?.map(segment => segment.text).filter(Boolean).join('\n\n')
    payload.system = !promptCaching
      ? converted.system
      : segments?.length && segmentText === converted.system
        ? wrapSystemSegmentsWithCache(segments)
        : cacheableSystemPrompt(converted.system)
  }
  if (isOAuthToken) {
    // pi-ai: the subscription endpoint REQUIRES the Claude Code identity as
    // the first system block, ahead of the caller's own system prompt.
    const identity = { type: 'text', text: ANTHROPIC_OAUTH_IDENTITY_PROMPT }
    const current = payload.system
    payload.system = typeof current === 'string'
      ? [identity, { type: 'text', text: current }]
      : Array.isArray(current)
        ? [identity, ...current]
        : [identity]
  }
  if (request.temperature !== undefined && (!plan.requested || request.temperature === 1)) {
    // Extended thinking requires temperature exactly 1; any other value is a
    // provider-side rejection, so it is omitted rather than sent.
    payload.temperature = request.temperature
  }
  if (request.topP !== undefined && !plan.requested) {
    // top_p sampling is likewise incompatible with extended thinking.
    payload.top_p = request.topP
  }
  if (request.stop?.length) {
    payload.stop_sequences = request.stop
  }
  if (plan.reasoningModel) {
    if (plan.requested) {
      if (plan.adaptive) {
        // Adaptive thinking (Kimi K3, adaptive Claude): the model decides how
        // much to think; the caller only nudges via output_config.effort.
        // pi-ai sends display: 'summarized' as the default view hint.
        payload.thinking = { type: 'adaptive', display: 'summarized' }
        if (plan.effort) {
          payload.output_config = { effort: plan.effort }
        }
      } else {
        // WHY budget_tokens: Anthropic extended thinking is budget-based, not
        // effort-based — the wire contract is { type: 'enabled', budget_tokens },
        // so the neutral ThinkingRequest's effort hint has no Anthropic meaning
        // and is intentionally not translated. The 10_000 fallback mirrors the
        // session-default budget in runtime/thinkingLevels.ts so an effort-only
        // directive still produces a valid budget.
        payload.thinking = {
          type: 'enabled',
          budget_tokens: thinkingBudget ?? 10_000,
        }
      }
    } else if (plan.canDisable) {
      // Explicit off: omitting the field leaves the server default, which for
      // always-thinking models is ON — "off" must be sent to mean off. Models
      // whose map marks off: null cannot disable; the field stays out.
      payload.thinking = { type: 'disabled' }
    }
  }
  if (request.tools?.length) {
    const normalize = isOAuthToken ? toClaudeCodeToolName : ((name: string): string => name)
    let wireTools = request.tools.map(tool => toolToAnthropic(tool, false, normalize))
    if (placement.enabled) {
      let immediate = [...placement.split.immediate]
      let deferred = [...placement.split.deferred.values()]
      // A request whose whole surface is deferred would send no callable
      // tools at all; pi-ai promotes everything to immediate in that corner.
      if (immediate.length === 0 && deferred.length > 0) {
        immediate = deferred
        deferred = []
      }
      wireTools = [
        ...immediate.map(tool => toolToAnthropic(tool, false, normalize)),
        ...deferred.map(tool => toolToAnthropic(tool, true, normalize)),
      ]
    }
    payload.tools = promptCaching ? wrapToolsWithCache(wireTools) : wireTools
    const choice = anthropicToolChoice(request.toolChoice)
    if (choice) {
      payload.tool_choice = choice
    }
  }
  if (request.extraBody) Object.assign(payload, request.extraBody)
  return payload
}

function anthropicHeaders(apiKey: string, version: string, accept: string): Record<string, string> {
  return {
    Accept: accept,
    'Content-Type': 'application/json',
    'User-Agent': 'xerxes-agents/0.4.0',
    'anthropic-version': version,
    'x-api-key': apiKey,
  }
}

function anthropicUserContent(content: MessageContent): AnthropicContent {
  if (typeof content === 'string') {
    return content
  }
  const blocks = content.flatMap(part => anthropicContentPart(part))
  return blocks.some(block => block.type === 'text' && block.text.trim())
    ? blocks
    : [...blocks, { type: 'text', text: '(see attached image)' }]
}

function anthropicContentPart(part: ContentPart): AnthropicContentBlock[] {
  if (part.type === 'text') {
    return part.text ? [{ type: 'text', text: part.text }] : []
  }
  const match = /^data:([^;,]+);base64,([A-Za-z0-9+/=\r\n]+)$/.exec(part.image_url.url)
  if (match?.[1] && match[2]) {
    return [{
      type: 'image',
      source: { type: 'base64', media_type: match[1], data: match[2].replaceAll(/\s/g, '') },
    }]
  }
  // Anthropic accepts base64 image sources, not arbitrary remote URLs.
  return [{ type: 'text', text: `[Image: ${part.image_url.url}]` }]
}

function toolToAnthropic(
  tool: ToolDefinition,
  deferLoading = false,
  normalizeToolName: (name: string) => string = (name): string => name,
): Record<string, unknown> {
  return {
    name: normalizeToolName(tool.function.name),
    description: tool.function.description,
    input_schema: tool.function.parameters,
    ...(deferLoading ? { defer_loading: true } : {}),
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
function anthropicFinishReason(stopReason: string, explanation = ''): string {
  if (stopReason === 'end_turn' || stopReason === 'stop_sequence') {
    return 'stop'
  }
  if (stopReason === 'max_tokens') {
    return 'length'
  }
  if (stopReason === 'tool_use') {
    return 'tool_calls'
  }
  if (stopReason === 'pause_turn') return 'stop'
  if (stopReason === 'refusal' || stopReason === 'sensitive') {
    throw new ProviderError('anthropic', explanation || `provider stopped with: ${stopReason}`)
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
      const partial = parseStreamingJson(call.arguments)
      const arguments_ = isJsonObject(partial) ? partial : parseToolArguments(call.arguments)
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
  const reasoningTokens = numberAt(asRecord(messageUsage.output_tokens_details), 'thinking_tokens')
    ?? numberAt(asRecord(deltaUsage.output_tokens_details), 'thinking_tokens')
  return {
    inputTokens: inputTokens ?? 0,
    outputTokens: outputTokens ?? 0,
    ...(cacheReadTokens === undefined ? {} : { cacheReadTokens }),
    ...(cacheCreationTokens === undefined ? {} : { cacheCreationTokens }),
    ...(reasoningTokens === undefined ? {} : { reasoningTokens }),
  }
}

/** Last-known usage counters for one stream; every field stays absent until reported. */
interface AnthropicStreamUsage {
  cacheCreationTokens?: number
  cacheReadTokens?: number
  inputTokens?: number
  outputTokens?: number
  reasoningTokens?: number
}

/**
 * Fold one SSE event's usage fragment into the running stream totals and
 * return a cumulative snapshot, or `undefined` when the event reports nothing.
 *
 * Anthropic's `message_delta` events carry only `output_tokens`, so each
 * per-event snapshot must be merged with the `message_start` totals instead of
 * fabricating `inputTokens: 0` — a fabricated zero otherwise overwrites the
 * real prompt-token count downstream.
 */
function trackAnthropicUsage(
  accumulator: AnthropicStreamUsage,
  event: Record<string, unknown>,
): TokenUsage | undefined {
  const messageUsage = asRecord(asRecord(event.message).usage)
  const deltaUsage = asRecord(event.usage)
  const inputTokens = numberAt(messageUsage, 'input_tokens') ?? numberAt(deltaUsage, 'input_tokens')
  const outputTokens = numberAt(messageUsage, 'output_tokens') ?? numberAt(deltaUsage, 'output_tokens')
  const cacheReadTokens = numberAt(messageUsage, 'cache_read_input_tokens') ?? numberAt(deltaUsage, 'cache_read_input_tokens')
  const cacheCreationTokens = numberAt(messageUsage, 'cache_creation_input_tokens')
    ?? numberAt(deltaUsage, 'cache_creation_input_tokens')
  const reasoningTokens = numberAt(asRecord(messageUsage.output_tokens_details), 'thinking_tokens')
    ?? numberAt(asRecord(deltaUsage.output_tokens_details), 'thinking_tokens')
  if (
    inputTokens === undefined
    && outputTokens === undefined
    && cacheReadTokens === undefined
    && cacheCreationTokens === undefined
    && reasoningTokens === undefined
  ) {
    return undefined
  }
  if (inputTokens !== undefined) {
    accumulator.inputTokens = inputTokens
  }
  if (outputTokens !== undefined) {
    accumulator.outputTokens = outputTokens
  }
  if (cacheReadTokens !== undefined) {
    accumulator.cacheReadTokens = cacheReadTokens
  }
  if (cacheCreationTokens !== undefined) {
    accumulator.cacheCreationTokens = cacheCreationTokens
  }
  if (reasoningTokens !== undefined) {
    accumulator.reasoningTokens = reasoningTokens
  }
  return {
    inputTokens: accumulator.inputTokens ?? 0,
    outputTokens: accumulator.outputTokens ?? 0,
    ...(accumulator.cacheReadTokens === undefined ? {} : { cacheReadTokens: accumulator.cacheReadTokens }),
    ...(accumulator.cacheCreationTokens === undefined ? {} : { cacheCreationTokens: accumulator.cacheCreationTokens }),
    ...(accumulator.reasoningTokens === undefined ? {} : { reasoningTokens: accumulator.reasoningTokens }),
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
