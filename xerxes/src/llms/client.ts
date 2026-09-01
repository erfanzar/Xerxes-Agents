// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'

import { parseStreamingJson } from '@earendil-works/pi-ai'

import { codexAuthHeaders, codexBaseUrl, CodexSession } from '../auth/codexAuth.js'
import {
  COPILOT_API_BASE_DEFAULT,
  copilotApiBase,
  copilotAuthHeaders,
  copilotRequestHeaders,
  CopilotSession,
} from '../auth/copilotAuth.js'
import { AnthropicOAuthSession, isAnthropicOAuthToken } from '../auth/anthropicOAuth.js'
import { KimiCodingOAuthSession } from '../auth/kimiCodingOAuth.js'
import { OpenRouterOAuthSession } from '../auth/openrouterOAuth.js'
import { XaiOAuthSession } from '../auth/xaiOAuth.js'
import { isGradedEffort } from './reasoningLevels.js'
import { ConfigurationError, ProviderError } from '../core/errors.js'
import { isPluginLlmProviderFactory } from '../extensions/plugins.js'
import type {
  PluginLlmProviderFactory,
  PluginLlmProviderOptions,
  PluginLlmProviderRegistry,
} from '../extensions/plugins.js'
import { ResponsesEventTranslator } from '../streaming/responsesApi.js'
import {
  buildCachedWebSocketRequestBody,
  codexWebSocketFallbackActive,
  CODEX_SSE_COMPRESSION_HEADER,
  compressRequestBodyZstd,
  continuationFromResponse,
  CodexWsApiError,
  isCodexRetryableWebSocketError,
  recordCodexWebSocketFallback,
  streamCodexWebSocket,
  type CodexWsContinuation,
} from '../streaming/codexWebSocket.js'
import { deterministicToolCallId } from '../streaming/toolCallIds.js'
import { SSEParser } from '../streaming/sse.js'
import { AnthropicMessagesClient } from './anthropic.js'
import { AzureOpenAiClient } from './azureOpenAi.js'
import { PiMessagesClient } from './piMessages.js'
import { DEFAULT_RADIUS_GATEWAY } from './radiusGateway.js'
import { BedrockConverseClient } from './bedrock.js'
import { createCloudflareWorkersAiClient } from './cloudflareWorkersAi.js'
import { MistralClient } from './mistral.js'
import { GoogleVertexClient } from './vertex.js'
import { GeminiClient } from './gemini.js'
import {
  createGrammarToolInputProperties,
  grammarToolInput,
  resolveGrammar,
} from './grammarTools.js'
import {
  completionsDeferredToolsMode,
  responsesDeferredToolsMode,
  splitDeferredTools,
} from './deferredTools.js'
import { OllamaClient } from './ollama.js'
import { piCatalogModelCapabilities } from './piModelCatalog.js'
import type { ChatMessage, MessageContent, OpenAiChatMessage } from '../types/messages.js'
import { messageText, messagesToOpenAi } from '../types/messages.js'
import type { JsonObject, ToolCall, ToolChoice, ToolDefinition } from '../types/toolCalls.js'
import { isJsonObject, parseToolArguments } from '../types/toolCalls.js'
import {
  type ProviderName,
  type ProviderOverrides,
  bareModel,
  getApiKey,
  getProviderConfig,
  isProviderName,
  providerDefaultHeaders,
  providerModel,
  resolveProvider,
} from './providerRegistry.js'
import { DEFAULT_TEMPERATURE } from './samplingDefaults.js'
import type { SystemPromptSegment } from '../streaming/promptCaching.js'

export interface TokenUsage {
  readonly cacheCreationTokens?: number
  readonly cacheReadTokens?: number
  readonly inputTokens: number
  readonly outputTokens: number
  /** Provider-reported reasoning tokens; absent when the provider does not expose them. */
  readonly reasoningTokens?: number
  /**
   * The OpenAI `service_tier` the provider actually served (`response.service_tier`),
   * or the requested tier when the provider echoes none. Absent when unsupported.
   */
  readonly serviceTier?: string
}

/**
 * Why a completion was requested.
 *
 * Without this dimension every call looks like the user's own turn, so
 * housekeeping work (compaction, titling, memory extraction) is invisible in
 * cost accounting and cannot be routed to a cheaper model. `main` is the only
 * value that represents the user-facing agent loop; everything else is
 * housekeeping by definition, which is what {@link isHousekeepingQuerySource}
 * relies on so new sources never have to be added to a second list.
 */
export type QuerySource =
  | 'main'
  | 'compaction'
  | 'session_title'
  | 'memory_extraction'
  | 'tool_result_summary'
  | 'classification'
  | 'speculation'

/** The user-facing agent loop; the one query source that is not housekeeping. */
export const MAIN_QUERY_SOURCE: QuerySource = 'main'

/** Every known query source, for validating values restored from persistence. */
export const QUERY_SOURCES: readonly QuerySource[] = Object.freeze([
  'main',
  'compaction',
  'session_title',
  'memory_extraction',
  'tool_result_summary',
  'classification',
  'speculation',
])

/** Narrow an untrusted value (config, stored ledger, wire payload) to a query source. */
export function isQuerySource(value: unknown): value is QuerySource {
  return typeof value === 'string' && (QUERY_SOURCES as readonly string[]).includes(value)
}

/** True for every source except the main agent loop, i.e. spend the user did not ask for directly. */
export function isHousekeepingQuerySource(value: QuerySource): boolean {
  return value !== MAIN_QUERY_SOURCE
}

export interface CompletionRequest {
  /** Provider-specific JSON fields sent alongside the standard chat payload. */
  readonly extraBody?: Readonly<Record<string, unknown>>
  readonly frequencyPenalty?: number
  readonly maxTokens?: number
  readonly messages: readonly ChatMessage[]
  readonly minP?: number
  readonly model: string
  readonly presencePenalty?: number
  /**
   * Local routing/accounting annotation naming why the call was made. Never
   * sent on the wire: payload builders copy known fields explicitly, and a
   * stray `querySource` key would be rejected by providers that reject
   * unknown body fields. Optional so existing callers keep compiling.
   */
  readonly querySource?: QuerySource
  /**
   * Named system-prompt sources, ordered stable-first. Adapters that support
   * prefix caching place the breakpoint between the stable and volatile halves
   * instead of caching one joined string, whose most volatile byte would
   * otherwise decide the hit rate for all of it.
   */
  readonly systemSegments?: readonly SystemPromptSegment[]
  /** Provider-neutral extended-thinking request; adapters map it to their wire shape. */
  readonly thinking?: ThinkingRequest
  readonly repetitionPenalty?: number
  /**
   * OpenAI processing tier (`auto`, `default`, `flex`, `priority`). Sent as
   * `service_tier` on Responses-family payloads only — Pi's compat data shows
   * chat-completions providers reject it.
   */
  readonly serviceTier?: string
  /** Stable durable session id used for provider prompt-cache routing when supported. */
  readonly sessionId?: string
  readonly stop?: readonly string[]
  readonly temperature?: number
  readonly toolChoice?: ToolChoice
  readonly tools?: readonly ToolDefinition[]
  readonly topK?: number
  readonly topP?: number
}

/** Provider-neutral extended-thinking request resolved per turn. */
export interface ThinkingRequest {
  /** Token budget for budget-based thinking APIs (Anthropic, Zhipu). */
  readonly budgetTokens?: number
  /** Effort hint for effort-based reasoning APIs (OpenAI-compatible). */
  readonly effort?: string
}

/** Provider-neutral incremental response from a model adapter. */
export interface LlmDelta {  readonly content?: string
  readonly finishReason?: string
  readonly thinking?: string
  readonly thinkingSignature?: string
  readonly toolCalls?: readonly ToolCall[]
  readonly usage?: TokenUsage
}

/** Fully collected provider-neutral completion returned by {@link completeLlm}. */
export interface LlmCompletion {
  readonly content: string
  readonly finishReason?: string
  readonly thinking?: string
  readonly thinkingSignature?: string
  readonly toolCalls: readonly ToolCall[]
  readonly usage?: TokenUsage
}

/** Stable model metadata, equivalent to the legacy BaseLLM model summary. */
export interface LlmModelInfo {
  /** Provider-reported context capacity supplied by the caller; absent means unknown. */
  readonly maxModelLen?: number
  readonly maxTokens: number
  readonly model: string
  readonly provider: ProviderName
  readonly stream: boolean
  readonly temperature: number
}

/** Per-call settings included in a model metadata summary. */
export interface LlmModelInfoOptions {
  readonly contextLimit?: number
  readonly maxTokens?: number
  readonly stream?: boolean
  readonly temperature?: number
}

export interface LlmClient {
  /**
   * Optionally produce a fully collected response without exposing provider wire data.
   * Stream-only clients remain valid: {@link completeLlm} collects their deltas.
   */
  complete?(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion>
  /** Optional resource cleanup for SDK-backed or plugin clients. */
  close?(): Promise<void> | void
  stream(request: CompletionRequest, signal?: AbortSignal): AsyncIterable<LlmDelta>
}

export type FetchImplementation = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

export interface OpenAiCompatibleClientOptions {
  readonly apiKey?: string
  readonly baseUrl?: string
  readonly fetchImplementation?: FetchImplementation
  /** Enable Anthropic ephemeral prompt-cache breakpoints when that transport is selected. */
  readonly promptCaching?: boolean
  readonly providerName: ProviderName
  /** Use the OpenAI Responses endpoint instead of chat completions when supported by the host. */
  readonly responsesApi?: boolean
  /**
   * Resolve per-request authorization headers instead of a static API key.
   *
   * Subscription-backed providers carry a short-lived OAuth token that has to
   * be refreshed between turns, so the credential cannot be captured once at
   * construction the way an API key can.
   */
  readonly resolveAuthHeaders?: (
    signal?: AbortSignal,
    request?: CompletionRequest,
  ) => Promise<Record<string, string>>
  /**
   * Codex transport preference (pi-ai): `auto` tries WebSocket with SSE
   * fallback, `websocket-cached` adds delta continuation bodies, `websocket`
   * pins WS with full context, `sse` pins HTTP. Only consulted for
   * `openai-codex`; default `auto`.
   */
  readonly codexTransport?: 'auto' | 'sse' | 'websocket' | 'websocket-cached'
}

/** Options used by the native client factory, including optional plugin provider lookup. */
export interface LlmClientFactoryOptions extends Omit<OpenAiCompatibleClientOptions, 'providerName'> {
  /** Injected ChatGPT session; defaults to the stored one for `openai-codex`. */
  readonly codexSession?: CodexSession
  /** Injected GitHub Copilot session; defaults to the stored one for `github-copilot`. */
  readonly copilotSession?: CopilotSession
  /** Injected Anthropic subscription session; defaults to the stored one for `anthropic`. */
  readonly anthropicOAuthSession?: AnthropicOAuthSession
  /** Injected Kimi Code subscription session; defaults to the stored one for `kimi-code`. */
  readonly kimiOAuthSession?: KimiCodingOAuthSession
  /** Injected OpenRouter OAuth session; defaults to the stored one for `openrouter`. */
  readonly openrouterOAuthSession?: OpenRouterOAuthSession
  /** Injected xAI subscription session; defaults to the stored one for `xai`. */
  readonly xaiOAuthSession?: XaiOAuthSession
  readonly pluginRegistry?: PluginLlmProviderRegistry
}

interface OpenAiToolCallDelta {
  /** Grammar-constrained custom tool call fragment (pi-ai custom tools). */
  readonly custom?: {
    readonly input?: string
    readonly name?: string
  }
  readonly function?: {
    readonly arguments?: string
    readonly name?: string
  }
  readonly id?: string
  readonly index?: number
}

interface PendingToolCall {
  arguments: string
  /** Raw grammar-constrained text for OpenAI `custom` tool calls. */
  customInput?: string
  id: string | undefined
  name: string
}

/**
 * Whether this model on this provider accepts OpenAI `custom` grammar tools,
 * straight from pi-ai's compat flags in the generated catalog.
 */
function supportsOpenAiGrammarTools(providerName: ProviderName, model: string): boolean {
  return piCatalogModelCapabilities(model, providerName)?.compat?.supportsOpenAIGrammarTools === true
}

/**
 * Native-fetch OpenAI-compatible SSE client. Provider-specific stream parsers
 * feed this same neutral delta vocabulary, keeping the agent loop independent
 * of vendor JSON shapes.
 */
export class OpenAiCompatibleClient implements LlmClient {
  private readonly apiKey: string
  private readonly baseUrl: string
  private readonly fetchImplementation: FetchImplementation
  private readonly providerName: ProviderName
  private readonly resolveAuthHeaders:
    | ((signal?: AbortSignal, request?: CompletionRequest) => Promise<Record<string, string>>)
    | undefined

  constructor(options: OpenAiCompatibleClientOptions) {
    const providerConfig = getProviderConfig(options.providerName)
    this.providerName = options.providerName
    this.apiKey = options.apiKey ?? getApiKey(options.providerName)
    this.baseUrl = options.baseUrl ?? providerConfig.baseUrl ?? ''
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.resolveAuthHeaders = options.resolveAuthHeaders

    if (!this.baseUrl) {
      throw new ConfigurationError('base_url', `No base URL is configured for ${options.providerName}`)
    }
  }

  /** Static API-key headers, or freshly resolved OAuth headers when configured. */
  private async headers(
    accept: string,
    signal?: AbortSignal,
    request?: CompletionRequest,
  ): Promise<Record<string, string>> {
    const base = openAiCompatibleHeaders(this.providerName, this.apiKey, accept)
    if (!this.resolveAuthHeaders) return base
    return { ...base, ...(await this.resolveAuthHeaders(signal, request)) }
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    const endpoint = new URL('chat/completions', withTrailingSlash(this.baseUrl)).toString()
    const response = await this.fetchImplementation(endpoint, {
      method: 'POST',
      headers: await this.headers('application/json', signal, request),
      body: JSON.stringify(openAiCompatiblePayload(request, this.providerName, false)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw openAiHttpError(
        this.providerName,
        `completion request failed (${response.status}): ${body.slice(0, 4_096)}`,
        response,
      )
    }

    const responseBody = parseJsonObject(await response.text(), this.providerName)
    const choice = firstChoice(responseBody)
    if (!choice) {
      throw new ProviderError(this.providerName, 'completion response did not include a choice')
    }
    const grammarProperties = createGrammarToolInputProperties(
      request.tools,
      supportsOpenAiGrammarTools(this.providerName, request.model),
    )
    const message = asRecord(choice.message)
    const pendingToolCalls = new Map<number, PendingToolCall>()
    mergeToolDeltas(pendingToolCalls, arrayAt(message, 'tool_calls'))
    const content = openAiMessageContent(message.content)
    const thinking = stringAt(message, 'reasoning_content')
      ?? stringAt(message, 'reasoning')
      ?? stringAt(message, 'reasoning_text')
    const finishReason = validatedOpenAiFinishReason(stringAt(choice, 'finish_reason'), this.providerName)
    const usage = openAiUsage(asRecord(responseBody.usage))

    return {
      content,
      toolCalls: completedToolCalls(pendingToolCalls, grammarProperties),
      ...(finishReason === undefined ? {} : { finishReason }),
      ...(thinking === undefined ? {} : { thinking }),
      ...(usage === undefined ? {} : { usage }),
    }
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const endpoint = new URL('chat/completions', withTrailingSlash(this.baseUrl)).toString()
    const response = await this.fetchImplementation(endpoint, {
      method: 'POST',
      headers: await this.headers('text/event-stream', signal, request),
      body: JSON.stringify(openAiCompatiblePayload(request, this.providerName, true)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw openAiHttpError(
        this.providerName,
        `stream request failed (${response.status}): ${body.slice(0, 4_096)}`,
        response,
      )
    }
    if (!response.body) {
      throw new ProviderError(this.providerName, 'stream request returned no response body')
    }

    const grammarProperties = createGrammarToolInputProperties(
      request.tools,
      supportsOpenAiGrammarTools(this.providerName, request.model),
    )
    const pendingToolCalls = new Map<number, PendingToolCall>()
    let emittedToolCalls = false
    let terminal = false
    for await (const data of sseData(response.body)) {
      if (data === '[DONE]') {
        terminal = true
        break
      }
      const chunk = parseJsonObject(data, this.providerName)
      throwIfStreamError(chunk, this.providerName)
      const choice = firstChoice(chunk)
      const delta = asRecord(choice?.delta)
      const content = stringAt(delta, 'content')
      const thinking = stringAt(delta, 'reasoning_content')
        ?? stringAt(delta, 'reasoning')
        ?? stringAt(delta, 'reasoning_text')
      mergeToolDeltas(pendingToolCalls, arrayAt(delta, 'tool_calls'))
      const finishReason = validatedOpenAiFinishReason(stringAt(choice, 'finish_reason'), this.providerName)
      const usage = openAiUsage(asRecord(chunk.usage)) ?? openAiUsage(asRecord(choice?.usage))

      const event: {
        content?: string
        finishReason?: string
        thinking?: string
        toolCalls?: readonly ToolCall[]
        usage?: TokenUsage
      } = {}
      if (content) {
        event.content = content
      }
      if (thinking) {
        event.thinking = thinking
      }
      if (usage) {
        event.usage = usage
      }
      if (finishReason) {
        event.finishReason = finishReason
        terminal = true
      }
      if (finishReason && pendingToolCalls.size) {
        event.toolCalls = completedToolCalls(pendingToolCalls, grammarProperties)
        emittedToolCalls = true
      }
      if (Object.keys(event).length) {
        yield event
      }
    }

    if (!terminal) {
      throw new ProviderError(this.providerName, 'stream ended before a terminal completion event')
    }
    if (!emittedToolCalls && pendingToolCalls.size) {
      yield { toolCalls: completedToolCalls(pendingToolCalls, grammarProperties) }
    }
  }
}

/**
 * Native-fetch Responses API client using the same neutral streaming deltas
 * as chat-completions providers. It is opt-in because OpenAI-compatible
 * providers do not all expose this endpoint.
 */
export class ResponsesApiClient implements LlmClient {
  private readonly apiKey: string
  private readonly baseUrl: string
  private readonly fetchImplementation: FetchImplementation
  private readonly providerName: ProviderName
  private readonly resolveAuthHeaders:
    | ((signal?: AbortSignal, request?: CompletionRequest) => Promise<Record<string, string>>)
    | undefined
  private readonly codexTransport: 'auto' | 'sse' | 'websocket' | 'websocket-cached' | undefined

  constructor(options: OpenAiCompatibleClientOptions) {
    const providerConfig = getProviderConfig(options.providerName)
    this.providerName = options.providerName
    this.apiKey = options.apiKey ?? getApiKey(options.providerName)
    this.baseUrl = options.baseUrl ?? providerConfig.baseUrl ?? ''
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.resolveAuthHeaders = options.resolveAuthHeaders
    this.codexTransport = options.codexTransport
    if (!this.baseUrl) {
      throw new ConfigurationError('base_url', 'No base URL is configured for ' + options.providerName)
    }
  }

  /** Static API-key headers, or freshly resolved OAuth headers when configured. */
  private async headers(
    accept: string,
    signal?: AbortSignal,
    request?: CompletionRequest,
  ): Promise<Record<string, string>> {
    const base = responsesHeaders(this.providerName, this.apiKey, accept)
    if (!this.resolveAuthHeaders) return base
    return { ...base, ...(await this.resolveAuthHeaders(signal, request)) }
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    // The ChatGPT backend serves streaming responses only and answers a
    // non-streamed request with `400 Stream must be set to true`. Collecting
    // our own stream keeps every non-streaming caller — /compact, session
    // titling, memory extraction — working instead of failing on a transport
    // detail they have no reason to know about.
    if (this.providerName === 'openai-codex') {
      return collectLlmCompletion(this.stream(request, signal))
    }
    const endpoint = new URL('responses', withTrailingSlash(this.baseUrl)).toString()
    const response = await this.fetchImplementation(endpoint, {
      method: 'POST',
      headers: await this.headers('application/json', signal, request),
      body: JSON.stringify(responsesPayload(request, this.providerName, false)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw openAiHttpError(
        this.providerName,
        'Responses API completion request failed (' + response.status + '): ' + body.slice(0, 4_096),
        response,
      )
    }
    return parseResponsesCompletion(parseJsonObject(await response.text(), this.providerName))
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    if (this.providerName === 'openai-codex') {
      yield* this.streamCodex(request, signal)
      return
    }
    const endpoint = new URL('responses', withTrailingSlash(this.baseUrl)).toString()
    const response = await this.fetchImplementation(endpoint, {
      method: 'POST',
      headers: await this.headers('text/event-stream', signal, request),
      body: JSON.stringify(responsesPayload(request, this.providerName, true)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw openAiHttpError(
        this.providerName,
        'Responses API stream request failed (' + response.status + '): ' + body.slice(0, 4_096),
        response,
      )
    }
    if (!response.body) {
      throw new ProviderError(this.providerName, 'Responses API stream returned no response body')
    }

    const translator = new ResponsesEventTranslator(createGrammarToolInputProperties(
      request.tools,
      supportsOpenAiGrammarTools(this.providerName, request.model),
    ))
    for await (const data of sseData(response.body)) {
      if (data === '[DONE]') break
      const event = parseJsonObject(data, this.providerName)
      for (const delta of translator.translate(event)) yield delta
    }
    translator.finish()
  }

  /**
   * Codex streaming with pi-ai's transport contract: WebSocket first (unless
   * pinned to SSE or the session already fell back), one reconnect retry on
   * connection-limit/oversize failures, one full-context retry when the
   * server lost the previous response, sticky SSE fallback after any other
   * pre-first-event WS failure, and zstd-compressed SSE bodies.
   */
  private async *streamCodex(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const transport = this.codexTransport ?? 'auto'
    const sessionId = request.sessionId
    const wsAllowed = transport !== 'sse'
      && !(sessionId !== undefined && codexWebSocketFallbackActive(sessionId))
    let retriedConnectionLimit = false
    let retriedMissingPrevious = false
    while (wsAllowed) {
      let emitted = false
      try {
        for await (const delta of this.streamCodexWebSocket(request, signal)) {
          emitted = true
          yield delta
        }
        return
      } catch (error) {
        // A failure after the first event cannot be retried or downgraded:
        // the consumer already saw output the retry would duplicate.
        if (emitted) throw error
        if (isCodexRetryableWebSocketError(error) && !retriedConnectionLimit) {
          retriedConnectionLimit = true
          continue
        }
        if (error instanceof CodexWsApiError
          && error.code === 'previous_response_not_found'
          && !retriedMissingPrevious) {
          retriedMissingPrevious = true
          if (sessionId) codexContinuations.delete(sessionId)
          continue
        }
        if (sessionId) recordCodexWebSocketFallback(sessionId)
        break
      }
    }
    yield* this.streamCodexSse(request, signal)
  }

  /** WebSocket transport: one uncompressed `response.create` frame per request. */
  private async *streamCodexWebSocket(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const transport = this.codexTransport ?? 'auto'
    const sessionId = request.sessionId
    const grammarProperties = createGrammarToolInputProperties(
      request.tools,
      supportsOpenAiGrammarTools(this.providerName, request.model),
    )
    const authHeaders = await this.headers('application/json', signal, request)
    const wsHeaders: Record<string, string> = {
      ...authHeaders,
      'OpenAI-Beta': CODEX_WEBSOCKET_BETA_HEADER,
    }
    delete wsHeaders.Accept
    delete wsHeaders.accept
    delete wsHeaders['Content-Type']
    delete wsHeaders['content-type']
    if (sessionId) {
      wsHeaders['x-client-request-id'] = sessionId.slice(0, 64)
      wsHeaders['session-id'] = sessionId.slice(0, 64)
    }
    const body = responsesPayload(request, this.providerName, true)
    const useCachedContext = transport === 'auto' || transport === 'websocket-cached'
    const continuation = sessionId && useCachedContext ? codexContinuations.get(sessionId) : undefined
    const prepared = continuation
      ? buildCachedWebSocketRequestBody(body, continuation)
      : { body, usedDelta: false }

    const translator = new ResponsesEventTranslator(grammarProperties)
    let responseId: string | undefined
    let assistantText = ''
    let thinkingSignature: string | undefined
    for await (const event of streamCodexWebSocket(prepared.body, {
      baseUrl: this.baseUrl,
      headers: wsHeaders,
      ...(sessionId ? { sessionId } : {}),
      ...(wsHeaders['chatgpt-account-id'] ? { accountId: wsHeaders['chatgpt-account-id'] } : {}),
      transport,
      ...(signal ? { signal } : {}),
    })) {
      if (event.type === 'response.created') {
        const id = asRecord(event.response).id
        if (typeof id === 'string' && id) responseId = id
      }
      for (const delta of translator.translate(event)) {
        if (delta.content) assistantText += delta.content
        if (delta.thinkingSignature) thinkingSignature = delta.thinkingSignature
        yield delta
      }
    }
    translator.finish()

    // Connection-scoped continuation (pi-ai): the store:false backend keeps
    // state on the socket, so the next request on this pooled connection can
    // send just the new tail behind previous_response_id.
    if (sessionId && useCachedContext && responseId) {
      const assistantItems = codexAssistantResponseItems(assistantText, thinkingSignature, translator.usage.toolCalls)
      // The continuation anchors on the FULL body: the next prefix check
      // compares against the complete input, not this turn's delta view.
      const next = continuationFromResponse(body, responseId, assistantItems)
      if (next) codexContinuations.set(sessionId, next)
    }
  }

  /** SSE fallback with pi-ai's zstd request compression for the Codex backend. */
  private async *streamCodexSse(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const endpoint = new URL('responses', withTrailingSlash(this.baseUrl)).toString()
    const rawBody = JSON.stringify(responsesPayload(request, this.providerName, true))
    const compressed = compressRequestBodyZstd(rawBody)
    const headers = await this.headers('text/event-stream', signal, request)
    if (compressed) Object.assign(headers, CODEX_SSE_COMPRESSION_HEADER)
    const response = await this.fetchImplementation(endpoint, {
      method: 'POST',
      headers,
      // TS lib.dom's BodyInit predates Uint8Array bodies; Bun fetch accepts them.
      body: (compressed ?? rawBody) as BodyInit,      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw openAiHttpError(
        this.providerName,
        'Responses API stream request failed (' + response.status + '): ' + body.slice(0, 4_096),
        response,
      )
    }
    if (!response.body) {
      throw new ProviderError(this.providerName, 'Responses API stream returned no response body')
    }
    const translator = new ResponsesEventTranslator(createGrammarToolInputProperties(
      request.tools,
      supportsOpenAiGrammarTools(this.providerName, request.model),
    ))
    for await (const data of sseData(response.body)) {
      if (data === '[DONE]') break
      const event = parseJsonObject(data, this.providerName)
      for (const delta of translator.translate(event)) yield delta
    }
    translator.finish()
  }
}

/** pi-ai's WebSocket-only beta flag; the SSE path keeps `responses=experimental`. */
const CODEX_WEBSOCKET_BETA_HEADER = 'responses_websockets=2026-02-06'

/** Connection-scoped Codex continuations keyed by session id. */
const codexContinuations = new Map<string, CodexWsContinuation>()

/**
 * Rebuild the assistant turn's Responses input items exactly as
 * messagesToResponsesInput will emit them next request, so the delta
 * continuation's strict prefix-extension check can actually hit.
 */
function codexAssistantResponseItems(
  text: string,
  thinkingSignature: string | undefined,
  toolCalls: readonly ToolCall[],
): Record<string, unknown>[] {
  const items: Record<string, unknown>[] = []
  const reasoningItem = responsesReasoningItem(thinkingSignature)
  if (reasoningItem) items.push(reasoningItem)
  if (text) items.push({ role: 'assistant', content: text })
  for (const call of toolCalls) {
    items.push({
      type: 'function_call',
      call_id: call.id,
      name: call.function.name,
      arguments: JSON.stringify(call.function.arguments),
    })
  }
  return items
}

/** Preserve HTTP retry metadata without mixing provider headers into user-facing messages. */
function openAiHttpError(provider: ProviderName, message: string, response: Response): ProviderError {
  const retryAfterSeconds = parseRetryAfterHeader(response.headers.get('retry-after'))
  return new ProviderError(provider, message, undefined, {
    status: response.status,
    ...(retryAfterSeconds === undefined ? {} : { retryAfterSeconds }),
  })
}

/** Parse either Retry-After delta-seconds or an HTTP date into a non-negative delay. */
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

/** Build the currently supported native streaming client for a configured model. */
export function createLlmClient(
  model: string,
  overrides: ProviderOverrides = {},
  options: LlmClientFactoryOptions = {},
): LlmClient {
  model = requireConfiguredModel(model)
  const pluginProvider = selectedPluginProvider(model, overrides, options.pluginRegistry)
  if (pluginProvider) {
    const client = pluginProvider.factory.createClient({
      model: bareModel(model),
      options: pluginProviderOptions(options),
      overrides,
      providerName: pluginProvider.name,
      requestedModel: model,
    })
    if (!isLlmClient(client)) {
      throw new ConfigurationError('provider', `Plugin provider '${pluginProvider.name}' returned an invalid LlmClient`)
    }
    return client
  }

  const providerName = resolveProvider(model, overrides)
  const providerConfig = getProviderConfig(providerName)
  const configuredApiKey = typeof overrides.api_key === 'string' ? overrides.api_key : options.apiKey
  const configuredBaseUrl = typeof overrides.base_url === 'string'
    ? overrides.base_url
    : typeof overrides.custom_base_url === 'string'
      ? overrides.custom_base_url
      : options.baseUrl
  if (providerConfig.transport === 'anthropic') {
    // The subscription session is consulted per request; a missing session
    // falls back to the ordinary API-key path untouched (pi-ai resolution
    // order: stored OAuth credential → ANTHROPIC_AUTH_TOKEN → API key).
    const anthropicOAuth = options.anthropicOAuthSession ?? new AnthropicOAuthSession()
    return new AnthropicMessagesClient({
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
      ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
      ...(options.promptCaching === undefined ? {} : { promptCaching: options.promptCaching }),
      providerName: providerConfig.name,
      resolveOAuthToken: async signal => {
        try {
          const credential = await anthropicOAuth.credential(signal)
          // Only subscription tokens take the OAuth surface (pi-ai parity);
          // an ambient ANTHROPIC_AUTH_TOKEN without `sk-ant-oat` is handled
          // by the session itself, not mistaken for an API key here.
          return isAnthropicOAuthToken(credential.access) ? credential.access : undefined
        } catch (error) {
          // Not signed in is the normal API-key path; refresh/network
          // failures must surface rather than silently degrade.
          if (error instanceof ConfigurationError) return undefined
          throw error
        }
      },
    })
  }
  if (providerConfig.transport !== 'openai') {
    throw new ConfigurationError('provider', `${providerName} requires its dedicated adapter.`)
  }
  if (providerName === 'openai-codex') {
    // Always the Responses transport, always OAuth: the ChatGPT backend
    // exposes no chat-completions route and accepts no API key.
    const session = options.codexSession ?? new CodexSession()
    const transportOverride = overrides.codex_transport
    const codexTransport = transportOverride === 'sse'
      || transportOverride === 'websocket'
      || transportOverride === 'websocket-cached'
      || transportOverride === 'auto'
      ? transportOverride
      : undefined
    if (transportOverride !== undefined && codexTransport === undefined) {
      throw new ConfigurationError(
        'codex_transport',
        `unknown Codex transport '${String(transportOverride)}'; expected auto, sse, websocket, or websocket-cached`,
      )
    }
    return new ResponsesApiClient({
      ...options,
      providerName,
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : { baseUrl: codexBaseUrl() }),
      resolveAuthHeaders: async signal => codexAuthHeaders(await session.credential(signal)),
      ...(codexTransport ? { codexTransport } : {}),
    })
  }
  if (providerName === 'github-copilot') {
    // The GitHub OAuth token mints a short-lived proxy token whose proxy-ep
    // claim also decides the api host (copilotApiBase): enterprise tokens
    // route to a different host than the individual default, so the request
    // URL is re-anchored per credential rather than pinned at construction.
    const session = options.copilotSession ?? new CopilotSession()
    const baseFetch = options.fetchImplementation ?? fetch
    return new OpenAiCompatibleClient({
      ...options,
      providerName,
      baseUrl: configuredBaseUrl ?? providerConfig.baseUrl ?? COPILOT_API_BASE_DEFAULT,
      ...(configuredBaseUrl
        ? {}
        : {
          fetchImplementation: async (input, init) => {
            const credential = await session.credential()
            const url = new URL(String(input))
            return baseFetch(
              new URL(url.pathname.replace(/^\/+/, '/') + url.search, withTrailingSlash(copilotApiBase(credential.access))).toString(),
              init,
            )
          },
        }),
      resolveAuthHeaders: async (signal, request) => {
        const credential = await session.credential(signal)
        return {
          ...copilotAuthHeaders(credential),
          ...copilotRequestHeaders({
            hasImages: request?.messages.some(
              message => Array.isArray(message.content)
                && message.content.some(part => part.type === 'image_url'),
            ) ?? false,
            ...(request?.messages.at(-1)?.role
              ? { lastMessageRole: request.messages.at(-1)?.role ?? '' }
              : {}),
          }),
        }
      },
    })
  }
  if (providerName === 'azure') {
    return new AzureOpenAiClient({
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
      ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
    })
  }
  if (providerName === 'radius') {
    // Pi's own gateway speaks the pi-messages wire protocol; its catalog is
    // live (radiusGateway.ts) rather than static.
    return new PiMessagesClient(bareModel(model), {
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      baseUrl: configuredBaseUrl ?? providerConfig.baseUrl ?? DEFAULT_RADIUS_GATEWAY,
      providerName: 'radius',
      ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
    })
  }
  if (providerName === 'amazon-bedrock') {
    // Auth is AWS-native (SigV4 credential chain or a Bedrock bearer token);
    // an `api_key` override maps to the bearer-token auth scheme.
    return new BedrockConverseClient({
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
    })
  }
  // Dedicated transports: intercepted before responses_api/multi-api routing
  // because their native APIs do not offer those surfaces.
  if (providerName === 'google-vertex') {
    return new GoogleVertexClient({
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
      ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
    })
  }
  if (providerName === 'mistral') {
    return new MistralClient({
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
      ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
    })
  }
  if (providerName === 'cloudflare-workers-ai') {
    return createCloudflareWorkersAiClient({
      ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
      overrides: configuredApiKey ? { apiKey: configuredApiKey } : {},
    })
  }
  const multiApiBase = MULTI_API_PROVIDERS.has(providerName)
    ? routeMultiApiProvider(model, providerName, overrides, options, configuredApiKey, configuredBaseUrl)
    : undefined
  if (multiApiBase) return multiApiBase
  const useResponsesApi = options.responsesApi === true || overrides.responses_api === true
  if (useResponsesApi) {
    return new ResponsesApiClient({
      ...options,
      providerName,
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
    })
  }
  // Subscription-backed providers: a stored OAuth credential provides the
  // Bearer header; without one the ordinary API-key path is kept untouched.
  if (providerName === 'kimi-code') {
    const session = options.kimiOAuthSession ?? new KimiCodingOAuthSession()
    return new OpenAiCompatibleClient({
      ...options,
      providerName,
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
      resolveAuthHeaders: async signal => {
        try {
          const credential = await session.credential(signal)
          return { Authorization: `Bearer ${credential.access}` }
        } catch (error) {
          if (error instanceof ConfigurationError) return {}
          throw error
        }
      },
    })
  }
  if (providerName === 'openrouter') {
    const session = options.openrouterOAuthSession ?? new OpenRouterOAuthSession()
    return new OpenAiCompatibleClient({
      ...options,
      providerName,
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
      resolveAuthHeaders: async signal => {
        try {
          const credential = await session.credential(signal)
          return { Authorization: `Bearer ${credential.access}` }
        } catch (error) {
          if (error instanceof ConfigurationError) return {}
          throw error
        }
      },
    })
  }
  if (providerName === 'xai') {
    const session = options.xaiOAuthSession ?? new XaiOAuthSession()
    return new OpenAiCompatibleClient({
      ...options,
      providerName,
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
      resolveAuthHeaders: async signal => {
        try {
          const credential = await session.credential(signal)
          return { Authorization: `Bearer ${credential.access}` }
        } catch (error) {
          if (error instanceof ConfigurationError) return {}
          throw error
        }
      },
    })
  }
  if (providerName === 'gemini') {
    return new GeminiClient({
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      ...(configuredBaseUrl ? { baseUrl: nativeGeminiBaseUrl(configuredBaseUrl) } : {}),
      ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
    })
  }
  if (providerName === 'ollama') {
    const topK = typeof overrides.top_k === 'number' ? overrides.top_k : undefined
    return new OllamaClient({
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
      ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
      ...(topK === undefined ? {} : { topK }),
    })
  }
  return new OpenAiCompatibleClient({
    ...options,
    providerName,
    ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
    ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
  })
}

/**
 * Gateways that serve different models over different protocols
 * (pi-ai multi-api providers). The catalog entry's api field — not a blanket
 * provider default — decides the transport, because these hosts genuinely
 * lack a unified endpoint: OpenCode Zen's Claude models only speak
 * anthropic-messages, its GPT models only responses, and so on.
 */
const MULTI_API_PROVIDERS: ReadonlySet<ProviderName> = new Set([
  'cloudflare-ai-gateway',
  'fireworks',
  'opencode',
  'opencode-go',
])

function routeMultiApiProvider(
  model: string,
  providerName: ProviderName,
  overrides: ProviderOverrides,
  options: LlmClientFactoryOptions,
  configuredApiKey: string | undefined,
  configuredBaseUrl: string | undefined,
): LlmClient | undefined {
  const entry = piCatalogModelCapabilities(model, providerName)
  if (!entry) return undefined
  const baseUrl = configuredBaseUrl ?? cloudflareGatewayBaseUrl(entry.baseUrl, providerName)
  const shared = {
    ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
    ...(baseUrl ? { baseUrl } : {}),
    ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
  }
  switch (entry.api) {
    case 'anthropic-messages':
      return new AnthropicMessagesClient({
        ...shared,
        ...(options.promptCaching === undefined ? {} : { promptCaching: options.promptCaching }),
      })
    case 'openai-responses':
      return new ResponsesApiClient({ ...options, providerName, ...shared })
    case 'google-generative-ai':
      return new GeminiClient({
        ...shared,
        ...(baseUrl ? { baseUrl: nativeGeminiBaseUrl(baseUrl) } : {}),
      })
    default:
      return new OpenAiCompatibleClient({ ...options, providerName, ...shared })
  }
}

/**
 * Cloudflare AI Gateway catalog URLs are account-templated
 * (`.../v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/...`); substitute
 * the environment, and fail loudly rather than sending a literal brace URL.
 */
function cloudflareGatewayBaseUrl(template: string | undefined, providerName: ProviderName): string | undefined {
  if (!template?.includes('{')) return template
  const accountId = process.env.CLOUDFLARE_ACCOUNT_ID?.trim()
  const gatewayId = process.env.CLOUDFLARE_GATEWAY_ID?.trim()
  if (!accountId || !gatewayId) {
    throw new ConfigurationError(
      'cloudflare_ai_gateway',
      `${providerName} requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_GATEWAY_ID ` +
      '(or an explicit base_url) — the gateway URL is account-scoped',
    )
  }
  return template
    .replaceAll('{CLOUDFLARE_ACCOUNT_ID}', accountId)
    .replaceAll('{CLOUDFLARE_GATEWAY_ID}', gatewayId)
}

/** Reject execution that would otherwise guess a provider/model from an empty ID. */export function requireConfiguredModel(model: string | undefined): string {
  const configured = model?.trim() ?? ''
  if (!configured) {
    throw new ConfigurationError(
      'model',
      'is not configured; select a provider model or pass an explicit model in runtime configuration',
    )
  }
  return configured
}

/**
 * Default overall deadline for a {@link completeLlm} call.
 *
 * Sized generously: it exists to turn a stalled upstream into an error, not
 * to cut off slow-but-working generations. Housekeeping completions (pre-turn
 * auto-compaction, session titling, memory extraction) have no UI watching
 * them, so without a deadline they could hang a turn forever.
 */
export const DEFAULT_COMPLETION_DEADLINE_MS = 180_000

/** Raised when a completion exceeds its deadline rather than failing on its own. */
export class CompletionDeadlineError extends Error {
  constructor(
    readonly timeoutMs: number,
    options?: { readonly cause?: unknown },
  ) {
    super(`LLM completion did not finish within ${timeoutMs}ms`, options)
    this.name = 'CompletionDeadlineError'
  }
}

/** Per-call controls for {@link completeLlm}. */
export interface CompleteLlmOptions {
  /**
   * Overall deadline in milliseconds, overriding
   * {@link DEFAULT_COMPLETION_DEADLINE_MS}. A caller signal still aborts
   * immediately and independently of this deadline.
   */
  readonly timeoutMs?: number
}

/**
 * Generate one complete, provider-neutral response.
 *
 * Dedicated adapters can make a native non-streaming request through their optional
 * `complete` method. Stream-only plugins and adapters are collected losslessly from
 * the same delta vocabulary instead, so adding this API does not invalidate them.
 *
 * The call carries a default deadline (see {@link DEFAULT_COMPLETION_DEADLINE_MS})
 * combined with any caller-provided signal: whichever fires first aborts the
 * transport. The work is additionally raced against that same abort so THIS
 * caller observes the failure promptly even when a stalled transport never
 * inspects its signal — precisely the housekeeping stall the deadline exists
 * for. The race listener is detached on settle.
 */
export async function completeLlm(
  client: LlmClient,
  request: CompletionRequest,
  signal?: AbortSignal,
  options: CompleteLlmOptions = {},
): Promise<LlmCompletion> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_COMPLETION_DEADLINE_MS
  const deadline = AbortSignal.timeout(timeoutMs)
  // Bun 1.3 provides AbortSignal.any; combining keeps one wire into the
  // transport while the deadline stays attributable for error translation.
  const combined = signal ? AbortSignal.any([signal, deadline]) : deadline

  let onCombinedAbort: (() => void) | undefined
  try {
    const work = typeof client.complete === 'function'
      ? client.complete(request, combined)
      : collectLlmCompletion(client.stream(request, combined))
    return await Promise.race([work, abortRejection(combined, listener => {
      onCombinedAbort = listener
    })])
  } catch (error) {
    if (deadline.aborted && !signal?.aborted) {
      throw new CompletionDeadlineError(timeoutMs, { cause: error })
    }
    throw error
  } finally {
    if (onCombinedAbort) combined.removeEventListener('abort', onCombinedAbort)
  }
}

/**
 * Reject with the signal's abort reason as soon as it fires.
 *
 * `register` hands the listener back so the caller can detach it on settle;
 * otherwise every completed call would leave a reaction parked until the
 * deadline eventually fires.
 */
function abortRejection(signal: AbortSignal, register: (listener: () => void) => void): Promise<never> {
  return new Promise((_, reject) => {
    const listener = () => reject(signal.reason)
    register(listener)
    if (signal.aborted) {
      listener()
      return
    }
    signal.addEventListener('abort', listener, { once: true })
  })
}

/** Collect a provider-neutral stream into a non-streaming completion result. */
export async function collectLlmCompletion(stream: AsyncIterable<LlmDelta>): Promise<LlmCompletion> {
  const content: string[] = []
  const thinking: string[] = []
  const toolCalls = new Map<string, ToolCall>()
  const toolCallOccurrences = new Map<string, number>()
  let finishReason: string | undefined
  let thinkingSignature: string | undefined
  let usage: TokenUsage | undefined

  for await (const delta of stream) {
    if (delta.content) {
      content.push(delta.content)
    }
    if (delta.thinking) {
      thinking.push(delta.thinking)
    }
    if (delta.thinkingSignature) {
      thinkingSignature = delta.thinkingSignature
    }
    if (delta.finishReason) {
      finishReason = delta.finishReason
    }
    if (delta.usage) {
      usage = mergeTokenUsage(usage, delta.usage)
    }
    for (const toolCall of delta.toolCalls ?? []) {
      // Id-less calls get a deterministic id derived from name+arguments, so
      // two identical id-less calls share one key. Keying by raw id would
      // silently drop the second call, so repeats get an occurrence suffix.
      const occurrence = toolCallOccurrences.get(toolCall.id) ?? 0
      toolCallOccurrences.set(toolCall.id, occurrence + 1)
      toolCalls.set(occurrence === 0 ? toolCall.id : `${toolCall.id}#${occurrence}`, toolCall)
    }
  }

  return {
    content: content.join(''),
    toolCalls: [...toolCalls.values()],
    ...(finishReason === undefined ? {} : { finishReason }),
    ...(thinking.length ? { thinking: thinking.join('') } : {}),
    ...(thinkingSignature === undefined ? {} : { thinkingSignature }),
    ...(usage === undefined ? {} : { usage }),
  }
}

/**
 * Process streamed text with a callback and return the complete visible text.
 *
 * Tool, usage, and thinking deltas remain available to callers that need the
 * whole neutral event stream; this helper mirrors the legacy text callback API.
 */
export async function processLlmStream(
  stream: AsyncIterable<LlmDelta>,
  onText: (content: string, delta: LlmDelta) => Promise<void> | void,
): Promise<string> {
  const content: string[] = []
  for await (const delta of stream) {
    if (!delta.content) {
      continue
    }
    content.push(delta.content)
    await onText(delta.content, delta)
  }
  return content.join('')
}

/** Close a client when it owns a provider SDK or other resource; fetch clients are a no-op. */
export async function closeLlmClient(client: LlmClient): Promise<void> {
  if (typeof client.close === 'function') {
    await client.close()
  }
}

/** Run an operation with a client and always close a resource-owning implementation afterward. */
export async function withLlmClient<Result>(
  client: LlmClient,
  operation: (client: LlmClient) => Promise<Result> | Result,
): Promise<Result> {
  try {
    return await operation(client)
  } finally {
    await closeLlmClient(client)
  }
}

/** Return provider/model metadata without performing an unauthenticated network probe. */
export function getLlmModelInfo(
  model: string,
  options: LlmModelInfoOptions = {},
  overrides: ProviderOverrides = {},
): LlmModelInfo {
  return {
    provider: resolveProvider(model, overrides),
    model,
    temperature: options.temperature ?? DEFAULT_TEMPERATURE,
    maxTokens: options.maxTokens ?? 2_048,
    ...(options.contextLimit === undefined ? {} : { maxModelLen: options.contextLimit }),
    stream: options.stream ?? false,
  }
}

/** Prepend an optional system instruction without mutating the caller's transcript. */
export function formatLlmMessages(
  messages: readonly ChatMessage[],
  systemPrompt?: string,
): ChatMessage[] {
  if (!systemPrompt) {
    return [...messages]
  }
  return [{ role: 'system', content: systemPrompt }, ...messages]
}

function selectedPluginProvider(
  model: string,
  overrides: ProviderOverrides,
  registry: PluginLlmProviderRegistry | undefined,
): { readonly factory: PluginLlmProviderFactory; readonly name: string } | undefined {
  if (!registry) return undefined
  const name = requestedPluginProviderName(model, overrides)
  if (!name || isProviderName(name.toLowerCase())) return undefined
  const factory = registry.getProvider(name)
  if (!factory) return undefined
  if (!isPluginLlmProviderFactory(factory)) {
    throw new ConfigurationError('provider', `Plugin provider '${name}' must expose createClient(request)`)
  }
  return { factory, name }
}

function requestedPluginProviderName(model: string, overrides: ProviderOverrides): string | undefined {
  const configured = typeof overrides.provider === 'string'
    ? overrides.provider
    : typeof overrides.provider_type === 'string'
      ? overrides.provider_type
      : undefined
  if (configured?.trim()) return configured.trim()
  const slash = model.indexOf('/')
  return slash > 0 ? model.slice(0, slash) : undefined
}

function pluginProviderOptions(options: LlmClientFactoryOptions): PluginLlmProviderOptions {
  return {
    ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
    ...(options.baseUrl === undefined ? {} : { baseUrl: options.baseUrl }),
    ...(options.fetchImplementation === undefined ? {} : { fetchImplementation: options.fetchImplementation }),
    ...(options.promptCaching === undefined ? {} : { promptCaching: options.promptCaching }),
    ...(options.responsesApi === undefined ? {} : { responsesApi: options.responsesApi }),
  }
}

function isLlmClient(value: unknown): value is LlmClient {
  return typeof value === 'object' && value !== null && typeof (value as { stream?: unknown }).stream === 'function'
}

function responsesToolDefinition(
  tool: ToolDefinition,
  supportsGrammarTools = false,
  deferLoading = false,
): Record<string, unknown> {
  const grammar = resolveGrammar(tool, supportsGrammarTools)
  if (grammar) {
    return {
      type: 'custom',
      name: tool.function.name,
      description: tool.function.description,
      format: { type: 'grammar', syntax: grammar.syntax, definition: grammar.definition },
      ...(deferLoading ? { defer_loading: true } : {}),
    }
  }
  return {
    type: 'function',
    name: tool.function.name,
    description: tool.function.description,
    parameters: tool.function.parameters,
    ...(deferLoading ? { defer_loading: true } : {}),
  }
}

/** Short deterministic id fragment for synthetic tool-search replay items (pi-ai `pi_tool_load_<hash>`). */
function shortHash(value: string): string {
  return createHash('sha256').update(value).digest('hex').slice(0, 12)
}

function responsesToolChoice(choice: ToolChoice): string {
  if (choice === 'any') return 'required'
  return choice
}

function responsesReasoningItem(signature: string | undefined): Record<string, unknown> | undefined {
  if (!signature) return undefined
  try {
    const parsed: unknown = JSON.parse(signature)
    const item = asRecord(parsed)
    return item.type === 'reasoning' ? item : undefined
  } catch {
    return undefined
  }
}

/**
 * Translate the neutral transcript into Responses API input items.
 *
 * Assistant tool calls become `function_call` items and tool replies become
 * `function_call_output` items; multipart user content uses `input_text` and
 * `input_image` parts. Chat-completions message shapes are not valid input.
 */
function messagesToResponsesInput(
  messages: readonly ChatMessage[],
  grammarProperties: ReadonlyMap<string, string> = new Map(),
  deferredTools: ReadonlyMap<string, ToolDefinition> = new Map(),
  deferredMode: 'additional-tools' | 'tool-search' | undefined = undefined,
  deferLoading = false,
): Record<string, unknown>[] {
  const input: Record<string, unknown>[] = []
  // Custom tool results are distinguished by the call id the assistant used:
  // grammar calls replay as custom_tool_call_output, everything else as
  // function_call_output (pi-ai replay contract).
  const customCallIds = new Set<string>()
  // Deferred schemas are introduced exactly once, at the result that loaded
  // them, through the provider's native load protocol.
  const loadedToolNames = new Set<string>()
  const deferredLoadItems = (message: ToolMessageLike): Record<string, unknown>[] => {
    const names = (message.added_tool_names ?? []).filter(
      name => deferredTools.has(name) && !loadedToolNames.has(name),
    )
    if (!names.length || deferredMode === undefined) return []
    for (const name of names) loadedToolNames.add(name)
    const tools = names.map(name => {
      const definition = deferredTools.get(name)
      return definition === undefined ? undefined : responsesToolDefinition(definition, grammarProperties.has(name), deferLoading)
    }).filter((definition): definition is Record<string, unknown> => definition !== undefined)
    if (!tools.length) return []
    if (deferredMode === 'additional-tools') {
      return [{ type: 'additional_tools', role: 'developer', tools }]
    }
    const searchCallId = `xerxes_tool_load_${shortHash(`${message.tool_call_id}:${names.join(',')}`)}`
    return [
      {
        type: 'tool_search_call',
        call_id: searchCallId,
        execution: 'client',
        status: 'completed',
        arguments: { query: names.join(' '), limit: names.length },
      },
      {
        type: 'tool_search_output',
        call_id: searchCallId,
        execution: 'client',
        status: 'completed',
        tools,
      },
    ]
  }
  for (const message of messages) {
    if (message.role === 'assistant') {
      const reasoningItem = responsesReasoningItem(message.thinking_signature)
      if (reasoningItem) input.push(reasoningItem)
      const text = messageText(message)
      if (text) {
        input.push({ role: 'assistant', content: text })
      }
      for (const call of message.tool_calls ?? []) {
        const grammarProperty = grammarProperties.get(call.function.name)
        if (grammarProperty !== undefined) {
          customCallIds.add(call.id)
          input.push({
            type: 'custom_tool_call',
            call_id: call.id,
            name: call.function.name,
            input: grammarToolInput(call.function.name, grammarProperty, call.function.arguments),
          })
          continue
        }
        input.push({
          type: 'function_call',
          call_id: call.id,
          name: call.function.name,
          arguments: JSON.stringify(call.function.arguments),
        })
      }
      continue
    }
    if (message.role === 'tool') {
      input.push(customCallIds.has(message.tool_call_id)
        ? { type: 'custom_tool_call_output', call_id: message.tool_call_id, output: message.content }
        : { type: 'function_call_output', call_id: message.tool_call_id, output: message.content })
      input.push(...deferredLoadItems(message))
      continue
    }
    input.push({ role: message.role, content: responsesMessageContent(message.content) })
  }
  return input
}

type ToolMessageLike = Pick<Extract<ChatMessage, { role: 'tool' }>, 'added_tool_names' | 'tool_call_id'>

function responsesMessageContent(content: MessageContent): unknown {
  if (typeof content === 'string') {
    return content
  }
  return content.map(part => part.type === 'text'
    ? { type: 'input_text', text: part.text }
    : {
      type: 'input_image',
      image_url: part.image_url.url,
      ...(part.image_url.detail ? { detail: part.image_url.detail } : {}),
    })
}

function responsesPayload(
  request: CompletionRequest,
  providerName: ProviderName,
  stream: boolean,
): Record<string, unknown> {
  const systemPrompt = request.messages
    .filter(message => message.role === 'system')
    .map(messageText)
    .filter(Boolean)
    .join('\n\n')
  const grammarSupport = piCatalogModelCapabilities(request.model, providerName)
    ?.compat?.supportsOpenAIGrammarTools === true
  const grammarProperties = createGrammarToolInputProperties(request.tools, grammarSupport)
  const deferredMode = responsesDeferredToolsMode(providerName, request.model)
  const deferredSplit = splitDeferredTools(request.tools, request.messages, deferredMode !== undefined)
  const payload: Record<string, unknown> = {
    model: providerModel(request.model, providerName),
    input: messagesToResponsesInput(
      request.messages.filter(message => message.role !== 'system'),
      grammarProperties,
      deferredSplit.deferred,
      deferredMode,
      // tool-search replay items always mark the discovered schemas deferred.
      deferredMode === 'tool-search',
    ),
    stream,
    store: false,
    ...(systemPrompt ? { instructions: systemPrompt } : {}),
  }
  addResponsesSampling(payload, request, providerName)
  if (request.serviceTier !== undefined) {
    payload.service_tier = request.serviceTier
  }
  if (request.thinking?.effort) {
    payload.include = ['reasoning.encrypted_content']
  }
  if (request.tools?.length) {
    // Only immediate tools ride the top-level array in native deferred modes;
    // deferred schemas enter through additional_tools / tool-search replay
    // items anchored at the result that loaded them (pi-ai).
    const wireTools = deferredMode === undefined ? request.tools : deferredSplit.immediate
    if (wireTools.length) {
      payload.tools = wireTools.map(tool => responsesToolDefinition(tool, grammarSupport))
    }
    if (request.toolChoice) payload.tool_choice = responsesToolChoice(request.toolChoice)
  }
  // The Responses API has no cache_control breakpoints: it caches long
  // prefixes automatically, but only routes a repeat to the machine holding
  // that prefix when the request carries a stable cache key. Without one an
  // agent loop re-reads the same system prompt at full price every turn.
  const cacheKey = promptCacheKey(request, providerName)
  if (cacheKey) {
    payload.prompt_cache_key = cacheKey
  }
  if (providerName === 'openai-codex') {
    // The subscription backend accepts a strict subset of the Responses
    // schema and answers anything outside it with `400 Unsupported
    // parameter`, never by ignoring the field. It caps output by plan rather
    // than per request, and it is not the stateful Responses host, so both
    // the output cap and the sampling knobs have to come back off.
    delete payload.max_output_tokens
    delete payload.temperature
    delete payload.top_p
    payload.store = false
  }
  return payload
}

/**
 * Stable cache key for the reusable head of a conversation.
 *
 * Derived from the model plus the leading system prompt — the part that is
 * identical across every turn of a session — so repeats of the same prefix
 * route to the same backend and hit its cache. Volatile tail messages are
 * deliberately excluded: folding them in would mint a fresh key each turn and
 * guarantee a miss, which is the same as sending no key at all.
 */
function providerCacheKey(sessionId: string): string {
  const normalized = sessionId.trim()
  return normalized.length <= 64
    ? normalized
    : createHash('sha256').update(normalized).digest('hex')
}

function promptCacheKey(request: CompletionRequest, providerName: ProviderName): string | undefined {
  // Scoped to the hosts documented to accept it. `responses_api` can be turned
  // on for third-party OpenAI-compatible endpoints, and a strict one answers an
  // unrecognized field with a 400 rather than ignoring it — the Codex backend
  // does exactly that. A cache hit is not worth breaking those requests.
  if (providerName !== 'openai' && providerName !== 'openai-codex') {
    return undefined
  }
  if (request.sessionId) return providerCacheKey(request.sessionId)
  const stable = request.systemSegments?.length
    ? request.systemSegments.map(segment => segment.text).join('\n')
    : request.messages.find(message => message.role === 'system')?.content
  const text = typeof stable === 'string'
    ? stable
    : (stable ?? []).map(part => part.type === 'text' ? part.text : '').join('')
  if (!text.trim()) {
    return undefined
  }
  const digest = createHash('sha256')
    .update(`${providerName}\0${providerModel(request.model, providerName)}\0${text}`)
    .digest('hex')
  return `xerxes-${digest.slice(0, 32)}`
}

function responsesHeaders(providerName: ProviderName, apiKey: string, accept: string): Record<string, string> {
  const headers: Record<string, string> = {
    Accept: accept,
    'Content-Type': 'application/json',
    ...providerDefaultHeaders(providerName),
  }
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`
  return headers
}

function openAiMessagesForProvider(
  messages: readonly ChatMessage[],
  providerName: ProviderName,
  grammarProperties: ReadonlyMap<string, string> = new Map(),
): OpenAiChatMessage[] {
  const converted = messagesToOpenAi(messages, grammarProperties)
  if (providerName !== 'deepseek') return converted
  return converted.map(message => message.role === 'assistant' && message.tool_calls?.length
    ? { ...message, reasoning_content: message.reasoning_content ?? '' }
    : message)
}

/**
 * Kimi's deferred-tools mode (pi-ai `deferredToolsMode: "kimi"`): deferred
 * schemas leave the top-level tools array and enter through a synthetic
 * system message carrying `tools` (and no content) right after the tool
 * result that loaded them, once per name per conversation.
 */
function withKimiDeferredToolMessages(
  converted: readonly OpenAiChatMessage[],
  messages: readonly ChatMessage[],
  deferred: ReadonlyMap<string, ToolDefinition>,
  grammarSupport: boolean,
): Record<string, unknown>[] {
  const output: Record<string, unknown>[] = []
  const loaded = new Set<string>()
  for (const [index, message] of messages.entries()) {
    const wire = converted[index]
    if (wire !== undefined) output.push(wire as unknown as Record<string, unknown>)
    if (message.role !== 'tool') continue
    const names = (message.added_tool_names ?? []).filter(name => deferred.has(name) && !loaded.has(name))
    if (!names.length) continue
    for (const name of names) loaded.add(name)
    const tools = names
      .map(name => deferred.get(name))
      .filter((tool): tool is ToolDefinition => tool !== undefined)
      .map(tool => completionsToolDefinition(tool, grammarSupport))
    if (tools.length) output.push({ role: 'system', tools })
  }
  return output
}

/**
 * Chat-completions grammar tool shape (pi-ai): unlike the Responses flat
 * `format`, completions nests the grammar under `custom.format.grammar`.
 */
function completionsToolDefinition(
  tool: ToolDefinition,
  supportsGrammarTools: boolean,
): Record<string, unknown> {
  const grammar = resolveGrammar(tool, supportsGrammarTools)
  if (grammar) {
    return {
      type: 'custom',
      custom: {
        name: tool.function.name,
        description: tool.function.description,
        format: {
          type: 'grammar',
          grammar: { syntax: grammar.syntax, definition: grammar.definition },
        },
      },
    }
  }
  return {
    type: 'function',
    function: {
      name: tool.function.name,
      description: tool.function.description,
      parameters: tool.function.parameters,
    },
  }
}

function openAiCompatiblePayload(
  request: CompletionRequest,
  providerName: ProviderName,
  stream: boolean,
): Record<string, unknown> {
  const modelCapabilities = piCatalogModelCapabilities(request.model, providerName)
  const grammarSupport = modelCapabilities?.compat?.supportsOpenAIGrammarTools === true
  const grammarProperties = createGrammarToolInputProperties(request.tools, grammarSupport)
  const kimiDeferred = completionsDeferredToolsMode(providerName, request.model) === 'kimi'
  const deferredSplit = splitDeferredTools(request.tools, request.messages, kimiDeferred)
  const payload: Record<string, unknown> = {
    model: providerModel(request.model, providerName),
    messages: kimiDeferred
      ? withKimiDeferredToolMessages(
        openAiMessagesForProvider(request.messages, providerName, grammarProperties),
        request.messages,
        deferredSplit.deferred,
        grammarSupport,
      )
      : openAiMessagesForProvider(request.messages, providerName, grammarProperties),
    stream,
  }
  // OpenAI's chat-completions prompt-cache key routes a repeated session
  // prefix to the same backend; other hosts may reject the extension.
  if (providerName === 'openai' && request.sessionId) {
    payload.prompt_cache_key = providerCacheKey(request.sessionId)
  }
  addSampling(payload, request, providerName)
  if (request.tools?.length) {
    const wireTools = kimiDeferred ? deferredSplit.immediate : request.tools
    if (wireTools.length) {
      payload.tools = wireTools.map(tool => completionsToolDefinition(tool, grammarSupport))
    }
    if (modelCapabilities?.compat?.zaiToolStream === true || providerName === 'zhipu') payload.tool_stream = true
    if (request.toolChoice) {
      payload.tool_choice = request.toolChoice === 'any' ? 'required' : request.toolChoice
    }
  }
  if (stream && providerName !== 'minimax') {
    payload.stream_options = { include_usage: true }
  }
  return payload
}

function openAiCompatibleHeaders(providerName: ProviderName, apiKey: string, accept: string): Record<string, string> {
  const headers: Record<string, string> = {
    Accept: accept,
    'Content-Type': 'application/json',
    ...providerDefaultHeaders(providerName),
  }
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`
  }
  return headers
}

/**
 * Sampling fields for the Responses API, which is not chat-completions with a
 * different path.
 *
 * It renames the output cap to `max_output_tokens`, carries reasoning effort
 * as a nested object, and rejects the chat-completions penalty and stop
 * parameters outright with a 400 rather than ignoring them — so the neutral
 * request is translated here instead of reusing {@link addSampling}.
 */
function addResponsesSampling(
  payload: Record<string, unknown>,
  request: CompletionRequest,
  providerName: ProviderName,
): void {
  if (request.temperature !== undefined && supportsTemperature(providerName, request.model, request.temperature)) {
    payload.temperature = request.temperature
  }
  if (request.maxTokens !== undefined) {
    payload.max_output_tokens = Math.max(16, request.maxTokens)
  }
  if (request.topP !== undefined) {
    payload.top_p = request.topP
  }
  const effort = request.thinking?.effort
  if (isGradedEffort(effort)) {
    // Pi's Responses adapter requests only the effort; `summary: 'auto'` is a
    // native Responses extension that strict third-party hosts reject.
    payload.reasoning = { effort }
  }
}

function addSampling(
  payload: Record<string, unknown>,
  request: CompletionRequest,
  providerName: ProviderName,
): void {
  if (request.temperature !== undefined && supportsTemperature(providerName, request.model, request.temperature)) {
    payload.temperature = request.temperature
  }
  if (request.maxTokens !== undefined) {
    const maxTokensField = piCatalogModelCapabilities(request.model, providerName)?.compat?.maxTokensField
    payload[maxTokensField === 'max_tokens' || maxTokensField === 'max_completion_tokens'
      ? maxTokensField
      : 'max_tokens'] = request.maxTokens
  }
  if (request.topP !== undefined) {
    payload.top_p = request.topP
  }
  if (request.frequencyPenalty !== undefined) {
    payload.frequency_penalty = request.frequencyPenalty
  }
  if (request.presencePenalty !== undefined) {
    payload.presence_penalty = request.presencePenalty
  }
  if (request.stop?.length) {
    payload.stop = request.stop
  }
  const effort = request.thinking?.effort
  const capabilities = piCatalogModelCapabilities(request.model, providerName)
  const thinkingFormat = capabilities?.compat?.thinkingFormat
  const thinkingMap = capabilities?.thinkingLevelMap
  const mappedEffort = effort ? thinkingMap?.[effort] : undefined
  const reasoningEffort = mappedEffort === null
    ? undefined
    : mappedEffort ?? (isGradedEffort(effort) ? effort : undefined)
  const thinkingEnabled = request.thinking !== undefined && effort !== 'off' && effort !== 'none'
  // pi-ai guards every thinking field on model.reasoning: a catalog entry
  // that says the model does not reason gets no thinking knobs at all, and a
  // map whose `off` is null means the model CANNOT disable thinking (sending
  // `disabled` would be a provider-side rejection). Unknown models keep the
  // provider-name heuristics untouched.
  const reasoningModel = capabilities?.reasoning !== false
  if (!reasoningModel) {
    // no thinking fields
  } else if (thinkingFormat === 'zai' || providerName === 'zhipu') {
    payload.thinking = thinkingEnabled
      ? { type: 'enabled', clear_thinking: false }
      : { type: 'disabled' }
    if (capabilities?.compat?.supportsReasoningEffort !== false && reasoningEffort) {
      payload.reasoning_effort = reasoningEffort
    }
  } else if (thinkingFormat === 'deepseek' || providerName === 'deepseek') {
    if (thinkingEnabled) {
      payload.thinking = { type: 'enabled' }
    } else if (thinkingMap?.off !== null) {
      payload.thinking = { type: 'disabled' }
    }
    if (thinkingEnabled && capabilities?.compat?.supportsReasoningEffort !== false && reasoningEffort) {
      payload.reasoning_effort = reasoningEffort
    }
  } else if (thinkingFormat === 'openrouter' || providerName === 'openrouter') {
    // pi: an explicit off maps through thinkingLevelMap.off ('none' when the
    // map names nothing); on without a graded effort leaves the field out so
    // the provider default applies — never `effort: 'none'` for an on turn.
    if (thinkingEnabled && reasoningEffort) {
      payload.reasoning = { effort: reasoningEffort }
    } else if (!thinkingEnabled && thinkingMap?.off !== null) {
      payload.reasoning = { effort: typeof thinkingMap?.off === 'string' ? thinkingMap.off : 'none' }
    }
  } else if (thinkingFormat === 'qwen' || providerName === 'qwen') {
    payload.enable_thinking = thinkingEnabled
    if (capabilities?.compat?.supportsReasoningEffort !== false && reasoningEffort) {
      payload.reasoning_effort = reasoningEffort
    }
  } else if (reasoningEffort) {
    payload.reasoning_effort = reasoningEffort
  } else if (!thinkingEnabled && typeof thinkingMap?.off === 'string' && capabilities?.compat?.supportsReasoningEffort) {
    // OpenAI-style: off lands as the map's own off word (e.g. 'none').
    payload.reasoning_effort = thinkingMap.off
  }
  if (request.thinking?.budgetTokens !== undefined) {
    payload.thinking_budget = request.thinking.budgetTokens
  }
  if (request.extraBody) {
    for (const [key, value] of Object.entries(request.extraBody)) {
      if (!OPENAI_COMPATIBLE_RESERVED_BODY_FIELDS.has(key)) {
        payload[key] = value
      }
    }
  }
  if (!supportsExtendedSampling(providerName)) {
    return
  }
  if (request.topK !== undefined) {
    payload.top_k = request.topK
  }
  if (request.minP !== undefined) {
    payload.min_p = request.minP
  }
  if (request.repetitionPenalty !== undefined) {
    payload.repetition_penalty = request.repetitionPenalty
  }
}

const OPENAI_COMPATIBLE_RESERVED_BODY_FIELDS = new Set([
  'model',
  'messages',
  'stream',
  'stream_options',
  'temperature',
  'max_tokens',
  'max_completion_tokens',
  'store',
  'prompt_cache_key',
  'prompt_cache_retention',
  'top_p',
  'frequency_penalty',
  'presence_penalty',
  'stop',
  'tools',
  'tool_choice',
  'reasoning_effort',
  'reasoning',
  'thinking',
  'thinking_budget',
  'enable_thinking',
  'tool_stream',
])

/** Kimi Code fixes temperature at 1 and rejects every other explicit value. */
function supportsTemperature(providerName: ProviderName, model: string, temperature: number): boolean {
  if (providerName === 'kimi-code') return temperature === 1
  const capabilities = piCatalogModelCapabilities(model, providerName)
  if (providerName === 'openai' && capabilities?.api === 'openai-responses' && capabilities.reasoning) return false
  return capabilities?.compat?.supportsTemperature !== false
}

/** Only providers that document these non-standard OpenAI-compatible fields receive them. */
function supportsExtendedSampling(providerName: ProviderName): boolean {
  return providerName === 'openrouter'
}

function withTrailingSlash(value: string): string {
  return value.endsWith('/') ? value : `${value}/`
}

/** Convert the registry's official OpenAI-compatibility root to Gemini's native REST root. */
function nativeGeminiBaseUrl(value: string): string {
  let url: URL
  try {
    url = new URL(value)
  } catch {
    return value
  }
  if (url.hostname !== 'generativelanguage.googleapis.com') {
    return value
  }
  url.pathname = url.pathname.replace(/\/openai\/?$/, '/')
  return url.toString()
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function arrayAt(value: Record<string, unknown>, key: string): unknown[] {
  const item = value[key]
  return Array.isArray(item) ? item : []
}

function stringAt(value: Record<string, unknown> | undefined, key: string): string | undefined {
  const item = value?.[key]
  return typeof item === 'string' ? item : undefined
}

function firstChoice(chunk: Record<string, unknown>): Record<string, unknown> | undefined {
  const choices = arrayAt(chunk, 'choices')
  return choices.length ? asRecord(choices[0]) : undefined
}

/** Some gateways deliver a terminal error payload as an in-stream chunk instead of an HTTP error. */
function throwIfStreamError(chunk: Record<string, unknown>, providerName: string): void {
  const error = chunk.error
  if (error === undefined) {
    return
  }
  if (typeof error === 'string') {
    throw new ProviderError(providerName, `stream returned API error: ${error}`)
  }
  const record = asRecord(error)
  const code = record.code
  const label = typeof code === 'string' || typeof code === 'number' ? ` (${String(code)})` : ''
  const message = stringAt(record, 'message') ?? ''
  throw new ProviderError(providerName, `stream returned API error${label}: ${message || 'unknown error'}`)
}

function mergeToolDeltas(target: Map<number, PendingToolCall>, values: unknown[]): void {
  let lastIndex: number | undefined
  for (const value of values) {
    const delta = asRecord(value) as OpenAiToolCallDelta
    // Providers may omit `index` on continuation chunks; append those to the
    // most recent tool call instead of opening a nameless new entry. An
    // index-less delta that introduces a *different* id or function name is
    // not a continuation, though: a provider streaming parallel calls without
    // indices would otherwise merge every call into one entry, concatenating
    // argument fragments into invalid JSON and overwriting the first name.
    let index = typeof delta.index === 'number'
      ? delta.index
      : lastIndex ?? (target.size ? Math.max(...target.keys()) : 0)
    if (typeof delta.index !== 'number') {
      const candidate = target.get(index)
      const announcedId = typeof delta.id === 'string' && delta.id ? delta.id : undefined
      const announcedName = typeof delta.function?.name === 'string' && delta.function.name
        ? delta.function.name
        : undefined
      const startsNewCall = candidate !== undefined
        && (candidate.id !== undefined || candidate.name !== '')
        && ((announcedId !== undefined && announcedId !== candidate.id)
          || (announcedName !== undefined && announcedName !== candidate.name))
      if (startsNewCall) {
        index = Math.max(...target.keys()) + 1
      }
    }
    const existing: PendingToolCall = target.get(index) ?? { id: undefined, name: '', arguments: '' }
    // A `custom` entry is a grammar-constrained tool call (pi-ai): the input
    // is raw grammar text appended per chunk, never JSON-fragment arguments.
    // A chunk carrying both is a function call and takes that path.
    const customDelta = asRecord(delta.custom)
    const isCustom = Object.keys(customDelta).length > 0 && delta.function === undefined
    const functionDelta = delta.function
    const customName = typeof customDelta.name === 'string' ? customDelta.name : undefined
    target.set(index, {
      id: typeof delta.id === 'string' && delta.id ? delta.id : existing.id,
      name: isCustom
        ? (customName ?? existing.name)
        : (typeof functionDelta?.name === 'string' ? functionDelta.name : existing.name),
      arguments: `${existing.arguments}${typeof functionDelta?.arguments === 'string' ? functionDelta.arguments : ''}`,
      ...(isCustom || existing.customInput !== undefined
        ? { customInput: `${existing.customInput ?? ''}${typeof customDelta.input === 'string' ? customDelta.input : ''}` }
        : {}),
    })
    lastIndex = index
  }
}

function completedToolCalls(
  values: Map<number, PendingToolCall>,
  grammarProperties: ReadonlyMap<string, string> = new Map(),
): ToolCall[] {
  return [...values.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, value]) => {
      if (!value.name) {
        throw new ProviderError('openai-compatible', 'provider returned a tool call without a function name')
      }
      // Custom (grammar) calls carry raw constrained text, surfaced as the
      // tool's single string input property with no JSON parsing (pi-ai).
      const arguments_ = value.customInput !== undefined
        ? ({ [grammarProperties.get(value.name) ?? 'input']: value.customInput } satisfies JsonObject)
        : (() => {
          const partial = parseStreamingJson(value.arguments)
          return isJsonObject(partial) ? partial : parseToolArguments(value.arguments)
        })()
      return {
        id: value.id ?? deterministicToolCallId(value.name, arguments_),
        type: 'function' as const,
        function: {
          name: value.name,
          arguments: arguments_,
        },
      }
    })
}

function validatedOpenAiFinishReason(reason: string | undefined, providerName: ProviderName): string | undefined {
  if (reason === undefined || ['stop', 'end', 'length', 'function_call', 'tool_calls'].includes(reason)) return reason
  throw new ProviderError(providerName, `provider finish_reason: ${reason}`)
}

function openAiUsage(value: Record<string, unknown>): TokenUsage | undefined {
  const inputDetails = asRecord(value.prompt_tokens_details)
  const cacheReadTokens = numberAt(inputDetails, 'cached_tokens')
    ?? numberAt(value, 'prompt_cache_hit_tokens')
    ?? numberAt(value, 'cached_tokens')
  const cacheCreationTokens = numberAt(inputDetails, 'cache_write_tokens')
  const cacheMissTokens = numberAt(value, 'prompt_cache_miss_tokens')
  const inputTokens = numberAt(value, 'prompt_tokens')
    ?? (cacheReadTokens === undefined && cacheMissTokens === undefined
      ? undefined
      : (cacheReadTokens ?? 0) + (cacheMissTokens ?? 0))
  const outputTokens = numberAt(value, 'completion_tokens')
  if (inputTokens === undefined && outputTokens === undefined) {
    return undefined
  }
  const outputDetails = asRecord(value.completion_tokens_details)
  const reasoningTokens = numberAt(outputDetails, 'reasoning_tokens')
  return {
    inputTokens: Math.max(0, (inputTokens ?? 0) - (cacheReadTokens ?? 0) - (cacheCreationTokens ?? 0)),
    outputTokens: outputTokens ?? 0,
    ...(cacheReadTokens === undefined ? {} : { cacheReadTokens }),
    ...(cacheCreationTokens === undefined ? {} : { cacheCreationTokens }),
    ...(reasoningTokens === undefined ? {} : { reasoningTokens }),
  }
}

function openAiMessageContent(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  if (!Array.isArray(value)) {
    return ''
  }
  return value.map(part => {
    const record = asRecord(part)
    return stringAt(record, 'text') ?? stringAt(record, 'content') ?? ''
  }).join('')
}

function parseResponsesCompletion(response: Record<string, unknown>): LlmCompletion {
  throwIfResponsesCompletionError(response)
  const content: string[] = []
  const thinking: string[] = []
  const toolCalls: ToolCall[] = []
  let thinkingSignature: string | undefined
  const output = arrayAt(response, 'output')
  for (const [index, rawItem] of output.entries()) {
    const item = asRecord(rawItem)
    const type = stringAt(item, 'type')
    if (type === 'message') {
      for (const rawPart of arrayAt(item, 'content')) {
        const part = asRecord(rawPart)
        const partType = stringAt(part, 'type')
        if (partType === 'output_text') {
          const text = stringAt(part, 'text')
          if (text) content.push(text)
        } else if (partType === 'reasoning') {
          const text = stringAt(part, 'text')
          if (text) thinking.push(text)
        }
      }
      continue
    }
    if (type === 'reasoning') {
      const summary = arrayAt(item, 'summary')
        .map(part => stringAt(asRecord(part), 'text') ?? '')
        .join('')
      if (summary) thinking.push(summary)
      thinkingSignature = JSON.stringify(item)
      continue
    }
    if (type !== 'function_call' && type !== 'tool_call') {
      continue
    }
    const name = stringAt(item, 'name')
    if (!name) {
      throw new ProviderError('responses', `function call ${index} is missing a name`)
    }
    const rawArguments = item.arguments as string | JsonObject | undefined
    const partialArguments = typeof rawArguments === 'string' ? parseStreamingJson(rawArguments) : rawArguments
    const arguments_ = isJsonObject(partialArguments)
      ? partialArguments
      : parseToolArguments(rawArguments)
    const id = stringAt(item, 'call_id') || stringAt(item, 'id') || deterministicToolCallId(name, arguments_)
    toolCalls.push({ id, type: 'function', function: { name, arguments: arguments_ } })
  }
  const usage = responsesUsage(asRecord(response.usage))
  const serviceTier = stringAt(response, 'service_tier') || undefined
  const status = stringAt(response, 'status') || undefined
  const finishReason = status === 'incomplete'
    ? responsesIncompleteFinishReason(response)
    : toolCalls.length
      ? 'tool_calls'
      : status === 'completed'
        ? 'stop'
        : status
  return {
    content: content.join(''),
    toolCalls,
    ...(finishReason === undefined ? {} : { finishReason }),
    ...(thinking.length ? { thinking: thinking.join('') } : {}),
    ...(thinkingSignature ? { thinkingSignature } : {}),
    ...(usage === undefined
      ? {}
      : { usage: serviceTier === undefined ? usage : { ...usage, serviceTier } }),
  }
}

/** Map non-streaming incomplete responses onto the streaming finish vocabulary. */
function responsesIncompleteFinishReason(response: Record<string, unknown>): string {
  const reason = stringAt(asRecord(response.incomplete_details), 'reason')
  return reason === 'max_output_tokens' ? 'length' : reason ?? 'incomplete'
}

/** Normalize semantic failures returned inside a successful Responses HTTP envelope. */
function throwIfResponsesCompletionError(response: Record<string, unknown>): void {
  const status = stringAt(response, 'status')
  if (status !== 'failed' && status !== 'error') return

  const error = asRecord(response.error)
  const code = error.code
  const label = typeof code === 'string' || typeof code === 'number' ? ` (${String(code)})` : ''
  const message = stringAt(error, 'message') ?? stringAt(response, 'message') ?? 'unknown error'
  throw new ProviderError('responses', `stream returned API error${label}: ${message}`)
}

function responsesUsage(value: Record<string, unknown>): TokenUsage | undefined {
  const inputTokens = numberAt(value, 'input_tokens')
  const outputTokens = numberAt(value, 'output_tokens')
  if (inputTokens === undefined && outputTokens === undefined) {
    return undefined
  }
  const inputDetails = asRecord(value.input_tokens_details)
  const outputDetails = asRecord(value.output_tokens_details)
  const cacheReadTokens = numberAt(value, 'cache_read_tokens') ?? numberAt(inputDetails, 'cached_tokens')
  const cacheCreationTokens = numberAt(value, 'cache_creation_tokens')
    ?? numberAt(inputDetails, 'cache_write_tokens')
    ?? numberAt(outputDetails, 'cache_creation_tokens')
  const reasoningTokens = numberAt(outputDetails, 'reasoning_tokens')
  return {
    inputTokens: freshPromptTokens(inputTokens ?? 0, cacheReadTokens, cacheCreationTokens),
    outputTokens: outputTokens ?? 0,
    ...(cacheReadTokens === undefined ? {} : { cacheReadTokens }),
    ...(cacheCreationTokens === undefined ? {} : { cacheCreationTokens }),
    ...(reasoningTokens === undefined ? {} : { reasoningTokens }),
  }
}

function freshPromptTokens(
  inputTokens: number,
  cacheReadTokens: number | undefined,
  cacheCreationTokens: number | undefined,
): number {
  return Math.max(0, inputTokens - (cacheReadTokens ?? 0) - (cacheCreationTokens ?? 0))
}

function mergeTokenUsage(current: TokenUsage | undefined, next: TokenUsage): TokenUsage {
  if (!current) {
    return next
  }
  const cacheCreationTokens = maxDefined(current.cacheCreationTokens, next.cacheCreationTokens)
  const cacheReadTokens = maxDefined(current.cacheReadTokens, next.cacheReadTokens)
  const reasoningTokens = maxDefined(current.reasoningTokens, next.reasoningTokens)
  return {
    inputTokens: Math.max(current.inputTokens, next.inputTokens),
    outputTokens: Math.max(current.outputTokens, next.outputTokens),
    ...(cacheCreationTokens === undefined ? {} : { cacheCreationTokens }),
    ...(cacheReadTokens === undefined ? {} : { cacheReadTokens }),
    ...(reasoningTokens === undefined ? {} : { reasoningTokens }),
  }
}

function maxDefined(left: number | undefined, right: number | undefined): number | undefined {
  if (left === undefined) {
    return right
  }
  if (right === undefined) {
    return left
  }
  return Math.max(left, right)
}

function numberAt(value: Record<string, unknown>, key: string): number | undefined {
  const item = value[key]
  return typeof item === 'number' && Number.isFinite(item) ? item : undefined
}

function parseJsonObject(data: string, providerName: string): Record<string, unknown> {
  try {
    return asRecord(JSON.parse(data) as unknown)
  } catch (error) {
    throw new ProviderError(providerName, `invalid SSE JSON: ${data.slice(0, 200)}`, error)
  }
}

async function* sseData(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  const parser = new SSEParser()

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }
      parser.feed(decoder.decode(value, { stream: true }))
      for (const event of parser.drain()) {
        yield event.data
      }
    }
    parser.feed(decoder.decode())
    parser.feed('\n\n')
    for (const event of parser.drain()) {
      yield event.data
    }
  } finally {
    try {
      await reader.cancel()
    } catch {
      // Cleanup after an early exit must not mask the primary stream failure.
    }
    reader.releaseLock()
  }
}

export const internalSseData = sseData
