// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError, ProviderError } from '../core/errors.js'
import { isPluginLlmProviderFactory } from '../extensions/plugins.js'
import type {
  PluginLlmProviderFactory,
  PluginLlmProviderOptions,
  PluginLlmProviderRegistry,
} from '../extensions/plugins.js'
import { ResponsesEventTranslator } from '../streaming/responsesApi.js'
import { deterministicToolCallId } from '../streaming/toolCallIds.js'
import { SSEParser } from '../streaming/sse.js'
import { AnthropicMessagesClient } from './anthropic.js'
import { GeminiClient } from './gemini.js'
import { OllamaClient } from './ollama.js'
import type { ChatMessage, MessageContent } from '../types/messages.js'
import { messageText, messagesToOpenAi } from '../types/messages.js'
import type { JsonObject, ToolCall, ToolChoice, ToolDefinition } from '../types/toolCalls.js'
import { parseToolArguments } from '../types/toolCalls.js'
import {
  type ProviderName,
  type ProviderOverrides,
  bareModel,
  getApiKey,
  getContextLimit,
  getProviderConfig,
  isProviderName,
  providerDefaultHeaders,
  providerModel,
  resolveProvider,
} from './providerRegistry.js'
import { DEFAULT_TEMPERATURE } from './samplingDefaults.js'

export interface TokenUsage {
  readonly cacheCreationTokens?: number
  readonly cacheReadTokens?: number
  readonly inputTokens: number
  readonly outputTokens: number
  /** Provider-reported reasoning tokens; absent when the provider does not expose them. */
  readonly reasoningTokens?: number
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
  readonly repetitionPenalty?: number
  readonly stop?: readonly string[]
  readonly temperature?: number
  readonly toolChoice?: ToolChoice
  readonly tools?: readonly ToolDefinition[]
  readonly topK?: number
  readonly topP?: number
}

/** Provider-neutral incremental response from a model adapter. */
export interface LlmDelta {
  readonly content?: string
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
  readonly maxModelLen: number
  readonly maxTokens: number
  readonly model: string
  readonly provider: ProviderName
  readonly stream: boolean
  readonly temperature: number
}

/** Per-call settings included in a model metadata summary. */
export interface LlmModelInfoOptions {
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
}

/** Options used by the native client factory, including optional plugin provider lookup. */
export interface LlmClientFactoryOptions extends Omit<OpenAiCompatibleClientOptions, 'providerName'> {
  readonly pluginRegistry?: PluginLlmProviderRegistry
}

interface OpenAiToolCallDelta {
  readonly function?: {
    readonly arguments?: string
    readonly name?: string
  }
  readonly id?: string
  readonly index?: number
}

interface PendingToolCall {
  arguments: string
  id: string | undefined
  name: string
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

  constructor(options: OpenAiCompatibleClientOptions) {
    const providerConfig = getProviderConfig(options.providerName)
    this.providerName = options.providerName
    this.apiKey = options.apiKey ?? getApiKey(options.providerName)
    this.baseUrl = options.baseUrl ?? providerConfig.baseUrl ?? ''
    this.fetchImplementation = options.fetchImplementation ?? fetch

    if (!this.baseUrl) {
      throw new ConfigurationError('base_url', `No base URL is configured for ${options.providerName}`)
    }
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    const endpoint = new URL('chat/completions', withTrailingSlash(this.baseUrl)).toString()
    const response = await this.fetchImplementation(endpoint, {
      method: 'POST',
      headers: openAiCompatibleHeaders(this.providerName, this.apiKey, 'application/json'),
      body: JSON.stringify(openAiCompatiblePayload(request, this.providerName, false)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw new ProviderError(
        this.providerName,
        `completion request failed (${response.status}): ${body.slice(0, 4_096)}`,
      )
    }

    const responseBody = parseJsonObject(await response.text(), this.providerName)
    const choice = firstChoice(responseBody)
    if (!choice) {
      throw new ProviderError(this.providerName, 'completion response did not include a choice')
    }
    const message = asRecord(choice.message)
    const pendingToolCalls = new Map<number, PendingToolCall>()
    mergeToolDeltas(pendingToolCalls, arrayAt(message, 'tool_calls'))
    const content = openAiMessageContent(message.content)
    const thinking = stringAt(message, 'reasoning_content') ?? stringAt(message, 'reasoning')
    const finishReason = stringAt(choice, 'finish_reason')
    const usage = openAiUsage(asRecord(responseBody.usage))

    return {
      content,
      toolCalls: completedToolCalls(pendingToolCalls),
      ...(finishReason === undefined ? {} : { finishReason }),
      ...(thinking === undefined ? {} : { thinking }),
      ...(usage === undefined ? {} : { usage }),
    }
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const endpoint = new URL('chat/completions', withTrailingSlash(this.baseUrl)).toString()
    const response = await this.fetchImplementation(endpoint, {
      method: 'POST',
      headers: openAiCompatibleHeaders(this.providerName, this.apiKey, 'text/event-stream'),
      body: JSON.stringify(openAiCompatiblePayload(request, this.providerName, true)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw new ProviderError(
        this.providerName,
        `stream request failed (${response.status}): ${body.slice(0, 4_096)}`,
      )
    }
    if (!response.body) {
      throw new ProviderError(this.providerName, 'stream request returned no response body')
    }

    const pendingToolCalls = new Map<number, PendingToolCall>()
    let emittedToolCalls = false
    for await (const data of sseData(response.body)) {
      if (data === '[DONE]') {
        break
      }
      const chunk = parseJsonObject(data, this.providerName)
      throwIfStreamError(chunk, this.providerName)
      const choice = firstChoice(chunk)
      const delta = asRecord(choice?.delta)
      const content = stringAt(delta, 'content')
      const thinking = stringAt(delta, 'reasoning_content') ?? stringAt(delta, 'reasoning')
      mergeToolDeltas(pendingToolCalls, arrayAt(delta, 'tool_calls'))
      const finishReason = stringAt(choice, 'finish_reason')
      const usage = openAiUsage(asRecord(chunk.usage))

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
      }
      if (finishReason && pendingToolCalls.size) {
        event.toolCalls = completedToolCalls(pendingToolCalls)
        emittedToolCalls = true
      }
      if (Object.keys(event).length) {
        yield event
      }
    }

    if (!emittedToolCalls && pendingToolCalls.size) {
      yield { toolCalls: completedToolCalls(pendingToolCalls) }
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

  constructor(options: OpenAiCompatibleClientOptions) {
    const providerConfig = getProviderConfig(options.providerName)
    this.providerName = options.providerName
    this.apiKey = options.apiKey ?? getApiKey(options.providerName)
    this.baseUrl = options.baseUrl ?? providerConfig.baseUrl ?? ''
    this.fetchImplementation = options.fetchImplementation ?? fetch
    if (!this.baseUrl) {
      throw new ConfigurationError('base_url', 'No base URL is configured for ' + options.providerName)
    }
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    const endpoint = new URL('responses', withTrailingSlash(this.baseUrl)).toString()
    const response = await this.fetchImplementation(endpoint, {
      method: 'POST',
      headers: responsesHeaders(this.providerName, this.apiKey, 'application/json'),
      body: JSON.stringify(responsesPayload(request, this.providerName, false)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw new ProviderError(
        this.providerName,
        'Responses API completion request failed (' + response.status + '): ' + body.slice(0, 4_096),
      )
    }
    return parseResponsesCompletion(parseJsonObject(await response.text(), this.providerName))
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const endpoint = new URL('responses', withTrailingSlash(this.baseUrl)).toString()
    const response = await this.fetchImplementation(endpoint, {
      method: 'POST',
      headers: responsesHeaders(this.providerName, this.apiKey, 'text/event-stream'),
      body: JSON.stringify(responsesPayload(request, this.providerName, true)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw new ProviderError(
        this.providerName,
        'Responses API stream request failed (' + response.status + '): ' + body.slice(0, 4_096),
      )
    }
    if (!response.body) {
      throw new ProviderError(this.providerName, 'Responses API stream returned no response body')
    }

    const translator = new ResponsesEventTranslator()
    for await (const data of sseData(response.body)) {
      if (data === '[DONE]') break
      const event = parseJsonObject(data, this.providerName)
      for (const delta of translator.translate(event)) yield delta
    }
    const final = translator.finish()
    if (final) yield final
  }
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
    return new AnthropicMessagesClient({
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
      ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
      ...(options.promptCaching === undefined ? {} : { promptCaching: options.promptCaching }),
    })
  }
  if (providerConfig.transport !== 'openai') {
    throw new ConfigurationError('provider', `${providerName} requires its dedicated adapter.`)
  }
  const useResponsesApi = options.responsesApi === true || overrides.responses_api === true
  if (useResponsesApi) {
    return new ResponsesApiClient({
      ...options,
      providerName,
      ...(configuredApiKey ? { apiKey: configuredApiKey } : {}),
      ...(configuredBaseUrl ? { baseUrl: configuredBaseUrl } : {}),
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

/** Reject execution that would otherwise guess a provider/model from an empty ID. */
export function requireConfiguredModel(model: string | undefined): string {
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
 * Generate one complete, provider-neutral response.
 *
 * Dedicated adapters can make a native non-streaming request through their optional
 * `complete` method. Stream-only plugins and adapters are collected losslessly from
 * the same delta vocabulary instead, so adding this API does not invalidate them.
 */
export async function completeLlm(
  client: LlmClient,
  request: CompletionRequest,
  signal?: AbortSignal,
): Promise<LlmCompletion> {
  if (typeof client.complete === 'function') {
    return client.complete(request, signal)
  }
  return collectLlmCompletion(client.stream(request, signal))
}

/** Collect a provider-neutral stream into a non-streaming completion result. */
export async function collectLlmCompletion(stream: AsyncIterable<LlmDelta>): Promise<LlmCompletion> {
  const content: string[] = []
  const thinking: string[] = []
  const toolCalls = new Map<string, ToolCall>()
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
      toolCalls.set(toolCall.id, toolCall)
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
    maxModelLen: getContextLimit(model, overrides),
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

function responsesToolDefinition(tool: ToolDefinition): Record<string, unknown> {
  return {
    type: 'function',
    name: tool.function.name,
    description: tool.function.description,
    parameters: tool.function.parameters,
  }
}

function responsesToolChoice(choice: ToolChoice): string {
  if (choice === 'any') return 'required'
  return choice
}

/**
 * Translate the neutral transcript into Responses API input items.
 *
 * Assistant tool calls become `function_call` items and tool replies become
 * `function_call_output` items; multipart user content uses `input_text` and
 * `input_image` parts. Chat-completions message shapes are not valid input.
 */
function messagesToResponsesInput(messages: readonly ChatMessage[]): Record<string, unknown>[] {
  const input: Record<string, unknown>[] = []
  for (const message of messages) {
    if (message.role === 'assistant') {
      const text = messageText(message)
      if (text) {
        input.push({ role: 'assistant', content: text })
      }
      for (const call of message.tool_calls ?? []) {
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
      input.push({ type: 'function_call_output', call_id: message.tool_call_id, output: message.content })
      continue
    }
    input.push({ role: message.role, content: responsesMessageContent(message.content) })
  }
  return input
}

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
  const payload: Record<string, unknown> = {
    model: providerModel(request.model, providerName),
    input: messagesToResponsesInput(request.messages),
    stream,
  }
  addSampling(payload, request, providerName)
  if (request.tools?.length) {
    payload.tools = request.tools.map(responsesToolDefinition)
    if (request.toolChoice) payload.tool_choice = responsesToolChoice(request.toolChoice)
  }
  return payload
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

function openAiCompatiblePayload(
  request: CompletionRequest,
  providerName: ProviderName,
  stream: boolean,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    model: providerModel(request.model, providerName),
    messages: messagesToOpenAi(request.messages),
    stream,
  }
  addSampling(payload, request, providerName)
  if (request.tools?.length) {
    payload.tools = request.tools
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

function addSampling(
  payload: Record<string, unknown>,
  request: CompletionRequest,
  providerName: ProviderName,
): void {
  if (request.temperature !== undefined && supportsTemperature(providerName, request.temperature)) {
    payload.temperature = request.temperature
  }
  if (request.maxTokens !== undefined) {
    payload.max_tokens = request.maxTokens
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
  if (request.extraBody) {
    Object.assign(payload, request.extraBody)
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

/** Kimi Code fixes temperature at 1 and rejects every other explicit value. */
function supportsTemperature(providerName: ProviderName, temperature: number): boolean {
  return providerName !== 'kimi-code' || temperature === 1
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
    // most recent tool call instead of opening a nameless new entry.
    const index = typeof delta.index === 'number'
      ? delta.index
      : lastIndex ?? (target.size ? Math.max(...target.keys()) : 0)
    const existing: PendingToolCall = target.get(index) ?? { id: undefined, name: '', arguments: '' }
    const functionDelta = delta.function
    target.set(index, {
      id: typeof delta.id === 'string' && delta.id ? delta.id : existing.id,
      name: typeof functionDelta?.name === 'string' ? functionDelta.name : existing.name,
      arguments: `${existing.arguments}${typeof functionDelta?.arguments === 'string' ? functionDelta.arguments : ''}`,
    })
    lastIndex = index
  }
}

function completedToolCalls(values: Map<number, PendingToolCall>): ToolCall[] {
  return [...values.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, value]) => {
      if (!value.name) {
        throw new ProviderError('openai-compatible', 'provider returned a tool call without a function name')
      }
      const arguments_ = parseToolArguments(value.arguments)
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

function openAiUsage(value: Record<string, unknown>): TokenUsage | undefined {
  const inputTokens = numberAt(value, 'prompt_tokens')
  const outputTokens = numberAt(value, 'completion_tokens')
  if (inputTokens === undefined && outputTokens === undefined) {
    return undefined
  }
  const outputDetails = asRecord(value.completion_tokens_details)
  const reasoningTokens = numberAt(outputDetails, 'reasoning_tokens')
  return {
    inputTokens: inputTokens ?? 0,
    outputTokens: outputTokens ?? 0,
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
  const content: string[] = []
  const thinking: string[] = []
  const toolCalls: ToolCall[] = []
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
      continue
    }
    if (type !== 'function_call' && type !== 'tool_call') {
      continue
    }
    const name = stringAt(item, 'name')
    if (!name) {
      throw new ProviderError('responses', `function call ${index} is missing a name`)
    }
    const arguments_ = parseToolArguments(item.arguments as string | JsonObject | undefined)
    const id = stringAt(item, 'call_id') || stringAt(item, 'id') || deterministicToolCallId(name, arguments_)
    toolCalls.push({ id, type: 'function', function: { name, arguments: arguments_ } })
  }
  const usage = responsesUsage(asRecord(response.usage))
  const finishReason = stringAt(response, 'status') || undefined
  return {
    content: content.join(''),
    toolCalls,
    ...(finishReason === undefined ? {} : { finishReason }),
    ...(thinking.length ? { thinking: thinking.join('') } : {}),
    ...(usage === undefined ? {} : { usage }),
  }
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
    ?? numberAt(outputDetails, 'cache_creation_tokens')
  const reasoningTokens = numberAt(outputDetails, 'reasoning_tokens')
  return {
    inputTokens: inputTokens ?? 0,
    outputTokens: outputTokens ?? 0,
    ...(cacheReadTokens === undefined ? {} : { cacheReadTokens }),
    ...(cacheCreationTokens === undefined ? {} : { cacheCreationTokens }),
    ...(reasoningTokens === undefined ? {} : { reasoningTokens }),
  }
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
