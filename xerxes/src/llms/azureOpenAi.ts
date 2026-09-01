// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Azure OpenAI Responses provider (`/openai/v1/responses?api-version=…`).
 *
 * Azure is the Responses API behind a different front door: key header is
 * `api-key` (never `Authorization: Bearer`), the model field names a
 * *deployment*, the version rides a query parameter, and per Pi's compat data
 * the gateway rejects `service_tier` and `prompt_cache_retention` outright —
 * so this adapter reuses the parent Responses payload semantics minus those
 * fields, and streams through the shared {@link ResponsesEventTranslator}.
 */

import { createHash } from 'node:crypto'

import { parseStreamingJson } from '@earendil-works/pi-ai'

import { ConfigurationError, ProviderError } from '../core/errors.js'
import { ResponsesEventTranslator } from '../streaming/responsesApi.js'
import { deterministicToolCallId } from '../streaming/toolCallIds.js'
import type { ChatMessage, MessageContent } from '../types/messages.js'
import { messageText } from '../types/messages.js'
import type { JsonObject, ToolCall, ToolChoice, ToolDefinition } from '../types/toolCalls.js'
import { isJsonObject, parseToolArguments } from '../types/toolCalls.js'
import type {
  CompletionRequest,
  FetchImplementation,
  LlmClient,
  LlmCompletion,
  LlmDelta,
  TokenUsage,
} from './client.js'
import { internalSseData } from './client.js'
import { isGradedEffort } from './reasoningLevels.js'

/** Diagnostic label carried by every ProviderError this adapter raises. */
const PROVIDER_LABEL = 'azure-openai'

/** Host suffixes Azure routes the Responses v1 surface on. */
const AZURE_HOST_SUFFIXES: readonly string[] = [
  '.openai.azure.com',
  '.cognitiveservices.azure.com',
  '.ai.azure.com',
]

/** Environment shape this adapter reads; injectable for tests. */
export type AzureEnv = Readonly<Record<string, string | undefined>>

export interface AzureOpenAiOptions {
  readonly apiKey?: string
  readonly resourceName?: string
  /** Full Responses root; wins over the resource-name default. */
  readonly baseUrl?: string
  /** `api-version` query parameter; defaults to `AZURE_OPENAI_API_VERSION` then `v1`. */
  readonly apiVersion?: string
  /** Static per-model deployment names, checked before the environment map. */
  readonly deploymentName?: string
  /** Per-model deployment overrides resolved at request time. */
  readonly deploymentNameMap?: Readonly<Record<string, string>>
  readonly fetchImplementation?: FetchImplementation
}

/** Trim a possibly-ambient string option; empty and whitespace-only mean absent. */
function stringOption(value: string | undefined): string | undefined {
  const trimmed = value?.trim()
  return trimmed ? trimmed : undefined
}

/** Strip a routing prefix (`azure/gpt-4o` → `gpt-4o`) for model-id comparisons. */
function bareModelId(model: string): string {
  const trimmed = model.trim()
  const slash = trimmed.lastIndexOf('/')
  return slash >= 0 ? trimmed.slice(slash + 1) : trimmed
}

/**
 * Resolve the Responses root.
 *
 * Precedence: explicit `options.baseUrl`, then `AZURE_OPENAI_BASE_URL`, then
 * the resource-name default. Documented Azure host suffixes are normalized to
 * the `/openai/v1` path because their older paths (`/openai/deployments/…`)
 * belong to the legacy completions surface this adapter does not speak.
 */
export function resolveAzureBaseUrl(options: AzureOpenAiOptions, env: AzureEnv): string {
  const configured = stringOption(options.baseUrl) ?? stringOption(env.AZURE_OPENAI_BASE_URL)
  if (configured) return normalizeAzureBaseUrl(configured)
  const resourceName = stringOption(options.resourceName) ?? stringOption(env.AZURE_OPENAI_RESOURCE_NAME)
  if (!resourceName) {
    throw new ConfigurationError(
      'base_url',
      'Azure OpenAI requires options.baseUrl, AZURE_OPENAI_BASE_URL, or a resource name '
        + '(options.resourceName / AZURE_OPENAI_RESOURCE_NAME)',
    )
  }
  return `https://${resourceName}.openai.azure.com/openai/v1`
}

function normalizeAzureBaseUrl(value: string): string {
  let url: URL
  try {
    url = new URL(value)
  } catch {
    throw new ConfigurationError('base_url', `Azure OpenAI base URL is not a valid URL: ${value}`)
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') {
    throw new ConfigurationError('base_url', `Azure OpenAI base URL must use http(s): ${value}`)
  }
  if (AZURE_HOST_SUFFIXES.some(suffix => url.hostname.toLowerCase().endsWith(suffix))) {
    url.pathname = '/openai/v1'
    url.search = ''
    url.hash = ''
  }
  return url.toString().replace(/\/+$/, '')
}

/**
 * Resolve the deployment name a request's `model` field must carry.
 *
 * Precedence: explicit `options.deploymentName`, then the
 * `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` environment map
 * (`"modelId=deployment,modelId2=deployment2"`), then the bare model id.
 */
export function resolveAzureDeployment(model: string, options: AzureOpenAiOptions, env: AzureEnv): string {
  const explicit = stringOption(options.deploymentName)
  if (explicit) return explicit
  const mapped = deploymentFromMap(stringOption(env.AZURE_OPENAI_DEPLOYMENT_NAME_MAP), model)
  if (mapped) return mapped
  const staticMapped = options.deploymentNameMap?.[bareModelId(model)] ?? options.deploymentNameMap?.[model.trim()]
  return stringOption(staticMapped) ?? bareModelId(model)
}

function deploymentFromMap(raw: string | undefined, model: string): string | undefined {
  if (!raw) return undefined
  const identifiers = new Set([model.trim(), bareModelId(model)].filter(Boolean))
  for (const entry of raw.split(',')) {
    const separator = entry.indexOf('=')
    if (separator <= 0) continue
    const key = entry.slice(0, separator).trim()
    const deployment = entry.slice(separator + 1).trim()
    if (key && deployment && identifiers.has(key)) return deployment
  }
  return undefined
}

/** Build the Responses payload with the parent's field semantics, minus Azure-rejected fields. */
function azureResponsesPayload(
  request: CompletionRequest,
  deployment: string,
  stream: boolean,
): Record<string, unknown> {
  const systemPrompt = request.messages
    .filter(message => message.role === 'system')
    .map(messageText)
    .filter(Boolean)
    .join('\n\n')
  const payload: Record<string, unknown> = {
    model: deployment,
    input: messagesToResponsesInput(request.messages.filter(message => message.role !== 'system')),
    stream,
    store: false,
    ...(systemPrompt ? { instructions: systemPrompt } : {}),
  }
  if (request.temperature !== undefined) payload.temperature = request.temperature
  if (request.maxTokens !== undefined) payload.max_output_tokens = Math.max(16, request.maxTokens)
  if (request.topP !== undefined) payload.top_p = request.topP
  const effort = request.thinking?.effort
  if (isGradedEffort(effort)) payload.reasoning = { effort }
  if (effort) payload.include = ['reasoning.encrypted_content']
  if (request.tools?.length) {
    payload.tools = request.tools.map(azureToolDefinition)
    if (request.toolChoice) payload.tool_choice = azureToolChoice(request.toolChoice)
  }
  // Long prefixes cache automatically; the key only routes repeats to the
  // machine holding that prefix. No system-segment fallback here: the parent's
  // host-scoped derivation is keyed to openai hosts, and Azure gateways vary.
  if (request.sessionId) payload.prompt_cache_key = azureCacheKey(request.sessionId)
  // `service_tier` and `prompt_cache_retention` are deliberately never sent:
  // Pi's compat data shows the Azure gateway rejects both with a 400.
  return payload
}

function azureToolDefinition(tool: ToolDefinition): Record<string, unknown> {
  return {
    type: 'function',
    name: tool.function.name,
    description: tool.function.description,
    parameters: tool.function.parameters,
  }
}

function azureToolChoice(choice: ToolChoice): string {
  return choice === 'any' ? 'required' : choice
}

/** Stable cache key: the session id when short, its digest when long. */
function azureCacheKey(sessionId: string): string {
  const normalized = sessionId.trim()
  return normalized.length <= 64 ? normalized : createHash('sha256').update(normalized).digest('hex')
}

/** Translate the neutral transcript into Responses input items (parent semantics). */
function messagesToResponsesInput(messages: readonly ChatMessage[]): Record<string, unknown>[] {
  const input: Record<string, unknown>[] = []
  for (const message of messages) {
    if (message.role === 'assistant') {
      const reasoningItem = azureReasoningItem(message.thinking_signature)
      if (reasoningItem) input.push(reasoningItem)
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
    input.push({ role: message.role, content: azureMessageContent(message.content) })
  }
  return input
}

function azureMessageContent(content: MessageContent): unknown {
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

function azureReasoningItem(signature: string | undefined): Record<string, unknown> | undefined {
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
 * Native-fetch Azure OpenAI Responses client sharing the neutral delta stream.
 */
export class AzureOpenAiClient implements LlmClient {
  private readonly apiKey: string
  private readonly baseUrl: string
  private readonly apiVersion: string
  private readonly options: AzureOpenAiOptions
  private readonly env: AzureEnv
  private readonly fetchImplementation: FetchImplementation

  constructor(options: AzureOpenAiOptions = {}, env: AzureEnv = process.env) {
    this.options = options
    this.env = env
    this.apiKey = stringOption(options.apiKey) ?? stringOption(env.AZURE_OPENAI_API_KEY) ?? ''
    if (!this.apiKey) {
      throw new ConfigurationError(
        'api_key',
        'Azure OpenAI requires an API key (options.apiKey or AZURE_OPENAI_API_KEY)',
      )
    }
    this.baseUrl = resolveAzureBaseUrl(options, env)
    this.apiVersion = stringOption(options.apiVersion) ?? stringOption(env.AZURE_OPENAI_API_VERSION) ?? 'v1'
    this.fetchImplementation = options.fetchImplementation ?? fetch
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    const response = await this.fetchImplementation(this.endpoint(), {
      method: 'POST',
      headers: azureHeaders(this.apiKey, 'application/json'),
      body: JSON.stringify(azureResponsesPayload(request, this.deploymentOf(request), false)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw azureHttpError(
        `completion request failed (${response.status}): ${body.slice(0, 4_096)}`,
        response,
      )
    }
    return parseAzureCompletion(parseAzureJson(await response.text()))
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const response = await this.fetchImplementation(this.endpoint(), {
      method: 'POST',
      headers: azureHeaders(this.apiKey, 'text/event-stream'),
      body: JSON.stringify(azureResponsesPayload(request, this.deploymentOf(request), true)),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw azureHttpError(`stream request failed (${response.status}): ${body.slice(0, 4_096)}`, response)
    }
    if (!response.body) {
      throw new ProviderError(PROVIDER_LABEL, 'Azure OpenAI stream returned no response body')
    }
    const translator = new ResponsesEventTranslator()
    for await (const data of internalSseData(response.body)) {
      if (data === '[DONE]') break
      for (const delta of translator.translate(parseAzureJson(data))) yield delta
    }
    translator.finish()
  }

  private endpoint(): string {
    const url = new URL(`${this.baseUrl}/responses`)
    url.searchParams.set('api-version', this.apiVersion)
    return url.toString()
  }

  private deploymentOf(request: CompletionRequest): string {
    return resolveAzureDeployment(request.model, this.options, this.env)
  }
}

function azureHeaders(apiKey: string, accept: string): Record<string, string> {
  // Azure authenticates with the `api-key` header; it does not accept
  // `Authorization: Bearer` on API-key resources.
  return {
    Accept: accept,
    'Content-Type': 'application/json',
    'api-key': apiKey,
  }
}

/** Preserve HTTP retry metadata without mixing provider headers into user-facing messages. */
function azureHttpError(message: string, response: Response): ProviderError {
  const retryAfterMilliseconds = parseRetryAfterHeader(response.headers.get('retry-after-ms'))
  const retryAfterSeconds = retryAfterMilliseconds === undefined
    ? parseRetryAfterHeader(response.headers.get('retry-after'))
    : retryAfterMilliseconds / 1_000
  return new ProviderError(PROVIDER_LABEL, message, undefined, {
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

function parseAzureJson(data: string): Record<string, unknown> {
  try {
    return asRecord(JSON.parse(data) as unknown)
  } catch (error) {
    throw new ProviderError(PROVIDER_LABEL, `invalid SSE JSON: ${data.slice(0, 200)}`, error)
  }
}

function parseAzureCompletion(response: Record<string, unknown>): LlmCompletion {
  throwIfAzureCompletionError(response)
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
      throw new ProviderError(PROVIDER_LABEL, `function call ${index} is missing a name`)
    }
    const rawArguments = item.arguments as string | JsonObject | undefined
    const partialArguments = typeof rawArguments === 'string' ? parseStreamingJson(rawArguments) : rawArguments
    const arguments_ = isJsonObject(partialArguments)
      ? partialArguments
      : parseToolArguments(rawArguments)
    const id = stringAt(item, 'call_id') || stringAt(item, 'id') || deterministicToolCallId(name, arguments_)
    toolCalls.push({ id, type: 'function', function: { name, arguments: arguments_ } })
  }
  const usage = azureUsage(asRecord(response.usage))
  const serviceTier = stringAt(response, 'service_tier') || undefined
  const status = stringAt(response, 'status') || undefined
  const finishReason = status === 'incomplete'
    ? azureIncompleteFinishReason(response)
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
function azureIncompleteFinishReason(response: Record<string, unknown>): string {
  const reason = stringAt(asRecord(response.incomplete_details), 'reason')
  return reason === 'max_output_tokens' ? 'length' : reason ?? 'incomplete'
}

/** Normalize semantic failures returned inside a successful HTTP envelope. */
function throwIfAzureCompletionError(response: Record<string, unknown>): void {
  const status = stringAt(response, 'status')
  if (status !== 'failed' && status !== 'error') return
  const error = asRecord(response.error)
  const code = error.code
  const label = typeof code === 'string' || typeof code === 'number' ? ` (${String(code)})` : ''
  const message = stringAt(error, 'message') ?? stringAt(response, 'message') ?? 'unknown error'
  throw new ProviderError(PROVIDER_LABEL, `Azure OpenAI returned API error${label}: ${message}`)
}

function azureUsage(value: Record<string, unknown>): TokenUsage | undefined {
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
    // The reported input total covers cached and cache-written tokens too;
    // fresh input is what the caller is uniquely billed for.
    inputTokens: Math.max(0, (inputTokens ?? 0) - (cacheReadTokens ?? 0) - (cacheCreationTokens ?? 0)),
    outputTokens: outputTokens ?? 0,
    ...(cacheReadTokens === undefined ? {} : { cacheReadTokens }),
    ...(cacheCreationTokens === undefined ? {} : { cacheCreationTokens }),
    ...(reasoningTokens === undefined ? {} : { reasoningTokens }),
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function arrayAt(value: Record<string, unknown>, key: string): readonly unknown[] {
  const item = value[key]
  return Array.isArray(item) ? item : []
}

function stringAt(value: Record<string, unknown>, key: string): string | undefined {
  const item = value[key]
  return typeof item === 'string' && item ? item : undefined
}

function numberAt(value: Record<string, unknown>, key: string): number | undefined {
  const item = value[key]
  return typeof item === 'number' && Number.isFinite(item) ? item : undefined
}
