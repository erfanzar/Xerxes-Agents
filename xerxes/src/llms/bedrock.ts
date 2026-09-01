// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Amazon Bedrock Converse provider (pi-ai `bedrock-converse-stream` parity).
 *
 * Like pi-ai, all AWS mechanics — SigV4 signing, the SDK credential chain
 * (env keys, `AWS_PROFILE` config files, ECS/web-identity roles), bearer-token
 * auth, region and endpoint selection — are delegated to
 * `@aws-sdk/client-bedrock-runtime`; this adapter mirrors pi-ai's resolution
 * order on top of it and translates Converse stream events into Xerxes's
 * neutral deltas. Auth resolution order (pi-ai `bedrockAuth.resolve`):
 * injected key → `AWS_BEARER_TOKEN_BEDROCK` → `AWS_PROFILE` →
 * `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` (+ optional session token) →
 * container/web-identity chain (SDK default).
 */

import {
  BedrockRuntimeClient,
  BedrockRuntimeServiceException,
  CachePointType,
  CacheTTL,
  ConverseStreamCommand,
  ImageFormat,
} from '@aws-sdk/client-bedrock-runtime'
import { NodeHttpHandler } from '@smithy/node-http-handler'
import { parseStreamingJson } from '@earendil-works/pi-ai'

import { ProviderError } from '../core/errors.js'
import { deterministicToolCallId } from '../streaming/toolCallIds.js'
import type { ContentPart, MessageContent } from '../types/messages.js'
import { messageText } from '../types/messages.js'
import type { ToolCall, ToolDefinition } from '../types/toolCalls.js'
import { isJsonObject } from '../types/toolCalls.js'
import type { CompletionRequest, LlmClient, LlmCompletion, LlmDelta, ThinkingRequest, TokenUsage } from './client.js'
import { collectLlmCompletion } from './client.js'
import { piCatalogModelCapabilities, type PiModelCapabilities } from './piModelCatalog.js'

/** Diagnostic label carried by every ProviderError this adapter raises. */
const PROVIDER_LABEL = 'amazon-bedrock'

/** Placeholder Bedrock requires for otherwise-blank message content. */
const EMPTY_TEXT_PLACEHOLDER = '<empty>'

/** Matches the placeholder the Anthropic path uses for redacted thinking. */
const REDACTED_THINKING_PLACEHOLDER = '[Reasoning redacted]'

/** pi-ai: tokens always left for the answer when a budget shares the response ceiling. */
const MIN_ANSWER_TOKENS = 1024

/** pi-ai default thinking budgets when the caller gives no explicit budget. */
function defaultThinkingBudget(effort: string | undefined): number {
  if (effort === 'minimal') return 1024
  if (effort === 'low') return 2048
  if (effort === 'medium') return 8192
  return 16384
}

/** Environment shape this adapter reads; injectable for tests. */
export type BedrockEnv = Readonly<Record<string, string | undefined>>

/** Human-readable prefixes for Bedrock SDK exception names (pi-ai parity). */
const BEDROCK_ERROR_PREFIXES: Readonly<Record<string, string>> = {
  InternalServerException: 'Internal server error',
  ModelStreamErrorException: 'Model stream error',
  ValidationException: 'Validation error',
  ThrottlingException: 'Throttling error',
  ServiceUnavailableException: 'Service unavailable',
}

/** Point at the AWS docs when a model rejects the account's data-retention mode. */
const BEDROCK_DATA_RETENTION_DOCS_URL = 'https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html'

/** Options the Bedrock client resolves per request. */
export interface BedrockClientOptions {
  /** Bedrock bearer token (API-key auth); wins over ambient AWS credentials. */
  readonly apiKey?: string
  /** Override the model catalog endpoint (custom VPC/proxy endpoints). */
  readonly baseUrl?: string
  /** Injectable environment; defaults to `process.env`. */
  readonly env?: BedrockEnv
  /** Override the client constructor; injectable for deterministic tests. */
  readonly createClient?: BedrockClientFactory
}

/**
 * Transport seam for tests: the resolved config in, a Converse stream out.
 * Production wires this to {@link BedrockRuntimeClient}; tests script events.
 */
export type BedrockClientFactory = (config: BedrockResolvedConfig) => BedrockConverseTransport

export interface BedrockConverseTransport {
  send(input: BedrockConverseInput, signal?: AbortSignal): Promise<{ readonly stream: AsyncIterable<unknown> }>
}

/**
 * One assembled Converse request body, plain JSON. The adapter builds plain
 * records and the production transport casts once at the SDK boundary: the
 * generated block unions are stricter than the wire (optional members under
 * `exactOptionalPropertyTypes`) without adding runtime information.
 */
export type BedrockConverseInput = Record<string, unknown>

/**
 * One Converse content block, assembled as plain JSON. The SDK's generated
 * block union is strict about optional members under
 * `exactOptionalPropertyTypes`, so the adapter builds plain records and casts
 * once at the SDK boundary.
 */
export type BedrockContentBlock = Record<string, unknown>

/** One Converse message: role plus plain content blocks. */
export interface BedrockMessage {
  readonly content: readonly BedrockContentBlock[]
  readonly role: 'assistant' | 'user'
}

/** The SDK client settings this adapter resolved, asserted in tests. */
export interface BedrockResolvedConfig {
  readonly authSchemePreference?: readonly string[]
  readonly credentials?: {
    readonly accessKeyId: string
    readonly secretAccessKey: string
    readonly sessionToken?: string
  }
  readonly endpoint?: string
  readonly profile?: string
  readonly region?: string
  /** `'http1'` pins `NodeHttpHandler`; `'http2'` keeps the SDK default handler. */
  readonly requestHandler?: 'http1' | 'http2'
  readonly token?: { readonly token: string }
}

interface ResolvedBedrockModel {
  readonly adaptiveThinking: boolean
  readonly capabilities: PiModelCapabilities | undefined
  readonly isClaude: boolean
  readonly modelId: string
  readonly supportsCaching: boolean
}

/** Remove unpaired Unicode surrogates that break JSON serialization (pi-ai parity). */
function sanitizeSurrogates(text: string): string {
  return text.replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, '')
}

/** pi-ai `normalizeToolCallId`: Bedrock rejects ids outside `^[a-zA-Z0-9_-]+$` (≤64). */
export function normalizeBedrockToolCallId(id: string): string {
  const sanitized = id.replaceAll(/[^a-zA-Z0-9_-]/g, '_')
  return sanitized.length > 64 ? sanitized.slice(0, 64) : sanitized
}

function envValue(env: BedrockEnv, name: string): string | undefined {
  const value = env[name]?.trim()
  return value ? value : undefined
}

/** pi-ai adaptive-thinking check: Opus 4.6+, Sonnet 4.6+ (model-id match). */
function supportsAdaptiveThinking(modelId: string): boolean {
  const lower = modelId.toLowerCase()
  return ['opus-4-6', 'opus-4-7', 'opus-4-8', 'opus-5', 'sonnet-4-6', 'sonnet-5', 'fable-5']
    .some(marker => lower.includes(marker))
}

/** pi-ai: only Anthropic Claude models accept the reasoning signature field. */
function isClaudeModelId(modelId: string): boolean {
  const lower = modelId.toLowerCase()
  return lower.includes('anthropic.claude') || lower.includes('anthropic/claude')
}

/** pi-ai `supportsPromptCaching`: Claude 3.5 Haiku, 3.7 Sonnet, 4.x, 5.x only. */
function supportsPromptCaching(modelId: string, env: BedrockEnv): boolean {
  const lower = modelId.toLowerCase()
  if (!lower.includes('claude')) {
    // Application inference profiles hide the model name in their ARN; allow
    // an explicit opt-in the way pi-ai does.
    return envValue(env, 'AWS_BEDROCK_FORCE_CACHE') === '1'
  }
  if (['fable-5', 'opus-5', 'sonnet-5'].some(marker => lower.includes(marker))) return true
  if (/-4-/.test(lower)) return true
  if (lower.includes('claude-3-7-sonnet')) return true
  if (lower.includes('claude-3-5-haiku')) return true
  return false
}

/** Resolve the model id + catalog capabilities that shape one request. */
export function resolveBedrockModel(request: CompletionRequest, env: BedrockEnv): ResolvedBedrockModel {
  const configured = request.model.trim()
  const slash = configured.indexOf('/')
  const modelId = slash >= 0 ? configured.slice(slash + 1) : configured
  const capabilities = piCatalogModelCapabilities(configured, 'amazon-bedrock')
  return {
    adaptiveThinking: supportsAdaptiveThinking(modelId),
    capabilities,
    isClaude: isClaudeModelId(modelId),
    modelId,
    supportsCaching: supportsPromptCaching(modelId, env),
  }
}

/** Region embedded in a Bedrock inference-profile ARN, if any. */
function arnRegion(modelId: string): string | undefined {
  return /^arn:aws(?:-[a-z0-9-]+)?:bedrock:([a-z0-9-]+):/.exec(modelId)?.[1]
}

/** Region encoded in a standard Bedrock runtime endpoint hostname. */
function standardEndpointRegion(baseUrl: string | undefined): string | undefined {
  if (!baseUrl) return undefined
  try {
    const { hostname } = new URL(baseUrl)
    return /^bedrock-runtime(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?$/.exec(hostname.toLowerCase())?.[1]
  } catch {
    return undefined
  }
}

/** Env region in pi-ai's precedence: AWS_REGION then AWS_DEFAULT_REGION. */
function configuredRegion(env: BedrockEnv): string | undefined {
  return envValue(env, 'AWS_REGION') ?? envValue(env, 'AWS_DEFAULT_REGION')
}

/**
 * pi-ai endpoint pinning: custom (non-standard) endpoints are always used
 * explicitly; a standard regional endpoint is only pinned when neither a
 * region nor an ambient AWS_PROFILE could otherwise route the SDK correctly.
 */
export function resolveBedrockEndpoint(
  baseUrl: string | undefined,
  region: string | undefined,
  hasAmbientProfile: boolean,
): string | undefined {
  const endpointRegion = standardEndpointRegion(baseUrl)
  if (!endpointRegion) return baseUrl
  if (!region && !hasAmbientProfile) return baseUrl
  return undefined
}

/** pi-ai GovCloud detection: region prefix or a GovCloud-scoped model id. */
function isGovCloudTarget(modelId: string, env: BedrockEnv): boolean {
  const region = configuredRegion(env)
  return region?.toLowerCase().startsWith('us-gov-')
    || modelId.toLowerCase().startsWith('us-gov.')
    || modelId.toLowerCase().startsWith('arn:aws-us-gov:')
}

/**
 * pi-ai's client-config resolution: profile/bearer/credential env handling,
 * the ARN > env > endpoint > us-east-1 region chain, and endpoint pinning.
 * Returns the settings handed to the SDK client.
 */
export function resolveBedrockConfig(options: {
  readonly baseUrl: string | undefined
  readonly bearerToken: string | undefined
  readonly env: BedrockEnv
  readonly modelId: string
}): BedrockResolvedConfig {
  const { baseUrl, bearerToken, env, modelId } = options
  const profile = envValue(env, 'AWS_PROFILE')
  const hasAmbientConfiguredProfile = Boolean(profile)
  const region = configuredRegion(env)
  const endpointRegion = standardEndpointRegion(baseUrl)
  const useExplicitEndpoint = resolveBedrockEndpoint(baseUrl, region, hasAmbientConfiguredProfile) !== undefined

  const arnMatch = arnRegion(modelId)
  let resolvedRegion: string | undefined
  if (arnMatch) {
    resolvedRegion = arnMatch
  } else if (region) {
    resolvedRegion = region
  } else if (endpointRegion && useExplicitEndpoint) {
    resolvedRegion = endpointRegion
  } else if (!hasAmbientConfiguredProfile) {
    resolvedRegion = 'us-east-1'
  }

  const skipAuth = envValue(env, 'AWS_BEDROCK_SKIP_AUTH') === '1'
  let credentials: BedrockResolvedConfig['credentials']
  if (skipAuth) {
    credentials = { accessKeyId: 'dummy-access-key', secretAccessKey: 'dummy-secret-key' }
  } else if (!bearerToken && !profile) {
    const accessKeyId = envValue(env, 'AWS_ACCESS_KEY_ID')
    const secretAccessKey = envValue(env, 'AWS_SECRET_ACCESS_KEY')
    if (accessKeyId && secretAccessKey) {
      const sessionToken = envValue(env, 'AWS_SESSION_TOKEN')
      credentials = { accessKeyId, secretAccessKey, ...(sessionToken ? { sessionToken } : {}) }
    }
  }

  const useBearerToken = bearerToken !== undefined && !skipAuth
  return {
    ...(useBearerToken ? { authSchemePreference: ['httpBearerAuth'] as const } : {}),
    ...(credentials ? { credentials } : {}),
    ...(useExplicitEndpoint && baseUrl ? { endpoint: baseUrl } : {}),
    ...(profile ? { profile } : {}),
    ...(resolvedRegion ? { region: resolvedRegion } : {}),
    // HTTP/1.1 by default: the SDK ≥3.798 defaults to NodeHttp2Handler and
    // Bun's node:http2 client is not battle-tested against event-stream
    // responses. `AWS_BEDROCK_FORCE_HTTP2=1` opts into the SDK default.
    requestHandler: envValue(env, 'AWS_BEDROCK_FORCE_HTTP2') === '1' ? 'http2' : 'http1',
    ...(useBearerToken ? { token: { token: bearerToken } } : {}),
  }
}

function createProductionClient(config: BedrockResolvedConfig): BedrockConverseTransport {
  const client = new BedrockRuntimeClient({
    ...(config.authSchemePreference ? { authSchemePreference: [...config.authSchemePreference] } : {}),
    ...(config.credentials ? { credentials: { ...config.credentials } } : {}),
    ...(config.endpoint ? { endpoint: config.endpoint } : {}),
    ...(config.profile ? { profile: config.profile } : {}),
    ...(config.region ? { region: config.region } : {}),
    ...(config.token ? { token: { ...config.token } } : {}),
    ...(config.requestHandler === 'http1' ? { requestHandler: new NodeHttpHandler() } : {}),
  })
  // The single SDK-boundary cast: assembled plain JSON → command input.
  type CommandInput = ConstructorParameters<typeof ConverseStreamCommand>[0]
  return {
    async send(input, signal) {
      const response = await client.send(
        new ConverseStreamCommand(input as unknown as CommandInput),
        ...(signal ? [{ abortSignal: signal }] as const : []),
      )
      return {
        stream: response.stream ?? (async function* empty() {
          // A response without an event stream carries nothing to translate.
        })(),
      }
    },
  }
}

function imageFormat(mimeType: string): (typeof ImageFormat)[keyof typeof ImageFormat] {
  switch (mimeType) {
    case 'image/jpeg':
    case 'image/jpg':
      return ImageFormat.JPEG
    case 'image/png':
      return ImageFormat.PNG
    case 'image/gif':
      return ImageFormat.GIF
    case 'image/webp':
      return ImageFormat.WEBP
    default:
      throw new ProviderError(PROVIDER_LABEL, `unsupported image type: ${mimeType}`)
  }
}

function imageBlock(part: Extract<ContentPart, { type: "image_url" }>): BedrockContentBlock {
  const match = /^data:([^;,]+);base64,([A-Za-z0-9+/=\r\n]+)$/.exec(part.image_url.url)
  if (!match?.[1] || !match[2]) {
    throw new ProviderError(PROVIDER_LABEL, 'Bedrock requires base64 data URLs for images')
  }
  return {
    image: {
      format: imageFormat(match[1]),
      source: { bytes: Uint8Array.from(atob(match[2].replaceAll(/\s/g, '')), char => char.charCodeAt(0)) },
    },
  }
}

/** Blank-only text is dropped where Bedrock allows, replaced where it is required. */
function optionalTextBlock(text: string): BedrockContentBlock | undefined {
  const sanitized = sanitizeSurrogates(text)
  return sanitized.trim().length === 0 ? undefined : { text: sanitized }
}

function requiredTextBlock(text: string): BedrockContentBlock {
  return optionalTextBlock(text) ?? { text: EMPTY_TEXT_PLACEHOLDER }
}

/** Bedrock documents reject empty-string keys coming from hand-built schemas. */
function sanitizeBedrockDocument(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sanitizeBedrockDocument)
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([key]) => key.length > 0)
        .map(([key, nested]) => [key, sanitizeBedrockDocument(nested)]),
    )
  }
  return value
}

function userContentBlocks(content: MessageContent): BedrockContentBlock[] {
  if (typeof content === 'string') return [requiredTextBlock(content)]
  const blocks: BedrockContentBlock[] = []
  for (const part of content) {
    if (part.type === 'text') {
      const block = optionalTextBlock(part.text)
      if (block) blocks.push(block)
      continue
    }
    blocks.push(imageBlock(part))
  }
  if (blocks.length === 0) blocks.push({ text: EMPTY_TEXT_PLACEHOLDER })
  return blocks
}

function toolResultContent(content: string): BedrockContentBlock[] {
  const block = optionalTextBlock(content)
  return block ? [block] : [{ text: EMPTY_TEXT_PLACEHOLDER }]
}

/**
 * pi-ai thinking budget math: the caller's explicit budget wins, otherwise the
 * level default; never exceed the ceiling minus {@link MIN_ANSWER_TOKENS}.
 */
function thinkingBudgetFor(thinking: ThinkingRequest): number {
  const effort = thinking.effort === 'xhigh' || thinking.effort === 'max' ? 'high' : thinking.effort
  const defaultBudget = effort !== undefined ? defaultThinkingBudget(effort) : undefined
  return thinking.budgetTokens ?? defaultBudget ?? defaultThinkingBudget('high')
}

/**
 * pi-ai `buildAdditionalModelRequestFields`: Claude-only thinking config —
 * adaptive (`output_config.effort`) for Opus 4.6+/Sonnet 4.6+, budget-based
 * otherwise, with the interleaved-thinking beta on non-adaptive models and
 * the `display` field omitted on GovCloud.
 */
export function buildBedrockThinkingFields(options: {
  readonly adaptiveThinking: boolean
  /** Explicit budget resolved by the caller; falls back to the effort default. */
  readonly budgetTokens?: number
  readonly capabilities: PiModelCapabilities | undefined
  readonly govCloud: boolean
  readonly thinking: ThinkingRequest
}): Record<string, unknown> | undefined {
  const { adaptiveThinking, budgetTokens, capabilities, govCloud, thinking } = options
  if (!capabilities?.reasoning) return undefined
  // GovCloud Bedrock rejects the Claude thinking.display field for now.
  const display = govCloud ? undefined : 'summarized'
  if (adaptiveThinking) {
    const level = thinking.effort ?? 'high'
    const mapped = capabilities.thinkingLevelMap?.[level]
    const effort = typeof mapped === 'string'
      ? mapped
      : level === 'minimal' || level === 'low'
        ? 'low'
        : level === 'medium'
          ? 'medium'
          : 'high'
    return {
      thinking: { type: 'adaptive', ...(display !== undefined ? { display } : {}) },
      output_config: { effort },
    }
  }
  return {
    thinking: {
      type: 'enabled',
      budget_tokens: budgetTokens ?? thinkingBudgetFor(thinking),
      ...(display !== undefined ? { display } : {}),
    },
    anthropic_beta: ['interleaved-thinking-2025-05-14'],
  }
}

/** Convert one request's neutral messages into Bedrock Converse messages. */
export function buildBedrockMessages(
  request: CompletionRequest,
  model: ResolvedBedrockModel,
  env: BedrockEnv,
): BedrockMessage[] {
  // pi-ai `resolveCacheRetention`: PI_CACHE_RETENTION=long selects 1-hour
  // cache points; anything else is the short default.
  const longTermCaching = envValue(env, 'PI_CACHE_RETENTION') === 'long'
  const cachingEnabled = model.supportsCaching
  const cachePoint: BedrockContentBlock | undefined = cachingEnabled
    ? { cachePoint: { type: CachePointType.DEFAULT, ...(longTermCaching ? { ttl: CacheTTL.ONE_HOUR } : {}) } }
    : undefined

  const messages: BedrockMessage[] = []
  const toolCallIdMap = new Map<string, string>()

  for (let index = 0; index < request.messages.length; index++) {
    const message = request.messages[index]
    if (!message || message.role === 'system') continue
    if (message.role === 'user') {
      messages.push({ role: "user", content: userContentBlocks(message.content) })
      continue
    }
    if (message.role === 'assistant') {
      const content: BedrockContentBlock[] = []
      const textBlock = optionalTextBlock(messageText(message))
      if (textBlock) content.push(textBlock)
      for (const call of message.tool_calls ?? []) {
        const toolUseId = normalizeBedrockToolCallId(call.id)
        toolCallIdMap.set(call.id, toolUseId)
        content.push({
          toolUse: {
            toolUseId,
            name: call.function.name,
            input: sanitizeBedrockDocument(call.function.arguments) as Record<string, unknown>,
          },
        })
      }
      const thinking = message.thinking
      const thinkingSignature = message.thinking_signature
      if (thinking && thinking.trim().length > 0) {
        // Signatures arrive after thinking deltas, and only Anthropic models
        // accept the signature field; anything else replays as plain text.
        if (model.isClaude && thinkingSignature && thinkingSignature.trim().length > 0) {
          content.push({
            reasoningContent: { reasoningText: { text: sanitizeSurrogates(thinking), signature: thinkingSignature } },
          })
        } else {
          content.push({ text: sanitizeSurrogates(thinking) })
        }
      }
      // Bedrock rejects messages with empty content arrays (aborted turns).
      if (content.length > 0) {
        messages.push({ role: "assistant", content })
      }
      continue
    }
    // Tool results: Bedrock requires all tool results in one user message, so
    // consecutive tool messages are grouped and the cache point trails them.
    const toolResults: BedrockContentBlock[] = []
    let cursor = index
    while (cursor < request.messages.length && request.messages[cursor]?.role === 'tool') {
      const toolMessage = request.messages[cursor]
      if (toolMessage?.role === 'tool') {
        const toolUseId = toolCallIdMap.get(toolMessage.tool_call_id)
          ?? normalizeBedrockToolCallId(toolMessage.tool_call_id)
        toolResults.push({
          toolResult: {
            toolUseId,
            content: toolResultContent(toolMessage.content),
            status: toolMessage.is_error ? 'error' : 'success',
          },
        })
      }
      cursor += 1
    }
    index = cursor - 1
    if (toolResults.length > 0) {
      if (cachePoint) toolResults.push(cachePoint)
      messages.push({ role: "user", content: toolResults })
    }
  }
  return messages
}

/** Convert neutral tools + choice into Bedrock's `toolConfig` (pi-ai parity). */
export interface BedrockToolConfig {
  readonly toolChoice?: Record<string, unknown>
  readonly tools: readonly {
    readonly toolSpec: {
      readonly description?: string
      readonly inputSchema: { readonly json: Record<string, unknown> }
      readonly name: string
    }
  }[]
}

export function buildBedrockToolConfig(
  tools: readonly ToolDefinition[] | undefined,
  toolChoice: CompletionRequest['toolChoice'],
): BedrockToolConfig | undefined {
  if (!tools?.length || toolChoice === 'none') return undefined
  const bedrockTools = tools.map(tool => ({
    toolSpec: {
      description: tool.function.description,
      inputSchema: { json: sanitizeBedrockDocument(tool.function.parameters) as Record<string, unknown> },
      name: tool.function.name,
    },
  }))
  let choice: Record<string, unknown> | undefined
  if (toolChoice === 'auto') choice = { auto: {} }
  if (toolChoice === 'any') choice = { any: {} }
  return {
    tools: bedrockTools,
    ...(choice ? { toolChoice: choice } : {}),
  }
}

/**
 * Build the full Converse stream command input for one request. Exported for
 * deterministic tests; the client calls this on every turn.
 */
export function buildBedrockConverseInput(
  request: CompletionRequest,
  options: {
    readonly env: BedrockEnv
    readonly model: ResolvedBedrockModel
  },
): BedrockConverseInput {
  const { env, model } = options
  const capabilities = model.capabilities

  // System prompt: neutral system messages join the Converse `system` blocks,
  // with the cache breakpoint after the stable prefix (pi-ai buildSystemPrompt).
  const systemTexts = request.messages
    .filter(message => message.role === 'system')
    .map(message => messageText(message))
    .filter(text => text.trim().length > 0)
  const systemJoined = systemTexts.join('\n\n')
  const longTermCaching = envValue(env, 'PI_CACHE_RETENTION') === 'long'
  const system: BedrockContentBlock[] | undefined = systemJoined
    ? [
      { text: sanitizeSurrogates(systemJoined) },
      ...(model.supportsCaching
        ? [{ cachePoint: { type: CachePointType.DEFAULT, ...(longTermCaching ? { ttl: CacheTTL.ONE_HOUR } : {}) } } satisfies BedrockContentBlock]
        : []),
    ]
    : undefined

  // pi-ai: only Claude models get a default output cap from the catalog.
  const modelMaxTokens = model.isClaude ? capabilities?.maxOutputTokens : undefined
  let maxTokens = request.maxTokens ?? modelMaxTokens
  const thinking = request.thinking
  let thinkingBudget: number | undefined
  let thinkingFields: Record<string, unknown> | undefined
  if (thinking && capabilities?.reasoning && model.isClaude) {
    // pi-ai adjustMaxTokensForThinking: the response ceiling grows to the
    // caller's cap plus the thinking budget, capped by the model maximum;
    // the budget then clamps so MIN_ANSWER_TOKENS remain for the answer.
    thinkingBudget = thinkingBudgetFor(thinking)
    if (request.maxTokens !== undefined) {
      maxTokens = modelMaxTokens === undefined
        ? request.maxTokens + thinkingBudget
        : Math.min(request.maxTokens + thinkingBudget, modelMaxTokens)
    }
    if (maxTokens !== undefined && maxTokens <= thinkingBudget) {
      thinkingBudget = Math.min(thinkingBudget, Math.max(0, maxTokens - MIN_ANSWER_TOKENS))
    }
    thinkingFields = buildBedrockThinkingFields({
      adaptiveThinking: model.adaptiveThinking,
      capabilities,
      govCloud: isGovCloudTarget(model.modelId, env),
      thinking,
      ...(thinkingBudget !== undefined ? { budgetTokens: thinkingBudget } : {}),
    })
  }

  const toolConfig = buildBedrockToolConfig(request.tools, request.toolChoice)
  const input: Record<string, unknown> = {
    modelId: model.modelId,
    messages: buildBedrockMessages(request, model, env),
    ...(system ? { system } : {}),
    inferenceConfig: {
      ...(maxTokens !== undefined ? { maxTokens } : {}),
      ...(request.temperature !== undefined ? { temperature: request.temperature } : {}),
    },
    ...(toolConfig ? { toolConfig } : {}),
    ...(thinkingFields ? { additionalModelRequestFields: thinkingFields } : {}),
  }
  if (request.extraBody) Object.assign(input, request.extraBody)
  return input
}

/** pi-ai `mapStopReason`: Converse stop reasons → Xerxes finish reasons. */
export function bedrockFinishReason(stopReason: string | undefined): string {
  switch (stopReason) {
    case 'end_turn':
    case 'stop_sequence':
    case 'pause_turn':
      return 'stop'
    case 'max_tokens':
    case 'model_context_window_exceeded':
      return 'length'
    case 'tool_use':
      return 'tool_calls'
    default:
      throw new ProviderError(PROVIDER_LABEL, `provider stopped with: ${stopReason ?? 'unknown reason'}`)
  }
}

/** pi-ai `formatBedrockError`: stable prefixes so retry logic can pattern-match. */
export function formatBedrockError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error)
  const hint = /data retention mode/i.test(message)
    ? ` See ${BEDROCK_DATA_RETENTION_DOCS_URL} for supported data retention modes.`
    : ''
  if (error instanceof BedrockRuntimeServiceException) {
    const prefix = BEDROCK_ERROR_PREFIXES[error.name] ?? error.name
    return `${prefix}: ${message}${hint}`
  }
  return `${message}${hint}`
}

function bedrockHttpError(error: unknown): ProviderError {
  const metadata = (error as { readonly $metadata?: { readonly httpStatusCode?: number } } | undefined)?.$metadata
  return new ProviderError(PROVIDER_LABEL, formatBedrockError(error), undefined, {
    ...(typeof metadata?.httpStatusCode === 'number' ? { status: metadata.httpStatusCode } : {}),
  })
}

interface BedrockPendingToolCall {
  readonly id: string
  readonly name: string
  partialJson: string
}

/**
 * Bedrock Converse streaming client (pi-ai `bedrock-converse-stream`).
 *
 * Converse has no non-streaming surface in this adapter: `complete()` collects
 * the stream, the way the subscription-backed providers do.
 */
export class BedrockConverseClient implements LlmClient {
  private readonly apiKey: string | undefined
  private readonly baseUrl: string | undefined
  private readonly createClient: BedrockClientFactory
  private readonly env: BedrockEnv

  constructor(options: BedrockClientOptions = {}) {
    this.apiKey = options.apiKey
    this.baseUrl = options.baseUrl
    this.createClient = options.createClient ?? createProductionClient
    this.env = options.env ?? process.env
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    return collectLlmCompletion(this.stream(request, signal))
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const model = resolveBedrockModel(request, this.env)
    const bearerToken = this.apiKey ?? envValue(this.env, 'AWS_BEARER_TOKEN_BEDROCK')
    const config = resolveBedrockConfig({
      baseUrl: this.baseUrl ?? 'https://bedrock-runtime.us-east-1.amazonaws.com',
      bearerToken,
      env: this.env,
      modelId: model.modelId,
    })
    const client = this.createClient(config)
    const input = buildBedrockConverseInput(request, { env: this.env, model })

    let response: { readonly stream: AsyncIterable<unknown> }
    try {
      response = await client.send(input, signal)
    } catch (error) {
      throw bedrockHttpError(error)
    }

    const pendingToolCalls = new Map<number, BedrockPendingToolCall>()
    let emittedToolCalls = false
    let receivedStop = false
    try {
      for await (const rawEvent of response.stream) {
        const event = asRecord(rawEvent)
        if (!event) continue
        if (event.contentBlockDelta !== undefined && event.contentBlockDelta !== null) {
          const deltaEvent = asRecord(event.contentBlockDelta)
          const delta = asRecord(deltaEvent?.delta)
          const blockIndex = numberAt(deltaEvent, 'contentBlockIndex')
          if (delta) {
            const text = stringAt(delta, 'text')
            if (text !== undefined) yield { content: text }
            const toolUse = asRecord(delta.toolUse)
            const toolInput = typeof toolUse?.input === 'string' ? toolUse.input : undefined
            if (toolUse && toolInput !== undefined && blockIndex !== undefined) {
              const pending = pendingToolCalls.get(blockIndex)
              if (pending) pending.partialJson += toolInput
            }
            const reasoning = asRecord(delta.reasoningContent)
            if (reasoning) {
              const reasoningText = stringAt(asRecord(reasoning.reasoningText), 'text')
              if (reasoningText) yield { thinking: reasoningText }
              const signature = stringAt(asRecord(reasoning.reasoningText), 'signature')
              if (signature) yield { thinkingSignature: signature }
              const redacted = reasoning.redactedContent
              // Encrypted reasoning from non-Anthropic models arrives as an
              // opaque byte payload; keep it base64 in `thinkingSignature` the
              // way the Anthropic path stores redacted thinking.
              if (redacted instanceof Uint8Array && redacted.length > 0) {
                yield {
                  thinking: REDACTED_THINKING_PLACEHOLDER,
                  thinkingSignature: Buffer.from(redacted).toString('base64'),
                }
              }
            }
          }
          continue
        }
        if (event.contentBlockStart !== undefined && event.contentBlockStart !== null) {
          const startEvent = asRecord(event.contentBlockStart)
          const blockIndex = numberAt(startEvent, 'contentBlockIndex')
          const toolUse = asRecord(asRecord(startEvent?.start)?.toolUse)
          if (toolUse && blockIndex !== undefined) {
            pendingToolCalls.set(blockIndex, {
              id: stringAt(toolUse, 'toolUseId') ?? '',
              name: stringAt(toolUse, 'name') ?? '',
              partialJson: '',
            })
          }
          continue
        }
        if (event.contentBlockStop !== undefined && event.contentBlockStop !== null) {
          const stopEvent = asRecord(event.contentBlockStop)
          const blockIndex = numberAt(stopEvent, 'contentBlockIndex')
          const pending = blockIndex === undefined ? undefined : pendingToolCalls.get(blockIndex)
          if (pending && blockIndex !== undefined) {
            pendingToolCalls.delete(blockIndex)
            emittedToolCalls = true
            yield { toolCalls: [finalizeBedrockToolCall(pending)] }
          }
          continue
        }
        if (event.messageStop !== undefined && event.messageStop !== null) {
          const stopEvent = asRecord(event.messageStop)
          receivedStop = true
          if (pendingToolCalls.size) {
            const calls = [...pendingToolCalls.entries()]
              .sort(([left], [right]) => left - right)
              .map(([, pending]) => finalizeBedrockToolCall(pending))
            pendingToolCalls.clear()
            emittedToolCalls = true
            yield { toolCalls: calls }
          }
          yield { finishReason: bedrockFinishReason(stringAt(stopEvent, 'stopReason')) }
          continue
        }
        if (event.metadata !== undefined && event.metadata !== null) {
          const usage = asRecord(asRecord(event.metadata)?.usage)
          if (usage) yield { usage: bedrockUsage(usage) }
          continue
        }
        for (const exceptionName of [
          'internalServerException',
          'modelStreamErrorException',
          'validationException',
          'throttlingException',
          'serviceUnavailableException',
        ]) {
          if (event[exceptionName] !== undefined && event[exceptionName] !== null) {
            throw event[exceptionName]
          }
        }
      }
    } catch (error) {
      if (error instanceof ProviderError) throw error
      if (signal?.aborted) {
        throw new ProviderError(PROVIDER_LABEL, 'request was aborted')
      }
      throw bedrockHttpError(error)
    }
    if (!receivedStop) {
      throw new ProviderError(PROVIDER_LABEL, 'Bedrock stream ended without a stop reason')
    }
    // A stream can settle without stopping every tool block (pi-ai finalize).
    if (!emittedToolCalls && pendingToolCalls.size) {
      const calls = [...pendingToolCalls.entries()]
        .sort(([left], [right]) => left - right)
        .map(([, pending]) => finalizeBedrockToolCall(pending))
      yield { toolCalls: calls }
    }
  }
}

function finalizeBedrockToolCall(pending: BedrockPendingToolCall): ToolCall {
  if (!pending.name) {
    throw new ProviderError(PROVIDER_LABEL, 'toolUse block missing a name')
  }
  const partial = parseStreamingJson(pending.partialJson || '{}')
  const arguments_ = isJsonObject(partial) ? partial : {}
  return {
    id: pending.id || deterministicToolCallId(pending.name, arguments_),
    type: 'function',
    function: { name: pending.name, arguments: arguments_ },
  }
}

function bedrockUsage(usage: Record<string, unknown>): TokenUsage {
  const cacheReadTokens = numberAt(usage, 'cacheReadInputTokens')
  const cacheCreationTokens = numberAt(usage, 'cacheWriteInputTokens')
  return {
    inputTokens: numberAt(usage, 'inputTokens') ?? 0,
    outputTokens: numberAt(usage, 'outputTokens') ?? 0,
    ...(cacheReadTokens !== undefined ? { cacheReadTokens } : {}),
    ...(cacheCreationTokens !== undefined ? { cacheCreationTokens } : {}),
  }
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

function stringAt(record: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = record?.[key]
  return typeof value === 'string' ? value : undefined
}

function numberAt(record: Record<string, unknown> | undefined, key: string): number | undefined {
  const value = record?.[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}
