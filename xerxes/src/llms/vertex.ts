// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Google Vertex AI native `generateContent` streaming transport (pi-ai `google-vertex`). */

import { createSign } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

import { ConfigurationError, ProviderError } from '../core/errors.js'
import type { ChatMessage, ContentPart } from '../types/messages.js'
import { messageText } from '../types/messages.js'
import { parseToolArguments, type JsonObject, type ToolCall, type ToolChoice, type ToolDefinition } from '../types/toolCalls.js'
import type { CompletionRequest, FetchImplementation, LlmClient, LlmCompletion, LlmDelta, TokenUsage } from './client.js'
import { internalSseData } from './client.js'
import type { PiModelCapabilities } from './piModelCatalog.js'
import { piCatalogModelCapabilities } from './piModelCatalog.js'
import { bareModel } from './providerRegistry.js'

const API_VERSION = 'v1'
const VERTEX_EXPRESS_API_ROOT = 'https://aiplatform.googleapis.com'
/** pi-ai's placeholder marker: a stored credential that means "use ADC, not an API key". */
const GCP_VERTEX_CREDENTIALS_MARKER = 'gcp-vertex-credentials'
const CLOUD_PLATFORM_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'
const METADATA_TOKEN_URL =
  'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token'
const METADATA_PROBE_TIMEOUT_MS = 2_000
/** Refresh the cached access token this far ahead of its reported expiry. */
const TOKEN_REFRESH_SKEW_MS = 60_000
/** Map Vertex finish reasons to the subset Xerxes propagates; the safety family is fatal. */
const FATAL_FINISH_REASONS = new Set([
  'SAFETY',
  'BLOCKLIST',
  'PROHIBITED_CONTENT',
  'SPII',
  'RECITATION',
  'LANGUAGE',
  'MALFORMED_FUNCTION_CALL',
  'IMAGE_SAFETY',
  'IMAGE_PROHIBITED_CONTENT',
  'IMAGE_RECITATION',
  'IMAGE_OTHER',
  'IMAGE_NO_IMAGE',
  'UNEXPECTED_TOOL_CALL',
  'FINISH_REASON_UNSPECIFIED',
  'OTHER',
])

export interface GoogleVertexClientOptions {
  /**
   * Vertex express-mode API key. pi-ai's stored-credential markers (`<...>`
   * placeholders and `gcp-vertex-credentials`) are rejected so ADC is used
   * instead of shipping a placeholder to the wire.
   */
  readonly apiKey?: string
  /** API root override; `{location}` materializes from the resolved location. */
  readonly baseUrl?: string
  readonly project?: string
  readonly location?: string
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly fetchImplementation?: FetchImplementation
  /** Home directory for the gcloud ADC well-known file; defaults to `~`. */
  readonly home?: string
}

interface CachedGoogleToken {
  readonly expiresAtMs: number
  readonly key: string
  readonly token: string
}

const tokenCache = new Map<string, CachedGoogleToken>()

/** Strip lone UTF-16 surrogates so a malformed transcript cannot 400 the request. */
function sanitizeSurrogates(value: string): string {
  return value.replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, '')
}

function trimSlash(value: string): string {
  return value.replace(/\/+$/, '')
}

function envValue(name: string, env: Readonly<Record<string, string | undefined>>): string | undefined {
  const value = env[name]?.trim()
  return value ? value : undefined
}

/** pi-ai's placeholder rejection: `<...>` placeholders never reach the wire. */
function isPlaceholderApiKey(apiKey: string): boolean {
  return /^<[^>]+>$/.test(apiKey)
}

export function resolveVertexApiKey(options: GoogleVertexClientOptions): string | undefined {
  const apiKey = (options.apiKey ?? options.env?.GOOGLE_VERTEX_API_KEY ?? '').trim()
  if (!apiKey || apiKey === GCP_VERTEX_CREDENTIALS_MARKER || isPlaceholderApiKey(apiKey)) {
    return undefined
  }
  return apiKey
}

export function resolveVertexProject(options: GoogleVertexClientOptions): string {
  const project = options.project?.trim()
    || envValue('GOOGLE_CLOUD_PROJECT', options.env ?? process.env)
    || envValue('GCLOUD_PROJECT', options.env ?? process.env)
  if (!project) {
    throw new ConfigurationError(
      'GOOGLE_CLOUD_PROJECT',
      'Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options.',
    )
  }
  return project
}

export function resolveVertexLocation(options: GoogleVertexClientOptions): string {
  const location = options.location?.trim()
    || envValue('GOOGLE_CLOUD_LOCATION', options.env ?? process.env)
  if (!location) {
    throw new ConfigurationError(
      'GOOGLE_CLOUD_LOCATION',
      'Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.',
    )
  }
  return location
}

interface ServiceAccountKeyFile {
  readonly client_email: string
  readonly private_key: string
  readonly token_uri?: string
  readonly type: string
}

interface AuthorizedUserKeyFile {
  readonly client_id: string
  readonly client_secret: string
  readonly refresh_token: string
  readonly type: string
}

type GcpKeyFile = ServiceAccountKeyFile | AuthorizedUserKeyFile

function isServiceAccountFile(value: GcpKeyFile): value is ServiceAccountKeyFile {
  return value.type === 'service_account' && typeof (value as ServiceAccountKeyFile).client_email === 'string'
}

function wellKnownAdcPath(home: string): string {
  return join(home, '.config', 'gcloud', 'application_default_credentials.json')
}

async function loadKeyFile(path: string): Promise<GcpKeyFile | undefined> {
  let raw: string
  try {
    raw = await readFile(path, 'utf8')
  } catch {
    return undefined
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    throw new ConfigurationError('GOOGLE_APPLICATION_CREDENTIALS', `credential file ${path} is not valid JSON`)
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new ConfigurationError('GOOGLE_APPLICATION_CREDENTIALS', `credential file ${path} is not a JSON object`)
  }
  const record = parsed as Record<string, unknown>
  const type = record.type
  if (type === 'service_account'
    && typeof record.client_email === 'string'
    && typeof record.private_key === 'string') {
    return {
      client_email: record.client_email,
      private_key: record.private_key,
      ...(typeof record.token_uri === 'string' ? { token_uri: record.token_uri } : {}),
      type,
    }
  }
  if (type === 'authorized_user'
    && typeof record.client_id === 'string'
    && typeof record.client_secret === 'string'
    && typeof record.refresh_token === 'string') {
    return {
      client_id: record.client_id,
      client_secret: record.client_secret,
      refresh_token: record.refresh_token,
      type,
    }
  }
  throw new ConfigurationError(
    'GOOGLE_APPLICATION_CREDENTIALS',
    `credential file ${path} has unsupported type '${String(type)}'; expected service_account or authorized_user`,
  )
}

function base64Url(value: string | Uint8Array): string {
  return Buffer.from(value).toString('base64url')
}

/** Sign the service-account JWT assertion (RS256) exactly as google-auth-library does. */
function signServiceAccountJwt(keyFile: ServiceAccountKeyFile, nowSeconds: number): string {
  const header = base64Url(JSON.stringify({ alg: 'RS256', typ: 'JWT' }))
  const claims = base64Url(JSON.stringify({
    aud: keyFile.token_uri ?? 'https://oauth2.googleapis.com/token',
    exp: nowSeconds + 3_600,
    iat: nowSeconds,
    iss: keyFile.client_email,
    scope: CLOUD_PLATFORM_SCOPE,
  }))
  const signer = createSign('RSA-SHA256')
  signer.update(`${header}.${claims}`)
  const signature = signer.sign(keyFile.private_key)
  return `${header}.${claims}.${base64Url(new Uint8Array(signature))}`
}

async function fetchJson(
  fetchImplementation: FetchImplementation,
  url: string,
  init: RequestInit,
): Promise<Record<string, unknown>> {
  const response = await fetchImplementation(url, init)
  const body = await response.text()
  if (!response.ok) {
    throw new ProviderError('google-vertex', `GCP token request failed (${response.status}): ${body.slice(0, 512)}`)
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(body)
  } catch {
    throw new ProviderError('google-vertex', 'GCP token response was not JSON')
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new ProviderError('google-vertex', 'GCP token response was not an object')
  }
  return parsed as Record<string, unknown>
}

async function tokenFromKeyFile(
  fetchImplementation: FetchImplementation,
  keyFile: GcpKeyFile,
  path: string,
): Promise<{ accessToken: string; expiresInSeconds: number }> {
  if (isServiceAccountFile(keyFile)) {
    const tokenUri = keyFile.token_uri ?? 'https://oauth2.googleapis.com/token'
    const payload = await fetchJson(fetchImplementation, tokenUri, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        assertion: signServiceAccountJwt(keyFile, Math.floor(Date.now() / 1000)),
        grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer',
      }).toString(),
    })
    return tokenFromPayload(payload, path)
  }
  const payload = await fetchJson(fetchImplementation, 'https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: keyFile.client_id,
      client_secret: keyFile.client_secret,
      grant_type: 'refresh_token',
      refresh_token: keyFile.refresh_token,
    }).toString(),
  })
  return tokenFromPayload(payload, path)
}

function tokenFromPayload(
  payload: Record<string, unknown>,
  source: string,
): { accessToken: string; expiresInSeconds: number } {
  const accessToken = payload.access_token
  const expiresInSeconds = payload.expires_in
  if (typeof accessToken !== 'string' || !accessToken) {
    throw new ProviderError('google-vertex', `GCP credential ${source} returned no access_token`)
  }
  return {
    accessToken,
    expiresInSeconds: typeof expiresInSeconds === 'number' && Number.isFinite(expiresInSeconds)
      ? expiresInSeconds
      : 3_600,
  }
}

/**
 * Resolve a Vertex access token following google-auth-library's order:
 * `GOOGLE_APPLICATION_CREDENTIALS`, the gcloud ADC well-known file, then the
 * compute metadata server. Tokens are cached per credential until near expiry.
 */
export async function resolveVertexAccessToken(
  options: GoogleVertexClientOptions,
  fetchImplementation: FetchImplementation,
  nowMs = Date.now(),
): Promise<string> {
  const env = options.env ?? process.env
  const explicitPath = envValue('GOOGLE_APPLICATION_CREDENTIALS', env)
  const home = options.home ?? envValue('HOME', env) ?? ''
  const adcPath = explicitPath ?? (home ? wellKnownAdcPath(home) : undefined)
  const keyFile = adcPath ? await loadKeyFile(adcPath) : undefined
  const cacheKey = keyFile ? `file:${adcPath}` : 'metadata'

  const cached = tokenCache.get(cacheKey)
  if (cached && cached.expiresAtMs - TOKEN_REFRESH_SKEW_MS > nowMs) {
    return cached.token
  }

  let accessToken: string
  let expiresInSeconds: number
  if (keyFile) {
    ({ accessToken, expiresInSeconds } = await tokenFromKeyFile(fetchImplementation, keyFile, adcPath ?? 'file'))
  } else {
    // Last resort in google-auth-library's chain: the GCE/run metadata server.
    let response: Response
    try {
      response = await fetchImplementation(METADATA_TOKEN_URL, {
        headers: { 'Metadata-Flavor': 'Google' },
        signal: AbortSignal.timeout(METADATA_PROBE_TIMEOUT_MS),
      })
    } catch (error) {
      throw new ConfigurationError(
        'GOOGLE_APPLICATION_CREDENTIALS',
        `No Application Default Credentials found (metadata server unreachable: ${String(error)}). `
        + 'Set GOOGLE_APPLICATION_CREDENTIALS, run `gcloud auth application-default login`, '
        + 'or provide a Vertex express API key.',
      )
    }
    const body = await response.text()
    if (!response.ok) {
      throw new ProviderError('google-vertex', `metadata server token request failed (${response.status})`)
    }
    let parsed: unknown
    try {
      parsed = JSON.parse(body)
    } catch {
      throw new ProviderError('google-vertex', 'metadata server token response was not JSON')
    }
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      throw new ProviderError('google-vertex', 'metadata server token response was not an object')
    }
    ({ accessToken, expiresInSeconds } = tokenFromPayload(parsed as Record<string, unknown>, 'metadata server'))
  }

  tokenCache.set(cacheKey, {
    key: cacheKey,
    token: accessToken,
    expiresAtMs: nowMs + expiresInSeconds * 1_000,
  })
  return accessToken
}

/** Test hook: drop cached GCP access tokens. */
export function clearVertexTokenCache(): void {
  tokenCache.clear()
}

function requiresToolCallId(modelId: string): boolean {
  if (modelId.startsWith('claude-') || modelId.startsWith('gpt-oss-')) return true
  const match = modelId.toLowerCase().match(/^gemini(?:-live)?-(\d+)/)
  return match?.[1] !== undefined && Number.parseInt(match[1], 10) >= 3
}

const JSON_SCHEMA_META_DECLARATIONS = new Set([
  '$schema',
  '$id',
  '$anchor',
  '$dynamicAnchor',
  '$vocabulary',
  '$comment',
  '$defs',
  'definitions',
])

/** pi-ai sanitizeForOpenApi: strip JSON-schema meta declarations Vertex rejects. */
function sanitizeForOpenApi(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(entry => sanitizeForOpenApi(entry))
  if (typeof value !== 'object' || value === null) return value
  const result: Record<string, unknown> = {}
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (JSON_SCHEMA_META_DECLARATIONS.has(key)) continue
    result[key] = sanitizeForOpenApi(entry)
  }
  return result
}

type VertexPart = {
  text?: string
  thought?: boolean
  thoughtSignature?: string
  functionCall?: { args?: JsonObject; id?: string; name?: string }
  functionResponse?: {
    id?: string
    name?: string
    parts?: { inlineData: { data: string; mimeType: string } }[]
    response?: Record<string, string>
  }
  inlineData?: { data: string; mimeType: string }
}

interface VertexContent {
  parts: VertexPart[]
  role: 'model' | 'user'
}

/** pi-ai thought-signature validation: base64 TYPE_BYTES only, same-model replay. */
function isValidThoughtSignature(signature: string | undefined): signature is string {
  if (!signature) return false
  if (signature.length % 4 !== 0) return false
  return /^[A-Za-z0-9+/]+={0,2}$/.test(signature)
}

function imagePartFromDataUrl(url: string): VertexPart | undefined {
  const match = /^data:([^;,]+);base64,([A-Za-z0-9+/=\r\n]+)$/.exec(url)
  if (!match?.[1] || !match[2]) return undefined
  return { inlineData: { mimeType: match[1], data: match[2].replaceAll(/\s/g, '') } }
}

function userParts(content: ContentPart[] | string): VertexPart[] {
  if (typeof content === 'string') {
    const text = sanitizeSurrogates(content)
    return text ? [{ text }] : []
  }
  const parts: VertexPart[] = []
  for (const part of content) {
    if (part.type === 'text') {
      const text = sanitizeSurrogates(part.text)
      if (text) parts.push({ text })
      continue
    }
    const image = imagePartFromDataUrl(part.image_url.url)
    if (image) parts.push(image)
  }
  return parts
}

/**
 * Convert neutral messages to Vertex `contents` (pi-ai google-shared
 * convertMessages): system prompts are extracted by the caller, assistant
 * turns carry text/thought/functionCall parts, and consecutive tool results
 * merge into a single user turn of functionResponse parts.
 */
export function vertexContentsFromMessages(
  messages: readonly ChatMessage[],
  modelId: string,
): VertexContent[] {
  const contents: VertexContent[] = []
  const toolNamesById = new Map<string, string>()
  const pushFunctionResponse = (part: VertexPart): void => {
    const last = contents.at(-1)
    if (last?.role === 'user' && last.parts.some(entry => entry.functionResponse !== undefined)) {
      last.parts.push(part)
      return
    }
    contents.push({ role: 'user', parts: [part] })
  }

  for (const message of messages) {
    if (message.role === 'system') continue
    if (message.role === 'user') {
      const parts = userParts(message.content)
      if (parts.length) contents.push({ role: 'user', parts })
      continue
    }
    if (message.role === 'assistant') {
      const parts: VertexPart[] = []
      const text = sanitizeSurrogates(messageText(message))
      if (text) parts.push({ text })
      if (message.thinking?.trim()) {
        // Xerxes persists one thinking trace per assistant message; replay it
        // as a thought part, carrying a stored signature when it is valid
        // base64 (pi-ai only replays same-model signatures — Xerxes messages
        // do not record the originating model, so validity is the gate).
        const signature = isValidThoughtSignature(message.thinking_signature)
          ? message.thinking_signature
          : undefined
        parts.push({
          thought: true,
          text: sanitizeSurrogates(message.thinking),
          ...(signature ? { thoughtSignature: signature } : {}),
        })
      }
      for (const call of message.tool_calls ?? []) {
        toolNamesById.set(call.id, call.function.name)
        parts.push({
          functionCall: {
            args: parseToolArguments(call.function.arguments),
            ...(requiresToolCallId(modelId) ? { id: call.id } : {}),
            name: call.function.name,
          },
        })
      }
      if (parts.length) contents.push({ role: 'model', parts })
      continue
    }

    // Tool result: Gemini's functionResponse wants the tool NAME, which the
    // transcript's matching assistant tool_call provides.
    const name = message.name ?? toolNamesById.get(message.tool_call_id) ?? message.tool_call_id
    const content = message.content || '(no tool output)'
    const responseValue = sanitizeSurrogates(content)
    pushFunctionResponse({
      functionResponse: {
        ...(requiresToolCallId(modelId) ? { id: message.tool_call_id } : {}),
        name,
        response: message.is_error ? { error: responseValue } : { output: responseValue },
      },
    })
  }
  return contents
}

function thinkingLevelFromEffort(effort: string | undefined): string | undefined {
  switch (effort) {
    case 'minimal':
      return 'MINIMAL'
    case 'low':
      return 'LOW'
    case 'medium':
      return 'MEDIUM'
    case 'high':
      return 'HIGH'
    default:
      return undefined
  }
}

function isGemini3ProModel(modelId: string): boolean {
  return /gemini-3(?:\.\d+)?-pro/.test(modelId.toLowerCase())
}

function isGemini3FlashModel(modelId: string): boolean {
  const id = modelId.toLowerCase()
  return /gemini-3(?:\.\d+)?-flash/.test(id) || id === 'gemini-flash-latest' || id === 'gemini-flash-lite-latest'
}

/** pi-ai getDisabledThinkingConfig: Gemini 3 cannot turn thinking fully off. */
function disabledThinkingConfig(modelId: string): Record<string, unknown> {
  if (isGemini3ProModel(modelId)) return { thinkingLevel: 'LOW' }
  if (isGemini3FlashModel(modelId)) return { thinkingLevel: 'MINIMAL' }
  return { thinkingBudget: 0 }
}

interface VertexPayload {
  contents: VertexContent[]
  model: string
  readonly config?: {
    generationConfig?: { maxOutputTokens?: number; temperature?: number }
    systemInstruction?: { parts: VertexPart[] }
    thinkingConfig?: Record<string, unknown>
    toolConfig?: { functionCallingConfig: { mode: 'ANY' | 'AUTO' | 'NONE' } }
    tools?: { functionDeclarations: Record<string, unknown>[] }[]
  }
}

function toolChoiceMode(choice: ToolChoice | undefined): 'ANY' | 'AUTO' | 'NONE' | undefined {
  switch (choice) {
    case 'any':
      return 'ANY'
    case 'auto':
      return 'AUTO'
    case 'none':
      return 'NONE'
    default:
      return undefined
  }
}

/** pi-ai buildParams: the native generateContent request body. */
export function vertexPayload(request: CompletionRequest): VertexPayload {
  const modelId = bareModel(request.model)
  const capabilities: PiModelCapabilities | undefined = piCatalogModelCapabilities(modelId, 'google-vertex')
  const config: NonNullable<VertexPayload['config']> = {}
  const generationConfig: { maxOutputTokens?: number; temperature?: number } = {}
  if (request.temperature !== undefined) generationConfig.temperature = request.temperature
  if (request.maxTokens !== undefined) generationConfig.maxOutputTokens = request.maxTokens
  if (Object.keys(generationConfig).length) config.generationConfig = generationConfig

  const systemParts = request.messages
    .filter(message => message.role === 'system')
    .map(message => sanitizeSurrogates(messageText(message)))
    .filter(Boolean)
  if (systemParts.length) {
    config.systemInstruction = { parts: [{ text: systemParts.join('\n\n') }] }
  }

  if (request.tools?.length) {
    config.tools = [{
      functionDeclarations: request.tools.map(tool => ({
        name: tool.function.name,
        description: tool.function.description,
        parametersJsonSchema: sanitizeForOpenApi(tool.function.parameters),
      })),
    }]
    const mode = toolChoiceMode(request.toolChoice)
    if (mode) config.toolConfig = { functionCallingConfig: { mode } }
  }

  if (request.thinking && capabilities?.reasoning !== false) {
    const thinkingConfig: Record<string, unknown> = { includeThoughts: true }
    const mapped = request.thinking.effort === undefined || request.thinking.effort === 'off'
      ? undefined
      : capabilities?.thinkingLevelMap?.[request.thinking.effort]
    const level = typeof mapped === 'string'
      ? thinkingLevelFromEffort(mapped.toLowerCase())
      : thinkingLevelFromEffort(request.thinking.effort)
    if (level) {
      thinkingConfig.thinkingLevel = level
    } else if (request.thinking.budgetTokens !== undefined) {
      thinkingConfig.thinkingBudget = request.thinking.budgetTokens
    }
    config.thinkingConfig = thinkingConfig
  } else if (request.thinking && capabilities?.reasoning === false) {
    // Explicitly disabled thinking on a reasoning model still needs a config
    // so Gemini 3 models do not leak hidden thoughts into the reply.
    config.thinkingConfig = disabledThinkingConfig(modelId)
  }

  return {
    model: modelId,
    contents: vertexContentsFromMessages(
      request.messages.filter(message => message.role !== 'system'),
      modelId,
    ),
    ...(Object.keys(config).length ? { config } : {}),
  }
}

function streamEndpoint(options: {
  apiKey: string | undefined
  baseUrl: string | undefined
  location: string | undefined
  model: string
  method: 'generateContent' | 'streamGenerateContent'
  project: string | undefined
}): { url: string; headers: Record<string, string> } {
  let root = options.baseUrl?.trim() || undefined
  if (root) {
    root = trimSlash(root.replaceAll('{location}', options.location ?? ''))
  }
  const path = options.apiKey
    ? `${root ?? VERTEX_EXPRESS_API_ROOT}/${API_VERSION}/publishers/google/models/${options.model}:${options.method}`
    : `${root ?? `${options.location}-aiplatform.googleapis.com`}/${API_VERSION}/projects/${options.project}/locations/${options.location}/publishers/google/models/${options.model}:${options.method}`
  const url = `${path.startsWith('http') ? path : `https://${path}`}${options.method === 'streamGenerateContent' ? '?alt=sse' : ''}`
  return {
    url,
    headers: options.apiKey ? { 'x-goog-api-key': options.apiKey } : {},
  }
}
interface VertexChunk {
  candidates?: {
    content?: { parts?: { functionCall?: { args?: unknown; id?: string; name?: string }; text?: string; thought?: boolean; thoughtSignature?: string }[] }
    finishReason?: string
  }[]
  responseId?: string
  usageMetadata?: {
    cachedContentTokenCount?: number
    candidatesTokenCount?: number
    promptTokenCount?: number
    thoughtsTokenCount?: number
    totalTokenCount?: number
  }
}

function vertexUsage(metadata: NonNullable<VertexChunk['usageMetadata']>): TokenUsage {
  const cached = metadata.cachedContentTokenCount ?? 0
  const thoughts = metadata.thoughtsTokenCount ?? 0
  return {
    inputTokens: Math.max(0, (metadata.promptTokenCount ?? 0) - cached),
    outputTokens: (metadata.candidatesTokenCount ?? 0) + thoughts,
    ...(cached ? { cacheReadTokens: cached } : {}),
    ...(thoughts ? { reasoningTokens: thoughts } : {}),
  }
}

interface PendingVertexToolCall {
  arguments: JsonObject
  readonly id: string
  readonly name: string
}

let vertexToolCallCounter = 0

/** pi-ai's synthesized id shape when Vertex provides none. */
function synthesizedToolCallId(name: string): string {
  vertexToolCallCounter += 1
  return `${name}_${Date.now()}_${vertexToolCallCounter}`
}

async function requestVertexStream(
  request: CompletionRequest,
  options: GoogleVertexClientOptions,
  stream: boolean,
  signal: AbortSignal | undefined,
): Promise<Response> {
  const fetchImplementation = options.fetchImplementation ?? fetch
  const apiKey = resolveVertexApiKey(options)
  const location = apiKey ? undefined : resolveVertexLocation(options)
  const project = apiKey ? undefined : resolveVertexProject(options)
  const endpoint = streamEndpoint({
    apiKey,
    baseUrl: options.baseUrl,
    location,
    model: bareModel(request.model),
    method: stream ? 'streamGenerateContent' : 'generateContent',
    project,
  })
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...endpoint.headers,
  }
  if (!apiKey) {
    headers.Authorization = `Bearer ${await resolveVertexAccessToken(options, fetchImplementation)}`
  }
  const response = await fetchImplementation(endpoint.url, {
    method: 'POST',
    headers,
    body: JSON.stringify(vertexPayload(request)),
    ...(signal ? { signal } : {}),
  })
  if (!response.ok) {
    const body = await response.text()
    throw new ProviderError(
      'google-vertex',
      `generateContent request failed (${response.status}): ${body.slice(0, 4_096)}`,
    )
  }
  return response
}

function throwOnFatalFinish(reason: string | undefined): void {
  if (reason && FATAL_FINISH_REASONS.has(reason)) {
    throw new ProviderError('google-vertex', `Provider stopped with: ${reason}`)
  }
}

function mapFinishReason(reason: string | undefined, hasToolCalls: boolean): string | undefined {
  if (!reason) return undefined
  if (reason === 'STOP') return hasToolCalls ? 'tool_calls' : 'stop'
  if (reason === 'MAX_TOKENS') return 'length'
  return undefined
}

export class GoogleVertexClient implements LlmClient {
  private readonly options: GoogleVertexClientOptions

  constructor(options: GoogleVertexClientOptions = {}) {
    this.options = options
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    const response = await requestVertexStream(request, this.options, false, signal)
    const chunk = await response.json() as VertexChunk
    const candidate = chunk.candidates?.[0]
    throwOnFatalFinish(candidate?.finishReason)

    const content: string[] = []
    const thinking: string[] = []
    let thinkingSignature: string | undefined
    const toolCalls: ToolCall[] = []
    for (const part of candidate?.content?.parts ?? []) {
      if (part.functionCall) {
        const args = (part.functionCall.args ?? {}) as JsonObject
        toolCalls.push({
          id: part.functionCall.id || synthesizedToolCallId(part.functionCall.name ?? ''),
          type: "function",
          function: { name: part.functionCall.name ?? "", arguments: args },
        })
        continue
      }
      if (typeof part.text !== 'string' || part.text === '') continue
      if (part.thought === true) {
        thinking.push(part.text)
        const signature = isValidThoughtSignature(part.thoughtSignature) ? part.thoughtSignature : undefined
        if (signature) thinkingSignature = signature
        continue
      }
      content.push(part.text)
      if (isValidThoughtSignature(part.thoughtSignature)) thinkingSignature = part.thoughtSignature
    }

    const finishReason = mapFinishReason(candidate?.finishReason, toolCalls.length > 0)
    return {
      content: content.join(''),
      toolCalls,
      ...(finishReason ? { finishReason } : {}),
      ...(thinking.length ? { thinking: thinking.join('') } : {}),
      ...(thinkingSignature ? { thinkingSignature } : {}),
      ...(chunk.usageMetadata ? { usage: vertexUsage(chunk.usageMetadata) } : {}),
    }
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const response = await requestVertexStream(request, this.options, true, signal)
    if (!response.body) {
      throw new ProviderError('google-vertex', 'streamGenerateContent returned no response body')
    }
    const pendingToolCalls = new Map<number, PendingVertexToolCall>()
    let rawFinishReason: string | undefined
    let usage: TokenUsage | undefined
    let thinkingSignature: string | undefined

    for await (const data of internalSseData(response.body)) {
      if (data === '[DONE]') break
      let chunk: VertexChunk
      try {
        chunk = JSON.parse(data) as VertexChunk
      } catch {
        throw new ProviderError('google-vertex', 'streamGenerateContent produced a malformed SSE event')
      }
      const candidate = chunk.candidates?.[0]
      for (const [index, part] of (candidate?.content?.parts ?? []).entries()) {
        if (part.functionCall) {
          const args = (part.functionCall.args ?? {}) as JsonObject
          pendingToolCalls.set(index, {
            id: part.functionCall.id || synthesizedToolCallId(part.functionCall.name ?? ''),
            name: part.functionCall.name ?? '',
            arguments: args,
          })
          continue
        }
        if (typeof part.text !== 'string' || part.text === '') continue
        const signature = isValidThoughtSignature(part.thoughtSignature) ? part.thoughtSignature : undefined
        if (signature) thinkingSignature = signature
        if (part.thought === true) {
          yield {
            thinking: part.text,
            ...(signature ? { thinkingSignature: signature } : {}),
          }
          continue
        }
        yield {
          content: part.text,
          ...(signature ? { thinkingSignature: signature } : {}),
        }
      }
      if (candidate?.finishReason) {
        throwOnFatalFinish(candidate.finishReason)
        rawFinishReason = candidate.finishReason
      }
      if (chunk.usageMetadata) usage = vertexUsage(chunk.usageMetadata)
    }

    // STOP means tool calls when any were collected, plain stop otherwise —
    // decided after the stream so the ordering of finish vs parts is irrelevant.
    const finish = rawFinishReason === 'MAX_TOKENS'
      ? 'length'
      : rawFinishReason === 'STOP'
        ? (pendingToolCalls.size ? 'tool_calls' : 'stop')
        : undefined

    if (pendingToolCalls.size) {
      yield {
        toolCalls: [...pendingToolCalls.values()].map(call => ({
          id: call.id,
          type: 'function' as const,
          function: { name: call.name, arguments: call.arguments },
        })),
      }
    }
    if (usage) yield { usage }
    if (finish) yield { finishReason: finish }
  }
}
