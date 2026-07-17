// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { timingSafeEqual } from 'node:crypto'

import type { CompletionRequest, LlmClient, LlmDelta, TokenUsage } from '../llms/client.js'
import { ThinkingParser } from '../streaming/thinkingParser.js'
import type { ToolCall } from '../types/toolCalls.js'
import {
  CortexCompletionService,
  type CortexCompletionMetadata,
  type CortexCompletionRequest,
} from './cortexCompletionService.js'
import {
  ApiRequestError,
  parseChatCompletionRequest,
  toOpenAiToolCalls,
  toOpenAiUsage,
  type OpenAiUsage,
} from './protocol.js'

export type LlmClientResolver = (model: string) => LlmClient

const DEFAULT_HOSTNAME = '127.0.0.1'
const DEFAULT_MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024
const MAX_RATE_LIMIT_KEYS = 10_000

/** Browser cross-origin policy applied when the API is called from a web page. */
export interface OpenAiApiCorsOptions {
  /** Origins allowed to read API responses. Use `'*'` only for public, credential-free APIs. */
  readonly origins: '*' | readonly string[]
  /** Whether browsers may include credentials. This cannot be combined with a wildcard origin. */
  readonly allowCredentials?: boolean
  /** Request headers accepted during a browser preflight. */
  readonly allowHeaders?: readonly string[]
  /** HTTP methods accepted during a browser preflight. */
  readonly allowMethods?: readonly string[]
  /** Browser preflight cache lifetime in seconds. */
  readonly maxAge?: number
}

/** Optional bearer-token protection for API routes. `/health` remains available for probes by default. */
export interface OpenAiApiBearerAuthOptions {
  /** The exact token expected after the `Bearer` authorization scheme. */
  readonly token: string
  /** Paths that bypass bearer authentication. Defaults to `['/health']`. */
  readonly exemptPaths?: readonly string[]
}

/** Per-key sliding-window request limit. It is inactive unless supplied. */
export interface OpenAiApiRateLimitOptions {
  /** Maximum requests allowed from one key during a window. */
  readonly maxRequests: number
  /** Window length in milliseconds. Defaults to one minute. */
  readonly windowMs?: number
  /**
   * Derives a stable bucket key. Bun listeners receive the peer address; direct
   * `fetch()` calls leave it undefined and therefore use the global bucket.
   */
  readonly key?: (request: Request, clientAddress: string | undefined) => string
}

export interface OpenAiApiServerOptions {
  /** One client for every exposed model, or a resolver for model-specific clients. */
  readonly llm: LlmClient | LlmClientResolver
  /** Model ids the endpoint advertises and accepts. */
  readonly models: readonly string[]
  /** Optional native Cortex backend for models whose id contains `cortex`. */
  readonly cortex?: CortexCompletionService
  /** Enables explicit browser CORS handling. No CORS headers are emitted by default. */
  readonly cors?: OpenAiApiCorsOptions
  /** Enables bearer authentication for all paths except configured exemptions. */
  readonly auth?: OpenAiApiBearerAuthOptions
  /**
   * Rejects chat-completion request bodies larger than this many bytes.
   * Defaults to 16 MiB; set to 0 to disable the limit.
   */
  readonly maxRequestBodyBytes?: number
  /** Enables in-memory sliding-window request limiting. Disabled by default. */
  readonly rateLimit?: OpenAiApiRateLimitOptions
  /** Injectable millisecond clock used to derive OpenAI's epoch-second timestamps. */
  readonly now?: () => number
  /** Injectable id source for deterministic integration tests. */
  readonly responseId?: () => string
}

export interface OpenAiApiHandler {
  fetch(request: Request): Promise<Response>
}

export interface OpenAiApiListenOptions {
  /**
   * Interface to bind. Defaults to loopback (`127.0.0.1`) so an unauthenticated
   * server is not exposed on every interface; pass a public address explicitly.
   */
  readonly hostname?: string
  readonly port?: number
  /**
   * Seconds a connection may idle before Bun closes it. Defaults to 0 (disabled)
   * so long-running completions and quiet SSE streams are not cut off mid-turn.
   */
  readonly idleTimeout?: number
}

interface CompletionAggregate {
  readonly content: string
  readonly finishReason: string
  readonly toolCalls: readonly ToolCall[]
  readonly usage: OpenAiUsage
}

interface CorsConfiguration {
  readonly allowCredentials: boolean
  readonly allowHeaders: readonly string[]
  readonly allowHeaderNames: ReadonlySet<string>
  readonly allowMethods: readonly string[]
  readonly allowMethodNames: ReadonlySet<string>
  readonly maxAge: number | undefined
  readonly origins: '*' | ReadonlySet<string>
}

interface BearerAuthConfiguration {
  readonly exemptPaths: ReadonlySet<string>
  readonly token: string
}

interface RateLimitConfiguration {
  readonly key: (request: Request, clientAddress: string | undefined) => string
  readonly limiter: SlidingWindowRateLimiter
}

type RateLimitDecision =
  | { readonly allowed: true; readonly headers: Headers }
  | { readonly allowed: false; readonly headers: Headers; readonly retryAfterSeconds: number }

class RequestBodyTooLargeError extends Error {
  constructor() {
    super('Request body too large.')
    this.name = 'RequestBodyTooLargeError'
  }
}

class InvalidContentLengthError extends Error {
  constructor() {
    super('Invalid Content-Length header.')
    this.name = 'InvalidContentLengthError'
  }
}

class SlidingWindowRateLimiter {
  private readonly requestTimes = new Map<string, number[]>()
  private requestsSinceCleanup = 0

  constructor(
    private readonly maxRequests: number,
    private readonly windowMs: number,
  ) {}

  consume(key: string, now: number): RateLimitDecision {
    const windowStart = now - this.windowMs
    const activeRequests = (this.requestTimes.get(key) ?? []).filter(timestamp => timestamp > windowStart)
    this.cleanupStaleEntries(windowStart)
    if (activeRequests.length >= this.maxRequests) {
      this.requestTimes.set(key, activeRequests)
      const retryAfterSeconds = Math.max(1, Math.ceil(((activeRequests[0] ?? now) + this.windowMs - now) / 1_000))
      return {
        allowed: false,
        retryAfterSeconds,
        headers: rateLimitHeaders(this.maxRequests, 0, activeRequests[0] ?? now, this.windowMs),
      }
    }

    activeRequests.push(now)
    this.requestTimes.set(key, activeRequests)
    this.evictOldestKeyWhenFull()
    return {
      allowed: true,
      headers: rateLimitHeaders(
        this.maxRequests,
        this.maxRequests - activeRequests.length,
        activeRequests[0] ?? now,
        this.windowMs,
      ),
    }
  }

  /** Keep the key map bounded under multi-IP floods by evicting the idlest key. */
  private evictOldestKeyWhenFull(): void {
    if (this.requestTimes.size <= MAX_RATE_LIMIT_KEYS) {
      return
    }
    let oldestKey: string | undefined
    let oldestSeen = Number.POSITIVE_INFINITY
    for (const [key, timestamps] of this.requestTimes) {
      const lastSeen = timestamps[timestamps.length - 1] ?? 0
      if (lastSeen < oldestSeen) {
        oldestSeen = lastSeen
        oldestKey = key
      }
    }
    if (oldestKey !== undefined) {
      this.requestTimes.delete(oldestKey)
    }
  }

  private cleanupStaleEntries(windowStart: number): void {
    this.requestsSinceCleanup += 1
    if (this.requestsSinceCleanup < 100) {
      return
    }
    this.requestsSinceCleanup = 0
    for (const [key, timestamps] of this.requestTimes) {
      const activeRequests = timestamps.filter(timestamp => timestamp > windowStart)
      if (activeRequests.length) {
        this.requestTimes.set(key, activeRequests)
      } else {
        this.requestTimes.delete(key)
      }
    }
  }
}

/**
 * Small Bun-native OpenAI-compatible HTTP surface over the portable LLM client.
 * It deliberately owns no session state: callers send the complete conversation
 * in each request, as required by the chat-completions protocol.
 */
export class OpenAiApiServer implements OpenAiApiHandler {
  private readonly auth: BearerAuthConfiguration | undefined
  private readonly clientResolver: LlmClientResolver
  private readonly cortex: CortexCompletionService | undefined
  private readonly cors: CorsConfiguration | undefined
  private readonly maxRequestBodyBytes: number | undefined
  private readonly models: readonly string[]
  private readonly now: () => number
  private readonly rateLimit: RateLimitConfiguration | undefined
  private readonly responseId: () => string

  constructor(options: OpenAiApiServerOptions) {
    this.auth = options.auth === undefined ? undefined : normalizeBearerAuth(options.auth)
    this.cortex = options.cortex
    this.cors = options.cors === undefined ? undefined : normalizeCors(options.cors)
    this.maxRequestBodyBytes = normalizeMaxRequestBodyBytes(
      options.maxRequestBodyBytes ?? DEFAULT_MAX_REQUEST_BODY_BYTES,
    )
    this.rateLimit = options.rateLimit === undefined ? undefined : normalizeRateLimit(options.rateLimit)
    this.models = [...new Set(options.models)]
    this.now = options.now ?? Date.now
    this.responseId = options.responseId ?? (() => `chatcmpl-${crypto.randomUUID()}`)
    const llm = options.llm
    this.clientResolver = typeof llm === 'function' ? llm : () => llm
  }

  async fetch(request: Request, clientAddress: string | undefined = undefined): Promise<Response> {
    const preflight = this.corsPreflight(request)
    if (preflight) {
      return preflight
    }

    const authenticationFailure = this.authenticationFailure(request)
    if (authenticationFailure) {
      return this.decorateResponse(request, authenticationFailure)
    }

    let rateLimitDecision: RateLimitDecision | null
    try {
      rateLimitDecision = this.consumeRateLimit(request, clientAddress)
    } catch {
      return this.decorateResponse(request, apiError(500, 'Internal server error.', 'api_error', null, null))
    }
    if (rateLimitDecision && !rateLimitDecision.allowed) {
      return this.decorateResponse(
        request,
        apiError(
          429,
          'Rate limit exceeded.',
          'rate_limit_error',
          null,
          'rate_limit_exceeded',
          { 'Retry-After': String(rateLimitDecision.retryAfterSeconds) },
        ),
        rateLimitDecision.headers,
      )
    }

    const pathname = new URL(request.url).pathname
    let response: Response
    if (pathname === '/health') {
      response = request.method === 'GET'
        ? json({ status: 'healthy', agents: this.models.length })
        : methodNotAllowed('GET')
    } else if (pathname === '/v1/models') {
      response = request.method === 'GET' ? this.listModels() : methodNotAllowed('GET')
    } else if (pathname === '/v1/chat/completions') {
      response = request.method === 'POST' ? await this.chatCompletions(request) : methodNotAllowed('POST')
    } else {
      response = apiError(404, 'Not found.', 'invalid_request_error', null, 'not_found')
    }
    return this.decorateResponse(request, response, rateLimitDecision?.headers)
  }

  /** Start this handler on Bun's native HTTP server, bound to loopback unless overridden. */
  listen(options: OpenAiApiListenOptions = {}) {
    const hostname = options.hostname ?? DEFAULT_HOSTNAME
    if (this.auth === undefined && !isLoopbackHostname(hostname)) {
      console.warn(
        `OpenAiApiServer is binding to '${hostname}' without bearer authentication; ` +
        'the OpenAI-compatible API will be reachable without credentials. ' +
        'Configure the auth option or keep the default 127.0.0.1 loopback bind.',
      )
    }
    return Bun.serve({
      ...options,
      hostname,
      idleTimeout: options.idleTimeout ?? 0,
      fetch: (request, server) => this.fetch(request, server.requestIP(request)?.address),
    })
  }

  private listModels(): Response {
    const created = this.epochSeconds()
    return json({
      object: 'list',
      data: this.models.map(id => ({ id, object: 'model', created, owned_by: 'xerxes' })),
    })
  }

  private async chatCompletions(request: Request): Promise<Response> {
    let input: unknown
    try {
      input = await readJsonBody(request, this.maxRequestBodyBytes)
    } catch (error) {
      if (error instanceof RequestBodyTooLargeError) {
        return apiError(413, error.message, 'invalid_request_error', null, 'request_too_large')
      }
      if (error instanceof InvalidContentLengthError) {
        return apiError(400, error.message, 'invalid_request_error', null, 'invalid_content_length')
      }
      return apiError(400, 'Invalid JSON request body.', 'invalid_request_error', null, null)
    }

    let parsed: ReturnType<typeof parseChatCompletionRequest>
    try {
      parsed = parseChatCompletionRequest(input)
    } catch (error) {
      if (error instanceof ApiRequestError) {
        return apiError(400, error.message, 'invalid_request_error', error.parameter, null)
      }
      return apiError(400, 'Invalid chat completion request.', 'invalid_request_error', null, null)
    }
    if (!this.models.includes(parsed.completion.model)) {
      return apiError(
        404,
        `The model '${parsed.completion.model}' does not exist.`,
        'invalid_request_error',
        'model',
        'model_not_found',
      )
    }

    if (isCortexModel(parsed.completion.model)) {
      return this.cortexChatCompletions(parsed, request.signal)
    }

    let client: LlmClient
    try {
      client = this.clientResolver(parsed.completion.model)
    } catch {
      return apiError(500, 'Internal server error.', 'api_error', null, null)
    }
    const id = this.responseId()
    const created = this.epochSeconds()
    if (parsed.stream) {
      return this.streamingCompletion(client, parsed.completion, parsed.includeUsage, id, created, request.signal)
    }
    try {
      const aggregate = await collectCompletion(client, parsed.completion, request.signal)
      return json({
        id,
        object: 'chat.completion',
        created,
        model: parsed.completion.model,
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: aggregate.content,
            ...(aggregate.toolCalls.length ? { tool_calls: toOpenAiToolCalls(aggregate.toolCalls) } : {}),
          },
          finish_reason: aggregate.finishReason,
        }],
        usage: aggregate.usage,
      })
    } catch {
      return apiError(500, 'Internal server error.', 'api_error', null, null)
    }
  }

  private async cortexChatCompletions(
    parsed: ReturnType<typeof parseChatCompletionRequest>,
    requestSignal: AbortSignal,
  ): Promise<Response> {
    const cortex = this.cortex
    if (cortex === undefined) {
      return apiError(
        404,
        'Cortex is not enabled on this server.',
        'invalid_request_error',
        'model',
        'cortex_unavailable',
      )
    }
    let metadata: CortexCompletionMetadata | undefined
    try {
      metadata = cortexMetadata(parsed.metadata)
    } catch (error) {
      if (error instanceof ApiRequestError) {
        return apiError(400, error.message, 'invalid_request_error', error.parameter, null)
      }
      return apiError(400, 'Invalid Cortex metadata.', 'invalid_request_error', 'metadata', null)
    }
    const completion: CortexCompletionRequest = {
      model: parsed.completion.model,
      messages: parsed.completion.messages,
      ...(metadata === undefined ? {} : { metadata }),
    }
    if (parsed.stream) {
      return this.streamingCortexCompletion(cortex, completion, requestSignal)
    }
    try {
      return json(await cortex.createCompletion(completion, requestSignal))
    } catch {
      return apiError(500, 'Internal server error.', 'api_error', null, null)
    }
  }

  private streamingCompletion(
    client: LlmClient,
    completion: CompletionRequest,
    includeUsage: boolean | undefined,
    id: string,
    created: number,
    requestSignal: AbortSignal,
  ): Response {
    const upstreamAbort = new AbortController()
    const abortUpstream = () => upstreamAbort.abort()
    requestSignal.addEventListener('abort', abortUpstream, { once: true })
    let cancelled = false
    const encoder = new TextEncoder()
    const body = new ReadableStream<Uint8Array>({
      start: async controller => {
        const send = (payload: unknown): void => {
          if (!cancelled) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`))
          }
        }
        const sendDone = (): void => {
          if (!cancelled) {
            controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          }
        }
        const sendChunk = (
          delta: Record<string, unknown>,
          finishReason: string | null = null,
          usage?: OpenAiUsage,
        ): void => {
          send({
            id,
            object: 'chat.completion.chunk',
            created,
            model: completion.model,
            choices: [{ index: 0, delta, finish_reason: finishReason }],
            ...(usage ? { usage } : {}),
          })
        }

        const parser = new ThinkingParser()
        let finishReason: string | undefined
        let lastUsage: TokenUsage | undefined
        let toolCalls: readonly ToolCall[] = []
        try {
          sendChunk({ role: 'assistant' })
          for await (const delta of client.stream(completion, upstreamAbort.signal)) {
            finishReason = delta.finishReason ?? finishReason
            lastUsage = delta.usage ?? lastUsage
            toolCalls = delta.toolCalls ?? toolCalls
            for (const text of visibleText(parser, delta)) {
              sendChunk({ content: text })
            }
            if (delta.toolCalls?.length) {
              sendChunk({
                tool_calls: toOpenAiToolCalls(delta.toolCalls).map((toolCall, index) => ({ index, ...toolCall })),
              })
            }
          }
          for (const text of visibleTail(parser)) {
            sendChunk({ content: text })
          }
          const usage = lastUsage !== undefined && includeUsage !== false ? toOpenAiUsage(lastUsage) : undefined
          sendChunk({}, resolvedFinishReason(finishReason, toolCalls), usage)
          sendDone()
        } catch {
          if (!cancelled) {
            send({ error: openAiError('Internal server error.', 'api_error', null, null) })
            sendDone()
          }
        } finally {
          requestSignal.removeEventListener('abort', abortUpstream)
          if (!cancelled) {
            controller.close()
          }
        }
      },
      cancel: () => {
        cancelled = true
        requestSignal.removeEventListener('abort', abortUpstream)
        upstreamAbort.abort()
      },
    })
    return new Response(body, {
      headers: {
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'Content-Type': 'text/event-stream; charset=utf-8',
      },
    })
  }

  private streamingCortexCompletion(
    cortex: CortexCompletionService,
    completion: CortexCompletionRequest,
    requestSignal: AbortSignal,
  ): Response {
    const upstreamAbort = new AbortController()
    const abortUpstream = () => upstreamAbort.abort(requestSignal.reason)
    requestSignal.addEventListener('abort', abortUpstream, { once: true })
    let cancelled = false
    const encoder = new TextEncoder()
    const body = new ReadableStream<Uint8Array>({
      start: async controller => {
        try {
          for await (const frame of cortex.createStreamingCompletion(completion, upstreamAbort.signal)) {
            if (!cancelled) {
              controller.enqueue(encoder.encode(frame))
            }
          }
        } catch {
          if (!cancelled) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: openAiError('Internal server error.', 'api_error', null, null) })}\n\n`))
            controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          }
        } finally {
          requestSignal.removeEventListener('abort', abortUpstream)
          if (!cancelled) {
            controller.close()
          }
        }
      },
      cancel: () => {
        cancelled = true
        requestSignal.removeEventListener('abort', abortUpstream)
        upstreamAbort.abort()
      },
    })
    return new Response(body, {
      headers: {
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'Content-Type': 'text/event-stream; charset=utf-8',
      },
    })
  }

  private epochSeconds(): number {
    return Math.floor(this.now() / 1_000)
  }

  private authenticationFailure(request: Request): Response | null {
    if (!this.auth || this.auth.exemptPaths.has(new URL(request.url).pathname)) {
      return null
    }
    const authorization = request.headers.get('authorization')
    if (authorization !== null && bearerAuthorizationMatches(authorization, this.auth.token)) {
      return null
    }
    return apiError(
      401,
      'Invalid authentication credentials.',
      'authentication_error',
      null,
      'invalid_api_key',
      { 'WWW-Authenticate': 'Bearer' },
    )
  }

  private consumeRateLimit(request: Request, clientAddress: string | undefined): RateLimitDecision | null {
    if (!this.rateLimit) {
      return null
    }
    const key = this.rateLimit.key(request, clientAddress)
    if (!key) {
      throw new Error('Rate-limit keys must be non-empty strings.')
    }
    return this.rateLimit.limiter.consume(key, this.now())
  }

  private corsPreflight(request: Request): Response | null {
    const cors = this.cors
    if (!cors || request.method !== 'OPTIONS' || !request.headers.has('access-control-request-method')) {
      return null
    }
    const headers = this.corsHeaders(request)
    if (!headers) {
      return apiError(403, 'CORS origin is not allowed.', 'invalid_request_error', null, 'cors_origin_denied')
    }

    const requestedMethod = request.headers.get('access-control-request-method')?.toUpperCase()
    if (!requestedMethod || !cors.allowMethodNames.has(requestedMethod)) {
      return withHeaders(
        apiError(405, 'CORS method is not allowed.', 'invalid_request_error', null, 'cors_method_denied'),
        headers,
      )
    }
    const requestedHeaders = splitHeaderValues(request.headers.get('access-control-request-headers'))
    if (!requestedHeaders.every(header => cors.allowHeaderNames.has('*') || cors.allowHeaderNames.has(header))) {
      return withHeaders(
        apiError(400, 'CORS request headers are not allowed.', 'invalid_request_error', null, 'cors_headers_denied'),
        headers,
      )
    }

    headers.set('Access-Control-Allow-Methods', cors.allowMethods.join(', '))
    headers.set('Access-Control-Allow-Headers', cors.allowHeaders.join(', '))
    if (cors.maxAge !== undefined) {
      headers.set('Access-Control-Max-Age', String(cors.maxAge))
    }
    appendVary(headers, 'Access-Control-Request-Method')
    appendVary(headers, 'Access-Control-Request-Headers')
    return new Response(null, { status: 204, headers })
  }

  private corsHeaders(request: Request): Headers | null {
    if (!this.cors) {
      return null
    }
    const origin = request.headers.get('origin')
    if (!origin || (this.cors.origins !== '*' && !this.cors.origins.has(origin))) {
      return null
    }
    const headers = new Headers()
    if (this.cors.origins === '*') {
      headers.set('Access-Control-Allow-Origin', '*')
    } else {
      headers.set('Access-Control-Allow-Origin', origin)
      appendVary(headers, 'Origin')
    }
    if (this.cors.allowCredentials) {
      headers.set('Access-Control-Allow-Credentials', 'true')
    }
    return headers
  }

  private decorateResponse(
    request: Request,
    response: Response,
    rateHeaders: Headers | undefined = undefined,
  ): Response {
    if (rateHeaders) {
      withHeaders(response, rateHeaders)
    }
    const corsHeaders = this.corsHeaders(request)
    if (corsHeaders) {
      withHeaders(response, corsHeaders)
    }
    return response
  }
}

export function createOpenAiApiServer(options: OpenAiApiServerOptions): OpenAiApiServer {
  return new OpenAiApiServer(options)
}

function normalizeCors(options: OpenAiApiCorsOptions): CorsConfiguration {
  const allowCredentials = options.allowCredentials ?? false
  const origins = options.origins
  if (origins === '*') {
    if (allowCredentials) {
      throw new Error('CORS wildcard origins cannot allow credentials.')
    }
  } else if (!Array.isArray(origins) || origins.length === 0) {
    throw new Error('CORS origins must contain at least one origin.')
  }

  const allowHeaders = normalizeValues(options.allowHeaders ?? ['Authorization', 'Content-Type'], 'CORS header')
  const allowMethods = normalizeValues(
    options.allowMethods ?? ['GET', 'POST', 'OPTIONS'],
    'CORS method',
    value => value.toUpperCase(),
  )
  const maxAge = options.maxAge
  if (maxAge !== undefined) {
    assertNonNegativeSafeInteger(maxAge, 'CORS maxAge')
  }

  return {
    allowCredentials,
    allowHeaders,
    allowHeaderNames: new Set(allowHeaders.map(header => header.toLowerCase())),
    allowMethods,
    allowMethodNames: new Set(allowMethods),
    maxAge,
    origins: origins === '*' ? origins : new Set(normalizeValues(origins, 'CORS origin')),
  }
}

function normalizeBearerAuth(options: OpenAiApiBearerAuthOptions): BearerAuthConfiguration {
  if (!isNonEmptyString(options.token)) {
    throw new Error('Bearer authentication requires a non-empty token.')
  }
  const exemptPaths = options.exemptPaths ?? ['/health']
  if (!Array.isArray(exemptPaths) || exemptPaths.some(path => !isNonEmptyString(path) || !path.startsWith('/'))) {
    throw new Error('Bearer-auth exempt paths must be non-empty absolute paths.')
  }
  return { token: options.token, exemptPaths: new Set(exemptPaths) }
}

function normalizeMaxRequestBodyBytes(value: number): number | undefined {
  assertNonNegativeSafeInteger(value, 'maxRequestBodyBytes')
  return value === 0 ? undefined : value
}

function normalizeRateLimit(options: OpenAiApiRateLimitOptions): RateLimitConfiguration {
  assertPositiveSafeInteger(options.maxRequests, 'rateLimit.maxRequests')
  const windowMs = options.windowMs ?? 60_000
  assertPositiveSafeInteger(windowMs, 'rateLimit.windowMs')
  if (options.key !== undefined && typeof options.key !== 'function') {
    throw new Error('rateLimit.key must be a function.')
  }
  return {
    limiter: new SlidingWindowRateLimiter(options.maxRequests, windowMs),
    key: options.key ?? ((_request, clientAddress) => clientAddress ?? 'global'),
  }
}

function normalizeValues(
  values: readonly string[],
  label: string,
  transform: (value: string) => string = value => value,
): readonly string[] {
  if (values.length === 0) {
    throw new Error(`${label} values must not be empty.`)
  }
  const normalized = new Set<string>()
  for (const value of values) {
    if (!isNonEmptyString(value)) {
      throw new Error(`${label} values must be non-empty strings.`)
    }
    normalized.add(transform(value.trim()))
  }
  return [...normalized]
}

function assertNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} must be a non-negative safe integer.`)
  }
}

function assertPositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive safe integer.`)
  }
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

async function readJsonBody(request: Request, maxBytes: number | undefined): Promise<unknown> {
  if (maxBytes === undefined) {
    return request.json()
  }
  const contentLength = request.headers.get('content-length')
  if (contentLength !== null) {
    const declaredLength = Number(contentLength)
    if (!/^\d+$/.test(contentLength) || !Number.isSafeInteger(declaredLength)) {
      throw new InvalidContentLengthError()
    }
    if (declaredLength > maxBytes) {
      throw new RequestBodyTooLargeError()
    }
  }
  const body = await readBoundedBody(request, maxBytes)
  return JSON.parse(new TextDecoder().decode(body))
}

async function readBoundedBody(request: Request, maxBytes: number): Promise<Uint8Array> {
  const reader = request.body?.getReader()
  if (!reader) {
    return new Uint8Array()
  }
  const chunks: Uint8Array[] = []
  let size = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }
      if (!value) {
        continue
      }
      size += value.byteLength
      if (size > maxBytes) {
        await reader.cancel().catch(() => undefined)
        throw new RequestBodyTooLargeError()
      }
      chunks.push(value)
    }
  } finally {
    reader.releaseLock()
  }

  const body = new Uint8Array(size)
  let offset = 0
  for (const chunk of chunks) {
    body.set(chunk, offset)
    offset += chunk.byteLength
  }
  return body
}

async function collectCompletion(
  client: LlmClient,
  completion: CompletionRequest,
  signal: AbortSignal,
): Promise<CompletionAggregate> {
  const parser = new ThinkingParser()
  const content: string[] = []
  let finishReason: string | undefined
  let lastUsage: TokenUsage | undefined
  let toolCalls: readonly ToolCall[] = []
  for await (const delta of client.stream(completion, signal)) {
    finishReason = delta.finishReason ?? finishReason
    lastUsage = delta.usage ?? lastUsage
    toolCalls = delta.toolCalls ?? toolCalls
    content.push(...visibleText(parser, delta))
  }
  content.push(...visibleTail(parser))
  return {
    content: content.join(''),
    finishReason: resolvedFinishReason(finishReason, toolCalls),
    toolCalls,
    usage: toOpenAiUsage(lastUsage),
  }
}

function visibleText(parser: ThinkingParser, delta: LlmDelta): string[] {
  if (!delta.content) {
    return []
  }
  return parser.process(delta.content).flatMap(part => part.type === 'text' ? [part.text] : [])
}

function visibleTail(parser: ThinkingParser): string[] {
  return parser.process('').flatMap(part => part.type === 'text' ? [part.text] : [])
}

function resolvedFinishReason(reason: string | undefined, toolCalls: readonly ToolCall[]): string {
  if (reason !== undefined) {
    return normalizeOpenAiFinishReason(reason)
  }
  return toolCalls.length ? 'tool_calls' : 'stop'
}

/**
 * Map provider-specific finish vocabularies (Responses API statuses, Anthropic
 * stop reasons, Gemini finish reasons) onto the chat-completions enum. Mid-stream
 * failures already surface through the SSE error-frame path, so provider error
 * markers and unknown values degrade to `stop` here.
 */
function normalizeOpenAiFinishReason(reason: string): string {
  switch (reason.toLowerCase()) {
    case 'stop':
    case 'end_turn':
    case 'stop_sequence':
    case 'completed':
      return 'stop'
    case 'length':
    case 'max_tokens':
    case 'incomplete':
      return 'length'
    case 'tool_calls':
    case 'tool_use':
      return 'tool_calls'
    case 'content_filter':
    case 'safety':
    case 'recitation':
      return 'content_filter'
    default:
      return 'stop'
  }
}

function isCortexModel(model: string): boolean {
  return model.toLowerCase().includes('cortex')
}

function bearerAuthorizationMatches(authorization: string, token: string): boolean {
  const actualBytes = Buffer.from(authorization)
  const expectedBytes = Buffer.from(`Bearer ${token}`)
  return actualBytes.byteLength === expectedBytes.byteLength && timingSafeEqual(actualBytes, expectedBytes)
}

function isLoopbackHostname(hostname: string): boolean {
  const normalized = hostname.trim().toLowerCase()
  return normalized === '127.0.0.1' || normalized === '::1' || normalized === '[::1]' || normalized === 'localhost'
}

function cortexMetadata(value: unknown): CortexCompletionMetadata | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (!isObjectRecord(value)) {
    throw new ApiRequestError('metadata must be an object.', 'metadata')
  }
  const background = optionalCortexString(value.background, 'metadata.background')
  const processType = optionalCortexString(value.process_type, 'metadata.process_type')
  const taskMode = optionalCortexBoolean(value.task_mode, 'metadata.task_mode')
  const metadata: CortexCompletionMetadata = {
    ...(background === undefined ? {} : { background }),
    ...(processType === undefined ? {} : { process_type: processType }),
    ...(taskMode === undefined ? {} : { task_mode: taskMode }),
  }
  return Object.keys(metadata).length ? metadata : undefined
}

function optionalCortexString(value: unknown, parameter: string): string | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (typeof value !== 'string') {
    throw new ApiRequestError(`${parameter} must be a string.`, parameter)
  }
  return value
}

function optionalCortexBoolean(value: unknown, parameter: string): boolean | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (typeof value !== 'boolean') {
    throw new ApiRequestError(`${parameter} must be a boolean.`, parameter)
  }
  return value
}

function isObjectRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function rateLimitHeaders(maxRequests: number, remaining: number, oldestRequest: number, windowMs: number): Headers {
  return new Headers({
    'X-RateLimit-Limit': String(maxRequests),
    'X-RateLimit-Remaining': String(remaining),
    'X-RateLimit-Reset': String(Math.ceil((oldestRequest + windowMs) / 1_000)),
  })
}

function splitHeaderValues(value: string | null): string[] {
  if (!value) {
    return []
  }
  return value
    .split(',')
    .map(item => item.trim().toLowerCase())
    .filter(Boolean)
}

function appendVary(headers: Headers, value: string): void {
  const existing = splitHeaderValues(headers.get('Vary'))
  if (existing.includes(value.toLowerCase())) {
    return
  }
  headers.set('Vary', [...existing, value].join(', '))
}

function withHeaders(response: Response, headers: Headers): Response {
  headers.forEach((value, name) => {
    if (name.toLowerCase() === 'vary') {
      appendVary(response.headers, value)
    } else {
      response.headers.set(name, value)
    }
  })
  return response
}

function json(value: unknown, init: ResponseInit = {}): Response {
  return Response.json(value, init)
}

function methodNotAllowed(allow: string): Response {
  return apiError(405, 'Method not allowed.', 'invalid_request_error', null, 'method_not_allowed', { Allow: allow })
}

function apiError(
  status: number,
  message: string,
  type: string,
  parameter: string | null,
  code: string | null,
  headers: HeadersInit | undefined = undefined,
): Response {
  return json({ error: openAiError(message, type, parameter, code) }, { status, ...(headers ? { headers } : {}) })
}

function openAiError(message: string, type: string, parameter: string | null, code: string | null) {
  return { message, type, param: parameter, code }
}
