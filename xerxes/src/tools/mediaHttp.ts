// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError, ProviderError, ValidationError } from '../core/errors.js'
import type { JsonObject, JsonValue } from '../types/toolCalls.js'

/** Injectable HTTP boundary for remote media providers. */
export type MediaFetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

/**
 * Hard response-body caps. A malicious or buggy media endpoint must not be
 * able to exhaust process memory through an unbounded `arrayBuffer()`/`text()`
 * read: binary (audio/image) bodies are capped at 64 MiB and JSON bodies at
 * 8 MiB, both far above legitimate provider payloads.
 */
export const MAX_BINARY_RESPONSE_BYTES = 64 * 1_024 * 1_024
export const MAX_JSON_RESPONSE_BYTES = 8 * 1_024 * 1_024

/**
 * Explicit credentials and transport settings for an HTTP media provider.
 *
 * An API key is required unless `allowUnauthenticated` is deliberately set
 * for a local gateway. No environment variables are consulted here.
 */
export interface HttpMediaClientOptions {
  readonly allowUnauthenticated?: boolean
  readonly apiKey?: string
  readonly authHeader?: string
  readonly authScheme?: string
  readonly baseUrl: string
  readonly fetchImplementation?: MediaFetch
  readonly headers?: Readonly<Record<string, string>>
  /** Override for the 64 MiB binary response-body cap; must be a positive integer. */
  readonly maxBinaryResponseBytes?: number
  /** Override for the 8 MiB JSON response-body cap; must be a positive integer. */
  readonly maxJsonResponseBytes?: number
  readonly providerName?: string
}

/** Minimal native-fetch client shared by image, speech, and vision adapters. */
export class HttpMediaClient {
  private readonly apiKey: string
  private readonly authHeader: string
  private readonly authScheme: string
  private readonly baseUrl: URL
  private readonly fetchImplementation: MediaFetch
  private readonly headers: Readonly<Record<string, string>>
  private readonly maxBinaryResponseBytes: number
  private readonly maxJsonResponseBytes: number
  readonly providerName: string

  constructor(options: HttpMediaClientOptions) {
    this.baseUrl = normalizedBaseUrl(options.baseUrl)
    this.apiKey = options.apiKey?.trim() ?? ''
    this.authHeader = nonEmptySetting(options.authHeader ?? 'Authorization', 'authHeader')
    this.authScheme = options.authScheme ?? 'Bearer'
    this.headers = Object.freeze({ ...(options.headers ?? {}) })
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.maxBinaryResponseBytes = positiveByteLimit(options.maxBinaryResponseBytes ?? MAX_BINARY_RESPONSE_BYTES, 'maxBinaryResponseBytes')
    this.maxJsonResponseBytes = positiveByteLimit(options.maxJsonResponseBytes ?? MAX_JSON_RESPONSE_BYTES, 'maxJsonResponseBytes')
    this.providerName = options.providerName?.trim() || 'media'

    if (!this.apiKey && !options.allowUnauthenticated) {
      throw new ConfigurationError(
        'media.apiKey',
        `${this.providerName} media requests require an explicit API key; `
          + 'pass allowUnauthenticated=true only for an intentionally unauthenticated local gateway.',
      )
    }
  }

  async postBinary(path: string, payload: JsonValue, signal?: AbortSignal): Promise<Uint8Array> {
    const response = await this.request(path, {
      body: JSON.stringify(payload),
      headers: { Accept: '*/*', 'Content-Type': 'application/json' },
      ...(signal === undefined ? {} : { signal }),
    })
    return readBoundedBody(response, this.maxBinaryResponseBytes, this.providerName, 'binary')
  }

  async postForm(path: string, form: FormData, signal?: AbortSignal): Promise<JsonValue> {
    const response = await this.request(path, { body: form, ...(signal === undefined ? {} : { signal }) })
    return parseJsonResponse(response, this.providerName, this.maxJsonResponseBytes)
  }

  async postJson(path: string, payload: JsonValue, signal?: AbortSignal): Promise<JsonValue> {
    const response = await this.request(path, {
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
      ...(signal === undefined ? {} : { signal }),
    })
    return parseJsonResponse(response, this.providerName, this.maxJsonResponseBytes)
  }

  private async request(
    path: string,
    options: { readonly body?: BodyInit; readonly headers?: Readonly<Record<string, string>>; readonly signal?: AbortSignal },
  ): Promise<Response> {
    const headers: Record<string, string> = {
      Accept: 'application/json',
      ...this.headers,
      ...options.headers,
    }
    if (this.apiKey) {
      headers[this.authHeader] = this.authScheme ? `${this.authScheme} ${this.apiKey}` : this.apiKey
    }

    // Path validation errors must surface as-is, not as provider reachability failures.
    const target = this.url(path)
    let response: Response
    try {
      response = await this.fetchImplementation(target, {
        method: 'POST',
        headers,
        ...(options.body === undefined ? {} : { body: options.body }),
        ...(options.signal === undefined ? {} : { signal: options.signal }),
      })
    } catch (error) {
      throw new ProviderError(this.providerName, 'media request could not reach the configured provider', error)
    }
    if (!response.ok) {
      // Provider response bodies can contain prompts, binary data, or secrets.
      throw new ProviderError(this.providerName, `media request failed with HTTP ${response.status}`, undefined, {
        status: response.status,
      })
    }
    return response
  }

  private url(path: string): URL {
    const normalized = path.replace(/^\/+/, '')
    if (!normalized) {
      throw new ValidationError('media.path', 'must be a non-empty provider-relative path', path)
    }
    // `new URL()` resolves dot segments, so `../admin` would silently escape
    // the configured API path prefix (https://host/v1/ + ../admin -> /admin).
    // Same-origin, so not SSRF, but it breaks path confinement; reject it.
    if (normalized.split('/').includes('..')) {
      throw new ValidationError('media.path', 'must not contain parent-directory (..) segments', path)
    }
    return new URL(normalized, this.baseUrl)
  }
}

/** Decode base64 supplied through a JSON tool call without accepting malformed data silently. */
export function decodeMediaBase64(value: string, field: string): Uint8Array {
  const normalized = value.replace(/\s/g, '')
  if (!normalized || !/^[A-Za-z0-9+/]*={0,2}$/.test(normalized) || normalized.length % 4 === 1) {
    throw new ValidationError(field, 'must be valid non-empty base64 data', value)
  }
  return new Uint8Array(Buffer.from(normalized, 'base64'))
}

export function encodeMediaBase64(value: Uint8Array): string {
  return Buffer.from(value).toString('base64')
}

export function requiredMediaType(value: string, prefix: 'audio/' | 'image/', field: string): string {
  const normalized = value.trim().toLowerCase()
  if (!normalized.startsWith(prefix) || !/^[a-z0-9.+-]+\/[a-z0-9.+-]+$/.test(normalized)) {
    throw new ValidationError(field, `must be a valid ${prefix} media type`, value)
  }
  return normalized
}

export function jsonArray(value: JsonValue, field: string): readonly JsonValue[] {
  if (!Array.isArray(value)) {
    throw new ProviderError('media', `provider response field ${field} must be an array`)
  }
  return value
}

export function jsonObject(value: JsonValue, field: string): JsonObject {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ProviderError('media', `provider response field ${field} must be an object`)
  }
  return value
}

export function jsonString(value: JsonValue | undefined): string | undefined {
  return typeof value === 'string' ? value : undefined
}

function nonEmptySetting(value: string, field: string): string {
  const normalized = value.trim()
  if (!normalized) {
    throw new ConfigurationError(`media.${field}`, 'must not be empty')
  }
  return normalized
}

function positiveByteLimit(value: number, field: string): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new ConfigurationError(`media.${field}`, 'must be a positive integer byte limit')
  }
  return value
}

function normalizedBaseUrl(value: string): URL {
  let url: URL
  try {
    url = new URL(value)
  } catch (error) {
    throw new ConfigurationError('media.baseUrl', 'must be an absolute HTTP(S) URL', { cause: error })
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new ConfigurationError('media.baseUrl', 'must use HTTP or HTTPS')
  }
  if (!url.pathname.endsWith('/')) {
    url.pathname += '/'
  }
  return url
}

async function parseJsonResponse(response: Response, providerName: string, maxBytes: number): Promise<JsonValue> {
  const bytes = await readBoundedBody(response, maxBytes, providerName, 'JSON')
  const body = new TextDecoder().decode(bytes)
  if (!body) {
    throw new ProviderError(providerName, 'media provider returned an empty JSON response')
  }
  try {
    return JSON.parse(body) as JsonValue
  } catch (error) {
    throw new ProviderError(providerName, 'media provider returned invalid JSON', error)
  }
}

/**
 * Stream a provider response body with a hard byte cap. `arrayBuffer()` and
 * `text()` buffer the full body without any limit, so an over-limit or
 * never-ending response is cancelled and rejected instead of exhausting
 * memory. A declared content-length over the cap fails before reading.
 */
async function readBoundedBody(
  response: Response,
  maxBytes: number,
  providerName: string,
  kind: string,
): Promise<Uint8Array> {
  const declared = Number(response.headers.get('content-length'))
  if (Number.isFinite(declared) && declared > maxBytes) {
    throw new ProviderError(providerName, `media provider ${kind} response exceeded the ${maxBytes}-byte limit`)
  }
  const stream = response.body
  if (stream === null) return new Uint8Array(0)
  const reader = stream.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  let overflow = false
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      total += value?.byteLength ?? 0
      if (total > maxBytes) {
        overflow = true
        break
      }
      if (value) chunks.push(value)
    }
  } finally {
    if (overflow) await reader.cancel().catch(() => undefined)
    reader.releaseLock()
  }
  if (overflow) {
    throw new ProviderError(providerName, `media provider ${kind} response exceeded the ${maxBytes}-byte limit`)
  }
  const body = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    body.set(chunk, offset)
    offset += chunk.byteLength
  }
  return body
}
