// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError, ProviderError, ValidationError } from '../core/errors.js'
import type { JsonObject, JsonValue } from '../types/toolCalls.js'

/** Injectable HTTP boundary for remote media providers. */
export type MediaFetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

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
  readonly providerName: string

  constructor(options: HttpMediaClientOptions) {
    this.baseUrl = normalizedBaseUrl(options.baseUrl)
    this.apiKey = options.apiKey?.trim() ?? ''
    this.authHeader = nonEmptySetting(options.authHeader ?? 'Authorization', 'authHeader')
    this.authScheme = options.authScheme ?? 'Bearer'
    this.headers = Object.freeze({ ...(options.headers ?? {}) })
    this.fetchImplementation = options.fetchImplementation ?? fetch
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
    return new Uint8Array(await response.arrayBuffer())
  }

  async postForm(path: string, form: FormData, signal?: AbortSignal): Promise<JsonValue> {
    const response = await this.request(path, { body: form, ...(signal === undefined ? {} : { signal }) })
    return parseJsonResponse(response, this.providerName)
  }

  async postJson(path: string, payload: JsonValue, signal?: AbortSignal): Promise<JsonValue> {
    const response = await this.request(path, {
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
      ...(signal === undefined ? {} : { signal }),
    })
    return parseJsonResponse(response, this.providerName)
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

    let response: Response
    try {
      response = await this.fetchImplementation(this.url(path), {
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

async function parseJsonResponse(response: Response, providerName: string): Promise<JsonValue> {
  const body = await response.text()
  if (!body) {
    throw new ProviderError(providerName, 'media provider returned an empty JSON response')
  }
  try {
    return JSON.parse(body) as JsonValue
  } catch (error) {
    throw new ProviderError(providerName, 'media provider returned invalid JSON', error)
  }
}
