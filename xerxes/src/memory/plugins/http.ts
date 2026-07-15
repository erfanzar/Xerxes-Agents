// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject, JsonValue } from '../../types/toolCalls.js'
import type { ExternalMemoryAction, ExternalMemoryUpstream } from './base.js'

/** A request emitted by an HTTP-backed external-memory adapter. */
export interface MemoryPluginHttpRequest {
  readonly body?: JsonObject
  readonly headers: Readonly<Record<string, string>>
  readonly method: 'DELETE' | 'GET' | 'POST'
  readonly url: string
}

/** The JSON-decoded response returned by an embedding-owned HTTP transport. */
export interface MemoryPluginHttpResponse {
  readonly body?: JsonValue
  readonly ok: boolean
  readonly status: number
}

/** Explicit network boundary. Adapters never select `fetch` or an SDK themselves. */
export interface MemoryPluginHttpTransport {
  request(request: MemoryPluginHttpRequest): Promise<MemoryPluginHttpResponse>
}

/** Build one outbound request for a particular standard memory action. */
export type MemoryPluginHttpRequestBuilder = (
  action: ExternalMemoryAction,
  arguments_: JsonObject,
) => MemoryPluginHttpRequest

/** Inputs for a reusable HTTP-to-memory-upstream adapter. */
export interface HttpMemoryUpstreamOptions {
  readonly providerName: string
  readonly requestFor: MemoryPluginHttpRequestBuilder
  readonly transport: MemoryPluginHttpTransport
}

/** Raised when a confirmed HTTP response rejects a configured memory request. */
export class MemoryPluginHttpError extends Error {
  readonly providerName: string
  readonly status: number

  constructor(providerName: string, status: number) {
    super(`${providerName} request failed with HTTP ${status}`)
    this.name = 'MemoryPluginHttpError'
    this.providerName = providerName
    this.status = status
  }
}

/**
 * Adapts a host-injected JSON HTTP transport to an external-memory upstream.
 *
 * A no-body successful delete becomes `{ removed: true }` only after the
 * transport confirms a successful HTTP response; no request is invented or
 * treated as successful before that confirmation.
 */
export class HttpMemoryUpstream implements ExternalMemoryUpstream {
  private readonly providerName: string
  private readonly requestFor: MemoryPluginHttpRequestBuilder
  private readonly transport: MemoryPluginHttpTransport

  constructor(options: HttpMemoryUpstreamOptions) {
    this.providerName = requireProviderName(options.providerName)
    this.requestFor = options.requestFor
    this.transport = options.transport
  }

  async call(action: ExternalMemoryAction, arguments_: JsonObject): Promise<JsonValue> {
    const response = await this.transport.request(this.requestFor(action, arguments_))
    if (!isHttpResponse(response)) throw new Error(`${this.providerName} transport returned an invalid response`)
    if (!response.ok) throw new MemoryPluginHttpError(this.providerName, response.status)
    if (response.body !== undefined) return response.body
    return action === 'remove' ? { removed: true } : {}
  }
}

/** Build a validated request URL from a provider base URL, a relative route, and query parameters. */
export function memoryPluginUrl(
  baseUrl: string,
  path: string,
  query: Readonly<Record<string, number | string>> = {},
): string {
  let base: URL
  try {
    base = new URL(baseUrl)
  } catch (error) {
    throw new Error(`invalid memory plugin base URL: ${baseUrl}`, { cause: error })
  }
  if (base.protocol !== 'http:' && base.protocol !== 'https:') {
    throw new Error(`memory plugin base URL must use HTTP or HTTPS: ${baseUrl}`)
  }
  base.hash = ''
  base.search = ''
  base.pathname = `${base.pathname.replace(/\/+$/, '')}/`
  const url = new URL(path.replace(/^\/+/, ''), base)
  for (const [name, value] of Object.entries(query)) url.searchParams.set(name, String(value))
  return url.toString()
}

/** Encode a backend entry identifier before placing it in an HTTP path segment. */
export function memoryPluginPathSegment(value: string): string {
  return encodeURIComponent(value)
}

function isHttpResponse(value: unknown): value is MemoryPluginHttpResponse {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const candidate = value as { readonly ok?: unknown; readonly status?: unknown }
  return typeof candidate.ok === 'boolean' && Number.isSafeInteger(candidate.status)
}

function requireProviderName(value: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error('memory plugin provider name must be a non-empty string')
  }
  return value.trim()
}
