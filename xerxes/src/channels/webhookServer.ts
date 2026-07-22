// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { timingSafeEqual } from 'node:crypto'

import type { WebhookCapableChannel, WebhookHeaders } from './webhooks.js'
import type { ChannelManager } from './manager.js'

const DEFAULT_MAX_BODY_BYTES = 1_048_576

export interface ChannelWebhookServerOptions {
  /**
   * Bearer token required for the channel list endpoint. When omitted the
   * list endpoint is unauthenticated and safe only because the default bind
   * is loopback (127.0.0.1); configure a token before binding publicly.
   */
  readonly authToken?: string
  readonly host?: string
  readonly manager: ChannelManager
  readonly maxBodyBytes?: number
  readonly pathPrefix?: string
  readonly port?: number
}

/**
 * Bun-native HTTP receiver for configured webhook-capable channel adapters.
 *
 * It intentionally exposes only a list endpoint and POST delivery to known
 * host-configured channel names. Provider signature checks remain the
 * responsibility of each channel adapter. The list endpoint enumerates
 * configured channels, so it requires `authToken` whenever one is configured;
 * without a token it relies on the loopback default bind.
 */
export class ChannelWebhookServer {
  readonly authToken: string
  readonly host: string
  readonly manager: ChannelManager
  readonly maxBodyBytes: number
  readonly pathPrefix: string
  readonly port: number
  private server: Bun.Server<undefined> | undefined

  constructor(options: ChannelWebhookServerOptions) {
    this.manager = options.manager
    this.authToken = options.authToken?.trim() ?? ''
    this.host = nonBlank(options.host) ?? '127.0.0.1'
    this.port = portValue(options.port ?? 0)
    this.maxBodyBytes = bodyLimit(options.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES)
    this.pathPrefix = normalizedPrefix(options.pathPrefix ?? '/channels')
  }

  get url(): URL | undefined {
    return this.server?.url
  }

  get listeningPort(): number | undefined {
    return this.server?.port
  }

  start(): void {
    if (this.server) return
    this.server = Bun.serve({
      hostname: this.host,
      port: this.port,
      fetch: request => this.handle(request),
    })
  }

  async stop(): Promise<void> {
    const server = this.server
    if (!server) return
    this.server = undefined
    await Promise.resolve(server.stop(true))
  }

  async handle(request: Request): Promise<Response> {
    const url = new URL(request.url)
    if (url.pathname === this.pathPrefix) {
      if (request.method !== 'GET') return methodNotAllowed('GET')
      if (this.authToken && !bearerMatches(request.headers.get('authorization'), this.authToken)) {
        return new Response('unauthorized', { status: 401 })
      }
      return json({ ok: true, channels: this.manager.list() })
    }
    const name = webhookName(url.pathname, this.pathPrefix)
    if (!name) return new Response('Not Found', { status: 404 })
    if (request.method !== 'POST') return methodNotAllowed('POST')
    const channel = this.manager.registry.get(name)
    if (!channel) return new Response('unknown channel ' + JSON.stringify(name), { status: 404 })
    if (!isWebhookCapable(channel)) {
      return new Response('channel does not support webhooks', { status: 400 })
    }
    let body: Uint8Array
    try {
      body = await readBody(request, Math.min(this.maxBodyBytes, channelBodyLimit(channel)))
    } catch (error) {
      if (error instanceof WebhookBodyTooLargeError) return new Response('request body too large', { status: 413 })
      return new Response('invalid request body', { status: 400 })
    }
    const response = await channel.handleWebhook(headersFrom(request.headers), body)
    return new Response(response.body, {
      status: response.status,
      ...(response.headers === undefined ? {} : { headers: response.headers }),
    })
  }
}

class WebhookBodyTooLargeError extends Error {}

async function readBody(request: Request, limit: number): Promise<Uint8Array> {
  const declared = request.headers.get('content-length')
  if (declared && (!/^\d+$/.test(declared) || Number(declared) > limit)) {
    throw new WebhookBodyTooLargeError()
  }
  if (!request.body) return new Uint8Array()
  const reader = request.body.getReader()
  const chunks: Uint8Array[] = []
  let size = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      size += value.byteLength
      if (size > limit) throw new WebhookBodyTooLargeError()
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

function isWebhookCapable(value: unknown): value is WebhookCapableChannel {
  return typeof value === 'object' && value !== null && 'handleWebhook' in value
    && typeof value.handleWebhook === 'function'
}

function channelBodyLimit(channel: WebhookCapableChannel): number {
  if (!('maxWebhookBodyBytes' in channel)) return Number.MAX_SAFE_INTEGER
  const value = channel.maxWebhookBodyBytes
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0
    ? value
    : Number.MAX_SAFE_INTEGER
}

function headersFrom(headers: Headers): WebhookHeaders {
  return Object.fromEntries(headers.entries())
}

function webhookName(pathname: string, prefix: string): string | undefined {
  const expectedSuffix = '/webhook'
  if (!pathname.startsWith(prefix + '/') || !pathname.endsWith(expectedSuffix)) return undefined
  const name = pathname.slice(prefix.length + 1, -expectedSuffix.length)
  if (!name || name.includes('/')) return undefined
  try {
    return decodeURIComponent(name)
  } catch {
    return undefined
  }
}

function bearerMatches(headerValue: string | null, expected: string): boolean {
  const match = /^Bearer\s+(\S+)\s*$/.exec(headerValue ?? '')
  if (!match) return false
  const supplied = Buffer.from(match[1] ?? '', 'utf8')
  const wanted = Buffer.from(expected, 'utf8')
  return supplied.byteLength === wanted.byteLength && timingSafeEqual(supplied, wanted)
}

function methodNotAllowed(allowed: string): Response {
  return new Response('Method Not Allowed', { status: 405, headers: { Allow: allowed } })
}

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  })
}

function nonBlank(value: string | undefined): string | undefined {
  const normalized = value?.trim()
  return normalized || undefined
}

function normalizedPrefix(value: string): string {
  const normalized = value.trim().replace(/\/+$/, '') || '/channels'
  if (!normalized.startsWith('/')) throw new TypeError('pathPrefix must start with a slash')
  return normalized
}

function portValue(value: number): number {
  if (!Number.isInteger(value) || value < 0 || value > 65_535) {
    throw new RangeError('port must be an integer between 0 and 65535')
  }
  return value
}

function bodyLimit(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError('maxBodyBytes must be a positive safe integer')
  return value
}
