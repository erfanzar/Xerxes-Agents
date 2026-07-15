// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ChannelMessage } from './types.js'

/** A callback that supplies a fresh access token for an HTTP API call. */
export type ChannelAccessTokenProvider = () => string | Promise<string>

/**
 * Transport support intentionally exposed by adapters that rely on an HTTP
 * callback/API pair instead of owning a provider's persistent connection.
 */
export interface RelayChannelTransport {
  /** Inbound delivery must be relayed into `handleWebhook` by the host. */
  readonly inbound: 'webhook-relay'
  /** The endpoint used for outbound delivery. */
  readonly outbound: 'http-api' | 'incoming-webhook'
  /** Provider transports deliberately outside this dependency-free adapter. */
  readonly unsupported: readonly string[]
}

/**
 * Features deliberately not emulated by the Bun runtime today.
 *
 * Email has a real normalization adapter and a native SMTP sender. Direct
 * IMAP polling remains host-owned, so applications inject an IMAP transport
 * or relay inbound mail through the webhook endpoint.
 */
export const UNSUPPORTED_CHANNEL_TRANSPORTS = {
  email_imap: {
    inbound: 'host-transport-required',
    outbound: 'bun-native-smtp-or-host-transport',
    reason: 'Bun includes direct SMTP delivery; inject an IMAP transport or use a mail-to-webhook relay for inbound mail.',
  },
} as const

/** Convert a JSON value to an object without trusting arbitrary arrays. */
export function recordValue(value: unknown): Readonly<Record<string, unknown>> {
  return isRecord(value) ? value : {}
}

/** Convert scalar JSON IDs and text values into normalized strings. */
export function stringValue(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  if (typeof value === 'number' || typeof value === 'bigint') {
    return String(value)
  }
  return ''
}

/** Return an array only when a provider field is actually an array. */
export function arrayValue(value: unknown): readonly unknown[] {
  return Array.isArray(value) ? value : []
}

/** Resolve a configured static token or a per-send refresh callback. */
export async function resolveAccessToken(
  staticToken: string,
  provider: ChannelAccessTokenProvider | undefined,
  providerName: string,
): Promise<string> {
  const supplied = provider ? await provider() : ''
  const token = supplied || staticToken
  if (!token) {
    throw new Error(`${providerName} access token unavailable`)
  }
  return token
}

/** Reject a provider's documented zero-is-success API envelope without logging it. */
export function throwOnProviderErrorCode(
  response: unknown,
  providerName: string,
  codeField: string,
): void {
  const code = recordValue(response)[codeField]
  if (code === undefined || code === null || code === 0 || code === '0') {
    return
  }
  throw new Error(`${providerName} API request failed`)
}

/** Require a usable outbound destination before issuing a provider request. */
export function outboundDestination(message: ChannelMessage, providerName: string): string {
  const destination = message.roomId ?? message.channelUserId
  if (!destination) {
    throw new TypeError(`${providerName} outbound messages require roomId or channelUserId`)
  }
  return destination
}

/** Add a query parameter without string concatenation or accidental escaping. */
export function urlWithQuery(url: string, name: string, value: string): string {
  const parsed = new URL(url)
  parsed.searchParams.set(name, value)
  return parsed.toString()
}

/** Reject configuration values that would only fail later in an outbound turn. */
export function requiredOption(value: string, name: string): string {
  if (!value.trim()) {
    throw new TypeError(`${name} must not be empty`)
  }
  return value
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
