// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Shared OAuth flow primitives for the subscription-backed provider sessions
 * (anthropic, kimi-code, openrouter, xai, radius), mirroring pi-ai's
 * `auth/oauth/{pkce,device-code,oauth-page}.ts`.
 *
 * Conventions differ from pi-ai in one place, deliberately: credentials carry
 * `expires` as seconds since the epoch (pi-ai stores skew-adjusted
 * milliseconds), matching the Copilot/Codex session convention in this
 * repository. Refresh skew is applied at resolve time instead.
 */

/** Canonical credential shape shared by every OAuth flow session. */
export interface OAuthFlowCredential {
  /** Access token sent to the provider. */
  access: string
  /** Long-lived refresh token; empty when the provider issues none (OpenRouter). */
  refresh: string
  /** Access-token expiry in epoch seconds. */
  expires: number
  /** OAuth scope the server granted, when it reports one. */
  scope?: string
}

/** Generate a PKCE S256 pair (RFC 7636) via Web Crypto. */
export async function generatePkceS256(): Promise<{ verifier: string; challenge: string }> {
  const verifierBytes = new Uint8Array(32)
  crypto.getRandomValues(verifierBytes)
  const verifier = base64UrlEncode(verifierBytes)
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier))
  return { verifier, challenge: base64UrlEncode(new Uint8Array(digest)) }
}

function base64UrlEncode(bytes: Uint8Array): string {
  let binary = ''
  for (const byte of bytes) binary += String.fromCharCode(byte)
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '')
}

const CANCEL_MESSAGE = 'Login cancelled'
const TIMEOUT_MESSAGE = 'Device flow timed out'
const SLOW_DOWN_TIMEOUT_MESSAGE =
  'Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again.'
const MINIMUM_INTERVAL_MS = 1_000
/** RFC 8628 §3.2: a missing `interval` means the client polls every 5 seconds. */
const DEFAULT_POLL_INTERVAL_SECONDS = 5
/** RFC 8628 §3.5: `slow_down` grows the polling interval by 5 seconds. */
const SLOW_DOWN_INTERVAL_INCREMENT_MS = 5_000

export type DeviceCodePollResult<T> =
  | { status: 'pending' }
  | { status: 'slow_down'; intervalSeconds?: number }
  | { status: 'failed'; message: string }
  | { status: 'complete'; value: T }

export interface DeviceCodePollOptions<T> {
  readonly intervalSeconds?: number
  readonly expiresInSeconds?: number
  readonly waitBeforeFirstPoll?: boolean
  readonly poll: () => Promise<DeviceCodePollResult<T>>
  readonly signal?: AbortSignal
  readonly sleep?: (ms: number) => Promise<void>
  readonly now?: () => number
}

/** Abortable sleep that rejects with the cancel message when the signal fires. */
async function abortableSleep(
  ms: number,
  signal: AbortSignal | undefined,
  sleep: (ms: number) => Promise<void>,
): Promise<void> {
  if (signal?.aborted) throw new Error(CANCEL_MESSAGE)
  if (!signal) {
    await sleep(ms)
    return
  }
  let onAbort: () => void | undefined
  const cancelled = new Promise<never>((_resolve, reject) => {
    onAbort = () => reject(new Error(CANCEL_MESSAGE))
    signal.addEventListener('abort', onAbort, { once: true })
  })
  try {
    await Promise.race([sleep(ms), cancelled])
  } finally {
    // `onAbort` is assigned synchronously above when a signal is present.
    if (onAbort!) signal.removeEventListener('abort', onAbort)
  }
}

/**
 * RFC 8628 device-code polling loop (pi-ai `pollOAuthDeviceCodeFlow`):
 * honours server-provided intervals, grows by 5s per slow_down, and times out
 * at the device code's expiry.
 */
export async function pollDeviceCodeFlow<T>(options: DeviceCodePollOptions<T>): Promise<T> {
  const now = options.now ?? (() => Date.now() / 1_000)
  const sleep = options.sleep ?? (ms => new Promise<void>(resolve => setTimeout(resolve, ms)))
  const deadline = options.expiresInSeconds !== undefined ? now() + options.expiresInSeconds : Number.POSITIVE_INFINITY
  let intervalMs = Math.max(MINIMUM_INTERVAL_MS, Math.floor((options.intervalSeconds ?? DEFAULT_POLL_INTERVAL_SECONDS) * 1_000))
  let slowDownResponses = 0
  if (options.waitBeforeFirstPoll && deadline - now() > 0) {
    await abortableSleep(Math.min(intervalMs, (deadline - now()) * 1_000), options.signal, sleep)
  }
  while (now() < deadline) {
    if (options.signal?.aborted) throw new Error(CANCEL_MESSAGE)
    const result = await options.poll()
    if (result.status === 'complete') return result.value
    if (result.status === 'failed') throw new Error(result.message)
    if (result.status === 'slow_down') {
      slowDownResponses += 1
      // A server-provided interval is the new minimum (GitHub behaviour);
      // only falling back to a client-tracked increment risks polling early
      // forever under WSL/VM clock drift.
      intervalMs = result.intervalSeconds !== undefined && Number.isFinite(result.intervalSeconds) && result.intervalSeconds > 0
        ? Math.max(MINIMUM_INTERVAL_MS, Math.floor(result.intervalSeconds * 1_000))
        : Math.max(MINIMUM_INTERVAL_MS, intervalMs + SLOW_DOWN_INTERVAL_INCREMENT_MS)
    }
    const remainingMs = (deadline - now()) * 1_000
    if (remainingMs <= 0) break
    await abortableSleep(Math.min(intervalMs, remainingMs), options.signal, sleep)
  }
  throw new Error(slowDownResponses > 0 ? SLOW_DOWN_TIMEOUT_MESSAGE : TIMEOUT_MESSAGE)
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
}

/** Minimal loopback landing page shown after a browser callback. */
export function oauthCallbackHtml(outcome: 'success' | 'error', message: string, details?: string): string {
  const heading = outcome === 'success' ? 'Signed in to Xerxes' : 'Sign-in failed'
  return `<!doctype html><meta charset="utf-8"><title>${escapeHtml(heading)}</title>`
    + `<body style="font:16px system-ui;padding:3rem;text-align:center"><h1>${escapeHtml(heading)}</h1>`
    + `<p>${escapeHtml(message)}</p>`
    + (details ? `<pre style="white-space:pre-wrap;word-break:break-word">${escapeHtml(details)}</pre>` : '')
    + '</body>'
}

/** Read a `Response` body as a JSON object, or `undefined` when it is not one. */
export async function readJsonObject(response: Response): Promise<Record<string, unknown> | undefined> {
  try {
    const parsed: unknown = await response.json()
    return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : undefined
  } catch {
    return undefined
  }
}

/** String field from a JSON object; empty and non-string values count as absent. */
export function stringField(record: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = record?.[key]
  return typeof value === 'string' && value ? value : undefined
}

/** Finite positive number field from a JSON object. */
export function positiveNumberField(record: Record<string, unknown> | undefined, key: string): number | undefined {
  const value = record?.[key]
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined
}
