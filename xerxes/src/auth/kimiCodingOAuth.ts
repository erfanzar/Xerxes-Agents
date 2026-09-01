// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { type OAuthFetch } from '../mcp/oauth.js'
// Licensed under the Apache License, Version 2.0.

/**
 * Kimi Code (subscription) OAuth, mirroring pi-ai `auth/oauth/kimi-coding.ts`:
 * RFC 8628 device authorization against auth.kimi.com with JSON responses.
 * The access token authenticates api.kimi.com/coding as a Bearer header.
 */

import { ConfigurationError, ProviderError } from '../core/errors.js'
import { xerxesHome as defaultXerxesHome } from '../daemon/paths.js'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import {
  pollDeviceCodeFlow,
  positiveNumberField,
  readJsonObject,
  stringField,
  type OAuthFlowCredential,
} from './oauthFlows.js'

export const KIMI_CODE_OAUTH_PROVIDER = 'kimi-code'

/** pi-ai registers this OAuth client for the Kimi Code subscription flow. */
export const KIMI_CODE_OAUTH_CLIENT_ID = '17e5f671-d194-4dfb-9706-5516cb48c098'
export const KIMI_CODE_OAUTH_HOST_DEFAULT = 'https://auth.kimi.com'
const DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
const DEFAULT_POLL_INTERVAL_SECONDS = 5
const REQUEST_TIMEOUT_MS = 30_000
const REFRESH_MAX_RETRIES = 3
/** Refresh this far ahead of expiry so a token cannot die mid-stream. */
export const KIMI_CODE_REFRESH_SKEW_SECONDS = 300

export interface KimiCodeDeviceAuthorization {
  readonly deviceCode: string
  readonly userCode: string
  readonly verificationUri: string
  readonly verificationUriComplete: string
  readonly intervalSeconds: number
  readonly expiresInSeconds: number
}

export interface KimiCodingOAuthSessionOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly fetchImplementation?: OAuthFetch
  /** Overrides `<xerxesHome>` for the credential file location. */
  readonly xerxesHome?: string
  /** Seconds since the epoch; injected so expiry logic is testable. */
  readonly now?: () => number
  /** Injectable delay so device-flow pacing is observable without real sleeps. */
  readonly sleep?: (ms: number) => Promise<void>
}

/** OAuth host override (pi-ai: KIMI_CODE_OAUTH_HOST, then KIMI_OAUTH_HOST). */
export function kimiCodeOauthHost(environment: Readonly<Record<string, string | undefined>>): string {
  const override = environment['KIMI_CODE_OAUTH_HOST']?.trim() || environment['KIMI_OAUTH_HOST']?.trim()
  return (override || KIMI_CODE_OAUTH_HOST_DEFAULT).replace(/\/+$/, '')
}

/** The verification URI opens in the user's browser; only http(s) URLs are trusted. */
function trustedHttpUrl(value: unknown): string | undefined {
  if (typeof value !== 'string' || !value) return undefined
  try {
    const url = new URL(value)
    if (url.protocol !== 'https:' && url.protocol !== 'http:') return undefined
    return url.href
  } catch {
    return undefined
  }
}

function requestDeadline(timeoutMs: number, signal?: AbortSignal): {
  readonly dispose: () => void
  readonly signal: AbortSignal
} {
  const controller = new AbortController()
  const timer = setTimeout(
    () => controller.abort(new Error(`Kimi Code request timed out after ${timeoutMs}ms`)),
    timeoutMs,
  )
  const dispose = (): void => clearTimeout(timer)
  if (!signal) return { dispose, signal: controller.signal }
  if (signal.aborted) controller.abort(signal.reason)
  else signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true })
  return { dispose, signal: controller.signal }
}

/** Validate a token payload into the canonical credential (pi-ai parity). */
export function kimiCodeTokenFromResponse(
  payload: Record<string, unknown> | undefined,
  operation: string,
  nowSeconds: number,
): OAuthFlowCredential {
  const accessToken = stringField(payload, 'access_token')
  const refreshToken = stringField(payload, 'refresh_token')
  const expiresIn = positiveNumberField(payload, 'expires_in')
  if (!accessToken || !refreshToken || expiresIn === undefined) {
    throw new ProviderError(
      KIMI_CODE_OAUTH_PROVIDER,
      `Kimi Code token ${operation} response missing fields: ${JSON.stringify(payload ?? null)}`,
    )
  }
  // pi-ai stores the raw expiry; Xerxes applies refresh skew at resolve time.
  return { access: accessToken, refresh: refreshToken, expires: Math.floor(nowSeconds + expiresIn) }
}

/** Start the device authorization grant (pi-ai `startDeviceAuthorization`). */
export async function startKimiCodeDeviceAuthorization(
  oauthHost: string,
  options: {
    readonly fetchImplementation?: OAuthFetch
    readonly signal?: AbortSignal
  } = {},
): Promise<KimiCodeDeviceAuthorization> {
  const request = options.fetchImplementation ?? fetch
  const deadline = requestDeadline(REQUEST_TIMEOUT_MS, options.signal)
  let response: Response
  try {
    response = await request(`${oauthHost}/api/oauth/device_authorization`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        Accept: 'application/json',
      },
      body: new URLSearchParams({ client_id: KIMI_CODE_OAUTH_CLIENT_ID }).toString(),
      signal: deadline.signal,
    })
  } finally {
    deadline.dispose()
  }
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new ProviderError(
      KIMI_CODE_OAUTH_PROVIDER,
      `Kimi Code device authorization failed with status ${response.status}${text ? `: ${text.slice(0, 512)}` : ''}`,
    )
  }
  const json = await readJsonObject(response)
  const deviceCode = stringField(json, 'device_code')
  const userCode = stringField(json, 'user_code')
  const verificationUri = trustedHttpUrl(json && 'verification_uri' in json ? json['verification_uri'] : undefined)
  const verificationUriComplete = trustedHttpUrl(
    json && 'verification_uri_complete' in json ? json['verification_uri_complete'] : undefined,
  )
  if (!deviceCode || !userCode || !verificationUri || !verificationUriComplete) {
    throw new ProviderError(
      KIMI_CODE_OAUTH_PROVIDER,
      `Invalid Kimi Code device authorization response: ${JSON.stringify(json ?? null)}`,
    )
  }
  return {
    deviceCode,
    userCode,
    verificationUri,
    verificationUriComplete,
    intervalSeconds: positiveNumberField(json, 'interval') ?? DEFAULT_POLL_INTERVAL_SECONDS,
    expiresInSeconds: positiveNumberField(json, 'expires_in') ?? DEVICE_CODE_TIMEOUT_SECONDS,
  }
}

/** Single device-token poll, classified for the shared poller. */
async function pollKimiCodeTokenOnce(
  oauthHost: string,
  device: KimiCodeDeviceAuthorization,
  options: {
    readonly fetchImplementation?: OAuthFetch
    readonly signal?: AbortSignal
    readonly now?: () => number
  },
): Promise<
  { status: 'pending' }
  | { status: 'slow_down'; intervalSeconds?: number }
  | { status: 'failed'; message: string }
  | { status: 'complete'; value: OAuthFlowCredential }
> {
  const request = options.fetchImplementation ?? fetch
  const deadline = requestDeadline(REQUEST_TIMEOUT_MS, options.signal)
  let response: Response
  try {
    response = await request(`${oauthHost}/api/oauth/token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        Accept: 'application/json',
      },
      body: new URLSearchParams({
        client_id: KIMI_CODE_OAUTH_CLIENT_ID,
        device_code: device.deviceCode,
        grant_type: 'urn:ietf:params:oauth:grant-type:device_code',
      }).toString(),
      signal: deadline.signal,
    })
  } finally {
    deadline.dispose()
  }

  if (response.status >= 500) {
    const text = await response.text().catch(() => '')
    return {
      status: 'failed',
      message: `Kimi Code device token request failed with status ${response.status}${text ? `: ${text}` : ''}`,
    }
  }

  const json = await readJsonObject(response)
  if (response.ok && stringField(json, 'access_token')) {
    return {
      status: 'complete',
      value: kimiCodeTokenFromResponse(json, "poll", (options.now ?? (() => Date.now() / 1_000))()),
    }
  }

  const error = json?.['error']
  const description = typeof json?.['error_description'] === 'string' ? `: ${json['error_description']}` : ''
  if (error === 'authorization_pending') return { status: 'pending' }
  if (error === 'slow_down') {
    const interval = positiveNumberField(json, 'interval')
    return { status: 'slow_down', ...(interval === undefined ? {} : { intervalSeconds: interval }) }
  }
  if (error === 'expired_token') {
    return { status: 'failed', message: 'Kimi Code device authorization expired. Please restart login.' }
  }
  if (error === 'access_denied') {
    return { status: 'failed', message: 'Kimi Code login was denied.' }
  }
  return {
    status: 'failed',
    message: `Kimi Code device token request failed (status ${response.status})${typeof error === 'string' ? `: ${error}${description}` : ''}`,
  }
}

/** Refresh with pi-ai's retry ladder: 401/403/invalid_grant are fatal, 429/5xx back off. */
export async function refreshKimiCodeToken(
  oauthHost: string,
  refreshToken: string,
  options: {
    readonly fetchImplementation?: OAuthFetch
    readonly signal?: AbortSignal
    readonly sleep?: (ms: number) => Promise<void>
    readonly now?: () => number
  } = {},
): Promise<OAuthFlowCredential> {
  const request = options.fetchImplementation ?? fetch
  const sleep = options.sleep ?? (ms => new Promise<void>(resolve => setTimeout(resolve, ms)))
  let lastError: Error | undefined
  for (let attempt = 0; attempt <= REFRESH_MAX_RETRIES; attempt++) {
    if (attempt > 0) await sleep(1_000 * 2 ** (attempt - 1))
    if (options.signal?.aborted) throw new Error('Kimi Code token refresh aborted')

    const deadline = requestDeadline(REQUEST_TIMEOUT_MS, options.signal)
    let response: Response
    try {
      response = await request(`${oauthHost}/api/oauth/token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          Accept: 'application/json',
        },
        body: new URLSearchParams({
          client_id: KIMI_CODE_OAUTH_CLIENT_ID,
          grant_type: 'refresh_token',
          refresh_token: refreshToken,
        }).toString(),
        signal: deadline.signal,
      })
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error))
      continue
    } finally {
      deadline.dispose()
    }

    const json = await readJsonObject(response)
    if (response.ok) {
      return kimiCodeTokenFromResponse(json, "refresh", (options.now ?? (() => Date.now() / 1_000))())
    }
    // Unauthorized: the stored credential is dead; re-login is required.
    if (response.status === 401 || response.status === 403 || json?.['error'] === 'invalid_grant') {
      const description = typeof json?.['error_description'] === 'string' ? `: ${json['error_description']}` : ''
      throw new ProviderError(
        KIMI_CODE_OAUTH_PROVIDER,
        `Kimi Code token refresh unauthorized (status ${response.status})${description}`,
      )
    }
    if ((response.status === 429 || response.status >= 500) && attempt < REFRESH_MAX_RETRIES) {
      lastError = new Error(`Kimi Code token refresh failed with status ${response.status}`)
      continue
    }
    throw new ProviderError(
      KIMI_CODE_OAUTH_PROVIDER,
      `Kimi Code token refresh failed with status ${response.status}${json ? `: ${JSON.stringify(json).slice(0, 512)}` : ''}`,
    )
  }
  throw lastError ?? new ProviderError(KIMI_CODE_OAUTH_PROVIDER, 'Kimi Code token refresh failed')
}

/**
 * Owns the stored Kimi Code OAuth credential: single-flight resolution,
 * device-flow login, refresh with backoff, persistence under `<xerxesHome>/auth/`.
 */
export class KimiCodingOAuthSession {
  private readonly environment: Readonly<Record<string, string | undefined>>
  private readonly fetchImplementation: OAuthFetch | undefined
  private readonly home: string
  private readonly now: () => number
  private readonly sleep: (ms: number) => Promise<void>
  private pending: Promise<OAuthFlowCredential> | undefined

  constructor(options: KimiCodingOAuthSessionOptions = {}) {
    this.environment = options.environment ?? process.env
    this.fetchImplementation = options.fetchImplementation
    this.home = options.xerxesHome ?? defaultXerxesHome(this.environment)
    this.now = options.now ?? (() => Date.now() / 1_000)
    this.sleep = options.sleep ?? (ms => new Promise<void>(resolve => setTimeout(resolve, ms)))
  }

  /** Return a usable credential, refreshing when at or near expiry. */
  async credential(signal?: AbortSignal): Promise<OAuthFlowCredential> {
    if (this.pending) return this.pending
    const flight = this.resolve(signal)
    const tracked = flight.finally(() => {
      if (this.pending === tracked) this.pending = undefined
    })
    this.pending = tracked
    return tracked
  }

  /** Re-mint the access token from the stored refresh token. */
  async refresh(credential: OAuthFlowCredential, signal?: AbortSignal): Promise<OAuthFlowCredential> {
    if (!credential.refresh) {
      throw new ConfigurationError(
        'kimi_code_oauth',
        "Kimi Code credential carries no refresh token. Run 'xerxes auth login kimi'.",
      )
    }
    const fresh = await refreshKimiCodeToken(kimiCodeOauthHost(this.environment), credential.refresh, {
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
      ...(signal ? { signal } : {}),
      sleep: this.sleep,
      now: this.now,
    })
    await this.persist(fresh)
    return fresh
  }

  /**
   * Run the RFC 8628 device flow: requests a device code, hands the user
   * code to `onUserCode`, then polls until Kimi issues the token or the
   * device code expires.
   */
  async login(
    onUserCode: (userCode: string, verificationUri: string) => void,
    signal?: AbortSignal,
  ): Promise<OAuthFlowCredential> {
    const oauthHost = kimiCodeOauthHost(this.environment)
    const device = await startKimiCodeDeviceAuthorization(oauthHost, {
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
      ...(signal ? { signal } : {}),
    })
    onUserCode(device.userCode, device.verificationUriComplete)
    const token = await pollDeviceCodeFlow({
      intervalSeconds: device.intervalSeconds,
      expiresInSeconds: device.expiresInSeconds,
      waitBeforeFirstPoll: true,
      ...(signal ? { signal } : {}),
      sleep: this.sleep,
      now: this.now,
      poll: () => pollKimiCodeTokenOnce(oauthHost, device, {
        ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
        ...(signal ? { signal } : {}),
        now: this.now,
      }),
    })
    await this.persist(token)
    return token
  }

  /** The persisted credential without refreshing, for `auth status`. */
  async stored(): Promise<OAuthFlowCredential | undefined> {
    return this.loadStored()
  }

  /** Remove the stored credential. Returns whether one existed. */
  async logout(): Promise<boolean> {
    try {
      await rm(this.credentialPath())
      return true
    } catch {
      return false
    }
  }

  /** Where the credential persists, for status/logout surfaces. */
  storedPath(): string {
    return this.credentialPath()
  }

  private isExpired(expires: number): boolean {
    return expires - KIMI_CODE_REFRESH_SKEW_SECONDS <= this.now()
  }

  private async resolve(signal?: AbortSignal): Promise<OAuthFlowCredential> {
    const stored = await this.loadStored()
    if (stored && !this.isExpired(stored.expires)) return stored
    if (stored) return this.refresh(stored, signal)
    throw new ConfigurationError(
      'kimi_code_oauth',
      "No Kimi Code subscription session found. Run 'xerxes auth login kimi'.",
    )
  }

  private credentialPath(): string {
    return join(this.home, 'auth', 'kimi-code-oauth.json')
  }

  private async loadStored(): Promise<OAuthFlowCredential | undefined> {
    let raw: string
    try {
      raw = await readFile(this.credentialPath(), 'utf8')
    } catch {
      return undefined
    }
    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch {
      return undefined
    }
    const record = parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : undefined
    const access = stringField(record, 'access')
    const refresh = stringField(record, 'refresh')
    const expires = record && typeof record['expires'] === 'number' ? record['expires'] : undefined
    if (!access || !refresh || expires === undefined) return undefined
    const scope = stringField(record, 'scope')
    return { access, refresh, expires, ...(scope ? { scope } : {}) }
  }

  private async persist(credential: OAuthFlowCredential): Promise<void> {
    const path = this.credentialPath()
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, `${JSON.stringify(credential, null, 2)}\n`, 'utf8')
  }
}
