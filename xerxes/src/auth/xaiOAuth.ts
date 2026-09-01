// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { type OAuthFetch } from '../mcp/oauth.js'
// Licensed under the Apache License, Version 2.0.

/**
 * xAI OAuth device-code flow, mirroring pi-ai `auth/oauth/xai.ts`: RFC 8628
 * against auth.x.ai for the Grok/X subscription surface. The access token is
 * used as the API key for the xAI OpenAI-compatible endpoint.
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

export const XAI_OAUTH_PROVIDER = 'xai'

/** pi-ai registers this OAuth client for the xAI subscription flow. */
export const XAI_OAUTH_CLIENT_ID = 'b1a00492-073a-47ea-816f-4c329264a828'
export const XAI_OAUTH_SCOPE = 'openid profile email offline_access grok-cli:access api:access'
export const XAI_DEVICE_CODE_URL = 'https://auth.x.ai/oauth2/device/code'
export const XAI_TOKEN_URL = 'https://auth.x.ai/oauth2/token'
/** Refresh this far ahead of expiry to avoid a token dying mid-request. */
export const XAI_REFRESH_SKEW_SECONDS = 300
const DEFAULT_TOKEN_LIFETIME_SECONDS = 3_600

export interface XaiDeviceCode {
  readonly deviceCode: string
  readonly userCode: string
  readonly verificationUri: string
  readonly verificationUriComplete?: string
  readonly intervalSeconds?: number
  readonly expiresInSeconds: number
}

export interface XaiOAuthSessionOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly fetchImplementation?: OAuthFetch
  /** Overrides `<xerxesHome>` for the credential file location. */
  readonly xerxesHome?: string
  /** Seconds since the epoch; injected so expiry logic is testable. */
  readonly now?: () => number
  /** Injectable delay so device-flow pacing is observable without real sleeps. */
  readonly sleep?: (ms: number) => Promise<void>
}

// The verification URI opens in the user's browser; force https so a malicious
// response cannot make the opener launch something else.
function validateVerificationUri(raw: unknown): string {
  if (typeof raw !== 'string' || !raw) {
    throw new ProviderError(XAI_OAUTH_PROVIDER, 'Invalid xAI OAuth response field: verification_uri')
  }
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    throw new ProviderError(XAI_OAUTH_PROVIDER, 'Untrusted verification URI in xAI OAuth response')
  }
  if (url.protocol !== 'https:') {
    throw new ProviderError(XAI_OAUTH_PROVIDER, 'Untrusted verification URI in xAI OAuth response')
  }
  return url.href
}

function requestDeadline(timeoutMs: number, signal?: AbortSignal): {
  readonly dispose: () => void
  readonly signal: AbortSignal
} {
  const controller = new AbortController()
  const timer = setTimeout(
    () => controller.abort(new Error(`xAI request timed out after ${timeoutMs}ms`)),
    timeoutMs,
  )
  const dispose = (): void => clearTimeout(timer)
  if (!signal) return { dispose, signal: controller.signal }
  if (signal.aborted) controller.abort(signal.reason)
  else signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true })
  return { dispose, signal: controller.signal }
}

async function postForm(
  url: string,
  fields: Record<string, string>,
  signal?: AbortSignal,
  context: { readonly fetchImplementation?: OAuthFetch } = {},
): Promise<{ ok: boolean; status: number; body: Record<string, unknown> }> {
  const deadline = requestDeadline(30_000, signal)
  let response: Response
  try {
    response = await (context.fetchImplementation ?? fetch)(url, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams(fields).toString(),
      signal: deadline.signal,
    })
  } catch (error) {
    if (signal?.aborted) throw new Error('Login cancelled')
    throw error
  } finally {
    deadline.dispose()
  }
  const body = await readJsonObject(response)
  return { ok: response.ok, status: response.status, body: body ?? {} }
}

function requestFailure(action: string, response: { status: number; body: Record<string, unknown> }): ProviderError {
  const error = stringField(response.body, 'error')
  const description = stringField(response.body, 'error_description')
  const detail = [error, description].filter(Boolean).join(': ')
  return new ProviderError(
    XAI_OAUTH_PROVIDER,
    `xAI OAuth ${action} failed (HTTP ${response.status})${detail ? `: ${detail}` : ''}`,
  )
}

/** Parse a device-authorization payload (pi-ai `parseDeviceCode`). */
export function xaiDeviceCodeFromResponse(body: Record<string, unknown>): XaiDeviceCode {
  const deviceCode = stringField(body, 'device_code')
  const userCode = stringField(body, 'user_code')
  const expiresInSeconds = positiveNumberField(body, 'expires_in')
  if (!deviceCode || !userCode || expiresInSeconds === undefined) {
    throw new ProviderError(XAI_OAUTH_PROVIDER, 'Invalid xAI OAuth device response fields')
  }
  // RFC 8628 allows interval 0; fall back to the poller's default instead of
  // failing on non-positive or malformed values.
  const intervalSeconds = positiveNumberField(body, 'interval')
  const complete = body['verification_uri_complete']
  return {
    deviceCode,
    userCode,
    verificationUri: validateVerificationUri(body['verification_uri']),
    ...(typeof complete === 'string' && complete ? { verificationUriComplete: validateVerificationUri(complete) } : {}),
    ...(intervalSeconds === undefined ? {} : { intervalSeconds }),
    expiresInSeconds,
  }
}

/** Parse a token payload; xAI may omit refresh_token when it did not rotate. */
export function xaiCredentialFromTokenResponse(
  body: Record<string, unknown>,
  nowSeconds: number,
  previousRefreshToken?: string,
): OAuthFlowCredential {
  const access = stringField(body, 'access_token')
  if (!access) {
    throw new ProviderError(XAI_OAUTH_PROVIDER, 'Invalid xAI OAuth response field: access_token')
  }
  const rotated = body['refresh_token'] !== undefined ? stringField(body, 'refresh_token') : previousRefreshToken
  if (!rotated) {
    throw new ProviderError(XAI_OAUTH_PROVIDER, 'Invalid xAI OAuth response field: refresh_token')
  }
  const lifetime = body['expires_in'] === undefined ? DEFAULT_TOKEN_LIFETIME_SECONDS : positiveNumberField(body, 'expires_in')
  if (lifetime === undefined) {
    throw new ProviderError(XAI_OAUTH_PROVIDER, 'Invalid xAI OAuth response field: expires_in')
  }
  // pi-ai banks the 5-minute skew into the stored expiry; Xerxes stores the
  // raw expiry and applies the skew at resolve time.
  return { access, refresh: rotated, expires: Math.floor(nowSeconds + lifetime) }
}

/** Request a device code (pi-ai `requestDeviceCode`). */
export async function requestXaiDeviceCode(options: {
  readonly fetchImplementation?: OAuthFetch
  readonly signal?: AbortSignal
} = {}): Promise<XaiDeviceCode> {
  const response = await postForm(XAI_DEVICE_CODE_URL, {
    client_id: XAI_OAUTH_CLIENT_ID,
    scope: XAI_OAUTH_SCOPE,
    referrer: 'xerxes',
  }, options.signal, {
    ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
  })
  if (!response.ok) throw requestFailure('device authorization', response)
  return xaiDeviceCodeFromResponse(response.body)
}

/** Refresh an xAI credential, keeping the previous refresh token when unrotated. */
export async function refreshXaiToken(
  refreshToken: string,
  options: {
    readonly fetchImplementation?: OAuthFetch
    readonly signal?: AbortSignal
    readonly now?: () => number
  } = {},
): Promise<OAuthFlowCredential> {
  const response = await postForm(XAI_TOKEN_URL, {
    grant_type: 'refresh_token',
    client_id: XAI_OAUTH_CLIENT_ID,
    refresh_token: refreshToken,
  }, options.signal, {
    ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
  })
  if (!response.ok) throw requestFailure('token refresh', response)
  return xaiCredentialFromTokenResponse(response.body, (options.now ?? (() => Date.now() / 1_000))(), refreshToken)
}

/**
 * Owns the stored xAI OAuth credential: device-flow login, refresh, and
 * persistence under `<xerxesHome>/auth/`.
 */
export class XaiOAuthSession {
  private readonly environment: Readonly<Record<string, string | undefined>>
  private readonly fetchImplementation: OAuthFetch | undefined
  private readonly home: string
  private readonly now: () => number
  private readonly sleep: (ms: number) => Promise<void>
  private pending: Promise<OAuthFlowCredential> | undefined

  constructor(options: XaiOAuthSessionOptions = {}) {
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
        'xai_oauth',
        "xAI credential carries no refresh token. Run 'xerxes auth login xai'.",
      )
    }
    const fresh = await refreshXaiToken(credential.refresh, {
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
      ...(signal ? { signal } : {}),
      now: this.now,
    })
    await this.persist(fresh)
    return fresh
  }

  /**
   * Run the RFC 8628 device flow: requests a device code, hands the user
   * code to `onUserCode`, then polls until xAI issues the token or the
   * device code expires.
   */
  async login(
    onUserCode: (userCode: string, verificationUri: string) => void,
    signal?: AbortSignal,
  ): Promise<OAuthFlowCredential> {
    const device = await requestXaiDeviceCode({
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
      ...(signal ? { signal } : {}),
    })
    onUserCode(device.userCode, device.verificationUriComplete ?? device.verificationUri)
    const credential = await pollDeviceCodeFlow({
      ...(device.intervalSeconds === undefined ? {} : { intervalSeconds: device.intervalSeconds }),
      expiresInSeconds: device.expiresInSeconds,
      waitBeforeFirstPoll: true,
      ...(signal ? { signal } : {}),
      sleep: this.sleep,
      now: this.now,
      poll: async () => {
        const response = await postForm(XAI_TOKEN_URL, {
          grant_type: 'urn:ietf:params:oauth:grant-type:device_code',
          client_id: XAI_OAUTH_CLIENT_ID,
          device_code: device.deviceCode,
        }, signal, {
          ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
        })
        // RFC 8628 poll responses arrive as HTTP 200 plus an `error` field;
        // classify that before treating the body as a token.
        const error = stringField(response.body, 'error')
        if (error === 'authorization_pending') return { status: 'pending' }
        if (error === 'slow_down') {
          const interval = positiveNumberField(response.body, 'interval')
          return { status: 'slow_down', ...(interval === undefined ? {} : { intervalSeconds: interval }) }
        }
        if (error === 'access_denied' || error === 'authorization_denied') {
          return { status: 'failed', message: 'xAI device authorization was denied' }
        }
        if (error === 'expired_token') {
          return { status: 'failed', message: 'xAI device code expired' }
        }
        if (error || !response.ok) {
          return { status: 'failed', message: requestFailure('device token polling', response).message }
        }
        return { status: 'complete', value: xaiCredentialFromTokenResponse(response.body, this.now()) }
      },
    })
    await this.persist(credential)
    return credential
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
    return expires - XAI_REFRESH_SKEW_SECONDS <= this.now()
  }

  private async resolve(signal?: AbortSignal): Promise<OAuthFlowCredential> {
    const stored = await this.loadStored()
    if (stored && !this.isExpired(stored.expires)) return stored
    if (stored) return this.refresh(stored, signal)
    throw new ConfigurationError(
      'xai_oauth',
      "No xAI subscription session found. Run 'xerxes auth login xai'.",
    )
  }

  private credentialPath(): string {
    return join(this.home, 'auth', 'xai-oauth.json')
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
