// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { type OAuthFetch } from '../mcp/oauth.js'
// Licensed under the Apache License, Version 2.0.

/**
 * Radius gateway OAuth, mirroring pi-ai `auth/oauth/radius.ts`: a pi-messages
 * gateway fronts its own OAuth endpoints, with browser-PKCE and device-code
 * login methods against `{gateway}/v1/oauth/*`.
 *
 * Xerxes has no pi-messages transport yet, so this module provides the
 * credential and its lifecycle only — there is deliberately no LLM provider
 * wired to it (an OpenAI-transport `radius` entry would be an invented
 * integration).
 */

import { timingSafeEqual } from 'node:crypto'

import { ConfigurationError, ProviderError } from '../core/errors.js'
import { xerxesHome as defaultXerxesHome } from '../daemon/paths.js'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { openInBrowser } from './codexLogin.js'
import {
  generatePkceS256,
  oauthCallbackHtml,
  pollDeviceCodeFlow,
  positiveNumberField,
  readJsonObject,
  stringField,
  type OAuthFlowCredential,
} from './oauthFlows.js'

export const RADIUS_OAUTH_PROVIDER = 'radius'

export const RADIUS_OAUTH_CLIENT_ID = 'pi-gateway'
export const RADIUS_OAUTH_SCOPE = 'gateway offline_access'
const CALLBACK_PORT = 1456
const CALLBACK_PATH = '/oauth/callback'
/** Refresh this far ahead of expiry so a token cannot die mid-stream. */
export const RADIUS_REFRESH_SKEW_SECONDS = 60

export interface RadiusCredential extends OAuthFlowCredential {
  /** The gateway this credential authenticates against. */
  readonly gateway: string
}

/** pi-ai `normalizeRadiusGatewayUrl`: default the scheme, strip trailing slashes. */
export function normalizeRadiusGatewayUrl(value: string): string {
  const withScheme = /^https?:\/\//iu.test(value) ? value : `https://${value}`
  return withScheme.replace(/\/+$/u, '')
}

export interface RadiusOAuthSessionOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly fetchImplementation?: OAuthFetch
  /** Overrides `<xerxesHome>` for the credential file location. */
  readonly xerxesHome?: string
  /** Seconds since the epoch; injected so expiry logic is testable. */
  readonly now?: () => number
  /** Injectable delay so device-flow pacing is observable without real sleeps. */
  readonly sleep?: (ms: number) => Promise<void>
}

class RadiusOAuthResponseError extends ProviderError {
  /** The RFC 6749 `error` code the gateway returned, when present. */
  readonly oauthError: string | undefined

  constructor(
    status: number,
    oauthError: string | undefined,
    description: string | undefined,
    message: string,
  ) {
    const detail = oauthError
      ? description ? `${oauthError}: ${description}` : oauthError
      : description || String(status)
    super(RADIUS_OAUTH_PROVIDER, `${message}: ${detail}`)
    this.name = 'RadiusOAuthResponseError'
    this.oauthError = oauthError
  }
}

async function readOAuthResponseError(response: Response, message: string): Promise<RadiusOAuthResponseError> {
  const text = await response.text().catch(() => '')
  let oauthError: string | undefined
  let description: string | undefined
  if (text) {
    try {
      const data = JSON.parse(text) as Record<string, unknown>
      oauthError = stringField(data, 'error')
      description = stringField(data, 'error_description')
    } catch {
      description = text
    }
  }
  return new RadiusOAuthResponseError(response.status, oauthError, description, message)
}

async function requestRadiusToken(
  gateway: string,
  body: Record<string, string>,
  options: {
    readonly fetchImplementation?: OAuthFetch
    readonly signal?: AbortSignal
    readonly now?: () => number
  },
): Promise<RadiusCredential> {
  const request = options.fetchImplementation ?? fetch
  let response: Response
  try {
    response = await request(new URL('/v1/oauth/token', gateway), {
      method: 'POST',
      headers: { accept: 'application/json', 'content-type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams(body).toString(),
      ...(options.signal ? { signal: options.signal } : {}),
    })
  } catch (error) {
    if (options.signal?.aborted) throw new Error('Login cancelled')
    throw error
  }
  if (!response.ok) {
    throw await readOAuthResponseError(response, 'Radius OAuth token request failed')
  }
  const data = await readJsonObject(response)
  // RFC 8628 token endpoints answer poll requests with HTTP 200 plus an
  // `error` field (authorization_pending, slow_down, …); surface that as the
  // same typed error so the poller can classify it.
  const oauthError = stringField(data, 'error')
  if (oauthError) {
    throw new RadiusOAuthResponseError(
      response.status,
      oauthError,
      stringField(data, 'error_description'),
      'Radius OAuth token request failed',
    )
  }
  const access = stringField(data, 'access_token')
  const refresh = stringField(data, 'refresh_token')
  const expiresIn = positiveNumberField(data, 'expires_in')
  if (!access || !refresh || expiresIn === undefined) {
    throw new ProviderError(RADIUS_OAUTH_PROVIDER, 'Radius OAuth token response is missing required fields')
  }
  const scope = stringField(data, 'scope')
  return {
    gateway,
    access,
    refresh,
    // pi-ai banks its 60s skew into the stored expiry; Xerxes stores the raw
    // expiry and applies the skew at resolve time.
    expires: Math.floor((options.now ?? (() => Date.now() / 1_000))() + expiresIn),
    ...(scope ? { scope } : {}),
  }
}

function constantTimeEquals(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, 'utf8')
  const expected = Buffer.from(right, 'utf8')
  // timingSafeEqual throws on a length mismatch, which is itself the answer.
  return leftBytes.byteLength === expected.byteLength && timingSafeEqual(leftBytes, expected)
}

/** Discover the gateway's interactive authorization endpoint. */
export async function loadRadiusOAuthDiscovery(
  gateway: string,
  options: {
    readonly fetchImplementation?: OAuthFetch
    readonly signal?: AbortSignal
  } = {},
): Promise<string> {
  const request = options.fetchImplementation ?? fetch
  let response: Response
  try {
    response = await request(new URL('/v1/oauth', gateway), {
      headers: { accept: 'application/json' },
      ...(options.signal ? { signal: options.signal } : {}),
    })
  } catch (error) {
    if (options.signal?.aborted) throw new Error('Login cancelled')
    throw error
  }
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new ProviderError(
      RADIUS_OAUTH_PROVIDER,
      `Could not load Radius OAuth config from ${gateway}: ${response.status} ${text.slice(0, 256)}`,
    )
  }
  const discovery = await readJsonObject(response)
  const authorizationEndpoint = stringField(discovery, 'authorizationEndpoint')
  if (!authorizationEndpoint) {
    throw new ProviderError(RADIUS_OAUTH_PROVIDER, `Invalid Radius OAuth config from ${gateway}`)
  }
  return authorizationEndpoint
}

/** Browser-PKCE login against the discovered authorization endpoint. */
export async function loginRadiusWithBrowser(
  gateway: string,
  authorizationEndpoint: string,
  options: {
    readonly signal?: AbortSignal
    readonly openUrl?: (url: string) => void
    readonly now?: () => number
  } = {},
): Promise<RadiusCredential> {
  const { verifier, challenge } = await generatePkceS256()
  const state = crypto.randomUUID()
  const redirectUri = `http://127.0.0.1:${CALLBACK_PORT}${CALLBACK_PATH}`
  const authorizeUrl = new URL(authorizationEndpoint)
  authorizeUrl.search = new URLSearchParams({
    response_type: 'code',
    client_id: RADIUS_OAUTH_CLIENT_ID,
    redirect_uri: redirectUri,
    scope: RADIUS_OAUTH_SCOPE,
    code_challenge: challenge,
    code_challenge_method: 'S256',
    handoff: 'url',
    state,
  }).toString()

  let callbackCode: string | undefined
  let wake: (() => void) | undefined
  const codeReceived = new Promise<void>(resolve => {
    wake = resolve
  })
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: CALLBACK_PORT,
    fetch(request) {
      const url = new URL(request.url)
      if (url.pathname !== CALLBACK_PATH) {
        return new Response(oauthCallbackHtml('error', 'Callback route not found.'), {
          status: 404,
          headers: { 'Content-Type': 'text/html; charset=utf-8' },
        })
      }
      if (!constantTimeEquals(url.searchParams.get('state') ?? '', state)) {
        return new Response(oauthCallbackHtml('error', 'OAuth state mismatch.'), {
          status: 400,
          headers: { 'Content-Type': 'text/html; charset=utf-8' },
        })
      }
      const error = url.searchParams.get('error')
      if (error) {
        wake?.()
        return new Response(oauthCallbackHtml('error', url.searchParams.get('error_description') ?? error), {
          status: 400,
          headers: { 'Content-Type': 'text/html; charset=utf-8' },
        })
      }
      const code = url.searchParams.get('code')
      if (!code) {
        return new Response(oauthCallbackHtml('error', 'Missing authorization code.'), {
          status: 400,
          headers: { 'Content-Type': 'text/html; charset=utf-8' },
        })
      }
      callbackCode = code
      wake?.()
      return new Response(oauthCallbackHtml('success', 'Signed in to Radius. You may now close this page.'), {
        headers: { 'Content-Type': 'text/html; charset=utf-8' },
      })
    },
  })

  try {
    ;(options.openUrl ?? openInBrowser)(authorizeUrl.toString())
    await codeReceived
    if (!callbackCode) {
      if (options.signal?.aborted) throw new Error('Login cancelled')
      throw new Error('OAuth callback did not complete.')
    }
    return await requestRadiusToken(gateway, {
      grant_type: 'authorization_code',
      client_id: RADIUS_OAUTH_CLIENT_ID,
      redirect_uri: redirectUri,
      code: callbackCode,
      code_verifier: verifier,
    }, options)
  } finally {
    void server.stop(true)
  }
}

/** Request a device authorization (pi-ai `requestDeviceAuthorization`). */
export async function requestRadiusDeviceAuthorization(
  gateway: string,
  options: {
    readonly fetchImplementation?: OAuthFetch
    readonly signal?: AbortSignal
  } = {},
): Promise<{
  deviceCode: string
  userCode: string
  verificationUri: string
  expiresInSeconds: number
  intervalSeconds?: number
}> {
  const request = options.fetchImplementation ?? fetch
  let response: Response
  try {
    response = await request(new URL('/v1/oauth/device', gateway), {
      method: 'POST',
      headers: { accept: 'application/json', 'content-type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ client_id: RADIUS_OAUTH_CLIENT_ID, scope: RADIUS_OAUTH_SCOPE }).toString(),
      ...(options.signal ? { signal: options.signal } : {}),
    })
  } catch (error) {
    if (options.signal?.aborted) throw new Error('Login cancelled')
    throw error
  }
  if (!response.ok) {
    throw await readOAuthResponseError(response, 'Radius OAuth device authorization failed')
  }
  const data = await readJsonObject(response)
  const deviceCode = stringField(data, 'device_code')
  const userCode = stringField(data, 'user_code')
  const verificationUri = stringField(data, 'verification_uri')
  const expiresInSeconds = positiveNumberField(data, 'expires_in')
  if (!deviceCode || !userCode || !verificationUri || expiresInSeconds === undefined) {
    throw new ProviderError(RADIUS_OAUTH_PROVIDER, 'Radius OAuth device authorization response is missing required fields')
  }
  const intervalSeconds = positiveNumberField(data, 'interval')
  return {
    deviceCode,
    userCode,
    verificationUri,
    expiresInSeconds,
    ...(intervalSeconds === undefined ? {} : { intervalSeconds }),
  }
}

/** Device-code login (pi-ai `loginWithDeviceCode`). */
export async function loginRadiusWithDeviceCode(
  gateway: string,
  options: {
    readonly fetchImplementation?: OAuthFetch
    readonly signal?: AbortSignal
    readonly now?: () => number
    readonly sleep?: (ms: number) => Promise<void>
    readonly onUserCode?: (userCode: string, verificationUri: string) => void
  } = {},
): Promise<RadiusCredential> {
  const device = await requestRadiusDeviceAuthorization(gateway, options)
  options.onUserCode?.(device.userCode, device.verificationUri)
  return pollDeviceCodeFlow({
    ...(device.intervalSeconds === undefined ? {} : { intervalSeconds: device.intervalSeconds }),
    expiresInSeconds: device.expiresInSeconds,
    ...(options.signal ? { signal: options.signal } : {}),
    ...(options.sleep ? { sleep: options.sleep } : {}),
    ...(options.now ? { now: options.now } : {}),
    poll: async () => {
      try {
        const credentials = await requestRadiusToken(gateway, {
          grant_type: 'urn:ietf:params:oauth:grant-type:device_code',
          client_id: RADIUS_OAUTH_CLIENT_ID,
          device_code: device.deviceCode,
        }, options)
        return { status: 'complete', value: credentials }
      } catch (error) {
        if (!(error instanceof RadiusOAuthResponseError)) throw error
        switch (error.oauthError) {
          case 'authorization_pending':
            return { status: 'pending' }
          case 'slow_down':
            return { status: 'slow_down' }
          case 'expired_token':
            return { status: 'failed', message: 'Device authorization expired.' }
          case 'access_denied':
            return { status: 'failed', message: 'Device authorization was denied.' }
          default:
            throw error
        }
      }
    },
  })
}

/**
 * Owns the stored Radius gateway credential, bound to one gateway URL:
 * browser/device login, refresh, persistence under `<xerxesHome>/auth/`.
 */
export class RadiusOAuthSession {
  private readonly environment: Readonly<Record<string, string | undefined>>
  private readonly fetchImplementation: OAuthFetch | undefined
  private readonly home: string
  private readonly now: () => number
  private readonly sleep: (ms: number) => Promise<void>
  private pending: Promise<RadiusCredential> | undefined

  constructor(options: RadiusOAuthSessionOptions = {}) {
    this.environment = options.environment ?? process.env
    this.fetchImplementation = options.fetchImplementation
    this.home = options.xerxesHome ?? defaultXerxesHome(this.environment)
    this.now = options.now ?? (() => Date.now() / 1_000)
    this.sleep = options.sleep ?? (ms => new Promise<void>(resolve => setTimeout(resolve, ms)))
  }

  /** The gateway this session authenticates against (arg wins, then env). */
  resolveGateway(explicit?: string): string {
    const value = explicit?.trim()
      || this.environment['RADIUS_GATEWAY']?.trim()
      || ''
    if (!value) {
      throw new ConfigurationError(
        'radius_gateway',
        "No Radius gateway configured. Pass one to 'xerxes auth login radius <gateway>' or set RADIUS_GATEWAY.",
      )
    }
    return normalizeRadiusGatewayUrl(value)
  }

  /** Return a usable credential, refreshing when at or near expiry. */
  async credential(explicitGateway?: string, signal?: AbortSignal): Promise<RadiusCredential> {
    if (this.pending) return this.pending
    const flight = this.resolve(explicitGateway, signal)
    const tracked = flight.finally(() => {
      if (this.pending === tracked) this.pending = undefined
    })
    this.pending = tracked
    return tracked
  }

  /** Re-mint the access token from the stored refresh token. */
  async refresh(credential: RadiusCredential, signal?: AbortSignal): Promise<RadiusCredential> {
    if (!credential.refresh) {
      throw new ConfigurationError(
        'radius_oauth',
        "Radius credential carries no refresh token. Run 'xerxes auth login radius'.",
      )
    }
    const fresh = await requestRadiusToken(credential.gateway, {
      grant_type: 'refresh_token',
      client_id: RADIUS_OAUTH_CLIENT_ID,
      refresh_token: credential.refresh,
    }, {
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
      ...(signal ? { signal } : {}),
      now: this.now,
    })
    await this.persist(fresh)
    return fresh
  }

  /**
   * Run the browser-PKCE login (or `method: 'device'` for the device-code
   * flow, used when signing in from another device) and persist the result.
   */
  async login(options: {
    readonly gateway?: string
    readonly method?: 'browser' | 'device'
    readonly signal?: AbortSignal
    readonly openUrl?: (url: string) => void
    readonly onUserCode?: (userCode: string, verificationUri: string) => void
  } = {}): Promise<RadiusCredential> {
    const gateway = this.resolveGateway(options.gateway)
    const shared = {
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
      ...(options.signal ? { signal: options.signal } : {}),
      now: this.now,
      sleep: this.sleep,
    }
    const credential = options.method === 'device'
      ? await loginRadiusWithDeviceCode(gateway, { ...shared, ...(options.onUserCode ? { onUserCode: options.onUserCode } : {}) })
      : await loginRadiusWithBrowser(gateway, await loadRadiusOAuthDiscovery(gateway, shared), {
        ...(options.signal ? { signal: options.signal } : {}),
        ...(options.openUrl ? { openUrl: options.openUrl } : {}),
        now: this.now,
        ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
      })
    await this.persist(credential)
    return credential
  }

  /** The persisted credential without refreshing, for `auth status`. */
  async stored(): Promise<RadiusCredential | undefined> {
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
    return expires - RADIUS_REFRESH_SKEW_SECONDS <= this.now()
  }

  private async resolve(explicitGateway: string | undefined, signal?: AbortSignal): Promise<RadiusCredential> {
    const gateway = this.resolveGateway(explicitGateway)
    const stored = await this.loadStored()
    if (stored && stored.gateway === gateway && !this.isExpired(stored.expires)) return stored
    if (stored && stored.gateway === gateway) return this.refresh(stored, signal)
    throw new ConfigurationError(
      'radius_oauth',
      `No Radius session for ${gateway}. Run 'xerxes auth login radius ${gateway}'.`,
    )
  }

  private credentialPath(): string {
    return join(this.home, 'auth', 'radius-oauth.json')
  }

  private async loadStored(): Promise<RadiusCredential | undefined> {
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
    const gateway = stringField(record, 'gateway')
    const expires = record && typeof record['expires'] === 'number' ? record['expires'] : undefined
    if (!access || !refresh || !gateway || expires === undefined) return undefined
    const scope = stringField(record, 'scope')
    return { access, refresh, gateway, expires, ...(scope ? { scope } : {}) }
  }

  private async persist(credential: RadiusCredential): Promise<void> {
    const path = this.credentialPath()
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, `${JSON.stringify(credential, null, 2)}\n`, 'utf8')
  }
}
