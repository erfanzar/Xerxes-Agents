// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomBytes } from 'node:crypto'

import { requireSkillText, skillJsonObject, type SkillFetch } from '../http.js'

/** OAuth scopes required by the bundled Google Workspace skill. */
export const GOOGLE_WORKSPACE_SCOPES = [
  'https://www.googleapis.com/auth/gmail.readonly',
  'https://www.googleapis.com/auth/gmail.send',
  'https://www.googleapis.com/auth/gmail.modify',
  'https://www.googleapis.com/auth/calendar',
  'https://www.googleapis.com/auth/drive.readonly',
  'https://www.googleapis.com/auth/contacts.readonly',
  'https://www.googleapis.com/auth/spreadsheets',
  'https://www.googleapis.com/auth/documents.readonly',
] as const

export const GOOGLE_AUTHORIZATION_ENDPOINT = 'https://accounts.google.com/o/oauth2/v2/auth'
export const GOOGLE_TOKEN_ENDPOINT = 'https://oauth2.googleapis.com/token'
export const GOOGLE_REVOCATION_ENDPOINT = 'https://oauth2.googleapis.com/revoke'

/** A Google user token. It is intentionally never stringified or logged by this module. */
export interface GoogleWorkspaceToken {
  readonly accessToken: string
  readonly expiresAt?: number
  readonly refreshToken?: string
  readonly scopes: readonly string[]
  readonly tokenType: string
}

/** PKCE state held only by caller-provided storage between authorization steps. */
export interface GooglePendingAuthorization {
  readonly codeVerifier: string
  readonly createdAt: number
  readonly redirectUri: string
  readonly state: string
}

/**
 * Explicit persistence boundary for Google Workspace authorization material.
 *
 * There is deliberately no default disk path or environment lookup: hosts choose
 * an encrypted credential store, keychain, or short-lived in-memory store.
 */
export interface GoogleWorkspaceAuthorizationStorage {
  loadPendingAuthorization(): Promise<GooglePendingAuthorization | undefined>
  loadToken(): Promise<GoogleWorkspaceToken | undefined>
  removePendingAuthorization(): Promise<void>
  removeToken(): Promise<void>
  savePendingAuthorization(pending: GooglePendingAuthorization): Promise<void>
  saveToken(token: GoogleWorkspaceToken): Promise<void>
}

/** Optional browser boundary. The OAuth client never opens a browser implicitly. */
export interface GoogleAuthorizationBrowser {
  open(url: string): Promise<void>
}

/** Explicit client configuration obtained from a Google OAuth client-secret document. */
export interface GoogleWorkspaceOAuthConfig {
  readonly authorizationEndpoint?: string
  readonly clientId: string
  readonly clientSecret?: string
  readonly redirectUri: string
  readonly revocationEndpoint?: string
  readonly scopes?: readonly string[]
  readonly tokenEndpoint?: string
}

/** Runtime dependencies for a native Google Workspace OAuth client. */
export interface GoogleWorkspaceOAuthClientOptions {
  readonly browser?: GoogleAuthorizationBrowser
  readonly fetchImplementation: SkillFetch
  readonly now?: () => number
  readonly randomBytes?: (size: number) => Uint8Array
  readonly storage: GoogleWorkspaceAuthorizationStorage
}

export interface GoogleAuthorizationRequest {
  readonly codeVerifier: string
  readonly state: string
  readonly url: string
}

export interface GoogleAuthorizationStatus {
  readonly missingScopes: readonly string[]
  readonly state: 'authorized' | 'expired' | 'missing' | 'partial'
}

export interface GoogleOAuthClientSecretEntry {
  readonly auth_uri?: unknown
  readonly client_id?: unknown
  readonly client_secret?: unknown
  readonly token_uri?: unknown
}

export interface GoogleOAuthClientSecretDocument {
  readonly installed?: GoogleOAuthClientSecretEntry
  readonly web?: GoogleOAuthClientSecretEntry
}

/** Error whose messages intentionally exclude authorization codes and tokens. */
export class GoogleWorkspaceAuthorizationError extends Error {
  constructor(message: string, options: { readonly cause?: unknown } = {}) {
    super(message, options)
    this.name = 'GoogleWorkspaceAuthorizationError'
  }
}

/**
 * Native authorization-code-with-PKCE client for Google Workspace.
 *
 * It needs explicit storage and transport dependencies, so importing a bundled
 * skill can neither discover credentials nor create a token file as a side effect.
 */
export class GoogleWorkspaceOAuthClient {
  private readonly authorizationEndpoint: string
  private readonly browser: GoogleAuthorizationBrowser | undefined
  private readonly config: RequiredGoogleWorkspaceOAuthConfig
  private readonly fetchImplementation: SkillFetch
  private readonly now: () => number
  private readonly randomBytes: (size: number) => Uint8Array
  private readonly storage: GoogleWorkspaceAuthorizationStorage

  constructor(config: GoogleWorkspaceOAuthConfig, options: GoogleWorkspaceOAuthClientOptions) {
    this.config = normalizeConfig(config)
    this.authorizationEndpoint = this.config.authorizationEndpoint
    this.browser = options.browser
    this.fetchImplementation = options.fetchImplementation
    this.now = options.now ?? (() => Date.now() / 1_000)
    this.randomBytes = options.randomBytes ?? randomBytes
    this.storage = options.storage
  }

  /** Begin a browser authorization flow and persist its PKCE verifier through the supplied store. */
  async beginAuthorization(options: { readonly openBrowser?: boolean } = {}): Promise<GoogleAuthorizationRequest> {
    const codeVerifier = base64Url(this.randomBytes(48))
    const state = base64Url(this.randomBytes(24))
    if (!codeVerifier || !state) throw new GoogleWorkspaceAuthorizationError('could not generate OAuth PKCE state')
    const pending: GooglePendingAuthorization = {
      codeVerifier,
      createdAt: this.now(),
      redirectUri: this.config.redirectUri,
      state,
    }
    await this.storage.savePendingAuthorization(pending)
    const url = this.authorizationUrl(pending)
    if (options.openBrowser === true) {
      if (!this.browser) throw new GoogleWorkspaceAuthorizationError('a browser adapter is required to open authorization URLs')
      await this.browser.open(url)
    }
    return { codeVerifier, state, url }
  }

  /** Complete a browser callback or exchange a bare authorization code after state validation. */
  async completeAuthorization(callbackOrCode: string, signal?: AbortSignal): Promise<GoogleWorkspaceToken> {
    const pending = await this.storage.loadPendingAuthorization()
    if (!pending) throw new GoogleWorkspaceAuthorizationError('no pending Google OAuth authorization exists; start authorization again')
    const callback = parseAuthorizationCallback(callbackOrCode)
    if (callback.state !== undefined && callback.state !== pending.state) {
      throw new GoogleWorkspaceAuthorizationError('Google OAuth callback state did not match the pending authorization')
    }
    const token = await this.requestToken({
      code: callback.code,
      code_verifier: pending.codeVerifier,
      grant_type: 'authorization_code',
      redirect_uri: pending.redirectUri,
    }, signal)
    await this.storage.saveToken(token)
    await this.storage.removePendingAuthorization()
    return token
  }

  /** Return an access token, refreshing it before use when it is expired or close to expiry. */
  async accessToken(signal?: AbortSignal): Promise<string> {
    const token = await this.loadUsableToken(signal)
    return token.accessToken
  }

  /** Return the current authorization status without printing or exposing credential values. */
  async authorizationStatus(): Promise<GoogleAuthorizationStatus> {
    const token = await this.storage.loadToken()
    if (!token) return { missingScopes: [], state: 'missing' }
    if (isExpired(token, this.now())) return { missingScopes: missingScopes(token.scopes, this.config.scopes), state: 'expired' }
    const missing = missingScopes(token.scopes, this.config.scopes)
    return { missingScopes: missing, state: missing.length ? 'partial' : 'authorized' }
  }

  /** Refresh a stored refresh token and write only the new token through the supplied store. */
  async refresh(signal?: AbortSignal): Promise<GoogleWorkspaceToken> {
    const current = await this.storage.loadToken()
    if (!current?.refreshToken) throw new GoogleWorkspaceAuthorizationError('no Google OAuth refresh token is available')
    const refreshed = await this.requestToken({
      grant_type: 'refresh_token',
      refresh_token: current.refreshToken,
    }, signal)
    const token: GoogleWorkspaceToken = {
      ...refreshed,
      refreshToken: refreshed.refreshToken ?? current.refreshToken,
      scopes: refreshed.scopes.length ? refreshed.scopes : current.scopes,
    }
    await this.storage.saveToken(token)
    return token
  }

  /** Revoke the stored token remotely, then clear local authorization material on success. */
  async revoke(signal?: AbortSignal): Promise<boolean> {
    const token = await this.storage.loadToken()
    if (!token) return false
    const response = await this.fetchGoogle(this.config.revocationEndpoint, {
      body: new URLSearchParams({ token: token.refreshToken ?? token.accessToken }).toString(),
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      method: 'POST',
      ...(signal === undefined ? {} : { signal }),
    })
    if (!response.ok) throw new GoogleWorkspaceAuthorizationError(`Google token revocation failed with HTTP ${response.status}`)
    await this.storage.removeToken()
    await this.storage.removePendingAuthorization()
    return true
  }

  private authorizationUrl(pending: GooglePendingAuthorization): string {
    const url = new URL(this.authorizationEndpoint)
    url.searchParams.set('access_type', 'offline')
    url.searchParams.set('client_id', this.config.clientId)
    url.searchParams.set('code_challenge', pkceChallenge(pending.codeVerifier))
    url.searchParams.set('code_challenge_method', 'S256')
    url.searchParams.set('include_granted_scopes', 'true')
    url.searchParams.set('prompt', 'consent')
    url.searchParams.set('redirect_uri', pending.redirectUri)
    url.searchParams.set('response_type', 'code')
    url.searchParams.set('scope', this.config.scopes.join(' '))
    url.searchParams.set('state', pending.state)
    return url.toString()
  }

  private async loadUsableToken(signal?: AbortSignal): Promise<GoogleWorkspaceToken> {
    const token = await this.storage.loadToken()
    if (!token) throw new GoogleWorkspaceAuthorizationError('Google Workspace is not authorized; begin authorization first')
    return isExpired(token, this.now()) ? this.refresh(signal) : token
  }

  private async requestToken(fields: Readonly<Record<string, string>>, signal?: AbortSignal): Promise<GoogleWorkspaceToken> {
    const payload: Record<string, string> = { client_id: this.config.clientId, ...fields }
    if (this.config.clientSecret !== undefined) payload.client_secret = this.config.clientSecret
    const response = await this.fetchGoogle(this.config.tokenEndpoint, {
      body: new URLSearchParams(payload).toString(),
      headers: { Accept: 'application/json', 'Content-Type': 'application/x-www-form-urlencoded' },
      method: 'POST',
      ...(signal === undefined ? {} : { signal }),
    })
    if (!response.ok) throw new GoogleWorkspaceAuthorizationError(`Google OAuth token request failed with HTTP ${response.status}`)
    let payloadValue: unknown
    try {
      payloadValue = await response.json()
    } catch (error) {
      throw new GoogleWorkspaceAuthorizationError('Google OAuth token response was not valid JSON', { cause: error })
    }
    return parseToken(payloadValue, this.now(), this.config.scopes)
  }

  private async fetchGoogle(input: RequestInfo | URL, init: RequestInit): Promise<Response> {
    try {
      return await this.fetchImplementation(input, init)
    } catch (error) {
      throw new GoogleWorkspaceAuthorizationError('Google OAuth request failed', { cause: error })
    }
  }
}

/** Parse an OAuth client-secret JSON document supplied by a host without reading it from disk. */
export function googleWorkspaceOAuthConfigFromClientSecret(
  document: GoogleOAuthClientSecretDocument,
  options: { readonly redirectUri: string; readonly scopes?: readonly string[] },
): GoogleWorkspaceOAuthConfig {
  const entry = document.installed ?? document.web
  if (!entry) throw new GoogleWorkspaceAuthorizationError('Google OAuth client secret must contain an installed or web client')
  return {
    authorizationEndpoint: requiredSecretText(entry.auth_uri, 'auth_uri'),
    clientId: requiredSecretText(entry.client_id, 'client_id'),
    clientSecret: requiredSecretText(entry.client_secret, 'client_secret'),
    redirectUri: requireSkillText(options.redirectUri, 'redirectUri'),
    ...(options.scopes === undefined ? {} : { scopes: options.scopes }),
    tokenEndpoint: requiredSecretText(entry.token_uri, 'token_uri'),
  }
}

/** Parse either a full OAuth redirect URL or a bare authorization code. */
export function parseAuthorizationCallback(callbackOrCode: string): { readonly code: string; readonly state?: string } {
  const value = requireSkillText(callbackOrCode, 'callbackOrCode')
  if (!/^https?:\/\//i.test(value)) return { code: value }
  let url: URL
  try {
    url = new URL(value)
  } catch (error) {
    throw new GoogleWorkspaceAuthorizationError('Google OAuth callback URL is invalid', { cause: error })
  }
  const errorName = url.searchParams.get('error')
  if (errorName) throw new GoogleWorkspaceAuthorizationError(`Google OAuth authorization was denied (${errorName})`)
  const code = url.searchParams.get('code')
  if (!code) throw new GoogleWorkspaceAuthorizationError('Google OAuth callback did not contain an authorization code')
  const state = url.searchParams.get('state')
  return { code, ...(state === null ? {} : { state }) }
}

/** Convert a stored token record to a redaction-safe native value object. */
export function googleWorkspaceTokenFromRecord(value: unknown): GoogleWorkspaceToken {
  return parseToken(value, 0, [])
}

/** Convert a native token into the stable JSON shape a caller-owned store may persist. */
export function googleWorkspaceTokenRecord(token: GoogleWorkspaceToken): Readonly<Record<string, unknown>> {
  return {
    access_token: token.accessToken,
    expires_at: token.expiresAt ?? null,
    refresh_token: token.refreshToken ?? null,
    scopes: [...token.scopes],
    token_type: token.tokenType,
  }
}

/** Build an OAuth PKCE S256 challenge. */
export function pkceChallenge(verifier: string): string {
  if (!verifier || !/^[\x20-\x7e]+$/.test(verifier)) {
    throw new GoogleWorkspaceAuthorizationError('PKCE verifier must be a non-empty ASCII string')
  }
  return createHash('sha256').update(verifier, 'ascii').digest('base64url')
}

interface RequiredGoogleWorkspaceOAuthConfig {
  readonly authorizationEndpoint: string
  readonly clientId: string
  readonly clientSecret: string | undefined
  readonly redirectUri: string
  readonly revocationEndpoint: string
  readonly scopes: readonly string[]
  readonly tokenEndpoint: string
}

function normalizeConfig(config: GoogleWorkspaceOAuthConfig): RequiredGoogleWorkspaceOAuthConfig {
  const scopes = [...(config.scopes ?? GOOGLE_WORKSPACE_SCOPES)].map(scope => requireSkillText(scope, 'Google OAuth scope'))
  if (!scopes.length) throw new GoogleWorkspaceAuthorizationError('Google OAuth scopes must not be empty')
  return {
    authorizationEndpoint: absoluteUrl(config.authorizationEndpoint ?? GOOGLE_AUTHORIZATION_ENDPOINT, 'authorizationEndpoint'),
    clientId: requireSkillText(config.clientId, 'clientId'),
    clientSecret: config.clientSecret === undefined ? undefined : requireSkillText(config.clientSecret, 'clientSecret'),
    redirectUri: absoluteUrl(config.redirectUri, 'redirectUri'),
    revocationEndpoint: absoluteUrl(config.revocationEndpoint ?? GOOGLE_REVOCATION_ENDPOINT, 'revocationEndpoint'),
    scopes: Object.freeze(scopes),
    tokenEndpoint: absoluteUrl(config.tokenEndpoint ?? GOOGLE_TOKEN_ENDPOINT, 'tokenEndpoint'),
  }
}

function parseToken(value: unknown, now: number, fallbackScopes: readonly string[]): GoogleWorkspaceToken {
  const record = skillJsonObject(value, 'Google OAuth token response')
  const accessToken = requiredSecretText(record.access_token, 'access_token')
  const expiresIn = optionalPositiveNumber(record.expires_in, 'expires_in')
  const expiresAt = record.expires_at === undefined || record.expires_at === null
    ? expiresIn === undefined ? undefined : now + expiresIn
    : optionalPositiveNumber(record.expires_at, 'expires_at')
  const refreshToken = optionalText(record.refresh_token, 'refresh_token')
  const tokenType = optionalText(record.token_type, 'token_type') ?? 'Bearer'
  const scopes = scopesFromValue(record.scopes ?? record.scope, fallbackScopes)
  return {
    accessToken,
    ...(expiresAt === undefined ? {} : { expiresAt }),
    ...(refreshToken === undefined ? {} : { refreshToken }),
    scopes: Object.freeze(scopes),
    tokenType,
  }
}

function base64Url(value: Uint8Array): string {
  return Buffer.from(value).toString('base64url')
}

function isExpired(token: GoogleWorkspaceToken, now: number): boolean {
  return token.expiresAt !== undefined && now >= token.expiresAt - 30
}

function missingScopes(granted: readonly string[], required: readonly string[]): string[] {
  const grantedSet = new Set(granted)
  return required.filter(scope => !grantedSet.has(scope))
}

function scopesFromValue(value: unknown, fallback: readonly string[]): string[] {
  if (value === undefined || value === null) return [...fallback]
  if (typeof value === 'string') return value.split(/\s+/).filter(Boolean)
  if (Array.isArray(value) && value.every(scope => typeof scope === 'string' && scope.trim())) return [...value]
  throw new GoogleWorkspaceAuthorizationError('Google OAuth scopes must be a space-separated string or string array')
}

function optionalPositiveNumber(value: unknown, name: string): number | undefined {
  if (value === undefined || value === null) return undefined
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new GoogleWorkspaceAuthorizationError(`${name} must be a non-negative finite number`)
  }
  return value
}

function optionalText(value: unknown, name: string): string | undefined {
  if (value === undefined || value === null) return undefined
  return requiredSecretText(value, name)
}

function requiredSecretText(value: unknown, name: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new GoogleWorkspaceAuthorizationError(`${name} must be a non-empty string`)
  return value.trim()
}

function absoluteUrl(value: string, name: string): string {
  const text = requireSkillText(value, name)
  try {
    return new URL(text).toString()
  } catch (error) {
    throw new GoogleWorkspaceAuthorizationError(`${name} must be an absolute URL`, { cause: error })
  }
}
