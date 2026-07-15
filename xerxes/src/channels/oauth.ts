// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomBytes } from 'node:crypto'

import { CredentialStorage, defaultCredentialStorage } from '../auth/storage.js'
import { OAuthError, OAuthToken, type OAuthFetch } from '../mcp/oauth.js'

const DEFAULT_INSTALL_ID = 'default'
const DEFAULT_REFRESH_SKEW_SECONDS = 60
const DEFAULT_STATE_TTL_SECONDS = 600
const RESERVED_AUTHORIZE_PARAMETERS = new Set([
  'client_id',
  'redirect_uri',
  'response_type',
  'scope',
  'state',
])
const STORAGE_KEY_PREFIX = 'channel-oauth'

/** Static OAuth configuration for a channel provider installation flow. */
export interface ChannelOAuthProvider {
  readonly authorizeUrl: string
  readonly clientId: string
  readonly clientSecret?: string
  readonly extraAuthorizeParams?: Readonly<Record<string, string>>
  readonly name: string
  readonly redirectUri: string
  readonly scopes?: readonly string[]
  readonly tokenUrl: string
}

/** Secure persistence boundary for installation-scoped channel credentials. */
export interface ChannelOAuthTokenStore {
  load(key: string): Promise<OAuthToken | undefined>
  save(key: string, token: OAuthToken): Promise<void>
}

/** Dependencies and policy values for one channel OAuth client. */
export interface ChannelOAuthClientOptions {
  readonly fetchImplementation?: OAuthFetch
  readonly now?: () => number
  readonly refreshSkewSeconds?: number
  readonly stateGenerator?: () => string
  readonly stateTtlSeconds?: number
  readonly storage?: ChannelOAuthTokenStore
}

export interface ChannelAuthorizeContext {
  readonly state: string
  readonly url: string
}

export interface CompleteChannelAuthorizeOptions {
  readonly code: string
  readonly installId?: string
  readonly state: string
}

/** Raised when a callback state is absent, expired, or has already been used. */
export class ChannelOAuthStateError extends OAuthError {
  constructor(message: string) {
    super(message)
    this.name = 'ChannelOAuthStateError'
  }
}

/**
 * OAuth authorization-code client for channel installations.
 *
 * Unlike the general OAuth helper, this client persists separate credentials for
 * each channel installation, consumes callback state exactly once, and serializes
 * refreshes per installation so refresh-token rotation cannot race.
 */
export class ChannelOAuthClient {
  readonly provider: ChannelOAuthProvider

  private readonly fetchImplementation: OAuthFetch
  private readonly now: () => number
  private readonly refreshLocks = new Map<string, Promise<OAuthToken>>()
  private readonly refreshSkewSeconds: number
  private readonly stateGenerator: () => string
  private readonly states = new Map<string, number>()
  private readonly stateTtlSeconds: number
  private readonly storage: ChannelOAuthTokenStore

  constructor(provider: ChannelOAuthProvider, options: ChannelOAuthClientOptions = {}) {
    this.provider = normalizeProvider(provider)
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.now = options.now ?? (() => Date.now() / 1000)
    this.refreshSkewSeconds = positiveFinite(
      options.refreshSkewSeconds ?? DEFAULT_REFRESH_SKEW_SECONDS,
      'refreshSkewSeconds',
    )
    this.stateGenerator = options.stateGenerator ?? defaultState
    this.stateTtlSeconds = positiveFinite(options.stateTtlSeconds ?? DEFAULT_STATE_TTL_SECONDS, 'stateTtlSeconds')
    this.storage = options.storage ?? credentialChannelOAuthStore()
  }

  /** Issue a CSRF state and render the provider's browser authorization URL. */
  beginAuthorize(): ChannelAuthorizeContext {
    const state = this.issueState()
    const url = new URL(this.provider.authorizeUrl)
    url.searchParams.set('response_type', 'code')
    url.searchParams.set('client_id', this.provider.clientId)
    url.searchParams.set('redirect_uri', this.provider.redirectUri)
    url.searchParams.set('state', state)
    if (this.provider.scopes?.length) {
      url.searchParams.set('scope', this.provider.scopes.join(' '))
    }
    for (const [name, value] of Object.entries(this.provider.extraAuthorizeParams ?? {})) {
      url.searchParams.set(name, value)
    }
    return Object.freeze({ state, url: url.toString() })
  }

  /** Validate and consume a callback state, then persist the resulting install token. */
  async completeAuthorize(options: CompleteChannelAuthorizeOptions): Promise<OAuthToken> {
    if (!this.consumeState(options.state)) {
      throw new ChannelOAuthStateError('OAuth callback state is invalid, expired, or already consumed')
    }
    if (!options.code.trim()) {
      throw new OAuthError('OAuth authorization code must not be empty')
    }
    const installId = normalizeInstallId(options.installId)
    const token = await this.requestToken({
      client_id: this.provider.clientId,
      code: options.code,
      grant_type: 'authorization_code',
      redirect_uri: this.provider.redirectUri,
      ...clientSecretField(this.provider),
    })
    await this.storage.save(channelOAuthStorageKey(this.provider.name, installId), token)
    return token
  }

  /** Load the persisted token for one channel installation without using the network. */
  async getToken(installId = DEFAULT_INSTALL_ID): Promise<OAuthToken | undefined> {
    return this.storage.load(channelOAuthStorageKey(this.provider.name, normalizeInstallId(installId)))
  }

  /** Return a current token, refreshing an expired installation token once when possible. */
  async getValidToken(installId = DEFAULT_INSTALL_ID): Promise<OAuthToken | undefined> {
    const normalizedInstallId = normalizeInstallId(installId)
    const token = await this.getToken(normalizedInstallId)
    if (!token || !token.isExpired(this.refreshSkewSeconds, this.now())) {
      return token
    }
    try {
      return await this.refresh(normalizedInstallId)
    } catch {
      return undefined
    }
  }

  /** Refresh and persist one installation token, coalescing concurrent callers per install ID. */
  async refresh(installId = DEFAULT_INSTALL_ID): Promise<OAuthToken> {
    const normalizedInstallId = normalizeInstallId(installId)
    const active = this.refreshLocks.get(normalizedInstallId)
    if (active) {
      return active
    }
    const refresh = this.refreshStoredToken(normalizedInstallId)
    this.refreshLocks.set(normalizedInstallId, refresh)
    try {
      return await refresh
    } finally {
      if (this.refreshLocks.get(normalizedInstallId) === refresh) {
        this.refreshLocks.delete(normalizedInstallId)
      }
    }
  }

  private consumeState(state: string): boolean {
    const issuedAt = this.states.get(state)
    this.states.delete(state)
    this.pruneStates()
    return issuedAt !== undefined && this.now() - issuedAt <= this.stateTtlSeconds
  }

  private issueState(): string {
    this.pruneStates()
    for (let attempt = 0; attempt < 8; attempt += 1) {
      const state = this.stateGenerator()
      if (!state || !state.trim()) {
        throw new OAuthError('OAuth state generator returned an empty state')
      }
      if (!this.states.has(state)) {
        this.states.set(state, this.now())
        return state
      }
    }
    throw new OAuthError('OAuth state generator repeatedly returned an existing state')
  }

  private pruneStates(): void {
    const cutoff = this.now() - this.stateTtlSeconds
    for (const [state, issuedAt] of this.states) {
      if (issuedAt < cutoff) {
        this.states.delete(state)
      }
    }
  }

  private async refreshStoredToken(installId: string): Promise<OAuthToken> {
    const current = await this.getToken(installId)
    if (!current) {
      throw new OAuthError('No OAuth token is stored for this channel installation')
    }
    if (!current.isExpired(this.refreshSkewSeconds, this.now())) {
      return current
    }
    if (!current.refreshToken) {
      throw new OAuthError('No OAuth refresh token is available; reauthorize this channel installation')
    }
    const refreshed = await this.requestToken({
      client_id: this.provider.clientId,
      grant_type: 'refresh_token',
      refresh_token: current.refreshToken,
      ...clientSecretField(this.provider),
    })
    const token = refreshed.refreshToken
      ? refreshed
      : new OAuthToken({
          accessToken: refreshed.accessToken,
          ...(refreshed.expiresAt === undefined ? {} : { expiresAt: refreshed.expiresAt }),
          refreshToken: current.refreshToken,
          scopes: refreshed.scopes,
          tokenType: refreshed.tokenType,
        })
    await this.storage.save(channelOAuthStorageKey(this.provider.name, installId), token)
    return token
  }

  private async requestToken(fields: Readonly<Record<string, string>>): Promise<OAuthToken> {
    let response: Response
    try {
      response = await this.fetchImplementation(this.provider.tokenUrl, {
        body: new URLSearchParams(fields).toString(),
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        method: 'POST',
      })
    } catch (error) {
      throw new OAuthError('OAuth token request failed', undefined, error)
    }
    if (!response.ok) {
      throw new OAuthError(`OAuth token request failed with HTTP ${response.status}`, response.status)
    }

    let body: string
    try {
      body = await response.text()
    } catch (error) {
      throw new OAuthError('OAuth token response could not be read', response.status, error)
    }
    return tokenFromPayload(parseTokenPayload(body, response.headers.get('content-type')), this.now())
  }
}

/** Adapt the encrypted credential store to the installation-scoped token-store boundary. */
export function credentialChannelOAuthStore(
  credentials: CredentialStorage = defaultCredentialStorage(),
): ChannelOAuthTokenStore {
  return Object.freeze({
    load: (key: string) => credentials.load(key),
    save: async (key: string, token: OAuthToken) => {
      await credentials.save(key, token)
    },
  })
}

/** Return an opaque, filename-safe credential key for a provider/install pair. */
export function channelOAuthStorageKey(providerName: string, installId = DEFAULT_INSTALL_ID): string {
  const provider = requiredIdentifier(providerName, 'provider name')
  const install = requiredIdentifier(installId, 'installId')
  return `${STORAGE_KEY_PREFIX}.${digest(provider)}.${digest(install)}`
}

function normalizeProvider(provider: ChannelOAuthProvider): ChannelOAuthProvider {
  const name = requiredIdentifier(provider.name, 'provider name')
  const clientId = requiredIdentifier(provider.clientId, 'clientId')
  const authorizeUrl = httpsUrl(provider.authorizeUrl, 'authorizeUrl')
  const tokenUrl = httpsUrl(provider.tokenUrl, 'tokenUrl')
  const redirectUri = absoluteUrl(provider.redirectUri, 'redirectUri')
  const scopes = Object.freeze((provider.scopes ?? []).map(scope => requiredIdentifier(scope, 'OAuth scope')))
  const extraAuthorizeParams = normalizeExtraAuthorizeParams(provider.extraAuthorizeParams)
  return Object.freeze({
    authorizeUrl,
    clientId,
    ...(provider.clientSecret === undefined ? {} : { clientSecret: provider.clientSecret }),
    ...(extraAuthorizeParams === undefined ? {} : { extraAuthorizeParams }),
    name,
    redirectUri,
    scopes,
    tokenUrl,
  })
}

function normalizeExtraAuthorizeParams(
  parameters: Readonly<Record<string, string>> | undefined,
): Readonly<Record<string, string>> | undefined {
  if (parameters === undefined) {
    return undefined
  }
  const normalized: Record<string, string> = {}
  for (const [name, value] of Object.entries(parameters)) {
    const key = requiredIdentifier(name, 'OAuth authorize parameter name')
    if (RESERVED_AUTHORIZE_PARAMETERS.has(key)) {
      throw new OAuthError(`OAuth authorize parameter ${key} is controlled by ChannelOAuthClient`)
    }
    if (typeof value !== 'string' || value.includes('\r') || value.includes('\n')) {
      throw new OAuthError(`OAuth authorize parameter ${key} must be a newline-free string`)
    }
    normalized[key] = value
  }
  return Object.freeze(normalized)
}

function clientSecretField(provider: ChannelOAuthProvider): Record<string, string> {
  return provider.clientSecret ? { client_secret: provider.clientSecret } : {}
}

function tokenFromPayload(payload: Record<string, unknown>, now: number): OAuthToken {
  const nested = asRecordOrUndefined(payload.authed_user)
  const accessToken = nonEmptyString(payload.access_token) ?? nonEmptyString(nested?.access_token)
  if (!accessToken) {
    throw new OAuthError('OAuth token response missing access_token')
  }
  const expiresIn = optionalNumber(payload.expires_in, 'expires_in')
  if (expiresIn !== undefined && expiresIn < 0) {
    throw new OAuthError('OAuth token response field expires_in must not be negative')
  }
  const refreshToken = optionalString(payload.refresh_token, 'refresh_token')
  const tokenType = optionalString(payload.token_type, 'token_type')
  return new OAuthToken({
    accessToken,
    ...(expiresIn === undefined ? {} : { expiresAt: now + expiresIn }),
    ...(refreshToken === undefined ? {} : { refreshToken }),
    scopes: tokenScopes(payload),
    ...(tokenType === undefined ? {} : { tokenType }),
  })
}

function tokenScopes(payload: Record<string, unknown>): readonly string[] {
  const scope = payload.scope ?? payload.scopes
  if (scope === undefined || scope === null || scope === '') {
    return []
  }
  if (typeof scope === 'string') {
    return scope.split(/\s+/).filter(Boolean)
  }
  if (Array.isArray(scope) && scope.every(value => typeof value === 'string')) {
    return scope
  }
  throw new OAuthError('OAuth token response scopes must be a string or string array')
}

function parseTokenPayload(body: string, contentType: string | null): Record<string, unknown> {
  const trimmed = body.trim()
  if (!trimmed) {
    throw new OAuthError('OAuth token response was empty')
  }
  if (contentType?.toLowerCase().includes('application/json') || trimmed.startsWith('{')) {
    try {
      return asRecord(JSON.parse(trimmed) as unknown, 'OAuth token response must be a JSON object')
    } catch (error) {
      if (error instanceof OAuthError) {
        throw error
      }
      throw new OAuthError('OAuth token response was not valid JSON', undefined, error)
    }
  }
  const values = new URLSearchParams(trimmed)
  const payload: Record<string, string> = {}
  for (const [key, value] of values) {
    payload[key] = value
  }
  return payload
}

function normalizeInstallId(installId: string | undefined): string {
  return requiredIdentifier(installId ?? DEFAULT_INSTALL_ID, 'installId')
}

function requiredIdentifier(value: string, field: string): string {
  const normalized = value.trim()
  if (!normalized) {
    throw new OAuthError(`${field} must not be empty`)
  }
  return normalized
}

function absoluteUrl(value: string, field: string): string {
  try {
    return new URL(value).toString()
  } catch (error) {
    throw new OAuthError(`${field} must be an absolute URL`, undefined, error)
  }
}

function httpsUrl(value: string, field: string): string {
  const url = new URL(absoluteUrl(value, field))
  if (url.protocol !== 'https:') {
    throw new OAuthError(`${field} must use HTTPS`)
  }
  return url.toString()
}

function positiveFinite(value: number, field: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new OAuthError(`${field} must be a positive finite number`)
  }
  return value
}

function asRecord(value: unknown, message: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new OAuthError(message)
  }
  return value as Record<string, unknown>
}

function asRecordOrUndefined(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

function optionalString(value: unknown, field: string): string | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (typeof value !== 'string') {
    throw new OAuthError(`OAuth token response field ${field} must be a string`)
  }
  return value
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === 'string' && value ? value : undefined
}

function optionalNumber(value: unknown, field: string): number | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : Number.NaN
  if (!Number.isFinite(parsed)) {
    throw new OAuthError(`OAuth token response field ${field} must be a finite number`)
  }
  return parsed
}

function digest(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('base64url').slice(0, 32)
}

function defaultState(): string {
  return Buffer.from(randomBytes(24)).toString('base64url')
}
