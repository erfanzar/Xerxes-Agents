// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomBytes } from 'node:crypto'

import { ClientError } from '../core/errors.js'

const DEFAULT_REDIRECT_URI = 'http://127.0.0.1:5454/callback'
const PKCE_RANDOM_BYTES = 96

export interface OAuthConfig {
  readonly authorizeUrl: string
  readonly clientId: string
  readonly redirectUri?: string
  readonly scopes?: readonly string[]
  readonly tokenUrl: string
}

export interface OAuthTokenInit {
  readonly accessToken: string
  readonly expiresAt?: number
  readonly refreshToken?: string
  readonly scopes?: readonly string[]
  readonly tokenType?: string
}

/** JSON representation used by the on-disk credential store. */
export interface OAuthTokenRecord {
  readonly access_token: string
  readonly expires_at: number | null
  readonly refresh_token: string | null
  readonly scopes: readonly string[]
  readonly token_type: string
}

export interface OAuthFetchOptions {
  readonly fetchImplementation?: OAuthFetch
  readonly now?: () => number
  readonly signal?: AbortSignal
}

export interface OAuthPkceOptions {
  /** Test-only deterministic verifier. Production callers should omit this. */
  readonly verifier?: string
}

export interface OAuthPkcePair {
  readonly challenge: string
  readonly verifier: string
}

export type OAuthFetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

/** A token-exchange failure that avoids putting token endpoint response bodies in logs. */
export class OAuthError extends ClientError {
  readonly status: number | undefined

  constructor(message: string, status?: number, cause: unknown = undefined) {
    super('oauth', message, cause, status === undefined ? {} : { status })
    this.status = status
  }
}

/** OAuth 2.1 access/refresh-token pair. Expiry timestamps are epoch seconds. */
export class OAuthToken {
  readonly accessToken: string
  readonly expiresAt: number | undefined
  readonly refreshToken: string | undefined
  readonly scopes: readonly string[]
  readonly tokenType: string

  constructor(input: OAuthTokenInit) {
    if (!input.accessToken) {
      throw new OAuthError('accessToken must not be empty')
    }
    const tokenType = input.tokenType ?? 'Bearer'
    if (!tokenType) {
      throw new OAuthError('tokenType must not be empty')
    }
    if (input.expiresAt !== undefined && !Number.isFinite(input.expiresAt)) {
      throw new OAuthError('expiresAt must be a finite epoch timestamp')
    }

    this.accessToken = input.accessToken
    this.refreshToken = input.refreshToken
    this.tokenType = tokenType
    this.expiresAt = input.expiresAt
    this.scopes = Object.freeze([...(input.scopes ?? [])])
  }

  /** Returns true when this token expires within the specified clock skew. */
  isExpired(skewSeconds = 30, now = Date.now() / 1000): boolean {
    return this.expiresAt !== undefined && now >= this.expiresAt - skewSeconds
  }

  /** Convert into the stable snake_case credential-file format. */
  toRecord(): OAuthTokenRecord {
    return {
      access_token: this.accessToken,
      refresh_token: this.refreshToken ?? null,
      token_type: this.tokenType,
      expires_at: this.expiresAt ?? null,
      scopes: [...this.scopes],
    }
  }

  toJSON(): OAuthTokenRecord {
    return this.toRecord()
  }

  /** Parse a successful OAuth token endpoint response. */
  static fromResponse(payload: unknown, now = Date.now() / 1000): OAuthToken {
    const record = asRecord(payload, 'OAuth token response must be a JSON object')
    const accessToken = requiredString(record, 'access_token')
    const expiresIn = optionalNumber(record.expires_in, 'expires_in')
    const refreshToken = optionalString(record.refresh_token, 'refresh_token')
    const scope = record.scope
    const tokenType = optionalString(record.token_type, 'token_type')
    if (scope !== undefined && typeof scope !== 'string') {
      throw new OAuthError('OAuth token response field scope must be a string')
    }
    return new OAuthToken({
      accessToken,
      ...(refreshToken === undefined ? {} : { refreshToken }),
      ...(tokenType === undefined ? {} : { tokenType }),
      ...(expiresIn === undefined ? {} : { expiresAt: now + expiresIn }),
      ...(scope ? { scopes: scope.split(/\s+/).filter(Boolean) } : {}),
    })
  }

  /** Parse a credential record written by {@link toRecord}. */
  static fromRecord(payload: unknown): OAuthToken {
    const record = asRecord(payload, 'OAuth credential record must be a JSON object')
    const accessToken = requiredString(record, 'access_token')
    const refreshToken = nullableString(record.refresh_token, 'refresh_token')
    const tokenType = record.token_type === undefined ? 'Bearer' : requiredString(record, 'token_type')
    const expiresAt = nullableNumber(record.expires_at, 'expires_at')
    const scopes = stringArray(record.scopes, 'scopes')
    return new OAuthToken({
      accessToken,
      ...(refreshToken === undefined ? {} : { refreshToken }),
      tokenType,
      ...(expiresAt === undefined ? {} : { expiresAt }),
      scopes,
    })
  }
}

/** Return a fresh verifier/challenge pair for the OAuth PKCE S256 method. */
export function generatePkcePair(options: OAuthPkceOptions = {}): OAuthPkcePair {
  const verifier = options.verifier ?? base64Url(randomBytes(PKCE_RANDOM_BYTES))
  return { verifier, challenge: pkceChallenge(verifier) }
}

/** Return the PKCE S256 challenge for an existing verifier. */
export function pkceChallenge(verifier: string): string {
  if (!verifier || !/^[\x20-\x7e]+$/.test(verifier)) {
    throw new OAuthError('PKCE verifier must be a non-empty ASCII string')
  }
  return base64Url(createHash('sha256').update(verifier, 'ascii').digest())
}

/** Build the browser URL for an authorization-code-with-PKCE request. */
export function buildAuthorizeUrl(
  config: OAuthConfig,
  options: { readonly codeChallenge: string; readonly state: string },
): string {
  const resolved = resolveConfig(config)
  if (!options.state) {
    throw new OAuthError('OAuth state must not be empty')
  }
  if (!options.codeChallenge) {
    throw new OAuthError('OAuth code challenge must not be empty')
  }

  let url: URL
  try {
    url = new URL(resolved.authorizeUrl)
  } catch (error) {
    throw new OAuthError('OAuth authorizeUrl must be an absolute URL', undefined, error)
  }
  url.searchParams.set('response_type', 'code')
  url.searchParams.set('client_id', resolved.clientId)
  url.searchParams.set('redirect_uri', resolved.redirectUri)
  url.searchParams.set('state', options.state)
  url.searchParams.set('code_challenge', options.codeChallenge)
  url.searchParams.set('code_challenge_method', 'S256')
  if (resolved.scopes.length) {
    url.searchParams.set('scope', resolved.scopes.join(' '))
  } else {
    url.searchParams.delete('scope')
  }
  return url.toString()
}

/** Exchange an authorization code and PKCE verifier for a token. */
export async function exchangeCode(
  config: OAuthConfig,
  options: OAuthFetchOptions & { readonly code: string; readonly codeVerifier: string },
): Promise<OAuthToken> {
  const resolved = resolveConfig(config)
  if (!options.code) {
    throw new OAuthError('OAuth authorization code must not be empty')
  }
  if (!options.codeVerifier) {
    throw new OAuthError('OAuth PKCE verifier must not be empty')
  }
  const payload = await requestToken(resolved, {
    grant_type: 'authorization_code',
    code: options.code,
    redirect_uri: resolved.redirectUri,
    client_id: resolved.clientId,
    code_verifier: options.codeVerifier,
  }, options)
  return OAuthToken.fromResponse(payload, options.now?.())
}

/** Refresh a token, retaining its old refresh token when the provider omits a replacement. */
export async function refreshToken(
  config: OAuthConfig,
  token: OAuthToken,
  options: OAuthFetchOptions = {},
): Promise<OAuthToken> {
  const resolved = resolveConfig(config)
  if (!token.refreshToken) {
    throw new OAuthError('no refreshToken available')
  }
  const payload = await requestToken(resolved, {
    grant_type: 'refresh_token',
    refresh_token: token.refreshToken,
    client_id: resolved.clientId,
  }, options)
  const refreshed = OAuthToken.fromResponse(payload, options.now?.())
  if (refreshed.refreshToken) {
    return refreshed
  }
  return new OAuthToken({
    accessToken: refreshed.accessToken,
    refreshToken: token.refreshToken,
    tokenType: refreshed.tokenType,
    ...(refreshed.expiresAt === undefined ? {} : { expiresAt: refreshed.expiresAt }),
    scopes: refreshed.scopes,
  })
}

interface ResolvedOAuthConfig {
  readonly authorizeUrl: string
  readonly clientId: string
  readonly redirectUri: string
  readonly scopes: readonly string[]
  readonly tokenUrl: string
}

function resolveConfig(config: OAuthConfig): ResolvedOAuthConfig {
  if (!config.clientId) {
    throw new OAuthError('OAuth clientId must not be empty')
  }
  if (!config.authorizeUrl) {
    throw new OAuthError('OAuth authorizeUrl must not be empty')
  }
  if (!config.tokenUrl) {
    throw new OAuthError('OAuth tokenUrl must not be empty')
  }
  return {
    clientId: config.clientId,
    authorizeUrl: config.authorizeUrl,
    tokenUrl: config.tokenUrl,
    redirectUri: config.redirectUri ?? DEFAULT_REDIRECT_URI,
    scopes: Object.freeze([...(config.scopes ?? [])]),
  }
}

async function requestToken(
  config: ResolvedOAuthConfig,
  fields: Readonly<Record<string, string>>,
  options: OAuthFetchOptions,
): Promise<Record<string, unknown>> {
  const fetchImplementation = options.fetchImplementation ?? fetch
  let response: Response
  try {
    response = await fetchImplementation(config.tokenUrl, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams(fields).toString(),
      ...(options.signal ? { signal: options.signal } : {}),
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
  return parseTokenPayload(body, response.headers.get('content-type'))
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

function asRecord(value: unknown, message: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new OAuthError(message)
  }
  return value as Record<string, unknown>
}

function requiredString(record: Record<string, unknown>, key: string): string {
  const value = record[key]
  if (typeof value !== 'string' || !value) {
    throw new OAuthError(`OAuth token response missing required ${key}`)
  }
  return value
}

function optionalString(value: unknown, key: string): string | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (typeof value !== 'string') {
    throw new OAuthError(`OAuth token response field ${key} must be a string`)
  }
  return value
}

function nullableString(value: unknown, key: string): string | undefined {
  return optionalString(value, key)
}

function optionalNumber(value: unknown, key: string): number | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  const number = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : Number.NaN
  if (!Number.isFinite(number)) {
    throw new OAuthError(`OAuth token response field ${key} must be a finite number`)
  }
  return number
}

function nullableNumber(value: unknown, key: string): number | undefined {
  return optionalNumber(value, key)
}

function stringArray(value: unknown, key: string): readonly string[] {
  if (value === undefined || value === null) {
    return []
  }
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string')) {
    throw new OAuthError(`OAuth credential field ${key} must be a string array`)
  }
  return value as string[]
}

function base64Url(value: Uint8Array): string {
  return Buffer.from(value).toString('base64').replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '')
}
