// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * ChatGPT (Codex) OAuth session.
 *
 * A Plus/Pro/Business ChatGPT subscription can drive the Codex backend at
 * `chatgpt.com/backend-api/codex` with an OAuth access token instead of a
 * metered `OPENAI_API_KEY`. The constants below are the public client
 * parameters the open-source Codex CLI ships — this is the same first-party
 * flow, not a private API — including the environment overrides it honors.
 *
 * The access token is a JWT whose `https://api.openai.com/auth` claim carries
 * the account id and plan, so nothing beyond the standard {@link OAuthToken}
 * has to be persisted: identity is derived from the token on use.
 */

import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

import { ConfigurationError, ProviderError } from '../core/errors.js'
import { OAuthToken, type OAuthConfig, type OAuthFetch } from '../mcp/oauth.js'
import { CredentialStorage } from './storage.js'

/** Provider key under which the Codex session is stored. */
export const CODEX_PROVIDER = 'openai-codex'

/** Public client id of the Codex CLI's OAuth application. */
export const CODEX_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'

export const CODEX_AUTHORIZE_URL = 'https://auth.openai.com/oauth/authorize'
export const CODEX_TOKEN_URL = 'https://auth.openai.com/oauth/token'
export const CODEX_REVOKE_URL = 'https://auth.openai.com/oauth/revoke'

/** Subscription-backed Responses endpoint host. */
export const CODEX_BASE_URL = 'https://chatgpt.com/backend-api/codex'

/**
 * `offline_access` is what yields the refresh token; without it the session
 * dies with the first access token and every turn needs a browser round trip.
 */
export const CODEX_SCOPES = ['openid', 'profile', 'email', 'offline_access'] as const

/** Identifies the calling surface to the backend, as the Codex CLI does. */
export const CODEX_ORIGINATOR = 'codex_cli_rs'

/**
 * Refresh this far ahead of expiry. A token that passes the check at request
 * build time but expires in flight surfaces as a 401 mid-stream, which the
 * agent loop cannot retry cleanly once tokens have been emitted.
 */
export const CODEX_REFRESH_SKEW_SECONDS = 120

/** Claims Xerxes reads out of a Codex access token. */
export interface CodexClaims {
  readonly accountId: string | undefined
  readonly email: string | undefined
  readonly expiresAt: number | undefined
  readonly planType: string | undefined
}

/** A resolved, ready-to-send Codex credential. */
export interface CodexCredential {
  readonly accessToken: string
  readonly accountId: string | undefined
  readonly planType: string | undefined
}

/**
 * OAuth client configuration for the ChatGPT authorization-code flow.
 *
 * The token URL honors `CODEX_REFRESH_TOKEN_URL_OVERRIDE` so a staging or
 * proxied environment stays configurable the same way the Codex CLI allows.
 */
export function codexOAuthConfig(
  redirectUri: string,
  environment: Readonly<Record<string, string | undefined>> = process.env,
): OAuthConfig {
  return {
    clientId: environment.CODEX_APP_SERVER_LOGIN_CLIENT_ID?.trim() || CODEX_CLIENT_ID,
    authorizeUrl: CODEX_AUTHORIZE_URL,
    tokenUrl: environment.CODEX_REFRESH_TOKEN_URL_OVERRIDE?.trim() || CODEX_TOKEN_URL,
    scopes: [...CODEX_SCOPES],
    redirectUri,
  }
}

/** Base URL for Codex inference, overridable for staging hosts. */
export function codexBaseUrl(
  environment: Readonly<Record<string, string | undefined>> = process.env,
): string {
  return environment.CODEX_BASE_URL?.trim() || CODEX_BASE_URL
}

/**
 * Decode the unverified payload of a JWT.
 *
 * The token is not validated here and must never be trusted for authorization
 * — the backend is the only authority on that. Xerxes reads it purely to route
 * the request (which account) and to report the plan to the user.
 */
export function decodeJwtClaims(token: string): Record<string, unknown> | undefined {
  const payload = token.split('.')[1]
  if (!payload) return undefined
  try {
    const decoded = Buffer.from(payload, 'base64url').toString('utf8')
    const parsed: unknown = JSON.parse(decoded)
    return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : undefined
  } catch {
    return undefined
  }
}

/** Read the account id, plan, and expiry a Codex access token carries. */
export function codexClaims(accessToken: string): CodexClaims {
  const claims = decodeJwtClaims(accessToken) ?? {}
  const auth = asRecord(claims['https://api.openai.com/auth'])
  const profile = asRecord(claims['https://api.openai.com/profile'])
  return {
    accountId: stringField(auth, 'chatgpt_account_id'),
    email: stringField(profile, 'email'),
    expiresAt: typeof claims.exp === 'number' ? claims.exp : undefined,
    planType: stringField(auth, 'chatgpt_plan_type'),
  }
}

/** Root of the Codex CLI's own state, honoring `CODEX_HOME`. */
export function codexCliHome(
  environment: Readonly<Record<string, string | undefined>> = process.env,
  homeDirectory = homedir(),
): string {
  return environment.CODEX_HOME?.trim() || join(homeDirectory, '.codex')
}

/**
 * Adopt the Codex CLI's own session if the user has already run `codex login`.
 *
 * This is the difference between "authenticate again in a second tool" and
 * Xerxes simply working on a machine that is already signed in. Returns
 * undefined whenever the file is absent, unreadable, or not an OAuth session
 * (an API-key-only `auth.json` has no `tokens`).
 */
export async function importCodexCliTokens(
  environment: Readonly<Record<string, string | undefined>> = process.env,
  homeDirectory = homedir(),
): Promise<OAuthToken | undefined> {
  let raw: string
  try {
    raw = await readFile(join(codexCliHome(environment, homeDirectory), 'auth.json'), 'utf8')
  } catch {
    return undefined
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return undefined
  }

  const tokens = asRecord(asRecord(parsed)?.tokens)
  const accessToken = stringField(tokens, 'access_token')
  if (!accessToken) return undefined
  const refresh = stringField(tokens, 'refresh_token')
  const expiresAt = codexClaims(accessToken).expiresAt

  return new OAuthToken({
    accessToken,
    ...(refresh === undefined ? {} : { refreshToken: refresh }),
    ...(expiresAt === undefined ? {} : { expiresAt }),
    scopes: [...CODEX_SCOPES],
  })
}

export interface CodexSessionOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly fetchImplementation?: OAuthFetch
  readonly homeDirectory?: string
  /** Seconds since the epoch; injected so expiry logic is testable. */
  readonly now?: () => number
  readonly storage?: CredentialStorage
}

/**
 * Resolves a live Codex bearer token, refreshing and re-persisting as needed.
 *
 * Resolution order is Xerxes' own stored session first, then the Codex CLI's
 * — so an explicit `xerxes auth login codex` always wins over an ambient CLI
 * login, but neither is required if the other is present.
 */
export class CodexSession {
  private readonly environment: Readonly<Record<string, string | undefined>>
  private readonly fetchImplementation: OAuthFetch | undefined
  private readonly homeDirectory: string
  private readonly now: () => number
  private pending: Promise<CodexCredential> | undefined
  private readonly storage: CredentialStorage

  constructor(options: CodexSessionOptions = {}) {
    this.environment = options.environment ?? process.env
    this.fetchImplementation = options.fetchImplementation
    this.homeDirectory = options.homeDirectory ?? homedir()
    this.now = options.now ?? (() => Date.now() / 1000)
    this.storage = options.storage ?? CredentialStorage.default()
  }

  /**
   * Return a usable credential, refreshing when it is at or near expiry.
   *
   * Concurrent callers share one in-flight resolution: a parallel fan-out of
   * subagents would otherwise race N refreshes against the same refresh token,
   * and a provider that rotates it on use invalidates every loser.
   */
  async credential(signal?: AbortSignal): Promise<CodexCredential> {
    this.pending ??= this.resolve(signal).finally(() => {
      this.pending = undefined
    })
    return this.pending
  }

  /** Persist a freshly minted session, replacing any stored one. */
  async store(token: OAuthToken): Promise<void> {
    await this.storage.save(CODEX_PROVIDER, token)
  }

  /** Forget the stored session. Returns false when there was nothing to remove. */
  async logout(): Promise<boolean> {
    return this.storage.remove(CODEX_PROVIDER)
  }

  /** Load the stored session without refreshing it. */
  async stored(): Promise<OAuthToken | undefined> {
    return this.storage.load(CODEX_PROVIDER)
  }

  private async resolve(signal?: AbortSignal): Promise<CodexCredential> {
    const stored = await this.storage.load(CODEX_PROVIDER)
    const token = stored ?? (await importCodexCliTokens(this.environment, this.homeDirectory))
    if (!token) {
      throw new ConfigurationError(
        'codex_auth',
        "No ChatGPT session found. Run 'xerxes auth login codex', or sign in with the Codex CLI.",
      )
    }

    if (!token.isExpired(CODEX_REFRESH_SKEW_SECONDS, this.now())) {
      // A CLI-sourced token is adopted into Xerxes' own store so the next
      // resolution does not depend on the CLI still being installed.
      if (!stored) await this.store(token)
      return credentialFrom(token)
    }

    const refreshed = await this.refresh(token, signal)
    await this.store(refreshed)
    return credentialFrom(refreshed)
  }

  private async refresh(token: OAuthToken, signal?: AbortSignal): Promise<OAuthToken> {
    if (!token.refreshToken) {
      throw new ConfigurationError(
        'codex_auth',
        "ChatGPT session expired and carries no refresh token. Run 'xerxes auth login codex'.",
      )
    }

    const config = codexOAuthConfig('', this.environment)
    const request = this.fetchImplementation ?? fetch
    const response = await request(config.tokenUrl, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        client_id: config.clientId,
        grant_type: 'refresh_token',
        refresh_token: token.refreshToken,
      }).toString(),
      ...(signal ? { signal } : {}),
    })

    if (!response.ok) {
      const body = await response.text()
      // 429 here means the plan's quota is exhausted, not that the credential
      // is bad — re-authenticating cannot lift a usage cap, so telling the
      // user to log in again would cost them a browser round trip and still
      // fail. The two cases must not be collapsed.
      if (response.status === 429) {
        const retryAfter = response.headers.get('retry-after')
        throw new ProviderError(
          CODEX_PROVIDER,
          `ChatGPT plan quota is exhausted (429); the session is still valid`
          + `${retryAfter ? `, retry after ${retryAfter}s` : ''}. ${body.slice(0, 512)}`,
        )
      }
      throw new ProviderError(
        CODEX_PROVIDER,
        `ChatGPT token refresh failed (${response.status}): ${body.slice(0, 512)}`,
      )
    }

    const payload: unknown = await response.json()
    const next = OAuthToken.fromResponse(payload, this.now())
    // OpenAI omits `refresh_token` when it is unchanged; dropping it here
    // would strand the session at the next expiry.
    if (next.refreshToken) return next
    return new OAuthToken({
      accessToken: next.accessToken,
      refreshToken: token.refreshToken,
      tokenType: next.tokenType,
      ...(next.expiresAt === undefined ? {} : { expiresAt: next.expiresAt }),
      scopes: next.scopes,
    })
  }
}

/** A reasoning effort a model accepts, as the provider describes it. */
export interface CodexReasoningLevel {
  readonly description: string | undefined
  readonly effort: string
}

/** One entry of the Codex backend's plan-scoped model catalog. */
export interface CodexModel {
  readonly contextLimit: number | undefined
  readonly defaultReasoningLevel: string | undefined
  readonly displayName: string | undefined
  /**
   * True when the model is coupled to the Codex CLI's own agent harness
   * rather than being generally callable.
   *
   * Xerxes uses the ChatGPT subscription as an entitlement; it is not a Codex
   * harness host. These models expect Codex's code-mode tool protocol and its
   * multi-agent orchestration, so driving them from a generic tool loop is
   * using them outside what they are built for.
   */
  readonly harnessCoupled: boolean
  readonly id: string
  /**
   * Efforts this specific model accepts. The set genuinely differs per model —
   * some accept `ultra`, others stop at `xhigh` — so it is read from the
   * catalog rather than assumed.
   */
  readonly reasoningLevels: readonly CodexReasoningLevel[]
}

/**
 * Client version reported to the catalog endpoint, which requires it.
 *
 * The backend gates the catalog on this: an omitted value is a 400, and a
 * value it considers too old can return a reduced list. It is intentionally a
 * plain constant rather than Xerxes' own version, which the backend has never
 * heard of.
 */
export const CODEX_CLIENT_VERSION = '0.144.4'

export interface CodexModelCatalogOptions {
  readonly baseUrl?: string
  readonly fetchImplementation?: typeof fetch
  /**
   * Drop models the Codex CLI drives through its own harness.
   *
   * Off by default: the flag describes how the Codex CLI uses a model, not a
   * restriction on calling it, and these models answer ordinary Responses
   * requests with ordinary tool calls. Hiding them would remove capability the
   * subscription pays for.
   */
  readonly excludeHarnessModels?: boolean
  readonly signal?: AbortSignal
}

/**
 * Fetch the models the signed-in plan may actually use.
 *
 * The catalog is plan-scoped, so it is discovered live rather than hard-coded:
 * a Plus account and a Pro account see different lists, and a static table
 * would offer models the backend then refuses.
 */
export async function fetchCodexModelCatalog(
  credential: CodexCredential,
  options: CodexModelCatalogOptions = {},
): Promise<readonly CodexModel[]> {
  const base = options.baseUrl ?? codexBaseUrl()
  const url = `${base.replace(/\/+$/, '')}/models?client_version=${encodeURIComponent(CODEX_CLIENT_VERSION)}`
  const request = options.fetchImplementation ?? fetch
  const response = await request(url, {
    headers: { ...codexAuthHeaders(credential), Accept: 'application/json' },
    ...(options.signal ? { signal: options.signal } : {}),
  })

  if (!response.ok) {
    throw new ProviderError(
      CODEX_PROVIDER,
      `Codex model catalog request failed (${response.status}): ${(await response.text()).slice(0, 512)}`,
    )
  }

  const payload = asRecord(await response.json())
  const entries = Array.isArray(payload?.models) ? payload.models : []
  const models: CodexModel[] = []
  for (const entry of entries) {
    const record = asRecord(entry)
    const id = stringField(record, 'id') ?? stringField(record, 'slug')
    if (!id) continue
    const contextLimit = record?.context_window
    models.push({
      id,
      displayName: stringField(record, 'display_name'),
      contextLimit: typeof contextLimit === 'number' && contextLimit > 0 ? contextLimit : undefined,
      defaultReasoningLevel: stringField(record, 'default_reasoning_level'),
      harnessCoupled: isHarnessCoupled(record),
      reasoningLevels: reasoningLevelsFrom(record?.supported_reasoning_levels),
    })
  }
  return options.excludeHarnessModels ? models.filter(model => !model.harnessCoupled) : models
}

/**
 * Detect a model the Codex CLI drives through its own harness.
 *
 * Reported as metadata, not used to hide anything: Xerxes runs its own agent
 * loop against the plan's entitlement, and these models serve ordinary
 * Responses requests — verified with a normal tool call — so the flag says how
 * the Codex CLI treats them rather than what a client is allowed to call.
 *
 * Keyed on capability flags rather than model names so it keeps describing the
 * right models as the catalog changes.
 *
 * - `tool_mode: "code_mode_only"` — the Codex CLI drives these with its
 *   code-mode tool protocol.
 * - `multi_agent_version` — participates in the harness's multi-agent flow.
 * - `use_responses_lite` — the Codex CLI uses a lighter request shape.
 */
function isHarnessCoupled(record: Record<string, unknown> | undefined): boolean {
  return (
    stringField(record, 'tool_mode') === 'code_mode_only'
    || stringField(record, 'multi_agent_version') !== undefined
    || record?.use_responses_lite === true
  )
}

/** Headers that authorize one Codex backend request. */
export function codexAuthHeaders(credential: CodexCredential, sessionId?: string): Record<string, string> {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${credential.accessToken}`,
    originator: CODEX_ORIGINATOR,
    'OpenAI-Beta': 'responses=experimental',
  }
  // The backend routes usage to a workspace by this header; without it a
  // multi-workspace account is billed against the wrong one or rejected.
  if (credential.accountId) headers['chatgpt-account-id'] = credential.accountId
  if (sessionId) headers.session_id = sessionId
  return headers
}

/**
 * Parse the catalog's reasoning-level list.
 *
 * Entries are objects carrying an effort plus a human description, but a bare
 * string is accepted too so a leaner catalog shape does not silently drop
 * every level and leave the model looking like it supports none.
 */
function reasoningLevelsFrom(value: unknown): readonly CodexReasoningLevel[] {
  if (!Array.isArray(value)) return []
  const levels: CodexReasoningLevel[] = []
  for (const entry of value) {
    if (typeof entry === 'string' && entry.trim()) {
      levels.push({ effort: entry.trim(), description: undefined })
      continue
    }
    const record = asRecord(entry)
    const effort = stringField(record, 'effort')
    if (effort) {
      levels.push({ effort, description: stringField(record, 'description') })
    }
  }
  return levels
}

function credentialFrom(token: OAuthToken): CodexCredential {
  const claims = codexClaims(token.accessToken)
  return {
    accessToken: token.accessToken,
    accountId: claims.accountId,
    planType: claims.planType,
  }
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

function stringField(record: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = record?.[key]
  return typeof value === 'string' && value ? value : undefined
}
