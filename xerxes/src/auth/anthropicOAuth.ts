// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { type OAuthFetch } from '../mcp/oauth.js'
// Licensed under the Apache License, Version 2.0.

/**
 * Anthropic OAuth (Claude Pro/Max subscription), mirroring pi-ai
 * `auth/oauth/anthropic.ts`: authorization-code + PKCE against
 * claude.ai/authorize with a loopback callback on the registered port,
 * raced against a manual paste of the redirect URL for headless machines,
 * then token exchange and refresh at platform.claude.com.
 *
 * The resulting access token authenticates the Anthropic messages API with
 * `Authorization: Bearer` plus the Claude Code identity surface — see
 * `isAnthropicOAuthToken`/`anthropicOAuthHeaders` and the transport wiring in
 * `src/llms/anthropic.ts`.
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
  type OAuthFlowCredential,
  readJsonObject,
} from './oauthFlows.js'

export const ANTHROPIC_OAUTH_PROVIDER = 'anthropic'

/** pi-ai registers this OAuth client for the Claude subscription flow. */
export const ANTHROPIC_OAUTH_CLIENT_ID = '9d1c250a-e61b-44d9-88ed-5944d1962f5e'
export const ANTHROPIC_AUTHORIZE_URL = 'https://claude.ai/oauth/authorize'
export const ANTHROPIC_TOKEN_URL = 'https://platform.claude.com/v1/oauth/token'
export const ANTHROPIC_CALLBACK_PORT = 53_692
export const ANTHROPIC_CALLBACK_PATH = '/callback'
export const ANTHROPIC_REDIRECT_URI = `http://localhost:${ANTHROPIC_CALLBACK_PORT}${ANTHROPIC_CALLBACK_PATH}`
export const ANTHROPIC_OAUTH_SCOPES =
  'org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload'

/** Refresh this far ahead of expiry so a token cannot die mid-stream. */
export const ANTHROPIC_REFRESH_SKEW_SECONDS = 300
/** Deadline for one token exchange/refresh HTTP call. */
export const ANTHROPIC_REQUEST_TIMEOUT_MS = 30_000
/** Give the user a real chance to finish a browser sign-in before giving up. */
export const ANTHROPIC_LOGIN_TIMEOUT_MS = 10 * 60 * 1_000

/** Claude Code identity headers the OAuth surface requires (pi-ai parity). */
export const CLAUDE_CODE_USER_AGENT = 'claude-cli/2.1.75'

export type AnthropicEnvironment = Readonly<Record<string, string | undefined>>

/**
 * Decide whether an Anthropic credential is an OAuth token (pi-ai
 * `isOAuthToken`): only subscription tokens carry the `sk-ant-oat` marker,
 * and only they may use the OAuth headers and Claude Code identity.
 */
export function isAnthropicOAuthToken(token: string): boolean {
  return token.includes('sk-ant-oat')
}

/**
 * Headers for an OAuth-token Anthropic request (pi-ai parity): Bearer
 * authorization instead of x-api-key, the Claude Code + OAuth beta flags,
 * and the Claude Code CLI identity.
 */
export function anthropicOAuthHeaders(accessToken: string, isOAuthToken: boolean): Record<string, string> {
  if (!isOAuthToken) {
    // ANTHROPIC_AUTH_TOKEN-style bearer credentials keep plain Bearer
    // semantics without claiming the Claude Code surface.
    return { Authorization: `Bearer ${accessToken}` }
  }
  return {
    Authorization: `Bearer ${accessToken}`,
    'anthropic-beta': 'claude-code-20250219,oauth-2025-04-20',
    'User-Agent': CLAUDE_CODE_USER_AGENT,
    'x-app': 'cli',
  }
}

/** The system-prompt identity block the OAuth surface requires. */
export const ANTHROPIC_OAUTH_IDENTITY_PROMPT = 'You are Claude Code, Anthropic\u2019s official CLI for Claude.'

/** Claude Code canonical tool names (pi-ai stealth mode); matched case-insensitively. */
const CLAUDE_CODE_TOOLS = [
  'Read',
  'Write',
  'Edit',
  'Bash',
  'Grep',
  'Glob',
  'AskUserQuestion',
  'EnterPlanMode',
  'ExitPlanMode',
  'KillShell',
  'NotebookEdit',
  'Skill',
  'Task',
  'TaskOutput',
  'TodoWrite',
  'WebFetch',
  'WebSearch',
] as const

const claudeCodeToolLookup = new Map(CLAUDE_CODE_TOOLS.map(name => [name.toLowerCase(), name]))

/**
 * Rename a tool to its Claude Code canonical casing (pi-ai `toClaudeCodeName`).
 * The subscription endpoint requires the Claude Code naming surface; unknown
 * names pass through unchanged.
 */
export function toClaudeCodeToolName(name: string): string {
  return claudeCodeToolLookup.get(name.toLowerCase()) ?? name
}

/** Parse a pasted redirect URL / `code#state` pair / bare code (pi-ai parity). */
export function parseAnthropicAuthorizationInput(input: string): { code?: string; state?: string } {
  const value = input.trim()
  if (!value) return {}
  try {
    const url = new URL(value)
    const code = url.searchParams.get('code') ?? undefined
    const state = url.searchParams.get('state') ?? undefined
    return {
      ...(code === undefined ? {} : { code }),
      ...(state === undefined ? {} : { state }),
    }
  } catch {
    // not a URL
  }
  if (value.includes('#')) {
    const [code, state] = value.split('#', 2)
    return {
      ...(code === undefined ? {} : { code }),
      ...(state === undefined ? {} : { state }),
    }
  }
  if (value.includes('code=')) {
    const params = new URLSearchParams(value)
    const code = params.get('code') ?? undefined
    const state = params.get('state') ?? undefined
    return {
      ...(code === undefined ? {} : { code }),
      ...(state === undefined ? {} : { state }),
    }
  }
  return { code: value }
}

export interface AnthropicLoginOptions {
  readonly signal?: AbortSignal
  readonly timeoutMs?: number
  /** Overrides the default browser open (tests). */
  readonly openUrl?: (url: string) => void
  readonly callbackHost?: string
  readonly callbackPort?: number
  /**
   * Headless fallback: resolves the pasted redirect URL / authorization code.
   * When provided it races the loopback callback exactly as pi-ai races its
   * manual prompt; the first source to produce a code wins.
   */
  readonly manualInput?: () => Promise<string | undefined>
}

/** Bind the loopback callback server and wait for the code (or manual paste). */
async function loginWithBrowserFlow(
  options: AnthropicLoginOptions,
  pkce: { verifier: string; challenge: string },
  context: AnthropicTokenRequestContext = {},
): Promise<OAuthFlowCredential> {
  const expectedState = pkce.verifier
  const callbackHost = options.callbackHost ?? '127.0.0.1'
  const callbackPort = options.callbackPort ?? ANTHROPIC_CALLBACK_PORT

  const authParams = new URLSearchParams({
    code: 'true',
    client_id: ANTHROPIC_OAUTH_CLIENT_ID,
    response_type: 'code',
    redirect_uri: ANTHROPIC_REDIRECT_URI,
    scope: ANTHROPIC_OAUTH_SCOPES,
    code_challenge: pkce.challenge,
    code_challenge_method: 'S256',
    state: expectedState,
  })

  const authorizeUrl = `${ANTHROPIC_AUTHORIZE_URL}?${authParams.toString()}`

  let settled = false
  let callbackCode: string | undefined
  let callbackState: string | undefined
  let manualInput: string | undefined
  let manualError: Error | undefined
  let wake: (() => void) | undefined
  const codeReceived = new Promise<void>(resolve => {
    wake = resolve
  })

  const server = Bun.serve({
    hostname: callbackHost,
    port: callbackPort,
    fetch(request) {
      const url = new URL(request.url)
      if (url.pathname !== ANTHROPIC_CALLBACK_PATH) {
        return new Response(oauthCallbackHtml('error', 'Callback route not found.'), {
          status: 404,
          headers: { 'Content-Type': 'text/html; charset=utf-8' },
        })
      }
      const error = url.searchParams.get('error')
      if (error) {
        return new Response(oauthCallbackHtml('error', 'Anthropic authentication did not complete.', `Error: ${error}`), {
          status: 400,
          headers: { 'Content-Type': 'text/html; charset=utf-8' },
        })
      }
      const code = url.searchParams.get('code')
      const state = url.searchParams.get('state')
      if (!code || !state) {
        return new Response(oauthCallbackHtml('error', 'Missing code or state parameter.'), {
          status: 400,
          headers: { 'Content-Type': 'text/html; charset=utf-8' },
        })
      }
      if (!constantTimeEquals(state, expectedState)) {
        return new Response(oauthCallbackHtml('error', 'State mismatch.'), {
          status: 400,
          headers: { 'Content-Type': 'text/html; charset=utf-8' },
        })
      }
      callbackCode = code
      callbackState = state
      wake?.()
      return new Response(oauthCallbackHtml('success', 'Anthropic authentication completed. You can close this window.'), {
        headers: { 'Content-Type': 'text/html; charset=utf-8' },
      })
    },
  })

  const manualPromise = options.manualInput?.().then(
    input => {
      manualInput = input
      wake?.()
    },
    error => {
      manualError = error instanceof Error ? error : new Error(String(error))
      wake?.()
    },
  )

  ;(options.openUrl ?? openInBrowser)(authorizeUrl)

  const timeoutMs = options.timeoutMs ?? ANTHROPIC_LOGIN_TIMEOUT_MS
  let deadlineHit = false
  const timer = setTimeout(() => {
    deadlineHit = true
    wake?.()
  }, timeoutMs)
  const onAbort = () => wake?.()
  options.signal?.addEventListener('abort', onAbort, { once: true })
  if (options.signal?.aborted) onAbort()

  try {
    await codeReceived
    let code: string | undefined
    let state: string | undefined
    if (callbackCode) {
      code = callbackCode
      state = callbackState
    } else if (deadlineHit) {
      throw new Error('Anthropic sign-in timed out before the browser returned')
    } else if (manualError) {
      throw manualError
    } else if (manualInput) {
      const parsed = parseAnthropicAuthorizationInput(manualInput)
      if (parsed.state && parsed.state !== pkce.verifier) throw new Error('OAuth state mismatch')
      code = parsed.code
      state = parsed.state ?? pkce.verifier
    }
    if (options.signal?.aborted) throw new Error('Login cancelled')
    if (!code) throw new Error('Missing authorization code')
    if (!state) throw new Error('Missing OAuth state')
    return await exchangeAnthropicAuthorizationCode(code, state, pkce.verifier, options.signal, context)
  } finally {
    clearTimeout(timer)
    options.signal?.removeEventListener('abort', onAbort)
    void manualPromise
    void server.stop(true)
  }
}

function constantTimeEquals(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, 'utf8')
  const rightBytes = Buffer.from(right, 'utf8')
  return leftBytes.byteLength === rightBytes.byteLength && timingSafeEqual(leftBytes, rightBytes)
}

interface AnthropicTokenRequestContext {
  readonly fetchImplementation?: OAuthFetch
}

async function postJson(
  url: string,
  body: Record<string, string>,
  signal: AbortSignal | undefined,
  context: AnthropicTokenRequestContext = {},
): Promise<Record<string, unknown>> {
  const request = context.fetchImplementation ?? fetch
  const deadline = requestDeadline(ANTHROPIC_REQUEST_TIMEOUT_MS, signal)
  let response: Response
  try {
    response = await request(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(body),
      signal: deadline.signal,
    })
  } finally {
    deadline.dispose()
  }
  const payload = await readJsonObject(response)
  if (!response.ok) {
    const text = payload ? JSON.stringify(payload) : ''
    throw new ProviderError(
      ANTHROPIC_OAUTH_PROVIDER,
      `Anthropic token request failed (HTTP ${response.status})${text ? `: ${text.slice(0, 512)}` : ''}`,
    )
  }
  if (!payload) {
    throw new ProviderError(ANTHROPIC_OAUTH_PROVIDER, 'Anthropic token request returned invalid JSON')
  }
  return payload
}

/** Bound one request, honouring a caller's abort as well as the deadline. */
function requestDeadline(timeoutMs: number, signal?: AbortSignal): {
  readonly dispose: () => void
  readonly signal: AbortSignal
} {
  const controller = new AbortController()
  const timer = setTimeout(
    () => controller.abort(new Error(`Anthropic request timed out after ${timeoutMs}ms`)),
    timeoutMs,
  )
  const dispose = (): void => clearTimeout(timer)
  if (!signal) return { dispose, signal: controller.signal }
  if (signal.aborted) controller.abort(signal.reason)
  else signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true })
  return { dispose, signal: controller.signal }
}

function credentialFromTokenResponse(payload: Record<string, unknown>, nowMs: number): OAuthFlowCredential {
  const access = payload['access_token']
  const refresh = payload['refresh_token']
  const expiresIn = payload['expires_in']
  if (typeof access !== 'string' || !access || typeof refresh !== 'string' || !refresh
    || typeof expiresIn !== 'number' || !Number.isFinite(expiresIn) || expiresIn <= 0) {
    throw new ProviderError(ANTHROPIC_OAUTH_PROVIDER, `Anthropic token response missing fields: ${JSON.stringify(payload).slice(0, 512)}`)
  }
  const scope = typeof payload['scope'] === 'string' ? payload['scope'] : undefined
  return {
    access,
    refresh,
    // pi-ai banks the 5-minute refresh skew into the stored expiry; Xerxes
    // stores the raw expiry and applies the skew at resolve time.
    expires: Math.floor(nowMs / 1_000 + expiresIn),
    ...(scope ? { scope } : {}),
  }
}

/** Exchange an authorization code (and PKCE verifier) for tokens. */
export async function exchangeAnthropicAuthorizationCode(
  code: string,
  state: string,
  verifier: string,
  signal?: AbortSignal,
  context: AnthropicTokenRequestContext = {},
): Promise<OAuthFlowCredential> {
  const payload = await postJson(ANTHROPIC_TOKEN_URL, {
    grant_type: 'authorization_code',
    client_id: ANTHROPIC_OAUTH_CLIENT_ID,
    code,
    state,
    redirect_uri: ANTHROPIC_REDIRECT_URI,
    code_verifier: verifier,
  }, signal, context)
  return credentialFromTokenResponse(payload, Date.now())
}

/** Refresh an Anthropic OAuth credential. */
export async function refreshAnthropicToken(
  refreshToken: string,
  signal?: AbortSignal,
  context: AnthropicTokenRequestContext = {},
): Promise<OAuthFlowCredential> {
  const payload = await postJson(ANTHROPIC_TOKEN_URL, {
    grant_type: 'refresh_token',
    client_id: ANTHROPIC_OAUTH_CLIENT_ID,
    refresh_token: refreshToken,
  }, signal, context)
  return credentialFromTokenResponse(payload, Date.now())
}

export interface AnthropicOAuthSessionOptions {
  readonly environment?: AnthropicEnvironment
  readonly fetchImplementation?: OAuthFetch
  /** Overrides `<xerxesHome>` for the credential file location. */
  readonly xerxesHome?: string
  /** Seconds since the epoch; injected so expiry logic is testable. */
  readonly now?: () => number
  readonly sleep?: (ms: number) => Promise<void>
}

/**
 * Owns the stored Anthropic OAuth credential: single-flight resolution,
 * refresh ahead of expiry, persistence under `<xerxesHome>/auth/`.
 */
export class AnthropicOAuthSession {
  private readonly environment: AnthropicEnvironment
  private readonly fetchImplementation: OAuthFetch | undefined
  private readonly home: string
  private readonly now: () => number
  private pending: Promise<OAuthFlowCredential> | undefined

  constructor(options: AnthropicOAuthSessionOptions = {}) {
    this.environment = options.environment ?? process.env
    this.fetchImplementation = options.fetchImplementation
    this.home = options.xerxesHome ?? defaultXerxesHome(this.environment)
    this.now = options.now ?? (() => Date.now() / 1_000)
  }

  /**
   * Return a usable credential, refreshing when at or near expiry.
   *
   * Without a stored session the ambient `ANTHROPIC_AUTH_TOKEN` /
   * `ANTHROPIC_OAUTH_TOKEN` is returned as an unrefreshable credential
   * (pi-ai's env resolution order), so API-key users are untouched and
   * explicit bearer tokens still reach the OAuth request path.
   */
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
        'anthropic_oauth',
        "Anthropic credential carries no refresh token. Run 'xerxes auth login anthropic'.",
      )
    }
    const fresh = await refreshAnthropicToken(credential.refresh, signal, {
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
    const next: OAuthFlowCredential = { ...fresh }
    await this.persist(next)
    return next
  }

  /**
   * Run the PKCE browser flow: opens claude.ai/authorize, waits for the
   * loopback callback on port 53692 (or a manual paste via `manualInput`),
   * exchanges the code, and persists the session.
   */
  async login(options: AnthropicLoginOptions = {}): Promise<OAuthFlowCredential> {
    const pkce = await generatePkceS256()
    const credential = await loginWithBrowserFlow(options, pkce, {
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
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
    return expires - ANTHROPIC_REFRESH_SKEW_SECONDS <= this.now()
  }

  /** Ambient bearer token from the environment (pi-ai env order). */
  private environmentToken(): string | undefined {
    for (const key of ['ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_OAUTH_TOKEN']) {
      const value = this.environment[key]?.trim()
      if (value) return value
    }
    return undefined
  }

  private async resolve(signal?: AbortSignal): Promise<OAuthFlowCredential> {
    const stored = await this.loadStored()
    if (stored && !this.isExpired(stored.expires)) return stored
    if (stored) return this.refresh(stored, signal)
    const ambient = this.environmentToken()
    if (ambient) {
      return { access: ambient, refresh: '', expires: Number.MAX_SAFE_INTEGER }
    }
    throw new ConfigurationError(
      'anthropic_oauth',
      "No Anthropic subscription session found. Run 'xerxes auth login anthropic'.",
    )
  }

  private credentialPath(): string {
    return join(this.home, 'auth', 'anthropic-oauth.json')
  }

  /** Where the credential persists, for status/logout surfaces. */
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
    const access = record && typeof record['access'] === 'string' ? record['access'] : undefined
    const refresh = record && typeof record['refresh'] === 'string' ? record['refresh'] : undefined
    const expires = record && typeof record['expires'] === 'number' ? record['expires'] : undefined
    if (!access || !refresh || expires === undefined) return undefined
    const scope = record && typeof record['scope'] === 'string' ? record['scope'] : undefined
    return { access, refresh, expires, ...(scope ? { scope } : {}) }
  }

  private async persist(credential: OAuthFlowCredential): Promise<void> {
    const path = this.credentialPath()
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, `${JSON.stringify(credential, null, 2)}\n`, 'utf8')
  }
}
