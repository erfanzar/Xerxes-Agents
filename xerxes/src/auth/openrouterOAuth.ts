// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { type OAuthFetch } from '../mcp/oauth.js'
// Licensed under the Apache License, Version 2.0.

/**
 * OpenRouter OAuth PKCE flow, mirroring pi-ai `auth/oauth/openrouter.ts`.
 *
 * OpenRouter exchanges an authorization code for a permanent, user-controlled
 * API key rather than an expiring token pair: `refresh` is the identity and
 * the credential never expires. The callback is a one-shot loopback server on
 * an ephemeral port, raced against a manual paste so headless sessions can
 * complete the login.
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

export const OPENROUTER_OAUTH_PROVIDER = 'openrouter'

export const OPENROUTER_AUTHORIZE_URL = 'https://openrouter.ai/auth'
export const OPENROUTER_TOKEN_URL = 'https://openrouter.ai/api/v1/auth/keys'
/** Give the user a real chance to finish a browser sign-in before giving up. */
export const OPENROUTER_LOGIN_TIMEOUT_MS = 5 * 60 * 1_000
const TOKEN_EXCHANGE_TIMEOUT_MS = 30_000

export interface OpenRouterLoginOptions {
  readonly signal?: AbortSignal
  readonly timeoutMs?: number
  /** Overrides the default browser open (tests). */
  readonly openUrl?: (url: string) => void
  /** Injectable token-exchange fetch (tests). */
  readonly fetchImplementation?: OAuthFetch
  /**
   * Headless fallback: resolves the pasted redirect URL / authorization code.
   * Races the loopback callback; the first source to produce a code wins.
   */
  readonly manualInput?: () => Promise<string | undefined>
}

/** Parse a pasted redirect URL or bare code (pi-ai parity). */
export function parseOpenRouterAuthorizationInput(input: string): string | undefined {
  const value = input.trim()
  if (!value) return undefined
  try {
    return new URL(value).searchParams.get('code') ?? undefined
  } catch {
    // not a URL
  }
  if (value.includes('code=')) {
    return new URLSearchParams(value).get('code') ?? undefined
  }
  return value
}

function errorDetail(body: Record<string, unknown>): string | undefined {
  if (typeof body['error_description'] === 'string') return body['error_description']
  if (typeof body['message'] === 'string') return body['message']
  if (typeof body['error'] === 'string') return body['error']
  const nested = body['error']
  if (nested !== null && typeof nested === 'object' && !Array.isArray(nested)) {
    const message = (nested as Record<string, unknown>)['message']
    if (typeof message === 'string') return message
  }
  return undefined
}

/** Exchange an authorization code for the permanent user-controlled API key. */
export async function exchangeOpenRouterAuthorizationCode(
  code: string,
  verifier: string,
  signal?: AbortSignal,
  context: { readonly fetchImplementation?: OAuthFetch } = {},
): Promise<OAuthFlowCredential> {
  if (signal?.aborted) throw new Error('Login cancelled')
  const controller = new AbortController()
  const reason = signal?.reason
  const onAbort = (): void => controller.abort(reason)
  signal?.addEventListener('abort', onAbort, { once: true })
  const timer = setTimeout(
    () => controller.abort(new Error('OpenRouter OAuth token exchange timed out')),
    TOKEN_EXCHANGE_TIMEOUT_MS,
  )

  let response: Response
  let body: Record<string, unknown> = {}
  try {
    response = await (context.fetchImplementation ?? fetch)(OPENROUTER_TOKEN_URL, {
      method: 'POST',
      headers: { accept: 'application/json', 'content-type': 'application/json' },
      body: JSON.stringify({ code, code_verifier: verifier, code_challenge_method: 'S256' }),
      signal: controller.signal,
    })
    const parsed = await readJsonObject(response)
    if (parsed) body = parsed
    else if (response.ok) throw new ProviderError(OPENROUTER_OAUTH_PROVIDER, 'OpenRouter OAuth returned invalid JSON')
  } catch (error) {
    if (signal?.aborted) throw new Error('Login cancelled')
    if (controller.signal.aborted) throw new Error('OpenRouter OAuth token exchange timed out')
    throw error
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }

  if (!response.ok) {
    const detail = errorDetail(body)
    throw new ProviderError(
      OPENROUTER_OAUTH_PROVIDER,
      `OpenRouter OAuth key exchange failed (HTTP ${response.status})${detail ? `: ${detail}` : ''}`,
    )
  }
  const key = body['key']
  if (typeof key !== 'string' || !key) {
    throw new ProviderError(OPENROUTER_OAUTH_PROVIDER, 'OpenRouter OAuth response carries no "key"')
  }
  // A user-controlled API key never expires and never rotates.
  return { access: key, refresh: '', expires: Number.MAX_SAFE_INTEGER }
}

function constantTimeEquals(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, 'utf8')
  const rightBytes = Buffer.from(right, 'utf8')
  return leftBytes.byteLength === rightBytes.byteLength && timingSafeEqual(leftBytes, rightBytes)
}

interface OpenRouterCallback {
  readonly callbackUrl: string
  /** Stop listening without settling the login (manual input takes over). */
  close(): void
  /** Hand the login over to manual entry unless a callback already claimed it. */
  cancelWait(): void
  waitForCredential(): Promise<OAuthFlowCredential | undefined>
}

/** Bind the one-shot loopback server that performs the exchange itself. */
async function startCallbackServer(
  callbackPath: string,
  verifier: string,
  signal: AbortSignal | undefined,
  timeoutMs: number,
  context: { readonly fetchImplementation?: OAuthFetch } = {},
): Promise<OpenRouterCallback> {
  if (signal?.aborted) throw new Error('Login cancelled')
  let resolveCredential: (credential: OAuthFlowCredential | undefined) => void = () => {}
  let rejectCredential: (error: Error) => void = () => {}
  const credential = new Promise<OAuthFlowCredential | undefined>((resolve, reject) => {
    resolveCredential = resolve
    rejectCredential = reject
  })
  let claimed = false
  let settled = false
  let server: ReturnType<typeof Bun.serve>
  let timeout: ReturnType<typeof setTimeout> | undefined
  let onAbort: (() => void) | undefined

  const finish = (result: { credential: OAuthFlowCredential | undefined } | { error: Error }): void => {
    if (settled) return
    settled = true
    close()
    if ('credential' in result) resolveCredential(result.credential)
    else rejectCredential(result.error)
  }

  const close = (): void => {
    if (timeout) clearTimeout(timeout)
    if (onAbort && signal) signal.removeEventListener('abort', onAbort)
    void server.stop(true)
  }

  server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    fetch(request) {
      const requestUrl = new URL(request.url)
      if (requestUrl.pathname !== callbackPath) {
        return new Response(oauthCallbackHtml('error', 'OAuth callback route not found.'), {
          status: 404,
          headers: { 'Content-Type': 'text/html; charset=utf-8' },
        })
      }
      if (claimed || settled) {
        return new Response(oauthCallbackHtml('error', 'This OAuth callback has already been used.'), {
          status: 409,
          headers: { 'Content-Type': 'text/html; charset=utf-8' },
        })
      }
      const oauthError = requestUrl.searchParams.get('error')
      if (oauthError) {
        const description = requestUrl.searchParams.get('error_description') ?? oauthError
        finish({ error: new Error(`OpenRouter authorization failed: ${description}`) })
        return new Response(oauthCallbackHtml('error', 'OpenRouter authorization was denied.', description), {
          status: 400,
          headers: { 'Content-Type': 'text/html; charset=utf-8' },
        })
      }
      const code = requestUrl.searchParams.get('code')
      if (!code) {
        return new Response(oauthCallbackHtml('error', 'OpenRouter returned no authorization code.'), {
          status: 400,
          headers: { 'Content-Type': 'text/html; charset=utf-8' },
        })
      }
      claimed = true
      // Bun's fetch handler may return a promise; the exchange settles the
      // login exactly once (guarded by `claimed`) and then renders the page.
      return new Promise<Response>(resolveExchangePage => {
        exchangeOpenRouterAuthorizationCode(code, verifier, signal, context).then(
          result => {
            finish({ credential: result })
            resolveExchangePage(new Response(
              oauthCallbackHtml('success', 'Signed in to OpenRouter. You may now close this page.'),
              { headers: { 'Content-Type': 'text/html; charset=utf-8' } },
            ))
          },
          error => {
            const message = error instanceof Error ? error.message : 'Unknown token exchange error'
            finish({ error: error instanceof Error ? error : new Error(message) })
            resolveExchangePage(new Response(
              oauthCallbackHtml('error', 'OpenRouter key exchange failed.', message),
              { status: 502, headers: { 'Content-Type': 'text/html; charset=utf-8' } },
            ))
          },
        )
      })
    },
  })

  // Bun exposes the bound address as a URL; an ephemeral port resolves here.
  const port = Number(server.url.port)
  if (!port) {
    void server.stop(true)
    throw new Error('Could not determine the OpenRouter OAuth callback port')
  }

  if (signal) {
    onAbort = () => finish({ error: new Error('Login cancelled') })
    signal.addEventListener('abort', onAbort, { once: true })
  }
  timeout = setTimeout(() => finish({ error: new Error('OpenRouter OAuth login timed out') }), timeoutMs)
  if (signal?.aborted) {
    close()
    throw new Error('Login cancelled')
  }

  const callbackUrl = `http://127.0.0.1:${port}${callbackPath}`
  return {
    callbackUrl,
    close,
    // A claimed callback is already exchanging its code; let that settle.
    cancelWait: () => {
      if (!claimed) finish({ credential: undefined })
    },
    waitForCredential: () => credential,
  }
}

/**
 * Owns the stored OpenRouter credential: PKCE browser login against the
 * ephemeral loopback callback, identity refresh, persistence under
 * `<xerxesHome>/auth/`.
 */
export class OpenRouterOAuthSession {
  private readonly home: string
  private readonly now: () => number
  private pending: Promise<OAuthFlowCredential> | undefined

  constructor(options: {
    readonly xerxesHome?: string
    readonly now?: () => number
  } = {}) {
    this.home = options.xerxesHome ?? defaultXerxesHome(process.env)
    this.now = options.now ?? (() => Date.now() / 1_000)
  }

  /**
   * Return the credential. OpenRouter keys never expire, so resolution is a
   * plain read — but a stored credential with `refresh` cleared still
   * resolves, matching pi-ai's identity refresh.
   */
  async credential(signal?: AbortSignal): Promise<OAuthFlowCredential> {
    signal?.throwIfAborted()
    if (this.pending) return this.pending
    const stored = await this.stored()
    if (stored) return stored
    throw new ConfigurationError(
      'openrouter_oauth',
      "No OpenRouter OAuth session found. Run 'xerxes auth login openrouter'.",
    )
  }

  /** Identity refresh: the stored key is the credential (pi-ai parity). */
  async refresh(credential: OAuthFlowCredential, _signal?: AbortSignal): Promise<OAuthFlowCredential> {
    return credential
  }

  /** Run the PKCE browser flow (with manual-paste fallback) and persist. */
  async login(options: OpenRouterLoginOptions = {}): Promise<OAuthFlowCredential> {
    const { verifier, challenge } = await generatePkceS256()
    const callbackPath = `/oauth/callback/${crypto.randomUUID()}`
    const timeoutMs = options.timeoutMs ?? OPENROUTER_LOGIN_TIMEOUT_MS
    const callback = await startCallbackServer(callbackPath, verifier, options.signal, timeoutMs, {
      ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
    })

    const authorizeUrl = new URL(OPENROUTER_AUTHORIZE_URL)
    authorizeUrl.search = new URLSearchParams({
      callback_url: callback.callbackUrl,
      code_challenge: challenge,
      code_challenge_method: 'S256',
    }).toString()

    let manualInput: string | undefined
    let manualError: Error | undefined
    const manualPromise = options.manualInput?.().then(
      input => {
        manualInput = input
        callback.cancelWait()
      },
      error => {
        manualError = error instanceof Error ? error : new Error(String(error))
        callback.cancelWait()
      },
    )

    try {
      ;(options.openUrl ?? openInBrowser)(authorizeUrl.toString())
      const credential = await callback.waitForCredential()
      if (manualError) throw manualError
      if (credential) {
        await this.persist(credential)
        return credential
      }
      await manualPromise
      if (manualError) throw manualError
      const code = manualInput ? parseOpenRouterAuthorizationInput(manualInput) : undefined
      if (!code) throw new Error('Missing authorization code')
      const manual = await exchangeOpenRouterAuthorizationCode(code, verifier, options.signal, {
        ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
      })
      await this.persist(manual)
      return manual
    } finally {
      void manualPromise
      callback.close()
    }
  }

  /** The persisted credential, for `auth status`. */
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

  private credentialPath(): string {
    return join(this.home, 'auth', 'openrouter-oauth.json')
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
    const access = record && typeof record['access'] === 'string' ? record['access'] : undefined
    const refresh = record && typeof record['refresh'] === 'string' ? record['refresh'] : undefined
    const expires = record && typeof record['expires'] === 'number' ? record['expires'] : undefined
    if (!access || expires === undefined) return undefined
    return { access, refresh: refresh ?? '', expires }
  }

  private async persist(credential: OAuthFlowCredential): Promise<void> {
    const path = this.credentialPath()
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, `${JSON.stringify(credential, null, 2)}\n`, 'utf8')
  }
}
