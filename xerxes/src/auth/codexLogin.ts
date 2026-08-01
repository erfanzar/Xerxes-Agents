// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Interactive ChatGPT sign-in: authorization code + PKCE over a loopback
 * callback, the same shape the open-source Codex CLI uses.
 *
 * The callback listener is bound to 127.0.0.1 rather than 0.0.0.0 so the
 * authorization code — which is bearer-equivalent until it is exchanged — is
 * never reachable from another host on the network.
 */

import { spawn } from 'node:child_process'
import { timingSafeEqual } from 'node:crypto'

import { ProviderError } from '../core/errors.js'
import {
  buildAuthorizeUrl,
  exchangeCode,
  generatePkcePair,
  OAuthToken,
  type OAuthFetch,
} from '../mcp/oauth.js'
import { CODEX_PROVIDER, codexOAuthConfig } from './codexAuth.js'

/** Port the Codex OAuth application has registered for its loopback callback. */
export const CODEX_CALLBACK_PORT = 1455
export const CODEX_CALLBACK_PATH = '/auth/callback'

/** Give the user a real chance to complete a browser sign-in before giving up. */
export const CODEX_LOGIN_TIMEOUT_MS = 10 * 60 * 1_000

export interface CodexLoginOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly fetchImplementation?: OAuthFetch
  /** Called with the authorize URL; defaults to opening the default browser. */
  readonly openUrl?: (url: string) => void
  readonly port?: number
  readonly signal?: AbortSignal
  readonly timeoutMs?: number
}

export interface CodexLoginResult {
  readonly token: OAuthToken
}

/** The redirect URI registered for the Codex OAuth application. */
export function codexRedirectUri(port = CODEX_CALLBACK_PORT): string {
  return `http://localhost:${port}${CODEX_CALLBACK_PATH}`
}

/**
 * Run the full browser sign-in and return the resulting session.
 *
 * Resolves only after the authorization code has been exchanged, so a caller
 * that awaits this has a token it can persist immediately.
 */
export async function loginWithChatGpt(options: CodexLoginOptions = {}): Promise<CodexLoginResult> {
  const port = options.port ?? CODEX_CALLBACK_PORT
  const config = codexOAuthConfig(codexRedirectUri(port), options.environment ?? process.env)
  const { verifier, challenge } = generatePkcePair()
  const state = crypto.randomUUID()

  const authorizeUrl = withCodexAuthorizeParams(
    buildAuthorizeUrl(config, { state, codeChallenge: challenge }),
  )

  const callback = awaitAuthorizationCode({
    expectedState: state,
    port,
    ...(options.signal ? { signal: options.signal } : {}),
    timeoutMs: options.timeoutMs ?? CODEX_LOGIN_TIMEOUT_MS,
  })

  ;(options.openUrl ?? openInBrowser)(authorizeUrl)

  const code = await callback
  const token = await exchangeCode(config, {
    code,
    codeVerifier: verifier,
    ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
    ...(options.signal ? { signal: options.signal } : {}),
  })
  return { token }
}

/**
 * Add the parameters the Codex authorization flow expects on top of the
 * standard PKCE set: organization claims in the id_token, and the simplified
 * consent screen the CLI is approved for.
 */
function withCodexAuthorizeParams(url: string): string {
  const parsed = new URL(url)
  parsed.searchParams.set('id_token_add_organizations', 'true')
  parsed.searchParams.set('codex_cli_simplified_flow', 'true')
  return parsed.toString()
}

interface CallbackOptions {
  readonly expectedState: string
  readonly port: number
  readonly signal?: AbortSignal
  readonly timeoutMs: number
}

/** Serve the loopback callback exactly until one authorization code arrives. */
function awaitAuthorizationCode(options: CallbackOptions): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    let settled = false
    const server = Bun.serve({
      hostname: '127.0.0.1',
      port: options.port,
      fetch(request) {
        const url = new URL(request.url)
        if (url.pathname !== CODEX_CALLBACK_PATH) {
          return new Response('not found', { status: 404 })
        }

        const error = url.searchParams.get('error')
        if (error) {
          const description = url.searchParams.get('error_description') ?? ''
          finish(undefined, new ProviderError(CODEX_PROVIDER, `ChatGPT sign-in failed: ${error} ${description}`.trim()))
          return htmlResponse('Sign-in failed. You can close this tab and try again.', 400)
        }

        const code = url.searchParams.get('code') ?? ''
        const state = url.searchParams.get('state') ?? ''
        // A mismatched state is a cross-site request forgery attempt against
        // the callback, so the code that came with it is not ours to use.
        if (!constantTimeEquals(state, options.expectedState)) {
          finish(undefined, new ProviderError(CODEX_PROVIDER, 'ChatGPT sign-in returned a mismatched OAuth state'))
          return htmlResponse('Sign-in state mismatch. Nothing was saved.', 400)
        }
        if (!code) {
          finish(undefined, new ProviderError(CODEX_PROVIDER, 'ChatGPT sign-in returned no authorization code'))
          return htmlResponse('Sign-in returned no authorization code.', 400)
        }

        finish(code, undefined)
        return htmlResponse('Signed in to Xerxes. You can close this tab.', 200)
      },
    })

    const timer = setTimeout(() => {
      finish(undefined, new ProviderError(CODEX_PROVIDER, 'ChatGPT sign-in timed out before the browser returned'))
    }, options.timeoutMs)

    const onAbort = () => {
      finish(undefined, new ProviderError(CODEX_PROVIDER, 'ChatGPT sign-in was cancelled'))
    }
    options.signal?.addEventListener('abort', onAbort, { once: true })

    function finish(code: string | undefined, error: Error | undefined): void {
      if (settled) return
      settled = true
      clearTimeout(timer)
      options.signal?.removeEventListener('abort', onAbort)
      // Stop listening before settling: the loopback port stays bound until
      // the server is closed, and a second `xerxes auth login` would then
      // fail to bind rather than starting a fresh flow.
      void server.stop(true)
      if (error) reject(error)
      else resolve(code as string)
    }
  })
}

function htmlResponse(message: string, status: number): Response {
  return new Response(
    `<!doctype html><meta charset="utf-8"><title>Xerxes</title>`
    + `<body style="font:16px system-ui;padding:3rem;text-align:center">${message}</body>`,
    { status, headers: { 'Content-Type': 'text/html; charset=utf-8' } },
  )
}

function constantTimeEquals(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, 'utf8')
  const rightBytes = Buffer.from(right, 'utf8')
  // timingSafeEqual throws on a length mismatch, which is itself the answer.
  return leftBytes.byteLength === rightBytes.byteLength && timingSafeEqual(leftBytes, rightBytes)
}

/**
 * Open the authorize URL in the default browser.
 *
 * Spawned detached with stdio ignored so the browser outlives the CLI process
 * and never writes into a terminal the caller may have in raw mode.
 */
function openInBrowser(url: string): void {
  const command = process.platform === 'darwin'
    ? { command: 'open', args: [] as string[] }
    : process.platform === 'win32'
      ? { command: 'cmd', args: ['/c', 'start', ''] }
      : { command: 'xdg-open', args: [] as string[] }
  try {
    const child = spawn(command.command, [...command.args, url], { detached: true, stdio: 'ignore' })
    // A missing opener surfaces asynchronously; an unhandled 'error' on a
    // ChildProcess would take the process down mid-login.
    child.on('error', () => {})
    child.unref()
  } catch {
    // Non-fatal: the caller prints the URL for manual opening.
  }
}
