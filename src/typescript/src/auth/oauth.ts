// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomBytes } from 'node:crypto'

import {
  buildAuthorizeUrl,
  exchangeCode,
  generatePkcePair,
  refreshToken,
  type OAuthConfig,
  type OAuthFetch,
  type OAuthFetchOptions,
  OAuthToken,
} from '../mcp/oauth.js'

export { OAuthToken }
export type { OAuthConfig }

export interface AuthorizeContext {
  readonly codeVerifier: string
  readonly state: string
  readonly url: string
}

export interface OAuthClientOptions {
  readonly fetchImplementation?: OAuthFetch
  readonly now?: () => number
  readonly stateGenerator?: () => string
}

/** Authorization-code-with-PKCE client bound to one provider configuration. */
export class OAuthClient {
  readonly config: OAuthConfig

  private readonly fetchImplementation: OAuthFetch | undefined
  private readonly now: (() => number) | undefined
  private readonly stateGenerator: () => string

  constructor(config: OAuthConfig, options: OAuthClientOptions = {}) {
    this.config = config
    this.fetchImplementation = options.fetchImplementation
    this.now = options.now
    this.stateGenerator = options.stateGenerator ?? defaultState
  }

  /** Start an OAuth flow and return the browser URL plus callback-verification material. */
  beginAuthorize(): AuthorizeContext {
    const { verifier, challenge } = generatePkcePair()
    const state = this.stateGenerator()
    if (!state) {
      throw new Error('OAuth state generator returned an empty state')
    }
    return {
      url: buildAuthorizeUrl(this.config, { state, codeChallenge: challenge }),
      state,
      codeVerifier: verifier,
    }
  }

  /** Exchange an authorization code after the caller has verified its callback state. */
  async finishAuthorize(code: string, codeVerifier: string, signal?: AbortSignal): Promise<OAuthToken> {
    return exchangeCode(this.config, {
      code,
      codeVerifier,
      ...this.requestOptions(signal),
    })
  }

  /** Refresh a provider token. */
  async refresh(token: OAuthToken, signal?: AbortSignal): Promise<OAuthToken> {
    return refreshToken(this.config, token, this.requestOptions(signal))
  }

  private requestOptions(signal?: AbortSignal): OAuthFetchOptions {
    return {
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
      ...(this.now ? { now: this.now } : {}),
      ...(signal ? { signal } : {}),
    }
  }
}

/** OAuth configuration for the standard GitHub user-token flow. */
export function githubPatPreset(clientId: string): OAuthConfig {
  return {
    clientId,
    authorizeUrl: 'https://github.com/login/oauth/authorize',
    tokenUrl: 'https://github.com/login/oauth/access_token',
    scopes: ['read:user'],
  }
}

/** OAuth configuration for OpenAI's authorization-code flow. */
export function openaiPreset(clientId: string): OAuthConfig {
  return {
    clientId,
    authorizeUrl: 'https://auth.openai.com/oauth/authorize',
    tokenUrl: 'https://auth.openai.com/oauth/token',
    scopes: ['openid', 'profile'],
  }
}

/** OAuth configuration for Anthropic / Claude Code credential bridging. */
export function anthropicPreset(clientId: string): OAuthConfig {
  return {
    clientId,
    authorizeUrl: 'https://console.anthropic.com/oauth/authorize',
    tokenUrl: 'https://console.anthropic.com/oauth/token',
    scopes: ['openid'],
  }
}

/** OAuth configuration for GitHub Copilot bearer-token issuance. */
export function copilotPreset(clientId: string): OAuthConfig {
  return {
    clientId,
    authorizeUrl: 'https://github.com/login/oauth/authorize',
    tokenUrl: 'https://github.com/login/oauth/access_token',
    scopes: ['copilot'],
  }
}

function defaultState(): string {
  return Buffer.from(randomBytes(24)).toString('base64url')
}
