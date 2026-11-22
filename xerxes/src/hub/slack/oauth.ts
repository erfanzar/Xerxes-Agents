// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomBytes } from 'node:crypto'
import { SlackTransportError, type SlackIdentity } from './types.js'
import { identity, record, teamId, userId, userToken } from './wire.js'

import { USER_SCOPES } from './manifest.js'

export { USER_SCOPES }
const STATE_TTL_MS = 10 * 60_000
const MAX_PENDING = 256

export interface SlackOAuthConfig {
  readonly clientId: string
  readonly clientSecret: string
  readonly redirectUri: string
}
export interface SlackOAuthStart {
  readonly authorizeUrl: string
  readonly expiresAt: number
}
export interface SlackOAuthCallback {
  readonly state: string
  readonly code?: string
  readonly error?: string
}

function bindingKey(binding: string): string {
  if (typeof binding !== 'string' || !binding || binding.length > 1024) throw new SlackTransportError('invalid_request')
  return createHash('sha256').update(binding).digest('hex')
}

export class SlackOAuthState {
  readonly #config: SlackOAuthConfig | undefined
  readonly #pending = new Map<string, { binding: string; expiresAt: number; expected?: SlackIdentity }>()

  constructor(config: SlackOAuthConfig | undefined, private readonly now: () => number) {
    if (!config) return
    try {
      const uri = new URL(config.redirectUri)
      if (!/^[A-Za-z0-9._-]{1,256}$/u.test(config.clientId) || !/^[A-Za-z0-9._-]{1,256}$/u.test(config.clientSecret)
        || !config.redirectUri.startsWith('https://') || uri.protocol !== 'https:' || uri.username || uri.password
        || /[\s\\?#]/u.test(config.redirectUri) || config.redirectUri.length > 2048) throw new Error()
    } catch { throw new SlackTransportError('invalid_request') }
    this.#config = { ...config }
  }

  begin(binding: string, expected?: SlackIdentity): SlackOAuthStart {
    if (!this.#config) throw new SlackTransportError('oauth_unsupported')
    const key = bindingKey(binding)
    const account = expected === undefined ? undefined : identity(expected)
    const now = this.now()
    for (const [state, entry] of this.#pending) if (entry.expiresAt <= now) this.#pending.delete(state)
    if (this.#pending.size >= MAX_PENDING) throw new SlackTransportError('invalid_state')
    const state = randomBytes(32).toString('base64url')
    const expiresAt = now + STATE_TTL_MS
    this.#pending.set(state, { binding: key, expiresAt, ...(account ? { expected: account } : {}) })
    const url = new URL('https://slack.com/oauth/v2/authorize')
    url.search = new URLSearchParams({
      client_id: this.#config.clientId, redirect_uri: this.#config.redirectUri, user_scope: USER_SCOPES.join(','), state,
    }).toString()
    return { authorizeUrl: url.toString(), expiresAt }
  }

  consume(binding: string, callback: SlackOAuthCallback): { body: URLSearchParams; authorization: string; expected?: SlackIdentity } {
    if (!this.#config) throw new SlackTransportError('oauth_unsupported')
    const key = bindingKey(binding)
    if (typeof callback?.state !== 'string' || !/^[A-Za-z0-9_-]{43}$/u.test(callback.state)) throw new SlackTransportError('invalid_state')
    const entry = this.#pending.get(callback.state)
    if (!entry || entry.binding !== key) throw new SlackTransportError('invalid_state')
    // Consume before parsing the code or awaiting I/O, including explicit denials and failed exchanges.
    this.#pending.delete(callback.state)
    if (entry.expiresAt <= this.now()) throw new SlackTransportError('invalid_state')
    if (callback.error !== undefined) throw new SlackTransportError('oauth_denied')
    if (typeof callback.code !== 'string' || !/^[\x21-\x7e]{1,2048}$/u.test(callback.code)) throw new SlackTransportError('invalid_request')
    const { clientId, clientSecret, redirectUri } = this.#config
    return {
      body: new URLSearchParams({ code: callback.code, redirect_uri: redirectUri }),
      authorization: `Basic ${Buffer.from(`${clientId}:${clientSecret}`).toString('base64')}`,
      ...(entry.expected ? { expected: entry.expected } : {}),
    }
  }
}

export interface SlackUserGrant { readonly account: SlackIdentity; readonly token: string }

export function parseScopedUserGrant(team: unknown, user: unknown, token: unknown, scope: unknown): SlackUserGrant {
  if (!teamId(team) || !userId(user) || !userToken(token) || typeof scope !== 'string') throw new SlackTransportError('oauth_unsupported')
  const granted = new Set(scope.split(',').map(value => value.trim()))
  if (granted.size !== USER_SCOPES.length || USER_SCOPES.some(value => !granted.has(value))) throw new SlackTransportError('oauth_unsupported')
  return { account: { teamId: team, userId: user }, token }
}

export function parseUserGrant(data: Record<string, unknown>): SlackUserGrant {
  const team = record(data.team)
  const user = record(data.authed_user)
  if (data.is_enterprise_install === true || data.access_token !== undefined || user?.token_type !== 'user'
    || user.refresh_token !== undefined || user.expires_in !== undefined) throw new SlackTransportError('oauth_unsupported')
  return parseScopedUserGrant(team?.id, user.id, user.access_token, user.scope)
}
