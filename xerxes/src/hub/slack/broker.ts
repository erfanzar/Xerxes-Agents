// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomBytes } from 'node:crypto'
import { checkAbort, parseSlackJson, type SlackFetch } from './http.js'
import { parseScopedUserGrant, type SlackUserGrant } from './oauth.js'
import { SlackTransportError } from './types.js'

export const SLACK_BROKER_ORIGIN_ENV = 'XERXES_HUB_SLACK_BROKER_ORIGIN'
export const SLACK_BROKER_ENROLLMENT_ENV = 'XERXES_HUB_SLACK_BROKER_ENROLLMENT_TOKEN'
const AUTHORIZATION_PATH = '/v1/slack/authorize'
const FETCH_TIMEOUT_MS = 10_000

export function readSlackBrokerOrigin(input: string): string {
  try {
    if (input.length > 2048 || !/^https:\/\/[^/?#\\\s]+\/?$/u.test(input)) throw new Error()
    const url = new URL(input)
    if (url.username || url.password || input.includes('@')) throw new Error()
    return url.origin
  } catch { throw new Error(`${SLACK_BROKER_ORIGIN_ENV} must be a trusted HTTPS origin without a path, credentials, query, or fragment.`) }
}

export type SlackBrokerAttempt = { readonly id: string; readonly verifier: string }
type SlackBrokerResult = { readonly status: 'pending' | 'denied' | 'expired' }
  | { readonly status: 'ready'; readonly grant: SlackUserGrant }

export class SlackBroker {
  readonly origin: string
  readonly #enrollmentToken: string

  constructor(origin: string, enrollmentToken: string, private readonly fetchPort: SlackFetch = (url, init) => fetch(url, init)) {
    this.origin = readSlackBrokerOrigin(origin)
    if (!/^[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]$/u.test(enrollmentToken)) {
      throw new Error(`${SLACK_BROKER_ENROLLMENT_ENV} must be 32 random bytes encoded as unpadded base64url.`)
    }
    this.#enrollmentToken = enrollmentToken
  }

  async start(signal: AbortSignal): Promise<{ attempt: SlackBrokerAttempt; authorizeUrl: string; pairingCode?: string }> {
    const verifier = randomBytes(32).toString('base64url')
    const result = await this.#post('start', { challenge: createHash('sha256').update(verifier).digest('base64url') }, signal)
    if (typeof result.id !== 'string' || !/^[A-Za-z0-9_-]{1,256}$/u.test(result.id)
      || typeof result.browser_url !== 'string' || result.browser_url.length > 4096 || /[\s\\]/u.test(result.browser_url)) {
      throw new SlackTransportError('invalid_response')
    }
    let url: URL
    try { url = new URL(result.browser_url) } catch { throw new SlackTransportError('invalid_response') }
    const decoded = decodeURIComponentSafe(url.href)
    const pairingCode = result.pairing_code
    if (url.origin !== this.origin || url.pathname !== AUTHORIZATION_PATH || url.username || url.password || url.hash
      || decoded.includes(verifier) || result.id.includes(verifier)
      || decoded.includes(this.#enrollmentToken) || result.id.includes(this.#enrollmentToken)
      || (pairingCode !== undefined && (typeof pairingCode !== 'string' || !/^[A-Za-z0-9_-]{22}$/u.test(pairingCode)
        || decoded.includes(pairingCode) || result.id.includes(pairingCode)))) throw new SlackTransportError('invalid_response')
    return { attempt: { id: result.id, verifier }, authorizeUrl: url.href, ...(pairingCode === undefined ? {} : { pairingCode }) }
  }

  async redeem(attempt: SlackBrokerAttempt, signal: AbortSignal): Promise<SlackBrokerResult> {
    const result = await this.#post('redeem', attempt, signal)
    switch (result.status) {
      case 'pending': case 'denied': case 'expired': return { status: result.status }
      case 'ready': return { status: 'ready', grant: parseScopedUserGrant(result.team_id, result.user_id, result.access_token, result.scope) }
      default: throw new SlackTransportError('invalid_response')
    }
  }

  async #post(path: 'start' | 'redeem', body: Record<string, string>, signal: AbortSignal): Promise<Record<string, unknown>> {
    checkAbort(signal)
    const deadline = AbortSignal.timeout(FETCH_TIMEOUT_MS)
    const requestSignal = AbortSignal.any([signal, deadline])
    const unavailable = () => new SlackTransportError(signal.aborted ? 'aborted' : deadline.aborted ? 'timeout' : 'network_error')
    let cancel: () => void = () => {}
    const aborted = new Promise<never>((_, reject) => {
      cancel = () => reject(unavailable())
      requestSignal.addEventListener('abort', cancel, { once: true })
    })
    const exchange = async () => {
      let response: Response
      try {
        response = await this.fetchPort(`${this.origin}/v1/slack/${path}`, { method: 'POST', redirect: 'error', signal: requestSignal,
          headers: { 'content-type': 'application/json', accept: 'application/json', authorization: `Bearer ${this.#enrollmentToken}` },
          body: JSON.stringify(body) })
      } catch { throw unavailable() }
      if (requestSignal.aborted || response.redirected || !response.ok) {
        void response.body?.cancel().catch(() => {})
        if (!requestSignal.aborted && !response.redirected && response.status === 429) {
          const delay = response.headers.get('retry-after') ?? ''
          throw new SlackTransportError('rate_limited', /^\d{1,8}$/u.test(delay) ? Number(delay) : undefined)
        }
        throw unavailable()
      }
      return parseSlackJson(response, requestSignal, 16 * 1024)
    }
    try { return await Promise.race([aborted, exchange()]) }
    finally { requestSignal.removeEventListener('abort', cancel) }
  }
}

function decodeURIComponentSafe(url: string): string {
  try { return decodeURIComponent(url) } catch { throw new SlackTransportError('invalid_response') }
}
