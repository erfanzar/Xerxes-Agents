// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isRecord } from '../hub/protocol.js'

/**
 * Same-origin HTTP calls the phone makes outside the WebSocket: session
 * refresh, pairing, push registration. Cookies are httpOnly and never read
 * here; the CSRF token comes back in the refresh response body and is sent in
 * `x-xerxes-csrf` on mutating requests. No token ever goes into a URL.
 */

export const HUB_HTTP = Object.freeze({
  refresh: '/api/hub/v1/session/refresh',
  pairConsume: '/api/hub/v1/pair/consume',
  pushVapid: '/api/hub/v1/push/vapid-key',
  pushSubscribe: '/api/hub/v1/push/subscribe',
})

export type FetchLike = (input: string, init: RequestInit) => Promise<Response>

export interface HttpResult {
  readonly status: number
  readonly body: Readonly<Record<string, unknown>>
}

export type RefreshOutcome =
  | { readonly status: 'ok'; readonly csrfToken?: string }
  /** 202: the device exists but the owner has not confirmed it yet. */
  | { readonly status: 'pending' }
  /** 401 only. `reason` is the Hub's refusal code (`no_session`: this browser was never paired). */
  | { readonly status: 'denied'; readonly reason?: string }
  | { readonly status: 'unavailable'; readonly message: string }

export type ConsumeOutcome =
  /** `active` is true when the Hub needs no desktop confirmation (the cookies are already usable). */
  | { readonly status: 'accepted'; readonly active: boolean; readonly comparisonCode?: string }
  | { readonly status: 'rejected'; readonly message: string }
  | { readonly status: 'unavailable'; readonly message: string }

export interface PairingBody {
  readonly pairing_id: string
  readonly code: string
  readonly device_label: string
}

export interface HubHttp {
  csrf(): string | undefined
  refresh(): Promise<RefreshOutcome>
  consumePairing(body: PairingBody): Promise<ConsumeOutcome>
  json(method: 'GET' | 'POST' | 'DELETE', path: string, body?: Readonly<Record<string, unknown>>): Promise<HttpResult>
}

async function readBody(response: Response): Promise<Readonly<Record<string, unknown>>> {
  const text = await response.text()
  if (text.length === 0) return {}
  try {
    const value = JSON.parse(text) as unknown
    return isRecord(value) ? value : {}
  } catch {
    // A non-JSON body (an HTML error page from a proxy) carries no fields we can use.
    return {}
  }
}

/** Identity routes answer `{error:{code,message}}`; other routes may use a top-level `message`. */
function messageOf(body: Readonly<Record<string, unknown>>, fallback: string): string {
  const nested = isRecord(body.error) && typeof body.error.message === 'string' ? body.error.message : undefined
  if (nested !== undefined && nested.length > 0) return nested
  return typeof body.message === 'string' && body.message.length > 0 ? body.message : fallback
}

export function createHubHttp(fetchFn: FetchLike): HubHttp {
  let csrfToken: string | undefined

  async function json(method: 'GET' | 'POST' | 'DELETE', path: string, body?: Readonly<Record<string, unknown>>): Promise<HttpResult> {
    const headers: Record<string, string> = { accept: 'application/json' }
    if (body !== undefined) headers['content-type'] = 'application/json'
    if (method !== 'GET' && csrfToken !== undefined) headers['x-xerxes-csrf'] = csrfToken
    const response = await fetchFn(path, {
      method,
      headers,
      credentials: 'same-origin',
      cache: 'no-store',
      referrerPolicy: 'no-referrer',
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    })
    return { status: response.status, body: await readBody(response) }
  }

  return {
    csrf: () => csrfToken,
    json,
    async refresh() {
      let result: HttpResult
      try {
        result = await json('POST', HUB_HTTP.refresh, {})
      } catch (error) {
        return { status: 'unavailable', message: error instanceof Error ? error.message : 'Network error' }
      }
      // Only 401 means the sign-in is gone. A 403 here is an Origin-header problem (a proxy or an
      // extension), not a revoked device: signing out for it locked phones out for no reason.
      if (result.status === 401) {
        const reason = isRecord(result.body.error) && typeof result.body.error.reason === 'string' ? result.body.error.reason : undefined
        return reason === undefined ? { status: 'denied' } : { status: 'denied', reason }
      }
      if (result.status === 403) return { status: 'unavailable', message: messageOf(result.body, 'The Hub refused this request (403). Retrying.') }
      if (result.status === 202) return { status: 'pending' }
      if (result.status >= 200 && result.status < 300) {
        const token = typeof result.body.csrf_token === 'string' && result.body.csrf_token.length > 0 ? result.body.csrf_token : undefined
        if (token !== undefined) csrfToken = token
        return token === undefined ? { status: 'ok' } : { status: 'ok', csrfToken: token }
      }
      return { status: 'unavailable', message: messageOf(result.body, `The Hub answered ${result.status}.`) }
    },
    async consumePairing(body) {
      let result: HttpResult
      try {
        result = await json('POST', HUB_HTTP.pairConsume, { ...body })
      } catch (error) {
        return { status: 'unavailable', message: error instanceof Error ? error.message : 'Network error' }
      }
      if (result.status >= 200 && result.status < 300) {
        const code = typeof result.body.comparison_code === 'string' && result.body.comparison_code.length > 0 ? result.body.comparison_code : undefined
        const active = result.body.status === 'active'
        return code === undefined ? { status: 'accepted', active } : { status: 'accepted', active, comparisonCode: code }
      }
      if (result.status >= 400 && result.status < 500 && result.status !== 429) {
        return { status: 'rejected', message: messageOf(result.body, 'The pairing code was not accepted. Create a new one on the desktop and scan it again.') }
      }
      return { status: 'unavailable', message: messageOf(result.body, `The Hub answered ${result.status}. Try again in a moment.`) }
    },
  }
}
