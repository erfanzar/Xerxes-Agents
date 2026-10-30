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
  /**
   * One same-origin call. A 401 (or a stale-CSRF 403) is answered by one session refresh and one
   * retry, so a lapsed 5-minute browser session never surfaces as a wrong error. Throws
   * `HubHttpTimeout` when the Hub does not answer in time.
   */
  json(method: 'GET' | 'POST' | 'DELETE', path: string, body?: Readonly<Record<string, unknown>>): Promise<HttpResult>
  /** Refresh unless a session with more than `marginMs` (default 60 s) left is already held. */
  ensureFresh(marginMs?: number): Promise<RefreshOutcome>
  /** Local-clock expiry of the current browser session, when one is known. */
  sessionExpiresAt(): number | undefined
  /**
   * Keep the browser session alive: refresh at about 70% of each session's lifetime while
   * `active()` holds (a hidden page skips it; the next `ensureFresh` on resume catches up).
   * Returns a stop function.
   */
  keepAlive(active: () => boolean): () => void
}

export interface HubHttpTimers {
  setTimeout(callback: () => void, ms: number): unknown
  clearTimeout(handle: unknown): void
}

export interface HubHttpOptions {
  /** Per-request deadline. Default 8 s. */
  readonly timeoutMs?: number
  readonly now?: () => number
  readonly timers?: HubHttpTimers
}

/** The Hub did not answer within the request deadline. */
export class HubHttpTimeout extends Error {
  constructor() {
    super('The Hub did not answer. Check the connection and try again.')
    this.name = 'HubHttpTimeout'
  }
}

const DEFAULT_TIMEOUT_MS = 8_000
/** The Hub's browser session lifetime; used when the reply's expiry is missing or the clocks disagree. */
const DEFAULT_SESSION_MS = 5 * 60_000
const KEEPALIVE_FRACTION = 0.7
const KEEPALIVE_RETRY_MS = 30_000
/** Refresh refusals that a concurrent winner in the same cookie jar can explain (see refreshOnce). */
const RACE_REFUSALS: ReadonlySet<string> = new Set(['spent_token', 'race'])
const RACE_RETRY_MS = 1_500

const defaultTimers: HubHttpTimers = {
  setTimeout: (callback, ms) => globalThis.setTimeout(callback, ms),
  clearTimeout: handle => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
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

/** A 403 whose message names the CSRF token: the token belonged to a session that has since lapsed. */
function staleCsrf(result: HttpResult): boolean {
  return result.status === 403 && /csrf/i.test(messageOf(result.body, ''))
}

/**
 * Remaining lifetime of a session the Hub just issued, by the phone's clock.
 * Phone and Hub clocks can disagree by minutes, so the result is bounded by
 * the Hub's normal session lifetime: refreshing early is harmless, late is not.
 */
function sessionLifetime(value: unknown, now: number): number {
  const at = typeof value === 'string' ? Date.parse(value) : Number.NaN
  const remaining = at - now
  return Number.isFinite(remaining) && remaining > 0 ? Math.min(remaining, DEFAULT_SESSION_MS) : DEFAULT_SESSION_MS
}

export function createHubHttp(fetchFn: FetchLike, options: HubHttpOptions = {}): HubHttp {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS
  const now = options.now ?? (() => Date.now())
  const timers = options.timers ?? defaultTimers
  let csrfToken: string | undefined
  let expiresAt: number | undefined
  let lifetimeMs = DEFAULT_SESSION_MS
  let inflight: Promise<RefreshOutcome> | undefined
  const sessionListeners = new Set<() => void>()

  /** One request under a deadline. The deadline also covers reading the body. */
  async function send(method: 'GET' | 'POST' | 'DELETE', path: string, body?: Readonly<Record<string, unknown>>): Promise<HttpResult> {
    const headers: Record<string, string> = { accept: 'application/json' }
    if (body !== undefined) headers['content-type'] = 'application/json'
    if (method !== 'GET' && csrfToken !== undefined) headers['x-xerxes-csrf'] = csrfToken
    const controller = new AbortController()
    let timer: unknown
    const deadline = new Promise<never>((_resolve, reject) => {
      timer = timers.setTimeout(() => {
        reject(new HubHttpTimeout())
        controller.abort()
      }, timeoutMs)
    })
    const attempt = (async (): Promise<HttpResult> => {
      const response = await fetchFn(path, {
        method,
        headers,
        credentials: 'same-origin',
        cache: 'no-store',
        referrerPolicy: 'no-referrer',
        signal: controller.signal,
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      })
      return { status: response.status, body: await readBody(response) }
    })()
    // When the deadline wins, the aborted fetch's own rejection is expected and dropped.
    attempt.catch(() => undefined)
    try {
      return await Promise.race([attempt, deadline])
    } finally {
      timers.clearTimeout(timer)
    }
  }

  async function json(method: 'GET' | 'POST' | 'DELETE', path: string, body?: Readonly<Record<string, unknown>>): Promise<HttpResult> {
    const first = await send(method, path, body)
    if (path === HUB_HTTP.refresh || path === HUB_HTTP.pairConsume) return first
    if (first.status !== 401 && !staleCsrf(first)) return first
    // The 5-minute browser session lapsed (a suspended phone): renew it once and retry once.
    const renewed = await refresh()
    if (renewed.status !== 'ok') return first
    return send(method, path, body)
  }

  async function refreshOnce(retryRace = true): Promise<RefreshOutcome> {
    let result: HttpResult
    try {
      result = await send('POST', HUB_HTTP.refresh, {})
    } catch (error) {
      return { status: 'unavailable', message: error instanceof Error ? error.message : 'Network error' }
    }
    // Only 401 means the sign-in is gone. A 403 here is an Origin-header problem (a proxy or an
    // extension), not a revoked device: signing out for it locked phones out for no reason.
    if (result.status === 401) {
      const reason = isRecord(result.body.error) && typeof result.body.error.reason === 'string' ? result.body.error.reason : undefined
      // A spent token or a lost race means another request of this cookie jar (a second tab, a
      // retried reply) rotated first and the jar may already hold its newer token. The Hub keeps
      // the cookies for these refusals, so one later retry reads the winner's token.
      if (retryRace && reason !== undefined && RACE_REFUSALS.has(reason)) {
        await new Promise<void>(resolve => { timers.setTimeout(resolve, RACE_RETRY_MS) })
        return refreshOnce(false)
      }
      csrfToken = undefined
      expiresAt = undefined
      return reason === undefined ? { status: 'denied' } : { status: 'denied', reason }
    }
    if (result.status === 403) return { status: 'unavailable', message: messageOf(result.body, 'The Hub refused this request (403). Retrying.') }
    if (result.status === 202) return { status: 'pending' }
    if (result.status >= 200 && result.status < 300) {
      const token = typeof result.body.csrf_token === 'string' && result.body.csrf_token.length > 0 ? result.body.csrf_token : undefined
      if (token !== undefined) csrfToken = token
      const at = now()
      lifetimeMs = sessionLifetime(result.body.session_expires_at, at)
      expiresAt = at + lifetimeMs
      for (const listener of [...sessionListeners]) listener()
      return token === undefined ? { status: 'ok' } : { status: 'ok', csrfToken: token }
    }
    return { status: 'unavailable', message: messageOf(result.body, `The Hub answered ${result.status}.`) }
  }

  /**
   * Concurrent refreshes share one request. Two parallel requests would both
   * present the same refresh token and rotate it twice.
   */
  function refresh(): Promise<RefreshOutcome> {
    if (inflight === undefined) {
      inflight = refreshOnce().finally(() => { inflight = undefined })
    }
    return inflight
  }

  return {
    csrf: () => csrfToken,
    json,
    refresh,
    sessionExpiresAt: () => expiresAt,
    async ensureFresh(marginMs = 60_000) {
      if (expiresAt !== undefined && expiresAt - now() > marginMs) {
        // A keep-alive tick skipped while hidden re-arms here (on resume): it refreshes at once
        // when the session is already past ~70% of its life.
        for (const listener of [...sessionListeners]) listener()
        return csrfToken === undefined ? { status: 'ok' } : { status: 'ok', csrfToken }
      }
      return refresh()
    },
    keepAlive(active) {
      let timer: unknown
      let stopped = false
      const arm = (ms: number): void => {
        if (timer !== undefined) timers.clearTimeout(timer)
        timer = timers.setTimeout(() => { timer = undefined; void tick() }, Math.max(1_000, ms))
      }
      const armFromSession = (): void => {
        if (stopped || expiresAt === undefined) return
        // Refresh at ~70% of the session's lifetime, measured from when it was issued.
        arm(expiresAt - now() - lifetimeMs * (1 - KEEPALIVE_FRACTION))
      }
      const tick = async (): Promise<void> => {
        if (stopped) return
        // Hidden: skip. The resume path calls ensureFresh, whose refresh re-arms this timer.
        if (!active()) return
        const outcome = await refresh()
        if (stopped) return
        // An ok refresh re-armed through the session listener; a denied one ends the device's sign-in.
        if (outcome.status === 'unavailable') arm(KEEPALIVE_RETRY_MS)
      }
      sessionListeners.add(armFromSession)
      armFromSession()
      return () => {
        stopped = true
        sessionListeners.delete(armFromSession)
        if (timer !== undefined) timers.clearTimeout(timer)
        timer = undefined
      }
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
