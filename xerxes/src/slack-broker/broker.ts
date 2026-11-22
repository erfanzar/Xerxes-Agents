// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, createHmac, randomBytes, timingSafeEqual } from 'node:crypto'
import { SlackHttp, type SlackFetch } from '../hub/slack/http.js'
import { parseUserGrant, SlackOAuthState } from '../hub/slack/oauth.js'
import { record } from '../hub/slack/wire.js'

const START = '/v1/slack/start'
const AUTHORIZE = '/v1/slack/authorize'
const CALLBACK = '/v1/slack/callback'
const REDEEM = '/v1/slack/redeem'
const TTL_MS = 10 * 60_000
const WINDOW_MS = 60_000
const MAX_FLOWS = 256
const MAX_BODY = 1024
const MAX_EXCHANGES = 8
const DIGEST = /^[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]$/u
const VERIFIER = /^[A-Za-z0-9._~-]{43,128}$/u

interface Grant {
  access_token: string
  team_id: string
  user_id: string
  scope: string
}
type Phase = { type: 'pair'; code: string }
  | { type: 'slack'; browser: string; state: string; oauth: SlackOAuthState; authorizeUrl: string }
  | { type: 'exchange'; browser: string }
  | { type: 'review' | 'ready'; browser: string; grant: Grant }
  | { type: 'denied'; browser: string }
interface Flow {
  readonly challenge: string
  readonly expires: number
  phase: Phase
}

const headers = {
  'cache-control': 'no-store', 'referrer-policy': 'no-referrer', 'x-content-type-options': 'nosniff',
  'content-security-policy': "default-src 'none'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
  'x-frame-options': 'DENY',
}
const json = (data: object, status = 200, extra?: Record<string, string>): Response => Response.json(data, { status, headers: { ...headers, ...extra } })
const escapeHtml = (value: string): string => value
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;')
  .replaceAll("'", '&#39;')
const page = (content: string, status = 200, extra?: Record<string, string>): Response => new Response(
  `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Connect Slack to Xerxes</title><main><h1>Connect Slack to Xerxes</h1>${content}</main></html>`,
  { status, headers: { ...headers, 'content-type': 'text/html; charset=utf-8', ...extra } },
)
// same-origin preserves the Origin header on same-origin form POSTs; no-referrer can make it null.
const formHeaders = { 'referrer-policy': 'same-origin' }
const invalid = (): Response => page('<p>This Slack authorization is invalid or expired. Return to Xerxes and start again.</p>', 400)
const random = (size: number): string => randomBytes(size).toString('base64url')
const sha = (value: string): string => createHash('sha256').update(value).digest('base64url')
const equal = (a: string, b: string): boolean => a.length === b.length && timingSafeEqual(Buffer.from(a), Buffer.from(b))
const matches = (pattern: RegExp, text: unknown): text is string => typeof text === 'string' && pattern.exec(text)?.[0] === text
const browserPath = (id: string): string => `${AUTHORIZE}?id=${id}`
const redirect = (location: string): Response => new Response(null, { status: 303, headers: { ...headers, location } })
const cookieName = (id: string): string => `__Host-xerxes-slack-${id}`
const cookie = (id: string, value: string, maxAge: number): string => `${cookieName(id)}=${value}; Secure; HttpOnly; SameSite=Lax; Path=/; Max-Age=${maxAge}`

async function body(request: Request): Promise<string | undefined> {
  const reader = request.body?.getReader()
  if (!reader) return undefined
  const decoder = new TextDecoder('utf-8', { fatal: true })
  let size = 0
  let text = ''
  try {
    while (true) {
      const next = await reader.read()
      if (next.done) return text + decoder.decode()
      size += next.value.byteLength
      if (size > MAX_BODY) return undefined
      text += decoder.decode(next.value, { stream: true })
    }
  } catch { return undefined } finally {
    void reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}

const type = (request: Request): string | undefined => request.headers.get('content-type')?.split(';')[0]?.trim().toLowerCase()
async function jsonBody(request: Request, keys: readonly string[]): Promise<Record<string, unknown> | undefined> {
  if (type(request) !== 'application/json' || request.headers.has('origin')) return undefined
  const raw = await body(request)
  if (raw === undefined) return undefined
  try {
    const data = record(JSON.parse(raw))
    if (data && Object.keys(data).length === keys.length && keys.every(key => Object.hasOwn(data, key))) return data
  } catch { /* malformed input */ }
  return undefined
}

export interface SlackBrokerOptions {
  readonly publicOrigin: string
  readonly clientId: string
  readonly clientSecret: string
  /** Operator-provisioned 32 random bytes encoded as unpadded base64url. */
  readonly enrollmentToken: string
  readonly fetch: SlackFetch
  readonly now?: () => number
}

export function createSlackBroker(options: SlackBrokerOptions): { handle(request: Request): Promise<Response>; sweep(): void } {
  let publicUrl: URL
  try {
    publicUrl = new URL(options.publicOrigin)
    if (publicUrl.protocol !== 'https:' || ![publicUrl.origin, `${publicUrl.origin}/`].includes(options.publicOrigin)) throw new Error()
  } catch { throw new Error('Slack broker requires a canonical public HTTPS origin.') }
  if (!matches(DIGEST, options.enrollmentToken)) throw new Error('Slack broker requires a valid enrollment token.')
  const enrolled = sha(options.enrollmentToken)
  const origin = publicUrl.origin
  const now = options.now ?? Date.now
  const config = { clientId: options.clientId, clientSecret: options.clientSecret, redirectUri: `${origin}${CALLBACK}` }
  // Validate the app credentials before the server starts accepting connections.
  new SlackOAuthState(config, now)
  const slack = new SlackHttp(options.fetch, 10_000)
  const key = randomBytes(32)
  const flows = new Map<string, Flow>()
  const states = new Map<string, string>()
  let exchanges = 0
  let windowEnd = 0, requests = 0, starts = 0

  const drop = (id: string, flow: Flow): void => {
    if (flow.phase.type === 'slack') states.delete(flow.phase.state)
    flows.delete(id)
  }
  const sweep = (): void => {
    const time = now()
    for (const [id, flow] of flows) if (flow.expires <= time) drop(id, flow)
  }
  const active = (id: string): Flow | undefined => {
    const flow = flows.get(id)
    if (!flow || flow.expires > now()) return flow
    drop(id, flow)
    return undefined
  }
  const sign = (id: string, nonce: string): string => createHmac('sha256', key).update(`${id}.${nonce}`).digest('base64url')
  const browser = (request: Request, id: string): string | undefined => {
    const name = `${cookieName(id)}=`
    const values = request.headers.get('cookie')?.split(';').map(part => part.trim()).filter(part => part.startsWith(name))
    if (values?.length !== 1) return undefined
    const [nonce, mac, extra] = values[0]?.slice(name.length).split('.') ?? []
    if (extra === undefined && matches(DIGEST, nonce) && matches(DIGEST, mac) && equal(sign(id, nonce), mac)) return nonce
    return undefined
  }

  async function start(request: Request): Promise<Response> {
    const data = await jsonBody(request, ['challenge'])
    if (!matches(DIGEST, data?.challenge)) return json({ error: 'invalid_request' }, 400)
    sweep()
    if (flows.size >= MAX_FLOWS) return json({ error: 'busy' }, 429, { 'retry-after': '60' })
    const id = random(32)
    const code = random(16)
    flows.set(id, { challenge: data.challenge, expires: now() + TTL_MS, phase: { type: 'pair', code: sha(code) } })
    return json({ id, browser_url: `${origin}${browserPath(id)}`, pairing_code: code })
  }

  async function authorize(request: Request, url: URL): Promise<Response> {
    const ids = url.searchParams.getAll('id')
    const id = ids[0]
    if (ids.length !== 1 || !matches(DIGEST, id)) return invalid()
    if (request.method === 'GET') {
      const flow = active(id)
      if (!flow) return invalid()
      const nonce = browser(request, id)
      const phase = flow.phase
      if (phase.type === 'pair') {
        const fresh = random(32)
        return page(`<p>Paste the pairing code shown in Xerxes Desktop. Do not use a code sent by someone else.</p><form method="post" action="${browserPath(id)}"><label>Pairing code <input name="code" required maxlength="22" autocomplete="off" autocapitalize="off" spellcheck="false"></label><button>Continue to Slack</button></form>`,
          200, { ...formHeaders, ...(nonce === undefined
            ? { 'set-cookie': cookie(id, `${fresh}.${sign(id, fresh)}`, Math.max(1, Math.ceil((flow.expires - now()) / 1000))) } : {}) })
      }
      if (phase.browser !== nonce) return invalid()
      // A link (rather than a cross-origin form redirect) works with form-action 'self'.
      if (phase.type === 'slack') return page(`<p><a rel="noreferrer" href="${escapeHtml(phase.authorizeUrl)}">Continue to Slack</a></p>`)
      if (phase.type === 'review') {
        return page(`<p>Slack authorized workspace <strong>${escapeHtml(phase.grant.team_id)}</strong> and user <strong>${escapeHtml(phase.grant.user_id)}</strong>. Verify these IDs before connecting this Slack account to Xerxes.</p><form method="post" action="${browserPath(id)}"><button name="decision" value="approve">Connect this account</button> <button name="decision" value="deny">Reject</button></form>`, 200, formHeaders)
      }
      return page(phase.type === 'denied' ? '<p>Slack authorization was not completed. Return to Xerxes.</p>'
        : phase.type === 'ready' ? '<p>Account approved. Return to Xerxes.</p>' : '<p>Slack authorization is in progress.</p>')
    }
    if (request.headers.get('origin') !== origin || type(request) !== 'application/x-www-form-urlencoded') return invalid()
    const nonce = browser(request, id)
    if (!nonce) return invalid()
    const raw = await body(request)
    const flow = active(id)
    if (raw === undefined || !flow) return invalid()
    const form = new URLSearchParams(raw)
    if (form.size !== 1) return invalid()
    const phase = flow.phase
    if (phase.type === 'pair') {
      const code = form.get('code')?.trim()
      if (!matches(/^[A-Za-z0-9_-]{22}$/u, code) || !equal(sha(code), phase.code)) return invalid()
      const oauth = new SlackOAuthState(config, now)
      const { authorizeUrl } = oauth.begin(nonce)
      const state = new URL(authorizeUrl).searchParams.get('state')
      if (!state) return invalid()
      flow.phase = { type: 'slack', browser: nonce, state, oauth, authorizeUrl }
      states.set(state, id)
      return redirect(browserPath(id))
    }
    if (phase.type !== 'review' || phase.browser !== nonce) return invalid()
    const decision = form.get('decision')
    if (decision !== 'approve' && decision !== 'deny') return invalid()
    flow.phase = decision === 'approve' ? { ...phase, type: 'ready' } : { type: 'denied', browser: nonce }
    return page(decision === 'approve' ? '<p>Account approved. Return to Xerxes.</p>' : '<p>Account rejected. Return to Xerxes.</p>',
      200, { 'set-cookie': cookie(id, '', 0) })
  }

  async function callback(request: Request, url: URL): Promise<Response> {
    const supplied = url.searchParams.getAll('state')
    const state = supplied[0]
    if (supplied.length !== 1 || !matches(DIGEST, state)) return invalid()
    const id = states.get(state)
    if (id === undefined) return invalid()
    const flow = active(id)
    const phase = flow?.phase
    if (!flow || phase?.type !== 'slack' || browser(request, id) !== phase.browser) return invalid()
    const codes = url.searchParams.getAll('code')
    const code = codes[0]
    if (!url.searchParams.has('error') && codes.length === 1 && matches(/^[\x21-\x7e]{1,2048}$/u, code) && exchanges >= MAX_EXCHANGES) {
      return page('<p>The Slack broker is busy. Retry this page shortly.</p>', 429, { 'retry-after': '2' })
    }
    states.delete(state)
    flow.phase = { type: 'denied', browser: phase.browser }
    if (url.searchParams.has('error') || codes.length !== 1 || !matches(/^[\x21-\x7e]{1,2048}$/u, code)) return redirect(browserPath(id))
    flow.phase = { type: 'exchange', browser: phase.browser }
    exchanges++
    try {
      const exchange = phase.oauth.consume(phase.browser, { state, code })
      const data = await slack.request('oauth.v2.access', exchange.authorization, exchange.body, request.signal)
      if (active(id) !== flow) return invalid()
      const { account, token } = parseUserGrant(data)
      const scope = record(data.authed_user)?.scope
      if (typeof scope !== 'string' || scope.length > 256) throw new Error()
      const verified = await slack.request('auth.test', `Bearer ${token}`, {}, request.signal)
      if (verified.team_id !== account.teamId || verified.user_id !== account.userId) throw new Error()
      if (active(id) !== flow) return invalid()
      flow.phase = { type: 'review', browser: phase.browser, grant: { access_token: token, team_id: account.teamId, user_id: account.userId, scope } }
    } catch {
      flow.phase = { type: 'denied', browser: phase.browser }
    } finally { exchanges-- }
    return redirect(browserPath(id))
  }

  async function redeem(request: Request): Promise<Response> {
    const data = await jsonBody(request, ['id', 'verifier'])
    if (!matches(DIGEST, data?.id) || !matches(VERIFIER, data.verifier)) return json({ error: 'invalid_request' }, 400)
    const flow = active(data.id)
    if (!flow || !equal(flow.challenge, sha(data.verifier))) return json({ status: 'expired' })
    if (flow.phase.type === 'denied') {
      drop(data.id, flow)
      return json({ status: 'denied' })
    }
    if (flow.phase.type !== 'ready') return json({ status: 'pending' })
    const { grant } = flow.phase
    drop(data.id, flow)
    return json({ status: 'ready', ...grant })
  }

  return { sweep, async handle(request) {
    if (request.url.length > 8192 || (request.headers.get('cookie')?.length ?? 0) > 8192) return json({ error: 'invalid_request' }, 400)
    if (request.headers.get('host') !== publicUrl.host) return json({ error: 'not_found' }, 404)
    const url = new URL(request.url)
    const path = url.pathname
    const allow = path === AUTHORIZE ? 'GET, POST' : path === CALLBACK ? 'GET' : path === START || path === REDEEM ? 'POST' : undefined
    if (!allow) return json({ error: 'not_found' }, 404)
    if (!allow.split(', ').includes(request.method)) return json({ error: 'method_not_allowed' }, 405, { allow })
    const time = now()
    if (time >= windowEnd) { windowEnd = time + WINDOW_MS; requests = 0; starts = 0 }
    const retryAfter = (): Record<string, string> => ({ 'retry-after': String(Math.max(1, Math.ceil((windowEnd - time) / 1000))) })
    if (++requests > 2048) return json({ error: 'rate_limited' }, 429, retryAfter())
    if (path === START) {
      const authorization = request.headers.get('authorization') ?? ''
      const credential = authorization.slice(7)
      if (!authorization.startsWith('Bearer ') || !matches(DIGEST, credential) || !equal(sha(credential), enrolled)) {
        return json({ error: 'unauthorized' }, 401, { 'www-authenticate': 'Bearer' })
      }
      if (++starts > 128) return json({ error: 'rate_limited' }, 429, retryAfter())
    }
    switch (path) {
      case START: return start(request)
      case REDEEM: return redeem(request)
      case AUTHORIZE: return authorize(request, url)
      default: return callback(request, url)
    }
  } }
}
