// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { createHash, randomBytes } from 'node:crypto'
import { createSlackBroker } from '../src/slack-broker/broker.js'

const origin = 'https://broker.example.test'
const routes = '/v1/slack/'
const scope = 'users:read,users:read.email,im:write,im:read,im:history,chat:write'
const secret = 'test-client-secret'
const enrollmentToken = randomBytes(32).toString('base64url')
const token = 'xoxp-test-token'
const grant = { ok: true, team: { id: 'T12345' }, authed_user: { id: 'U12345', token_type: 'user', access_token: token, scope } }

function harness(exchange: unknown = grant, verified: unknown = { ok: true, team_id: 'T12345', user_id: 'U12345' }) {
  let time = 10_000
  const requests: { url: string; init: RequestInit }[] = []
  const broker = createSlackBroker({ publicOrigin: origin, clientId: '123.456', clientSecret: secret, enrollmentToken, now: () => time,
    fetch: async (url, init) => {
      requests.push({ url, init })
      if (url === 'https://slack.com/api/oauth.v2.access') {
        if (exchange instanceof Error) throw exchange
        return exchange instanceof Promise ? await exchange : exchange instanceof Response ? exchange : Response.json(exchange)
      }
      if (url === 'https://slack.com/api/auth.test') return Response.json(verified)
      throw new Error('Unexpected fetch')
    },
  })
  const send = (path: string, init?: RequestInit) => {
    const headers = new Headers(init?.headers)
    if (!headers.has('host')) headers.set('host', new URL(origin).host)
    return broker.handle(new Request(new URL(path, origin), { ...init, headers }))
  }
  const post = (route: string, payload: unknown) => send(routes + route, {
    method: 'POST', headers: { 'content-type': 'application/json', ...(route === 'start' ? { authorization: `Bearer ${enrollmentToken}` } : {}) }, body: JSON.stringify(payload),
  })
  const verifier = randomBytes(32).toString('base64url')
  const challenge = createHash('sha256').update(verifier).digest('base64url')
  async function start(): Promise<{ id: string; browser_url: string; pairing_code: string }> {
    const response = await post('start', { challenge })
    expect(response.status).toBe(200)
    expect(response.headers.get('cache-control')).toBe('no-store')
    return response.json()
  }
  async function visit(url: string): Promise<string> {
    const response = await send(url)
    expect(response.status).toBe(200)
    expect(await response.text()).toContain('Paste the pairing code')
    expect(response.headers.get('referrer-policy')).toBe('same-origin')
    const header = response.headers.get('set-cookie') ?? ''
    expect(header).toMatch(/^__Host-xerxes-slack-.*; Secure; HttpOnly; SameSite=Lax; Path=\/; Max-Age=600$/)
    return header.split(';')[0] ?? ''
  }
  const form = (path: string, cookie: string, fields: Record<string, string>, formOrigin = origin) => send(path, { method: 'POST',
    headers: { cookie, origin: formOrigin, 'content-type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams(fields) })
  async function authorize(path: string, cookie: string, code: string): Promise<URL> {
    expect((await form(path, cookie, { code })).status).toBe(303)
    const page = await send(path, { headers: { cookie } })
    const link = /href="([^"]+)"/.exec(await page.text())?.[1]
    return new URL(link?.replaceAll('&amp;', '&') ?? '')
  }
  return { send, post, start, visit, form, authorize, verifier, challenge, requests, advance(ms: number) { time += ms; broker.sweep() } }
}

const redirectPath = (response: Response): string => response.headers.get('location') ?? ''
const callbackPath = (authorize: URL, key = 'code', value = 'slack-secret-code'): string => `${routes}callback?${new URLSearchParams({ state: authorize.searchParams.get('state') ?? '', [key]: value })}`

describe('managed Slack OAuth broker', () => {
  test('refuses to start without a correctly formatted enrollment credential', () => {
    for (const invalid of ['', 'short', `${enrollmentToken}\n`]) {
      expect(() => createSlackBroker({ publicOrigin: origin, clientId: '123.456', clientSecret: secret, enrollmentToken: invalid,
        fetch: async () => { throw new Error('unexpected fetch') } })).toThrow('Slack broker requires a valid enrollment token.')
    }
  })

  test('rejects rogue Hub starts before parsing or allocating a flow', async () => {
    const h = harness()
    const attackerToken = randomBytes(32).toString('base64url')
    for (const auth of [undefined, `Bearer ${attackerToken}`, `Basic ${enrollmentToken}`, `Bearer ${enrollmentToken}, Bearer ${attackerToken}`]) {
      const response = await h.send(`${routes}start`, { method: 'POST', headers: { 'content-type': 'application/json',
        ...(auth === undefined ? {} : { authorization: auth }) }, body: JSON.stringify({ challenge: h.challenge }) })
      expect(response.status).toBe(401)
      expect(response.headers.get('www-authenticate')).toBe('Bearer')
      expect(response.headers.get('set-cookie')).toBeNull()
      expect(await response.json()).toEqual({ error: 'unauthorized' })
    }
    const query = await h.send(`${routes}start?enrollment_token=${enrollmentToken}`, { method: 'POST',
      headers: { 'content-type': 'application/json' }, body: JSON.stringify({ challenge: h.challenge, enrollment_token: enrollmentToken }) })
    expect(query.status).toBe(401)
    const response = await h.send(`${routes}start`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: 'not JSON' })
    expect(response.status).toBe(401)
    expect(h.requests).toHaveLength(0)
    const flow = await h.start()
    expect(JSON.stringify(flow)).not.toContain(enrollmentToken)
  })

  test('pairs the owner browser, verifies the Slack identity and releases the token once, only after explicit review', async () => {
    const h = harness()
    const flow = await h.start()
    expect(flow.browser_url).toBe(`${origin}${routes}authorize?id=${flow.id}`)
    expect(flow.browser_url).not.toContain(flow.pairing_code)
    const cookie = await h.visit(flow.browser_url)
    const copyingBrowser = await h.visit(flow.browser_url)
    expect(copyingBrowser).not.toBe(cookie)
    expect((await h.form(flow.browser_url, copyingBrowser, { code: 'x'.repeat(22) })).status).toBe(400)
    expect((await h.form(flow.browser_url, '', { code: flow.pairing_code })).status).toBe(400)
    expect((await h.form(flow.browser_url, cookie, { code: flow.pairing_code }, 'https://attacker.example.test')).status).toBe(400)
    const authorize = await h.authorize(flow.browser_url, cookie, flow.pairing_code)
    expect(authorize.origin + authorize.pathname).toBe('https://slack.com/oauth/v2/authorize')
    expect(authorize.searchParams.get('scope')).toBeNull()
    expect(authorize.searchParams.get('client_id')).toBe('123.456')
    expect(authorize.searchParams.get('redirect_uri')).toBe(`${origin}${routes}callback`)
    expect(authorize.searchParams.get('user_scope')).toBe(scope)
    expect((await h.send(flow.browser_url, { headers: { cookie: copyingBrowser } })).status).toBe(400)
    expect((await h.form(flow.browser_url, copyingBrowser, { code: flow.pairing_code })).status).toBe(400)
    const callback = callbackPath(authorize)
    expect((await h.send(callback)).status).toBe(400)
    expect((await h.send(callback, { headers: { cookie: copyingBrowser } })).status).toBe(400)
    expect((await h.send(`${callback}&state=${authorize.searchParams.get('state')}`, { headers: { cookie } })).status).toBe(400)
    expect(h.requests).toHaveLength(0)
    const parallel = await Promise.all([h.send(callback, { headers: { cookie } }), h.send(callback, { headers: { cookie } })])
    expect(parallel.map(response => response.status).sort()).toEqual([303, 400])
    const exchange = h.requests[0]
    expect(h.requests.map(request => request.url)).toEqual(['https://slack.com/api/oauth.v2.access', 'https://slack.com/api/auth.test'])
    expect(exchange?.init.method).toBe('POST')
    expect(exchange?.init.redirect).toBe('error')
    expect(new Headers(exchange?.init.headers).get('authorization')).toBe(`Basic ${Buffer.from(`123.456:${secret}`).toString('base64')}`)
    expect(new URLSearchParams(String(exchange?.init.body)).get('redirect_uri')).toBe(`${origin}${routes}callback`)
    expect(new URLSearchParams(String(exchange?.init.body)).get('code')).toBe('slack-secret-code')
    expect(new Headers(h.requests[1]?.init.headers).get('authorization')).toBe(`Bearer ${token}`)
    const resume = parallel.find(response => response.status === 303)
    if (!resume) throw new Error('Missing redirect')
    expect(resume.headers.get('referrer-policy')).toBe('no-referrer')
    expect(redirectPath(resume)).not.toContain('slack-secret-code')
    const page = await h.send(redirectPath(resume), { headers: { cookie } })
    const review = await page.text()
    expect(review).toContain('T12345')
    expect(review).toContain('U12345')
    expect(review).not.toContain(token)
    expect(review).not.toContain(secret)
    expect(await (await h.post('redeem', { id: flow.id, verifier: h.verifier })).json()).toEqual({ status: 'pending' })
    expect((await h.form(flow.browser_url, copyingBrowser, { decision: 'approve' })).status).toBe(400)
    expect((await h.form(flow.browser_url, cookie, { decision: 'approve' }, 'https://attacker.example.test')).status).toBe(400)
    expect((await h.form(flow.browser_url, cookie, { decision: 'approve' })).headers.get('set-cookie')).toContain('Max-Age=0')
    expect(await (await h.post('redeem', { id: flow.id, verifier: randomBytes(32).toString('base64url') })).json()).toEqual({ status: 'expired' })
    const redeemed = await Promise.all([h.post('redeem', { id: flow.id, verifier: h.verifier }), h.post('redeem', { id: flow.id, verifier: h.verifier })])
    const values = await Promise.all(redeemed.map(response => response.json()))
    expect(values).toContainEqual({ status: 'ready', access_token: token, team_id: 'T12345', user_id: 'U12345', scope })
    expect(values).toContainEqual({ status: 'expired' })
  })

  test('consumes Slack denial without fetching, and browser rejection discards a valid grant', async () => {
    for (const source of ['slack', 'browser']) {
      const h = harness()
      const flow = await h.start()
      const cookie = await h.visit(flow.browser_url)
      const authorize = await h.authorize(flow.browser_url, cookie, flow.pairing_code)
      const callback = source === 'slack' ? callbackPath(authorize, 'error', secret) : callbackPath(authorize)
      expect(redirectPath(await h.send(callback, { headers: { cookie } }))).not.toContain(secret)
      if (source === 'browser') expect((await h.form(flow.browser_url, cookie, { decision: 'deny' })).status).toBe(200)
      expect(await (await h.post('redeem', { id: flow.id, verifier: h.verifier })).json()).toEqual({ status: 'denied' })
      expect((await h.send(callbackPath(authorize), { headers: { cookie } })).status).toBe(400)
      expect(h.requests).toHaveLength(source === 'slack' ? 0 : 2)
    }
  })

  test('rejects unsupported grants, mismatched identities and upstream errors without disclosing them', async () => {
    for (const h of [
      harness({ ...grant, authed_user: { ...grant.authed_user, scope: 'chat:write' } }),
      harness({ ...grant, authed_user: { ...grant.authed_user, refresh_token: secret } }),
      harness(grant, { ok: true, team_id: 'T12345', user_id: 'U999' }),
      harness(new Error(`${secret} ${token}`)),
      harness(new Response(null, { status: 302, headers: { location: 'https://other.example.test' } })),
      harness(new Response('x'.repeat(2 * 1024 * 1024 + 1))),
    ]) {
      const flow = await h.start()
      const cookie = await h.visit(flow.browser_url)
      const authorize = await h.authorize(flow.browser_url, cookie, flow.pairing_code)
      const result = await h.send(callbackPath(authorize), { headers: { cookie } })
      const text = await (await h.send(redirectPath(result), { headers: { cookie } })).text()
      expect(text).not.toContain(secret)
      expect(text).not.toContain(token)
      expect(await (await h.post('redeem', { id: flow.id, verifier: h.verifier })).json()).toEqual({ status: 'denied' })
    }
  })

  test('an expired or cancelled exchange cannot stage a late grant', async () => {
    for (const reason of ['expired', 'cancelled']) {
      const exchange = Promise.withResolvers<Response>()
      const h = harness(exchange.promise)
      const flow = await h.start()
      const cookie = await h.visit(flow.browser_url)
      const authorize = await h.authorize(flow.browser_url, cookie, flow.pairing_code)
      const controller = new AbortController()
      const callback = h.send(callbackPath(authorize), { headers: { cookie }, signal: controller.signal })
      expect(h.requests).toHaveLength(1)
      if (reason === 'expired') h.advance(10 * 60_000)
      else controller.abort(new Error(secret))
      exchange.resolve(Response.json(grant))
      expect(await (await callback).text()).not.toContain(secret)
      expect(h.requests).toHaveLength(1)
      expect(await (await h.post('redeem', { id: flow.id, verifier: h.verifier })).json()).toEqual({ status: reason === 'expired' ? 'expired' : 'denied' })
    }
  })

  test('enforces flow TTL from start, exact destination, bounded bodies, and process-local rate and capacity limits', async () => {
    const h = harness()
    expect((await h.post('start', { challenge: 'x'.repeat(43) })).status).toBe(400)
    expect((await h.post('start', { challenge: h.challenge, redirect_uri: 'https://other.example.test' })).status).toBe(400)
    expect((await h.send(`${routes}start`, { method: 'POST', headers: { host: 'other.example.test', 'x-forwarded-host': 'broker.example.test' } })).status).toBe(404)
    expect((await h.send(`${routes}start`, { method: 'POST', headers: { origin, 'content-type': 'application/json', authorization: `Bearer ${enrollmentToken}` }, body: JSON.stringify({ challenge: h.challenge }) })).status).toBe(400)
    expect((await h.send(`${routes}start`, { method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${enrollmentToken}` }, body: `${' '.repeat(1024)}${JSON.stringify({ challenge: h.challenge })}` })).status).toBe(400)
    const flow = await h.start()
    const cookie = await h.visit(flow.browser_url)
    h.advance(9 * 60_000)
    const authorize = await h.authorize(flow.browser_url, cookie, flow.pairing_code)
    h.advance(60_000)
    expect((await h.send(callbackPath(authorize), { headers: { cookie } })).status).toBe(400)
    expect(await (await h.post('redeem', { id: flow.id, verifier: h.verifier })).json()).toEqual({ status: 'expired' })
    expect(h.requests).toHaveLength(0)

    for (let window = 0; window < 2; window++) {
      for (let i = 0; i < 128; i++) expect((await h.post('start', { challenge: h.challenge })).status).toBe(200)
      const limited = await h.post('start', { challenge: h.challenge })
      expect(limited.status).toBe(429)
      expect(limited.headers.get('retry-after')).toBe('60')
      h.advance(60_000)
    }
    expect(await (await h.post('start', { challenge: h.challenge })).json()).toEqual({ error: 'busy' })
    h.advance(10 * 60_000)
    expect((await h.post('start', { challenge: h.challenge })).status).toBe(200)
  })
})
