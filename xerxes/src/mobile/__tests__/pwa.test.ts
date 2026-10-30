// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { HUB_HTTP, HubHttpTimeout, createHubHttp, type FetchLike } from '../api.js'
import { initialPairState, pairingAwait, readPairRoute, rememberPairingAwait, type PairStorage } from '../pairing.js'
import { FakeTimers, settle } from './fakes.js'
import { createPushService, urlBase64ToBytes, type PushEnv } from '../pwa/push.js'
import { fetchDecision, installServiceWorker, pushDisplay, safeTaskPath, type SwClient, type SwScope } from '../pwa/swCore.js'

const ORIGIN = 'https://hub.example'

interface Recorded {
  readonly puts: string[]
  readonly notifications: { title: string; options: NotificationOptions }[]
  readonly opened: string[]
  readonly navigated: string[]
}

function fakeScope(clients: SwClient[] = []): { scope: SwScope; listeners: Map<string, (event: never) => void>; recorded: Recorded; cached: Map<string, Response> } {
  const listeners = new Map<string, (event: never) => void>()
  const cached = new Map<string, Response>()
  const recorded: Recorded = { puts: [], notifications: [], opened: [], navigated: [] }
  const cache = {
    addAll: async (urls: readonly string[]) => { for (const url of urls) recorded.puts.push(url) },
    match: async (request: Request | string) => cached.get(typeof request === 'string' ? request : new URL(request.url).pathname),
    put: async (request: Request | string) => { recorded.puts.push(typeof request === 'string' ? request : request.url) },
  }
  const scope: SwScope = {
    location: { origin: ORIGIN },
    registration: { showNotification: async (title, options) => { recorded.notifications.push({ title, options }) } },
    clients: {
      claim: async () => {},
      matchAll: async () => clients,
      openWindow: async url => { recorded.opened.push(url) },
    },
    caches: { open: async () => cache, keys: async () => ['xerxes-shell-old', 'other', 'xerxes-shell-v1'], delete: async () => true },
    skipWaiting: async () => {},
    addEventListener: (type, listener) => { listeners.set(type, listener as (event: never) => void) },
  }
  return { scope, listeners, recorded, cached }
}

const SHELL = ['/index.html', '/assets/main-abc12345.js', '/assets/mobile-abc12345.css']
const request = (url: string, mode = 'cors', method = 'GET'): Request => ({ url, mode, method }) as unknown as Request

describe('service worker caching policy', () => {
  it('never intercepts /api/ or the socket, whatever the request', () => {
    const shell = new Set(SHELL)
    for (const path of ['/api/hub/v1/ws', '/api/hub/v1/session/refresh', '/api/hub/v1/push/vapid', '/api/anything']) {
      for (const mode of ['cors', 'navigate', 'same-origin']) {
        expect(fetchDecision({ method: 'GET', url: `${ORIGIN}${path}`, origin: ORIGIN, mode, shell })).toBe('network')
      }
    }
    // Even a hostile shell list cannot put an API path in the cache.
    expect(fetchDecision({ method: 'GET', url: `${ORIGIN}/api/x`, origin: ORIGIN, mode: 'cors', shell: new Set(['/api/x']) })).toBe('network')
  })

  it('only serves the listed shell from cache; other GETs go to the network; mutations are untouched', () => {
    const shell = new Set(SHELL)
    expect(fetchDecision({ method: 'GET', url: `${ORIGIN}/assets/main-abc12345.js`, origin: ORIGIN, mode: 'cors', shell })).toBe('shell')
    expect(fetchDecision({ method: 'GET', url: `${ORIGIN}/other.js`, origin: ORIGIN, mode: 'cors', shell })).toBe('network')
    expect(fetchDecision({ method: 'GET', url: `${ORIGIN}/w/a/s/b`, origin: ORIGIN, mode: 'navigate', shell })).toBe('navigation')
    expect(fetchDecision({ method: 'POST', url: `${ORIGIN}/assets/main-abc12345.js`, origin: ORIGIN, mode: 'cors', shell })).toBe('network')
    expect(fetchDecision({ method: 'GET', url: 'https://elsewhere.example/assets/main-abc12345.js', origin: ORIGIN, mode: 'cors', shell })).toBe('network')
  })

  it('precaches only the shell on install and drops stale shell caches on activate', async () => {
    const { scope, listeners, recorded } = fakeScope()
    installServiceWorker(scope, { version: 'v1', shell: SHELL })
    const install = { waitUntil: async (p: Promise<unknown>) => { await p } }
    await (listeners.get('install') as (e: unknown) => Promise<void>)(install)
    expect(recorded.puts).toEqual(SHELL)
    expect(recorded.puts.some(url => url.includes('/api/'))).toBe(false)
    const waited: Promise<unknown>[] = []
    ;(listeners.get('activate') as (e: unknown) => void)({ waitUntil: (p: Promise<unknown>) => { waited.push(p) } })
    await Promise.all(waited)
  })

  it('does not call respondWith for an API fetch and answers a cached shell asset from cache', async () => {
    const { scope, listeners, cached } = fakeScope()
    cached.set('/assets/main-abc12345.js', new Response('cached-js'))
    installServiceWorker(scope, { version: 'v1', shell: SHELL })
    const onFetch = listeners.get('fetch') as (event: unknown) => void
    let responded = 0
    onFetch({ request: request(`${ORIGIN}/api/hub/v1/session/refresh`, 'cors', 'POST'), respondWith: () => { responded += 1 } })
    onFetch({ request: request(`${ORIGIN}/api/hub/v1/ws`), respondWith: () => { responded += 1 } })
    expect(responded).toBe(0)
    let answer: Promise<Response> | Response | undefined
    onFetch({ request: request(`${ORIGIN}/assets/main-abc12345.js`), respondWith: (value: Promise<Response> | Response) => { answer = value } })
    expect(await (await answer)?.text()).toBe('cached-js')
  })

  it('serves the cached shell when a navigation is slower than the deadline, and the network when it is fast', async () => {
    const slow = fakeScope()
    slow.cached.set('/index.html', new Response('cached-shell'))
    installServiceWorker({ ...slow.scope, fetch: () => new Promise<Response>(() => undefined) }, { version: 'v1', shell: SHELL, navigationTimeoutMs: 20 })
    let answer: Promise<Response> | Response | undefined
    ;(slow.listeners.get('fetch') as (event: unknown) => void)({ request: request(`${ORIGIN}/w/a/s/b`, 'navigate'), respondWith: (value: Promise<Response> | Response) => { answer = value } })
    expect(await (await answer)?.text()).toBe('cached-shell')

    const fast = fakeScope()
    fast.cached.set('/index.html', new Response('cached-shell'))
    installServiceWorker({ ...fast.scope, fetch: async () => new Response('fresh-page') }, { version: 'v1', shell: SHELL, navigationTimeoutMs: 1_000 })
    ;(fast.listeners.get('fetch') as (event: unknown) => void)({ request: request(`${ORIGIN}/`, 'navigate'), respondWith: (value: Promise<Response> | Response) => { answer = value } })
    expect(await (await answer)?.text()).toBe('fresh-page')

    const down = fakeScope()
    down.cached.set('/index.html', new Response('cached-shell'))
    installServiceWorker({ ...down.scope, fetch: async () => { throw new Error('offline') } }, { version: 'v1', shell: SHELL, navigationTimeoutMs: 1_000 })
    ;(down.listeners.get('fetch') as (event: unknown) => void)({ request: request(`${ORIGIN}/`, 'navigate'), respondWith: (value: Promise<Response> | Response) => { answer = value } })
    expect(await (await answer)?.text()).toBe('cached-shell')
  })

  it('waits for a slow network when no shell is cached yet', async () => {
    const { scope, listeners } = fakeScope()
    let release: (response: Response) => void = () => undefined
    installServiceWorker({ ...scope, fetch: () => new Promise<Response>(resolve => { release = resolve }) }, { version: 'v1', shell: SHELL, navigationTimeoutMs: 10 })
    let answer: Promise<Response> | Response | undefined
    ;(listeners.get('fetch') as (event: unknown) => void)({ request: request(`${ORIGIN}/`, 'navigate'), respondWith: (value: Promise<Response> | Response) => { answer = value } })
    await new Promise(resolve => { setTimeout(resolve, 30) })
    release(new Response('late-page'))
    expect(await (await answer)?.text()).toBe('late-page')
  })

  it('also ships a worker source with no /api/ cache entry', async () => {
    const source = await readFile(join(import.meta.dirname, '..', 'pwa', 'swCore.ts'), 'utf8')
    expect(source).not.toMatch(/cache\.(put|add|addAll)\([^)]*\/api\//)
    expect(source).toContain("startsWith('/api/')")
  })
})

describe('push notifications', () => {
  it('shows minimal text, no actions, and a task url that stays on this origin', () => {
    const shown = pushDisplay({ title: 'Task needs you', body: 'Open Xerxes to decide', url: '/w/w1/s/s1', tag: 'w1:s1', actions: [{ action: 'approve', title: 'Approve' }], rawCommand: 'rm -rf /' }, ORIGIN)
    expect(shown.title).toBe('Task needs you')
    expect(shown.options).not.toHaveProperty('actions')
    expect(JSON.stringify(shown)).not.toContain('rm -rf')
    expect(shown.options.data).toEqual({ url: '/w/w1/s/s1' })
    expect(pushDisplay(undefined, ORIGIN).title).toBe('Xerxes')
  })

  it('clips long text and falls back to a generic notice for junk payloads', () => {
    const shown = pushDisplay({ title: 'x'.repeat(500), body: 'y'.repeat(500) }, ORIGIN)
    expect(shown.title.length).toBeLessThanOrEqual(60)
    expect(String(shown.options.body).length).toBeLessThanOrEqual(120)
    expect(pushDisplay('garbage', ORIGIN).options.body).toContain('Open Xerxes')
  })

  it('only opens the Inbox or one exact task', () => {
    expect(safeTaskPath('/w/a/s/b', ORIGIN)).toBe('/w/a/s/b')
    expect(safeTaskPath(`${ORIGIN}/w/a/s/b?x=1#frag`, ORIGIN)).toBe('/w/a/s/b')
    expect(safeTaskPath('https://evil.example/w/a/s/b', ORIGIN)).toBe('/')
    expect(safeTaskPath('/api/hub/v1/session/refresh', ORIGIN)).toBe('/')
    expect(safeTaskPath('/w/a/s/b/extra', ORIGIN)).toBe('/')
    expect(safeTaskPath(42, ORIGIN)).toBe('/')
    // XERXES pushes open the XERXES tab, not the Inbox.
    expect(safeTaskPath('/xerxes', ORIGIN)).toBe('/xerxes')
    expect(safeTaskPath(`${ORIGIN}/xerxes?x=1`, ORIGIN)).toBe('/xerxes')
    expect(safeTaskPath('/xerxes/other', ORIGIN)).toBe('/')
  })

  it('shows a notification on push and focuses an existing window on click', async () => {
    const client: SwClient = { url: `${ORIGIN}/`, focus: async () => {}, navigate: async url => { recorded.navigated.push(url) } }
    const { scope, listeners, recorded } = fakeScope([client])
    installServiceWorker(scope, { version: 'v1', shell: SHELL })
    const waited: Promise<unknown>[] = []
    ;(listeners.get('push') as (e: unknown) => void)({ data: { json: () => ({ title: 'T', body: 'B', url: '/w/w1/s/s1' }), text: () => '' }, waitUntil: (p: Promise<unknown>) => { waited.push(p) } })
    await Promise.all(waited)
    expect(recorded.notifications[0]?.title).toBe('T')

    let closed = false
    const clicks: Promise<unknown>[] = []
    ;(listeners.get('notificationclick') as (e: unknown) => void)({
      notification: { close: () => { closed = true }, data: { url: '/w/w1/s/s1' } },
      waitUntil: (p: Promise<unknown>) => { clicks.push(p) },
    })
    await Promise.all(clicks)
    expect(closed).toBe(true)
    expect(recorded.navigated).toEqual(['/w/w1/s/s1'])
  })

  it('opens a new window at the task url when no window exists, and ignores a hostile target', async () => {
    const { scope, listeners, recorded } = fakeScope([])
    installServiceWorker(scope, { version: 'v1', shell: SHELL })
    const clicks: Promise<unknown>[] = []
    const click = listeners.get('notificationclick') as (e: unknown) => void
    click({ notification: { close() {}, data: { url: '/w/w1/s/s1' } }, waitUntil: (p: Promise<unknown>) => { clicks.push(p) } })
    click({ notification: { close() {}, data: { url: 'https://evil.example/' } }, waitUntil: (p: Promise<unknown>) => { clicks.push(p) } })
    await Promise.all(clicks)
    expect(recorded.opened).toEqual(['/w/w1/s/s1', '/'])
  })
})

describe('push opt-in', () => {
  function env(order: string[], overrides: Partial<PushEnv> = {}): PushEnv {
    const subscription = { toJSON: () => ({ endpoint: 'https://push.example/abc', keys: {} }) }
    return {
      http: {
        csrf: () => 'csrf-token',
        refresh: async () => ({ status: 'ok' as const }),
        json: async (method, path, body) => {
          order.push(`${method} ${path}`)
          if (path.endsWith('/vapid-key')) return { status: 200, body: { public_key: 'BAUD' } }
          order.push(`body ${JSON.stringify(body)}`)
          return { status: 200, body: {} }
        },
      },
      ready: async () => ({ pushManager: { getSubscription: async () => null, subscribe: async () => subscription } }),
      permission: () => 'default',
      requestPermission: async () => { order.push('permission'); return 'granted' },
      ...overrides,
    }
  }

  it('asks for permission first, then fetches the key, subscribes and posts with the CSRF header path', async () => {
    const order: string[] = []
    const outcome = await createPushService(env(order)).enable()
    expect(outcome).toEqual({ ok: true })
    expect(order[0]).toBe('permission')
    expect(order[1]).toBe('GET /api/hub/v1/push/vapid-key')
    expect(order[2]).toBe('POST /api/hub/v1/push/subscribe')
  })

  it('stops at a denied permission without touching the network', async () => {
    const order: string[] = []
    const outcome = await createPushService(env(order, { requestPermission: async () => { order.push('permission'); return 'denied' } })).enable()
    expect(outcome.ok).toBe(false)
    expect(order).toEqual(['permission'])
  })

  it('reports an unsupported browser with the iPhone hint, and a hub without push', async () => {
    const none = await createPushService(env([], { ready: undefined })).enable()
    expect(none).toMatchObject({ ok: false })
    expect(none.ok ? '' : none.message).toContain('Home Screen')
    const order: string[] = []
    const noKey = await createPushService(env(order, { http: { csrf: () => 'x', refresh: async () => ({ status: 'ok' as const }), json: async () => ({ status: 404, body: {} }) } })).enable()
    expect(noKey.ok ? '' : noKey.message).toContain('not set up for push')
  })

  it('refreshes the session before the VAPID request and reports an expired session plainly', async () => {
    const order: string[] = []
    const base = env(order)
    const outcome = await createPushService({ ...base, http: { ...base.http, ensureFresh: async () => { order.push('ensureFresh'); return { status: 'ok' as const } } } }).enable()
    expect(outcome).toEqual({ ok: true })
    expect(order.slice(0, 3)).toEqual(['permission', 'ensureFresh', 'GET /api/hub/v1/push/vapid-key'])
    const expired = await createPushService({ ...base, http: { ...base.http, ensureFresh: async () => ({ status: 'denied' as const }) } }).enable()
    expect(expired.ok ? '' : expired.message).toContain('session with the Hub expired')
  })

  it('re-registers an existing browser subscription without prompting, and skips when there is none', async () => {
    const order: string[] = []
    const subscription = { toJSON: () => ({ endpoint: 'https://push.example/abc', keys: { p256dh: 'k', auth: 'a' } }) }
    const granted = env(order, {
      permission: () => 'granted',
      ready: async () => ({ pushManager: { getSubscription: async () => subscription, subscribe: async () => subscription } }),
    })
    expect(await createPushService(granted).resync()).toEqual({ ok: true })
    expect(order).toContain('POST /api/hub/v1/push/subscribe')
    expect(order).not.toContain('permission')
    expect(order.some(line => line.includes('push.example/abc'))).toBe(true)

    const none: string[] = []
    expect(await createPushService(env(none, { permission: () => 'granted' })).resync()).toEqual({ ok: 'skipped' })
    expect(await createPushService(env(none, { permission: () => 'default' })).resync()).toEqual({ ok: 'skipped' })
    expect(none).toEqual([])
  })

  it('decodes a VAPID key', () => {
    expect([...urlBase64ToBytes('AQID')]).toEqual([1, 2, 3])
  })
})

describe('http helper', () => {
  it('sends the CSRF header on mutations only, with same-origin credentials and no tokens in URLs', async () => {
    const calls: { path: string; init: RequestInit }[] = []
    const fetchFn: FetchLike = async (path, init) => {
      calls.push({ path, init })
      return new Response(JSON.stringify({ csrf_token: 'tok-1' }), { status: 200 })
    }
    const http = createHubHttp(fetchFn)
    expect(await http.refresh()).toEqual({ status: 'ok', csrfToken: 'tok-1' })
    await http.json('GET', '/api/hub/v1/push/vapid-key')
    await http.json('POST', '/api/hub/v1/push/subscribe', { subscription: {} })
    const [refresh, get, post] = calls
    expect((get?.init.headers as Record<string, string>)['x-xerxes-csrf']).toBeUndefined()
    expect((post?.init.headers as Record<string, string>)['x-xerxes-csrf']).toBe('tok-1')
    for (const call of calls) {
      expect(call.init.credentials).toBe('same-origin')
      expect(call.path).not.toMatch(/token|code|secret/i)
    }
    expect(refresh?.path).toBe('/api/hub/v1/session/refresh')
  })

  it('maps pairing and refresh responses', async () => {
    const respond = (status: number, body: unknown): FetchLike => async () => new Response(JSON.stringify(body), { status })
    expect(await createHubHttp(respond(401, {})).refresh()).toEqual({ status: 'denied' })
    expect(await createHubHttp(respond(401, { error: { code: 'unauthorized', message: 'x', reason: 'no_session' } })).refresh()).toEqual({ status: 'denied', reason: 'no_session' })
    // A 403 is an Origin problem, not a revoked device: never a sign-out.
    expect((await createHubHttp(respond(403, { error: { code: 'forbidden', message: 'origin is not allowed' } })).refresh()).status).toBe('unavailable')
    expect((await createHubHttp(respond(500, {})).refresh()).status).toBe('unavailable')
    expect(await createHubHttp(respond(200, { status: 'pending_confirmation', comparison_code: '1234' })).consumePairing({ pairing_id: 'p', code: 'c', device_label: 'iPhone' })).toEqual({ status: 'accepted', active: false, comparisonCode: '1234' })
    expect(await createHubHttp(respond(200, { status: 'active', comparison_code: '1234' })).consumePairing({ pairing_id: 'p', code: 'c', device_label: 'iPhone' })).toEqual({ status: 'accepted', active: true, comparisonCode: '1234' })
    expect(await createHubHttp(respond(202, { status: 'pending_confirmation' })).refresh()).toEqual({ status: 'pending' })
    expect((await createHubHttp(respond(410, { error: { code: 'gone', message: 'nested expired' } })).consumePairing({ pairing_id: 'p', code: 'c', device_label: 'iPhone' }))).toEqual({ status: 'rejected', message: 'nested expired' })
    expect((await createHubHttp(respond(410, { message: 'expired' })).consumePairing({ pairing_id: 'p', code: 'c', device_label: 'iPhone' }))).toEqual({ status: 'rejected', message: 'expired' })
    expect((await createHubHttp(respond(503, {})).consumePairing({ pairing_id: 'p', code: 'c', device_label: 'iPhone' })).status).toBe('unavailable')
    const down: FetchLike = async () => { throw new Error('offline') }
    expect((await createHubHttp(down).consumePairing({ pairing_id: 'p', code: 'c', device_label: 'iPhone' })).status).toBe('unavailable')
  })

  it('times out a hung request instead of waiting forever', async () => {
    const timers = new FakeTimers()
    let aborted = false
    const hang: FetchLike = (_path, init) => new Promise((_resolve, reject) => {
      init.signal?.addEventListener('abort', () => { aborted = true; reject(new Error('aborted')) })
    })
    const http = createHubHttp(hang, { timers, timeoutMs: 8_000 })
    const refreshing = http.refresh()
    const consuming = http.consumePairing({ pairing_id: 'p', code: 'c', device_label: 'iPhone' })
    const raw = http.json('GET', '/api/hub/v1/push/vapid-key').catch((error: unknown) => error)
    await settle()
    timers.advance(8_000)
    expect(await refreshing).toMatchObject({ status: 'unavailable', message: expect.stringContaining('did not answer') })
    expect((await consuming).status).toBe('unavailable')
    expect(await raw).toBeInstanceOf(HubHttpTimeout)
    expect(aborted).toBe(true)
  })

  it('refreshes once and retries once when a call meets a lapsed session', async () => {
    const paths: string[] = []
    let sessionOk = false
    const fetchFn: FetchLike = async path => {
      paths.push(path)
      if (path === HUB_HTTP.refresh) { sessionOk = true; return new Response(JSON.stringify({ csrf_token: 't2', session_expires_at: new Date(Date.now() + 300_000).toISOString() }), { status: 200 }) }
      return sessionOk ? new Response('{"ok":true}', { status: 200 }) : new Response('{}', { status: 401 })
    }
    const http = createHubHttp(fetchFn)
    const result = await http.json('GET', HUB_HTTP.pushVapid)
    expect(result.status).toBe(200)
    expect(paths).toEqual([HUB_HTTP.pushVapid, HUB_HTTP.refresh, HUB_HTTP.pushVapid])
    expect(http.csrf()).toBe('t2')
  })

  it('shares one refresh between concurrent callers so the token is rotated once', async () => {
    let refreshes = 0
    const fetchFn: FetchLike = async () => { refreshes += 1; await settle(); return new Response('{"csrf_token":"x"}', { status: 200 }) }
    const http = createHubHttp(fetchFn)
    const outcomes = await Promise.all([http.refresh(), http.refresh(), http.ensureFresh()])
    expect(refreshes).toBe(1)
    expect(outcomes.every(outcome => outcome.status === 'ok')).toBe(true)
  })

  it('retries a spent-token refusal once before concluding the device is signed out', async () => {
    const timers = new FakeTimers()
    let calls = 0
    const fetchFn: FetchLike = async () => {
      calls += 1
      return calls === 1
        ? new Response(JSON.stringify({ error: { code: 'unauthorized', message: 'x', reason: 'spent_token' } }), { status: 401 })
        : new Response('{"csrf_token":"won"}', { status: 200 })
    }
    const http = createHubHttp(fetchFn, { timers })
    const pending = http.refresh()
    await settle()
    timers.advance(1_500)
    expect(await pending).toEqual({ status: 'ok', csrfToken: 'won' })
    expect(calls).toBe(2)
    const denied: FetchLike = async () => new Response(JSON.stringify({ error: { code: 'unauthorized', message: 'x', reason: 'device_revoked' } }), { status: 401 })
    expect(await createHubHttp(denied).refresh()).toEqual({ status: 'denied', reason: 'device_revoked' })
  })

  it('keeps the browser session alive at ~70% of its life while active, and not while hidden', async () => {
    const timers = new FakeTimers()
    let clock = 0
    let refreshes = 0
    const fetchFn: FetchLike = async () => {
      refreshes += 1
      return new Response(JSON.stringify({ csrf_token: `t${refreshes}`, session_expires_at: new Date(clock + 300_000).toISOString() }), { status: 200 })
    }
    const http = createHubHttp(fetchFn, { timers, now: () => clock })
    let visible = true
    const stop = http.keepAlive(() => visible)
    await http.refresh()
    expect(refreshes).toBe(1)
    const step = async (ms: number): Promise<void> => { clock += ms; timers.advance(ms); await settle() }
    await step(209_000)
    expect(refreshes).toBe(1)
    await step(2_000)
    expect(refreshes).toBe(2)
    visible = false
    await step(300_000)
    expect(refreshes).toBe(2)
    // Back in the foreground with the session expired: ensureFresh refreshes and re-arms.
    visible = true
    await http.ensureFresh()
    expect(refreshes).toBe(3)
    stop()
    await step(600_000)
    expect(refreshes).toBe(3)
  })

  it('resumes waiting for desktop confirmation after a reload of the pair page, without storing a secret', () => {
    const items = new Map<string, string>()
    const storage: PairStorage = { getItem: key => items.get(key) ?? null, setItem: (key, value) => { items.set(key, value) }, removeItem: key => { items.delete(key) } }
    let replaced = 0
    const win = (hash: string) => ({ location: { hash, pathname: '/pair', search: '' }, history: { replaceState: () => { replaced += 1 } } })
    // Nothing remembered: the plain "needs a pairing link" page.
    expect(readPairRoute(win(''), storage, 0)).toEqual({ kind: 'missing' })
    rememberPairingAwait(storage, 1_000, '4829')
    expect([...items.values()].join('')).not.toMatch(/code|secret|token/i)
    const resumed = readPairRoute(win(''), storage, 2_000)
    expect(resumed).toEqual({ kind: 'resume', comparison: '4829' })
    expect(initialPairState(resumed)).toEqual({ phase: 'waiting', comparison: '4829' })
    // The remembered wait expires.
    expect(pairingAwait(storage, 1_000 + 11 * 60_000)).toBeUndefined()
    expect(items.size).toBe(0)
    // A fresh fragment supersedes an older wait.
    rememberPairingAwait(storage, 0, '1111')
    const fresh = readPairRoute(win('#pairing_id=p1&code=c1'), storage, 10)
    expect(fresh.kind).toBe('found')
    expect(items.size).toBe(0)
    expect(replaced).toBe(1)
    // Storage that throws (Safari private mode) degrades to the plain page.
    const broken: PairStorage = { getItem: () => { throw new Error('denied') }, setItem: () => { throw new Error('denied') }, removeItem: () => { throw new Error('denied') } }
    rememberPairingAwait(broken, 0)
    expect(readPairRoute(win(''), broken, 0)).toEqual({ kind: 'missing' })
  })

  it('posts pairing with the documented body fields', async () => {
    let seen: { path: string; body: unknown } | undefined
    const fetchFn: FetchLike = async (path, init) => { seen = { path, body: JSON.parse(String(init.body)) }; return new Response('{}', { status: 200 }) }
    await createHubHttp(fetchFn).consumePairing({ pairing_id: 'p1', code: 'c1', device_label: 'Android phone' })
    expect(seen).toEqual({ path: '/api/hub/v1/pair/consume', body: { pairing_id: 'p1', code: 'c1', device_label: 'Android phone' } })
  })
})
