// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createHubHealth, readHealthz, registerHubHealthIpc, hubHealthChannels, HUB_FOCUS_RECHECK_MS, type HubHealthPort } from '../src/desktop/main/hubHealth.js'
import { asHubHealthState, hubNoticeKind, type HubHealthDismissed, type HubHealthState } from '../src/desktop/hubHealthShared.js'
import { HubHealthNoticeView, hubHealthStateOf } from '../src/desktop/renderer/HubHealthNotice.js'

function fakePort(options: { hub?: { target: string; origin: string }; answer?: () => Promise<{ status: number; body: string }> } = {}) {
  let clock = 1_000_000
  let dismissed: HubHealthDismissed = { not_configured: false, unreachable: false }
  const urls: string[] = []
  const port: HubHealthPort = {
    readHub: () => options.hub,
    get: async url => { urls.push(url); return options.answer ? options.answer() : { status: 200, body: '{"status":"ok","workspaces":{"total":2,"online":2,"degraded":0}}' } },
    readDismissed: () => dismissed,
    writeDismissed: value => { dismissed = value },
    now: () => clock,
  }
  return { port, urls, advance: (ms: number) => { clock += ms }, dismissed: () => dismissed }
}

const HUB = { target: 'xerxes@hub.example', origin: 'https://hub.example.ts.net' }

test('no saved Hub reports not_configured without any network request', async () => {
  const fake = fakePort()
  const state = await createHubHealth(fake.port).check()
  expect(state.kind).toBe('not_configured')
  expect(fake.urls).toEqual([])
  expect(hubNoticeKind(state)).toBe('not_configured')
})

test('a healthy Hub is online and probes only its public /healthz', async () => {
  const fake = fakePort({ hub: HUB })
  const state = await createHubHealth(fake.port).check()
  expect(state).toMatchObject({ kind: 'online', target: HUB.target, origin: HUB.origin, workspaces: { total: 2, online: 2, degraded: 0 } })
  expect(fake.urls).toEqual(['https://hub.example.ts.net/healthz'])
  expect(hubNoticeKind(state)).toBeNull()
})

test('a Hub that does not answer is unreachable with the reason, and concurrent checks share one request', async () => {
  const fake = fakePort({ hub: HUB, answer: () => Promise.reject(new Error('no answer within 10 s')) })
  const health = createHubHealth(fake.port)
  const [a, b] = await Promise.all([health.check(), health.check()])
  expect(a).toBe(b)
  expect(fake.urls).toHaveLength(1)
  expect(a).toMatchObject({ kind: 'unreachable', detail: 'no answer within 10 s' })
  expect(hubNoticeKind(a)).toBe('unreachable')
})

test('healthz answers map to degraded, non-Hub and HTTP failures honestly', () => {
  expect(readHealthz(200, '{"status":"degraded","workspaces":{"total":9,"online":7,"degraded":2}}')).toMatchObject({ kind: 'degraded', detail: '2 of 9 workspaces offline' })
  expect(readHealthz(200, '{"status":"identity_unavailable"}')).toMatchObject({ kind: 'degraded' })
  expect(readHealthz(200, '<html>parked domain</html>')).toMatchObject({ kind: 'unreachable', detail: 'the address answered, but not as a Xerxes Hub' })
  expect(readHealthz(502, '')).toMatchObject({ kind: 'unreachable', detail: 'the Hub answered HTTP 502' })
})

test("don't remind me is per kind, persists, and remind turns it back on", async () => {
  const fake = fakePort({ hub: HUB, answer: () => Promise.reject(new Error('timeout')) })
  const health = createHubHealth(fake.port)
  await health.check()
  const dismissed = await health.handle('dismiss', { kind: 'unreachable' })
  expect(dismissed.ok && hubNoticeKind(dismissed.state)).toBeNull()
  expect(fake.dismissed()).toEqual({ not_configured: false, unreachable: true })
  expect(await health.handle('dismiss', { kind: 'everything' })).toMatchObject({ ok: false, code: 'invalid_request' })
  expect(await health.handle('reboot', {})).toMatchObject({ ok: false, code: 'unknown_action' })
  const reminded = await health.handle('remind', { kind: 'unreachable' })
  expect(reminded.ok && hubNoticeKind(reminded.state)).toBe('unreachable')
})

test('focus re-checks are throttled and listeners receive each new state', async () => {
  const fake = fakePort({ hub: HUB })
  const health = createHubHealth(fake.port)
  const seen: string[] = []
  health.onChange(state => seen.push(state.kind))
  await health.check()
  health.recheckOnFocus()
  expect(fake.urls).toHaveLength(1)
  fake.advance(HUB_FOCUS_RECHECK_MS + 1)
  health.recheckOnFocus()
  await Promise.resolve(); await Promise.resolve()
  expect(fake.urls).toHaveLength(2)
  expect(seen[0]).toBe('online')
})

test('IPC registers one channel per action and turns a thrown error into a failure', async () => {
  const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>()
  const broken = { ...createHubHealth(fakePort().port), handle: async () => { throw new Error('disk\nfull') } }
  registerHubHealthIpc({ handle: (channel, listener) => { handlers.set(channel, listener) } }, broken)
  expect([...handlers.keys()]).toEqual([...hubHealthChannels()])
  expect(await handlers.get('hubhealth:status')!({}, {})).toEqual({ ok: false, code: 'failed', message: 'disk full' })
})

const state = (over: Partial<HubHealthState>): HubHealthState => ({
  kind: 'unreachable', target: HUB.target, origin: HUB.origin, detail: 'no answer within 10 s', workspaces: null,
  checked_at: '2099-01-01T00:00:00.000Z', dismissed: { not_configured: false, unreachable: false }, ...over,
})
const noop = () => {}
const view = (s: HubHealthState | undefined, hidden = false) => renderToStaticMarkup(createElement(HubHealthNoticeView, { state: s, hidden, checking: false, onSettings: noop, onRecheck: noop, onLater: noop, onDismiss: noop }))

test('the notice names an unreachable Hub with its address, reason and the actions', () => {
  const html = view(state({}))
  expect(html).toContain('The Hub is not reachable.')
  expect(html).toContain('https://hub.example.ts.net')
  expect(html).toContain('no answer within 10 s')
  expect(html).toContain('xerxes@hub.example')
  for (const label of ['Hub settings', 'Re-check', 'Not now', 'Don&#x27;t remind me']) expect(html).toContain(label)
})

test('the notice offers setup when no Hub is saved, and stays away when online, hidden or dismissed', () => {
  const setup = view(state({ kind: 'not_configured', target: null, origin: null, detail: null }))
  expect(setup).toContain('No Hub is set up.')
  expect(setup).toContain('Set up a Hub')
  expect(setup).not.toContain('Re-check')
  expect(view(state({ kind: 'online', detail: null }))).toBe('')
  expect(view(state({ kind: 'degraded' }))).toBe('')
  expect(view(state({}), true)).toBe('')
  expect(view(state({ dismissed: { not_configured: false, unreachable: true } }))).toBe('')
  expect(view(undefined)).toBe('')
})

test('states that cross the bridge are validated', () => {
  expect(asHubHealthState(state({}))).toEqual(state({}))
  expect(asHubHealthState({ ...state({}), kind: 'exploded' })).toBeUndefined()
  expect(asHubHealthState({ ...state({}), origin: 'javascript:alert(1)' })).toBeUndefined()
  expect(asHubHealthState({ ...state({}), dismissed: {} })).toBeUndefined()
  expect(hubHealthStateOf({ ok: true, state: state({}) })).toEqual(state({}))
  expect(hubHealthStateOf({ ok: false, code: 'failed', message: 'x' })).toBeUndefined()
})
