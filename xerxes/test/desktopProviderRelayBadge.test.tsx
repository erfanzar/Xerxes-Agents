// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { RelayBadge, relaySummary } from '../src/desktop/renderer/RelayBadge.js'
import { providerRelayView, Store } from '../src/desktop/renderer/store.js'

afterEach(() => { delete (globalThis as { window?: unknown }).window })

const activity = { ok: true, bound: true, live: true, destination: 'softnu-local', profile: 'codex', model: 'gpt-6.1-sol', requests: 3, inFlight: 1, lastAt: 1_791_000_000_000, expiresAt: 1_791_028_800_000 }

test('only a conversation bound to this computer produces a badge', () => {
  expect(providerRelayView({ ok: true, bound: false })).toBeNull()
  expect(providerRelayView({ ok: false })).toBeNull()
  expect(providerRelayView(activity)).toEqual({ bound: true, live: true, destination: 'softnu-local', profile: 'codex', model: 'gpt-6.1-sol', requests: 3, inFlight: 1, lastAt: 1_791_000_000_000, expiresAt: 1_791_028_800_000 })
  expect(providerRelayView({ ok: true, bound: true, requests: -1, inFlight: 'x', profile: '' })).toEqual({ bound: true, live: false, requests: 0, inFlight: 0 })
})

test('the badge says where prompts run, pulses while one is in flight, and warns when access ended', () => {
  const live = providerRelayView(activity)!
  const summary = relaySummary(live, 1_791_000_000_000)
  expect(summary).toContain('run on this computer through codex (gpt-6.1-sol), not on softnu-local')
  expect(summary).toContain('3 requests carried by this window · 1 in flight')
  const busy = renderToStaticMarkup(createElement(RelayBadge, { relay: live }))
  expect(busy).toContain('data-state="busy"')
  expect(busy).toContain('via this Mac')
  expect(busy).toContain('relaybadge__dot')
  const idle = renderToStaticMarkup(createElement(RelayBadge, { relay: { ...live, inFlight: 0 } }))
  expect(idle).toContain('data-state="live"')
  expect(idle).not.toContain('relaybadge__dot')
  const ended = { ...live, live: false, inFlight: 0, lastError: 'grant_expired' }
  const markup = renderToStaticMarkup(createElement(RelayBadge, { relay: ended }))
  expect(markup).toContain('data-state="ended"')
  expect(markup).toContain('Mac access ended')
  expect(relaySummary(ended)).toContain('its access has ended. It resumes when this window reconnects.')
  expect(relaySummary(ended)).toContain('Last error: grant_expired.')
})

function desktopWindow(scope: string, answer: unknown) {
  const asked: string[] = []
  let push: (() => void) | undefined
  ;(globalThis as { window?: unknown }).window = {
    xerxes: {
      onEvent: () => () => {},
      call: () => new Promise(() => {}),
      getWorkspace: () => Promise.resolve(null),
      getContextScope: () => Promise.resolve(scope),
      remote: (action: string) => { asked.push(action); return Promise.resolve(answer) },
      onProviderRelay: (handler: () => void) => { push = handler; return () => {} },
    },
  }
  return { asked, push: () => push?.() }
}

test('an SSH window shows the badge when this computer reports carrying its prompts', async () => {
  const desktop = desktopWindow('ssh:softnu-local', activity)
  const store = new Store()
  store.start()
  await Bun.sleep(5)
  desktop.push()
  await Bun.sleep(5)
  expect(desktop.asked).toContain('provider-activity')
  expect(store.getSnapshot().providerRelay).toMatchObject({ live: true, profile: 'codex', inFlight: 1 })
})

test('a local window never asks and never shows the badge', async () => {
  const desktop = desktopWindow('local', activity)
  const store = new Store()
  store.start()
  await Bun.sleep(5)
  desktop.push()
  await Bun.sleep(5)
  expect(desktop.asked).not.toContain('provider-activity')
  expect(store.getSnapshot().providerRelay).toBeNull()
})
