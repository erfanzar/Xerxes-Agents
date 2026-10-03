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
  expect(markup).toContain('Mac not connected')
  expect(relaySummary(ended)).toContain('which is not connected to it right now. It continues once this window reconnects.')
  // While this window is restoring the link, it says so instead of giving up.
  const restoring = { ...ended, reconnecting: true, lastFailure: 'The conversation changed or is working.' }
  const reconnecting = renderToStaticMarkup(createElement(RelayBadge, { relay: restoring }))
  expect(reconnecting).toContain('data-state="reconnecting"')
  expect(reconnecting).toContain('Reconnecting to this Mac…')
  expect(reconnecting).toContain('relaybadge__dot')
  expect(relaySummary(restoring)).toContain('this window is restoring it and retries until it is back')
  expect(relaySummary({ ...restoring, profile: undefined })).toContain('runs through its provider on this computer')
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

test('Settings rows say which provider an SSH task really uses', async () => {
  const { providerRowState } = await import('../src/desktop/renderer/store.js')
  const ssh = { storageScope: 'ssh:softnu-local:' }
  const relay = { bound: true, live: true, profile: 'codex', requests: 1, inFlight: 0 }
  // Running through this computer: that row is in use, the host's active one is only its default.
  expect(providerRowState({ ...ssh, providerRelay: relay }, { name: 'codex', active: false, signsIn: true })).toEqual({ inUse: true, viaMac: true, chip: 'via this Mac', status: 'in use' })
  expect(providerRowState({ ...ssh, providerRelay: relay }, { name: 'zai', active: true, signsIn: false })).toEqual({ inUse: false, viaMac: false, chip: 'host default', status: 'saved' })
  expect(providerRowState({ ...ssh, providerRelay: { ...relay, live: false } }, { name: 'codex', active: false, signsIn: true })).toMatchObject({ inUse: true, chip: 'Mac not connected' })
  expect(providerRowState({ ...ssh, providerRelay: { ...relay, live: false, reconnecting: true } }, { name: 'codex', active: false, signsIn: true })).toMatchObject({ inUse: true, chip: 'reconnecting' })
  // On the host's own providers: the active keyed one is in use; sign-in ones run on this Mac when picked.
  expect(providerRowState({ ...ssh, providerRelay: null }, { name: 'kimi', active: true, signsIn: false })).toEqual({ inUse: true, viaMac: false, chip: 'active', status: 'in use' })
  expect(providerRowState({ ...ssh, providerRelay: null }, { name: 'claude-code', active: false, signsIn: true })).toEqual({ inUse: false, viaMac: true, chip: null, status: 'runs on this Mac' })
  // A local window is unchanged.
  expect(providerRowState({ storageScope: '', providerRelay: null }, { name: 'codex', active: true, signsIn: true })).toEqual({ inUse: true, viaMac: false, chip: 'active', status: 'in use' })
})
