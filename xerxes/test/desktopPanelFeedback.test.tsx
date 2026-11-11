// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MacPermissionsView } from '../src/desktop/renderer/MacPermissionsPanel.js'
import { ChannelsCard } from '../src/desktop/renderer/ChannelsPanel.js'
import { Store } from '../src/desktop/renderer/store.js'
import { AutopilotPanel } from '../src/desktop/renderer/AutopilotPanel.js'
import { WatchPanel } from '../src/desktop/renderer/WatchPanel.js'
import { initialHubState } from '../src/client/hubStore.js'

test('an initial macOS permissions failure has an alert and usable Retry, not an endless spinner', () => {
  const noop = () => {}
  const html = renderToStaticMarkup(createElement(MacPermissionsView, { available: true, state: undefined, busy: null, error: 'Host unavailable', onOpen: noop, onRequest: noop, onRecheck: noop, onRemind: noop, onRelaunch: noop }))
  expect(html).toContain('role="alert"')
  expect(html).toContain('Host unavailable')
  expect(html).toContain('Retry')
  expect(html).not.toContain('Checking permissions…')
})

test('channel refresh owns pending/error feedback, preserves rows on failure, and latest refresh wins', async () => {
  const previous = globalThis.window
  const requests = [Promise.withResolvers<Record<string, unknown>>(), Promise.withResolvers<Record<string, unknown>>(), Promise.withResolvers<Record<string, unknown>>()]
  let count = 0
  Object.assign(globalThis, { window: { xerxes: { call: () => requests[count++]!.promise } } })
  try {
    const store = new Store()
    store.loadChannels()
    expect(store.getSnapshot().channelsLoading).toBe(true)
    requests[0]!.resolve({ ok: true, channels_available: true, channels_configured: true, channels: [{ name: 'telegram', enabled: true }] })
    await Bun.sleep(0)
    store.loadChannels()
    store.loadChannels()
    requests[2]!.reject(new Error('Refresh disconnected'))
    await Bun.sleep(0)
    requests[1]!.resolve({ ok: true, channels_available: false, channels_configured: false, channels: [] })
    await Bun.sleep(0)
    const snap = store.getSnapshot()
    expect(snap.channelsLoading).toBe(false)
    expect(snap.channelsError).toBe('Refresh disconnected')
    expect(snap.channels[0]?.name).toBe('telegram')
    const html = renderToStaticMarkup(createElement(ChannelsCard, { snap }))
    expect(html).toContain('role="alert"')
    expect(html).toContain('Refresh disconnected')
    expect(html).toContain('telegram')
  } finally {
    if (previous === undefined) delete (globalThis as { window?: unknown }).window
    else Object.assign(globalThis, { window: previous })
  }
})

test('unattributed Autopilot and Watch save errors appear before long lists, never below them', () => {
  const state = initialHubState(), onSave = async () => false, onRetry = () => {}
  const autopilot = renderToStaticMarkup(createElement(AutopilotPanel, { autopilot: { ...state.autopilot, phase: 'ready', canEdit: true, saveError: 'Autopilot save failed' }, live: true, onSave, onRetry }))
  const watch = renderToStaticMarkup(createElement(WatchPanel, { watch: { ...state.watch, phase: 'ready', canEdit: true, saveError: 'Watch save failed' }, tasks: [], workspaces: {}, live: true, onSave, onRetry }))
  expect(autopilot.indexOf('Autopilot save failed')).toBeLessThan(autopilot.indexOf('Workspaces'))
  expect(watch.indexOf('Watch save failed')).toBeLessThan(watch.indexOf('Watching'))
})
