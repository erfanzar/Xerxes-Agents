// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { expect, test } from 'bun:test'

import { createBridge } from '../src/desktop/bridgeApi.js'
import { ProviderAccountsCenter, ProviderLoginLink, formatAccountUsageWindow } from '../src/desktop/renderer/ProviderAccountsCenter.js'
import { providerAccountUsageWire } from '../src/desktop/main/globalProviderAccounts.js'
import { ActionDialog } from '../src/desktop/renderer/ActionDialog.js'

test('the global provider account center is available on Mac and SSH', () => {
  const html = renderToStaticMarkup(createElement(ProviderAccountsCenter, { accounts: async () => ({ accounts: [] }), sessionKey: 's1', turnActive: false, changed: () => {} }))
  for (const text of ['Provider accounts', 'One account center for this Mac and every SSH workspace', 'Claude', 'Codex', 'Sign in']) expect(html).toContain(text)
})

test('the sign-in URL stays visible and copyable while OAuth is unfinished', () => {
  const url = 'https://auth.example.com/authorize?state=public-state'
  const html = renderToStaticMarkup(createElement(ProviderLoginLink, { url, busy: true }))
  expect(html).toContain('Copy sign-in link')
  expect(html).toContain(`value="${url}"`)
  expect(html).toContain('Nothing is registered until sign-in completes')
})

test('OAuth usage is normalized and malformed percentages never render NaN', () => {
  const wire = providerAccountUsageWire({
    provider: 'claude', fetchedAt: Date.UTC(2026, 10, 10),
    windows: [{ label: '5-hour', usedPercent: 42.4, resetsAt: Date.UTC(2026, 10, 10, 12) }],
  })
  expect(wire).toMatchObject({ status: 'available', windows: [{ label: '5-hour', used_percent: 42.4, resets_at: '2026-11-10T12:00:00.000Z' }] })
  expect(formatAccountUsageWindow({ label: '5-hour', used_percent: Number.NaN, resets_at: null, detail: null })).toBe('5-hour: usage unavailable')
  expect(providerAccountUsageWire({ windows: [] })).toMatchObject({ status: 'unknown' })
  expect(() => providerAccountUsageWire({ windows: [{ label: 'weekly', usedPercent: Number.NaN }] })).toThrow('Invalid provider usage window')
})

test('account actions and usage cannot overlap', async () => {
  const css = await Bun.file(new URL('../src/desktop/renderer/atelier.css', import.meta.url)).text()
  expect(css).toContain('.provider-account .provcard__tools{position:static')
  expect(css).toContain('.provider-account__usage{display:flex;flex-direction:column')
  expect(css).toContain('overflow-wrap:anywhere')
})

test('credential-transfer confirmation is a named modal with visible progress and errors', () => {
  const html = renderToStaticMarkup(createElement(ActionDialog, {
    title: 'Sync Personal to SSH', confirmLabel: 'Confirm sync', busyLabel: 'Syncing…',
    busy: true, onConfirm: () => {}, onCancel: () => {}, error: 'Transfer failed. Nothing was acknowledged.',
    children: createElement('p', null, 'Transfer Personal to the training host?'),
  }))
  expect(html).toContain('role="dialog"')
  expect(html).toContain('aria-modal="true"')
  expect(html).toContain('aria-busy="true"')
  expect(html).toContain('Syncing…')
  expect(html).toContain('role="alert"')
  expect(html).toContain('Transfer Personal to the training host?')
})

test('the preload forwards global account actions without exposing credentials', async () => {
  const calls: unknown[][] = []
  const bridge = createBridge({ invoke: async (...args) => { calls.push(args); return { ok: true, accounts: [] } }, on: () => () => {}, setZoomFactor: () => {} })
  for (const action of ['list', 'login', 'copy-login-link', 'activate', 'usage', 'sync-review', 'sync', 'remove']) {
    await bridge.accounts(action, action === 'activate' ? { sessionKey: 's1' } : {})
  }
  expect(calls.map(call => call[1])).toEqual(['list', 'login', 'copy-login-link', 'activate', 'usage', 'sync-review', 'sync', 'remove'])
  expect(calls.every(call => call[0] === 'desktop:accounts')).toBe(true)
})

test('the preload forwards authorize URLs without waiting for OAuth completion', () => {
  const handlers = new Map<string, (value: unknown) => void>()
  const bridge = createBridge({ invoke: async () => ({ ok: true }), on: (channel, handler) => { handlers.set(channel, handler); return () => handlers.delete(channel) }, setZoomFactor: () => {} })
  const urls: string[] = []
  bridge.onAccountLogin?.(event => { if (event.type === 'authorization_url') urls.push(event.url) })
  handlers.get('desktop:account-login')?.({ type: 'authorization_url', login_id: 'login-1', url: 'https://auth.example.com/authorize' })
  expect(urls).toEqual(['https://auth.example.com/authorize'])
})
