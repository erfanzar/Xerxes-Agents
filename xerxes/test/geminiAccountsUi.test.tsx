// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { createBridge } from '../src/desktop/bridgeApi.js'
import { AccountRecordedUsage, GeminiAccountForm, ProviderAccountsCenter } from '../src/desktop/renderer/ProviderAccountsCenter.js'

test('the account center offers Gemini API keys, not Gemini OAuth', () => {
  const html = renderToStaticMarkup(createElement(ProviderAccountsCenter, { accounts: async () => ({ accounts: [] }), sessionKey: 's1', turnActive: false, changed: () => {} }))
  expect(html).toContain('<option value="gemini">Gemini (API key)</option>')
  expect(html).toContain('Gemini is API-key only')
  expect(html).toContain('no Gemini OAuth')
})

test('Gemini key entry is masked, never prefilled, and explicitly local until reviewed for SSH', () => {
  const html = renderToStaticMarkup(createElement(GeminiAccountForm, { busy: false, save: async () => {} }))
  expect(html).toContain('type="password"')
  expect(html).toContain('aria-label="Gemini API key"')
  expect(html).toContain('autoComplete="off"')
  expect(html).toContain('Save Gemini account')
  expect(html).toContain('Stored only on this computer')
  expect(html).toContain('Sync to SSH')
  expect(html).not.toContain('fixture-gemini-secret')
})

test('recorded consumption renders separately from quotas with unknown pricing left unknown', () => {
  const html = renderToStaticMarkup(createElement(AccountRecordedUsage, { report: { status: 'available', profile: 'gemini-Work', since: null, calls: 3, input_tokens: 100, output_tokens: 20, cache_read_tokens: 7, cache_write_tokens: 2 } }))
  for (const text of ['Recorded on this workspace', '3 calls', '100 input', '20 output', '7 cache read', '2 cache write', 'Price unavailable']) expect(html).toContain(text)
  expect(html).not.toContain('$0')
  expect(html).not.toContain('remaining')
  const unavailable = renderToStaticMarkup(createElement(AccountRecordedUsage, { report: { status: 'unavailable', profile: 'gemini-Work' } }))
  expect(unavailable).toContain('Recorded usage unavailable')
  expect(unavailable).not.toContain('0 calls')
})

test('the bridge forwards only explicit Gemini creation input and rejects unknown actions', async () => {
  const calls: unknown[][] = []
  const bridge = createBridge({ invoke: async (...args) => { calls.push(args); return { ok: true, account: { id: 'api:gemini-Work' } } }, on: () => () => {}, setZoomFactor: () => {} })
  const result = await bridge.accounts('create-gemini', { label: 'Work', apiKey: 'fixture-gemini-secret', model: 'gemini-2.5-flash' })
  expect(calls).toEqual([['desktop:accounts', 'create-gemini', { label: 'Work', apiKey: 'fixture-gemini-secret', model: 'gemini-2.5-flash' }]])
  expect(JSON.stringify(result)).not.toContain('fixture-gemini-secret')
  await expect(bridge.accounts('gemini-oauth', {})).rejects.toThrow('Unknown provider account action')
})

test('desktop account IPC validates Gemini fields without coercion and dispatches only to the account owner', async () => {
  const main = await Bun.file(new URL('../src/desktop/main.ts', import.meta.url)).text()
  expect(main).toContain("if (action === 'create-gemini')")
  expect(main).toContain("typeof params.apiKey !== 'string'")
  expect(main).toContain('accounts.createGemini(params.label, params.apiKey, params.model)')
})
