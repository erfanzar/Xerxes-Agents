// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { runBunProviderAccountLogin } from '../src/desktop/main/providerAccountLogin.js'
import { loginWithChatGpt } from '../src/auth/codexLogin.js'

test('OAuth login uses a standalone Bun helper instead of the running daemon RPC', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-oauth-helper-'))
  try {
    const entry = join(directory, 'helper.ts')
    await writeFile(entry, 'if (process.argv[2] !== "provider-account-login" || process.argv[3] !== "account-1") process.exit(2)\n')
    await runBunProviderAccountLogin('account-1', {
      entry,
      env: { ...process.env, XERXES_BUN: process.execPath },
    })
    const main = await Bun.file(new URL('../src/desktop/main.ts', import.meta.url)).text()
    expect(main).toContain('await runBunProviderAccountLogin(accountId, { onAuthorizationUrl })')
    expect(main).toContain('older\n          // daemon running which does not know provider_account.login')
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('OAuth helper failures remain observable without leaking process arguments', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-oauth-helper-fail-'))
  try {
    const entry = join(directory, 'helper.ts')
    await writeFile(entry, 'console.error("login failed safely"); process.exit(1)\n')
    await expect(runBunProviderAccountLogin('account-1', {
      entry,
      env: { ...process.env, XERXES_BUN: process.execPath },
    })).rejects.toThrow('login failed safely')
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('Codex publishes the copyable URL even when its callback port is occupied', async () => {
  const blocker = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: () => new Response('occupied') })
  try {
    const port = blocker.port
    if (port === undefined) throw new Error('Test callback port was not assigned')
    const urls: string[] = []
    let opened = 0
    await expect(loginWithChatGpt({
      port,
      onAuthorizationUrl: url => { urls.push(url) },
      openUrl: () => { opened += 1 },
    })).rejects.toThrow()
    expect(urls).toHaveLength(1)
    expect(urls[0]).toContain('https://')
    expect(opened).toBe(0)
  } finally { blocker.stop(true) }
})

test('Codex returns the callback page before closing its listener', async () => {
  const reservation = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: () => new Response('reserved') })
  const port = reservation.port
  reservation.stop(true)
  if (!port) throw new Error('Could not reserve a callback port')
  const exchanges: string[] = []
  let callbackResponse: Promise<string> | undefined
  const { token } = await loginWithChatGpt({
    port,
    timeoutMs: 2_000,
    openUrl: value => {
      const authorize = new URL(value)
      const redirect = authorize.searchParams.get('redirect_uri')
      const state = authorize.searchParams.get('state')
      if (!redirect || !state) throw new Error('Authorize URL is missing callback parameters')
      callbackResponse = fetch(`${redirect}?code=local-test-code&state=${encodeURIComponent(state)}`).then(response => response.text())
    },
    fetchImplementation: async (_url, init) => {
      exchanges.push(String(init?.body ?? ''))
      return new Response(JSON.stringify({ access_token: 'local-access', refresh_token: 'local-refresh', expires_in: 3600, token_type: 'Bearer' }))
    },
  })
  expect(token.accessToken).toBe('local-access')
  expect(await callbackResponse).toContain('Signed in to Xerxes')
  expect(exchanges).toHaveLength(1)
  expect(exchanges[0]).toContain('local-test-code')
})

test('the helper forwards the authorize URL before login completes', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-oauth-helper-url-'))
  try {
    const entry = join(directory, 'helper.ts')
    const url = 'https://auth.example.com/authorize?state=public-state'
    await writeFile(entry, `console.log(${JSON.stringify(JSON.stringify({ type: 'authorization_url', url }))}); console.error('Failed to start server. Is port 1455 in use?'); process.exit(1)\n`)
    const urls: string[] = []
    await expect(runBunProviderAccountLogin('account-1', {
      entry,
      env: { ...process.env, XERXES_BUN: process.execPath },
      onAuthorizationUrl: value => { urls.push(value) },
    })).rejects.toThrow('port 1455')
    expect(urls).toEqual([url])
  } finally { await rm(directory, { recursive: true, force: true }) }
})
