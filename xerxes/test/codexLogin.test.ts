// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { loginWithChatGpt } from '../src/auth/codexLogin.js'

test('Codex publishes its URL on a blocked callback port without opening a dead browser flow', async () => {
  const blocker = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: () => new Response('occupied') })
  try {
    const port = blocker.port
    if (port === undefined) throw new Error('Test callback port was not assigned')
    const urls: string[] = []
    let opened = 0
    await expect(loginWithChatGpt({ port, onAuthorizationUrl: url => { urls.push(url) }, openUrl: () => { opened += 1 } })).rejects.toThrow()
    expect(urls).toHaveLength(1)
    expect(urls[0]).toContain('https:')
    expect(opened).toBe(0)
  } finally { blocker.stop(true) }
})

test('Codex delivers the callback page before closing the listener', async () => {
  const reservation = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: () => new Response('reserved') })
  const port = reservation.port
  reservation.stop(true)
  if (!port) throw new Error('Could not reserve a callback port')
  let callbackResponse: Promise<string> | undefined
  const exchanges: string[] = []
  const { token } = await loginWithChatGpt({ port, timeoutMs: 2_000, openUrl: value => {
    const authorize = new URL(value), redirect = authorize.searchParams.get('redirect_uri'), state = authorize.searchParams.get('state')
    if (!redirect || !state) throw new Error('Missing callback parameters')
    callbackResponse = fetch(`${redirect}?code=local-test-code&state=${encodeURIComponent(state)}`).then(response => response.text())
  }, fetchImplementation: async (_url, init) => {
    exchanges.push(String(init?.body ?? ''))
    return Response.json({ access_token: 'local-access', refresh_token: 'local-refresh', expires_in: 3600, token_type: 'Bearer' })
  } })
  expect(token.accessToken).toBe('local-access')
  expect(await callbackResponse).toContain('Signed in to Xerxes')
  expect(exchanges).toHaveLength(1)
  expect(exchanges[0]).toContain('local-test-code')
})
