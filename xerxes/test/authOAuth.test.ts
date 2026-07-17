// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { OAuthClient, anthropicPreset, copilotPreset, githubPatPreset, openaiPreset } from '../src/auth/oauth.js'
import { CredentialStorage } from '../src/auth/storage.js'
import {
  OAuthToken,
  buildAuthorizeUrl,
  exchangeCode,
  generatePkcePair,
  refreshToken,
  type OAuthConfig,
  type OAuthFetch,
} from '../src/mcp/oauth.js'

const CONFIG: OAuthConfig = {
  clientId: 'client-id',
  authorizeUrl: 'https://oauth.example.test/authorize?prompt=consent',
  tokenUrl: 'https://oauth.example.test/token',
  scopes: ['read', 'write'],
}

test('OAuth PKCE generation, authorize URLs, and token parsing follow OAuth 2.1', () => {
  const pair = generatePkcePair({ verifier: 'fixed-verifier' })
  expect(pair).toEqual({
    verifier: 'fixed-verifier',
    challenge: '7MosA1dS6hiqNcSny0SqUWJbJo82pR0lNczg5YZ-GLI',
  })

  const parsed = new URL(buildAuthorizeUrl(CONFIG, { state: 'state-value', codeChallenge: pair.challenge }))
  expect(parsed.searchParams.get('prompt')).toBe('consent')
  expect(parsed.searchParams.get('response_type')).toBe('code')
  expect(parsed.searchParams.get('client_id')).toBe('client-id')
  expect(parsed.searchParams.get('redirect_uri')).toBe('http://127.0.0.1:5454/callback')
  expect(parsed.searchParams.get('state')).toBe('state-value')
  expect(parsed.searchParams.get('code_challenge')).toBe(pair.challenge)
  expect(parsed.searchParams.get('code_challenge_method')).toBe('S256')
  expect(parsed.searchParams.get('scope')).toBe('read write')

  const token = OAuthToken.fromResponse({
    access_token: 'access', refresh_token: 'refresh', token_type: 'Bearer', expires_in: 60, scope: 'read write',
  }, 1_000)
  expect(token.toRecord()).toEqual({
    access_token: 'access', refresh_token: 'refresh', token_type: 'Bearer', expires_at: 1_060, scopes: ['read', 'write'],
  })
  expect(token.isExpired(30, 1_029)).toBeFalse()
  expect(token.isExpired(30, 1_030)).toBeTrue()
})

test('OAuth exchange and refresh use form posts and preserve omitted refresh tokens', async () => {
  const requests: Array<{ readonly body: URLSearchParams; readonly url: string }> = []
  let requestCount = 0
  const fetchImplementation: OAuthFetch = async (input, init) => {
    requests.push({ body: new URLSearchParams(String(init?.body)), url: String(input) })
    requestCount += 1
    if (requestCount === 1) {
      return new Response(JSON.stringify({ access_token: 'first', refresh_token: 'long-lived', expires_in: 60 }), {
        headers: { 'Content-Type': 'application/json' },
      })
    }
    return new Response('access_token=second&expires_in=120', {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    })
  }

  const issued = await exchangeCode(CONFIG, {
    code: 'authorization-code', codeVerifier: 'verifier', fetchImplementation, now: () => 100,
  })
  const refreshed = await refreshToken(CONFIG, issued, { fetchImplementation, now: () => 200 })

  expect(requests).toHaveLength(2)
  expect(requests[0]?.url).toBe(CONFIG.tokenUrl)
  expect(requests[0]?.body).toEqual(new URLSearchParams({
    grant_type: 'authorization_code', code: 'authorization-code', redirect_uri: 'http://127.0.0.1:5454/callback',
    client_id: 'client-id', code_verifier: 'verifier',
  }))
  expect(requests[1]?.body).toEqual(new URLSearchParams({
    grant_type: 'refresh_token', refresh_token: 'long-lived', client_id: 'client-id',
  }))
  expect(refreshed.toRecord()).toEqual({
    access_token: 'second', refresh_token: 'long-lived', token_type: 'Bearer', expires_at: 320, scopes: [],
  })
})

test('provider presets and OAuthClient begin/finish flow are usable without an SDK', async () => {
  expect(githubPatPreset('github').authorizeUrl).toContain('github.com')
  expect(openaiPreset('openai').tokenUrl).toContain('openai.com')
  expect(anthropicPreset('anthropic').tokenUrl).toContain('anthropic.com')
  expect(copilotPreset('copilot').scopes).toEqual(['copilot'])

  const client = new OAuthClient(githubPatPreset('client-id'), {
    stateGenerator: () => 'fixed-state',
    fetchImplementation: async () => new Response(JSON.stringify({ access_token: 'access' }), {
      headers: { 'Content-Type': 'application/json' },
    }),
  })
  const context = client.beginAuthorize()
  expect(context.state).toBe('fixed-state')
  expect(context.codeVerifier).not.toBe('')
  expect(new URL(context.url).searchParams.get('state')).toBe('fixed-state')
  await expect(client.finishAuthorize('code', 'verifier')).resolves.toMatchObject({ accessToken: 'access' })
})

test('credential storage encrypts tokens, writes restricted atomic files, and rejects path escapes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-oauth-'))
  const credentialDirectory = join(root, 'credentials')
  const storage = new CredentialStorage(credentialDirectory, { credentialKey: 'test-only-key' })
  const first = new OAuthToken({ accessToken: 'secret-first', refreshToken: 'refresh-token', scopes: ['read'] })
  try {
    const path = await storage.save('github', first)
    const encrypted = await readFile(path, 'utf8')
    expect(encrypted).not.toContain('secret-first')
    expect(encrypted).toContain('aes-256-gcm')
    expect(await storage.load('github')).toEqual(first)

    await storage.save('github', new OAuthToken({ accessToken: 'secret-second' }))
    expect((await storage.load('github'))?.accessToken).toBe('secret-second')
    expect(await storage.listProviders()).toEqual(['github'])
    expect((await readdir(credentialDirectory)).some(name => name.endsWith('.tmp'))).toBeFalse()
    if (process.platform !== 'win32') {
      expect((await stat(path)).mode & 0o777).toBe(0o600)
    }

    await writeFile(join(credentialDirectory, 'legacy.json'), JSON.stringify({
      access_token: 'legacy-token', refresh_token: null, token_type: 'Bearer', expires_at: null, scopes: [],
    }), 'utf8')
    expect((await storage.load('legacy'))?.accessToken).toBe('legacy-token')
    await expect(storage.save('../outside', first)).rejects.toThrow('Credential provider')
    expect(await storage.remove('github')).toBeTrue()
    expect(await storage.remove('github')).toBeFalse()
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})


test('credential storage propagates real filesystem errors instead of masking them as logged-out', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-oauth-errors-'))
  try {
    const storage = new CredentialStorage(join(root, 'credentials'), { credentialKey: 'test-only-key' })
    // A missing credential file is a genuine "not logged in".
    expect(await storage.load('github')).toBeUndefined()
    // A directory where the credential file belongs is a real error (EISDIR) and must surface.
    await mkdir(join(root, 'credentials', 'github.json'), { recursive: true })
    await expect(storage.load('github')).rejects.toMatchObject({ code: 'EISDIR' })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
