// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ChannelOAuthClient,
  ChannelOAuthStateError,
  channelOAuthStorageKey,
  type ChannelOAuthProvider,
  type ChannelOAuthTokenStore,
} from '../src/channels/index.js'
import { OAuthToken, type OAuthFetch } from '../src/mcp/oauth.js'

const PROVIDER: ChannelOAuthProvider = {
  authorizeUrl: 'https://slack.example.test/oauth/v2/authorize?prompt=consent',
  clientId: 'client-id',
  clientSecret: 'client-secret',
  extraAuthorizeParams: { access_type: 'offline' },
  name: 'slack',
  redirectUri: 'https://xerxes.example.test/oauth/callback',
  scopes: ['chat:write', 'users:read'],
  tokenUrl: 'https://slack.example.test/oauth/v2/access',
}

class MemoryTokenStore implements ChannelOAuthTokenStore {
  readonly tokens = new Map<string, OAuthToken>()

  async load(key: string): Promise<OAuthToken | undefined> {
    return this.tokens.get(key)
  }

  async save(key: string, token: OAuthToken): Promise<void> {
    this.tokens.set(key, token)
  }
}

test('channel OAuth persists authorization-code tokens independently per installation', async () => {
  const storage = new MemoryTokenStore()
  const requests: URLSearchParams[] = []
  const fetchImplementation: OAuthFetch = async (_input, init) => {
    requests.push(new URLSearchParams(String(init?.body)))
    return new Response(JSON.stringify({
      access_token: 'workspace-access-token',
      expires_in: 3_600,
      refresh_token: 'workspace-refresh-token',
      scope: 'chat:write users:read',
      token_type: 'Bearer',
    }), { headers: { 'Content-Type': 'application/json' } })
  }
  const client = new ChannelOAuthClient(PROVIDER, {
    fetchImplementation,
    now: () => 1_000,
    stateGenerator: () => 'fixed-state',
    storage,
  })

  const authorization = client.beginAuthorize()
  const url = new URL(authorization.url)
  expect(authorization.state).toBe('fixed-state')
  expect(url.searchParams.get('prompt')).toBe('consent')
  expect(url.searchParams.get('response_type')).toBe('code')
  expect(url.searchParams.get('client_id')).toBe('client-id')
  expect(url.searchParams.get('redirect_uri')).toBe(PROVIDER.redirectUri)
  expect(url.searchParams.get('scope')).toBe('chat:write users:read')
  expect(url.searchParams.get('state')).toBe('fixed-state')
  expect(url.searchParams.get('access_type')).toBe('offline')

  const token = await client.completeAuthorize({
    code: 'authorization-code',
    installId: 'workspace-a',
    state: authorization.state,
  })

  expect(requests).toEqual([new URLSearchParams({
    client_id: 'client-id',
    client_secret: 'client-secret',
    code: 'authorization-code',
    grant_type: 'authorization_code',
    redirect_uri: PROVIDER.redirectUri,
  })])
  expect(token.toRecord()).toEqual({
    access_token: 'workspace-access-token',
    expires_at: 4_600,
    refresh_token: 'workspace-refresh-token',
    scopes: ['chat:write', 'users:read'],
    token_type: 'Bearer',
  })
  expect(await client.getToken('workspace-a')).toEqual(token)
  expect(await client.getToken('workspace-b')).toBeUndefined()
  expect(storage.tokens.get(channelOAuthStorageKey('slack', 'workspace-a'))).toEqual(token)
  expect(channelOAuthStorageKey('slack', 'workspace-a')).not.toContain('workspace-a')
})

test('channel OAuth states are one-time, expire, and cannot be overridden by provider parameters', async () => {
  const storage = new MemoryTokenStore()
  let now = 100
  let requests = 0
  const client = new ChannelOAuthClient(PROVIDER, {
    fetchImplementation: async () => {
      requests += 1
      return new Response(JSON.stringify({ access_token: 'access' }), {
        headers: { 'Content-Type': 'application/json' },
      })
    },
    now: () => now,
    stateGenerator: () => 'state',
    stateTtlSeconds: 600,
    storage,
  })

  const first = client.beginAuthorize()
  await client.completeAuthorize({ code: 'code', state: first.state })
  await expect(client.completeAuthorize({ code: 'code', state: first.state })).rejects.toBeInstanceOf(ChannelOAuthStateError)
  expect(requests).toBe(1)

  const expired = client.beginAuthorize()
  now = 701
  await expect(client.completeAuthorize({ code: 'code', state: expired.state })).rejects.toBeInstanceOf(ChannelOAuthStateError)
  expect(requests).toBe(1)

  expect(() => new ChannelOAuthClient({
    ...PROVIDER,
    extraAuthorizeParams: { state: 'attacker-controlled-state' },
  }, { storage })).toThrow('controlled by ChannelOAuthClient')
})

test('channel OAuth coalesces concurrent per-install refreshes and retains rotated credentials safely', async () => {
  const storage = new MemoryTokenStore()
  const installId = 'workspace-a'
  const key = channelOAuthStorageKey('slack', installId)
  await storage.save(key, new OAuthToken({
    accessToken: 'expired-access-token',
    expiresAt: 10,
    refreshToken: 'long-lived-refresh-token',
  }))

  let requests = 0
  let releaseResponse: (() => void) | undefined
  const responseGate = new Promise<void>(resolve => {
    releaseResponse = resolve
  })
  const client = new ChannelOAuthClient(PROVIDER, {
    fetchImplementation: async () => {
      requests += 1
      await responseGate
      return new Response(JSON.stringify({ access_token: 'fresh-access-token', expires_in: 3_600 }), {
        headers: { 'Content-Type': 'application/json' },
      })
    },
    now: () => 100,
    refreshSkewSeconds: 60,
    storage,
  })

  const first = client.getValidToken(installId)
  const second = client.getValidToken(installId)
  await waitFor(() => requests === 1)
  const release = releaseResponse
  if (!release) {
    throw new Error('refresh request did not start')
  }
  release()

  const [firstToken, secondToken] = await Promise.all([first, second])
  expect(requests).toBe(1)
  expect(firstToken?.toRecord()).toEqual({
    access_token: 'fresh-access-token',
    expires_at: 3_700,
    refresh_token: 'long-lived-refresh-token',
    scopes: [],
    token_type: 'Bearer',
  })
  expect(secondToken).toEqual(firstToken)
  expect((await storage.load(key))?.refreshToken).toBe('long-lived-refresh-token')
})

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) {
      return
    }
    await new Promise<void>(resolve => setTimeout(resolve, 0))
  }
  throw new Error('condition was not reached')
}
