// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  RadiusOAuthSession,
  normalizeRadiusGatewayUrl,
} from '../src/auth/radiusOAuth.js'

async function inTemporaryHome(run: (home: string) => Promise<void>): Promise<void> {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-radius-oauth-'))
  try {
    await run(home)
  } finally {
    await rm(home, { force: true, recursive: true })
  }
}

const NOW = 1_700_000_000

test('gateway URLs default the scheme and strip trailing slashes', () => {
  expect(normalizeRadiusGatewayUrl('gw.example.com')).toBe('https://gw.example.com')
  expect(normalizeRadiusGatewayUrl('https://gw.example.com/')).toBe('https://gw.example.com')
  expect(normalizeRadiusGatewayUrl('http://gw.example.com//')).toBe('http://gw.example.com')
})

test('the device flow starts, polls, persists, and binds the credential to the gateway', async () => {
  await inTemporaryHome(async home => {
    let polls = 0
    const session = new RadiusOAuthSession({
      xerxesHome: home,
      environment: {},
      now: () => NOW,
      sleep: (): Promise<void> => Promise.resolve(),
      fetchImplementation: async (url) => {
        const target = String(url)
        if (target.endsWith('/v1/oauth/device')) {
          return new Response(JSON.stringify({
            device_code: 'dc-1',
            user_code: 'ABCD',
            verification_uri: 'https://gw.example.com/activate',
            expires_in: 600,
            interval: 1,
          }))
        }
        if (target.endsWith('/v1/oauth/token')) {
          polls += 1
          if (polls === 1) {
            return new Response(JSON.stringify({ error: 'authorization_pending' }))
          }
          return new Response(JSON.stringify({
            access_token: 'rad-access',
            refresh_token: 'rad-refresh',
            expires_in: 1_200,
            scope: 'gateway',
          }))
        }
        throw new Error(`unexpected fetch: ${target}`)
      },
    })
    const seen: string[] = []
    const credential = await session.login({
      gateway: 'gw.example.com',
      method: 'device',
      onUserCode: (userCode, verificationUri) => seen.push(userCode, verificationUri),
    })
    expect(seen).toEqual(['ABCD', 'https://gw.example.com/activate'])
    expect(credential).toMatchObject({
      gateway: 'https://gw.example.com',
      access: 'rad-access',
      refresh: 'rad-refresh',
      expires: NOW + 1_200,
      scope: 'gateway',
    })

    // Stored credential resolves without any network traffic.
    const stored = await session.credential('gw.example.com')
    expect(stored.access).toBe('rad-access')

    // A different gateway than the stored credential is a usage error.
    const other = new RadiusOAuthSession({ xerxesHome: home, environment: {}, now: () => NOW })
    await expect(other.credential('https://other.example.com')).rejects.toThrow(/No Radius session/)

    expect(await session.logout()).toBe(true)
  })
})

test('refresh posts to the gateway token endpoint and re-persists', async () => {
  await inTemporaryHome(async home => {
    const session = new RadiusOAuthSession({
      xerxesHome: home,
      environment: {},
      now: () => NOW,
      fetchImplementation: async (url, init) => {
        expect(String(url)).toBe('https://gw.example.com/v1/oauth/token')
        const body = new URLSearchParams(String(init?.body))
        expect(body.get('grant_type')).toBe('refresh_token')
        expect(body.get('refresh_token')).toBe('rad-refresh')
        return new Response(JSON.stringify({
          access_token: 'rad-next',
          refresh_token: 'rad-refresh-2',
          expires_in: 900,
        }))
      },
    })
    const refreshed = await session.refresh({
      gateway: 'https://gw.example.com',
      access: 'rad-access',
      refresh: 'rad-refresh',
      expires: NOW - 1,
    })
    expect(refreshed.access).toBe('rad-next')
    expect(refreshed.gateway).toBe('https://gw.example.com')
  })
})

test('a missing gateway is an explicit usage error, not a silent guess', async () => {
  await inTemporaryHome(async home => {
    const session = new RadiusOAuthSession({ xerxesHome: home, environment: {}, now: () => NOW })
    expect(() => session.resolveGateway()).toThrow(/No Radius gateway configured/)
    expect(session.resolveGateway('  gw2.example.com ')).toBe('https://gw2.example.com')
    const fromEnv = new RadiusOAuthSession({
      xerxesHome: home,
      environment: { RADIUS_GATEWAY: 'env-gw.example.com/' },
      now: () => NOW,
    })
    expect(fromEnv.resolveGateway()).toBe('https://env-gw.example.com')
  })
})
