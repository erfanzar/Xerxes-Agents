// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  XAI_DEVICE_CODE_URL,
  XAI_OAUTH_CLIENT_ID,
  XAI_OAUTH_SCOPE,
  XAI_TOKEN_URL,
  XaiOAuthSession,
  xaiCredentialFromTokenResponse,
  xaiDeviceCodeFromResponse,
} from '../src/auth/xaiOAuth.js'
import { createLlmClient } from '../src/llms/client.js'

async function inTemporaryHome(run: (home: string) => Promise<void>): Promise<void> {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-xai-oauth-'))
  try {
    await run(home)
  } finally {
    await rm(home, { force: true, recursive: true })
  }
}

const NOW = 1_700_000_000

test('device code parsing validates the https verification URI', () => {
  const device = xaiDeviceCodeFromResponse({
    device_code: 'dc-1',
    user_code: 'WXYZ-1234',
    verification_uri: 'https://auth.x.ai/device',
    verification_uri_complete: 'https://auth.x.ai/device?code=WXYZ-1234',
    interval: 2,
    expires_in: 600,
  })
  expect(device.deviceCode).toBe('dc-1')
  expect(device.verificationUriComplete).toBe('https://auth.x.ai/device?code=WXYZ-1234')
  expect(device.intervalSeconds).toBe(2)

  expect(() => xaiDeviceCodeFromResponse({
    device_code: 'dc-1',
    user_code: 'x',
    verification_uri: 'http://insecure.example.com',
    expires_in: 600,
  })).toThrow(/Untrusted verification URI/)
})

test('token parsing keeps the previous refresh token when xAI does not rotate', () => {
  const rotated = xaiCredentialFromTokenResponse({
    access_token: 'a2',
    refresh_token: 'r2',
    expires_in: 1_800,
  }, NOW)
  expect(rotated).toEqual({ access: 'a2', refresh: 'r2', expires: NOW + 1_800 })

  const unrotated = xaiCredentialFromTokenResponse({
    access_token: 'a3',
    expires_in: 1_800,
  }, NOW, 'r-previous')
  expect(unrotated.refresh).toBe('r-previous')

  // No rotation and no previous refresh token is a hard error.
  expect(() => xaiCredentialFromTokenResponse({ access_token: 'a4' }, NOW)).toThrow(/refresh_token/)
})

test('login runs the device flow and persists the session', async () => {
  await inTemporaryHome(async home => {
    let polls = 0
    const session = new XaiOAuthSession({
      xerxesHome: home,
      environment: {},
      now: () => NOW,
      sleep: (): Promise<void> => Promise.resolve(),
      fetchImplementation: async (url, init) => {
        const target = String(url)
        if (target === XAI_DEVICE_CODE_URL) {
          const body = new URLSearchParams(String(init?.body))
          expect(body.get('client_id')).toBe(XAI_OAUTH_CLIENT_ID)
          expect(body.get('scope')).toBe(XAI_OAUTH_SCOPE)
          return new Response(JSON.stringify({
            device_code: 'dc-1',
            user_code: 'WXYZ-1234',
            verification_uri: 'https://auth.x.ai/device',
            expires_in: 600,
          }))
        }
        expect(target).toBe(XAI_TOKEN_URL)
        polls += 1
        if (polls === 1) return new Response(JSON.stringify({ error: 'authorization_pending' }))
        if (polls === 2) return new Response(JSON.stringify({ error: 'slow_down' }))
        return new Response(JSON.stringify({
          access_token: 'xai-access',
          refresh_token: 'xai-refresh',
          expires_in: 3_600,
        }))
      },
    })
    const credential = await session.login((userCode, verificationUri) => {
      expect(userCode).toBe('WXYZ-1234')
      expect(verificationUri).toBe('https://auth.x.ai/device')
    })
    expect(credential).toEqual({ access: 'xai-access', refresh: 'xai-refresh', expires: NOW + 3_600 })
    expect(await session.logout()).toBe(true)
  })
})

test('refresh keeps the stored refresh token when the response omits rotation', async () => {
  await inTemporaryHome(async home => {
    const session = new XaiOAuthSession({
      xerxesHome: home,
      environment: {},
      now: () => NOW,
      fetchImplementation: async () => new Response(JSON.stringify({
        access_token: 'xai-next',
        expires_in: 1_800,
      })),
    })
    const refreshed = await session.refresh({ access: 'old', refresh: 'xai-refresh', expires: NOW - 1 })
    expect(refreshed).toEqual({ access: 'xai-next', refresh: 'xai-refresh', expires: NOW + 1_800 })
  })
})

test('the xai factory injects the session bearer over the API-key path', async () => {
  const requests: { init?: RequestInit }[] = []
  const session = {
    credential: async () => ({ access: 'xai-session-token', refresh: 'r', expires: NOW + 3_600 }),
  }
  const client = createLlmClient('xai/grok-4', {}, {
    xaiOAuthSession: session as never,
    fetchImplementation: async (_input, init) => {
      requests.push(init === undefined ? {} : { init })
      return new Response(JSON.stringify({
        choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
      }))
    },
  })
  expect(client.complete).toBeTypeOf('function')
  await client.complete!({ model: 'xai/grok-4', messages: [{ role: 'user', content: 'hi' }] })
  const headers = requests[0]?.init?.headers as Record<string, string>
  expect(headers.Authorization).toBe('Bearer xai-session-token')
})
