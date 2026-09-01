// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  KIMI_CODE_OAUTH_CLIENT_ID,
  KimiCodingOAuthSession,
  kimiCodeOauthHost,
  startKimiCodeDeviceAuthorization,
} from '../src/auth/kimiCodingOAuth.js'
import { createLlmClient, OpenAiCompatibleClient } from '../src/llms/client.js'

async function inTemporaryHome(run: (home: string) => Promise<void>): Promise<void> {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-kimi-oauth-'))
  try {
    await run(home)
  } finally {
    await rm(home, { force: true, recursive: true })
  }
}

const NOW = 1_700_000_000
const instantSleep = (): Promise<void> => Promise.resolve()

test('the oauth host honours KIMI_CODE_OAUTH_HOST then KIMI_OAUTH_HOST and strips slashes', () => {
  expect(kimiCodeOauthHost({})).toBe('https://auth.kimi.com')
  expect(kimiCodeOauthHost({ KIMI_OAUTH_HOST: 'https://mirror.example.com/' })).toBe('https://mirror.example.com')
  expect(kimiCodeOauthHost({ KIMI_CODE_OAUTH_HOST: 'https://a.example.com//', KIMI_OAUTH_HOST: 'https://b.example.com' }))
    .toBe('https://a.example.com')
})

test('device authorization validates and normalizes the pi-ai response fields', async () => {
  const device = await startKimiCodeDeviceAuthorization('https://auth.example.com', {
    fetchImplementation: async () => new Response(JSON.stringify({
      device_code: 'dc-1',
      user_code: 'ABCD-EFGH',
      verification_uri: 'https://auth.kimi.com/device',
      verification_uri_complete: 'https://auth.kimi.com/device?code=ABCD-EFGH',
      interval: 3,
      expires_in: 600,
    })),
  })
  expect(device.deviceCode).toBe('dc-1')
  expect(device.verificationUriComplete).toBe('https://auth.kimi.com/device?code=ABCD-EFGH')
  expect(device.intervalSeconds).toBe(3)
  expect(device.expiresInSeconds).toBe(600)

  await expect(startKimiCodeDeviceAuthorization('https://auth.example.com', {
    fetchImplementation: async () => new Response(JSON.stringify({
      device_code: 'dc-1',
      user_code: 'x',
      verification_uri: 'file:///etc/passwd',
      verification_uri_complete: 'https://ok.example.com',
    })),
  })).rejects.toThrow(/Invalid Kimi Code device authorization/)
})

test('login polls through pending then completes, persisting the session', async () => {
  await inTemporaryHome(async home => {
    let polls = 0
    const session = new KimiCodingOAuthSession({
      xerxesHome: home,
      environment: {},
      now: () => NOW,
      sleep: instantSleep,
      fetchImplementation: async (url, init) => {
        const target = String(url)
        if (target.endsWith('/api/oauth/device_authorization')) {
          expect(String(init?.body)).toBe(`client_id=${encodeURIComponent(KIMI_CODE_OAUTH_CLIENT_ID)}`)
          return new Response(JSON.stringify({
            device_code: 'dc-1',
            user_code: 'ABCD-EFGH',
            verification_uri: 'https://auth.example.com/device',
            verification_uri_complete: 'https://auth.example.com/device?code=x',
            interval: 1,
            expires_in: 600,
          }))
        }
        polls += 1
        if (polls === 1) {
          return new Response(JSON.stringify({ error: 'authorization_pending' }))
        }
        return new Response(JSON.stringify({
          access_token: 'kc-access',
          refresh_token: 'kc-refresh',
          expires_in: 3_600,
        }))
      },
    })
    const credential = await session.login((userCode, verificationUri) => {
      expect(userCode).toBe('ABCD-EFGH')
      expect(verificationUri).toContain('https://')
    })
    expect(credential).toEqual({ access: 'kc-access', refresh: 'kc-refresh', expires: NOW + 3_600 })
    expect(await session.logout()).toBe(true)
  })
})

test('refresh retries 5xx then succeeds; 401 is fatal', async () => {
  await inTemporaryHome(async home => {
    let attempts = 0
    const session = new KimiCodingOAuthSession({
      xerxesHome: home,
      environment: {},
      now: () => NOW,
      sleep: instantSleep,
      fetchImplementation: async () => {
        attempts += 1
        if (attempts < 3) return new Response('{}', { status: 500 })
        return new Response(JSON.stringify({
          access_token: 'kc-next',
          refresh_token: 'kc-refresh-2',
          expires_in: 1_800,
        }))
      },
    })
    const refreshed = await session.refresh({ access: 'old', refresh: 'kc-refresh', expires: NOW - 1 })
    expect(refreshed.access).toBe('kc-next')
    expect(attempts).toBe(3)

    const dead = new KimiCodingOAuthSession({
      xerxesHome: home,
      environment: {},
      now: () => NOW,
      sleep: instantSleep,
      fetchImplementation: async () => new Response(JSON.stringify({ error: 'invalid_grant' }), { status: 401 }),
    })
    await expect(dead.refresh({ access: 'old', refresh: 'kc-refresh', expires: NOW - 1 }))
      .rejects.toThrow(/unauthorized/)
  })
})

test('the kimi-code factory injects the session bearer over the API-key path', async () => {
  const requests: { init?: RequestInit }[] = []
  const session = {
    credential: async () => ({ access: 'kc-session-token', refresh: 'r', expires: NOW + 3_600 }),
  }
  const client = createLlmClient('kimi-code/kimi-k3', {}, {
    kimiOAuthSession: session as never,
    fetchImplementation: async (_input, init) => {
      requests.push(init === undefined ? {} : { init })
      return new Response(JSON.stringify({
        choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
      }))
    },
  })
  expect(client).toBeInstanceOf(OpenAiCompatibleClient)
  expect(client.complete).toBeTypeOf('function')
  await client.complete!({
    model: 'kimi-code/kimi-k3',
    messages: [{ role: 'user', content: 'hi' }],
  })
  const headers = requests[0]?.init?.headers as Record<string, string>
  expect(headers.Authorization).toBe('Bearer kc-session-token')
})
