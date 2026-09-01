// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  OpenRouterOAuthSession,
  exchangeOpenRouterAuthorizationCode,
  parseOpenRouterAuthorizationInput,
} from '../src/auth/openrouterOAuth.js'
import { createLlmClient } from '../src/llms/client.js'

async function inTemporaryHome(run: (home: string) => Promise<void>): Promise<void> {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-openrouter-oauth-'))
  try {
    await run(home)
  } finally {
    await rm(home, { force: true, recursive: true })
  }
}

const NOW = 1_700_000_000

test('authorization input parses URLs, query strings, and bare codes', () => {
  expect(parseOpenRouterAuthorizationInput('https://example.com/cb?code=abc&state=x')).toBe('abc')
  expect(parseOpenRouterAuthorizationInput('code=abc')).toBe('abc')
  expect(parseOpenRouterAuthorizationInput(' bare ')).toBe('bare')
  expect(parseOpenRouterAuthorizationInput('')).toBeUndefined()
})

test('key exchange posts the pi-ai shape and stores a never-expiring credential', async () => {
  const requests: { url: string; init?: RequestInit }[] = []
  const credential = await exchangeOpenRouterAuthorizationCode('the-code', 'the-verifier', undefined, {
    fetchImplementation: async (url, init) => {
      requests.push({ url: String(url), ...(init === undefined ? {} : { init }) })
      return new Response(JSON.stringify({ key: 'sk-or-v1-permanent' }))
    },
  })
  expect(requests[0]?.url).toBe('https://openrouter.ai/api/v1/auth/keys')
  expect(JSON.parse(String(requests[0]?.init?.body))).toEqual({
    code: 'the-code',
    code_verifier: 'the-verifier',
    code_challenge_method: 'S256',
  })
  expect(credential).toEqual({ access: 'sk-or-v1-permanent', refresh: '', expires: Number.MAX_SAFE_INTEGER })
})

test('exchange failures surface the provider error detail', async () => {
  await expect(exchangeOpenRouterAuthorizationCode('bad', 'verifier', undefined, {
    fetchImplementation: async () => new Response(JSON.stringify({ error: { message: 'invalid code' } }), { status: 400 }),
  } as never)).rejects.toThrow(/invalid code/)
})

test('login exchanges a manually pasted code and persists the key', async () => {
  await inTemporaryHome(async home => {
    const session = new OpenRouterOAuthSession({ xerxesHome: home, now: () => NOW })
    const credential = await session.login({
      openUrl: () => {},
      timeoutMs: 2_000,
      manualInput: async () => 'pasted-code',
      fetchImplementation: async () => new Response(JSON.stringify({ key: 'exchanged-key' })),
    })
    expect(credential.access).toBe('exchanged-key')
    const stored = JSON.parse(await readFile(join(home, 'auth', 'openrouter-oauth.json'), 'utf8')) as Record<string, unknown>
    expect(stored['access']).toBe('exchanged-key')
    expect(await session.logout()).toBe(true)
  })
})

test('credential resolution: stored key wins, missing session is a usage error', async () => {
  await inTemporaryHome(async home => {
    const session = new OpenRouterOAuthSession({ xerxesHome: home, now: () => NOW })
    await expect(session.credential()).rejects.toThrow(/No OpenRouter OAuth session/)

    await session.login({
      openUrl: () => {},
      timeoutMs: 2_000,
      manualInput: async () => 'code-2',
      fetchImplementation: async () => new Response(JSON.stringify({ key: 'exchanged-key' })),
    })
    const credential = await session.credential()
    expect(credential.access).toBe('exchanged-key')
    // Identity refresh: the stored key is the credential.
    await expect(session.refresh(credential)).resolves.toBe(credential)
  })
})

test('the openrouter factory injects the session bearer over the API-key path', async () => {
  const requests: { init?: RequestInit }[] = []
  const session = {
    credential: async () => ({ access: 'or-session-key', refresh: '', expires: Number.MAX_SAFE_INTEGER }),
  }
  const client = createLlmClient('openrouter/openai/gpt-5', {}, {
    openrouterOAuthSession: session as never,
    fetchImplementation: async (_input, init) => {
      requests.push(init === undefined ? {} : { init })
      return new Response(JSON.stringify({
        choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
      }))
    },
  })
  expect(client.complete).toBeTypeOf('function')
  await client.complete!({ model: 'openrouter/openai/gpt-5', messages: [{ role: 'user', content: 'hi' }] })
  const headers = requests[0]?.init?.headers as Record<string, string>
  expect(headers.Authorization).toBe('Bearer or-session-key')
})
