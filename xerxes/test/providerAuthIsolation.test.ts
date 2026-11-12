// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { AnthropicOAuthSession } from '../src/auth/anthropicOAuth.js'
import { createLlmClient, type CompletionRequest, type FetchImplementation } from '../src/llms/client.js'

const request: CompletionRequest = {
  model: 'claude-sonnet-4-6',
  messages: [{ role: 'user', content: 'hello' }],
}

async function withFakeAmbientAnthropicAuth(run: (home: string) => Promise<void>): Promise<void> {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-provider-auth-isolation-'))
  const previous = {
    XERXES_HOME: process.env.XERXES_HOME,
    ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY,
    ANTHROPIC_AUTH_TOKEN: process.env.ANTHROPIC_AUTH_TOKEN,
  }
  process.env.XERXES_HOME = home
  process.env.ANTHROPIC_API_KEY = 'fake-ambient-api-key'
  process.env.ANTHROPIC_AUTH_TOKEN = 'sk-ant-oat01-fake-ambient'
  try {
    await run(home)
  } finally {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]
      else process.env[key] = value
    }
    await rm(home, { recursive: true, force: true })
  }
}

function fakeTransport(headers: Headers[]): FetchImplementation {
  return async (_input, init) => {
    headers.push(new Headers(init?.headers))
    return Response.json({
      content: [{ type: 'text', text: 'ok' }],
      stop_reason: 'end_turn',
      usage: { input_tokens: 1, output_tokens: 1 },
    })
  }
}

test('isolated Anthropic explicit keys win over ambient and injected OAuth', async () => {
  await withFakeAmbientAnthropicAuth(async home => {
    const headers: Headers[] = []
    const anthropicOAuthSession = new AnthropicOAuthSession({
      xerxesHome: join(home, 'named'),
      environment: { ANTHROPIC_AUTH_TOKEN: 'sk-ant-oat01-fake-named' },
    })
    for (const keySource of ['override', 'option'] as const) {
      const client = createLlmClient(request.model, {
        provider: 'anthropic',
        ...(keySource === 'override' ? { api_key: 'fake-explicit-override' } : {}),
      }, {
        isolatedAuth: true,
        apiKey: 'fake-explicit-option',
        anthropicOAuthSession,
        fetchImplementation: fakeTransport(headers),
      })
      if (!client.complete) throw new Error('Anthropic native completion is unavailable')
      expect(await client.complete(request)).toMatchObject({ content: 'ok' })
    }
    expect(headers.map(value => value.get('x-api-key'))).toEqual(['fake-explicit-override', 'fake-explicit-option'])
    expect(headers.map(value => value.get('authorization'))).toEqual([null, null])
  })
})

test('isolated Anthropic missing keys never borrow ambient API keys or OAuth', async () => {
  await withFakeAmbientAnthropicAuth(async () => {
    const headers: Headers[] = []
    for (const apiKey of [undefined, '']) {
      await expect((async () => {
        const client = createLlmClient(request.model, {
          provider: 'anthropic',
          ...(apiKey === undefined ? {} : { api_key: apiKey }),
        }, { isolatedAuth: true, fetchImplementation: fakeTransport(headers) })
        if (!client.complete) throw new Error('Anthropic native completion is unavailable')
        return client.complete(request)
      })()).rejects.toThrow('API key not provided')
    }
    expect(headers).toHaveLength(0)
  })
})

test('isolated Anthropic keeps the injected named OAuth session', async () => {
  await withFakeAmbientAnthropicAuth(async home => {
    const headers: Headers[] = []
    const client = createLlmClient(request.model, { provider: 'anthropic', api_key: '' }, {
      isolatedAuth: true,
      anthropicOAuthSession: new AnthropicOAuthSession({
        xerxesHome: join(home, 'named'),
        environment: { ANTHROPIC_AUTH_TOKEN: 'sk-ant-oat01-fake-named' },
      }),
      fetchImplementation: fakeTransport(headers),
    })
    if (!client.complete) throw new Error('Anthropic native completion is unavailable')
    expect(await client.complete(request)).toMatchObject({ content: 'ok' })
    expect(headers).toHaveLength(1)
    expect(headers[0]?.get('authorization')).toBe('Bearer sk-ant-oat01-fake-named')
    expect(headers[0]?.get('x-api-key')).toBeNull()
  })
})

test('isolated Anthropic missing named OAuth never falls back to ambient authentication', async () => {
  await withFakeAmbientAnthropicAuth(async home => {
    const headers: Headers[] = []
    const client = createLlmClient(request.model, { provider: 'anthropic', api_key: '' }, {
      isolatedAuth: true,
      anthropicOAuthSession: new AnthropicOAuthSession({ xerxesHome: join(home, 'named'), environment: {} }),
      fetchImplementation: fakeTransport(headers),
    })
    if (!client.complete) throw new Error('Anthropic native completion is unavailable')
    await expect(client.complete(request)).rejects.toThrow('No Anthropic subscription session')
    expect(headers).toHaveLength(0)
  })
})
