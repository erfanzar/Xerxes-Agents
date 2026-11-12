// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { discoverModelCatalog, discoveryApiKey, modelDiscoveryEndpoint } from '../src/daemon/modelDiscovery.js'

const root = 'https://generativelanguage.googleapis.com/v1beta'

test('native Gemini discovery normalizes compatibility roots, follows bounded page tokens and preserves limits', async () => {
  for (const suffix of ['', '/openai/', '/openai/models']) {
    expect(modelDiscoveryEndpoint(`${root}${suffix}`, 'gemini')).toBe(`${root}/models`)
  }
  const requests: URL[] = []
  expect(await discoverModelCatalog({ provider: 'gemini', baseUrl: `${root}/openai/`, environment: { GOOGLE_API_KEY: 'fake-google' } }, {
    fetchImplementation: async (input, init) => {
      const url = new URL(String(input))
      requests.push(url)
      expect(init?.redirect).toBe('error')
      expect(new Headers(init?.headers).get('x-goog-api-key')).toBe('fake-google')
      expect(new Headers(init?.headers).get('authorization')).toBeNull()
      expect(url.searchParams.has('key')).toBe(false)
      return requests.length === 1 ? Response.json({
        models: [
          { name: 'models/gemini-2.5-flash', displayName: 'Gemini Flash', inputTokenLimit: 1_048_576, outputTokenLimit: 65_536, supportedGenerationMethods: ['generateContent', 'countTokens'] },
          { name: 'models/embedding', supportedGenerationMethods: ['embedContent'] },
          { name: 'models/invalid', inputTokenLimit: -1, outputTokenLimit: 0, supportedGenerationMethods: ['generateContent'] },
        ], nextPageToken: 'opaque+/=?',
      }) : Response.json({ models: [
        { name: 'models/gemini-2.5-flash', supportedGenerationMethods: ['generateContent'] },
        { name: 'models/gemini-2.5-pro', supportedGenerationMethods: ['generateContent'] },
      ] })
    },
  })).toEqual([
    { id: 'gemini-2.5-flash', displayName: 'Gemini Flash', contextLimit: 1_048_576, maxOutputTokens: 65_536 },
    { id: 'invalid' },
    { id: 'gemini-2.5-pro' },
  ])
  expect(requests).toHaveLength(2)
  expect(requests[1]?.origin).toBe(new URL(root).origin)
  expect(requests[1]?.pathname).toBe('/v1beta/models')
  expect(requests[1]?.searchParams.get('pageToken')).toBe('opaque+/=?')
})

test('Gemini discovery respects explicit keys, probe opt-out and custom destination credential withholding', async () => {
  expect(discoveryApiKey('gemini', 'fake-explicit', { GEMINI_API_KEY: 'fake-env' }, root)).toBe('fake-explicit')
  expect(discoveryApiKey('gemini', undefined, { GOOGLE_API_KEY: 'fake-google' }, root)).toBe('fake-google')
  expect(discoveryApiKey('gemini', undefined, { GOOGLE_API_KEY: 'fake-google' }, 'https://proxy.example/v1beta')).toBe('')
  for (const input of [
    { baseUrl: 'https://proxy.example/v1beta' },
    { baseUrl: root, resolveProviderCredential: false },
  ]) {
    await discoverModelCatalog({ provider: 'gemini', environment: { GEMINI_API_KEY: 'fake-env' }, ...input }, {
      fetchImplementation: async (_input, init) => {
        expect(new Headers(init?.headers).get('x-goog-api-key')).toBeNull()
        return Response.json({ models: [] })
      },
    })
  }
})

test('Gemini discovery refuses repeated page tokens, redirects and aggregate oversized responses', async () => {
  let pages = 0
  await expect(discoverModelCatalog({ provider: 'gemini', apiKey: 'fake-key', baseUrl: root }, {
    fetchImplementation: async () => {
      pages++
      return Response.json({ models: [], nextPageToken: 'repeated' })
    },
  })).rejects.toThrow('pagination')
  expect(pages).toBe(2)

  await expect(discoverModelCatalog({ provider: 'gemini', apiKey: 'fake-key', baseUrl: root }, {
    fetchImplementation: async () => Response.json({ models: [], nextPageToken: 'next' }),
    maxResponseBytes: 60,
  })).rejects.toThrow('exceeded')

  await expect(discoverModelCatalog({ provider: 'gemini', apiKey: 'fake-key', baseUrl: root }, {
    fetchImplementation: async (_input, init) => {
      expect(init?.redirect).toBe('error')
      return new Response(null, { status: 302, headers: { location: 'https://untrusted.example/' } })
    },
  })).rejects.toThrow('HTTP 302')
})

test('a Google-hosted profile is discovered natively even when its wire says openai', async () => {
  const requests: URL[] = []
  const models = await discoverModelCatalog({ provider: 'openai', apiKey: 'fake-key', baseUrl: root }, {
    fetchImplementation: async (input, init) => {
      const url = new URL(String(input))
      requests.push(url)
      const headers = new Headers(init?.headers)
      // The endpoint decides the credentials: Google rejects a bearer token here.
      expect(headers.get('x-goog-api-key')).toBe('fake-key')
      expect(headers.get('authorization')).toBeNull()
      return Response.json({ models: [
        { name: 'models/gemini-4-carbon', inputTokenLimit: 1_048_576, outputTokenLimit: 65_536, supportedGenerationMethods: ['generateContent'] },
        { name: 'models/embedding-only', supportedGenerationMethods: ['embedContent'] },
      ] })
    },
  })
  expect(requests[0]?.pathname).toBe('/v1beta/models')
  expect(models).toEqual([
    { id: 'gemini-4-carbon', contextLimit: 1_048_576, maxOutputTokens: 65_536 },
  ])
})

test('a provider that reports no context window leaves it unknown instead of guessing', async () => {
  const models = await discoverModelCatalog({ provider: 'openai', apiKey: 'fake-key', baseUrl: 'https://qwen.example/v1' }, {
    fetchImplementation: async () => Response.json({ object: 'list', data: [
      { id: 'deepseek-v4.1-flash-1017', object: 'model', created: 1, owned_by: 'qwen-plan' },
    ] }),
  })
  expect(models).toEqual([{ id: 'deepseek-v4.1-flash-1017' }])
  expect(models[0]).not.toHaveProperty('contextLimit')
})

test('Gemini discovery revalidates public DNS before each page and caps unique pagination', async () => {
  let resolved = 0
  let fetched = 0
  await expect(discoverModelCatalog({ provider: 'gemini', apiKey: 'fake-key', baseUrl: root }, {
    resolveHostname: async () => ++resolved === 1 ? ['8.8.8.8'] : ['127.0.0.1'],
    fetchImplementation: async () => { fetched++; return Response.json({ models: [], nextPageToken: 'page2' }) },
  })).rejects.toThrow('Private or local')
  expect(fetched).toBe(1)
  fetched = 0
  await expect(discoverModelCatalog({ provider: 'gemini', apiKey: 'fake-key', baseUrl: root }, {
    fetchImplementation: async () => Response.json({ models: [], nextPageToken: `page${++fetched}` }),
  })).rejects.toThrow('pagination')
  expect(fetched).toBe(20)
})

test('Gemini discovery cancellation and timeout cover stalled fetches and response bodies', async () => {
  const controller = new AbortController()
  controller.abort(new Error('caller cancelled'))
  let fetched = false
  await expect(discoverModelCatalog({ provider: 'gemini', apiKey: 'fake-key', baseUrl: root, signal: controller.signal }, {
    fetchImplementation: async () => { fetched = true; return Response.json({ models: [] }) },
  })).rejects.toThrow('caller cancelled')
  expect(fetched).toBe(false)
  for (const hang of ['headers', 'body'] as const) {
    let cancelled = false
    await expect(discoverModelCatalog({ provider: 'gemini', apiKey: 'fake-key', baseUrl: root }, {
      timeoutMs: 5,
      fetchImplementation: async () => hang === 'headers' ? new Promise<Response>(() => {}) : new Response(new ReadableStream({ cancel() { cancelled = true } })),
    })).rejects.toThrow('timed out')
    if (hang === 'body') expect(cancelled).toBe(true)
  }
})
