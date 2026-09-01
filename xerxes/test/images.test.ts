// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import {
  buildParams,
  generateImagesViaOpenRouter,
  OPENROUTER_IMAGES_API,
  openrouterImagesApiProvider,
  parseUsage,
  sanitizeSurrogates,
} from '../src/images/openrouterImages.js'
import { generateImages, getImagesApiProvider, registerImagesApiProvider } from '../src/images/registry.js'
import type { ImagesContent, ImagesModel } from '../src/images/types.js'
import { imagesError } from '../src/images/types.js'

const MODEL: ImagesModel = {
  api: OPENROUTER_IMAGES_API,
  baseUrl: 'https://openrouter.example/api/v1',
  cost: { cacheRead: 0.01, cacheWrite: 0.02, input: 1, output: 2 },
  id: 'google/gemini-3-pro-image',
  input: ['text', 'image'],
  name: 'Gemini image',
  output: ['image', 'text'],
  provider: 'openrouter',
}

const IMAGE_MODEL: ImagesModel = { ...MODEL, output: ['image'] }

const PNG_BASE64 = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).toString('base64')

function chatResponse(body: Record<string, unknown>, status = 200, headers?: Record<string, string>): Response {
  return new Response(JSON.stringify(body), {
    status,
    ...(headers === undefined ? {} : { headers }),
  })
}

function sseFetch(responses: Response[], calls?: { url?: string; init?: RequestInit }[]): (input: RequestInfo | URL, init?: RequestInit) => Promise<Response> {
  let index = 0
  return async (url, init) => {
    calls?.push({ ...(init === undefined ? {} : { init }), url: String(url) })
    const response = responses[Math.min(index, responses.length - 1)]
    if (!response) throw new Error('no stubbed response left')
    index += 1
    return response
  }
}

describe('image api registry', () => {
  test('registers, resolves, and guards the api field', async () => {
    expect(getImagesApiProvider('nothing-here')).toBeUndefined()
    registerImagesApiProvider(openrouterImagesApiProvider())
    const provider = getImagesApiProvider(OPENROUTER_IMAGES_API)
    expect(provider?.api).toBe(OPENROUTER_IMAGES_API)

    // The api guard fires when a provider implementation is invoked for a
    // model naming a different api.
    await expect(provider?.generateImages({ ...MODEL, api: 'other-api' }, { input: [] }))
      .rejects.toThrow(/mismatched api/)
  })

  test('dispatch throws a typed error when no provider is registered', async () => {
    await expect(generateImages({ ...MODEL, api: 'never-registered' }, { input: [] }))
      .rejects.toThrow(/no image API provider registered/)
  })
})

describe('openrouter-images request building', () => {
  test('builds chat-completions params with modalities from the model surface', () => {
    const params = buildParams(MODEL, { input: [{ type: 'text', text: 'a cat' }] })
    expect(params).toEqual({
      model: 'google/gemini-3-pro-image',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'a cat' }] }],
      stream: false,
      modalities: ['image', 'text'],
    })
    const imageOnly = buildParams(IMAGE_MODEL, { input: [{ type: 'text', text: 'a cat' }] })
    expect(imageOnly.modalities).toEqual(['image'])

    const withImage = buildParams(MODEL, { input: [
      { type: 'text', text: 'redo' },
      { type: 'image', data: PNG_BASE64, mimeType: 'image/png' },
    ] })
    expect(withImage.messages).toEqual([{
      role: 'user',
      content: [
        { type: 'text', text: 'redo' },
        { type: 'image_url', image_url: { url: `data:image/png;base64,${PNG_BASE64}` } },
      ],
    }])
  })

  test('removes unpaired surrogates but keeps real emoji', () => {
    expect(sanitizeSurrogates('Hello 🙈 World')).toBe('Hello 🙈 World')
    expect(sanitizeSurrogates(`broken ${String.fromCharCode(0xD83D)} end`)).toBe('broken  end')
  })
})

describe('openrouter-images response parsing', () => {
  test('collects text and data-URL images and skips remote URLs', async () => {
    const calls: { url?: string; init?: RequestInit }[] = []
    const fetchImplementation = sseFetch([chatResponse({
      id: 'resp-1',
      choices: [{
        message: {
          role: 'assistant',
          content: 'Here you go',
          images: [
            { image_url: { url: `data:image/png;base64,${PNG_BASE64}` } },
            { image_url: 'https://cdn.example/skipped.png' },
            { image_url: { url: 'not-a-data-url' } },
          ],
        },
      }],
      usage: { prompt_tokens: 10, completion_tokens: 4, prompt_tokens_details: { cached_tokens: 6, cache_write_tokens: 2 } },
    })], calls)

    const result = await generateImagesViaOpenRouter(MODEL, { input: [{ type: 'text', text: 'a cat' }] }, {
      apiKey: 'sk-test',
      fetch: fetchImplementation,
    })

    expect(result.stopReason).toBe('stop')
    expect(result.responseId).toBe('resp-1')
    expect(calls[0]?.url).toBe('https://openrouter.example/api/v1/chat/completions')
    const headers = calls[0]?.init?.headers as Record<string, string>
    expect(headers.Authorization).toBe('Bearer sk-test')
    expect(result.output).toEqual([
      { type: 'text', text: 'Here you go' },
      { type: 'image', data: PNG_BASE64, mimeType: 'image/png' },
    ])
    // input = 10 - (6-2) - 2 = 4; costs at $1/M input, $2/M output, cache at $0.01/$0.02.
    expect(result.usage).toEqual({
      input: 4,
      output: 4,
      cacheRead: 4,
      cacheWrite: 2,
      totalTokens: 14,
      cost: {
        input: 4 / 1_000_000,
        output: 8 / 1_000_000,
        cacheRead: 4 / 100_000_000,
        cacheWrite: 4 / 100_000_000,
        total: 4 / 1_000_000 + 8 / 1_000_000 + 4 / 100_000_000 + 4 / 100_000_000,
      },
    })
  })

  test('parses usage without cache details and keeps the negative dynamic-cost sentinel', () => {
    const dynamic = parseUsage({ prompt_tokens: 3, completion_tokens: 2 }, {
      ...MODEL,
      cost: { cacheRead: 0, cacheWrite: 0, input: -1_000_000, output: -1_000_000 },
    })
    expect(dynamic?.cost.total).toBe(-5)
    expect(parseUsage(undefined, MODEL)).toBeUndefined()
  })
})

describe('openrouter-images failure behavior', () => {
  test('reports HTTP failures in-band without throwing', async () => {
    const result = await generateImagesViaOpenRouter(MODEL, { input: [] }, {
      apiKey: 'sk-test',
      fetch: sseFetch([chatResponse({ error: { message: 'nope' } }, 402)]),
    })
    expect(result.stopReason).toBe('error')
    expect(result.errorMessage).toContain('402')
    expect(result.output).toEqual([])
  })

  test('missing API key is an error result, not a throw', async () => {
    const result = await generateImagesViaOpenRouter(MODEL, { input: [] }, { fetch: sseFetch([]) })
    expect(result.stopReason).toBe('error')
    expect(result.errorMessage).toContain('no API key')
  })

  test('an aborted signal reports stopReason aborted', async () => {
    const controller = new AbortController()
    controller.abort()
    const result = await generateImagesViaOpenRouter(MODEL, { input: [] }, {
      apiKey: 'sk-test',
      fetch: sseFetch([chatResponse({ error: { message: 'boom' } }, 500)]),
      signal: controller.signal,
    })
    expect(result.stopReason).toBe('aborted')
    expect(imagesError(MODEL, 'x').stopReason).toBe('error')
  })
})

describe('openrouter-images retries', () => {
  test('retries retryable statuses honoring retry-after, then succeeds', async () => {
    const calls: { init?: RequestInit }[] = []
    const sleeps: number[] = []
    const fetchImplementation = sseFetch([
      chatResponse({ error: 'slow down' }, 429, { 'retry-after': '0.02' }),
      chatResponse({
        choices: [{ message: { content: 'ok', images: [{ image_url: { url: `data:image/png;base64,${PNG_BASE64}` } }] } }],
      }),
    ], calls)
    const result = await generateImagesViaOpenRouter(MODEL, { input: [{ type: 'text', text: 'hi' }] }, {
      apiKey: 'sk-test',
      fetch: fetchImplementation,
      maxRetries: 2,
      sleep: async ms => {
        sleeps.push(ms)
      },
    })
    expect(result.stopReason).toBe('stop')
    expect(calls.length).toBe(2)
    expect(sleeps).toEqual([20])
  })

  test('does not retry non-retryable statuses even with attempts left', async () => {
    const calls: { init?: RequestInit }[] = []
    await generateImagesViaOpenRouter(MODEL, { input: [] }, {
      apiKey: 'sk-test',
      fetch: sseFetch([chatResponse({}, 400)], calls),
      maxRetries: 3,
    })
    expect(calls.length).toBe(1)
  })

  test('a server retry delay beyond the cap aborts the retry loop', async () => {
    const calls: { init?: RequestInit }[] = []
    const result = await generateImagesViaOpenRouter(MODEL, { input: [] }, {
      apiKey: 'sk-test',
      fetch: sseFetch([chatResponse({}, 429, { 'retry-after': '120' })], calls),
      maxRetries: 3,
    })
    expect(calls.length).toBe(1)
    expect(result.stopReason).toBe('error')
    expect(result.errorMessage).toContain('retry delay')
  })
})
