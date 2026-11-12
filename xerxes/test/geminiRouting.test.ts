// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ProviderError } from '../src/core/errors.js'
import { createLlmClient, type CompletionRequest, type FetchImplementation, type LlmDelta } from '../src/llms/client.js'
import { GeminiClient } from '../src/llms/gemini.js'
import { PROVIDERS } from '../src/llms/providerRegistry.js'
import { GoogleVertexClient } from '../src/llms/vertex.js'

const MODEL = 'gemini-3.8-flash'
const KEY = 'AIza-native-key'
/**
 * The endpoint a Gemini profile saved through Google's OpenAI compatibility
 * documentation carries. Google answers `404 model_not_found — Model not
 * exist.` there for current ids, while the native surface serves them.
 */
const COMPAT_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'
const NATIVE_STREAM_URL = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:streamGenerateContent?alt=sse`

interface RecordedRequest {
  readonly url: URL
  readonly headers: Headers
  readonly body: Record<string, unknown>
}

test("an OpenAI-wire profile on Google's compatibility route reaches the native Generate Content API", async () => {
  const recorded: RecordedRequest[] = []
  const client = createLlmClient(MODEL, {
    provider: 'openai',
    base_url: COMPAT_BASE_URL,
    api_key: KEY,
  }, { fetchImplementation: recordingFetch(recorded, nativeStream()) })

  const deltas = await collect(client.stream(simpleRequest()))

  expect(recorded).toHaveLength(1)
  const request = recorded[0]!
  expect(request.url.toString()).toBe(NATIVE_STREAM_URL)
  expect(request.url.pathname).not.toContain('/openai/')
  // The model travels in the URL path, and the turn's text in `contents`.
  expect(request.url.pathname).toContain(`/models/${MODEL}:`)
  expect(request.body).toMatchObject({ contents: [{ role: 'user', parts: [{ text: 'Explain AI briefly' }] }] })
  // Google's own header, never the OpenAI bearer form.
  expect(request.headers.get('x-goog-api-key')).toBe(KEY)
  expect(request.headers.get('authorization')).toBeNull()

  expect(deltas.flatMap(delta => delta.content ?? []).join('')).toBe('Explain how AI works.')
  expect(deltas.flatMap(delta => delta.usage ?? [])).toEqual([{ inputTokens: 11, outputTokens: 6 }])
  expect(deltas.flatMap(delta => delta.finishReason ?? [])).toEqual(['stop'])
})

test("a profile on the registry's own Gemini endpoint is never sent to the compatibility layer", async () => {
  const recorded: RecordedRequest[] = []
  const client = createLlmClient(MODEL, {
    provider: 'gemini',
    api_key: KEY,
    base_url: PROVIDERS.gemini.baseUrl,
  }, { fetchImplementation: recordingFetch(recorded, nativeStream()) })

  await collect(client.stream(simpleRequest()))

  expect(recorded[0]?.url.toString()).toBe(NATIVE_STREAM_URL)
  expect(recorded[0]?.url.pathname).not.toContain('/openai/')
})

test('a gemini provider profile without a base URL override streams from the native endpoint', async () => {
  const recorded: RecordedRequest[] = []
  const client = createLlmClient(MODEL, { provider: 'gemini', api_key: KEY }, {
    fetchImplementation: recordingFetch(recorded, nativeStream()),
  })

  await collect(client.stream(simpleRequest()))

  expect(recorded[0]?.url.toString()).toBe(NATIVE_STREAM_URL)
  expect(recorded[0]?.headers.get('x-goog-api-key')).toBe(KEY)
})

test('a 404 from the native API names the model and the endpoint it was sent to', async () => {
  const client = createLlmClient(MODEL, { provider: 'openai', base_url: COMPAT_BASE_URL, api_key: KEY }, {
    fetchImplementation: async () => new Response(
      JSON.stringify({
        error: { code: 404, message: `models/${MODEL} is not found for API version v1beta`, status: 'NOT_FOUND' },
      }),
      { status: 404, statusText: 'Not Found' },
    ),
  })

  const failure = await collect(client.stream(simpleRequest())).then(
    () => { throw new Error('expected the request to fail') },
    (error: unknown) => error as ProviderError,
  )

  expect(failure).toBeInstanceOf(ProviderError)
  expect(failure.clientType).toBe('gemini')
  expect(failure.details).toMatchObject({ status: 404 })
  expect(failure.message).toContain(MODEL)
  expect(failure.message).toContain(`/v1beta/models/${MODEL}:streamGenerateContent`)
  expect(failure.message).toContain('is not found for API version v1beta')
  expect(failure.message).not.toContain(KEY)
})

test('an OpenAI profile against its own host keeps the OpenAI transport', async () => {
  const recorded: RecordedRequest[] = []
  const client = createLlmClient('gpt-5.2', { provider: 'openai', api_key: 'sk-test' }, {
    fetchImplementation: recordingFetch(recorded, openAiStream()),
  })

  expect(client).not.toBeInstanceOf(GeminiClient)
  await collect(client.stream({ model: 'gpt-5.2', messages: [{ role: 'user', content: 'hi' }] }))
  expect(recorded[0]?.url.toString()).toBe('https://api.openai.com/v1/chat/completions')
})

test('google-vertex keeps its own adapter, so the Gemini routing cannot capture it', () => {
  const client = createLlmClient('gemini-2.5-flash', {
    provider: 'google-vertex',
    base_url: 'https://generativelanguage.googleapis.com/v1beta',
  })

  expect(client).toBeInstanceOf(GoogleVertexClient)
  expect(client).not.toBeInstanceOf(GeminiClient)
})

function simpleRequest(): CompletionRequest {
  return { model: MODEL, messages: [{ role: 'user', content: 'Explain AI briefly' }] }
}

function recordingFetch(recorded: RecordedRequest[], response: () => Response): FetchImplementation {
  return async (input, init) => {
    recorded.push({
      url: new URL(String(input)),
      headers: new Headers(init?.headers),
      body: JSON.parse(String(init?.body)) as Record<string, unknown>,
    })
    return response()
  }
}

/** One native Generate Content SSE stream: text, then finish plus usage. */
function nativeStream(): () => Response {
  return () => sseResponse([
    { candidates: [{ content: { parts: [{ text: 'Explain ' }] } }] },
    { candidates: [{ content: { parts: [{ text: 'how AI works.' }] }, finishReason: 'STOP' }] },
    {
      candidates: [{ content: { parts: [] }, finishReason: 'STOP' }],
      usageMetadata: { promptTokenCount: 11, candidatesTokenCount: 6 },
    },
  ])
}

function openAiStream(): () => Response {
  return () => sseResponse([
    { choices: [{ delta: { content: 'hi' } }] },
    { choices: [{ delta: {}, finish_reason: 'stop' }] },
  ])
}

function sseResponse(events: readonly Record<string, unknown>[]): Response {
  const body = `${events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('')}data: [DONE]\n\n`
  return new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(body))
      controller.close()
    },
  }), { status: 200, headers: { 'content-type': 'text/event-stream' } })
}

async function collect(stream: AsyncIterable<LlmDelta>): Promise<LlmDelta[]> {
  const deltas: LlmDelta[] = []
  for await (const delta of stream) {
    deltas.push(delta)
  }
  return deltas
}