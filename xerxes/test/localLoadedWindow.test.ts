// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, test } from 'bun:test'

import {
  llamaServerWindow,
  loopbackOrigin,
  parseLmStudioModels,
  probeLoadedWindows,
  windowFor,
  type WindowFetch,
} from '../src/local/loadedWindow.js'

interface Call {
  readonly url: string
  readonly headers: Readonly<Record<string, string>>
}

function fakeFetch(routes: Record<string, () => Response | Promise<Response>>): { fetch: WindowFetch; calls: Call[] } {
  const calls: Call[] = []
  const fetch: WindowFetch = async (url, init) => {
    calls.push({ url, headers: init.headers })
    const route = routes[new URL(url).pathname]
    return route === undefined ? new Response('not found', { status: 404 }) : route()
  }
  return { fetch, calls }
}

const json = (body: unknown, status = 200): Response => new Response(JSON.stringify(body), { status })

// What LM Studio 0.3.x returns from /api/v0/models with one model loaded.
const lmStudioModels = {
  object: 'list',
  data: [
    {
      id: 'ternary-bonsai-2-27b-mlx',
      object: 'model',
      type: 'vlm',
      compatibility_type: 'mlx',
      state: 'loaded',
      max_context_length: 262144,
      loaded_context_length: 42752,
    },
    { id: 'qwen/qwen3-8b', object: 'model', type: 'llm', state: 'not-loaded', max_context_length: 40960 },
  ],
}

describe('probeLoadedWindows', () => {
  test('LM Studio: reports loaded_context_length for loaded models only', async () => {
    const { fetch, calls } = fakeFetch({ '/api/v0/models': () => json(lmStudioModels) })
    const result = await probeLoadedWindows({ baseUrl: 'http://127.0.0.1:1234/v1' }, fetch)
    expect(result).toMatchObject({ ok: true, engine: 'lmstudio' })
    expect(windowFor(result, 'ternary-bonsai-2-27b-mlx')).toBe(42752)
    expect(windowFor(result, 'qwen/qwen3-8b')).toBeUndefined()
    // llama-server's endpoint is not asked once LM Studio answered.
    expect(calls.map((call) => call.url)).toEqual(['http://127.0.0.1:1234/api/v0/models'])
  })

  test("llama-server: LM Studio's route 404s, /props gives the per-slot window for any model name", async () => {
    const { fetch, calls } = fakeFetch({
      '/props': () => json({ default_generation_settings: { n_ctx: 16384 }, total_slots: 1, model_path: '/m/bonsai.gguf' }),
    })
    const result = await probeLoadedWindows({ baseUrl: 'http://localhost:11581/v1' }, fetch)
    expect(result).toEqual({ ok: true, engine: 'llama-server', window: 16384 })
    expect(windowFor(result, 'ternary-bonsai-2-27b')).toBe(16384)
    expect(calls.map((call) => new URL(call.url).pathname)).toEqual(['/api/v0/models', '/props'])
  })

  test('llama-server: falls back to a top-level n_ctx', () => {
    expect(llamaServerWindow({ n_ctx: 8192 })).toBe(8192)
    expect(llamaServerWindow({ default_generation_settings: { n_ctx: 4096 }, n_ctx: 8192 })).toBe(4096)
    expect(llamaServerWindow({ default_generation_settings: {} })).toBeUndefined()
    expect(llamaServerWindow({ n_ctx: -1 })).toBeUndefined()
  })

  test('a cloud or unix-socket base URL is never probed', async () => {
    const { fetch, calls } = fakeFetch({ '/api/v0/models': () => json(lmStudioModels) })
    for (const baseUrl of ['https://api.openai.com/v1', 'http://10.0.0.5:1234/v1', 'unix:/tmp/llama.sock']) {
      const result = await probeLoadedWindows({ baseUrl }, fetch)
      expect(result).toEqual({ ok: false, reason: 'not an http(s) loopback endpoint' })
    }
    expect(calls).toEqual([])
    expect(loopbackOrigin('http://[::1]:8080/v1')).toBe('http://[::1]:8080')
  })

  test('a server with neither endpoint fails with a reason naming both', async () => {
    const { fetch } = fakeFetch({})
    const result = await probeLoadedWindows({ baseUrl: 'http://127.0.0.1:8000/v1' }, fetch)
    expect(result.ok).toBe(false)
    expect(!result.ok && result.reason).toContain('/api/v0/models')
    expect(!result.ok && result.reason).toContain('/props')
  })

  test('an LM Studio server that requires a key: the key is sent, and a refusal is reported, not retried elsewhere', async () => {
    const { fetch, calls } = fakeFetch({ '/api/v0/models': () => new Response('unauthorized', { status: 401 }) })
    const result = await probeLoadedWindows({ baseUrl: 'http://127.0.0.1:1234/v1', apiKey: 'lm-key' }, fetch)
    expect(result).toEqual({ ok: false, reason: 'http://127.0.0.1:1234/api/v0/models refused the request (HTTP 401)' })
    expect(calls).toHaveLength(1)
    expect(calls[0]?.headers.Authorization).toBe('Bearer lm-key')
  })

  test('a network error and invalid JSON become reasons, never throws', async () => {
    const refused: WindowFetch = async () => {
      throw new Error('connect ECONNREFUSED 127.0.0.1:1234')
    }
    expect(await probeLoadedWindows({ baseUrl: 'http://127.0.0.1:1234/v1' }, refused)).toEqual({
      ok: false,
      reason: 'connect ECONNREFUSED 127.0.0.1:1234',
    })
    const { fetch } = fakeFetch({ '/api/v0/models': () => new Response('{not json', { status: 200 }) })
    const result = await probeLoadedWindows({ baseUrl: 'http://127.0.0.1:1234/v1' }, fetch)
    expect(result).toEqual({ ok: false, reason: '/api/v0/models returned invalid JSON' })
  })

  test('a hung server times out and the request is aborted', async () => {
    let aborted = false
    const hung: WindowFetch = (_url, init) =>
      new Promise((_resolve, reject) => {
        init.signal.addEventListener('abort', () => {
          aborted = true
          reject(init.signal.reason)
        })
      })
    const started = performance.now()
    const result = await probeLoadedWindows({ baseUrl: 'http://127.0.0.1:1234/v1', timeoutMs: 20 }, hung)
    expect(performance.now() - started).toBeLessThan(1000)
    expect(aborted).toBe(true)
    expect(result).toEqual({ ok: false, reason: 'no answer from http://127.0.0.1:1234 within 20ms' })
  })

  test("the caller's cancellation aborts the probe", async () => {
    const controller = new AbortController()
    const hung: WindowFetch = (_url, init) =>
      new Promise((_resolve, reject) => init.signal.addEventListener('abort', () => reject(init.signal.reason)))
    const pending = probeLoadedWindows({ baseUrl: 'http://127.0.0.1:1234/v1', signal: controller.signal }, hung)
    controller.abort(new Error('turn cancelled'))
    expect(await pending).toEqual({ ok: false, reason: 'turn cancelled' })
  })
})

test('parseLmStudioModels rejects bodies that are not LM Studio model lists', () => {
  expect(parseLmStudioModels({ data: [{ id: 'm' }] })).toBeUndefined()
  expect(parseLmStudioModels({ object: 'list' })).toBeUndefined()
  expect(parseLmStudioModels({ data: [{ id: 'm', state: 'loaded', loaded_context_length: '4096' }] })).toEqual([
    { id: 'm', state: 'loaded' },
  ])
})
