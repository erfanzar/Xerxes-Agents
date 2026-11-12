// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ProfileStore, canonicalProfileName, profileLabel, signsInPerMachine } from '../src/bridge/profiles.js'
import { createLlmClient } from '../src/llms/client.js'
import { GeminiClient } from '../src/llms/gemini.js'
import { ambientApiKeyWithheldHint, getApiKey, resolveProvider } from '../src/llms/providerRegistry.js'
import { applyRunMode, readRunMode, runModeTiers } from '../src/daemon/runMode.js'
import { enableLocalOnly } from '../src/runtime/localOnly.js'

const request = { model: 'gemini-2.5-flash', messages: [{ role: 'user' as const, content: 'hello' }] }

test('Gemini keys resolve GEMINI_API_KEY before GOOGLE_API_KEY and stay on the vendor origin', () => {
  const environment = { GEMINI_API_KEY: 'fake-primary', GOOGLE_API_KEY: 'fake-google' }
  expect(getApiKey('gemini', {}, environment)).toBe('fake-primary')
  expect(getApiKey('gemini', {}, { GOOGLE_API_KEY: 'fake-google' })).toBe('fake-google')
  for (const destination of ['https://proxy.example/v1beta', 'http://generativelanguage.googleapis.com/v1beta']) {
    expect(getApiKey('gemini', {}, environment, destination)).toBe('')
  }
  expect(getApiKey('gemini', {}, environment, 'https://generativelanguage.googleapis.com/v1beta')).toBe('fake-primary')
  expect(ambientApiKeyWithheldHint('gemini', 'https://proxy.example', { GOOGLE_API_KEY: 'fake-google' }))
    .toContain('GOOGLE_API_KEY')
})

test('direct Gemini clients accept injected environment only on the configured vendor origin', async () => {
  const keys: Array<string | null> = []
  const fetchImplementation = async (_input: RequestInfo | URL, init?: RequestInit) => {
    keys.push(new Headers(init?.headers).get('x-goog-api-key'))
    expect(init?.redirect).toBe('error')
    return Response.json({ candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }] })
  }
  const environment = { GOOGLE_API_KEY: 'fake-google' }
  await new GeminiClient({ environment, fetchImplementation }).complete(request)
  expect(keys.at(-1)).toBe('fake-google')
  expect(() => new GeminiClient({ environment, apiKey: '', fetchImplementation })).toThrow('API key not provided')
  expect(() => new GeminiClient({ environment, baseUrl: 'https://proxy.example/v1beta', fetchImplementation }))
    .toThrow('API key not provided')
  await new GeminiClient({ environment, apiKey: 'fake-explicit', baseUrl: 'https://proxy.example/v1beta', fetchImplementation }).complete(request)
  expect(keys.at(-1)).toBe('fake-explicit')
})

test('Gemini factory uses native requests even with responses_api and honors isolated account keys', async () => {
  const urls: string[] = []
  const keys: Array<string | null> = []
  for (const compat of ['', '/openai/']) {
    const client = createLlmClient('gemini-2.5-flash', {
      provider: 'gemini', api_key: 'fake-account', responses_api: true,
      base_url: `https://generativelanguage.googleapis.com/v1beta${compat}`,
    }, {
      isolatedAuth: true,
      fetchImplementation: async (input, init) => {
        urls.push(String(input))
        keys.push(new Headers(init?.headers).get('x-goog-api-key'))
        return Response.json({ candidates: [{ content: { parts: [{ text: 'native' }] }, finishReason: 'STOP' }] })
      },
    })
    expect(client).toBeInstanceOf(GeminiClient)
    expect(await client.complete?.(request)).toMatchObject({ content: 'native', finishReason: 'stop' })
  }
  expect(keys).toEqual(['fake-account', 'fake-account'])
  expect(urls).toEqual(Array(2).fill('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'))
  for (const api_key of ['', undefined]) {
    expect(() => createLlmClient('gemini-2.5-flash', { provider: 'gemini', ...(api_key === undefined ? {} : { api_key }) }, { isolatedAuth: true }))
      .toThrow('API key not provided')
  }
})

test('saved Google native and compat endpoints infer Gemini, not OpenAI; saved local ids only change display', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-gemini-profiles-'))
  try {
    const store = new ProfileStore(join(directory, 'profiles.json'))
    for (const suffix of ['', '/openai']) {
      const baseUrl = `https://generativelanguage.googleapis.com/v1beta${suffix}`
      const profile = store.save({ name: `gemini${suffix ? '-compat' : ''}`, apiKey: 'fake-key', baseUrl, model: request.model })
      expect(profile.provider).toBe('gemini')
      expect(resolveProvider(request.model, { base_url: baseUrl })).toBe('gemini')
      expect(signsInPerMachine(profile.provider)).toBe(false)
    }
    const profile = store.save({ name: 'bonsai-local', apiKey: '', baseUrl: 'http://localhost:1234/v1', model: 'saved-model' })
    expect(profileLabel(profile.name)).toBe('local-model')
    expect(canonicalProfileName(profile.name)).toBe('bonsai-local')
    expect(store.get('bonsai-local')?.name).toBe('bonsai-local')
    expect(profileLabel('Claude Code custom')).toBe('Claude Code custom')
    expect(profileLabel('cc')).toBe('Claude Code')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('Gemini is a selectable Cloud/Hybrid lead but the Private runtime still rejects it before fetching', () => {
  const worker = { name: 'gemini', model: request.model, base_url: 'https://generativelanguage.googleapis.com/v1beta' }
  const local = { name: 'local-model', model: 'local', base_url: 'http://127.0.0.1:1234/v1' }
  for (const mode of ['cloud', 'hybrid'] as const) {
    const settings = applyRunMode({}, mode)
    expect(readRunMode({ saved: settings, loaded: settings, worker, env: {} }).mode).toBe(mode)
    const tiers = runModeTiers(mode, local, worker)
    if (mode === 'hybrid') expect(tiers.smart).toMatchObject({ provider_profile: 'gemini' })
    expect(createLlmClient(request.model, { provider: 'gemini', api_key: 'fake-key' })).toBeInstanceOf(GeminiClient)
  }
  const restore = enableLocalOnly({})
  try {
    expect(() => createLlmClient(request.model, { provider: 'gemini', api_key: 'fake-key' })).toThrow('fully local mode refuses')
  } finally {
    restore()
  }
})
