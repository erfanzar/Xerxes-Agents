// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, spyOn, test } from 'bun:test'
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ProviderAccountRegistry } from '../src/auth/providerAccountRegistry.js'
import { profileSessionOptions } from '../src/auth/profileSessionOptions.js'
import { createLlmClient } from '../src/llms/client.js'
import { ProfileStore } from '../src/bridge/profiles.js'
import { GlobalProviderAccounts } from '../src/desktop/main/globalProviderAccounts.js'
import { profileDiscoveryApiKey } from '../src/daemon/modelDiscovery.js'
import { GeminiClient } from '../src/llms/gemini.js'

async function fixture() {
  const directory = await mkdtemp(join(tmpdir(), 'xr-gemini-accounts-'))
  const profiles = new ProfileStore(join(directory, 'profiles.json'))
  return { directory, profiles, registry: new ProviderAccountRegistry(join(directory, 'accounts'), profiles.filePath, {}) }
}
const login = async () => { throw new Error('Gemini must not invoke OAuth') }

test('two named Gemini accounts use owner-only ProfileStore storage, never secret readback', async () => {
  const { directory, profiles, registry } = await fixture()
  try {
    const a = await registry.createGemini('Work', 'fixture-gemini-A', 'gemini-2.5-flash')
    const b = await registry.createGemini('Personal', 'fixture-gemini-B', 'gemini-2.5-flash')
    expect(a).toMatchObject({ id: 'api:gemini-Work', kind: 'api_key', profileName: 'gemini-Work', provider: 'gemini' })
    expect(b.id).not.toBe(a.id)
    expect(profiles.get(a.profileName)).toMatchObject({ api_key: 'fixture-gemini-A', base_url: 'https://generativelanguage.googleapis.com/v1beta', provider: 'gemini' })
    expect(profiles.chosenActive()).toBeUndefined()
    expect((await stat(profiles.filePath)).mode & 0o777).toBe(0o600)
    expect((await stat(directory)).mode & 0o777).toBe(0o700)
    await expect(stat(join(directory, 'accounts'))).rejects.toThrow()
    const readback = JSON.stringify([a, b, await registry.list(), await registry.usage(a.id)])
    expect(readback).not.toContain('fixture-gemini-')
    expect(readback).not.toContain('"api_key":')
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('Gemini creation rejects collisions and malformed inputs without changing existing profiles', async () => {
  const { directory, profiles, registry } = await fixture()
  try {
    await registry.createGemini('Work', 'fixture-gemini-A', 'gemini-2.5-flash')
    const before = await readFile(profiles.filePath, 'utf8')
    await expect(registry.createGemini('Work', 'fixture-gemini-B', 'gemini-2.5-flash')).rejects.toThrow('already exists')
    await expect(registry.createGemini('work', 'fixture-gemini-C', 'gemini-2.5-flash')).rejects.toThrow('already exists')
    for (const [label, key, model] of [['', 'fixture-A', 'gemini-2.5-flash'], ['Bad\nlabel', 'fixture-A', 'gemini-2.5-flash'], ['Bad', '', 'gemini-2.5-flash'], ['Bad', 'fixture-secret\nheader', 'gemini-2.5-flash'], ['Bad', 'fixture-A', '../invalid']]) {
      try { await registry.createGemini(label!, key!, model!); throw new Error('accepted invalid account') }
      catch (error) { expect(String(error)).not.toContain(key || 'not-a-secret'); expect(String(error)).not.toContain('accepted invalid account') }
    }
    expect(await readFile(profiles.filePath, 'utf8')).toBe(before)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('Gemini persistence failures redact lower-level errors and leave no account', async () => {
  const { directory, registry } = await fixture()
  const save = spyOn(ProfileStore.prototype, 'save').mockImplementation(() => { throw new Error('fixture-gemini-secret should never escape') })
  try {
    await expect(registry.createGemini('Work', 'fixture-gemini-secret', 'gemini-2.5-flash')).rejects.toThrow('Could not save Gemini account')
    expect(await registry.list()).toEqual([])
  } finally { save.mockRestore(); await rm(directory, { recursive: true, force: true }) }
})

test('A to B to A selects exact profiles and the native client sends only each selected fake key', async () => {
  const { directory, profiles, registry } = await fixture()
  try {
    const a = await registry.createGemini('Work', 'fixture-gemini-A', 'gemini-2.5-flash')
    const b = await registry.createGemini('Personal', 'fixture-gemini-B', 'gemini-2.5-flash')
    const observed: (string | null)[] = [], selected: unknown[] = []
    const center = new GlobalProviderAccounts(registry, undefined, async (method, params) => {
      if (method === 'provider_list') return { profiles: profiles.list().map(profile => ({ name: profile.name, provider: profile.provider, base_url: profile.base_url, has_api_key: Boolean(profile.api_key) })) }
      expect(method).toBe('provider_select'); selected.push(params.name)
      const profile = profiles.get(String(params.name))
      if (!profile?.api_key) throw new Error('Missing dedicated key')
      const client = new GeminiClient({ apiKey: profile.api_key, baseUrl: profile.base_url, fetchImplementation: async (_url, init) => {
        observed.push(new Headers(init?.headers).get('x-goog-api-key'))
        return Response.json({ candidates: [{ content: { role: 'model', parts: [{ text: 'offline fixture' }] }, finishReason: 'STOP' }] })
      } })
      await client.complete({ model: profile.model, messages: [{ role: 'user', content: 'offline fixture' }] })
      return { ok: true }
    }, login)
    for (const account of [a, b, a]) await center.activate(account.id, 'idle-session')
    expect(selected).toEqual([a.profileName, b.profileName, a.profileName])
    expect(observed).toEqual(['fixture-gemini-A', 'fixture-gemini-B', 'fixture-gemini-A'])
    await expect(center.activate(a.id, '')).rejects.toThrow('idle conversation')
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('a saved Gemini account with a missing key cannot borrow ambient credentials, while legacy env profiles still work', async () => {
  const { directory, profiles, registry } = await fixture()
  const ambient = process.env.GEMINI_API_KEY
  process.env.GEMINI_API_KEY = 'fixture-ambient-not-the-selected-account'
  try {
    const account = await registry.createGemini('Bound', 'fixture-bound-key', 'gemini-2.5-flash')
    const saved = profiles.get(account.profileName)!
    profiles.save({ name: saved.name, provider: saved.provider, baseUrl: saved.base_url, model: saved.model, apiKey: '', setActive: false })
    const reloaded = new ProfileStore(profiles.filePath).get(account.profileName)!
    expect(reloaded.isolated_auth).toBe(true)
    expect(() => profileDiscoveryApiKey(reloaded, { GEMINI_API_KEY: 'fixture-ambient' })).toThrow('dedicated API key')
    const observed: (string | null)[] = []
    const fetchImplementation = async (_url: string | URL | Request, init?: RequestInit) => {
      observed.push(new Headers(init?.headers).get('x-goog-api-key'))
      return Response.json({ candidates: [{ content: { role: 'model', parts: [{ text: 'offline fixture' }] }, finishReason: 'STOP' }] })
    }
    expect(() => createLlmClient(reloaded.model, { provider: reloaded.provider, api_key: reloaded.api_key, base_url: reloaded.base_url }, { ...profileSessionOptions(reloaded, registry), fetchImplementation })).toThrow('API key')
    expect(observed).toEqual([])
    const legacy = profiles.save({ name: 'legacy-env', provider: 'gemini', baseUrl: saved.base_url, model: saved.model, apiKey: '', setActive: false })
    const client = createLlmClient(legacy.model, { provider: legacy.provider, api_key: legacy.api_key, base_url: legacy.base_url }, { ...profileSessionOptions(legacy, registry), fetchImplementation })
    if (!client.complete) throw new Error('Gemini must support native completion')
    await client.complete({ model: legacy.model, messages: [{ role: 'user', content: 'offline fixture' }] })
    expect(observed).toEqual(['fixture-ambient-not-the-selected-account'])
  } finally {
    if (ambient === undefined) delete process.env.GEMINI_API_KEY
    else process.env.GEMINI_API_KEY = ambient
    await rm(directory, { recursive: true, force: true })
  }
})

test('Gemini usage is recorded, exact-profile attributed, and never invents quota or missing prices', async () => {
  const { directory, registry } = await fixture()
  try {
    const a = await registry.createGemini('Work', 'fixture-gemini-A', 'gemini-2.5-flash')
    const b = await registry.createGemini('Personal', 'fixture-gemini-B', 'gemini-2.5-flash')
    const calls: unknown[] = []
    const center = new GlobalProviderAccounts(registry, undefined, async (method, params) => {
      calls.push([method, params])
      return { ok: true, models_since: 1234, models: [
        { profile: a.profileName, model: 'gemini-2.5-flash', calls: 2, input_tokens: 100, output_tokens: 20, cache_read_tokens: 7, cache_write_tokens: 3 },
        { profile: b.profileName, model: 'gemini-2.5-flash', calls: 5, input_tokens: 900, output_tokens: 80, cache_read_tokens: 0, cache_write_tokens: 0, cost_usd: 0.25 },
        { profile: `${a.profileName}-other`, calls: 999, input_tokens: 999999, output_tokens: 999999 },
        { model: 'gemini-2.5-flash', calls: 999, input_tokens: 999999, output_tokens: 999999 },
      ] }
    }, login)
    expect(await center.usage(a.id)).toMatchObject({ status: 'unknown', remaining_tokens: null, recorded: { status: 'available', profile: a.profileName, since: 1234, calls: 2, input_tokens: 100, output_tokens: 20, cache_read_tokens: 7, cache_write_tokens: 3 } })
    const usageB = await center.usage(b.id)
    expect(usageB).toMatchObject({ recorded: { calls: 5, input_tokens: 900, output_tokens: 80, cost_usd: 0.25 } })
    expect(JSON.stringify(await center.usage(a.id))).not.toContain('cost_usd')
    expect(calls).toEqual(Array.from({ length: 3 }, () => ['usage.report', { refresh: false }]))
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('Gemini recorded totals combine exact-profile rows but omit partial pricing and reject malformed counters', async () => {
  const { directory, registry } = await fixture()
  try {
    const account = await registry.createGemini('Work', 'fixture-gemini-A', 'gemini-2.5-flash')
    const row = { profile: account.profileName, model: 'gemini-2.5-flash', calls: 1, input_tokens: 10, output_tokens: 5, cache_read_tokens: 3, cache_write_tokens: 2 }
    const combined = new GlobalProviderAccounts(registry, undefined, async () => ({ models: [{ ...row, cost_usd: 0.1 }, { ...row, model: 'gemini-2.5-pro' }] }), login)
    const result = await combined.usage(account.id)
    expect(result).toMatchObject({ recorded: { calls: 2, input_tokens: 20, output_tokens: 10, cache_read_tokens: 6, cache_write_tokens: 4 } })
    const recorded = result as { recorded: Record<string, unknown> }
    expect(recorded.recorded).not.toHaveProperty('cost_usd')
    const malformed = new GlobalProviderAccounts(registry, undefined, async () => ({ models: [{ ...row, input_tokens: Number.NaN }] }), login)
    expect(await malformed.usage(account.id)).toMatchObject({ recorded: { status: 'unavailable' } })
    const empty = new GlobalProviderAccounts(registry, undefined, async () => ({ models: [] }), login)
    expect(await empty.usage(account.id)).toMatchObject({ recorded: { status: 'available', calls: 0, input_tokens: 0 } })
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('Gemini usage ledger failures remain unknown and never expose daemon error text or keys', async () => {
  const { directory, registry } = await fixture()
  try {
    const account = await registry.createGemini('Work', 'fixture-gemini-secret', 'gemini-2.5-flash')
    for (const call of [async () => { throw new Error('fixture-gemini-secret daemon failure') }, async () => ({ ok: false, error: 'fixture-gemini-secret' }), async () => ({ ok: true })]) {
      const center = new GlobalProviderAccounts(registry, undefined, call, login)
      const result = await center.usage(account.id)
      expect(result).toMatchObject({ status: 'unknown', recorded: { status: 'unavailable' } })
      expect(JSON.stringify(result)).not.toContain('fixture-gemini-secret')
      expect(JSON.stringify(result)).not.toContain('cost_usd')
    }
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('Gemini removal is exact and a removed or empty account cannot borrow another credential', async () => {
  const { directory, profiles, registry } = await fixture()
  try {
    const a = await registry.createGemini('Work', 'fixture-gemini-A', 'gemini-2.5-flash')
    const b = await registry.createGemini('Personal', 'fixture-gemini-B', 'gemini-2.5-flash')
    let selected = false
    const center = new GlobalProviderAccounts(registry, undefined, async () => { selected = true; return { ok: true } }, login)
    expect(await center.remove(a.id)).toEqual({ ok: true })
    expect(profiles.get(a.profileName)).toBeUndefined()
    expect(profiles.get(b.profileName)?.api_key).toBe('fixture-gemini-B')
    await expect(center.activate(a.id, 's1')).rejects.toThrow('not found')
    await expect(registry.syncPayload(a.id)).rejects.toThrow()
    await expect(center.remove(a.id)).rejects.toThrow()
    expect(selected).toBe(false)
    profiles.save({ name: b.profileName, apiKey: '', provider: 'gemini', baseUrl: 'https://generativelanguage.googleapis.com/v1beta', model: 'gemini-2.5-flash', setActive: false })
    await expect(center.activate(b.id, 's1')).rejects.toThrow('not found')
    await expect(registry.syncPayload(b.id)).rejects.toThrow('dedicated API key')
    expect(selected).toBe(false)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('Gemini remote creation and listing never transfer; sync requires an explicit one-use review', async () => {
  const { directory, registry } = await fixture()
  try {
    const writes: unknown[] = []
    const center = new GlobalProviderAccounts(registry, { alias: 'fixture-host', target: 'user@example.test', workspacePath: '/workspace' }, async method => method === 'provider_list' ? { profiles: [] } : { ok: true, provider_account_isolation_v1_supported: true }, login, () => 1000, async payload => { writes.push(payload); return JSON.stringify({ ok: true, profile: 'gemini-Work' }) })
    const created = await center.createGemini('Work', 'fixture-gemini-A', 'gemini-2.5-flash')
    expect(JSON.stringify(created)).not.toContain('fixture-gemini-A')
    expect(await center.list()).toMatchObject({ accounts: [{ id: 'api:gemini-Work', available: false }] })
    const review = center.reviewSync('api:gemini-Work') as { review: string }
    expect(writes).toHaveLength(0)
    await expect(center.confirmSync(review.review, false)).rejects.toThrow('review')
    expect(writes).toHaveLength(0)
    const result = await center.confirmSync(review.review, true)
    expect(writes).toHaveLength(1)
    expect(JSON.stringify(writes)).toContain('fixture-gemini-A')
    expect(JSON.stringify(result)).not.toContain('fixture-gemini-A')
    await expect(center.confirmSync(review.review, true)).rejects.toThrow('review')
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('Gemini availability and activation reject a same-named unrelated remote profile until confirmed sync', async () => {
  const { directory, profiles, registry } = await fixture()
  try {
    const account = await registry.createGemini('Work', 'fixture-gemini-A', 'gemini-2.5-flash')
    let selected = 0, writes = 0
    const center = new GlobalProviderAccounts(registry, { alias: 'fixture-host', target: 'user@example.test', workspacePath: '/workspace' }, async method => {
      if (method === 'provider_list') return { profiles: [{ name: account.profileName, provider: 'gemini', base_url: 'https://generativelanguage.googleapis.com/v1beta' }] }
      if (method === 'provider_select') selected++
      return { ok: true, provider_account_isolation_v1_supported: true }
    }, login, () => 1000, async () => { writes++; return JSON.stringify({ ok: true, profile: account.profileName }) })
    expect(await center.list()).toMatchObject({ accounts: [{ available: false }] })
    await expect(center.activate(account.id, 's1')).rejects.toThrow('not available')
    expect(selected).toBe(0)
    expect(writes).toBe(0)
    const review = center.reviewSync(account.id) as { review: string }
    await center.confirmSync(review.review, true)
    expect(await center.list()).toMatchObject({ accounts: [{ available: true }] })
    await center.activate(account.id, 's1')
    expect(selected).toBe(1)
    expect(writes).toBe(1)
    profiles.save({ name: account.profileName, provider: 'gemini', baseUrl: 'https://generativelanguage.googleapis.com/v1beta', apiKey: 'fixture-gemini-replaced', model: 'gemini-2.5-flash', setActive: false })
    expect(await center.list()).toMatchObject({ accounts: [{ available: false }] })
    await expect(center.activate(account.id, 's1')).rejects.toThrow('not available')
    expect(selected).toBe(1)
    expect(writes).toBe(1)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('Gemini quota inspection performs no network probes and transfer failures never expose secrets', async () => {
  const { directory, registry } = await fixture()
  const network = spyOn(globalThis, 'fetch').mockImplementation(Object.assign(async () => { throw new Error('Unexpected network quota probe') }, { preconnect: () => { throw new Error('Unexpected network preconnect') } }))
  try {
    const account = await registry.createGemini('Work', 'fixture-gemini-A', 'gemini-2.5-flash')
    expect(await registry.usage(account.id)).toMatchObject({ status: 'unknown', remaining_tokens: null })
    expect(network).not.toHaveBeenCalled()
    const center = new GlobalProviderAccounts(registry, { alias: 'fixture-host', target: 'user@example.test', workspacePath: '/workspace' }, async () => ({ ok: true, provider_account_isolation_v1_supported: true }), login, () => 1000, async () => 'fixture-gemini-A invalid receiver output')
    const review = center.reviewSync(account.id) as { review: string }
    try { await center.confirmSync(review.review, true); throw new Error('accepted bad receipt') }
    catch (error) { expect(String(error)).toContain('not acknowledged'); expect(String(error)).not.toContain('fixture-gemini-A') }
    await expect(center.confirmSync(review.review, true)).rejects.toThrow('review')
  } finally { network.mockRestore(); await rm(directory, { recursive: true, force: true }) }
})

test('Gemini profile deletion failures are observable and never remove other accounts', async () => {
  const { directory, registry } = await fixture()
  try {
    const account = await registry.createGemini('Work', 'fixture-gemini-A', 'gemini-2.5-flash')
    const remove = spyOn(ProfileStore.prototype, 'delete').mockImplementation(() => { throw new Error('fixture-gemini-A internal failure') })
    try { await expect(registry.remove(account.id)).rejects.toThrow('Could not remove Gemini account') }
    finally { remove.mockRestore() }
    expect((await registry.list()).map(row => row.id)).toEqual([account.id])
  } finally { await rm(directory, { recursive: true, force: true }) }
})
