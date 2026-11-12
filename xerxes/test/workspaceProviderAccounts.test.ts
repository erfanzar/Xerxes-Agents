// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ProviderAccountRegistry } from '../src/auth/providerAccountRegistry.js'
import { profileSessionOptions } from '../src/auth/profileSessionOptions.js'
import { ProfileStore } from '../src/bridge/profiles.js'
import { sessionProvider } from '../src/daemon/sessionProvider.js'
import { createLlmClient } from '../src/llms/client.js'

test('A to B to A preserves the selected account in turn, compaction, and fallback clients', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-workspace-accounts-'))
  try {
    const root = join(directory, 'accounts')
    const profiles = new ProfileStore(join(directory, 'profiles.json'))
    const registry = new ProviderAccountRegistry(root, profiles.filePath)
    const a = await registry.create('codex', 'A')
    const b = await registry.create('codex', 'B')
    for (const account of [a, b]) {
      await mkdir(join(root, account.id, '.codex'), { recursive: true })
      await writeFile(join(root, account.id, '.codex', 'auth.json'), JSON.stringify({ tokens: {
        access_token: `fixture-${account.label}`, refresh_token: `fixture-refresh-${account.label}`,
        expires_at: Date.now() / 1000 + 3600, scopes: [], token_type: 'Bearer',
      } }))
    }
    const chatA = { metadata: { provider_profile: a.profileName } }
    const chatB = { metadata: { provider_profile: b.profileName } }
    const observed: (string | null)[] = []
    for (const chat of [chatA, chatB, chatA]) {
      profiles.setActive(chat.metadata.provider_profile)
      for (const purpose of ['main', 'compaction', 'fallback']) {
        const profile = sessionProvider(profiles, chat, 'gpt-5.6-codex')
        if (!profile) throw new Error('Missing pinned provider')
        const client = createLlmClient('gpt-5.6-codex', { provider: profile.provider, base_url: profile.base_url, api_key: profile.api_key, codex_transport: 'sse' }, {
          ...profileSessionOptions(profile, registry),
          fetchImplementation: async (_url, init) => {
            observed.push(new Headers(init?.headers).get('Authorization'))
            return new Response('data: {"type":"response.completed","response":{"status":"completed"}}\n\ndata: [DONE]\n\n', { headers: { 'Content-Type': 'text/event-stream' } })
          },
        })
        for await (const _event of client.stream({ model: 'gpt-5.6-codex', messages: [{ role: 'user', content: purpose }] })) { /* consume the actual native adapter */ }
      }
    }
    expect(observed).toEqual(['Bearer fixture-A', 'Bearer fixture-A', 'Bearer fixture-A', 'Bearer fixture-B', 'Bearer fixture-B', 'Bearer fixture-B', 'Bearer fixture-A', 'Bearer fixture-A', 'Bearer fixture-A'])
    expect(chatA.metadata.provider_profile).toBe(a.profileName)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('a missing isolated account fails instead of borrowing an ambient session', async () => {
  const registry = new ProviderAccountRegistry(join(tmpdir(), 'xr-account-missing-fixture'))
  const options = profileSessionOptions({ oauth_account_id: '33333333-3333-4333-8333-333333333333' }, registry)
  await expect(options.codexSession?.credential()).rejects.toThrow()
  expect(profileSessionOptions({})).toEqual({})
})

test('a missing named Claude sign-in cannot fall back to an ambient API key', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-claude-no-fallback-'))
  const previous = process.env.ANTHROPIC_API_KEY
  process.env.ANTHROPIC_API_KEY = 'fixture-ambient-key'
  try {
    const registry = new ProviderAccountRegistry(join(directory, 'accounts'), join(directory, 'profiles.json'))
    const account = await registry.create('claude', 'Empty')
    let fetched = false
    const client = createLlmClient('claude-sonnet-4-5', { provider: 'anthropic', api_key: '' }, {
      ...profileSessionOptions({ oauth_account_id: account.id }, registry),
      fetchImplementation: async () => {
        fetched = true
        return new Response('data: {"type":"message_stop"}\n\n', { headers: { 'Content-Type': 'text/event-stream' } })
      },
    })
    const run = async () => { for await (const _event of client.stream({ model: 'claude-sonnet-4-5', messages: [{ role: 'user', content: 'fixture' }] })) { /* consume */ } }
    await expect(run()).rejects.toThrow()
    expect(fetched).toBe(false)
  } finally {
    if (previous === undefined) delete process.env.ANTHROPIC_API_KEY
    else process.env.ANTHROPIC_API_KEY = previous
    await rm(directory, { recursive: true, force: true })
  }
})

test('account profile edits retain privacy, sampling, and model overrides', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-account-settings-'))
  try {
    const profiles = new ProfileStore(join(directory, 'profiles.json'))
    const registry = new ProviderAccountRegistry(join(directory, 'accounts'), profiles.filePath)
    const account = await registry.create('claude', 'Lead')
    const original = profiles.get(account.profileName)
    if (!original) throw new Error('Missing linked profile')
    profiles.save({ name: original.name, baseUrl: original.base_url, provider: original.provider, apiKey: '', model: original.model, dataTrust: 'cloud', promptProfile: 'lean', sampling: { temperature: 0.2 } })
    profiles.updateModelCapabilities(original.name, original.model, { contextLimit: 8192 })
    profiles.save({ name: original.name, baseUrl: original.base_url, provider: original.provider, apiKey: '', model: original.model })
    expect(profiles.get(original.name)).toMatchObject({ oauth_account_id: account.id, data_trust: 'cloud', prompt_profile: 'lean', sampling: { temperature: 0.2 }, model_overrides: { [original.model]: { context_limit: 8192 } } })
    profiles.save({ name: original.name, baseUrl: 'https://other.example/v1', provider: original.provider, apiKey: 'fixture-other', model: original.model })
    expect(profiles.get(original.name)?.oauth_account_id).toBeUndefined()
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('named-account identity reaches auxiliary and SSH-relay client factories too', async () => {
  for (const name of ['advisorWiring', 'providerRelays', 'titleGenerator', 'server']) {
    const source = await Bun.file(new URL(`../src/daemon/${name}.ts`, import.meta.url)).text()
    expect(source).toContain('profileSessionOptions(profile)')
  }
  const server = await Bun.file(new URL('../src/daemon/server.ts', import.meta.url)).text()
  expect(server).toContain('profileSessionOptions(profile).codexSession')
})

test('all daemon profile rebuilds pass account sessions as factory options, not wire overrides', async () => {
  const source = await Bun.file(new URL('../src/cli.ts', import.meta.url)).text()
  // Guard every profile-bound rebuild: turns, fallback, routing, planner and compaction.
  const calls = source.split('\n').filter(line => line.includes('createLlmClient(') && line.includes('profile.api_key'))
  expect(calls.length).toBeGreaterThanOrEqual(6)
  for (const call of calls) expect(call).toContain('profileSessionOptions(profile)')
  expect(source).not.toContain('...oauthSessions,')
})
