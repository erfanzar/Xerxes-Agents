// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, spyOn, test } from 'bun:test'

import { ProviderAccountRegistry } from '../src/auth/providerAccountRegistry.js'
import { ProfileStore } from '../src/bridge/profiles.js'
import { runtimeConnection } from '../src/daemon/runtimeConnection.js'
import { GlobalProviderAccounts } from '../src/desktop/main/globalProviderAccounts.js'

test('a duplicate account row for one profile never hides the account the profile belongs to', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-account-duplicate-'))
  try {
    const root = join(directory, 'accounts'), profilesFile = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(root, profilesFile)
    const real = await registry.create('claude', 'Personal')
    // A second row claiming the same linked profile — the shape left behind by
    // two registrations that each saw a free profile name.
    const stale = { id: '00000000-0000-4000-8000-000000000000', kind: 'claude' as const, label: 'Personal', profileName: real.profileName, createdAt: new Date(0).toISOString() }
    await writeFile(join(root, 'accounts.json'), JSON.stringify({ accounts: [stale, real] }, null, 2) + '\n')

    const listed = await registry.list()
    expect(listed.filter(account => account.profileName === real.profileName).map(account => account.id)).toEqual([real.id])
    // The survivor is the one the profile actually points at, so the receiver's
    // account-id check can pass.
    await mkdir(join(root, real.id, 'auth'), { recursive: true })
    await writeFile(join(root, real.id, 'auth', 'anthropic-oauth.json'), JSON.stringify({ access: 'fixture-access', refresh: 'fixture-refresh', expires: 4102444800 }))
    expect((await registry.syncPayload(real.id)).kind).toBe('claude')
    await expect(registry.syncPayload(stale.id)).rejects.toThrow('Provider account not found')
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('a signed-in account whose linked profile vanished is relinked, not hidden', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-account-relink-'))
  try {
    const root = join(directory, 'accounts'), profilesFile = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(root, profilesFile)
    const signedIn = await registry.create('codex', 'Erfan')
    const signedOut = await registry.create('codex', 'Omid')
    await mkdir(join(root, signedIn.id, '.codex'), { recursive: true })
    await writeFile(join(root, signedIn.id, '.codex', 'auth.json'), JSON.stringify({ tokens: { access_token: 'fixture' } }))
    // Another writer rewrote profiles.json without either linked profile.
    const store = new ProfileStore(profilesFile)
    store.delete(signedIn.profileName)
    store.delete(signedOut.profileName)

    const listed = await registry.list()
    expect(listed.map(account => account.id)).toContain(signedIn.id)
    expect(new ProfileStore(profilesFile).get(signedIn.profileName)?.oauth_account_id).toBe(signedIn.id)
    // No saved sign-in: nothing to relink, so it stays hidden.
    expect(listed.map(account => account.id)).not.toContain(signedOut.id)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

const CODEX_URL = 'https://chatgpt.com/backend-api/codex'
const fixtureCatalog = async () => ['plan-model-a', 'plan-model-b']

test('a Claude account on the broken anthropic transport moves to claude-code, keeping every other field', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-account-repair-claude-'))
  try {
    const root = join(directory, 'accounts'), profilesFile = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(root, profilesFile, {}, { codexModels: fixtureCatalog })
    const legacy = await registry.create('claude', 'Personal')
    const chosen = await registry.create('claude', 'Prism')
    const store = new ProfileStore(profilesFile)
    // Old default pair: transport and model both legacy.
    store.save({ name: legacy.profileName, apiKey: '', model: 'claude-sonnet-4-5', provider: 'anthropic', baseUrl: 'https://api.anthropic.com', setActive: false, oauthAccountId: legacy.id, sampling: { temperature: 0.3 } })
    store.updateModelCapabilities(legacy.profileName, 'claude-sonnet-4-5', { contextLimit: 123_000 })
    // The person chose another model on the broken transport: only the transport changes.
    store.save({ name: chosen.profileName, apiKey: '', model: 'claude-opus-4-1', provider: 'anthropic', baseUrl: 'https://api.anthropic.com', setActive: false, oauthAccountId: chosen.id })

    await registry.list()
    const after = new ProfileStore(profilesFile)
    expect(after.get(legacy.profileName)).toMatchObject({ provider: 'claude-code', model: 'claude-code/sonnet', base_url: 'https://api.anthropic.com', oauth_account_id: legacy.id, sampling: { temperature: 0.3 } })
    expect(after.get(legacy.profileName)?.model_overrides?.['claude-sonnet-4-5']).toEqual({ context_limit: 123_000 })
    expect(after.get(chosen.profileName)).toMatchObject({ provider: 'claude-code', model: 'claude-opus-4-1' })
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('a claude-code account choosing claude-sonnet-4-5 is never reverted', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-account-repair-keep-'))
  try {
    const root = join(directory, 'accounts'), profilesFile = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(root, profilesFile, {}, { codexModels: fixtureCatalog })
    const account = await registry.create('claude', 'Personal')
    new ProfileStore(profilesFile).updateConnection(account.profileName, { model: 'claude-sonnet-4-5' })
    await registry.list()
    await registry.list()
    expect(new ProfileStore(profilesFile).get(account.profileName)).toMatchObject({ provider: 'claude-code', model: 'claude-sonnet-4-5' })
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('a Codex account takes its live catalog model; only the exact legacy id is repaired', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-account-repair-codex-'))
  try {
    const root = join(directory, 'accounts'), profilesFile = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(root, profilesFile, {}, { codexModels: fixtureCatalog })
    const created = await registry.create('codex', 'Erfan')
    const store = new ProfileStore(profilesFile)
    store.updateConnection(created.profileName, { model: 'gpt-5.6-codex' })
    await registry.list()
    expect(new ProfileStore(profilesFile).get(created.profileName)).toMatchObject({ model: 'plan-model-a', base_url: CODEX_URL })
    // A model the person picked later is kept.
    new ProfileStore(profilesFile).updateConnection(created.profileName, { model: 'gpt-6-astra' })
    await registry.list()
    expect(new ProfileStore(profilesFile).get(created.profileName)?.model).toBe('gpt-6-astra')
    // Unreachable catalog: the legacy id stays rather than being guessed.
    const offline = new ProviderAccountRegistry(root, profilesFile, {}, { codexModels: async () => { throw new Error('offline') } })
    new ProfileStore(profilesFile).updateConnection(created.profileName, { model: 'gpt-5.6-codex' })
    await offline.list()
    expect(new ProfileStore(profilesFile).get(created.profileName)?.model).toBe('gpt-5.6-codex')
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('a relinked Codex profile uses the live catalog model', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-account-relink-live-'))
  try {
    const root = join(directory, 'accounts'), profilesFile = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(root, profilesFile, {}, { codexModels: fixtureCatalog })
    const account = await registry.create('codex', 'Erfan')
    await mkdir(join(root, account.id, '.codex'), { recursive: true })
    await writeFile(join(root, account.id, '.codex', 'auth.json'), JSON.stringify({ tokens: { access_token: 'fixture' } }))
    new ProfileStore(profilesFile).delete(account.profileName)
    await registry.list()
    expect(new ProfileStore(profilesFile).get(account.profileName)?.model).toBe('plan-model-a')
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('registering an account cannot leave two rows for the same linked profile', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-account-unique-'))
  try {
    const root = join(directory, 'accounts'), profilesFile = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(root, profilesFile)
    const first = await registry.create('claude', 'Personal')
    // Re-create the profile the first account owns under another id, as a
    // racing registration would.
    new ProfileStore(profilesFile).save({ name: first.profileName, apiKey: '', model: 'x', provider: 'anthropic', baseUrl: 'https://api.anthropic.com', setActive: false, oauthAccountId: '00000000-0000-4000-8000-000000000000' })
    const document = JSON.parse(await readFile(join(root, 'accounts.json'), 'utf8')) as { accounts: Array<{ profileName: string }> }
    const names = document.accounts.filter(row => row.profileName === first.profileName)
    expect(names).toHaveLength(1)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('multiple OAuth accounts have isolated directories and switchable profiles', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-account-registry-'))
  try {
    const root = join(directory, 'accounts'), profilesFile = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(root, profilesFile)
    const claude = await registry.create('claude', 'work')
    const codex = await registry.create('codex', 'personal')
    expect(claude.id).not.toBe(codex.id)
    expect(claude.profileName).not.toBe(codex.profileName)
    const profiles = new ProfileStore(profilesFile).list()
    const linked = profiles.filter(profile => profile.oauth_account_id)
    expect(linked).toHaveLength(2)
    expect(profiles.find(profile => profile.name === claude.profileName)?.oauth_account_id).toBe(claude.id)
    expect(profiles.find(profile => profile.name === codex.profileName)?.oauth_account_id).toBe(codex.id)
    expect((await registry.list()).map(account => account.kind)).toEqual(['claude', 'codex'])
    const claudeDir = join(root, claude.id), codexDir = join(root, codex.id)
    await mkdir(join(claudeDir, 'auth'), { recursive: true })
    await writeFile(join(claudeDir, 'auth', 'anthropic-oauth.json'), JSON.stringify({ access: 'claude-access', refresh: 'claude-refresh', expires: Date.now() / 1000 + 3600 }))
    await mkdir(join(codexDir, '.codex'), { recursive: true })
    await writeFile(join(codexDir, '.codex', 'auth.json'), JSON.stringify({ tokens: { access_token: 'codex-access', refresh_token: 'codex-refresh', token_type: 'Bearer', expires_at: Date.now() / 1000 + 3600, scopes: [] } }))
    const claudePayload = await registry.syncPayload(claude.id)
    const codexPayload = await registry.syncPayload(codex.id)
    expect(claudePayload.kind).toBe('claude')
    expect(codexPayload.kind).toBe('codex')
    expect(JSON.stringify(claudePayload)).toContain('claude-refresh')
    expect(JSON.stringify(codexPayload)).toContain('codex-refresh')
    expect(claudePayload.kind === 'claude' && claudePayload.profile.oauth_account_id).toBe(claude.id)
    // A Claude sign-in runs through the Claude Code transport, used once the person picked it.
    const connection = runtimeConnection({ channels: {}, control: {}, maxConcurrentTurns: 1, projectDirectory: directory, runtime: {}, workspace: {} }, linked[0], { claudeCodeChosen: true, claudeCodeInstalled: () => true })
    expect(connection?.oauthAccountId).toBe(claude.id)
    expect(registry.sessionOptions(claude.id).anthropicOAuthSession?.storedPath()).toBe(join(claudeDir, 'auth', 'anthropic-oauth.json'))
    await registry.remove(codex.id)
    expect((await registry.list()).map(account => account.id)).toEqual([claude.id])
    await expect(readFile(join(codexDir, '.codex', 'auth.json'), 'utf8')).rejects.toThrow()
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('registration is committed only after OAuth produced a credential', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-login-transaction-'))
  try {
    const root = join(directory, 'accounts'), file = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(root, file)
    const account = await registry.prepare('claude', 'work')
    expect(await registry.list()).toEqual([])
    expect(new ProfileStore(file).list().filter(profile => profile.oauth_account_id)).toEqual([])
    await expect(registry.finishLogin(account.id)).rejects.toThrow('no saved credential')
    expect(await registry.list()).toEqual([])
    expect(new ProfileStore(file).list().filter(profile => profile.oauth_account_id)).toEqual([])
    await mkdir(join(root, account.id, 'auth'), { recursive: true })
    await writeFile(join(root, account.id, 'auth', 'anthropic-oauth.json'), JSON.stringify({ access: 'access', refresh: 'refresh', expires: 4102444800 }))
    await registry.finishLogin(account.id)
    expect((await registry.list()).map(row => row.id)).toEqual([account.id])
    expect(new ProfileStore(file).get(account.profileName)?.oauth_account_id).toBe(account.id)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('OAuth login runs through the Bun daemon, not Electron', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-login-runtime-'))
  try {
    const root = join(directory, 'accounts'), file = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(root, file)
    spyOn(registry, 'usage').mockResolvedValue({ provider: 'claude', fetchedAt: 0, windows: [] })
    const calls: { method: string; params: Record<string, unknown> }[] = []
    const center = new GlobalProviderAccounts(registry, undefined, async () => ({ ok: true }), async (method, params) => {
      calls.push({ method, params })
      const accountId = String(params.account_id)
      await mkdir(join(root, accountId, 'auth'), { recursive: true })
      await writeFile(join(root, accountId, 'auth', 'anthropic-oauth.json'), JSON.stringify({ access: 'access', refresh: 'refresh', expires: 4102444800 }))
      await registry.finishLogin(accountId)
      return { ok: true }
    })
    const result = await center.login('claude', 'work') as { account: { id: string } }
    expect(calls).toEqual([{ method: 'provider_account.login', params: { account_id: result.account.id } }])
    expect((await registry.list()).map(account => account.id)).toEqual([result.account.id])
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('a failed or unfinished sign-in never leaves an account or linked profile', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-login-failure-'))
  try {
    const file = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(join(directory, 'accounts'), file)
    const center = new GlobalProviderAccounts(registry, undefined, async () => ({ ok: true }), async () => {
      throw new Error('Failed to start server. Is port 1455 in use?')
    })
    await expect(center.login('codex', 'Personal')).rejects.toThrow('port 1455')
    expect(await registry.list()).toEqual([])
    expect(new ProfileStore(file).list().filter(profile => profile.oauth_account_id)).toEqual([])
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('an unfinished helper and a mismatched sync receipt never report success', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-account-ack-'))
  try {
    const file = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(join(directory, 'accounts'), file)
    const unfinished = new GlobalProviderAccounts(registry, undefined, async () => ({ ok: true }), async () => ({ ok: true }))
    await expect(unfinished.login('claude', 'unfinished')).rejects.toThrow('did not complete')
    expect(await registry.list()).toEqual([])
    const account = await registry.create('claude', 'work')
    await Bun.write(join(directory, 'accounts', account.id, 'auth', 'anthropic-oauth.json'), JSON.stringify({ access: 'fixture-access', refresh: 'fixture-refresh', expires: 4102444800 }))
    let reloads = 0, writes = 0
    const center = new GlobalProviderAccounts(registry, { alias: 'host', target: 'user@example.com', workspacePath: '/work/repo' }, async method => {
      if (method === 'runtime.status') return { provider_account_isolation_v1_supported: true }
      if (method === 'runtime.reload') reloads++
      return { ok: true }
    }, async () => ({ ok: true }), () => 1000, async () => { writes++; return JSON.stringify({ ok: true, profile: 'wrong-profile' }) })
    const review = center.reviewSync(account.id) as { review: string }
    await expect(center.confirmSync(review.review, true)).rejects.toThrow('not acknowledged')
    await expect(center.confirmSync(review.review, true)).rejects.toThrow('expired')
    expect(writes).toBe(1)
    expect(reloads).toBe(0)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('the global center reviews sync and activates the linked profile', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-global-center-'))
  try {
    const root = join(directory, 'accounts'), file = join(directory, 'profiles.json')
    const registry = new ProviderAccountRegistry(root, file)
    const account = await registry.create('claude', 'work')
    await mkdir(join(root, account.id, 'auth'), { recursive: true })
    await writeFile(join(root, account.id, 'auth', 'anthropic-oauth.json'), JSON.stringify({ access: 'secret-access', refresh: 'secret-refresh', expires: 4102444800 }))
    const calls: { method: string; params: Record<string, unknown> }[] = [], writes: unknown[] = []
    const center = new GlobalProviderAccounts(registry, { alias: 'host', target: 'user@example.com', workspacePath: '/work/repo' }, async (method, params) => {
      calls.push({ method, params })
      if (method === 'runtime.status') return { provider_account_isolation_v1_supported: true }
      return method === 'provider_list' ? { ok: true, profiles: [{ name: account.profileName }] } : { ok: true }
    }, async () => ({ ok: true }), () => 1000, async payload => { writes.push(payload); return JSON.stringify({ ok: true, profile: account.profileName }) })
    const listed = await center.list() as { accounts: { id: string; available: boolean }[]; can_sync: boolean }
    expect(listed.accounts[0]).toMatchObject({ id: account.id, available: false })
    expect(JSON.stringify(listed)).not.toContain('secret-refresh')
    const review = center.reviewSync(account.id) as { review: string }
    expect(await center.confirmSync(review.review, true)).toMatchObject({ ok: true, profile: account.profileName })
    expect(writes).toHaveLength(1)
    await center.activate(account.id, 'session-1')
    expect(calls.map(call => call.method)).toEqual(['provider_list', 'runtime.status', 'runtime.reload', 'runtime.status', 'provider_list', 'provider_select'])
    expect(calls.at(-1)?.params).toEqual({ name: account.profileName, session_key: 'session-1' })
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('named API-key profiles join the global account list and sync payload', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-api-account-'))
  try {
    const file = join(directory, 'profiles.json')
    new ProfileStore(file).save({ name: 'zai-work', provider: 'zai', baseUrl: 'https://api.z.ai/api/paas/v4', apiKey: 'key', model: 'glm-5', setActive: false })
    const registry = new ProviderAccountRegistry(join(directory, 'accounts'), file)
    const accounts = await registry.list()
    expect(accounts[0]).toMatchObject({ id: 'api:zai-work', kind: 'api_key', profileName: 'zai-work' })
    const payload = await registry.syncPayload('api:zai-work')
    expect(payload.kind).toBe('api_key')
    expect(payload.kind === 'api_key' && payload.profile.api_key).toBe('key')
  } finally { await rm(directory, { recursive: true, force: true }) }
})
