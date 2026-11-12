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
    const connection = runtimeConnection({ channels: {}, control: {}, maxConcurrentTurns: 1, projectDirectory: directory, runtime: {}, workspace: {} }, linked[0])
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
    const center = new GlobalProviderAccounts(registry, { alias: 'host', target: 'user@example.com', workspacePath: '/work/repo' }, async () => { reloads++; return { ok: true } }, async () => ({ ok: true }), () => 1000, async () => { writes++; return JSON.stringify({ ok: true, profile: 'wrong-profile' }) })
    const review = center.reviewSync(account.id) as { review: string }
    await expect(center.confirmSync(review.review, true)).rejects.toThrow('different profile')
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
      return method === 'provider_list' ? { ok: true, profiles: [{ name: account.profileName }] } : { ok: true }
    }, async () => ({ ok: true }), () => 1000, async payload => { writes.push(payload); return JSON.stringify({ ok: true, profile: account.profileName }) })
    const listed = await center.list() as { accounts: { id: string; available: boolean }[]; can_sync: boolean }
    expect(listed.accounts[0]).toMatchObject({ id: account.id, available: true })
    expect(JSON.stringify(listed)).not.toContain('secret-refresh')
    const review = center.reviewSync(account.id) as { review: string }
    expect(await center.confirmSync(review.review, true)).toMatchObject({ ok: true, profile: account.profileName })
    expect(writes).toHaveLength(1)
    await center.activate(account.id, 'session-1')
    expect(calls.map(call => call.method)).toEqual(['provider_list', 'runtime.reload', 'provider_select'])
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
