// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

import { expect, spyOn, test } from 'bun:test'

import { ProviderAccountRegistry, type ProviderAccountKind } from '../src/auth/providerAccountRegistry.js'
import { ProfileStore } from '../src/bridge/profiles.js'
import { GlobalProviderAccounts } from '../src/desktop/main/globalProviderAccounts.js'

const target = { alias: 'fixture-host', target: 'user@example.test', workspacePath: '/workspace' }
const login = async () => { throw new Error('Sign-in must not start during account inspection') }
const kinds = ['claude', 'claude-code', 'codex', 'api_key'] as const

async function fixture(kind: ProviderAccountKind) {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-center-identity-'))
  const root = join(directory, 'accounts'), profiles = new ProfileStore(join(directory, 'profiles.json'))
  const registry = new ProviderAccountRegistry(root, profiles.filePath)
  const metadata = kind === 'api_key' ? undefined : await registry.create(kind, 'Work')
  if (!metadata) profiles.save({ name: 'api-work', provider: 'openai', baseUrl: 'https://api.example.test/v1', apiKey: 'fixture-secret-A', model: 'fixture-model', setActive: false })
  const account = (await registry.list())[0]
  if (!account) throw new Error('Missing fixture account')
  const profile = profiles.get(account.profileName)
  if (!profile) throw new Error('Missing fixture profile')
  const publicProfile = { name: profile.name, provider: profile.provider, base_url: profile.base_url }
  const changeCredential = async (suffix: string) => {
    if (kind === 'api_key') {
      profiles.save({ name: profile.name, provider: profile.provider, baseUrl: profile.base_url, apiKey: `fixture-secret-${suffix}`, model: profile.model, setActive: false })
    } else {
      const file = join(root, account.id, ...(kind === 'codex' ? ['.codex', 'auth.json'] : ['auth', 'anthropic-oauth.json']))
      await mkdir(dirname(file), { recursive: true })
      await writeFile(file, JSON.stringify(kind === 'codex'
        ? { tokens: { access_token: `fixture-secret-${suffix}`, refresh_token: `fixture-refresh-${suffix}`, token_type: 'Bearer', expires_at: 4102444800, scopes: [] } }
        : { access: `fixture-secret-${suffix}`, refresh: `fixture-refresh-${suffix}`, expires: 4102444800 }))
    }
  }
  await changeCredential('A')
  return { directory, registry, profiles, account, publicProfile, changeCredential }
}

for (const kind of kinds) {
  test(`${kind} SSH accounts refuse same-named remote credentials until a one-use confirmed sync`, async () => {
    const { directory, registry, account, publicProfile, changeCredential } = await fixture(kind)
    try {
      const calls: string[] = [], writes: unknown[] = []
      const call = async (method: string) => {
        calls.push(method)
        if (method === 'runtime.status') return { provider_account_isolation_v1_supported: true }
        if (method === 'provider_list') return { profiles: [{ ...publicProfile, oauth_account_id: 'unrelated-remote-login', has_api_key: true }] }
        expect(['runtime.reload', 'provider_select']).toContain(method)
        return { ok: true }
      }
      const center = new GlobalProviderAccounts(registry, target, call, login, () => 1000, async payload => {
        writes.push(payload)
        return JSON.stringify({ ok: true, profile: account.profileName })
      })
      expect(await center.list()).toMatchObject({ accounts: [{ id: account.id, available: false }] })
      await expect(center.activate(account.id, 's1')).rejects.toThrow('not available')
      expect(writes).toHaveLength(0)
      expect(calls).not.toContain('provider_select')
      const review = center.reviewSync(account.id) as { review: string }
      expect(writes).toHaveLength(0)
      await expect(center.confirmSync(review.review, false)).rejects.toThrow('review')
      expect(writes).toHaveLength(0)
      const receipt = await center.confirmSync(review.review, true)
      expect(receipt).toMatchObject({ ok: true, profile: account.profileName })
      expect(writes).toHaveLength(1)
      expect(JSON.stringify(writes)).toContain('fixture-secret-A')
      await expect(center.confirmSync(review.review, true)).rejects.toThrow('review')
      const listed = await center.list()
      expect(listed).toMatchObject({ accounts: [{ available: true }] })
      expect(JSON.stringify([receipt, listed])).not.toContain('fixture-secret')
      expect(JSON.stringify([receipt, listed])).not.toContain('fixture-refresh')
      await expect(center.activate(account.id, 's1')).resolves.toEqual({ ok: true })
      expect(calls.filter(method => method === 'provider_select')).toHaveLength(1)
      expect(calls.filter(method => method === 'runtime.reload')).toHaveLength(1)
      // A different center instance cannot inherit this session's confirmation.
      const reopened = new GlobalProviderAccounts(registry, target, call, login)
      expect(await reopened.list()).toMatchObject({ accounts: [{ available: false }] })
      await expect(reopened.activate(account.id, 's1')).rejects.toThrow('not available')
      // Rotating the local key or OAuth login invalidates the exact-credential proof.
      await changeCredential('B')
      expect(await center.list()).toMatchObject({ accounts: [{ available: false }] })
      await expect(center.activate(account.id, 's1')).rejects.toThrow('not available')
      expect(writes).toHaveLength(1)
      expect(calls.filter(method => method === 'provider_select')).toHaveLength(1)
    } finally { await rm(directory, { recursive: true, force: true }) }
  })

  test(`${kind} SSH sync and activation require the explicit isolation capability, never a version guess`, async () => {
    const { directory, registry, account, publicProfile } = await fixture(kind)
    try {
      let status: Record<string, unknown> | 'throw' = {}, writes = 0, selected = 0
      const calls: string[] = [], payloads = spyOn(registry, 'syncPayload')
      const center = new GlobalProviderAccounts(registry, target, async method => {
        calls.push(method)
        if (method === 'runtime.status') {
          if (status === 'throw') throw new Error('fixture-secret-A status error')
          return status
        }
        if (method === 'provider_list') return { profiles: [publicProfile] }
        if (method === 'provider_select') selected++
        return { ok: true }
      }, login, () => 1000, async () => {
        writes++
        return JSON.stringify({ ok: true, profile: account.profileName })
      })
      const unsupported: (Record<string, unknown> | 'throw')[] = [
        { version: '999.0.0' },
        { provider_account_isolation_v1_supported: false },
        { provider_account_isolation_v1_supported: 'true' },
        { provider_account_isolation_v1_supported: 1 },
        { ok: false, provider_account_isolation_v1_supported: true, error: 'fixture-secret-A status refusal' },
        'throw',
      ]
      for (const response of unsupported) {
        status = response
        calls.length = 0
        const review = center.reviewSync(account.id) as { review: string }
        let error: unknown
        try { await center.confirmSync(review.review, true) } catch (caught) { error = caught }
        expect(String(error)).toContain('update remote runtime to 0.8.35')
        expect(String(error)).not.toContain('fixture-secret')
        expect(calls).toEqual(['runtime.status'])
        await expect(center.confirmSync(review.review, true)).rejects.toThrow('review')
        calls.length = 0
        await expect(center.activate(account.id, 's1')).rejects.toThrow('update remote runtime to 0.8.35')
        expect(calls).toEqual(['runtime.status'])
      }
      expect(payloads).not.toHaveBeenCalled()
      expect(writes).toBe(0)
      expect(selected).toBe(0)
      status = { provider_account_isolation_v1_supported: true, version: '0.0.0' }
      const review = center.reviewSync(account.id) as { review: string }
      await center.confirmSync(review.review, true)
      await center.activate(account.id, 's1')
      expect(writes).toBe(1)
      expect(selected).toBe(1)
      // Probe again on every action: a reconnect/downgrade cannot reuse support.
      status = {}
      await expect(center.activate(account.id, 's1')).rejects.toThrow('update remote runtime to 0.8.35')
      const retry = center.reviewSync(account.id) as { review: string }
      await expect(center.confirmSync(retry.review, true)).rejects.toThrow('update remote runtime to 0.8.35')
      expect(writes).toBe(1)
      expect(selected).toBe(1)
      expect(await center.list()).toMatchObject({ accounts: [{ available: false }] })
      payloads.mockRestore()
    } finally { await rm(directory, { recursive: true, force: true }) }
  })

  test(`${kind} local availability and selection do not require sync or read credentials`, async () => {
    const { directory, registry, account, publicProfile } = await fixture(kind)
    try {
      const credentials = spyOn(registry, 'syncPayload').mockRejectedValue(new Error('fixture-secret: credentials must not be read'))
      const selected: unknown[] = []
      const center = new GlobalProviderAccounts(registry, undefined, async (method, params) => {
        if (method === 'provider_list') return { profiles: [publicProfile] }
        expect(method).toBe('provider_select')
        selected.push(params)
        return { ok: true, selected: true }
      }, login)
      expect(await center.list()).toMatchObject({ accounts: [{ available: true }] })
      await expect(center.activate(account.id, 's1')).resolves.toEqual({ ok: true, selected: true })
      expect(selected).toEqual([{ name: account.profileName, session_key: 's1' }])
      expect(credentials).not.toHaveBeenCalled()
      credentials.mockRestore()
    } finally { await rm(directory, { recursive: true, force: true }) }
  })

  test(`${kind} SSH credential lookup/selection failures are safe and failed resync revokes availability`, async () => {
    const { directory, registry, account, publicProfile } = await fixture(kind)
    try {
      let selection: 'ok' | 'refuse' | 'throw' = 'ok', failWrite = false
      const center = new GlobalProviderAccounts(registry, target, async method => {
        if (method === 'runtime.status') return { provider_account_isolation_v1_supported: true }
        if (method === 'provider_list') return { profiles: [publicProfile] }
        if (method === 'provider_select' && selection === 'refuse') return { ok: false, error: 'fixture-secret-A refusal' }
        if (method === 'provider_select' && selection === 'throw') throw new Error('fixture-secret-A selection error')
        return { ok: true }
      }, login, () => 1000, async () => failWrite ? 'fixture-secret-A uncertain write' : JSON.stringify({ ok: true, profile: account.profileName }))
      const review = center.reviewSync(account.id) as { review: string }
      await center.confirmSync(review.review, true)
      expect(await center.list()).toMatchObject({ accounts: [{ available: true }] })
      const lookup = spyOn(registry, 'syncPayload').mockRejectedValue(new Error('fixture-secret-A credential read failed'))
      try {
        const listed = await center.list()
        expect(listed).toMatchObject({ accounts: [{ available: false }] })
        expect(JSON.stringify(listed)).not.toContain('fixture-secret')
        await expect(center.activate(account.id, 's1')).rejects.toThrow('not available')
      } finally { lookup.mockRestore() }
      for (const failure of ['refuse', 'throw'] as const) {
        selection = failure
        let error: unknown
        try { await center.activate(account.id, 's1') } catch (caught) { error = caught }
        expect(String(error)).toContain('Could not switch Provider account')
        expect(String(error)).not.toContain('fixture-secret')
      }
      selection = 'ok'
      failWrite = true
      const retry = center.reviewSync(account.id) as { review: string }
      await expect(center.confirmSync(retry.review, true)).rejects.toThrow('not acknowledged')
      expect(await center.list()).toMatchObject({ accounts: [{ available: false }] })
      await expect(center.activate(account.id, 's1')).rejects.toThrow('not available')
    } finally { await rm(directory, { recursive: true, force: true }) }
  })

  test(`${kind} unacknowledged SSH writes/reloads remain unavailable and never expose secrets`, async () => {
    const { directory, registry, account, publicProfile } = await fixture(kind)
    try {
      for (const failure of ['malformed-receipt', 'mismatched-receipt', 'writer-error', 'reload-error', 'reload-refusal'] as const) {
        let writes = 0, selected = 0
        const center = new GlobalProviderAccounts(registry, target, async method => {
          if (method === 'runtime.status') return { provider_account_isolation_v1_supported: true }
          if (method === 'provider_list') return { profiles: [publicProfile] }
          if (method === 'provider_select') selected++
          if (method === 'runtime.reload' && failure === 'reload-error') throw new Error('fixture-secret-A reload error')
          if (method === 'runtime.reload' && failure === 'reload-refusal') return { ok: false, error: 'fixture-secret-A reload refusal' }
          return { ok: true }
        }, login, () => 1000, async () => {
          writes++
          if (failure === 'writer-error') throw new Error('fixture-secret-A writer error')
          if (failure === 'malformed-receipt') return 'fixture-secret-A invalid JSON'
          return JSON.stringify({ ok: true, profile: failure === 'mismatched-receipt' ? 'fixture-secret-A' : account.profileName })
        })
        const review = center.reviewSync(account.id) as { review: string }
        let error: unknown
        try { await center.confirmSync(review.review, true) } catch (caught) { error = caught }
        expect(String(error)).toContain('not acknowledged')
        expect(String(error)).not.toContain('fixture-secret')
        await expect(center.confirmSync(review.review, true)).rejects.toThrow('review')
        expect(writes).toBe(1)
        expect(await center.list()).toMatchObject({ accounts: [{ available: false }] })
        await expect(center.activate(account.id, 's1')).rejects.toThrow('not available')
        expect(selected).toBe(0)
      }
    } finally { await rm(directory, { recursive: true, force: true }) }
  })
}

test('legacy official Gemini OpenAI roots normalize before local/SSH availability checks', async () => {
  const { directory, registry, profiles } = await fixture('api_key')
  try {
    profiles.save({ name: 'legacy-gemini', provider: 'gemini', baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai', apiKey: 'fixture-gemini-key', model: 'gemini-2.5-flash', setActive: false })
    const accountId = 'api:legacy-gemini'
    const call = async (method: string) => {
      if (method === 'runtime.status') return { provider_account_isolation_v1_supported: true }
      if (method === 'provider_list') return { profiles: [{ name: 'legacy-gemini', provider: 'gemini', base_url: 'https://generativelanguage.googleapis.com/v1beta/openai' }] }
      return { ok: true }
    }
    const local = new GlobalProviderAccounts(registry, undefined, call, login)
    expect(await local.list()).toMatchObject({ accounts: expect.arrayContaining([expect.objectContaining({ id: accountId, available: true })]) })
    await expect(local.activate(accountId, 's1')).resolves.toEqual({ ok: true })
    const remote = new GlobalProviderAccounts(registry, target, call, login, () => 1000, async () => JSON.stringify({ ok: true, profile: 'legacy-gemini' }))
    expect(await remote.list()).toMatchObject({ accounts: expect.arrayContaining([expect.objectContaining({ id: accountId, available: false })]) })
    const review = remote.reviewSync(accountId) as { review: string }
    await remote.confirmSync(review.review, true)
    expect(await remote.list()).toMatchObject({ accounts: expect.arrayContaining([expect.objectContaining({ id: accountId, available: true })]) })
    await expect(remote.activate(accountId, 's1')).resolves.toEqual({ ok: true })
  } finally { await rm(directory, { recursive: true, force: true }) }
})
