// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ProviderAccountRegistry } from '../src/auth/providerAccountRegistry.js'
import { buildUsageReport, UsageReportCache } from '../src/auth/usageReport.js'
import { ProfileStore } from '../src/bridge/profiles.js'

test('named accounts use isolated sign-ins and distinct cached quota answers without deleting empty accounts', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xerxes-usage-accounts-'))
  try {
    const root = join(dir, 'accounts'), file = join(dir, 'profiles.json'), registry = new ProviderAccountRegistry(root, file)
    const first = await registry.create('codex', 'First'), second = await registry.create('codex', 'Second')
    const claude = await registry.create('claude', 'Personal'), empty = await registry.create('claude', 'Empty')
    for (const [account, token] of [[first, 'fixture-first'], [second, 'fixture-second']] as const) {
      await mkdir(join(root, account.id, '.codex'), { recursive: true })
      await writeFile(join(root, account.id, '.codex', 'auth.json'), JSON.stringify({ tokens: { access_token: token, refresh_token: 'fixture-refresh', token_type: 'Bearer', expires_at: 4102444800, scopes: [] } }))
    }
    await mkdir(join(root, claude.id, 'auth'), { recursive: true })
    await writeFile(join(root, claude.id, 'auth', 'anthropic-oauth.json'), JSON.stringify({ access: 'fixture-personal', refresh: 'fixture-refresh', expires: 4102444800 }))
    const seen: string[] = [], legacy: string[] = []
    const fetchImplementation = async (_url: string, init?: RequestInit) => {
      const token = new Headers(init?.headers).get('authorization') ?? ''
      seen.push(token)
      return new Response(JSON.stringify(token.includes('personal') ? { five_hour: { utilization: 33 } } : { rate_limit: { primary_window: { used_percent: token.includes('first') ? 11 : 22 } } }))
    }
    const profiles = new ProfileStore(file).list().filter(profile => profile.oauth_account_id)
    const cache = new UsageReportCache()
    const options = { cache, accountUsage: (id: string) => registry.subscriptionUsage(id, { fetchImplementation }), fetchUsage: async () => { legacy.push('legacy'); return { provider: 'codex', fetchedAt: 0, windows: [{ label: '5-hour', usedPercent: 90 }] } } }
    const report = await buildUsageReport([...profiles, { name: 'codex', provider: 'openai-codex', model: 'fixture', api_key: '', base_url: '' }], options)
    const percent = (name: string) => report.find(entry => entry.profile === name)?.windows[0]?.usedPercent
    expect(percent(first.profileName)).toBe(11)
    expect(percent(second.profileName)).toBe(22)
    expect(percent(claude.profileName)).toBe(33)
    expect(percent('codex')).toBe(90)
    expect(report.find(entry => entry.profile === empty.profileName)).toMatchObject({ status: 'error', windows: [] })
    expect(legacy).toEqual(['legacy'])
    expect(seen).toHaveLength(3)
    await buildUsageReport(profiles, options)
    expect(seen).toHaveLength(3)
    expect((await registry.list()).map(account => account.id)).toContain(empty.id)
  } finally { await rm(dir, { recursive: true, force: true }) }
})

test('an empty named Claude account never borrows an ambient OAuth token', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xerxes-no-ambient-usage-'))
  const previous = process.env.ANTHROPIC_AUTH_TOKEN
  process.env.ANTHROPIC_AUTH_TOKEN = 'fixture-ambient-token'
  try {
    const registry = new ProviderAccountRegistry(join(dir, 'accounts'), join(dir, 'profiles.json'))
    const account = await registry.create('claude', 'Empty')
    await expect(registry.sessionOptions(account.id).anthropicOAuthSession!.credential()).rejects.toThrow()
    await expect(registry.subscriptionUsage(account.id, { fetchImplementation: async () => { throw new Error('Must not fetch') } })).rejects.toThrow('no saved sign-in')
    expect(await registry.list()).toHaveLength(1)
  } finally {
    if (previous === undefined) delete process.env.ANTHROPIC_AUTH_TOKEN
    else process.env.ANTHROPIC_AUTH_TOKEN = previous
    await rm(dir, { recursive: true, force: true })
  }
})

test('missing named-account metadata is an error, never a shared-login fallback', async () => {
  let shared = 0
  const [entry] = await buildUsageReport([{ name: 'missing', provider: 'openai-codex', model: 'fixture', api_key: '', base_url: '', oauth_account_id: '00000000-0000-0000-0000-000000000000' }], {
    accountUsage: async () => { throw new Error('Provider account not found') },
    fetchUsage: async () => { shared++; return { provider: 'codex', fetchedAt: 0, windows: [] } },
  })
  expect(entry).toMatchObject({ status: 'error', message: 'Provider account not found' })
  expect(shared).toBe(0)
})
