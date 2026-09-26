// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { buildUsageReport, resetIn, UsageReportCache, usageSourceFor } from '../src/auth/usageReport.js'
import type { ProviderUsageReport } from '../src/auth/usage.js'

const profile = (name: string, provider: string, extra: { api_key?: string; active?: boolean; base_url?: string } = {}) =>
  ({ name, provider, api_key: extra.api_key ?? '', base_url: extra.base_url ?? '', model: `${provider}-model`, active: extra.active ?? false })

test('every imported profile gets an answer: windows, unsupported, or the reason', async () => {
  const calls: string[] = []
  const report = await buildUsageReport([
    profile('openai', 'openai', { api_key: 'sk' }),
    profile('zai', 'zai-coding', { api_key: 'z-key' }),
    profile('codex', 'openai-codex', { active: true }),
  ], {
    fetchUsage: async source => {
      calls.push(source)
      if (source === 'zai') throw new Error('Z.ai rejected the key (401)')
      return { provider: source, planType: 'pro', fetchedAt: 1_000, windows: [{ label: '5-hour', usedPercent: 12, resetAfterSeconds: 3_600 }, { label: 'weekly', usedPercent: 7 }] } satisfies ProviderUsageReport
    },
  })
  expect(report.map(entry => [entry.profile, entry.status])).toEqual([['codex', 'ok'], ['zai', 'error'], ['openai', 'unsupported']])
  expect(report[0]).toMatchObject({ active: true, plan: 'pro', source: 'codex', windows: [{ label: '5-hour', usedPercent: 12 }, { label: 'weekly', usedPercent: 7 }] })
  expect(report[1]!.message).toContain('401')
  expect(report[2]!.message).toContain('does not publish usage limits')
  expect(calls.sort()).toEqual(['codex', 'zai'])
})

test('profiles sharing one login are fetched once; separate keys are fetched separately; failures are not cached', async () => {
  const cache = new UsageReportCache(60_000)
  let fetches = 0
  let fail = true
  const fetchUsage = async (source: 'claude' | 'codex' | 'kimi' | 'zai') => {
    fetches += 1
    if (source === 'zai' && fail) throw new Error('offline')
    return { provider: source, fetchedAt: 0, windows: [] }
  }
  const profiles = [profile('codex-a', 'openai-codex'), profile('codex-b', 'codex'), profile('zai-1', 'zai', { api_key: 'k1' }), profile('zai-2', 'zai', { api_key: 'k2' })]
  await buildUsageReport(profiles, { cache, fetchUsage })
  expect(fetches).toBe(3) // one Codex login, two Z.ai keys
  fail = false
  await buildUsageReport(profiles, { cache, fetchUsage })
  expect(fetches).toBe(5) // the failed Z.ai fetches retried; Codex came from cache
  await buildUsageReport(profiles, { cache, fetchUsage, refresh: true })
  expect(fetches).toBe(8)
})

test('reset times read as a countdown', () => {
  const now = 1_000_000_000
  expect(resetIn({ resetsAt: now + (3 * 1_440 + 23 * 60) * 60_000 }, undefined, now)).toBe('3d 23h')
  expect(resetIn({ resetAfterSeconds: 4 * 3_600 + 12 * 60 }, now, now)).toBe('4h 12m')
  expect(resetIn({ resetsAt: now + 18 * 60_000 }, undefined, now)).toBe('18m')
  expect(resetIn({}, now, now)).toBeUndefined()
})

test('the host decides the source: a kimi profile on api.moonshot.cn is a Moonshot key, not a Kimi Code plan', () => {
  expect(usageSourceFor('kimi', 'https://api.moonshot.cn/v1')).toBe('moonshot')
  expect(usageSourceFor('kimi-code', 'https://api.kimi.com/coding/v1')).toBe('kimi')
  expect(usageSourceFor('openai', 'https://openrouter.ai/api/v1')).toBe('openrouter')
  expect(usageSourceFor('zhipu', 'https://api.z.ai/api/coding/paas/v4')).toBe('zai')
  expect(usageSourceFor('openai', 'https://api.openai.com/v1')).toBeUndefined()
})

test('pay-as-you-go keys report their balance or spend limit from the provider', async () => {
  const seen: string[] = []
  const fetchImplementation = async (url: string, init?: RequestInit) => {
    seen.push(`${url} ${(init?.headers as Record<string, string>).Authorization}`)
    const body = url.includes('openrouter') ? { data: { usage: 2.5, limit: 10, is_free_tier: false } }
      : url.includes('deepseek') ? { is_available: true, balance_infos: [{ currency: 'USD', total_balance: '4.20' }] }
      : { code: 0, data: { available_balance: 88.5 } }
    return new Response(JSON.stringify(body), { status: 200 })
  }
  const report = await buildUsageReport([
    profile('or', 'openrouter', { api_key: 'or-key' }),
    profile('ds', 'deepseek', { api_key: 'ds-key' }),
    profile('ms', 'kimi', { api_key: 'ms-key', base_url: 'https://api.moonshot.cn/v1' }),
    profile('nokey', 'deepseek'),
  ], { fetchImplementation })
  const byName = Object.fromEntries(report.map(entry => [entry.profile, entry]))
  expect(byName.or).toMatchObject({ status: 'ok', balance: '$7.50 left', windows: [{ label: 'Key limit', usedPercent: 25, detail: '$2.50 of $10.00' }] })
  expect(byName.ds).toMatchObject({ status: 'ok', balance: '$4.20 left', windows: [] })
  expect(byName.ms).toMatchObject({ status: 'ok', source: 'moonshot', balance: '¥88.50 left' })
  expect(byName.nokey).toMatchObject({ status: 'error', message: 'This profile has no API key saved.' })
  expect(seen.sort()).toEqual([
    'https://api.deepseek.com/user/balance Bearer ds-key',
    'https://api.moonshot.cn/v1/users/me/balance Bearer ms-key',
    'https://openrouter.ai/api/v1/key Bearer or-key',
  ])
})

test('built-in profiles the user never signed into are left out unless active', async () => {
  const { ConfigurationError } = await import('../src/core/errors.js')
  const signedOut = async () => { throw new ConfigurationError('codex_auth', 'No ChatGPT session found.') }
  const profiles = [profile('cc', 'claude-code'), profile('codex', 'openai-codex')]
  const hidden = await buildUsageReport(profiles, { fetchUsage: signedOut, hideWhenSignedOut: new Set(['cc', 'codex']) })
  expect(hidden).toEqual([])
  const active = await buildUsageReport([{ ...profiles[1]!, active: true }], { fetchUsage: signedOut, hideWhenSignedOut: new Set(['codex']) })
  expect(active).toMatchObject([{ profile: 'codex', status: 'error', message: 'No ChatGPT session found.' }])
  // A network failure is not "never signed in": it stays visible.
  const offline = await buildUsageReport([profiles[0]!], { fetchUsage: async () => { throw new Error('fetch failed') }, hideWhenSignedOut: new Set(['cc']) })
  expect(offline).toMatchObject([{ profile: 'cc', source: 'claude', status: 'error' }])
})

test('window labels keep the model a window covers and move quantities to the note', async () => {
  const { windowLabel, windowNote } = await import('../src/auth/usageView.js')
  expect([windowLabel({ label: 'weekly', usedPercent: 1, detail: 'opus' }), windowNote({ label: 'weekly', usedPercent: 1, detail: 'opus' })]).toEqual(['Weekly · opus', undefined])
  expect([windowLabel({ label: 'weekly', usedPercent: 1, detail: 'tokens_limit' }), windowNote({ label: 'weekly', usedPercent: 1, detail: 'tokens_limit' })]).toEqual(['Weekly', undefined])
  expect([windowLabel({ label: '5-hour', usedPercent: 1, detail: '1250000 remaining' }), windowNote({ label: '5-hour', usedPercent: 1, detail: '1250000 remaining' })]).toEqual(['5-hour', '1.3M left'])
  expect([windowLabel({ label: 'Key limit', usedPercent: 1, detail: '$2.50 of $10.00' }), windowNote({ label: 'Key limit', usedPercent: 1, detail: '$2.50 of $10.00' })]).toEqual(['Key limit', '$2.50 of $10.00'])
})

// Shapes captured from the live endpoints on 2026-09-24 (identifiers removed).
test('Kimi Code: the weekly total and 5-hour window are read the way Kimi’s own CLI reads them', async () => {
  const { fetchKimiUsage } = await import('../src/auth/usage.js')
  const body = {
    usage: { limit: '100', used: '86', remaining: '14', resetTime: '2026-09-26T19:46:33.714067Z' },
    limits: [{ window: { duration: 300, timeUnit: 'TIME_UNIT_MINUTE' }, detail: { limit: '100', remaining: '100', resetTime: '2026-09-24T13:46:33.714067Z' } }],
    usages: { limit_5h: { used_ratio: 0, reset_time: '2026-09-24T13:46:32Z' }, limit_7d: { used_ratio: 0, reset_time: '2026-09-26T19:46:32Z' } },
  }
  const report = await fetchKimiUsage('token', { fetchImplementation: async () => new Response(JSON.stringify(body)) })
  expect(report.windows).toEqual([
    { label: 'weekly', usedPercent: 86, resetsAt: Date.parse('2026-09-26T19:46:33.714Z'), detail: '14 remaining' },
    { label: '5-hour', usedPercent: 0, resetsAt: Date.parse('2026-09-24T13:46:33.714Z'), detail: '100 remaining' },
  ])
})

test('Z.ai: CREDIT_LIMIT windows are named by unit × number, not by the type', async () => {
  const { fetchZaiUsage } = await import('../src/auth/usage.js')
  const body = { code: 200, data: { level: 'max', limits: [
    { type: 'CREDIT_LIMIT', unit: 3, number: 5, usage: 28000, currentValue: 0, remaining: 28000, percentage: 0 },
    { type: 'CREDIT_LIMIT', unit: 6, number: 1, usage: 140000, currentValue: 0, remaining: 140000, percentage: 0, nextResetTime: 1790820130999 },
  ] } }
  const report = await fetchZaiUsage('key', { fetchImplementation: async () => new Response(JSON.stringify(body)) })
  expect(report.windows.map(window => [window.label, window.detail, window.resetsAt])).toEqual([
    ['5-hour', '28000 remaining', undefined],
    ['weekly', '140000 remaining', 1790820130999],
  ])
})

test('the fallback reads any quota payload: percentages, ratios, counts, durations, keyed windows', async () => {
  const { discoverUsageWindows, windowNameForSeconds } = await import('../src/auth/usage.js')
  expect(discoverUsageWindows({ data: { quotas: [
    { name: 'Requests', limit: 200, used: 50, reset_in: 600 },
    { window_seconds: 86_400, used_percent: 12 },
  ], five_hour: { utilization: 40, resets_at: '2026-09-24T12:00:00Z' }, balance: '0' } })).toEqual([
    { label: 'Requests', usedPercent: 25, resetAfterSeconds: 600, detail: '150 remaining' },
    { label: 'daily', usedPercent: 12 },
    { label: '5-hour', usedPercent: 40, resetsAt: Date.parse('2026-09-24T12:00:00Z') },
  ])
  expect(discoverUsageWindows({ credits: { balance: '0', has_credits: false } })).toEqual([])
  expect([18_000, 604_800, 86_400, 2_592_000, 7_200, 1_800].map(windowNameForSeconds)).toEqual(['5-hour', 'weekly', 'daily', 'monthly', '2-hour', '30-minute'])
})

test('Claude Code sign-in: keychain first, then ~/.claude/.credentials.json; expired stays visible', async () => {
  const { claudeCodeLogin } = await import('../src/auth/claudeCodeLogin.js')
  const stored = (token: string, expiresAt: number) => JSON.stringify({ claudeAiOauth: { accessToken: token, refreshToken: 'r', expiresAt, subscriptionType: 'max' } })
  const now = () => 1_000
  expect(await claudeCodeLogin({ platform: 'darwin', now, readKeychain: async () => stored('from-keychain', 2_000), readFileText: async () => stored('from-file', 2_000) }))
    .toEqual({ accessToken: 'from-keychain', expiresAt: 2_000, subscriptionType: 'max' })
  const paths: string[] = []
  expect((await claudeCodeLogin({ platform: 'linux', now, environment: { CLAUDE_CONFIG_DIR: '/cfg' }, readFileText: async path => { paths.push(path); return stored('from-file', 2_000) } })).accessToken).toBe('from-file')
  expect(paths).toEqual(['/cfg/.credentials.json'])
  const { ConfigurationError } = await import('../src/core/errors.js')
  const missing = await claudeCodeLogin({ platform: 'darwin', now, readKeychain: async () => undefined, readFileText: async () => { throw new Error('ENOENT') } }).catch(error => error)
  expect(missing).toBeInstanceOf(ConfigurationError)
  const expired = await claudeCodeLogin({ platform: 'darwin', now, readKeychain: async () => stored('old', 500), readFileText: async () => { throw new Error('ENOENT') } }).catch(error => error)
  expect(expired).not.toBeInstanceOf(ConfigurationError)
  expect(String(expired.message)).toContain('lapsed')
})

test('the cc profile reads its plan windows through Claude Code’s own sign-in and shows as "Claude Code"', async () => {
  const seen: string[] = []
  const report = await buildUsageReport([profile('cc', 'claude-code', { base_url: 'claude-code://local' } as never)], {
    hideWhenSignedOut: new Set(['cc']),
    claudeCodeLogin: async () => ({ accessToken: 'cc-token', subscriptionType: 'max' }),
    fetchImplementation: async (url: string, init?: RequestInit) => {
      seen.push(`${url} ${(init?.headers as Record<string, string>).Authorization}`)
      return new Response(JSON.stringify({ five_hour: { utilization: 22, resets_at: '2026-09-24T18:00:00Z' }, seven_day: { utilization: 41 } }))
    },
  })
  expect(report).toMatchObject([{ profile: 'cc', label: 'Claude Code', source: 'claude', status: 'ok', plan: 'max', windows: [{ label: '5-hour', usedPercent: 22 }, { label: 'weekly', usedPercent: 41 }] }])
  expect(seen).toEqual(['https://api.anthropic.com/api/oauth/usage Bearer cc-token'])
})

test('Claude Code sign-in: the freshest token wins across keychain and file, and an expiry says when', async () => {
  const { claudeCodeLogin } = await import('../src/auth/claudeCodeLogin.js')
  const stored = (token: string, expiresAt: number) => JSON.stringify({ claudeAiOauth: { accessToken: token, expiresAt } })
  const fresh = await claudeCodeLogin({ platform: 'darwin', now: () => 1_000, readKeychain: async () => stored('stale-keychain', 500), readFileText: async () => stored('fresh-file', 9_000) })
  expect(fresh.accessToken).toBe('fresh-file')
  const expired = await claudeCodeLogin({ platform: 'darwin', now: () => Date.parse('2026-09-24T12:00:00Z'), readKeychain: async () => stored('a', Date.parse('2026-09-23T19:25:00Z')), readFileText: async () => stored('b', Date.parse('2026-09-12T07:44:00Z')) }).then(() => new Error('resolved'), (error: Error) => error)
  // Never "sign-in expired": the person is signed in, only the access token lapsed.
  expect(expired.message).toMatch(/^You're signed in, but Claude Code's saved access token lapsed Sep 2[34]/)
  expect(expired.message).toContain('renews it on its next request')
})

test('a lapsed Claude Code token is read, never refreshed by Xerxes', async () => {
  const { claudeCodeLogin } = await import('../src/auth/claudeCodeLogin.js')
  const stored = (token: string, expiresAt: number) => JSON.stringify({ claudeAiOauth: { accessToken: token, expiresAt, refreshToken: 'rt' } })
  let reads = 0
  const lapsed = await claudeCodeLogin({ platform: 'darwin', now: () => 1_000, readKeychain: async () => { reads += 1; return stored('stale', 500) }, readFileText: async () => { throw new Error('ENOENT') } }).catch((error: Error) => error)
  expect(lapsed).toBeInstanceOf(Error)
  expect(reads).toBe(1)
  // Once Claude Code renews it on its own, the next read returns it.
  expect((await claudeCodeLogin({ platform: 'darwin', now: () => 1_000, readKeychain: async () => stored('renewed', 9_000), readFileText: async () => { throw new Error('ENOENT') } })).accessToken).toBe('renewed')
})
