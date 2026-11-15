// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ProfileStore } from '../src/bridge/profiles.js'
import { createAccountFailover } from '../src/daemon/accountFailoverWiring.js'
import { chooseFailover, parseFailoverSettings } from '../src/runtime/accountFailover.js'

const CODEX = 'https://chatgpt.com/backend-api/codex'

async function fixture(run: (store: ProfileStore) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-failover-'))
  try {
    const store = new ProfileStore(join(directory, 'profiles.json'))
    for (const name of ['codex-Omid', 'codex-Erfan']) store.save({ name, apiKey: '', model: 'gpt-6.1-sol', provider: 'openai-codex', baseUrl: CODEX, setActive: false })
    store.save({ name: 'claude-Personal', apiKey: '', model: 'claude-code/sonnet', provider: 'claude-code', baseUrl: 'https://api.anthropic.com', setActive: false })
    await run(store)
  } finally { await rm(directory, { recursive: true, force: true }) }
}

const usageOf = (peaks: Record<string, number | undefined>) => async (profiles: readonly { name: string }[]) =>
  profiles.map(profile => peaks[profile.name] === undefined
    ? { profile: profile.name, status: 'error', windows: [] }
    : { profile: profile.name, status: 'ok', windows: [{ usedPercent: peaks[profile.name]! }] })

test('chooseFailover picks the same-family sibling with the most room, never across families or to unknown usage', () => {
  const current = { name: 'codex-Omid', provider: 'openai-codex' }
  expect(chooseFailover(current, [
    { name: 'codex-Omid', provider: 'openai-codex', peakUsed: 100 },
    { name: 'codex-Erfan', provider: 'openai-codex', peakUsed: 30 },
    { name: 'codex-Third', provider: 'openai-codex', peakUsed: 10 },
    { name: 'claude-Personal', provider: 'claude-code', peakUsed: 0 },
  ], 95)).toEqual({ to: 'codex-Third', peakUsed: 10 })
  expect(chooseFailover(current, [{ name: 'codex-Erfan', provider: 'openai-codex', peakUsed: undefined }], 95)).toBeUndefined()
  expect(chooseFailover(current, [{ name: 'codex-Erfan', provider: 'openai-codex', peakUsed: 97 }], 95)).toBeUndefined()
  expect(chooseFailover({ name: 'kimi', provider: 'kimi-code' }, [{ name: 'other', provider: 'kimi-code', peakUsed: 0 }], 95)).toBeUndefined()
})

test('settings default on at 95% and parse off and a custom threshold', () => {
  expect(parseFailoverSettings({})).toEqual({ enabled: true, threshold: 95 })
  expect(parseFailoverSettings({ account_failover: 'off', account_failover_threshold: 80 })).toEqual({ enabled: false, threshold: 80 })
  expect(parseFailoverSettings({ account_failover_threshold: 500 }).threshold).toBe(95)
})

test('before a turn: switches only when the current account is near its limit', () => fixture(async store => {
  const near = createAccountFailover({ profiles: store, settings: () => ({ enabled: true, threshold: 95 }), usage: usageOf({ 'codex-Omid': 97, 'codex-Erfan': 4 }) })
  const session = { metadata: { provider_profile: 'codex-Omid' } as Record<string, unknown> }
  const switched = await near(session, 'gpt-6.1-sol', 'preflight')
  expect(switched).toMatchObject({ from: 'codex-Omid', to: 'codex-Erfan' })
  expect(session.metadata.provider_profile).toBe('codex-Erfan')

  const roomy = createAccountFailover({ profiles: store, settings: () => ({ enabled: true, threshold: 95 }), usage: usageOf({ 'codex-Omid': 40, 'codex-Erfan': 4 }) })
  const stay = { metadata: { provider_profile: 'codex-Omid' } as Record<string, unknown> }
  expect(await roomy(stay, 'gpt-6.1-sol', 'preflight')).toBeUndefined()
  expect(stay.metadata.provider_profile).toBe('codex-Omid')
}))

test('after a quota error: switches with a fresh read; no sibling with room or disabled means no switch', () => fixture(async store => {
  let refreshed = false
  const failover = createAccountFailover({ profiles: store, settings: () => ({ enabled: true, threshold: 95 }),
    usage: async (profiles, refresh) => { refreshed = refresh; return usageOf({ 'codex-Omid': 60, 'codex-Erfan': 10 })(profiles) } })
  const session = { metadata: { provider_profile: 'codex-Omid' } as Record<string, unknown> }
  expect(await failover(session, 'gpt-6.1-sol', 'quota')).toMatchObject({ to: 'codex-Erfan' })
  expect(refreshed).toBe(true)

  const full = createAccountFailover({ profiles: store, settings: () => ({ enabled: true, threshold: 95 }), usage: usageOf({ 'codex-Omid': 100, 'codex-Erfan': 99 }) })
  const stuck = { metadata: { provider_profile: 'codex-Omid' } as Record<string, unknown> }
  expect(await full(stuck, 'gpt-6.1-sol', 'quota')).toBeUndefined()
  expect(stuck.metadata.provider_profile).toBe('codex-Omid')

  const off = createAccountFailover({ profiles: store, settings: () => ({ enabled: false, threshold: 95 }), usage: usageOf({ 'codex-Omid': 100, 'codex-Erfan': 0 }) })
  expect(await off({ metadata: { provider_profile: 'codex-Omid' } }, 'gpt-6.1-sol', 'quota')).toBeUndefined()
}))

test('a Claude account never fails over to a Codex account', () => fixture(async store => {
  const failover = createAccountFailover({ profiles: store, settings: () => ({ enabled: true, threshold: 95 }), usage: usageOf({ 'claude-Personal': 100, 'codex-Erfan': 0 }) })
  const session = { metadata: { provider_profile: 'claude-Personal' } as Record<string, unknown> }
  expect(await failover(session, 'claude-code/sonnet', 'quota')).toBeUndefined()
  expect(session.metadata.provider_profile).toBe('claude-Personal')
}))
