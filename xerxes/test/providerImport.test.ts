// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { importProfiles, ProfileStore } from '../src/bridge/profiles.js'
import { copyProfilesToRemote, localKeyedProfiles } from '../src/desktop/main/profileSync.js'

async function inTemporaryHome(body: (root: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-import-'))
  try { await body(root) } finally { await rm(root, { recursive: true, force: true }) }
}

const mac = {
  active: 'openrouter',
  profiles: {
    openrouter: { name: 'openrouter', provider: 'openrouter', base_url: 'https://openrouter.ai/api/v1', api_key: 'or-key', model: 'deepseek/deepseek-v4.1-flash', model_capabilities: {}, sampling: {} },
    zai: { name: 'zai', provider: 'zhipu', base_url: 'https://api.z.ai/api/coding/paas/v4', api_key: 'zai-key', model: 'glm-5.3', model_capabilities: {}, sampling: { temperature: 0.2 } },
    'claude-code': { name: 'claude-code', provider: 'claude-code', base_url: 'claude-code://local', api_key: '', model: 'claude-code/opus', model_capabilities: {}, sampling: {} },
  },
}

test('the Mac\'s keyed profiles reach the host; sign-in profiles and the host selection stay put', async () => {
  await inTemporaryHome(async root => {
    const macFile = join(root, 'mac-profiles.json')
    await writeFile(macFile, JSON.stringify(mac))
    const hostFile = join(root, 'host-profiles.json')
    await writeFile(hostFile, JSON.stringify({ active: 'zai-glm', profiles: {
      'zai-glm': { name: 'zai-glm', provider: 'zhipu', base_url: 'https://api.z.ai/api/paas/v4', api_key: 'host-key', model: 'glm-5.3', model_capabilities: {}, sampling: {} },
    } }))
    const host = new ProfileStore(hostFile)
    const sent: unknown[] = []
    const result = await copyProfilesToRemote(async (method, params) => {
      expect(method).toBe('provider.import')
      sent.push(...(params.profiles as unknown[]))
      const outcome = importProfiles(host, params.profiles)
      return { ok: true, imported: [...outcome.imported] }
    }, macFile)

    expect(result).toEqual({ imported: ['openrouter', 'zai'] })
    // Profiles without a key never leave the Mac.
    expect(sent.map(profile => (profile as { name: string }).name)).toEqual(['openrouter', 'zai'])
    expect(host.active()?.name).toBe('zai-glm')
    expect(host.get('openrouter')?.api_key).toBe('or-key')
    expect(host.get('zai')?.sampling).toEqual({ temperature: 0.2 })
    expect(host.get('zai-glm')?.api_key).toBe('host-key')
    expect(JSON.parse(await readFile(hostFile, 'utf8')).active).toBe('zai-glm')
  })
})

test('a same-named host profile keeps its own endpoint, key, model and tuned limits on every connect', async () => {
  await inTemporaryHome(async root => {
    const hostFile = join(root, 'host-profiles.json')
    const hostOwn = { name: 'openrouter', provider: 'openrouter', base_url: 'https://openrouter.ai/api/v1/host', api_key: 'host-key', model: 'host/model',
      model_capabilities: { 'host/model': { context_limit: 64000 } }, model_overrides: { 'host/model': { max_output_tokens: 4096 } }, sampling: { temperature: 0.7 } }
    await writeFile(hostFile, JSON.stringify({ active: 'openrouter', profiles: { openrouter: hostOwn } }))
    const host = new ProfileStore(hostFile)
    const before = host.get('openrouter')
    expect(before).toMatchObject({ base_url: 'https://openrouter.ai/api/v1/host', api_key: 'host-key', model: 'host/model', sampling: { temperature: 0.7 },
      model_capabilities: { 'host/model': { context_limit: 64000 } }, model_overrides: { 'host/model': { max_output_tokens: 4096 } } })
    for (let connect = 0; connect < 2; connect++) {
      const outcome = importProfiles(host, Object.values(mac.profiles))
      expect(outcome.imported).toEqual(connect === 0 ? ['zai'] : [])
      expect(outcome.skipped).toContainEqual({ name: 'openrouter', reason: 'already on host' })
    }
    expect(host.get('openrouter')).toEqual(before)
    expect(host.get('zai')?.api_key).toBe('zai-key')
  })
})

test('the host rejects malformed and sign-in profiles with a reason', async () => {
  await inTemporaryHome(async root => {
    const host = new ProfileStore(join(root, 'profiles.json'))
    const outcome = importProfiles(host, [
      { name: '../escape', provider: 'openai', base_url: 'https://example.test', api_key: 'k', model: 'm' },
      { name: 'codex', provider: 'openai-codex', base_url: 'https://chatgpt.com/backend-api/codex', api_key: 'k', model: 'm' },
      { name: 'keyless', provider: 'openai', base_url: 'https://example.test', api_key: '', model: 'm' },
      { name: 'local-file', provider: 'openai', base_url: 'file:///etc', api_key: 'k', model: 'm' },
    ])
    expect(outcome.imported).toEqual([])
    expect(outcome.skipped.map(item => item.reason)).toEqual(['invalid profile name', 'signs in on each machine', 'no key to copy', 'invalid base_url'])
    expect(importProfiles(host, 'nope').skipped).toEqual([{ name: '', reason: 'profiles must be a list' }])
  })
})

test('an older host runtime without the method leaves the connection alone', async () => {
  await inTemporaryHome(async root => {
    const macFile = join(root, 'profiles.json')
    await writeFile(macFile, JSON.stringify(mac))
    const result = await copyProfilesToRemote(async () => { throw new Error('Unknown method: provider.import') }, macFile)
    expect(result).toEqual({ imported: [], error: 'Unknown method: provider.import' })
    expect(await localKeyedProfiles(join(root, 'missing.json'))).toEqual([])
  })
})
