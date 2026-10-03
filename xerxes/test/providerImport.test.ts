// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { connect } from 'node:net'

import { importProfiles, ProfileStore } from '../src/bridge/profiles.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { copyProfilesToRemote, localKeyedProfiles, watchProfiles } from '../src/desktop/main/profileSync.js'
import { rename } from 'node:fs/promises'

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
      expect(outcome.skipped).toContainEqual({ name: 'openrouter', reason: 'host uses a different endpoint' })
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

test('a key rotated on the Mac replaces the copy on the host; its model and tuned limits stay', async () => {
  await inTemporaryHome(async root => {
    const host = new ProfileStore(join(root, 'host-profiles.json'))
    expect(importProfiles(host, Object.values(mac.profiles)).imported).toEqual(['openrouter', 'zai'])
    host.updateModelCapabilities('zai', 'glm-5.3', { contextLimit: 200_000 })
    const rotated = Object.values(mac.profiles).map(profile => profile.name === 'zai' ? { ...profile, api_key: 'zai-new-key', model: 'mac/other' } : profile)
    const outcome = importProfiles(host, rotated)
    expect(outcome).toMatchObject({ imported: [], updated: ['zai'] })
    expect(outcome.skipped).toContainEqual({ name: 'openrouter', reason: 'already on host' })
    expect(host.get('zai')).toMatchObject({ api_key: 'zai-new-key', model: 'glm-5.3', sampling: { temperature: 0.2 } })
    expect(host.get('zai')?.model_overrides).toMatchObject({ 'glm-5.3': { context_limit: 200_000 } })
    expect(importProfiles(host, rotated).updated).toEqual([])
  })
})

/** One request at a time over the daemon socket; events are skipped. */
async function daemonCall(socketPath: string): Promise<{ call: (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>; close: () => void }> {
  const socket = connect({ path: socketPath })
  await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) })
  socket.setEncoding('utf8')
  let buffer = '', id = 0
  const pending = new Map<number, (result: Record<string, unknown>) => void>()
  socket.on('data', chunk => {
    buffer += String(chunk)
    for (let newline = buffer.indexOf('\n'); newline >= 0; newline = buffer.indexOf('\n')) {
      const frame = JSON.parse(buffer.slice(0, newline)) as { id?: number; result?: Record<string, unknown>; error?: unknown }
      buffer = buffer.slice(newline + 1)
      if (typeof frame.id === 'number') pending.get(frame.id)?.(frame.result ?? { error: frame.error })
    }
  })
  return {
    call: (method, params) => new Promise(resolve => { id += 1; pending.set(id, resolve); socket.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n') }),
    close: () => socket.destroy(),
  }
}

async function withHostDaemon(hostDocument: unknown, body: (call: (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>, host: ProfileStore) => Promise<void>): Promise<void> {
  await inTemporaryHome(async root => {
    const hostFile = join(root, 'profiles.json')
    if (hostDocument) await writeFile(hostFile, JSON.stringify(hostDocument))
    const profileStore = new ProfileStore(hostFile)
    const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
    const socketPath = join(root, 'daemon.sock')
    const server = new DaemonServer({ profileStore, runtime, socketPath })
    await server.start()
    const client = await daemonCall(socketPath)
    try {
      await client.call('initialize', { session_key: 'ssh-task' })
      await body(client.call, profileStore)
    } finally { client.close(); await server.stop() }
  })
}

test('a fresh SSH host starts on the Mac\'s active profile, so its model list is not empty', async () => {
  await withHostDaemon(undefined, async (call, host) => {
    expect(host.chosenActive()).toBeUndefined()
    const result = await call('provider.import', { profiles: Object.values(mac.profiles), active: 'zai' })
    expect(result).toMatchObject({ ok: true, imported: ['openrouter', 'zai'], selected: 'zai' })
    expect(host.chosenActive()).toBe('zai')
    expect((await call('session.status', { session_key: 'ssh-task' })).session).toMatchObject({ model: 'glm-5.3' })
  })
})

test('when the Mac\'s active profile is a sign-in one, a fresh host starts on the first copied profile', async () => {
  await withHostDaemon(undefined, async (call, host) => {
    const result = await call('provider.import', { profiles: Object.values(mac.profiles), active: 'codex' })
    expect(result).toMatchObject({ ok: true, selected: 'openrouter' })
    expect(host.chosenActive()).toBe('openrouter')
  })
})

test('a host that already chose a provider keeps it', async () => {
  const own = { name: 'own', provider: 'openai', base_url: 'https://api.example.test/v1', api_key: 'host-key', model: 'own-model', model_capabilities: {}, sampling: {} }
  await withHostDaemon({ active: 'own', profiles: { own } }, async (call, host) => {
    const result = await call('provider.import', { profiles: Object.values(mac.profiles), active: 'zai' })
    expect(result).toMatchObject({ ok: true, imported: ['openrouter', 'zai'] })
    expect(result.selected).toBeUndefined()
    expect(host.chosenActive()).toBe('own')
  })
})

test('a provider saved on the Mac (written by rename) triggers one sync; other files do not', async () => {
  await inTemporaryHome(async root => {
    const file = join(root, 'profiles.json')
    await writeFile(file, JSON.stringify(mac))
    let calls = 0
    const stop = watchProfiles(file, () => { calls++ }, 50)
    try {
      // macOS can report the setup write above late; start counting after it settles.
      await Bun.sleep(400)
      calls = 0
      await writeFile(join(root, 'other.json'), '{}')
      await Bun.sleep(200)
      expect(calls).toBe(0)
      // ProfileStore saves through a temporary file and a rename, twice here.
      for (let save = 0; save < 2; save++) {
        await writeFile(join(root, 'profiles.json.tmp'), JSON.stringify({ ...mac, active: 'zai' }))
        await rename(join(root, 'profiles.json.tmp'), file)
      }
      for (let i = 0; i < 50 && calls === 0; i++) await Bun.sleep(20)
      await Bun.sleep(150)
      expect(calls).toBe(1)
    } finally { stop() }
  })
})

test('the host marks sign-in profiles, so an SSH window runs them on the Mac', async () => {
  await withHostDaemon(undefined, async call => {
    await call('provider.import', { profiles: Object.values(mac.profiles), active: 'zai' })
    const listed = (await call('provider_list', {})).profiles as Array<{ name: string; provider: string; signs_in: boolean }>
    expect(listed.find(row => row.provider === 'claude-code')?.signs_in).toBe(true)
    expect(listed.find(row => row.provider === 'openai-codex')?.signs_in).toBe(true)
    expect(listed.find(row => row.name === 'zai')?.signs_in).toBe(false)
  })
})
