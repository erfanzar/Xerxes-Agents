// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { EventEmitter } from 'node:events'
import { mkdtemp, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Readable, Writable } from 'node:stream'

import { expect, test } from 'bun:test'

import { ProviderAccountRegistry } from '../src/auth/providerAccountRegistry.js'
import { ProviderAccountDirectory } from '../src/desktop/main/providerAccounts.js'
import { remoteProfileReceipt, remoteProfileReceiverSource, spawnSecretWriter } from '../src/desktop/main/remoteCredentials.js'
import { ensureRemoteProviderClis, parseProviderCliProbe } from '../src/desktop/main/remoteProviderSetup.js'

const target = { alias: 'host', target: 'user@example.com', workspacePath: '/work/repo' }

test('missing Claude and Codex are installed and verified on the SSH host', async () => {
  const calls: string[][] = []
  const outputs = [
    'XERXES_PROVIDER_CLIS {"claude":false,"codex":false}',
    'XERXES_PROVIDER_CLIS {"claude":true,"codex":true}',
  ]
  const status = await ensureRemoteProviderClis(target, new AbortController().signal, async (_binary, args) => {
    calls.push(args)
    return calls.length === 1 ? outputs[0]! : outputs[1]!
  })
  expect(status).toEqual({ claude: true, codex: true })
  expect(calls).toHaveLength(4)
  expect(calls[1]!.join(' ')).toContain('@anthropic-ai/claude-code')
  expect(calls[2]!.join(' ')).toContain('@openai/codex')
  expect(parseProviderCliProbe(outputs[1]!)).toEqual(status)
})

test('credential payloads use stdin only and receiver output contains no secret', async () => {
  const secret = 'sk-test-never-in-argv'
  const profile = { name: 'work', base_url: 'https://provider.test/v1', api_key: secret, model: 'model-a' }
  let argv = '', stdin = ''
  const child = new EventEmitter() as EventEmitter & { stdin: Writable; stdout: Readable; kill: () => void }
  child.stdin = new Writable({
    write(chunk, _encoding, callback) {
      stdin += String(chunk)
      callback()
      queueMicrotask(() => {
        child.stdout.push(JSON.stringify({ ok: true, profile: 'work' }) + '\n')
        child.emit('close', 0)
      })
    },
  })
  child.stdout = new Readable({ read() {} })
  child.kill = () => {}
  const output = await spawnSecretWriter(target, profile, new AbortController().signal, 1000, (_binary, args) => {
    argv = args.join(' ')
    return child as never
  })
  expect(remoteProfileReceipt(output)).toBe('work')
  expect(argv).not.toContain(secret)
  expect(stdin).toContain(secret)
  expect(remoteProfileReceiverSource()).not.toContain(secret)
  expect(remoteProfileReceiverSource()).toContain('mode: 0o600')
})

test('an already cancelled transfer never starts a credential writer', async () => {
  const controller = new AbortController()
  controller.abort()
  let started = false
  await expect(spawnSecretWriter(target, { name: 'work', base_url: 'https://provider.test/v1', api_key: 'fixture-key', model: 'fixture' }, controller.signal, 1000, () => { started = true; throw new Error('Must not spawn') })).rejects.toThrow('cancelled')
  expect(started).toBe(false)
})

test('the remote receiver preserves other profiles and writes mode 0600 atomically', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-profile-receiver-'))
  try {
    const home = join(directory, '.xerxes')
    await Bun.write(join(home, 'profiles.json'), JSON.stringify({ active: 'other', profiles: { other: { base_url: 'https://other.test', api_key: 'other-secret', model: 'other-model' } } }))
    const source = remoteProfileReceiverSource()
    const child = Bun.spawn([process.execPath, '-e', source], {
      stdin: new TextEncoder().encode(JSON.stringify({ name: 'work', base_url: 'https://provider.test/v1', api_key: 'secret', model: 'model-a' })),
      stdout: 'pipe', stderr: 'pipe',
      env: { ...process.env, HOME: directory },
    })
    const output = await new Response(child.stdout).text()
    expect(await child.exited).toBe(0)
    expect(output).toContain('"profile":"work"')
    expect(output).not.toContain('secret')
    const document = await Bun.file(join(home, 'profiles.json')).json() as { active: string; profiles: Record<string, { api_key: string }> }
    expect(document.active).toBe('other')
    expect(document.profiles.other?.api_key).toBe('other-secret')
    expect(document.profiles.work?.api_key).toBe('secret')
    expect((await stat(join(home, 'profiles.json'))).mode & 0o777).toBe(0o600)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('the receiver stores an isolated OAuth account and linked profile without echoing tokens', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-oauth-receiver-'))
  try {
    const accountId = '11111111-1111-4111-8111-111111111111'
    const source = remoteProfileReceiverSource()
    const child = Bun.spawn([process.execPath, '-e', source], {
      stdin: new TextEncoder().encode(JSON.stringify({
        kind: 'claude', accountId, label: 'work',
        credential: { access: 'oauth-access', refresh: 'oauth-refresh', expires: 4102444800 },
        profile: { name: 'claude-work', base_url: 'https://api.anthropic.com', api_key: '', model: 'claude-sonnet-4-5', provider: 'anthropic', oauth_account_id: accountId },
      })),
      stdout: 'pipe', stderr: 'pipe', env: { ...process.env, HOME: directory },
    })
    const output = await new Response(child.stdout).text()
    expect(await child.exited).toBe(0)
    expect(output).toContain('"profile":"claude-work"')
    expect(output).not.toContain('oauth-access')
    const root = join(directory, '.xerxes')
    const credential = await Bun.file(join(root, 'provider-accounts', accountId, 'auth', 'anthropic-oauth.json')).json() as { access: string }
    expect(credential.access).toBe('oauth-access')
    const profiles = await Bun.file(join(root, 'profiles.json')).json() as { profiles: Record<string, { oauth_account_id?: string }> }
    expect(profiles.profiles['claude-work']?.oauth_account_id).toBe(accountId)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('synced OAuth accounts retain identity, readable quota, and remote settings', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-synced-account-'))
  try {
    const registry = new ProviderAccountRegistry(join(directory, 'source-accounts'), join(directory, 'source-profiles.json'))
    const remoteHome = join(directory, 'remote'), remoteRoot = join(remoteHome, '.xerxes')
    const remoteRegistry = new ProviderAccountRegistry(join(remoteRoot, 'provider-accounts'), join(remoteRoot, 'profiles.json'))
    await Bun.write(join(remoteRoot, 'profiles.json'), JSON.stringify({ active: 'existing', retained_setting: { enabled: true }, profiles: { existing: { name: 'existing', base_url: 'http://localhost:8080/v1', provider: 'openai', api_key: '', model: 'local-fixture', sampling: {}, data_trust: 'local', prompt_profile: 'lean' } } }))
    for (const kind of ['claude', 'claude-code', 'codex'] as const) {
      const account = await registry.create(kind, kind)
      const credential = kind === 'codex'
        ? { tokens: { access_token: 'fixture-codex', refresh_token: 'fixture-refresh', token_type: 'Bearer', expires_at: 4102444800, scopes: [] } }
        : { access: 'fixture-claude', refresh: 'fixture-refresh', expires: 4102444800 }
      const credentialPath = kind === 'codex' ? ['.codex', 'auth.json'] : ['auth', 'anthropic-oauth.json']
      await Bun.write(join(directory, 'source-accounts', account.id, ...credentialPath), JSON.stringify(credential))
      const payload = await registry.syncPayload(account.id)
      const child = Bun.spawn([process.execPath, '-e', remoteProfileReceiverSource()], {
        stdin: new TextEncoder().encode(JSON.stringify(payload)), stdout: 'pipe', stderr: 'pipe', env: { ...process.env, HOME: remoteHome },
      })
      const receipt = await new Response(child.stdout).text()
      expect(await child.exited).toBe(0)
      expect(remoteProfileReceipt(receipt)).toBe(account.profileName)
      expect((await remoteRegistry.list()).find(row => row.id === account.id)?.kind).toBe(kind)
      const report = await remoteRegistry.subscriptionUsage(account.id, { fetchImplementation: async () => new Response(JSON.stringify(kind === 'codex' ? { rate_limit: { primary_window: { used_percent: 25 } } } : { five_hour: { utilization: 25 } })) })
      expect(report.windows[0]?.usedPercent).toBe(25)
      if (kind !== 'codex') expect((await remoteRegistry.sessionOptions(account.id).anthropicOAuthSession.stored())?.access).toBe('fixture-claude')
    }
    const document = await Bun.file(join(remoteRoot, 'profiles.json')).json() as Record<string, unknown>
    expect(document.active).toBe('existing')
    expect(document.retained_setting).toEqual({ enabled: true })
    expect(document.profiles).toMatchObject({ existing: { data_trust: 'local', prompt_profile: 'lean' } })
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('per-host binding is reviewed, writes once, switches session and reads quota', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-provider-accounts-'))
  try {
    const file = join(directory, 'profiles.json')
    await writeFile(file, JSON.stringify({ active: 'work', profiles: { work: { base_url: 'https://provider.test/v1', api_key: 'secret', model: 'model-a', provider: 'openai' } } }))
    const writes: unknown[] = [], calls: { method: string; params: Record<string, unknown> }[] = []
    let now = 1000
    const accounts = new ProviderAccountDirectory(target, file, async (method, params) => {
      calls.push({ method, params })
      return { ok: true }
    }, async profile => { writes.push(profile); return profile.name }, () => now)
    const review = await accounts.reviewBind('work')
    const bound = await accounts.confirmBind(review.review, true, new AbortController().signal)
    expect(bound.profile).toBe('work')
    expect(writes).toHaveLength(1)
    expect(calls.map(call => call.method)).toEqual(['runtime.reload'])
    expect((await accounts.list()).profiles).toEqual([{ name: 'work', model: 'model-a', provider: 'openai', bound: true }])
    await accounts.activate('session-1')
    expect(calls.at(-1)).toEqual({ method: 'provider_select', params: { name: 'work', session_key: 'session-1' } })
    await accounts.quota()
    expect(calls.at(-1)).toEqual({ method: 'usage.report', params: { refresh: false } })
    const second = await accounts.reviewBind('work')
    now = second.expires_at + 1
    await expect(accounts.confirmBind(second.review, true, new AbortController().signal)).rejects.toThrow('expired')
    expect(writes).toHaveLength(1)
  } finally { await rm(directory, { recursive: true, force: true }) }
})
