// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { chmod, mkdir, mkdtemp, readFile, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { CodexSession } from '../src/auth/codexAuth.js'
import { CredentialStorage } from '../src/auth/storage.js'
import { ProfileStore } from '../src/bridge/profiles.js'
import { OAuthToken } from '../src/mcp/oauth.js'
import { ConfigurationError } from '../src/core/errors.js'
import { createSignInReadinessProbe, type SignInReadinessState } from '../src/daemon/providerReadiness.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { DaemonRpc } from '../src/desktop/main/daemon.js'
import { DesktopProviderForwarding } from '../src/desktop/main/providerForwarding.js'
import { providerRelayView, runsOnThisComputer } from '../src/desktop/renderer/store.js'

const installed = () => '/usr/local/bin/claude'

test('the probe reports whether this machine\'s own Claude Code sign-in works, and nothing secret', async () => {
  const cases: Array<[() => string | undefined, () => Promise<unknown>, SignInReadinessState]> = [
    [() => undefined, async () => ({ accessToken: 'never-read' }), 'not_installed'],
    [installed, async () => ({ accessToken: 'synthetic-token' }), 'ready'],
    [installed, async () => { throw new ConfigurationError('claude_code', 'No Claude Code sign-in found.') }, 'not_signed_in'],
    // Only the short-lived token lapsed: Claude Code renews it itself.
    [installed, async () => { throw new Error("You're signed in, but Claude Code's saved access token lapsed Oct 23.") }, 'ready'],
    [installed, async () => { throw new Error('macOS keychain access was denied') }, 'unknown'],
  ]
  for (const [executable, login, state] of cases) {
    const probe = createSignInReadinessProbe({ environment: {}, executable, login })
    const result = await probe('claude-code')
    expect(result.state).toBe(state)
    expect(JSON.stringify(result)).not.toContain('synthetic-token')
    expect(JSON.stringify(result)).not.toContain('/usr/local/bin')
  }
  expect((await createSignInReadinessProbe({ environment: {}, executable: installed, login: async () => ({}) })('unsupported')).state).toBe('unknown')
})

test('Codex readiness accepts native or CLI OAuth without requiring the executable or exposing credentials', async () => {
  const valid = new OAuthToken({ accessToken: 'synthetic-codex-access', expiresAt: 2000 })
  const expired = new OAuthToken({ accessToken: 'synthetic-expired-access', expiresAt: 900 })
  const renewable = new OAuthToken({ accessToken: 'synthetic-expired-access', refreshToken: 'synthetic-codex-refresh', expiresAt: 900 })
  const cases: Array<{ installed: boolean; stored?: OAuthToken; cli?: OAuthToken; state: SignInReadinessState }> = [
    { installed: false, state: 'not_installed' },
    { installed: false, cli: valid, state: 'ready' },
    { installed: false, stored: valid, state: 'ready' },
    { installed: true, state: 'not_signed_in' },
    { installed: true, stored: valid, state: 'ready' },
    { installed: true, cli: valid, state: 'ready' },
    { installed: true, stored: renewable, state: 'ready' },
    { installed: true, cli: renewable, state: 'ready' },
    { installed: true, stored: expired, state: 'not_signed_in' },
    { installed: true, cli: expired, state: 'not_signed_in' },
    { installed: true, cli: new OAuthToken({ accessToken: 'synthetic-codex-access', expiresAt: 1100 }), state: 'not_signed_in' },
  ]
  for (const value of cases) {
    let reads = 0
    const probe = createSignInReadinessProbe({ environment: {}, now: () => 1_000_000,
      codexExecutable: () => value.installed ? '/host/bin/codex' : undefined,
      codexSession: { stored: async () => { reads++; return value.stored } },
      codexCliTokens: async () => { reads++; return value.cli },
      login: async () => { throw new Error('Claude sign-in must not be read') },
    })
    const result = await probe(' OpenAI-Codex ')
    expect(result.state).toBe(value.state)
    expect(reads).toBe(2)
    for (const secret of ['synthetic-codex-access', 'synthetic-codex-refresh', '/host/bin/codex']) {
      expect(JSON.stringify(result)).not.toContain(secret)
    }
  }
  const failed = createSignInReadinessProbe({ environment: {}, codexExecutable: installed,
    codexSession: { stored: async () => { throw new Error('synthetic-codex-secret at /host/private') } }, codexCliTokens: async () => undefined,
  })
  expect(await failed('openai-codex')).toEqual({ state: 'unknown', reason: 'Could not read Codex\'s sign-in on this machine.' })
})

test('Codex discovers a CLI on the host PATH and reads auth.json without invoking it or adopting tokens', async () => {
  const dir = await realpath(await mkdtemp(join(tmpdir(), 'xcodex-ready-')))
  try {
    const bin = join(dir, 'bin'), cliHome = join(dir, 'codex')
    await mkdir(bin); await mkdir(cliHome)
    const executable = join(bin, 'codex')
    // Finding the executable must not run it (or any sign-in command).
    await Bun.write(executable, '#!/bin/sh\nexit 99\n'); await chmod(executable, 0o700)
    const environment = { HOME: dir, PATH: bin, CODEX_HOME: cliHome }
    let fetches = 0
    const session = new CodexSession({ environment, homeDirectory: dir, storage: new CredentialStorage(join(dir, 'credentials')),
      fetchImplementation: async () => { fetches++; throw new Error('Readiness must not call the provider') },
    })
    const probe = createSignInReadinessProbe({ environment, homeDirectory: dir, codexSession: session, cacheMs: 0 })
    expect((await probe('openai-codex')).state).toBe('not_signed_in')
    const auth = JSON.stringify({ tokens: { access_token: 'synthetic-cli-access', refresh_token: 'synthetic-cli-refresh' } })
    const authPath = join(cliHome, 'auth.json')
    await Bun.write(authPath, auth)
    expect(await probe('openai-codex')).toEqual({ state: 'ready' })
    expect(await readFile(authPath, 'utf8')).toBe(auth)
    expect(await session.stored()).toBeUndefined()
    // API-key-only, malformed and expired/nonrenewable CLI files are not OAuth readiness.
    const expired = `e30.${Buffer.from(JSON.stringify({ exp: 1 })).toString('base64url')}.signature`
    for (const text of ['{broken', JSON.stringify({ OPENAI_API_KEY: 'synthetic-api-key' }), JSON.stringify({ tokens: { access_token: expired } })]) {
      await Bun.write(authPath, text)
      expect((await probe('openai-codex')).state).toBe('not_signed_in')
    }
    await session.store(new OAuthToken({ accessToken: 'synthetic-stored-access', refreshToken: 'synthetic-stored-refresh', expiresAt: 1 }))
    const storedPath = join(dir, 'credentials', 'openai-codex.json')
    const before = await readFile(storedPath, 'utf8')
    expect(await probe('openai-codex')).toEqual({ state: 'ready' })
    expect(await readFile(storedPath, 'utf8')).toBe(before)
    expect(fetches).toBe(0)
  } finally { await rm(dir, { recursive: true, force: true }) }
})

test('Codex readiness coalesces concurrent reads and expires its cache independently of Claude', async () => {
  let storedReads = 0, cliReads = 0, claudeReads = 0, clock = 0
  const probe = createSignInReadinessProbe({ environment: {}, executable: installed,
    login: async () => { claudeReads++; return {} }, codexExecutable: installed,
    codexSession: { stored: async () => { storedReads++; return undefined } },
    codexCliTokens: async () => { cliReads++; return new OAuthToken({ accessToken: 'synthetic' }) },
    now: () => clock, cacheMs: 1000,
  })
  await Promise.all([probe('openai-codex'), probe('OpenAI-Codex'), probe('claude-code')])
  expect([storedReads, cliReads, claudeReads]).toEqual([1, 1, 1])
  clock = 2000
  await probe('openai-codex')
  expect([storedReads, cliReads, claudeReads]).toEqual([2, 2, 1])
})

test('one answer is cached, so a keychain prompt is not raised on every list', async () => {
  let reads = 0, clock = 0
  const probe = createSignInReadinessProbe({ environment: {}, executable: installed, login: async () => { reads++; return {} }, now: () => clock, cacheMs: 1000 })
  await probe('claude-code'); await probe('Claude-Code')
  expect(reads).toBe(1)
  clock = 2000
  await probe('claude-code')
  expect(reads).toBe(2)
})

test('provider_list carries readiness only when asked and only for sign-in profiles', async () => {
  const dir = await realpath(await mkdtemp(join(tmpdir(), 'xpr-')))
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: dir, sessionDirectory: join(dir, 'sessions') })
  const profiles = new ProfileStore(join(dir, 'profiles.json'))
  profiles.save({ name: 'keyed', provider: 'openai', model: 'gpt-4o', apiKey: 'synthetic-key', baseUrl: 'https://fixture.invalid' })
  const server = new DaemonServer({ runtime, socketPath: join(dir, 'd.sock'), projectDirectory: dir, profileStore: profiles,
    providerReadiness: createSignInReadinessProbe({ environment: {}, executable: installed, login: async () => ({}),
      codexExecutable: installed, codexSession: { stored: async () => undefined },
      codexCliTokens: async () => new OAuthToken({ accessToken: 'synthetic-codex-access', refreshToken: 'synthetic-codex-refresh' }),
    }) })
  await server.start()
  const rpc = new DaemonRpc({ socketPath: join(dir, 'd.sock'), projectDir: dir, deadlineMs: 3000 })
  try {
    await rpc.call('initialize', {})
    const plain = await rpc.call('provider_list', {})
    expect(plain.readiness_supported).toBeUndefined()
    const rows = (await rpc.call('provider_list', { with_readiness: true }))
    expect(rows.readiness_supported).toBe(true)
    const list = rows.profiles as Array<Record<string, unknown>>
    expect(list.find(row => row.name === 'claude-code')).toMatchObject({ signs_in: true, ready: true, ready_state: 'ready' })
    expect(list.find(row => row.name === 'codex')).toMatchObject({ provider: 'openai-codex', signs_in: true, ready: true, ready_state: 'ready' })
    expect(list.find(row => row.name === 'keyed')?.ready).toBeUndefined()
    for (const secret of ['synthetic-key', 'synthetic-codex-access', 'synthetic-codex-refresh']) expect(JSON.stringify(rows)).not.toContain(secret)
  } finally { rpc.dispose(); await server.stop(); await runtime.shutdown(); await rm(dir, { recursive: true, force: true }) }
})

/** A desktop window over a fake host whose readiness answer the test controls. */
function forwardingWith(host: { readiness?: boolean; ready?: boolean; setModel?: boolean; relayed?: boolean; oldHost?: boolean }, name = 'claude-code') {
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  let grants = 0, bound = host.relayed === true
  const model = name === 'codex' ? 'gpt-6.1-sol' : 'claude-code/opus'
  const profile = { name, model, supported: true, credential_source: 'local provider authentication or environment', output_limit_mode: 'request-bound' }
  const forwarding = new DesktopProviderForwarding(async method => {
    if (method === 'provider.relay.inventory') return { ok: true, profiles: [profile] }
    if (method === 'provider.relay.authorize') return { ok: true, grant: { id: String(++grants).repeat(32) } }
    return { ok: true }
  }, async (method, params) => {
    calls.push({ method, params })
    if (method === 'provider_list') return host.readiness === false ? { ok: true, profiles: [{ name, signs_in: true }] }
      : { ok: true, readiness_supported: true, profiles: [{ name, provider: name === 'codex' ? 'openai-codex' : name, model, signs_in: true, ready: host.ready === true }] }
    if (method === 'set_model') {
      if (host.setModel === false) return { ok: false, error: 'refused' }
      // An older host ignores host_credentials and keeps the relayed route.
      if (params.host_credentials === true && host.oldHost !== true) { bound = false; return { ok: true, host_credentials: true } }
      return { ok: true }
    }
    if (method === 'provider.remote.bind') { bound = true; return { ok: true, binding: { workspace: '/project', profile: params.profile, model: params.model, binding: 'a'.repeat(32) } } }
    return { ok: true, provider_binding_session_guard_supported: true, provider_binding_busy: false, provider_binding_lost: false,
      session: { key: 'task', cwd: '/project', status: 'idle', turn_count: host.relayed ? 12 : 0,
        ...(bound ? { local_provider_label: `Local provider: ${name}`, local_provider_route: { profile: name, model } } : {}) } }
  }, 'tpu-host', '/project', async () => name)
  return { forwarding, calls, grants: () => grants }
}

test('a new SSH task runs on the host\'s own working Claude Code instead of relaying through this computer', async () => {
  const { forwarding, calls, grants } = forwardingWith({ ready: true })
  try {
    expect(await forwarding.follow()).toEqual({ status: 'skipped', reason: 'the host has its own claude-code sign-in' })
    expect(grants()).toBe(0)
    expect(calls.some(call => call.method === 'provider.remote.bind')).toBe(false)
    expect(calls.find(call => call.method === 'set_model')?.params).toEqual({ model: 'claude-code/opus', provider_profile: 'claude-code', session_key: 'task' })
    const shown = await forwarding.current()
    expect(shown).toMatchObject({ bound: false, destination: 'tpu-host', onHost: { profile: 'claude-code', model: 'claude-code/opus' } })
    const view = providerRelayView(shown)!
    // The badge rendering for this view is covered in desktopProviderRelayBadge.test.tsx.
    expect(view).toMatchObject({ bound: false, destination: 'tpu-host', onHost: { profile: 'claude-code', model: 'claude-code/opus' } })
  } finally { await forwarding.disconnect() }
})

test('a host without a working sign-in, an older host, or a refused switch still relays through this computer', async () => {
  for (const host of [{ ready: false }, { readiness: false }, { ready: true, setModel: false }]) {
    const { forwarding, grants } = forwardingWith(host)
    try {
      expect(await forwarding.follow()).toEqual({ status: 'bound' })
      expect(grants()).toBe(1)
      expect((await forwarding.current()).bound).toBe(true)
    } finally { await forwarding.disconnect() }
  }
})

test('a task already relayed through this computer moves to the host\'s own working sign-in once idle', async () => {
  const { forwarding, calls, grants } = forwardingWith({ ready: true, relayed: true })
  try {
    expect(await forwarding.follow()).toEqual({ status: 'skipped', reason: 'the host has its own claude-code sign-in' })
    expect(grants()).toBe(0)
    expect(calls.find(call => call.method === 'set_model')?.params).toEqual({ model: 'claude-code/opus', provider_profile: 'claude-code', session_key: 'task', host_credentials: true })
    expect(await forwarding.current()).toMatchObject({ bound: false, onHost: { profile: 'claude-code', model: 'claude-code/opus' } })
  } finally { await forwarding.disconnect() }
})

test('a relayed task stays on this computer when the host cannot run it, or is too old to confirm the switch', async () => {
  for (const host of [{ ready: false, relayed: true }, { ready: true, relayed: true, oldHost: true }]) {
    const { forwarding, grants } = forwardingWith(host)
    try {
      expect(await forwarding.follow()).toEqual({ status: 'bound' })
      expect(grants()).toBe(1)
      expect((await forwarding.current()).bound).toBe(true)
    } finally { await forwarding.disconnect() }
  }
})

test('an explicit "use via Mac" is kept; "run on the server" moves it back', async () => {
  const { forwarding, calls } = forwardingWith({ ready: true })
  try {
    await forwarding.useLocal('claude-code')
    expect((await forwarding.current()).bound).toBe(true)
    expect(await forwarding.follow()).toEqual({ status: 'skipped', reason: 'already following' })
    expect(calls.some(call => call.method === 'set_model' && call.params.host_credentials === true)).toBe(false)
    expect(await forwarding.useHost('claude-code')).toEqual({ ok: true })
    expect(await forwarding.current()).toMatchObject({ bound: false, onHost: { profile: 'claude-code' } })
  } finally { await forwarding.disconnect() }
  const old = forwardingWith({ ready: true, relayed: true, oldHost: true })
  try { await expect(old.forwarding.useHost('claude-code')).rejects.toThrow('no working claude-code sign-in') }
  finally { await old.forwarding.disconnect() }
})

test('host-ready Codex is used automatically for new and already-relayed SSH tasks without new grants', async () => {
  for (const relayed of [false, true]) {
    const { forwarding, calls, grants } = forwardingWith({ ready: true, relayed }, 'codex')
    try {
      expect(await forwarding.follow()).toEqual({ status: 'skipped', reason: 'the host has its own codex sign-in' })
      expect(grants()).toBe(0)
      expect(calls).toContainEqual({ method: 'provider_list', params: { with_readiness: true } })
      expect(calls.find(call => call.method === 'set_model')?.params).toEqual({ model: 'gpt-6.1-sol', provider_profile: 'codex', session_key: 'task', ...(relayed ? { host_credentials: true } : {}) })
      expect(calls.some(call => call.method === 'provider.remote.bind')).toBe(false)
      expect(await forwarding.current()).toMatchObject({ bound: false, onHost: { profile: 'codex', model: 'gpt-6.1-sol' } })
    } finally { await forwarding.disconnect() }
  }
})

test('Run on the server selects host Codex before any relay has been created', async () => {
  const { forwarding, calls, grants } = forwardingWith({ ready: true }, 'codex')
  try {
    expect(await forwarding.useHost('codex')).toEqual({ ok: true })
    expect(grants()).toBe(0)
    expect(calls.find(call => call.method === 'set_model')?.params).toEqual({ model: 'gpt-6.1-sol', provider_profile: 'codex', session_key: 'task' })
    expect(calls.some(call => call.method === 'provider.remote.bind')).toBe(false)
    expect(await forwarding.current()).toMatchObject({ bound: false, onHost: { profile: 'codex' } })
  } finally { await forwarding.disconnect() }
})

test('a host-ready sign-in row is no longer forced through this computer', () => {
  const ssh = { storageScope: 'ssh:tpu-host' }
  expect(runsOnThisComputer(ssh, { signsIn: true })).toBe(true)
  expect(runsOnThisComputer(ssh, { signsIn: true, ready: true })).toBe(false)
  expect(runsOnThisComputer({ storageScope: 'local' }, { signsIn: true })).toBe(false)
})
