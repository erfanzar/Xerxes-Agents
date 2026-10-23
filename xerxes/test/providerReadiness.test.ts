// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ProfileStore } from '../src/bridge/profiles.js'
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
  expect((await createSignInReadinessProbe({ environment: {}, executable: installed, login: async () => ({}) })('openai-codex')).state).toBe('unknown')
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
    providerReadiness: async provider => provider === 'claude-code' ? { state: 'ready' } : { state: 'unknown' } })
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
    expect(list.find(row => row.name === 'keyed')?.ready).toBeUndefined()
    expect(JSON.stringify(rows)).not.toContain('synthetic-key')
  } finally { rpc.dispose(); await server.stop(); await runtime.shutdown(); await rm(dir, { recursive: true, force: true }) }
})

/** A desktop window over a fake host whose readiness answer the test controls. */
function forwardingWith(host: { readiness?: boolean; ready?: boolean; setModel?: boolean }) {
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  let grants = 0, bound = false
  const profile = { name: 'claude-code', model: 'claude-code/opus', supported: true, credential_source: 'local provider authentication or environment', output_limit_mode: 'request-bound' }
  const forwarding = new DesktopProviderForwarding(async method => {
    if (method === 'provider.relay.inventory') return { ok: true, profiles: [profile] }
    if (method === 'provider.relay.authorize') return { ok: true, grant: { id: String(++grants).repeat(32) } }
    return { ok: true }
  }, async (method, params) => {
    calls.push({ method, params })
    if (method === 'provider_list') return host.readiness === false ? { ok: true, profiles: [{ name: 'claude-code', signs_in: true }] }
      : { ok: true, readiness_supported: true, profiles: [{ name: 'claude-code', provider: 'claude-code', model: '', signs_in: true, ready: host.ready === true }] }
    if (method === 'set_model') return host.setModel === false ? { ok: false, error: 'refused' } : { ok: true }
    if (method === 'provider.remote.bind') { bound = true; return { ok: true, binding: { workspace: '/project', profile: params.profile, model: params.model, binding: 'a'.repeat(32) } } }
    return { ok: true, provider_binding_session_guard_supported: true, provider_binding_busy: false, provider_binding_lost: false,
      session: { key: 'task', cwd: '/project', status: 'idle', turn_count: 0, ...(bound ? { local_provider_label: 'Local provider: claude-code' } : {}) } }
  }, 'tpu-host', '/project', async () => 'claude-code')
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

test('a host-ready sign-in row is no longer forced through this computer', () => {
  const ssh = { storageScope: 'ssh:tpu-host' }
  expect(runsOnThisComputer(ssh, { signsIn: true })).toBe(true)
  expect(runsOnThisComputer(ssh, { signsIn: true, ready: true })).toBe(false)
  expect(runsOnThisComputer({ storageScope: 'local' }, { signsIn: true })).toBe(false)
})
