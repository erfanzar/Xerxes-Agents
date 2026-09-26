// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ProfileStore } from '../src/bridge/profiles.js'
import { AgentSettingsStore } from '../src/agents/settingsStore.js'
import { inheritedSelectionValidator, profileInventoryHost, profileSelectionValidator } from '../src/runtime/profileInventory.js'

test('inherited selection validates alternate models against the captured parent connection', async () => {
  const requests: (string | null)[] = []
  // The endpoint reports the parent model's effort levels itself — the only
  // grounds on which an effort outside them may be refused.
  const endpoint = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: request => {
    requests.push(request.headers.get('authorization'))
    return Response.json({ data: [{ id: 'worker' }, { id: 'parent', supports_reasoning: true, think_efforts: { support: true, valid_efforts: ['low', 'high'] } }] })
  } })
  try {
    const connection = { provider: 'openai', model: 'parent', apiKey: 'parent-key', baseUrl: `${endpoint.url}v1`, permissionMode: 'accept-all' as const }
    const validate = inheritedSelectionValidator(connection)
    connection.apiKey = 'replacement-key'
    await validate('parent')
    expect(requests).toHaveLength(0)
    await validate('worker')
    await expect(validate('unknown')).rejects.toThrow('not configured or discovered')
    await expect(validate('parent', 'imaginary')).rejects.toThrow('Unsupported reasoning effort')
    expect(requests).toEqual(['Bearer parent-key', 'Bearer parent-key', 'Bearer parent-key'])
    await expect(validate('parent', undefined, AbortSignal.abort(new Error('cancelled')))).rejects.toThrow('cancelled')
  } finally { endpoint.stop(true) }
})

test('standalone Radius inventory reads gateway configuration and preserves credential ownership', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'radius-inventory-'))
  const requests: { path: string; auth: string | null }[] = []
  const endpoint = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: request => {
    requests.push({ path: new URL(request.url).pathname, auth: request.headers.get('authorization') })
    return Response.json({ baseUrl: 'https://example.invalid/messages', models: [{ id: 'radius-worker', name: 'Worker', reasoning: true, input: ['text'], cost: {}, contextWindow: 128000, maxTokens: 8192 }] })
  } })
  try {
    const profiles = new ProfileStore(join(directory, 'profiles.json'))
    profiles.save({ name: 'gateway', provider: 'radius', apiKey: 'radius-fixture-secret', baseUrl: String(endpoint.url), model: 'radius-worker' })
    const host = profileInventoryHost(profiles, new AgentSettingsStore(join(directory, 'settings.sqlite')))
    const result = await host('session', { provider_profile: 'gateway' })
    expect(result).toMatchObject({ entries: [{ model: 'radius-worker', context_window: 128000, max_output_tokens: 8192, context_source: 'provider', output_source: 'provider' }] })
    expect(JSON.stringify(result)).not.toContain('radius-fixture-secret')
    await profileSelectionValidator(profiles)('gateway', 'radius-worker')
    expect(requests).toEqual(Array.from({ length: 2 }, () => ({ path: '/v1/config', auth: 'Bearer radius-fixture-secret' })))
  } finally { endpoint.stop(true); await rm(directory, { recursive: true, force: true }) }
})

test('standalone Copilot inventory and selection use the subscription catalog and sanitize failures', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'copilot-inventory-'))
  try {
    const profiles = new ProfileStore(join(directory, 'profiles.json'))
    profiles.save({ name: 'copilot', provider: 'github-copilot', apiKey: '', baseUrl: 'https://example.invalid', model: 'configured-worker' })
    let fail = false, calls = 0
    const controller = new AbortController()
    const options = { copilotModels: async (signal?: AbortSignal) => {
      calls++
      expect(signal).toBe(controller.signal)
      if (fail) throw new Error('oauth-fixture-secret')
      return ['subscription-worker']
    } }
    const host = profileInventoryHost(profiles, new AgentSettingsStore(join(directory, 'settings.sqlite')), options)
    expect(await host('session', { provider_profile: 'copilot' }, controller.signal)).toMatchObject({ entries: [{ model: 'subscription-worker' }] })
    await profileSelectionValidator(profiles, options)('copilot', 'subscription-worker', undefined, controller.signal)
    expect(calls).toBe(2)
    fail = true
    await expect(host('session', { provider_profile: 'copilot' }, controller.signal)).rejects.toThrow('check the Copilot login')
    controller.abort(new Error('cancelled'))
    await expect(host('session', { provider_profile: 'copilot' }, controller.signal)).rejects.toThrow('cancelled')
    expect(calls).toBe(3)
  } finally { await rm(directory, { recursive: true, force: true }) }
})
test('standalone profile inventory discovers a configured endpoint, notes and provenance without exposing keys', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'profile-inventory-'))
  let fail = false
  const endpoint = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: () => fail ? new Response('fixture-secret', { status: 500 }) : Response.json({ data: [{ id: 'worker', context_length: 32000 }] }) })
  try {
    const profiles = new ProfileStore(join(directory, 'profiles.json'))
    profiles.save({ name: 'local', provider: 'openai', apiKey: 'fixture-secret', baseUrl: `${endpoint.url}v1`, model: 'worker' })
    const settings = new AgentSettingsStore(join(directory, 'settings.sqlite'))
    settings.saveRoutingNote('local', 'worker', 'Review complex code', 0)
    const host = profileInventoryHost(profiles, settings)
    await expect(profileSelectionValidator(profiles)('wrong-host-name', 'worker')).rejects.toThrow('Configured profiles for this model on the execution host: "local"')
    for (const provider_profile of ['', ' \t\n']) {
      expect(await host('session', { provider_profile, include_usage: false, query: '', offset: 0, limit: 1, revision: '' })).toMatchObject({ source: 'configured_profiles', mode: 'providers', entries: [expect.objectContaining({ provider_profile: expect.any(String) })] })
      await expect(host('session', { provider_profile, include_usage: true })).rejects.toThrow('requires provider_profile')
    }
    await expect(host('session', { provider_profile: 'missing' })).rejects.toThrow('Unknown provider profile')
    expect(await host('session', { provider_profile: ' local\t' })).toMatchObject({ entries: [expect.objectContaining({ model: 'worker' })] })
    const result = await host('session', { provider_profile: 'local' })
    expect(result).toMatchObject({ source: 'provider', entries: [{ model: 'worker', context_window: 32000, reasoning_source: 'provider_fallback' }] })
    expect(JSON.stringify(result)).toContain('Review complex code')
    expect(JSON.stringify(result)).not.toContain('fixture-secret')
    profiles.updateModelCapabilities('local', 'worker', { contextLimit: 16000, maxOutputTokens: 2048 })
    expect(await host('session', { provider_profile: 'local' })).toMatchObject({ entries: [{ context_window: 16000, context_source: 'override', max_output_tokens: 2048, output_source: 'override' }] })
    profiles.updateModelCapabilities('local', 'worker', { contextLimit: null, maxOutputTokens: null })
    expect(await host('session', { provider_profile: 'local' })).toMatchObject({ entries: [{ context_window: 32000, context_source: 'provider' }] })
    fail = true
    await expect(host('session', { provider_profile: 'local' })).rejects.toThrow()
    await expect(host('session', {}, AbortSignal.abort(new Error('cancelled')))).rejects.toThrow('cancelled')
  } finally { endpoint.stop(true); await rm(directory, { recursive: true, force: true }) }
})

test('standalone agent selections validate models, reasoning, cancellation and profile replacement', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'profile-selection-'))
  let onRequest = () => {}
  let requests = 0
  const endpoint = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: () => {
    requests++
    onRequest()
    return Response.json({ data: [{ id: 'worker', supports_reasoning: true, think_efforts: { support: true, valid_efforts: ['low', 'high'] } }] })
  } })
  try {
    const profiles = new ProfileStore(join(directory, 'profiles.json'))
    const profile = { name: 'local', provider: 'openai', apiKey: 'fixture-secret', baseUrl: `${endpoint.url}v1`, model: 'configured-worker' }
    profiles.save(profile)
    const validate = profileSelectionValidator(profiles)
    await validate('local', 'worker')
    await validate('local', 'configured-worker')
    await expect(validate('local', 'missing')).rejects.toThrow('not configured or discovered')
    await expect(validate('local', 'worker', 'imaginary')).rejects.toThrow('Unsupported reasoning effort')
    await expect(validate('missing', 'worker')).rejects.toThrow('unavailable')
    const before = requests
    await expect(validate('local', 'worker', undefined, AbortSignal.abort(new Error('cancelled')))).rejects.toThrow('cancelled')
    expect(requests).toBe(before)
    const abort = new AbortController()
    onRequest = () => abort.abort(new Error('cancelled during discovery'))
    await expect(validate('local', 'worker', undefined, abort.signal)).rejects.toThrow('cancelled during discovery')
    onRequest = () => { profiles.save({ ...profile, apiKey: 'replacement-secret' }) }
    await expect(validate('local', 'worker')).rejects.toThrow('changed during discovery')
  } finally { endpoint.stop(true); await rm(directory, { recursive: true, force: true }) }
})
