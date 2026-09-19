// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { modelInventory, type ModelInventoryPort } from '../src/runtime/modelInventory.js'
const profiles = [{ name: 'local', provider: 'custom', model: 'small', active: true, api_key: 'never-return', base_url: 'secret-endpoint' }, { name: 'second', provider: 'custom', model: 'large', active: false }]
const port: ModelInventoryPort = { profiles: () => profiles, discover: async () => ({ models: [{ id: 'small', context_limit: 32000, context_source: 'provider' }, { id: 'large' }], source: 'remote' }), reasoning: async () => ({ efforts: ['low', 'high'], source: 'provider_reported', shape: 'effort' }) }
test('empty optional profile fields list providers instead of looking up a blank profile', async () => {
  for (const provider_profile of ['', '  ']) {
    const result = await modelInventory(port, {provider_profile, include_usage:false, query:'', offset:0, limit:1, revision:''})
    expect(result.mode).toBe('providers')
    expect(result.configured_profiles).toBe(2)
    await expect(modelInventory(port, {provider_profile, include_usage:true})).rejects.toThrow('requires provider_profile')
  }
})
test('first-page retries recover from empty, stale and different-query revisions', async () => {
  const providers = await modelInventory(port, {})
  const current = await modelInventory(port, { provider_profile: 'local' })
  for (const revision of ['', 'stale', providers.revision]) {
    for (const offset of [undefined, 0]) {
      const refreshed = await modelInventory(port, { provider_profile: 'local', query: '', include_usage: true, revision, ...(offset === undefined ? {} : { offset }) })
      expect(refreshed.entries).toEqual(current.entries)
      expect(refreshed.revision).toBe(current.revision)
    }
  }
  const filtered = await modelInventory(port, { provider_profile: 'local', query: 'small', revision: current.revision })
  expect(filtered.entries.map(entry => entry.model)).toEqual(['small'])
  await expect(modelInventory(port, { provider_profile: 'local', offset: 1, revision: '' })).rejects.toThrow('omit revision')
  const first = await modelInventory(port, { provider_profile: 'local', limit: 1, offset: 0, revision: 'stale' })
  const next = await modelInventory(port, { provider_profile: 'local', offset: first.next_offset, revision: first.revision })
  expect(next.entries.map(entry => entry.model)).toEqual(['small'])
})
test('profile inventory is bounded, counts providers and never exposes connection credentials', async () => {
  const result = await modelInventory(port, { limit: 1 })
  expect(result.configured_profiles).toBe(2)
  expect(result.configured_providers).toBe(1)
  expect(result.next_offset).toBe(1)
  expect(JSON.stringify(result)).not.toContain('never-return')
  expect(JSON.stringify(result)).not.toContain('secret-endpoint')
  expect(result.quota).toMatchObject({ status: 'unknown', remaining_tokens: null })
  expect((await modelInventory(port, { offset: 1, revision: result.revision })).entries[0]?.provider_profile).toBe('second')
  await expect(modelInventory(port, { offset: 1 })).rejects.toThrow('revision')
  await expect(modelInventory({ ...port, profiles: () => profiles.slice(0, 1) }, { offset: 1, revision: result.revision })).rejects.toThrow('changed')
})
test('model detail keeps unknown capacity separate from subscription allowance and validates paging', async () => {
  const result = await modelInventory(port, { provider_profile: 'local', query: 'small' })
  expect(result.entries).toEqual([{ provider_profile: 'local', provider: 'custom', model: 'small', spawn_supported: true, context_window: 32000, max_output_tokens: null, context_source: 'provider', output_source: 'unknown', reasoning_efforts: ['low', 'high'], reasoning_source: 'provider_reported', reasoning_shape: 'effort', default_reasoning_effort: null }])
  await expect(modelInventory(port, { provider_profile: 'missing' })).rejects.toThrow('Unknown')
  await expect(modelInventory(port, { limit: 51 })).rejects.toThrow('limit')
})
test('cancellation and discovery failures do not masquerade as empty catalogs', async () => {
  await expect(modelInventory(port, {}, AbortSignal.abort(new Error('cancelled')))).rejects.toThrow('cancelled')
  await expect(modelInventory({ ...port, discover: async () => { throw new Error('discovery failed') } }, { provider_profile: 'local' })).rejects.toThrow('discovery failed')
  const controller = new AbortController()
  let reasoning = 0
  await expect(modelInventory({ ...port, discover: async () => { controller.abort(new Error('cancelled')); return { models: [{ id: 'small' }], source: 'remote' } }, reasoning: async () => { reasoning++; return { efforts: [], source: 'unknown', shape: 'inherent' } } }, { provider_profile: 'local' }, controller.signal)).rejects.toThrow('cancelled')
  expect(reasoning).toBe(0)
})
test('routing preferences are scoped and invalidate catalog pagination when changed', async () => {
  const notes = [{ provider_profile: 'local', model: '', note: 'Prefer for exploration', revision: 1 }, { provider_profile: 'local', model: 'small', note: 'Quick checks', revision: 1 }]
  const noted = { ...port, routingNotes: () => notes }
  const summary = await modelInventory(noted, {})
  expect(JSON.stringify(summary.entries[0])).toContain('Prefer for exploration')
  expect(JSON.stringify(summary.entries[0])).not.toContain('Quick checks')
  expect(JSON.stringify(summary.entries[1])).not.toContain('routing_notes')
  const detail = await modelInventory(noted, { provider_profile: 'local' })
  expect(JSON.stringify(detail.entries.find(entry => entry.model === 'small'))).toContain('Quick checks')
  expect(JSON.stringify(detail.entries.find(entry => entry.model === 'large'))).not.toContain('Quick checks')
  notes[0]!.note = ''
  notes[0]!.revision++
  await expect(modelInventory(noted, { offset: 1, revision: summary.revision })).rejects.toThrow('changed')
  expect(JSON.stringify((await modelInventory(noted, {})).entries)).not.toContain('routing_notes')
})
test('large UTF-8 notes paginate into intact responses within the tool byte limit', async () => {
  const names = Array.from({ length: 50 }, (_, i) => `model-${String(i).padStart(2, '0')}`)
  const inventory: ModelInventoryPort = { ...port,
    routingNotes: () => [{ provider_profile: 'local', model: '', note: '界'.repeat(2000), revision: 1 }, ...names.map(model => ({ provider_profile: 'local', model, note: '文'.repeat(2000), revision: 1 }))],
    discover: async () => ({ source: 'fixture', models: names.map(id => ({ id })) }),
  }
  let offset = 0, revision: string | undefined
  const seen: string[] = []
  do {
    const result = await modelInventory(inventory, { provider_profile: 'local', limit: 50, offset, ...(revision ? { revision } : {}) })
    expect(new TextEncoder().encode(JSON.stringify(result)).byteLength).toBeLessThanOrEqual(60000)
    seen.push(...result.entries.map(entry => entry.model as string))
    revision = result.revision
    if (result.next_offset === null) break
    expect(result.next_offset).toBeGreaterThan(offset)
    offset = result.next_offset
  } while (offset < names.length)
  expect(seen).toEqual(names)
})
test('reasoning discovery preserves fallback provenance, control shape and unknown defaults', async () => {
  for (const source of ['bundled_catalog', 'provider_fallback', 'unknown'] as const) {
    const result = await modelInventory({ ...port, reasoning: async () => ({ efforts: ['off', 'on'], source, shape: 'toggle' }) }, { provider_profile: 'local' })
    expect(result.entries[0]).toMatchObject({ reasoning_source: source, reasoning_shape: 'toggle', default_reasoning_effort: null })
  }
  const result = await modelInventory({ ...port, reasoning: async () => ({ efforts: ['low', 'high'], source: 'provider_reported', shape: 'effort', defaultEffort: 'high' }) }, { provider_profile: 'local' })
  expect(result.entries[0]?.default_reasoning_effort).toBe('high')
})
test('pagination invalidates when reasoning metadata or discovery provenance changes', async () => {
  let effort = 'high', source = 'remote'
  const inventory: ModelInventoryPort = { ...port,
    discover: async () => ({ ...(await port.discover('local')), source }),
    reasoning: async (_profile, model) => ({ efforts: ['low', model === 'large' ? effort : 'high'], defaultEffort: model === 'large' ? effort : 'high', source: 'provider_reported', shape: 'effort' }),
  }
  const first = await modelInventory(inventory, { provider_profile: 'local', limit: 1 })
  expect(first.next_offset).toBe(1)
  await modelInventory(inventory, { provider_profile: 'local', offset: 1, revision: first.revision })
  effort = 'medium'
  await expect(modelInventory(inventory, { provider_profile: 'local', offset: 1, revision: first.revision })).rejects.toThrow('changed')
  const refreshed = await modelInventory(inventory, { provider_profile: 'local', limit: 1 })
  source = 'profile'
  await expect(modelInventory(inventory, { provider_profile: 'local', offset: 1, revision: refreshed.revision })).rejects.toThrow('changed')
})
test('quota lookup is opt-in and bound to the requested provider profile', async () => {
  const requests: string[] = []
  const inventory: ModelInventoryPort = { ...port, quota: async name => { requests.push(name); return { status: 'unknown', remaining_tokens: null, reason: 'fixture unavailable' } } }
  await modelInventory(inventory, { provider_profile: 'local' })
  expect(requests).toEqual([])
  const result = await modelInventory(inventory, { provider_profile: 'local', include_usage: true })
  expect(requests).toEqual(['local'])
  expect(result.quota).toMatchObject({ reason: 'fixture unavailable' })
  await expect(modelInventory(inventory, { include_usage: true })).rejects.toThrow('requires provider_profile')
  await expect(modelInventory(inventory, { include_usage: 'yes' })).rejects.toThrow('Invalid')
})
