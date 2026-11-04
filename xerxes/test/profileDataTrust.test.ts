// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import { parseAgentIntelligenceConfig, resolveAgentIntelligenceTier } from '../src/agents/intelligence.js'
import { ProfileStore } from '../src/bridge/profiles.js'
import { ValidationError } from '../src/core/errors.js'
import type { DataTrust } from '../src/security/boundaryTypes.js'
import { deriveDataTrust } from '../src/security/dataBoundary.js'

const roots: string[] = []
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function store(): ProfileStore {
  const root = mkdtempSync(join(tmpdir(), 'xerxes-profile-trust-'))
  roots.push(root)
  return new ProfileStore(join(root, 'profiles.json'))
}

test('data_trust and prompt_profile round-trip through profiles.json', () => {
  const profiles = store()
  profiles.save({ name: 'gpu', baseUrl: 'https://gpu.lan/v1', apiKey: 'k', model: 'm', dataTrust: 'local', promptProfile: 'lean' })
  const loaded = new ProfileStore(profiles.filePath).get('gpu')
  expect(loaded?.data_trust).toBe('local')
  expect(loaded?.prompt_profile).toBe('lean')
  expect(deriveDataTrust(loaded)).toBe('local')
  // Omitted on a later save of the same connection: kept.
  profiles.save({ name: 'gpu', baseUrl: 'https://gpu.lan/v1', apiKey: 'k2', model: 'm' })
  expect(profiles.get('gpu')).toMatchObject({ data_trust: 'local', prompt_profile: 'lean' })
  // Moved to another endpoint: the explicit trust no longer applies.
  profiles.save({ name: 'gpu', baseUrl: 'https://api.example.com/v1', apiKey: 'k2', model: 'm' })
  expect(profiles.get('gpu')?.data_trust).toBeUndefined()
  expect(deriveDataTrust(profiles.get('gpu'))).toBe('cloud')
})

test('profiles without the new fields still load unchanged', () => {
  const profiles = store()
  writeFileSync(profiles.filePath, JSON.stringify({
    active: 'old',
    profiles: { old: { name: 'old', base_url: 'http://localhost:8080/v1', api_key: '', model: 'x', provider: 'openai', sampling: {} } },
  }))
  const old = profiles.get('old')
  expect(old).toBeDefined()
  expect(old && 'data_trust' in old).toBe(false)
  expect(old && 'prompt_profile' in old).toBe(false)
  expect(deriveDataTrust(old)).toBe('local')
  profiles.save({ name: 'old', baseUrl: 'http://localhost:8080/v1', apiKey: '', model: 'y' })
  const written = JSON.parse(readFileSync(profiles.filePath, 'utf8')) as { profiles: Record<string, Record<string, unknown>> }
  expect(Object.keys(written.profiles.old ?? {})).not.toContain('data_trust')
})

test('invalid values are refused on save and fail closed on load', () => {
  const profiles = store()
  expect(() => profiles.save({ name: 'p', baseUrl: 'http://localhost', apiKey: '', model: 'm', dataTrust: 'remote' as DataTrust })).toThrow(ValidationError)
  expect(() => profiles.save({ name: 'p', baseUrl: 'http://localhost', apiKey: '', model: 'm', promptProfile: 'tiny' as 'lean' })).toThrow(ValidationError)
  writeFileSync(profiles.filePath, JSON.stringify({
    active: null,
    profiles: { p: { name: 'p', base_url: 'http://127.0.0.1:1/v1', api_key: '', model: 'm', provider: 'openai', sampling: {}, data_trust: 'weird', prompt_profile: 'tiny' } },
  }))
  const loaded = profiles.get('p')
  expect(loaded?.data_trust).toBe('cloud')
  expect(loaded?.prompt_profile).toBeUndefined()
})

test('resolveAgentIntelligenceTier ignores an explicit model', () => {
  const config = parseAgentIntelligenceConfig({ default: 'light', light: { model: 'local-llm', provider_profile: 'local' }, smart: 'big' })
  expect(resolveAgentIntelligenceTier(config)).toEqual({ model: 'local-llm', provider_profile: 'local' })
  expect(resolveAgentIntelligenceTier(config, 'smart')).toEqual({ model: 'big' })
  expect(resolveAgentIntelligenceTier(config, 'balanced')).toBeUndefined()
  expect(resolveAgentIntelligenceTier(parseAgentIntelligenceConfig({ default: 'inherit' }))).toBeUndefined()
})
