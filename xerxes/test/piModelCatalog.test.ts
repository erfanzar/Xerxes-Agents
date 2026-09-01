// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  PI_MODEL_CATALOG_SOURCE,
  piCatalogContextWindow,
  piCatalogModelCapabilities,
  resolvePiContextWindow,
} from '../src/llms/piModelCatalog.js'

test('Pi catalog resolves capacities by provider and normalized model id', () => {
  expect(piCatalogContextWindow('glm-5.2', 'zhipu')).toBe(1_000_000)
  expect(piCatalogContextWindow('zai/glm-5.2', 'zai')).toBe(1_000_000)
  expect(piCatalogContextWindow('codex/gpt-5.6-sol', 'openai-codex')).toBe(272_000)
  expect(
    piCatalogContextWindow('openrouter/anthropic/claude-sonnet-4.5', 'openrouter'),
  ).toBeGreaterThan(0)
  expect(piCatalogModelCapabilities('glm-5.2', 'zhipu')).toMatchObject({
    api: 'openai-completions',
    contextLimit: 1_000_000,
    maxOutputTokens: 131_072,
    reasoning: true,
  })
  expect(piCatalogModelCapabilities('gpt-5.6-sol', 'openai-codex')).toMatchObject({
    api: 'openai-codex-responses',
    contextLimit: 272_000,
    maxOutputTokens: 128_000,
    reasoning: true,
  })
})

test('provider or user metadata overrides Pi catalog and misses stay unknown', () => {
  expect(resolvePiContextWindow({
    configuredContextWindow: 262_144,
    model: 'glm-5.2',
    provider: 'zhipu',
  })).toBe(262_144)
  expect(resolvePiContextWindow({ model: 'not-in-the-catalog', provider: 'custom' })).toBeUndefined()
  expect(resolvePiContextWindow({ model: 'glm-5.2', provider: 'custom' })).toBeUndefined()
})

test('generated catalog records reproducible Pi provenance', () => {
  expect(PI_MODEL_CATALOG_SOURCE.package).toBe('@earendil-works/pi-ai')
  expect(PI_MODEL_CATALOG_SOURCE.version).toMatch(/^\d+\.\d+\.\d+/u)
  expect(Date.parse(PI_MODEL_CATALOG_SOURCE.generated_at)).not.toBeNaN()
})
