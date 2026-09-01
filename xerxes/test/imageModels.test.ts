// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  DEFAULT_IMAGE_MODEL_REFERENCE,
  getImageModel,
  getImageModels,
  getImageProviders,
  IMAGE_MODEL_SOURCE,
  resolveImageModel,
} from '../src/images/imageModels.js'

test('the generated catalog mirrors the installed pi-ai package', () => {
  expect(IMAGE_MODEL_SOURCE.package).toBe('@earendil-works/pi-ai')
  // The generator stamps the pi-ai version; assert its shape so a broken
  // generation cannot pass silently.
  expect(IMAGE_MODEL_SOURCE.version).toMatch(/^\d+\.\d+\.\d+/)
  expect(getImageProviders()).toEqual(['openrouter'])
  // Keep pace with Pi's generated data; the count changes with upstream.
  expect(getImageModels().length).toBeGreaterThanOrEqual(50)
})

test('every catalog entry carries the wire fields generation needs', () => {
  for (const model of getImageModels()) {
    expect(model.api).toBe('openrouter-images')
    expect(model.baseUrl).toMatch(/^https:\/\//)
    expect(model.id.length).toBeGreaterThan(0)
    expect(model.name.length).toBeGreaterThan(0)
    expect(model.output).toContain('image')
    for (const value of Object.values(model.cost)) {
      expect(typeof value).toBe('number')
    }
  }
})

test('image model lookups match full route ids, which contain slashes', () => {
  const model = getImageModel('openrouter', 'google/gemini-3-pro-image')
  expect(model?.name).toContain('Gemini')
  expect(model?.api).toBe('openrouter-images')
  expect(getImageModel('openrouter', 'openrouter/auto')?.output).toContain('text')
  expect(getImageModel('openrouter', 'nonexistent')).toBeUndefined()
  expect(getImageModels('unknown-provider')).toEqual([])
})

test('resolveImageModel accepts provider-prefixed, bare, and default references', () => {
  expect(resolveImageModel('openrouter/recraft/recraft-v4').id).toBe('recraft/recraft-v4')
  expect(resolveImageModel('openai/gpt-image-2').id).toBe('openai/gpt-image-2')
  expect(resolveImageModel('bytedance-seed/seedream-4.5').id).toBe('bytedance-seed/seedream-4.5')
  const fallback = resolveImageModel(undefined)
  // The default reference is itself a full route id in the catalog.
  expect(fallback.id).toBe(DEFAULT_IMAGE_MODEL_REFERENCE)
  expect(fallback.provider).toBe('openrouter')
  expect(() => resolveImageModel('not-a-model')).toThrow(/unknown image model/)
})
