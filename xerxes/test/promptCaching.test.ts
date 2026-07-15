// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  EPHEMERAL_CACHE_CONTROL,
  SUPPORTS_CACHING,
  extractCacheTokens,
  wrapSystemWithCache,
  wrapToolsWithCache,
} from '../src/streaming/promptCaching.js'

test('prompt caching exposes Anthropic as the supported provider and marks non-empty system prompts', () => {
  expect(SUPPORTS_CACHING).toEqual(['anthropic'])
  expect(wrapSystemWithCache('')).toBe('')
  expect(wrapSystemWithCache('Stable instructions.')).toEqual([{
    type: 'text',
    text: 'Stable instructions.',
    cache_control: { type: 'ephemeral' },
  }])
})

test('tool cache wrapping copies schemas and leaves exactly one tail breakpoint', () => {
  const tools = [{
    name: 'ReadFile',
    cache_control: { type: 'ephemeral' },
    input_schema: { type: 'object' },
  }, {
    name: 'WriteFile',
    cache_control: { type: 'persistent' },
    input_schema: { type: 'object' },
  }]

  const wrapped = wrapToolsWithCache(tools)
  expect(wrapped).not.toBe(tools)
  expect(wrapped).toEqual([{
    name: 'ReadFile',
    input_schema: { type: 'object' },
  }, {
    name: 'WriteFile',
    input_schema: { type: 'object' },
    cache_control: EPHEMERAL_CACHE_CONTROL,
  }])
  expect(tools).toEqual([{
    name: 'ReadFile',
    cache_control: { type: 'ephemeral' },
    input_schema: { type: 'object' },
  }, {
    name: 'WriteFile',
    cache_control: { type: 'persistent' },
    input_schema: { type: 'object' },
  }])

  const empty: readonly Record<string, unknown>[] = []
  expect(wrapToolsWithCache(empty)).toBe(empty)
})

test('cache token extraction accepts record and SDK-shaped usage while defaulting malformed values', () => {
  expect(extractCacheTokens({
    cache_read_input_tokens: 13,
    cache_creation_input_tokens: 7,
  })).toEqual([13, 7])

  class Usage {
    readonly cache_creation_input_tokens = '5'
    readonly cache_read_input_tokens = 11.8
  }

  expect(extractCacheTokens(new Usage())).toEqual([11, 5])
  expect(extractCacheTokens({
    cache_read_input_tokens: Number.POSITIVE_INFINITY,
    cache_creation_input_tokens: 'not-a-number',
  })).toEqual([0, 0])
  expect(extractCacheTokens(undefined)).toEqual([0, 0])
})
