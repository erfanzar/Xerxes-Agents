// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { BackoffPolicy, CachingToolExecutor, JitterMode, ToolResultCache, hashArgs } from '../src/index.js'
import type { ToolCall } from '../src/types/toolCalls.js'

test('backoff policies preserve deterministic and bounded jitter schedules', () => {
  const deterministic = new BackoffPolicy({ mode: JitterMode.NONE, baseDelay: 1, maxDelay: 5, multiplier: 2 })
  expect([...deterministic.delays(4)]).toEqual([1, 2, 4, 5])
  const full = new BackoffPolicy({ mode: JitterMode.FULL, baseDelay: 4, random: () => 0.25 })
  expect(full.delay(0)).toBe(1)
  const equal = new BackoffPolicy({ mode: JitterMode.EQUAL, baseDelay: 4, random: () => 0.5 })
  expect(equal.delay(0)).toBe(3)
})

test('tool result cache has stable keys, TTL expiration, and LRU eviction', () => {
  let time = 0
  const cache = new ToolResultCache<string>({ maxEntries: 2, ttl: 10, now: () => time })
  cache.set('read', { b: 2, a: 1 }, 'one')
  expect(cache.get('read', { a: 1, b: 2 })).toBe('one')
  expect(hashArgs({ a: 1, b: 2 })).toBe(hashArgs({ b: 2, a: 1 }))
  cache.set('two', {}, 'two')
  cache.set('three', {}, 'three')
  expect(cache.get('read', { a: 1, b: 2 })).toBeUndefined()
  time = 11
  expect(cache.get('three', {})).toBeUndefined()
  expect(cache.statistics).toEqual({ hits: 1, misses: 2 })
})

test('caching tool executor only caches explicitly idempotent tool names', async () => {
  let calls = 0
  const executor = new CachingToolExecutor({
    execute: async () => 'value:' + ++calls,
  }, { cacheableTools: ['ReadFile'] })
  const read: ToolCall = {
    id: 'first',
    type: 'function',
    function: { name: 'ReadFile', arguments: { file_path: 'a.txt' } },
  }
  expect(await executor.execute(read, { metadata: {} })).toBe('value:1')
  expect(await executor.execute({ ...read, id: 'second' }, { metadata: {} })).toBe('value:1')
  expect(await executor.execute({
    ...read,
    id: 'third',
    function: { name: 'WriteFile', arguments: { file_path: 'a.txt' } },
  }, { metadata: {} })).toBe('value:2')
})
