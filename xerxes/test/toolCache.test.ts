// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ToolOutputCache,
  ToolOutputCacheError,
  extractToolFilePaths,
  toolCacheContentKey,
} from '../src/runtime/toolCache.js'

test('ToolOutputCache caches only the read-only allowlist with deterministic content keys', async () => {
  let calls = 0
  const cache = new ToolOutputCache({ fileStat: () => undefined, monotonicNow: () => 0 })
  const executor = cache.wrap(async () => 'value:' + ++calls)

  expect(await executor('ReadFile', { file_path: 'notes.md', options: { b: 2, a: 1 } })).toBe('value:1')
  expect(await executor('ReadFile', { options: { a: 1, b: 2 }, file_path: 'notes.md' })).toBe('value:1')
  expect(await executor('FileEditTool', { file_path: 'notes.md' })).toBe('value:2')
  expect(await executor('FileEditTool', { file_path: 'notes.md' })).toBe('value:3')
  expect(toolCacheContentKey('ReadFile', { b: 2, a: 1 })).toBe(toolCacheContentKey('ReadFile', { a: 1, b: 2 }))
  expect(cache.stats).toEqual({ hitRate: 0.5, hits: 1, misses: 1, size: 1 })
})

test('ToolOutputCache invalidates cache keys when path mtime or size changes', async () => {
  let calls = 0
  let version = { mtimeMs: 1, size: 4 }
  const cache = new ToolOutputCache({
    fileStat: path => path === '/workspace/file.ts' ? version : undefined,
    monotonicNow: () => 10,
  })
  const executor = cache.wrap(() => 'result:' + ++calls)

  expect(await executor('ReadFile', { file_path: '/workspace/file.ts' })).toBe('result:1')
  expect(await executor('ReadFile', { file_path: '/workspace/file.ts' })).toBe('result:1')
  version = { mtimeMs: 2, size: 4 }
  expect(await executor('ReadFile', { file_path: '/workspace/file.ts' })).toBe('result:2')
  version = { mtimeMs: 2, size: 5 }
  expect(await executor('ReadFile', { file_path: '/workspace/file.ts' })).toBe('result:3')
  expect(cache.stats).toMatchObject({ hits: 1, misses: 3 })
})

test('ToolOutputCache observes injected monotonic TTL and maintains bounded LRU order', () => {
  let time = 0
  const cache = new ToolOutputCache({
    fileStat: () => undefined,
    maxEntries: 2,
    monotonicNow: () => time,
    ttlSeconds: 5,
  })
  cache.put('ReadFile', { file_path: 'a' }, 'a')
  cache.put('ReadFile', { file_path: 'b' }, 'b')
  expect(cache.get('ReadFile', { file_path: 'a' })).toBe('a')
  cache.put('ReadFile', { file_path: 'c' }, 'c')
  expect(cache.get('ReadFile', { file_path: 'b' })).toBeUndefined()
  expect(cache.get('ReadFile', { file_path: 'a' })).toBe('a')
  time = 5.1
  expect(cache.get('ReadFile', { file_path: 'a' })).toBeUndefined()
  expect(cache.size).toBe(1)
})

test('ToolOutputCache supports targeted path/tool invalidation and full invalidation', () => {
  const cache = new ToolOutputCache({ fileStat: () => undefined })
  cache.put('ReadFile', { file_path: '/project/a.ts' }, 'a')
  cache.put('GrepTool', { pattern: '/project/**/*.ts' }, 'grep')
  cache.put('ListDir', { directory: '/project/subdir' }, 'list')

  expect(extractToolFilePaths({ pattern: '/project/**/*.ts' })).toEqual(['/project/**'])
  expect(cache.invalidate({ filePath: '/project/**' })).toBe(1)
  expect(cache.get('GrepTool', { pattern: '/project/**/*.ts' })).toBeUndefined()
  expect(cache.invalidate({ toolName: 'ReadFile' })).toBe(1)
  expect(cache.size).toBe(1)
  expect(cache.invalidate()).toBe(1)
  expect(cache.size).toBe(0)
  expect(() => cache.invalidate({ filePath: '' })).toThrow(ToolOutputCacheError)
})

test('ToolOutputCache skips oversized entries instead of pinning unbounded results', async () => {
  let calls = 0
  const cache = new ToolOutputCache({
    fileStat: () => undefined,
    maxEntryBytes: 16,
    monotonicNow: () => 0,
  })
  const executor = cache.wrap(async () => `result:${++calls}`)

  // A result under the byte cap is cached and served from the cache.
  expect(await executor('ReadFile', { file_path: 'a' })).toBe('result:1')
  expect(await executor('ReadFile', { file_path: 'a' })).toBe('result:1')

  // An over-cap result still executes but is never stored, so repeated calls
  // re-execute and the cache cannot grow past its per-entry byte budget.
  cache.put('ReadFile', { file_path: 'big' }, 'x'.repeat(17))
  expect(cache.get('ReadFile', { file_path: 'big' })).toBeUndefined()
  expect(cache.size).toBe(1)

  // Multi-byte results are measured in UTF-8 bytes, not string length.
  cache.put('ReadFile', { file_path: 'unicode' }, '€'.repeat(6)) // 18 bytes
  expect(cache.get('ReadFile', { file_path: 'unicode' })).toBeUndefined()

  expect(() => new ToolOutputCache({ maxEntryBytes: 0 })).toThrow(ToolOutputCacheError)
})
