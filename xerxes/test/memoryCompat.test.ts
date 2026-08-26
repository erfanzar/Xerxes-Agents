// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  MemoryCompatibilityError,
  MemoryEntry,
  MemoryStore,
  MemoryStoreConfigurationError,
  MemoryType,
  SimpleStorage,
} from '../src/memory/index.js'

test('MemoryEntry mirrors typed context, tags, and importance into its core item metadata', () => {
  const timestamp = new Date('2026-07-13T10:00:00.000Z')
  const entry = new MemoryEntry({
    agentId: 'planner',
    content: 'Preserve JSON-RPC protocol compatibility.',
    context: { subsystem: 'daemon' },
    importanceScore: 0.8,
    memoryType: MemoryType.SEMANTIC,
    metadata: { confidence: 0.95 },
    tags: ['protocol', 'durable'],
    timestamp,
  })

  entry.context.phase = 'rewrite'
  entry.tags.push('bun')
  entry.touch(timestamp)

  expect(entry.toMemoryItem().metadata).toEqual({
    confidence: 0.95,
    context: { subsystem: 'daemon', phase: 'rewrite' },
    importance: 0.8,
    tags: ['protocol', 'durable', 'bun'],
  })
  expect(entry.toRecord()).toMatchObject({
    agent_id: 'planner',
    content: 'Preserve JSON-RPC protocol compatibility.',
    context: { subsystem: 'daemon', phase: 'rewrite' },
    importance_score: 0.8,
    memory_type: MemoryType.SEMANTIC,
    tags: ['protocol', 'durable', 'bun'],
    timestamp: timestamp.toISOString(),
  })
  expect(entry.accessCount).toBe(1)
  expect(() => { entry.importanceScore = 2 }).toThrow(MemoryCompatibilityError)
})

test('MemoryStore indexes buckets while retaining the current tier as the storage authority', () => {
  const store = new MemoryStore({ maxLongTerm: 4, maxShortTerm: 2, maxWorking: 1 })
  const first = store.addMemory({
    agentId: 'agent-a',
    content: 'First scratch entry',
    importanceScore: 0.2,
    memoryType: MemoryType.SHORT_TERM,
    timestamp: new Date('2026-07-13T08:00:00.000Z'),
  })
  store.addMemory({
    agentId: 'agent-a',
    content: 'Second scratch entry',
    importanceScore: 0.4,
    memoryType: MemoryType.SHORT_TERM,
    timestamp: new Date('2026-07-13T09:00:00.000Z'),
  })
  const promoted = store.addMemory({
    agentId: 'agent-a',
    content: 'High-priority protocol invariant',
    importanceScore: 0.9,
    memoryType: MemoryType.SHORT_TERM,
    tags: ['protocol', 'urgent'],
    timestamp: new Date('2026-07-13T10:00:00.000Z'),
  })
  const semantic = store.addMemory({
    agentId: 'agent-b',
    content: 'Semantic API fact',
    importanceScore: 0.7,
    memoryType: MemoryType.SEMANTIC,
    tags: ['protocol'],
    timestamp: new Date('2026-07-13T11:00:00.000Z'),
  })
  store.addMemory({ agentId: 'agent-a', content: 'old working state', memoryType: MemoryType.WORKING })
  const working = store.addMemory({
    agentId: 'agent-a',
    content: 'current working state',
    memoryType: MemoryType.WORKING,
  })

  expect(store.memories[MemoryType.SHORT_TERM].map(entry => entry.content)).toEqual([
    'Second scratch entry',
    'High-priority protocol invariant',
  ])
  expect(store.shortTerm.retrieve(first.memoryId)).toBeUndefined()
  expect(store.shortTerm.retrieve(promoted.memoryId)).toBe(promoted.toMemoryItem())
  expect(store.longTerm.retrieve(semantic.memoryId)).toBe(semantic.toMemoryItem())
  expect(store.memories[MemoryType.WORKING]).toEqual([working])
  expect(store.retrieveMemories({ agentId: 'agent-a', tags: ['urgent'] })).toEqual([promoted])

  const summary = store.consolidateMemories({ agentId: 'agent-a', threshold: 0.8 })
  expect(store.memories[MemoryType.SHORT_TERM]).not.toContain(promoted)
  expect(store.memories[MemoryType.LONG_TERM].some(entry => entry.content === promoted.content)).toBe(true)
  expect(store.longTerm.search('protocol invariant').some(item => item.content === promoted.content)).toBe(true)
  expect(summary).toContain('Important facts:')
  expect(store.retrieveRecent({ minutesAgo: 90, now: new Date('2026-07-13T11:20:00.000Z') })).toContain(semantic)

  store.clearMemories({ agentId: 'agent-b' })
  expect(store.retrieveMemories({ memoryType: MemoryType.SEMANTIC })).toEqual([])
  expect(store.getStatistics()).toMatchObject({ cacheHitRate: 0, totalMemories: 3 })
})

test('MemoryStore writes SQLite only after explicit persistence opt-in and restores durable categories', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-memory-compat-'))
  const path = join(directory, 'compat-memory.db')
  try {
    const disabled = new MemoryStore({ persistence: { path, writeEnabled: false } })
    disabled.addMemory({ agentId: 'agent', content: 'must remain in memory', memoryType: MemoryType.LONG_TERM })
    disabled.close()
    expect(existsSync(path)).toBe(false)

    const enabled = new MemoryStore({ persistence: { path, writeEnabled: true } })
    const durable = enabled.addMemory({
      agentId: 'agent',
      content: 'Persist the semantic compatibility contract.',
      context: { release: 'typescript' },
      importanceScore: 0.9,
      memoryType: MemoryType.SEMANTIC,
      tags: ['durable'],
    })
    enabled.close()
    expect(existsSync(path)).toBe(true)

    const restored = new MemoryStore({ persistence: { path, writeEnabled: true } })
    expect(restored.persistenceEnabled).toBe(true)
    expect(restored.retrieveMemories({ memoryType: MemoryType.SEMANTIC })).toHaveLength(1)
    expect(restored.retrieveMemories({ memoryType: MemoryType.SEMANTIC }).at(0)?.memoryId).toBe(durable.memoryId)
    expect(restored.retrieveMemories({ memoryType: MemoryType.SEMANTIC }).at(0)?.context)
      .toEqual({ release: 'typescript' })
    restored.close()
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('MemoryStore rejects ambiguous storage and persistence configuration', () => {
  expect(() => new MemoryStore({
    persistence: { path: '/tmp/xerxes-memory.db', writeEnabled: true },
    storage: new SimpleStorage(),
  })).toThrow(MemoryStoreConfigurationError)
})

test('MemoryStore hydration restores entries without re-persisting them', () => {
  class CountingStorage extends SimpleStorage {
    saves = 0
    override save(key: string, data: unknown): boolean {
      this.saves += 1
      return super.save(key, data)
    }
  }
  const storage = new CountingStorage()
  const seed = new MemoryStore({ storage })
  seed.addMemory({
    agentId: 'agent',
    content: 'Durable boot entry',
    importanceScore: 0.9,
    memoryType: MemoryType.LONG_TERM,
  })
  expect(storage.saves).toBeGreaterThan(0)

  storage.saves = 0
  const restored = new MemoryStore({ storage })
  expect(restored.retrieveMemories({ memoryType: MemoryType.LONG_TERM })).toHaveLength(1)
  // Boot hydration must not trigger one re-persist (or re-embed) per restored entry.
  expect(storage.saves).toBe(0)
})

test('MemoryStore sqlite persistence restores short-term-tier entries into a fresh instance', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-memory-compat-durable-'))
  const path = join(directory, 'memory.db')
  try {
    const first = new MemoryStore({ persistence: { path, writeEnabled: true } })
    for (let index = 0; index < 5; index += 1) {
      first.addMemory({
        agentId: 'runner',
        content: `Scratch observation ${index}`,
        importanceScore: 0.3,
        memoryType: MemoryType.SHORT_TERM,
        timestamp: new Date(Date.UTC(2026, 6, 13, 8, index)),
      })
    }
    first.addMemory({
      agentId: 'runner',
      content: 'Semantic API invariant',
      importanceScore: 0.9,
      memoryType: MemoryType.SEMANTIC,
      timestamp: new Date(Date.UTC(2026, 6, 13, 9, 0)),
    })
    expect(first.size).toBe(6)
    first.close()

    // A fresh instance over the same path must restore all six entries: five
    // short-term rows through the now-durable short-term tier and one
    // semantic row through the long-term tier.
    const second = new MemoryStore({ persistence: { path, writeEnabled: true } })
    try {
      expect(second.size).toBe(6)
      expect(second.memories[MemoryType.SHORT_TERM].map(entry => entry.content)).toEqual(
        Array.from({ length: 5 }, (_, index) => `Scratch observation ${index}`),
      )
      expect(second.memories[MemoryType.SEMANTIC].map(entry => entry.content)).toEqual(['Semantic API invariant'])
      expect(second.retrieveMemories({ agentId: 'runner', limit: 10 }).map(entry => entry.content))
        .toContain('Semantic API invariant')
    } finally {
      second.close()
    }
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('boot hydration bounds memory without destroying durable rows', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-memory-compat-bootbound-'))
  const path = join(directory, 'memory.db')
  try {
    const seeder = new MemoryStore({ maxShortTerm: 6, persistence: { path, writeEnabled: true } })
    for (let index = 0; index < 6; index += 1) {
      seeder.addMemory({
        agentId: 'runner',
        content: `Durable scratch ${index}`,
        importanceScore: 0.4,
        memoryType: MemoryType.SHORT_TERM,
        timestamp: new Date(Date.UTC(2026, 7, 24, 10, index)),
      })
    }
    seeder.close()

    // An over-cap configuration bounds what boot keeps in memory...
    const bounded = new MemoryStore({ maxShortTerm: 3, persistence: { path, writeEnabled: true } })
    expect(bounded.memories[MemoryType.SHORT_TERM].map(entry => entry.content)).toEqual([
      'Durable scratch 3',
      'Durable scratch 4',
      'Durable scratch 5',
    ])
    bounded.close()

    // ...but hydration must not delete rows: a roomier instance still
    // restores the full history.
    const verifier = new MemoryStore({ persistence: { path, writeEnabled: true } })
    expect(verifier.size).toBe(6)
    verifier.close()
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('boot hydration restore order is deterministic for identical timestamps', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-memory-compat-order-'))
  const path = join(directory, 'memory.db')
  try {
    const sameTimestamp = new Date(Date.UTC(2026, 7, 24, 11, 0))
    const seeder = new MemoryStore({ persistence: { path, writeEnabled: true } })
    for (const content of ['alpha', 'beta', 'gamma', 'delta']) {
      seeder.addMemory({
        agentId: 'runner',
        content,
        importanceScore: 0.5,
        memoryType: MemoryType.SHORT_TERM,
        timestamp: sameTimestamp,
      })
    }
    seeder.close()

    const boots = [0, 1].map(() => {
      const store = new MemoryStore({ persistence: { path, writeEnabled: true } })
      try {
        return store.memories[MemoryType.SHORT_TERM].map(entry => ({ content: entry.content, id: entry.memoryId }))
      } finally {
        store.close()
      }
    })
    // Two boots over the same rows agree exactly...
    expect(boots[1]).toEqual(boots[0])
    // ...because identical timestamps are tie-broken by ascending memoryId.
    const firstBoot = boots[0]
    if (firstBoot === undefined) throw new Error('first boot missing')
    const ids = firstBoot.map(entry => entry.id)
    expect([...ids].sort((left, right) => left.localeCompare(right))).toEqual(ids)
    expect(firstBoot.map(entry => entry.content).sort()).toEqual(['alpha', 'beta', 'delta', 'gamma'])
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('explicit runtime eviction still trims durable rows after a non-destructive boot', () => {
  const storage = new SimpleStorage()
  const store = new MemoryStore({ maxShortTerm: 2, storage })
  for (let index = 0; index < 4; index += 1) {
    store.addMemory({
      agentId: 'runner',
      content: `Runtime eviction ${index}`,
      importanceScore: 0.3,
      memoryType: MemoryType.SHORT_TERM,
      timestamp: new Date(Date.UTC(2026, 7, 24, 12, index)),
    })
  }
  const stmRows = () => storage.listKeys().filter(key => key.startsWith('stm_'))
  expect(store.memories[MemoryType.SHORT_TERM]).toHaveLength(2)
  expect(stmRows()).toHaveLength(2)
})
