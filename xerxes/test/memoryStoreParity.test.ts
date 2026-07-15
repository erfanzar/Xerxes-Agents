// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  LEGACY_MEMORY_TYPES,
  MemoryEntry,
  MemoryStore,
  MemoryType,
} from '../src/memory/index.js'

test('memory-entry parity preserves typed fields, serialization, and mutable embeddings', () => {
  const timestamp = new Date('2026-07-13T10:00:00.000Z')
  const entry = new MemoryEntry({
    agentId: 'test-agent',
    content: 'Test memory',
    context: { key: 'value' },
    importanceScore: 0.8,
    memoryType: MemoryType.SHORT_TERM,
    tags: ['test', 'memory'],
    timestamp,
  })

  expect(entry.content).toBe('Test memory')
  expect(entry.memoryType).toBe(MemoryType.SHORT_TERM)
  expect(entry.agentId).toBe('test-agent')
  expect(entry.context).toEqual({ key: 'value' })
  expect(entry.importanceScore).toBe(0.8)
  expect(entry.tags).toEqual(['test', 'memory'])
  expect(entry.accessCount).toBe(0)
  expect(entry.lastAccessed).toBeUndefined()
  expect(entry.toRecord()).toMatchObject({
    agent_id: 'test-agent',
    content: 'Test memory',
    context: { key: 'value' },
    importance_score: 0.8,
    memory_type: MemoryType.SHORT_TERM,
    tags: ['test', 'memory'],
    timestamp: timestamp.toISOString(),
  })

  entry.embedding = Array.from({ length: 768 }, (_, index) => index / 768)
  expect(entry.embedding).toHaveLength(768)
})

test('memory-store parity initializes every bucket, enforces bounded categories, and retains typed records', () => {
  const store = new MemoryStore({ maxShortTerm: 2, maxWorking: 1 })

  expect(store.maxShortTerm).toBe(2)
  expect(store.maxWorking).toBe(1)
  expect(Object.keys(store.memories).sort()).toEqual([...LEGACY_MEMORY_TYPES].sort())
  expect(LEGACY_MEMORY_TYPES.every(memoryType => store.memories[memoryType].length === 0)).toBeTrue()

  store.addMemory({ agentId: 'agent', content: 'Memory 1', importanceScore: 0.5, memoryType: MemoryType.SHORT_TERM })
  store.addMemory({ agentId: 'agent', content: 'Memory 2', importanceScore: 0.5, memoryType: MemoryType.SHORT_TERM })
  store.addMemory({ agentId: 'agent', content: 'Memory 3', importanceScore: 0.5, memoryType: MemoryType.SHORT_TERM })
  store.addMemory({ agentId: 'agent', content: 'Working 1', memoryType: MemoryType.WORKING })
  store.addMemory({ agentId: 'agent', content: 'Working 2', memoryType: MemoryType.WORKING })

  expect(store.memories[MemoryType.SHORT_TERM].map(entry => entry.content)).toEqual(['Memory 2', 'Memory 3'])
  expect(store.memories[MemoryType.WORKING].map(entry => entry.content)).toEqual(['Working 2'])
})

test('memory-store parity retrieves by category, owner, tags, importance, and recent window before clearing selected buckets', () => {
  const store = new MemoryStore()
  const now = new Date('2026-07-13T12:00:00.000Z')
  const old = store.addMemory({
    agentId: 'agent-1',
    content: 'Old memory',
    memoryType: MemoryType.SHORT_TERM,
    timestamp: new Date('2026-07-13T09:00:00.000Z'),
  })
  const short = store.addMemory({
    agentId: 'agent-1',
    content: 'Short term',
    importanceScore: 0.6,
    memoryType: MemoryType.SHORT_TERM,
    tags: ['important', 'urgent'],
    timestamp: new Date('2026-07-13T11:30:00.000Z'),
  })
  const long = store.addMemory({
    agentId: 'agent-2',
    content: 'Long term',
    importanceScore: 0.9,
    memoryType: MemoryType.LONG_TERM,
    tags: ['important'],
    timestamp: new Date('2026-07-13T11:45:00.000Z'),
  })
  const episodic = store.addMemory({
    agentId: 'agent-1',
    content: 'Episodic',
    importanceScore: 0.2,
    memoryType: MemoryType.EPISODIC,
    tags: ['trivial'],
    timestamp: new Date('2026-07-13T11:50:00.000Z'),
  })

  expect(store.retrieveMemories({ memoryType: MemoryType.SHORT_TERM }).map(entry => entry.memoryId))
    .toEqual([short.memoryId, old.memoryId])
  expect(store.retrieveMemories({ agentId: 'agent-1' }).map(entry => entry.memoryId))
    .toEqual([episodic.memoryId, short.memoryId, old.memoryId])
  expect(store.retrieveMemories({ tags: ['urgent'] })).toEqual([short])
  expect(store.retrieveMemories({ minImportance: 0.8 })).toEqual([long])
  expect(store.retrieveMemories({ memoryTypes: [MemoryType.LONG_TERM, MemoryType.EPISODIC] }).map(entry => entry.memoryId))
    .toEqual([episodic.memoryId, long.memoryId])
  expect(store.retrieveRecent({ minutesAgo: 60, now }).map(entry => entry.memoryId))
    .toEqual([episodic.memoryId, long.memoryId, short.memoryId])

  store.clearMemories({ memoryType: MemoryType.SHORT_TERM })
  expect(store.memories[MemoryType.SHORT_TERM]).toEqual([])
  expect(store.memories[MemoryType.LONG_TERM]).toEqual([long])
  store.clearMemories()
  expect(LEGACY_MEMORY_TYPES.every(memoryType => store.memories[memoryType].length === 0)).toBeTrue()
})

test('memory-store parity consolidates important records and accepts large, duplicate-tagged Unicode content', () => {
  const store = new MemoryStore({ maxShortTerm: 5 })
  const contents = ['Important memory 0', 'Important memory 1', 'Important memory 2']
  for (const content of contents) {
    store.addMemory({
      agentId: 'agent-1',
      content,
      importanceScore: 0.9,
      memoryType: MemoryType.SHORT_TERM,
      tags: ['tag', 'tag', 'durable'],
    })
  }
  const longContent = `Test with special chars: 你好 🚀 \n\t\r${'x'.repeat(10_000)}`
  const edge = store.addMemory({
    agentId: 'agent-1',
    content: longContent,
    memoryType: MemoryType.SHORT_TERM,
    tags: ['tag', 'tag'],
  })

  const summary = store.consolidateMemories({ agentId: 'agent-1', threshold: 0.8 })
  expect(summary).toContain('Important facts:')
  expect(store.retrieveMemories({ memoryType: MemoryType.LONG_TERM }).map(entry => entry.content))
    .toEqual(expect.arrayContaining(contents))
  expect(edge.content).toBe(longContent)
  expect(edge.tags).toEqual(['tag', 'tag'])
  expect(store.getStatistics()).toMatchObject({ cacheHitRate: 0 })
})
