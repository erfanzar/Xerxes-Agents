// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  ContextualMemory,
  HashEmbedder,
  HybridRetriever,
  LongTermMemory,
  MemoryItem,
  RAGStorage,
  ShortTermMemory,
  SimpleStorage,
  SQLiteStorage,
  coerceText,
  makeMemoryProvider,
  makeTurnIndexerHook,
} from '../src/memory/index.js'

test('short-term memory bounds its working set, searches metadata, and serializes items', () => {
  const memory = new ShortTermMemory({ capacity: 2 })
  memory.save('first alpha', {}, { agentId: 'one' })
  const retained = memory.save('second beta', { topic: 'build' }, { userId: 'user' })
  memory.save('third beta', {}, { agentId: 'two' })

  expect(memory.size).toBe(2)
  expect(memory.retrieve(undefined, { topic: 'build' })).toEqual([retained])
  expect(memory.search('beta').map(item => item.content)).toEqual(['third beta', 'second beta'])
  expect(MemoryItem.fromRecord(retained.toRecord()).toRecord()).toEqual(retained.toRecord())
})

test('long-term memory uses semantic storage and hydrates records from a supplied backend', () => {
  const storage = new RAGStorage(new SimpleStorage(), new HashEmbedder(64))
  const memory = new LongTermMemory({ storage, maxItems: 5 })
  const relevant = memory.save('Bun offers a fast TypeScript runtime', {}, { importance: 0.9 })
  memory.save('A garden has tomatoes', {}, { importance: 0.1 })

  expect(memory.search('TypeScript runtime', 1).at(0)?.memoryId).toBe(relevant.memoryId)
  const restored = new LongTermMemory({ storage, maxItems: 5 })
  const recalled = restored.retrieve(relevant.memoryId)
  expect(Array.isArray(recalled) ? undefined : recalled?.content).toBe('Bun offers a fast TypeScript runtime')
})

test('contextual memory routes important records and promotes frequently accessed working memory', () => {
  const contextual = new ContextualMemory({
    longTerm: new LongTermMemory({ storage: new SimpleStorage() }),
    promotionThreshold: 2,
  })
  contextual.pushContext('project', { repository: 'xerxes' })
  const shortTerm = contextual.save('Track the daemon protocol carefully')
  const important = contextual.save('Never lose persisted user data', {}, { importance: 0.9 })

  contextual.retrieve(shortTerm.memoryId)
  contextual.retrieve(shortTerm.memoryId)
  const recalled = contextual.longTerm.retrieve(important.memoryId)
  expect(Array.isArray(recalled) ? undefined : recalled?.content).toBe('Never lose persisted user data')
  expect(contextual.longTerm.search('daemon', 5).some(item => item.content.includes('daemon'))).toBe(true)
  expect(contextual.getContextSummary()).toContain('Current context:')
})

test('Bun SQLite storage persists when explicitly enabled', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-memory-'))
  const path = join(directory, 'memory.db')
  try {
    const storage = new SQLiteStorage({ dbPath: path, writeEnabled: true })
    expect(storage.save('key', { value: 1 })).toBe(true)
    storage.close()
    const restored = new SQLiteStorage({ dbPath: path, writeEnabled: true })
    expect(restored.load('key')).toEqual({ value: 1 })
    expect(restored.listKeys()).toEqual(['key'])
    restored.close()
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('hybrid retrieval blends semantic, lexical, and recency scores', () => {
  const recent = new MemoryItem({ content: 'The daemon speaks JSON RPC over a Unix socket' })
  const unrelated = new MemoryItem({ content: 'Coconut trees grow in warm climates' })
  const results = new HybridRetriever(new HashEmbedder(64)).rank('JSON RPC daemon socket', [unrelated, recent])
  expect(results.at(0)?.item.memoryId).toBe(recent.memoryId)
  expect(results.at(0)?.score).toBeGreaterThan(0)
})

test('turn indexer normalizes assistant content and exposes a recall provider', () => {
  const memory = new LongTermMemory({ storage: new SimpleStorage() })
  const indexTurn = makeTurnIndexerHook(memory, { minChars: 5 })
  indexTurn({ agentId: 'coder', response: { content: [{ text: 'A useful response from the agent.' }] } })
  expect(memory.size).toBe(1)
  expect(makeMemoryProvider(memory)('useful', 3)).toEqual(['A useful response from the agent.'])
  expect(coerceText({ content: [{ text: 'one' }, 'two'] })).toBe('one\ntwo')
})
