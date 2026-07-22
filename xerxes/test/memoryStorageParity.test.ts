// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { Database } from 'bun:sqlite'
import { createHash } from 'node:crypto'
import { mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  FileStorage,
  HashEmbedder,
  RAGStorage,
  SimpleStorage,
  SQLiteStorage,
  type Embedder,
  type MemoryStorage,
} from '../src/memory/index.js'

test('storage parity keeps CRUD and non-semantic contracts consistent across memory, file, and SQLite fallback stores', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-storage-parity-'))
  const stores: MemoryStorage[] = [
    new SimpleStorage(),
    new FileStorage(join(directory, 'files')),
    new SQLiteStorage({ dbPath: join(directory, 'fallback.db'), writeEnabled: false }),
  ]
  try {
    for (const storage of stores) {
      expect(storage.load('missing')).toBeUndefined()
      expect(storage.delete('missing')).toBeFalse()
      expect(storage.save('agent_1', { value: 1 })).toBeTrue()
      expect(storage.save('agent_2', 'two')).toBeTrue()
      expect(storage.save('task_1', 'three')).toBeTrue()
      expect(storage.load('agent_1')).toEqual({ value: 1 })
      expect(storage.exists('agent_2')).toBeTrue()
      expect(storage.exists('missing')).toBeFalse()
      expect(storage.listKeys('agent')).toEqual(expect.arrayContaining(['agent_1', 'agent_2']))
      expect(storage.semanticSearch('anything')).toEqual([])
      expect(storage.supportsSemanticSearch()).toBeFalse()
      expect(storage.delete('task_1')).toBeTrue()
      expect(storage.clear()).toBe(2)
      expect(storage.listKeys()).toEqual([])
    }
  } finally {
    for (const storage of stores) {
      if (storage instanceof SQLiteStorage) storage.close()
    }
    rmSync(directory, { force: true, recursive: true })
  }
})

test('RAG storage parity persists embeddings separately, restores them, and keeps embedding records out of ordinary key listings', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-rag-parity-'))
  const root = join(directory, 'store')
  try {
    const backend = new FileStorage(root)
    const embedder = new CountingEmbedder()
    const first = new RAGStorage(backend, embedder)
    expect(first.save('doc1', 'the quick brown fox')).toBeTrue()
    expect(first.save('doc2', 'lazy dog jumps over moon')).toBeTrue()
    expect(embedder.calls).toBe(2)
    expect(backend.load(`${RAGStorage.embeddingKeyPrefix}doc1`)).toHaveLength(32)
    expect(first.listKeys().sort()).toEqual(['doc1', 'doc2'])
    expect(first.supportsSemanticSearch()).toBeTrue()

    const restored = new RAGStorage(new FileStorage(root), new HashEmbedder(32))
    expect(restored.semanticSearch('the quick', 2).map(result => result.key)).toContain('doc1')
    expect(restored.semanticSearch('unrelated terms', 10, 0.99).every(result => result.similarity >= 0.99)).toBeTrue()

    expect(restored.delete('doc1')).toBeTrue()
    expect(restored.load('doc1')).toBeUndefined()
    expect(restored.listKeys()).toEqual(['doc2'])
    expect(restored.save(`${RAGStorage.embeddingKeyPrefix}raw`, [0.1, 0.2])).toBeTrue()
    expect(restored.backend.load(`${RAGStorage.embeddingKeyPrefix}raw`)).toEqual([0.1, 0.2])
    expect(restored.clear()).toBeGreaterThan(0)
    expect(restored.backend.listKeys()).toEqual([])
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('RAG storage parity semantically ranks text and structured content through the native storage protocol', () => {
  const rag = new RAGStorage(new SimpleStorage(), new HashEmbedder(64))
  expect(rag.save('fruit', 'apple banana cherry')).toBeTrue()
  expect(rag.save('vehicle', { content: 'car bus train' })).toBeTrue()

  const fruit = rag.semanticSearch('apple banana', 1)
  const vehicle = rag.semanticSearch('car bus', 1)
  expect(fruit).toHaveLength(1)
  expect(fruit[0]?.key).toBe('fruit')
  expect(vehicle[0]?.key).toBe('vehicle')
})

test('file storage writes atomically and treats corrupt records as missing', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-file-storage-'))
  const root = join(directory, 'store')
  try {
    const storage = new FileStorage(root)
    expect(storage.save('good', { value: 1 })).toBeTrue()
    expect(readdirSync(root).filter(entry => entry.includes('.tmp'))).toEqual([])

    // Corrupt the persisted payload behind the index entry.
    const hash = createHash('md5').update('good').digest('hex')
    writeFileSync(join(root, `${hash}.json`), 'not json', 'utf8')
    expect(storage.exists('good')).toBeTrue()
    expect(storage.load('good')).toBeUndefined()

    // A corrupt index file is backed up and rebuilt from scanned data files
    // (recovered under their hash stems) instead of orphaning every record.
    writeFileSync(join(root, '_index.json'), 'not json', 'utf8')
    const restored = new FileStorage(root)
    expect(restored.listKeys()).toEqual([hash])
    expect(readdirSync(root).some(entry => entry.startsWith('_index.json.corrupt-'))).toBeTrue()
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('RAG storage skips a corrupt embedding entry during restore', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-rag-corrupt-'))
  const path = join(directory, 'memory.db')
  try {
    const database = new Database(path)
    database.run(`
      CREATE TABLE IF NOT EXISTS memory (
        key TEXT PRIMARY KEY,
        data TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    `)
    database.query('INSERT INTO memory (key, data, created_at, updated_at) VALUES (?, ?, ?, ?)')
      .run(`${RAGStorage.embeddingKeyPrefix}broken`, 'not json', 'now', 'now')
    database.close()

    // The corrupt row warns and returns undefined from backend.load; restore
    // must skip it, not brick startup.
    const rag = new RAGStorage(new SQLiteStorage({ dbPath: path, writeEnabled: true }), new HashEmbedder(16))
    expect(rag.semanticSearch('anything')).toEqual([])
    expect(rag.save('healthy', 'a clean record')).toBeTrue()
    expect(rag.semanticSearch('clean', 1).map(result => result.key)).toEqual(['healthy'])
    ;(rag.backend as SQLiteStorage).close()
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

class CountingEmbedder implements Embedder {
  readonly dimension = 32
  readonly name = 'counting-hash'
  calls = 0
  private readonly delegate = new HashEmbedder(this.dimension)

  embed(text: string): number[] {
    this.calls += 1
    return this.delegate.embed(text)
  }

  embedBatch(texts: readonly string[]): number[][] {
    return texts.map(text => this.embed(text))
  }
}
