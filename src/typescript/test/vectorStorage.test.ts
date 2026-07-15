// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { SQLiteVectorStorage, type Embedder } from '../src/memory/index.js'

class TopicEmbedder implements Embedder {
  readonly dimension = 2
  readonly name = 'topic-test'

  embed(text: string): number[] {
    const normalized = text.toLowerCase()
    return [
      normalized.includes('fruit') || normalized.includes('apple') ? 1 : 0,
      normalized.includes('vehicle') || normalized.includes('car') ? 1 : 0,
    ]
  }

  embedBatch(texts: readonly string[]): number[][] {
    return texts.map(text => this.embed(text))
  }
}

class FailingEmbedder implements Embedder {
  readonly dimension = 3
  readonly name = 'failing-test'

  embed(_text: string): number[] {
    throw new Error('embedding service unavailable')
  }

  embedBatch(texts: readonly string[]): number[][] {
    return texts.map(text => this.embed(text))
  }
}

test('SQLiteVectorStorage persists structured values, bytes, and semantic vectors across instances', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-vectors-'))
  const path = join(directory, 'vectors.db')
  const first = new SQLiteVectorStorage({ dbPath: path, embedder: new TopicEmbedder() })
  let restored: SQLiteVectorStorage | undefined
  try {
    const payload = {
      bytes: new Uint8Array([0, 1, 255]),
      nested: [new Uint8Array([7, 8])],
      title: 'fruit record',
    }
    expect(first.save('fruit', payload)).toBe(true)
    expect(first.save('vehicle', 'fast vehicle')).toBe(true)
    expect(first.exists('fruit')).toBe(true)

    const loaded = first.load('fruit') as { bytes: Uint8Array; nested: Uint8Array[]; title: string }
    expect([...loaded.bytes]).toEqual([0, 1, 255])
    expect([...loaded.nested[0] as Uint8Array]).toEqual([7, 8])
    expect(loaded.title).toBe('fruit record')
    expect(first.semanticSearch('apple fruit', 10, 0.5).map(result => result.key)).toEqual(['fruit'])
    first.close()

    restored = new SQLiteVectorStorage(path, new TopicEmbedder())
    expect(restored.load('vehicle')).toBe('fast vehicle')
    expect(restored.semanticSearch('vehicle', 1, 0.5).at(0)?.key).toBe('vehicle')
  } finally {
    restored?.close()
    first.close()
    rmSync(directory, { recursive: true, force: true })
  }
})

test('SQLiteVectorStorage sorts equal semantic scores and protects LIKE patterns', () => {
  const storage = new SQLiteVectorStorage({ dbPath: ':memory:', embedder: new TopicEmbedder() })
  try {
    expect(storage.save('zeta', 'fruit')).toBe(true)
    expect(storage.save('alpha', 'fruit')).toBe(true)
    expect(storage.save('literal_%', 'vehicle')).toBe(true)
    expect(storage.save('literal_x', 'vehicle')).toBe(true)

    expect(storage.semanticSearch('fruit', 10, 0.9).map(result => result.key)).toEqual(['alpha', 'zeta'])
    expect(storage.listKeys('literal_%')).toEqual(['literal_%'])
    expect(storage.delete('zeta')).toBe(true)
    expect(storage.delete('zeta')).toBe(false)
    expect(storage.clear()).toBe(3)
  } finally {
    storage.close()
  }
})

test('SQLiteVectorStorage degrades embedding failures and rejects unsafe payloads without partial writes', () => {
  const storage = new SQLiteVectorStorage({ dbPath: ':memory:', embedder: new FailingEmbedder() })
  try {
    expect(storage.save('kept', 'stored despite embedder failure')).toBe(true)
    expect(storage.load('kept')).toBe('stored despite embedder failure')
    expect(storage.semanticSearch('query')).toEqual([])
    expect(storage.save('undefined', { value: undefined })).toBe(false)
    expect(storage.save('bigint', 1n)).toBe(false)
    expect(storage.exists('undefined')).toBe(false)
    expect(storage.exists('bigint')).toBe(false)
  } finally {
    storage.close()
  }
})

test('SQLiteVectorStorage has an explicit, idempotent close lifecycle', () => {
  const storage = new SQLiteVectorStorage({ dbPath: ':memory:' })
  storage.close()
  storage.close()
  expect(() => storage.load('missing')).toThrow('SQLiteVectorStorage is closed')
})
