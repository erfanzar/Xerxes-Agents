// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, rmSync, unlinkSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

import { cosineSimilarity, getDefaultEmbedder, type Embedder } from './embedders.js'

export interface MemoryStorage {
  clear(): number
  delete(key: string): boolean
  exists(key: string): boolean
  listKeys(pattern?: string): string[]
  load(key: string): unknown | undefined
  save(key: string, data: unknown): boolean
  semanticSearch(query: string, limit?: number, threshold?: number): SemanticSearchResult[]
  supportsSemanticSearch(): boolean
}

export interface SemanticSearchResult {
  readonly data: unknown
  readonly key: string
  readonly similarity: number
}

/** Ephemeral in-process storage appropriate for tests and read-only sessions. */
export class SimpleStorage implements MemoryStorage {
  private readonly data = new Map<string, unknown>()

  clear(): number {
    const count = this.data.size
    this.data.clear()
    return count
  }

  delete(key: string): boolean {
    return this.data.delete(key)
  }

  exists(key: string): boolean {
    return this.data.has(key)
  }

  listKeys(pattern?: string): string[] {
    const keys = [...this.data.keys()]
    return pattern ? keys.filter(key => key.includes(pattern)) : keys
  }

  load(key: string): unknown | undefined {
    return this.data.get(key)
  }

  save(key: string, data: unknown): boolean {
    this.data.set(key, data)
    return true
  }

  semanticSearch(_query: string, _limit = 10, _threshold = 0): SemanticSearchResult[] {
    return []
  }

  supportsSemanticSearch(): boolean {
    return false
  }
}

/** JSON-file key/value backend whose hashed filenames prevent key-path traversal. */
export class FileStorage implements MemoryStorage {
  private readonly indexFile: string
  private index: Record<string, string>

  constructor(readonly directory = '.xerxes_memory') {
    mkdirSync(directory, { recursive: true })
    this.indexFile = join(directory, '_index.json')
    this.index = this.readIndex()
  }

  clear(): number {
    const keys = Object.keys(this.index)
    for (const key of keys) {
      const file = this.pathForKey(key)
      if (existsSync(file)) unlinkSync(file)
    }
    this.index = {}
    this.writeIndex()
    return keys.length
  }

  delete(key: string): boolean {
    if (!(key in this.index)) return false
    const file = this.pathForKey(key)
    if (existsSync(file)) unlinkSync(file)
    delete this.index[key]
    this.writeIndex()
    return true
  }

  exists(key: string): boolean {
    return key in this.index
  }

  listKeys(pattern?: string): string[] {
    const keys = Object.keys(this.index)
    return pattern ? keys.filter(key => key.includes(pattern)) : keys
  }

  load(key: string): unknown | undefined {
    const filename = this.index[key]
    if (!filename) return undefined
    const path = join(this.directory, filename)
    if (!existsSync(path)) return undefined
    return JSON.parse(readFileSync(path, 'utf8')) as unknown
  }

  save(key: string, data: unknown): boolean {
    const path = this.pathForKey(key)
    try {
      writeFileSync(path, JSON.stringify(data), 'utf8')
      this.index[key] = path.slice(this.directory.length + 1)
      this.writeIndex()
      return true
    } catch {
      return false
    }
  }

  semanticSearch(_query: string, _limit = 10, _threshold = 0): SemanticSearchResult[] {
    return []
  }

  supportsSemanticSearch(): boolean {
    return false
  }

  private pathForKey(key: string): string {
    const hash = createHash('md5').update(key).digest('hex')
    return join(this.directory, `${hash}.json`)
  }

  private readIndex(): Record<string, string> {
    if (!existsSync(this.indexFile)) return {}
    try {
      const parsed = JSON.parse(readFileSync(this.indexFile, 'utf8')) as unknown
      return isRecord(parsed) && Object.values(parsed).every(value => typeof value === 'string')
        ? parsed as Record<string, string>
        : {}
    } catch {
      return {}
    }
  }

  private writeIndex(): void {
    writeFileSync(this.indexFile, JSON.stringify(this.index), 'utf8')
  }
}

export interface SQLiteStorageOptions {
  readonly dbPath?: string
  readonly writeEnabled?: boolean
}

/** Bun SQLite backend retaining Xerxes' optional WRITE_MEMORY persistence switch. */
export class SQLiteStorage implements MemoryStorage {
  readonly dbPath: string
  readonly writeEnabled: boolean
  private readonly fallback = new SimpleStorage()
  private readonly database: Database | undefined

  constructor(options: SQLiteStorageOptions = {}) {
    this.dbPath = options.dbPath ?? '.xerxes_memory/memory.db'
    this.writeEnabled = options.writeEnabled ?? process.env.WRITE_MEMORY === '1'
    if (!this.writeEnabled) return
    mkdirSync(dirname(this.dbPath), { recursive: true })
    this.database = new Database(this.dbPath)
    this.database.run(`
      CREATE TABLE IF NOT EXISTS memory (
        key TEXT PRIMARY KEY,
        data TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    `)
    this.database.run('CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory(created_at)')
  }

  clear(): number {
    if (!this.database) return this.fallback.clear()
    const row = this.database.query('SELECT COUNT(*) AS count FROM memory').get() as { count?: number } | null
    this.database.run('DELETE FROM memory')
    return row?.count ?? 0
  }

  close(): void {
    this.database?.close()
  }

  delete(key: string): boolean {
    if (!this.database) return this.fallback.delete(key)
    const result = this.database.query('DELETE FROM memory WHERE key = ?').run(key)
    return result.changes > 0
  }

  exists(key: string): boolean {
    if (!this.database) return this.fallback.exists(key)
    return this.database.query('SELECT 1 FROM memory WHERE key = ? LIMIT 1').get(key) !== null
  }

  listKeys(pattern?: string): string[] {
    if (!this.database) return this.fallback.listKeys(pattern)
    const rows = this.database.query('SELECT key FROM memory ORDER BY created_at DESC').all() as Array<{ key: unknown }>
    const keys = rows.flatMap(row => typeof row.key === 'string' ? [row.key] : [])
    return pattern ? keys.filter(key => key.includes(pattern)) : keys
  }

  load(key: string): unknown | undefined {
    if (!this.database) return this.fallback.load(key)
    const row = this.database.query('SELECT data FROM memory WHERE key = ?').get(key) as { data?: unknown } | null
    if (!row || typeof row.data !== 'string') return undefined
    return JSON.parse(row.data) as unknown
  }

  save(key: string, data: unknown): boolean {
    if (!this.database) return this.fallback.save(key, data)
    try {
      const timestamp = new Date().toISOString()
      this.database.query(`
        INSERT INTO memory (key, data, created_at, updated_at) VALUES (?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at
      `).run(key, JSON.stringify(data), timestamp, timestamp)
      return true
    } catch {
      return false
    }
  }

  semanticSearch(_query: string, _limit = 10, _threshold = 0): SemanticSearchResult[] {
    return []
  }

  supportsSemanticSearch(): boolean {
    return false
  }
}

/** Adds locally-computed embeddings and semantic scan to any storage backend. */
export class RAGStorage implements MemoryStorage {
  static readonly embeddingKeyPrefix = '_emb_'

  private readonly embeddings = new Map<string, number[]>()

  constructor(
    readonly backend: MemoryStorage = new SimpleStorage(),
    readonly embedder: Embedder = getDefaultEmbedder(),
  ) {
    this.restoreEmbeddings()
  }

  clear(): number {
    this.embeddings.clear()
    for (const key of this.backend.listKeys(RAGStorage.embeddingKeyPrefix)) {
      this.backend.delete(key)
    }
    return this.backend.clear()
  }

  delete(key: string): boolean {
    this.embeddings.delete(key)
    this.backend.delete(`${RAGStorage.embeddingKeyPrefix}${key}`)
    return this.backend.delete(key)
  }

  exists(key: string): boolean {
    return this.backend.exists(key)
  }

  listKeys(pattern?: string): string[] {
    return this.backend.listKeys(pattern).filter(key => !key.startsWith(RAGStorage.embeddingKeyPrefix))
  }

  load(key: string): unknown | undefined {
    return this.backend.load(key)
  }

  save(key: string, data: unknown): boolean {
    if (key.startsWith(RAGStorage.embeddingKeyPrefix)) return this.backend.save(key, data)
    const saved = this.backend.save(key, data)
    if (!saved) return false
    const embedding = this.embedder.embed(dataToText(data))
    this.embeddings.set(key, embedding)
    this.backend.save(`${RAGStorage.embeddingKeyPrefix}${key}`, embedding)
    return true
  }

  semanticSearch(query: string, limit = 10, threshold = 0): SemanticSearchResult[] {
    const queryEmbedding = this.embedder.embed(query)
    const matches: SemanticSearchResult[] = []
    for (const [key, embedding] of this.embeddings) {
      const similarity = cosineSimilarity(queryEmbedding, embedding)
      if (similarity < threshold) continue
      const data = this.backend.load(key)
      if (data !== undefined) matches.push({ key, similarity, data })
    }
    return matches.sort((left, right) => right.similarity - left.similarity).slice(0, limit)
  }

  supportsSemanticSearch(): boolean {
    return true
  }

  private restoreEmbeddings(): void {
    for (const key of this.backend.listKeys(RAGStorage.embeddingKeyPrefix)) {
      if (!key.startsWith(RAGStorage.embeddingKeyPrefix)) continue
      const vector = this.backend.load(key)
      if (!Array.isArray(vector) || !vector.every(value => typeof value === 'number')) continue
      this.embeddings.set(key.slice(RAGStorage.embeddingKeyPrefix.length), [...vector])
    }
  }
}

function dataToText(value: unknown): string {
  if (typeof value === 'string') return value
  if (isRecord(value) && typeof value.content === 'string') return value.content
  return JSON.stringify(value)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
