// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { createHash, randomUUID } from 'node:crypto'
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs'
import { dirname, join, resolve } from 'node:path'

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

/**
 * Tenant-scoped view over a shared backend. Every key is transparently
 * prefixed, so per-user memory tiers over one backend only see, list, search,
 * and delete their own records.
 */
export class NamespacedStorage implements MemoryStorage {
  constructor(
    readonly backend: MemoryStorage,
    readonly prefix: string,
  ) {}

  clear(): number {
    let removed = 0
    for (const key of this.backend.listKeys(this.prefix)) {
      if (key.startsWith(this.prefix) && this.backend.delete(key)) removed += 1
    }
    return removed
  }

  delete(key: string): boolean {
    return this.backend.delete(this.scoped(key))
  }

  exists(key: string): boolean {
    return this.backend.exists(this.scoped(key))
  }

  listKeys(pattern?: string): string[] {
    const scoped = pattern === undefined ? this.prefix : this.scoped(pattern)
    return this.backend
      .listKeys(scoped)
      .flatMap(key => key.startsWith(this.prefix) ? [key.slice(this.prefix.length)] : [])
  }

  load(key: string): unknown | undefined {
    return this.backend.load(this.scoped(key))
  }

  save(key: string, data: unknown): boolean {
    return this.backend.save(this.scoped(key), data)
  }

  semanticSearch(query: string, limit = 10, threshold = 0): SemanticSearchResult[] {
    // Over-fetch because the backend ranks across every tenant before the
    // namespace filter below narrows the results back to this one.
    return this.backend
      .semanticSearch(query, limit * 4, threshold)
      .flatMap(result => result.key.startsWith(this.prefix)
        ? [{ ...result, key: result.key.slice(this.prefix.length) }]
        : [])
      .slice(0, limit)
  }

  supportsSemanticSearch(): boolean {
    return this.backend.supportsSemanticSearch()
  }

  private scoped(key: string): string {
    return `${this.prefix}${key}`
  }
}

/** JSON-file key/value backend whose hashed filenames prevent key-path traversal. */
export class FileStorage implements MemoryStorage {
  readonly directory: string
  private readonly indexFile: string
  private index: Record<string, string>
  /** Keys this instance removed since it last read the on-disk index. */
  private readonly removed = new Set<string>()

  constructor(directory = '.xerxes_memory') {
    // Normalize once so every later join/slice operates on the same canonical
    // prefix; a raw './mem' or 'mem/' would desynchronize index paths.
    this.directory = resolve(directory)
    mkdirSync(this.directory, { recursive: true })
    this.indexFile = join(this.directory, '_index.json')
    this.index = this.readIndex()
  }

  clear(): number {
    const entries = Object.entries(this.index)
    for (const [key, filename] of entries) {
      const file = join(this.directory, filename)
      if (existsSync(file)) unlinkSync(file)
      this.removed.add(key)
    }
    this.index = {}
    this.writeIndex()
    return entries.length
  }

  delete(key: string): boolean {
    const filename = this.index[key]
    if (!filename) return false
    const file = join(this.directory, filename)
    if (existsSync(file)) unlinkSync(file)
    delete this.index[key]
    this.removed.add(key)
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
    try {
      return JSON.parse(readFileSync(path, 'utf8')) as unknown
    } catch (error) {
      console.warn(`Skipping corrupt memory record ${key}:`, error)
      return undefined
    }
  }

  save(key: string, data: unknown): boolean {
    // Store the plain basename (`<md5>.json`) rather than slicing the joined
    // path by the raw directory length, which corrupted filenames whenever
    // the constructor received a non-normalized directory. Indexes written
    // before this fix already hold basenames for healthy directories, so the
    // on-disk format is unchanged; mangled legacy entries simply miss on
    // load exactly as they did before.
    const filename = this.fileNameForKey(key)
    const path = join(this.directory, filename)
    const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`
    try {
      writeFileSync(temporary, JSON.stringify(data), 'utf8')
      renameSync(temporary, path)
      this.index[key] = filename
      this.removed.delete(key)
      this.writeIndex()
      return true
    } catch {
      rmSync(temporary, { force: true })
      return false
    }
  }

  semanticSearch(_query: string, _limit = 10, _threshold = 0): SemanticSearchResult[] {
    return []
  }

  supportsSemanticSearch(): boolean {
    return false
  }

  private fileNameForKey(key: string): string {
    const hash = createHash('md5').update(key).digest('hex')
    return `${hash}.json`
  }

  private readIndex(): Record<string, string> {
    if (!existsSync(this.indexFile)) return {}
    const parsed = this.readIndexFile()
    if (parsed) return parsed
    // The index is corrupt or wrong-shaped. Back it up instead of silently
    // discarding it, then rebuild by scanning data files. Original keys are
    // unrecoverable (filenames are one-way md5 hashes), so recovered records
    // are exposed under their hash stem: they stay loadable, listable, and
    // deletable by clear() instead of being orphaned.
    const backup = `${this.indexFile}.corrupt-${Date.now()}`
    try {
      renameSync(this.indexFile, backup)
      console.warn(`Backed up corrupt memory index to ${backup}; rebuilding from data files.`)
    } catch (error) {
      console.warn('Could not back up corrupt memory index:', error)
    }
    return this.rebuildIndex()
  }

  private readIndexFile(): Record<string, string> | undefined {
    try {
      const parsed = JSON.parse(readFileSync(this.indexFile, 'utf8')) as unknown
      return isRecord(parsed) && Object.values(parsed).every(value => typeof value === 'string')
        ? parsed as Record<string, string>
        : undefined
    } catch {
      return undefined
    }
  }

  private rebuildIndex(): Record<string, string> {
    const rebuilt: Record<string, string> = {}
    for (const entry of readdirSync(this.directory)) {
      if (!/^[0-9a-f]{32}\.json$/.test(entry)) continue
      rebuilt[entry.slice(0, -'.json'.length)] = entry
    }
    return rebuilt
  }

  private writeIndex(): void {
    // Re-read and merge the on-disk index so two FileStorage instances
    // sharing one directory do not orphan each other's data files: the disk
    // copy contributes keys this instance never saw, this instance's entries
    // win for keys it wrote, and keys it deleted stay deleted.
    const disk = existsSync(this.indexFile) ? this.readIndexFile() : undefined
    const merged: Record<string, string> = { ...disk, ...this.index }
    for (const key of this.removed) delete merged[key]
    const temporary = `${this.indexFile}.${process.pid}.${randomUUID()}.tmp`
    try {
      writeFileSync(temporary, JSON.stringify(merged), 'utf8')
      renameSync(temporary, this.indexFile)
      // Deletions are durable now; drop the tombstones so a key legitimately
      // rewritten by another instance is not suppressed forever.
      this.removed.clear()
    } catch (error) {
      rmSync(temporary, { force: true })
      throw error
    }
  }
}

export interface SQLiteStorageOptions {
  readonly dbPath?: string
  readonly writeEnabled?: boolean
}

interface SQLiteMigration {
  readonly version: number
  readonly apply: (database: Database) => void
}

/**
 * Ordered additive schema migrations tracked through `PRAGMA user_version`.
 * v1 is the original memory-table schema; append new migrations with higher
 * versions instead of editing shipped ones.
 */
const SQLITE_MIGRATIONS: readonly SQLiteMigration[] = [
  {
    version: 1,
    apply: database => {
      database.run(`
        CREATE TABLE IF NOT EXISTS memory (
          key TEXT PRIMARY KEY,
          data TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
      `)
      database.run('CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory(created_at)')
    },
  },
]

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
    migrateSQLiteSchema(this.database)
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
    try {
      return JSON.parse(row.data) as unknown
    } catch (error) {
      console.warn(`Skipping corrupt memory record ${key}:`, error)
      return undefined
    }
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
      let vector: unknown
      try {
        vector = this.backend.load(key)
      } catch (error) {
        console.warn(`Skipping corrupt memory embedding ${key}:`, error)
        continue
      }
      if (!Array.isArray(vector) || !vector.every(value => typeof value === 'number')) continue
      this.embeddings.set(key.slice(RAGStorage.embeddingKeyPrefix.length), [...vector])
    }
  }
}

function migrateSQLiteSchema(database: Database): void {
  const row = database.query('PRAGMA user_version').get() as { user_version?: unknown } | null
  const current = typeof row?.user_version === 'number' ? row.user_version : 0
  for (const migration of [...SQLITE_MIGRATIONS].sort((left, right) => left.version - right.version)) {
    if (migration.version <= current) continue
    migration.apply(database)
    // PRAGMA statements do not accept bind parameters; versions are integer
    // literals defined in this module, so interpolation is safe.
    database.run(`PRAGMA user_version = ${migration.version}`)
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
