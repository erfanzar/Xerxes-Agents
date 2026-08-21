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
  statSync,
  utimesSync,
  writeFileSync,
} from 'node:fs'
import { dirname, join, resolve } from 'node:path'

import { cosineSimilarity, getDefaultEmbedder, type Embedder } from './embedders.js'

export type AccessStateUpdateResult = 'updated' | 'missing' | 'failed'

export interface MemoryStorage {
  clear(): number
  delete(key: string): boolean
  exists(key: string): boolean
  listKeys(pattern?: string): string[]
  load(key: string): unknown | undefined
  save(key: string, data: unknown): boolean
  semanticSearch(query: string, limit?: number, threshold?: number): SemanticSearchResult[]
  supportsSemanticSearch(): boolean
  /** Add an access delta without replacing the rest of a record when the backend supports it. */
  updateAccessState?(key: string, increment: number, lastAccessed: string): AccessStateUpdateResult
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

  updateAccessState(key: string, increment: number, lastAccessed: string): AccessStateUpdateResult {
    const current = this.data.get(key)
    if (!isRecord(current)) return current === undefined ? 'missing' : 'failed'
    const accessCount = typeof current.access_count === 'number' && Number.isInteger(current.access_count)
      ? current.access_count
      : 0
    this.data.set(key, { ...current, access_count: accessCount + increment, last_accessed: lastAccessed })
    return 'updated'
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
    // Ranking happens in the shared backend, so a fixed over-fetch factor can
    // omit this tenant completely when another tenant has enough stronger
    // matches. Ask for the complete backend candidate set, then scope and cap.
    const candidateLimit = this.backend.listKeys().length
    return this.backend
      .semanticSearch(query, candidateLimit, threshold)
      .flatMap(result => result.key.startsWith(this.prefix)
        ? [{ ...result, key: result.key.slice(this.prefix.length) }]
        : [])
      .slice(0, limit)
  }

  supportsSemanticSearch(): boolean {
    return this.backend.supportsSemanticSearch()
  }

  updateAccessState(key: string, increment: number, lastAccessed: string): AccessStateUpdateResult {
    return this.backend.updateAccessState?.(this.scoped(key), increment, lastAccessed) ?? 'failed'
  }

  private scoped(key: string): string {
    return `${this.prefix}${key}`
  }
}

export interface FileStorageOptions {
  readonly lockTimeoutMs?: number
  readonly staleLockMs?: number
}

/**
 * JSON-file key/value backend whose hashed filenames prevent key-path traversal.
 * Read operations refresh the cached index from disk, so multiple long-lived
 * instances over the same directory observe each other's writes.
 */
export class FileStorage implements MemoryStorage {
  readonly directory: string
  private readonly indexFile: string
  private readonly lockTimeoutMs: number
  private readonly staleLockMs: number
  private index: Record<string, string>

  constructor(directory = '.xerxes_memory', options: FileStorageOptions = {}) {
    // Normalize once so every later join/slice operates on the same canonical
    // prefix; a raw './mem' or 'mem/' would desynchronize index paths.
    this.directory = resolve(directory)
    this.lockTimeoutMs = nonNegativeInteger(options.lockTimeoutMs ?? 5_000, 'lockTimeoutMs')
    this.staleLockMs = positiveInteger(options.staleLockMs ?? 30_000, 'staleLockMs')
    mkdirSync(this.directory, { recursive: true })
    this.indexFile = join(this.directory, '_index.json')
    this.index = this.readIndex()
  }

  clear(): number {
    return this.withIndexLock((touchLock) => {
      const current = this.currentIndexUnderLock()
      const moved: Array<{ readonly backup: string; readonly file: string }> = []
      try {
        for (const filename of Object.values(current)) {
          const file = join(this.directory, filename)
          if (!existsSync(file)) continue
          const backup = `${file}.${process.pid}.${randomUUID()}.deleted`
          renameSync(file, backup)
          moved.push({ backup, file })
          touchLock()
        }
        this.persistIndex({}, touchLock)
        this.index = {}
        return Object.keys(current).length
      } catch (error) {
        for (const { backup, file } of moved.reverse()) {
          if (existsSync(backup)) renameSync(backup, file)
        }
        throw error
      } finally {
        // Backup cleanup is non-transactional: once the empty index has been
        // committed, a cleanup failure must not resurrect cleared payloads.
        for (const { backup } of moved) {
          try {
            rmSync(backup, { force: true })
          } catch (cleanupError) {
            console.warn(`Could not remove cleared memory backup ${backup}:`, cleanupError)
          }
        }
      }
    })
  }

  delete(key: string): boolean {
    return this.withIndexLock((touchLock) => {
      const current = this.currentIndexUnderLock()
      const filename = current[key]
      if (!filename) {
        this.index = current
        return false
      }
      const file = join(this.directory, filename)
      const backup = `${file}.${process.pid}.${randomUUID()}.deleted`
      const moved = existsSync(file)
      try {
        if (moved) renameSync(file, backup)
        delete current[key]
        this.persistIndex(current, touchLock)
        this.index = current
        return true
      } catch (error) {
        if (moved && existsSync(backup)) renameSync(backup, file)
        throw error
      } finally {
        // Backup cleanup is non-transactional: once the index commit has
        // succeeded, a cleanup failure must not roll back the deletion.
        if (moved) {
          try {
            rmSync(backup, { force: true })
          } catch (cleanupError) {
            console.warn(`Could not remove deleted memory backup ${backup}:`, cleanupError)
          }
        }
      }
    })
  }

  exists(key: string): boolean {
    this.refreshIndex()
    return key in this.index
  }

  listKeys(pattern?: string): string[] {
    this.refreshIndex()
    const keys = Object.keys(this.index)
    return pattern ? keys.filter(key => key.includes(pattern)) : keys
  }

  load(key: string): unknown | undefined {
    this.refreshIndex()
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
    // The payload and index share one lock transaction. In particular, two
    // same-key writers must not publish payloads around each other's index
    // commits, because every value for a key uses the same hashed filename.
    const filename = this.fileNameForKey(key)
    const path = join(this.directory, filename)
    const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`
    const backup = `${path}.${process.pid}.${randomUUID()}.replaced`
    try {
      return this.withIndexLock((touchLock) => {
        const current = this.currentIndexUnderLock()
        const replaced = existsSync(path)
        try {
          writeFileSync(temporary, JSON.stringify(data), 'utf8')
          touchLock()
          if (replaced) renameSync(path, backup)
          renameSync(temporary, path)
          current[key] = filename
          this.persistIndex(current, touchLock)
          this.index = current
          return true
        } catch (error) {
          rmSync(temporary, { force: true })
          rmSync(path, { force: true })
          if (replaced && existsSync(backup)) renameSync(backup, path)
          throw error
        }
      })
    } catch {
      return false
    } finally {
      // Cleanup is best-effort: a throwing rmSync must not escape because
      // save() promises a boolean result rather than an exception.
      try {
        rmSync(temporary, { force: true })
      } catch {
        // ignore
      }
      try {
        rmSync(backup, { force: true })
      } catch {
        // ignore
      }
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

  private refreshIndex(): void {
    // Read operations must see writes from other processes or long-lived
    // sibling instances. When the persisted index is readable, replace the
    // cached view; if it is absent or corrupt, keep the in-memory view so
    // records recovered from a corrupt index are not lost before the first
    // write persists them.
    if (!existsSync(this.indexFile)) return
    const current = this.readIndexFile()
    if (current) this.index = current
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

  private isStaleLock(lock: string): boolean {
    try {
      return Date.now() - statSync(lock).mtimeMs >= this.staleLockMs
    } catch (error) {
      if (isMissingFileError(error)) return false
      throw error
    }
  }

  private currentIndexUnderLock(): Record<string, string> {
    // A corrupt index is renamed during construction and rebuilt only in
    // memory, so preserve that recovered view until the first transaction
    // persists it. For an ordinary new store this is simply an empty object.
    if (!existsSync(this.indexFile)) return { ...this.index }
    const current = this.readIndexFile()
    if (!current) throw new Error(`Memory index became corrupt: ${this.indexFile}`)
    return current
  }

  private persistIndex(index: Record<string, string>, touchLock?: () => void): void {
    const temporary = `${this.indexFile}.${process.pid}.${randomUUID()}.tmp`
    try {
      writeFileSync(temporary, JSON.stringify(index), 'utf8')
      touchLock?.()
      renameSync(temporary, this.indexFile)
    } finally {
      rmSync(temporary, { force: true })
    }
  }

  private withIndexLock<T>(transaction: (touchLock: () => void) => T): T {
    // Serialize the complete payload+index transaction across processes.
    // Atomic index replacement alone cannot make payload publication safe,
    // especially when concurrent operations target the same hashed filename.
    const lock = `${this.indexFile}.lock`
    const deadline = Date.now() + this.lockTimeoutMs
    while (true) {
      try {
        mkdirSync(lock)
        break
      } catch (error) {
        if (!isFileExistsError(error)) throw error
        if (this.isStaleLock(lock) && this.validateStaleLock(lock)) {
          try {
            rmSync(lock, { force: true, recursive: true })
          } catch (removeError) {
            if (!isMissingFileError(removeError)) throw removeError
          }
          continue
        }
        if (Date.now() >= deadline) throw error
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 5)
      }
    }

    const touchLock = () => {
      const now = new Date()
      try {
        utimesSync(lock, now, now)
      } catch (refreshError) {
        if (!isMissingFileError(refreshError)) throw refreshError
      }
    }

    try {
      touchLock()
      return transaction(touchLock)
    } finally {
      try {
        rmSync(lock, { force: true, recursive: true })
      } catch {
        // Best-effort cleanup; another process may have already removed it.
      }
    }
  }

  private validateStaleLock(lock: string): boolean {
    // A live owner may refresh the lock mtime at the boundary between our
    // stale check and our removal attempt. Observe the mtime once, then wait
    // for a full staleLockMs window while re-checking; only remove the lock
    // if its mtime never changes and it stays stale throughout.
    let firstMtime: number
    let firstSeen: number
    try {
      const stats = statSync(lock)
      firstMtime = stats.mtimeMs
      firstSeen = Date.now()
    } catch (error) {
      return isMissingFileError(error)
    }

    while (true) {
      let currentMtime: number
      try {
        currentMtime = statSync(lock).mtimeMs
      } catch (error) {
        return isMissingFileError(error)
      }
      if (currentMtime !== firstMtime) return false
      if (Date.now() - currentMtime < this.staleLockMs) return false
      if (Date.now() - firstSeen >= this.staleLockMs) return true
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 5)
    }
  }
}

export interface SQLiteStorageOptions {
  /** Maximum time SQLite waits for another connection's write lock. */
  readonly busyTimeoutMs?: number
  readonly dbPath?: string
  readonly writeEnabled?: boolean
}

const DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 5_000

function sqliteBusyTimeout(value: number | undefined): number {
  const timeout = value ?? DEFAULT_SQLITE_BUSY_TIMEOUT_MS
  if (!Number.isSafeInteger(timeout) || timeout < 0) {
    throw new RangeError('busyTimeoutMs must be a non-negative safe integer')
  }
  return timeout
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
    try {
      // Install the handler before schema inspection/migrations, which can
      // themselves require a write lock when another process opens the store.
      this.database.run(`PRAGMA busy_timeout = ${sqliteBusyTimeout(options.busyTimeoutMs)}`)
      this.database.run('PRAGMA journal_mode = WAL')
      migrateSQLiteSchema(this.database)
    } catch (error) {
      this.database.close()
      throw error
    }
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

  updateAccessState(key: string, increment: number, lastAccessed: string): AccessStateUpdateResult {
    if (!this.database) return this.fallback.updateAccessState(key, increment, lastAccessed)
    try {
      this.database.run('BEGIN IMMEDIATE')
      const row = this.database.query('SELECT data FROM memory WHERE key = ?').get(key) as { data?: unknown } | null
      if (!row || typeof row.data !== 'string') {
        this.database.run('ROLLBACK')
        return 'missing'
      }
      const current = JSON.parse(row.data) as unknown
      if (!isRecord(current)) {
        this.database.run('ROLLBACK')
        return 'failed'
      }
      const accessCount = typeof current.access_count === 'number' && Number.isInteger(current.access_count)
        ? current.access_count
        : 0
      const updated = { ...current, access_count: accessCount + increment, last_accessed: lastAccessed }
      this.database.query('UPDATE memory SET data = ?, updated_at = ? WHERE key = ?')
        .run(JSON.stringify(updated), new Date().toISOString(), key)
      this.database.run('COMMIT')
      return 'updated'
    } catch {
      try {
        this.database.run('ROLLBACK')
      } catch {
        // No active transaction remains.
      }
      return 'failed'
    }
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

    // Prepare the derived value before either durable write. Publishing the
    // primary record first can leave it permanently unsearchable when
    // embedding or sidecar persistence fails.
    let embedding: number[]
    try {
      embedding = this.embedder.embed(dataToText(data))
    } catch {
      return false
    }

    const embeddingKey = `${RAGStorage.embeddingKeyPrefix}${key}`
    let hadEmbedding = false
    let previousEmbedding: unknown
    let sidecarWriteAttempted = false
    try {
      hadEmbedding = this.backend.exists(embeddingKey)
      previousEmbedding = hadEmbedding ? this.backend.load(embeddingKey) : undefined
      sidecarWriteAttempted = true
      if (!this.backend.save(embeddingKey, embedding)) return false
      if (!this.backend.save(key, data)) {
        this.restoreEmbeddingSidecar(embeddingKey, hadEmbedding, previousEmbedding)
        return false
      }
    } catch {
      if (sidecarWriteAttempted) {
        this.restoreEmbeddingSidecar(embeddingKey, hadEmbedding, previousEmbedding)
      }
      return false
    }

    // Do not expose the new vector in-process until both durable writes have
    // succeeded. A failed save therefore keeps the prior searchable view.
    this.embeddings.set(key, embedding)
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

  updateAccessState(key: string, increment: number, lastAccessed: string): AccessStateUpdateResult {
    if (key.startsWith(RAGStorage.embeddingKeyPrefix)) return 'failed'
    const result = this.backend.updateAccessState?.(key, increment, lastAccessed) ?? 'failed'
    if (result !== 'updated') return result
    const data = this.backend.load(key)
    if (data !== undefined) {
      const embedding = this.embedder.embed(dataToText(data))
      this.embeddings.set(key, embedding)
      this.backend.save(`${RAGStorage.embeddingKeyPrefix}${key}`, embedding)
    }
    return result
  }

  private restoreEmbeddingSidecar(key: string, existed: boolean, previous: unknown): void {
    try {
      if (existed) {
        this.backend.save(key, previous)
      } else {
        this.backend.delete(key)
      }
    } catch {
      // Best-effort compensation only: MemoryStorage has no transaction or
      // conditional-write contract, so a backend failure may be irreversible.
    }
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

function isFileExistsError(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'EEXIST'
}

function isMissingFileError(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function nonNegativeInteger(value: number, field: string): number {
  if (!Number.isInteger(value) || value < 0) throw new RangeError(`${field} must be a non-negative integer`)
  return value
}

function positiveInteger(value: number, field: string): number {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${field} must be a positive integer`)
  return value
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
