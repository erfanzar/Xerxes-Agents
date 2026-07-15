// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { homedir } from 'node:os'
import { mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'

import { cosineSimilarity, getDefaultEmbedder, type Embedder } from './embedders.js'
import type { MemoryStorage, SemanticSearchResult } from './storage.js'

const DEFAULT_DATABASE_PATH = '.xerxes_memory/vectors.db'
const BYTE_MARKER_TYPE = 'bytes'

interface ByteMarker {
  readonly [key: string]: JsonValue
  readonly __type__: typeof BYTE_MARKER_TYPE
  readonly data: string
}

type JsonValue = boolean | JsonObject | JsonValue[] | null | number | string
type JsonObject = { readonly [key: string]: JsonValue }

interface VectorRow {
  readonly data: unknown
  readonly embedding: unknown
  readonly key: unknown
}

export interface SQLiteVectorStorageOptions {
  /** SQLite database file. The default lives beneath the current workspace. */
  readonly dbPath?: string
  /** Encoder used for stored text and semantic-search queries. */
  readonly embedder?: Embedder
}

/**
 * Bun-native SQLite key/value store with durable dense embeddings.
 *
 * Vectors intentionally live as JSON in the same file as the persisted value,
 * making semantic retrieval portable without an SQLite vector extension. This
 * is suitable for the small-to-medium stores where a deterministic cosine scan
 * is less complex than operating an external vector database.
 */
export class SQLiteVectorStorage implements MemoryStorage {
  readonly dbPath: string
  readonly embedder: Embedder
  private closed = false
  private readonly database: Database

  constructor(options?: SQLiteVectorStorageOptions)
  constructor(dbPath?: string, embedder?: Embedder)
  constructor(optionsOrPath: SQLiteVectorStorageOptions | string | undefined = {}, positionalEmbedder?: Embedder) {
    const options = typeof optionsOrPath === 'string'
      ? { dbPath: optionsOrPath, ...(positionalEmbedder === undefined ? {} : { embedder: positionalEmbedder }) }
      : { ...(optionsOrPath ?? {}), ...(positionalEmbedder === undefined ? {} : { embedder: positionalEmbedder }) }
    this.dbPath = expandHome(options.dbPath ?? DEFAULT_DATABASE_PATH)
    this.embedder = options.embedder ?? getDefaultEmbedder()
    if (this.dbPath !== ':memory:') mkdirSync(dirname(this.dbPath), { recursive: true })
    this.database = new Database(this.dbPath)
    this.initialize()
  }

  clear(): number {
    this.assertOpen()
    const row = this.database.query('SELECT COUNT(*) AS count FROM vectors').get() as { count?: unknown } | null
    this.database.run('DELETE FROM vectors')
    return numberCount(row?.count)
  }

  /** Release the owned SQLite connection. Calling this more than once is safe. */
  close(): void {
    if (this.closed) return
    this.database.close()
    this.closed = true
  }

  delete(key: string): boolean {
    this.assertOpen()
    const result = this.database.query('DELETE FROM vectors WHERE key = ?').run(key)
    return result.changes > 0
  }

  exists(key: string): boolean {
    this.assertOpen()
    return this.database.query('SELECT 1 FROM vectors WHERE key = ? LIMIT 1').get(key) !== null
  }

  listKeys(pattern?: string): string[] {
    this.assertOpen()
    const statement = pattern === undefined || pattern.length === 0
      ? this.database.query('SELECT key FROM vectors ORDER BY created_at DESC, key ASC')
      : this.database.query("SELECT key FROM vectors WHERE key LIKE ? ESCAPE '\\' ORDER BY created_at DESC, key ASC")
    const rows = (pattern === undefined || pattern.length === 0
      ? statement.all()
      : statement.all(`%${escapeLike(pattern)}%`)) as unknown as Array<{ key?: unknown }>
    return rows.flatMap(row => typeof row.key === 'string' ? [row.key] : [])
  }

  load(key: string): unknown | undefined {
    this.assertOpen()
    const row = this.database.query('SELECT data FROM vectors WHERE key = ?').get(key) as { data?: unknown } | null
    return row ? deserialize(row.data) : undefined
  }

  save(key: string, data: unknown): boolean {
    this.assertOpen()
    let payload: string
    try {
      payload = serialize(data)
    } catch {
      return false
    }

    const embedding = this.embeddingFor(data)
    try {
      this.database.query(`
        INSERT INTO vectors (key, data, embedding, created_at) VALUES (?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
          data = excluded.data,
          embedding = excluded.embedding,
          created_at = excluded.created_at
      `).run(key, Buffer.from(payload, 'utf8'), JSON.stringify(embedding), new Date().toISOString())
      return true
    } catch {
      return false
    }
  }

  semanticSearch(query: string, limit = 10, threshold = 0): SemanticSearchResult[] {
    this.assertOpen()
    if (query.length === 0 || limit <= 0) return []
    let queryVector: number[]
    try {
      queryVector = validVector(this.embedder.embed(query)) ?? []
    } catch {
      return []
    }
    if (queryVector.length === 0) return []

    const results: SemanticSearchResult[] = []
    const rows = this.database
      .query('SELECT key, data, embedding FROM vectors ORDER BY key ASC')
      .all() as unknown as VectorRow[]
    for (const row of rows) {
      if (typeof row.key !== 'string') continue
      const embedding = parseEmbedding(row.embedding)
      if (!embedding) continue
      const similarity = cosineSimilarity(queryVector, embedding)
      if (similarity < threshold) continue
      const data = deserialize(row.data)
      if (data === undefined) continue
      results.push({ key: row.key, similarity, data })
    }
    return results.sort(compareSearchResults).slice(0, limit)
  }

  supportsSemanticSearch(): boolean {
    return true
  }

  private embeddingFor(data: unknown): number[] {
    const text = embeddingText(data)
    if (text.length === 0) return zeroVector(this.embedder)
    try {
      return validVector(this.embedder.embed(text)) ?? zeroVector(this.embedder)
    } catch {
      return zeroVector(this.embedder)
    }
  }

  private initialize(): void {
    this.database.run(`
      CREATE TABLE IF NOT EXISTS vectors (
        key TEXT PRIMARY KEY,
        data BLOB NOT NULL,
        embedding TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    `)
    this.database.run('CREATE INDEX IF NOT EXISTS idx_vectors_created ON vectors(created_at)')
  }

  private assertOpen(): void {
    if (this.closed) throw new Error('SQLiteVectorStorage is closed')
  }
}

function compareSearchResults(left: SemanticSearchResult, right: SemanticSearchResult): number {
  if (left.similarity !== right.similarity) return right.similarity - left.similarity
  return compareStrings(left.key, right.key)
}

function compareStrings(left: string, right: string): number {
  if (left < right) return -1
  if (left > right) return 1
  return 0
}

function deserialize(raw: unknown): unknown | undefined {
  const text = databaseText(raw)
  if (text === undefined) return undefined
  try {
    return decodeValue(JSON.parse(text) as JsonValue)
  } catch {
    return undefined
  }
}

function databaseText(value: unknown): string | undefined {
  if (typeof value === 'string') return value
  if (value instanceof Uint8Array) return Buffer.from(value).toString('utf8')
  return undefined
}

function decodeValue(value: JsonValue): unknown {
  if (isByteMarker(value)) return new Uint8Array(Buffer.from(value.data, 'base64'))
  if (Array.isArray(value)) return value.map(item => decodeValue(item))
  if (!isJsonObject(value)) return value
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, decodeValue(item)]))
}

function embeddingText(value: unknown): string {
  if (typeof value === 'string') return value
  if (!isRecord(value)) return ''
  try {
    return serialize(value)
  } catch {
    return ''
  }
}

function escapeLike(value: string): string {
  return value.replaceAll('\\', '\\\\').replaceAll('%', '\\%').replaceAll('_', '\\_')
}

function expandHome(path: string): string {
  if (path === '~') return homedir()
  return path.startsWith('~/') ? join(homedir(), path.slice(2)) : path
}

function isByteMarker(value: JsonValue): value is ByteMarker {
  if (!isJsonObject(value) || value.__type__ !== BYTE_MARKER_TYPE || typeof value.data !== 'string') return false
  const keys = Object.keys(value)
  return keys.length === 2 && keys.includes('__type__') && keys.includes('data')
}

function isRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function isJsonObject(value: JsonValue): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function numberCount(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function parseEmbedding(value: unknown): number[] | undefined {
  if (typeof value !== 'string') return undefined
  try {
    const parsed = JSON.parse(value) as unknown
    return Array.isArray(parsed) ? validVector(parsed) : undefined
  } catch {
    return undefined
  }
}

function serialize(value: unknown): string {
  return JSON.stringify(encodeValue(value))
}

function encodeValue(value: unknown, seen = new WeakSet<object>()): JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError('Only finite numbers can be stored as JSON')
    return value
  }
  if (value instanceof Uint8Array) return byteMarker(value)
  if (value instanceof ArrayBuffer) return byteMarker(new Uint8Array(value))
  if (Array.isArray(value)) {
    if (seen.has(value)) throw new TypeError('Circular values cannot be stored')
    seen.add(value)
    const encoded = value.map(item => encodeValue(item, seen))
    seen.delete(value)
    return encoded
  }
  if (!isRecord(value)) throw new TypeError(`Unsupported storage value: ${typeof value}`)
  if (seen.has(value)) throw new TypeError('Circular values cannot be stored')
  seen.add(value)
  const encoded: Record<string, JsonValue> = {}
  for (const key of Object.keys(value).sort(compareStrings)) encoded[key] = encodeValue(value[key], seen)
  seen.delete(value)
  return encoded
}

function byteMarker(bytes: Uint8Array): ByteMarker {
  return { __type__: BYTE_MARKER_TYPE, data: Buffer.from(bytes).toString('base64') }
}

function validVector(value: readonly number[]): number[] | undefined {
  if (value.length === 0 || !value.every(item => typeof item === 'number' && Number.isFinite(item))) return undefined
  return [...value]
}

function zeroVector(embedder: Embedder): number[] {
  const dimension = Number.isInteger(embedder.dimension) && embedder.dimension > 0 ? embedder.dimension : 1
  return Array<number>(dimension).fill(0)
}
