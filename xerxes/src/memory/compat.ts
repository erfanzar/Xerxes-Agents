// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolve } from 'node:path'

import {
  MemoryItem,
  type MemoryItemOptions,
  type MemoryItemRecord,
  type MemoryMetadata,
  type MemorySaveOptions,
  type MemoryStatistics,
  type MemoryUpdate,
} from './base.js'
import { ContextualMemory } from './contextualMemory.js'
import { HashEmbedder } from './embedders.js'
import { LongTermMemory } from './longTermMemory.js'
import { ShortTermMemory } from './shortTermMemory.js'
import { RAGStorage, SQLiteStorage, SimpleStorage, type MemoryStorage } from './storage.js'

/** Typed categories retained from the original bucketed memory API. */
export const MemoryType = Object.freeze({
  SHORT_TERM: 'short_term',
  LONG_TERM: 'long_term',
  EPISODIC: 'episodic',
  SEMANTIC: 'semantic',
  WORKING: 'working',
  PROCEDURAL: 'procedural',
} as const)

export type MemoryType = (typeof MemoryType)[keyof typeof MemoryType]

export const LEGACY_MEMORY_TYPES = Object.freeze([
  MemoryType.SHORT_TERM,
  MemoryType.LONG_TERM,
  MemoryType.EPISODIC,
  MemoryType.SEMANTIC,
  MemoryType.WORKING,
  MemoryType.PROCEDURAL,
] as const)

const STORE_MARKER = 'xerxes.memory.compat'
const STORE_MEMORY_TYPE = 'xerxes.memory.compat_type'
const UNBOUNDED_TIER_CAPACITY = Number.MAX_SAFE_INTEGER

export interface MemoryEntryOptions extends Omit<MemoryItemOptions, 'memoryType' | 'metadata'> {
  readonly context?: Readonly<Record<string, unknown>>
  readonly importanceScore?: number
  readonly memoryType?: MemoryType
  readonly metadata?: MemoryMetadata
  readonly tags?: readonly string[]
}

export interface MemoryEntryRecord extends MemoryItemRecord {
  readonly context: MemoryMetadata
  readonly importance_score: number
  readonly memory_type: MemoryType
  readonly tags: readonly string[]
}

/** Thrown when a compatibility-memory value is malformed. */
export class MemoryCompatibilityError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'MemoryCompatibilityError'
  }
}

/** Thrown when a memory-store configuration could make persistence ambiguous or unsafe. */
export class MemoryStoreConfigurationError extends MemoryCompatibilityError {
  constructor(message: string) {
    super(message)
    this.name = 'MemoryStoreConfigurationError'
  }
}

/**
 * Typed facade over one core MemoryItem.
 *
 * Its contextual fields are stored in the item's metadata, so a MemoryEntry
 * returned by MemoryStore remains backed by the actual short- or long-term tier.
 */
export class MemoryEntry {
  private item: MemoryItem

  constructor(options: MemoryEntryOptions) {
    const memoryType = requireMemoryType(options.memoryType ?? MemoryType.SHORT_TERM)
    const timestamp = options.timestamp === undefined ? undefined : requireDate(options.timestamp, 'timestamp')
    const lastAccessed = options.lastAccessed === undefined
      ? undefined
      : requireDate(options.lastAccessed, 'lastAccessed')
    const metadata = normalizeEntryMetadata(options.metadata, options.context, options.tags, options.importanceScore)
    const {
      context: _context,
      importanceScore: _importanceScore,
      lastAccessed: _lastAccessed,
      memoryType: _memoryType,
      metadata: _metadata,
      tags: _tags,
      timestamp: _timestamp,
      ...itemOptions
    } = options
    this.item = new MemoryItem({
      ...itemOptions,
      memoryType,
      metadata,
      ...(timestamp ? { timestamp } : {}),
      ...(lastAccessed ? { lastAccessed } : {}),
    })
  }

  static fromMemoryItem(item: MemoryItem): MemoryEntry {
    const memoryType = requireMemoryType(item.memoryType)
    const entry = new MemoryEntry({ content: item.content, memoryType })
    entry.item = item
    entry.item.metadata = normalizeStoredMetadata(item.metadata)
    return entry
  }

  get accessCount(): number {
    return this.item.accessCount
  }

  set accessCount(value: number) {
    this.item.accessCount = requireNonNegativeInteger(value, 'accessCount')
  }

  get agentId(): string | undefined {
    return this.item.agentId
  }

  set agentId(value: string | undefined) {
    this.item.agentId = value
  }

  get content(): string {
    return this.item.content
  }

  set content(value: string) {
    if (typeof value !== 'string') throw new MemoryCompatibilityError('content must be a string')
    this.item.content = value
  }

  get context(): MemoryMetadata {
    const context = this.item.metadata.context
    if (isRecord(context)) return context
    const next: MemoryMetadata = {}
    this.item.metadata.context = next
    return next
  }

  set context(value: Readonly<Record<string, unknown>>) {
    this.item.metadata.context = copyContext(value)
  }

  get conversationId(): string | undefined {
    return this.item.conversationId
  }

  set conversationId(value: string | undefined) {
    this.item.conversationId = value
  }

  get embedding(): number[] | undefined {
    return this.item.embedding
  }

  set embedding(value: readonly number[] | undefined) {
    this.item.embedding = value === undefined ? undefined : [...value]
  }

  get lastAccessed(): Date | undefined {
    return this.item.lastAccessed
  }

  set lastAccessed(value: Date | undefined) {
    this.item.lastAccessed = value === undefined ? undefined : requireDate(value, 'lastAccessed')
  }

  get memoryId(): string {
    return this.item.memoryId
  }

  get memoryType(): MemoryType {
    return requireMemoryType(this.item.memoryType)
  }

  set memoryType(value: MemoryType) {
    this.item.memoryType = requireMemoryType(value)
  }

  get metadata(): MemoryMetadata {
    return this.item.metadata
  }

  set metadata(value: MemoryMetadata) {
    this.item.metadata = normalizeStoredMetadata(value)
  }

  get importanceScore(): number {
    const value = this.item.metadata.importance
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? value : 0.5
  }

  set importanceScore(value: number) {
    this.item.metadata.importance = requireImportance(value)
  }

  get relevanceScore(): number {
    return this.item.relevanceScore
  }

  set relevanceScore(value: number) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new MemoryCompatibilityError('relevanceScore must be a finite number')
    }
    this.item.relevanceScore = value
  }

  get tags(): string[] {
    const tags = this.item.metadata.tags
    if (Array.isArray(tags) && tags.every(tag => typeof tag === 'string')) return tags
    const next: string[] = []
    this.item.metadata.tags = next
    return next
  }

  set tags(value: readonly string[]) {
    this.item.metadata.tags = copyTags(value)
  }

  get taskId(): string | undefined {
    return this.item.taskId
  }

  set taskId(value: string | undefined) {
    this.item.taskId = value
  }

  get timestamp(): Date {
    return this.item.timestamp
  }

  set timestamp(value: Date) {
    this.item.timestamp = requireDate(value, 'timestamp')
  }

  get userId(): string | undefined {
    return this.item.userId
  }

  set userId(value: string | undefined) {
    this.item.userId = value
  }

  toMemoryItem(): MemoryItem {
    return this.item
  }

  toRecord(): MemoryEntryRecord {
    return {
      ...this.item.toRecord(),
      memory_type: this.memoryType,
      context: { ...this.context },
      importance_score: this.importanceScore,
      tags: [...this.tags],
    }
  }

  touch(now = new Date()): void {
    this.item.touch(requireDate(now, 'now'))
  }
}

export interface MemoryStorePersistenceOptions {
  /** SQLite destination used only when writeEnabled is explicitly true. */
  readonly path: string
  /** Defaults to false so merely supplying a path cannot create or modify a database. */
  readonly writeEnabled?: boolean
}

export interface MemoryStoreOptions {
  readonly cacheSize?: number
  readonly defaultMemoryType?: MemoryType
  readonly embeddingDimension?: number
  readonly enableVectorSearch?: boolean
  readonly maxLongTerm?: number
  readonly maxShortTerm?: number
  readonly maxWorking?: number
  readonly persistence?: MemoryStorePersistenceOptions
  /** Optional dedicated backend for tests or application-owned storage. */
  readonly storage?: MemoryStorage
}

export type MemoryStoreAddOptions = Omit<MemoryEntryOptions, 'memoryId'> & {
  readonly agentId: string
}

export interface MemoryStoreRetrieveOptions {
  readonly agentId?: string
  readonly limit?: number
  readonly memoryType?: MemoryType
  readonly memoryTypes?: readonly MemoryType[]
  readonly minImportance?: number
  readonly tags?: readonly string[]
}

export interface MemoryStoreRecentOptions {
  readonly minutesAgo?: number
  readonly now?: Date
}

export interface MemoryStoreClearOptions {
  readonly agentId?: string
  readonly memoryType?: MemoryType
}

export interface MemoryStoreConsolidateOptions {
  readonly agentId?: string
  readonly threshold?: number
}

export interface MemoryStoreStatistics extends MemoryStatistics {
  readonly cacheHitRate: number
  readonly totalMemories: number
}

/**
 * Modern TypeScript facade for Xerxes' typed legacy memory buckets.
 *
 * Each entry is saved in the current short- or long-term tier; `memories` is
 * only the category index required by the legacy API, not a second store.
 */
export class MemoryStore {
  readonly cacheSize: number
  readonly contextualMemory: ContextualMemory
  readonly defaultMemoryType: MemoryType
  readonly embeddingDimension: number
  readonly enableVectorSearch: boolean
  readonly maxLongTerm: number
  readonly maxShortTerm: number
  readonly maxWorking: number
  readonly memories: Record<MemoryType, MemoryEntry[]>
  readonly persistenceEnabled: boolean
  private readonly sqliteStorage: SQLiteStorage | undefined

  constructor(options: MemoryStoreOptions = {}) {
    this.maxShortTerm = requirePositiveInteger(options.maxShortTerm ?? 100, 'maxShortTerm')
    this.maxWorking = requirePositiveInteger(options.maxWorking ?? 10, 'maxWorking')
    this.maxLongTerm = requirePositiveInteger(options.maxLongTerm ?? 10_000, 'maxLongTerm')
    this.cacheSize = requirePositiveInteger(options.cacheSize ?? 100, 'cacheSize')
    this.embeddingDimension = requirePositiveInteger(options.embeddingDimension ?? 768, 'embeddingDimension')
    this.enableVectorSearch = options.enableVectorSearch ?? false
    this.defaultMemoryType = requireMemoryType(options.defaultMemoryType ?? MemoryType.SHORT_TERM)

    const setup = createStorageSetup(options, this.enableVectorSearch, this.embeddingDimension)
    this.sqliteStorage = setup.sqliteStorage
    this.persistenceEnabled = setup.persistenceEnabled
    const longTerm = new LongTermMemory({
      enableEmbeddings: this.enableVectorSearch,
      maxItems: UNBOUNDED_TIER_CAPACITY,
      storage: setup.storage,
    })
    this.contextualMemory = new ContextualMemory({
      importanceThreshold: 0.7,
      longTerm,
      promotionThreshold: 3,
      shortTermCapacity: UNBOUNDED_TIER_CAPACITY,
    })
    this.memories = createBuckets()
    this.hydratePersistedEntries()
  }

  get longTerm(): LongTermMemory {
    return this.contextualMemory.longTerm
  }

  get shortTerm(): ShortTermMemory {
    return this.contextualMemory.shortTerm
  }

  get size(): number {
    return this.entries().length
  }

  /** Add an entry to its typed bucket and corresponding core memory tier. */
  addMemory(options: MemoryStoreAddOptions): MemoryEntry {
    if (typeof options.agentId !== 'string' || !options.agentId.trim()) {
      throw new MemoryCompatibilityError('agentId must be a non-empty string')
    }
    const memoryType = requireMemoryType(options.memoryType ?? this.defaultMemoryType)
    const draft = new MemoryEntry({ ...options, memoryType })
    const entry = this.persistEntry(draft, memoryType)
    this.memories[memoryType].push(entry)
    this.enforceLimit(memoryType)
    return entry
  }

  /** Close the SQLite handle created by explicit persistent configuration. */
  close(): void {
    this.sqliteStorage?.close()
  }

  /** Remove selected typed entries from both the bucket index and their core tier. */
  clear(): void {
    this.clearMemories()
  }

  clearMemories(options: MemoryStoreClearOptions = {}): void {
    const memoryTypes = options.memoryType === undefined ? LEGACY_MEMORY_TYPES : [requireMemoryType(options.memoryType)]
    for (const memoryType of memoryTypes) {
      const bucket = this.memories[memoryType]
      const retained: MemoryEntry[] = []
      for (const entry of bucket) {
        if (options.agentId === undefined || entry.agentId === options.agentId) {
          this.tierFor(memoryType).delete(entry.memoryId)
        } else {
          retained.push(entry)
        }
      }
      bucket.splice(0, bucket.length, ...retained)
    }
  }

  /** Promote important short, working, and episodic entries into long-term memory. */
  consolidateMemories(options: MemoryStoreConsolidateOptions = {}): string {
    const threshold = requireImportance(options.threshold ?? 0.7)
    const promoted = this.memories[MemoryType.LONG_TERM]
    for (const sourceType of [MemoryType.SHORT_TERM, MemoryType.WORKING, MemoryType.EPISODIC] as const) {
      const source = this.memories[sourceType]
      const retained: MemoryEntry[] = []
      for (const entry of source) {
        if (entry.importanceScore >= threshold && (options.agentId === undefined || entry.agentId === options.agentId)) {
          this.tierFor(sourceType).delete(entry.memoryId)
          promoted.push(this.persistEntry(entry, MemoryType.LONG_TERM))
        } else {
          retained.push(entry)
        }
      }
      source.splice(0, source.length, ...retained)
    }
    this.enforceLimit(MemoryType.LONG_TERM)

    const relevant = this.retrieveMemories({ ...(options.agentId ? { agentId: options.agentId } : {}), limit: 20 })
    if (relevant.length === 0) return ''
    const important = relevant.filter(entry => entry.importanceScore >= threshold).slice(0, 5)
    const lines: string[] = []
    if (important.length > 0) {
      lines.push('Important facts:')
      lines.push(...important.map(entry => `- [${entry.importanceScore.toFixed(1)}] ${entry.content}`))
    }
    const recent = relevant.filter(entry => !important.includes(entry)).slice(0, 5)
    if (recent.length > 0) {
      if (lines.length > 0) lines.push('')
      lines.push('Recent context:')
      lines.push(...recent.map(entry => `- ${entry.content}`))
    }
    return lines.join('\n')
  }

  /** Return aggregate statistics for the typed legacy bucket view. */
  getStatistics(): MemoryStoreStatistics {
    const entries = this.entries()
    const memoryTypes: Record<string, number> = {}
    const agents = new Set<string>()
    const users = new Set<string>()
    const conversations = new Set<string>()
    for (const entry of entries) {
      memoryTypes[entry.memoryType] = (memoryTypes[entry.memoryType] ?? 0) + 1
      if (entry.agentId) agents.add(entry.agentId)
      if (entry.userId) users.add(entry.userId)
      if (entry.conversationId) conversations.add(entry.conversationId)
    }
    return {
      cacheHitRate: 0,
      maxItems: undefined,
      memoryTypes,
      totalItems: entries.length,
      totalMemories: entries.length,
      uniqueAgents: agents.size,
      uniqueConversations: conversations.size,
      uniqueUsers: users.size,
    }
  }

  /** Select bucket entries by category, owner, tags, importance, and recency. */
  retrieveMemories(options: MemoryStoreRetrieveOptions = {}): MemoryEntry[] {
    const limit = requireLimit(options.limit ?? 10)
    const minImportance = requireImportance(options.minImportance ?? 0)
    const requestedTypes = options.memoryTypes
      ?? (options.memoryType === undefined ? LEGACY_MEMORY_TYPES : [options.memoryType])
    const memoryTypes = [...new Set(requestedTypes.map(requireMemoryType))]
    const requestedTags = options.tags === undefined ? undefined : copyTags(options.tags)
    const matches = memoryTypes.flatMap(memoryType => this.memories[memoryType]).filter(entry => {
      if (options.agentId !== undefined && entry.agentId !== options.agentId) return false
      if (entry.importanceScore < minImportance) return false
      return requestedTags === undefined || requestedTags.some(tag => entry.tags.includes(tag))
    })
    return matches.sort(compareEntriesByRecency).slice(0, limit)
  }

  /** Return entries created inside the requested recent time window. */
  retrieveRecent(options: MemoryStoreRecentOptions = {}): MemoryEntry[] {
    const minutesAgo = options.minutesAgo ?? 60
    if (typeof minutesAgo !== 'number' || !Number.isFinite(minutesAgo) || minutesAgo < 0) {
      throw new MemoryCompatibilityError('minutesAgo must be a non-negative finite number')
    }
    const now = options.now === undefined ? new Date() : requireDate(options.now, 'now')
    const cutoff = now.valueOf() - minutesAgo * 60_000
    return this.entries().filter(entry => entry.timestamp.valueOf() >= cutoff).sort(compareEntriesByRecency)
  }

  private enforceLimit(memoryType: MemoryType): void {
    const limit = memoryType === MemoryType.SHORT_TERM
      ? this.maxShortTerm
      : memoryType === MemoryType.WORKING
        ? this.maxWorking
        : memoryType === MemoryType.LONG_TERM
          ? this.maxLongTerm
          : undefined
    if (limit === undefined) return
    const bucket = this.memories[memoryType]
    while (bucket.length > limit) {
      const removed = bucket.shift()
      if (removed) this.tierFor(memoryType).delete(removed.memoryId)
    }
  }

  private entries(): MemoryEntry[] {
    return LEGACY_MEMORY_TYPES.flatMap(memoryType => this.memories[memoryType])
  }

  private hydratePersistedEntries(): void {
    // mostImportant is read-only: boot hydration must not touch access state or
    // trigger one re-persist/re-embed write per restored entry. Reversed to
    // ascending importance so enforceLimit evicts the least valuable entries.
    for (const item of this.longTerm.mostImportant(UNBOUNDED_TIER_CAPACITY).reverse()) {
      const memoryType = persistedMemoryType(item)
      if (!memoryType || !isLongTermType(memoryType)) continue
      this.memories[memoryType].push(MemoryEntry.fromMemoryItem(item))
    }
    this.enforceLimit(MemoryType.LONG_TERM)
  }

  private persistEntry(entry: MemoryEntry, memoryType: MemoryType): MemoryEntry {
    const metadata: MemoryMetadata = {
      ...entry.metadata,
      context: { ...entry.context },
      tags: [...entry.tags],
      importance: entry.importanceScore,
      [STORE_MARKER]: true,
      [STORE_MEMORY_TYPE]: memoryType,
    }
    const tier = this.tierFor(memoryType)
    const item = tier.save(entry.content, metadata, saveOptionsFor(entry))
    const updates: MemoryUpdate = {
      accessCount: entry.accessCount,
      memoryType,
      metadata,
      relevanceScore: entry.relevanceScore,
      timestamp: entry.timestamp,
      ...(entry.embedding ? { embedding: entry.embedding } : {}),
      ...(entry.lastAccessed ? { lastAccessed: entry.lastAccessed } : {}),
    }
    tier.update(item.memoryId, updates)
    return MemoryEntry.fromMemoryItem(item)
  }

  private tierFor(memoryType: MemoryType): LongTermMemory | ShortTermMemory {
    return isLongTermType(memoryType) ? this.longTerm : this.shortTerm
  }
}

interface StorageSetup {
  readonly persistenceEnabled: boolean
  readonly sqliteStorage: SQLiteStorage | undefined
  readonly storage: MemoryStorage
}

function compareEntriesByRecency(left: MemoryEntry, right: MemoryEntry): number {
  return right.timestamp.valueOf() - left.timestamp.valueOf() || left.memoryId.localeCompare(right.memoryId)
}

function copyContext(value: Readonly<Record<string, unknown>> | undefined): MemoryMetadata {
  if (value === undefined) return {}
  if (!isRecord(value)) throw new MemoryCompatibilityError('context must be an object')
  return { ...value }
}

function copyTags(value: readonly string[] | undefined): string[] {
  if (value === undefined) return []
  if (!Array.isArray(value) || !value.every(tag => typeof tag === 'string')) {
    throw new MemoryCompatibilityError('tags must be an array of strings')
  }
  return [...value]
}

function createBuckets(): Record<MemoryType, MemoryEntry[]> {
  return {
    [MemoryType.SHORT_TERM]: [],
    [MemoryType.LONG_TERM]: [],
    [MemoryType.EPISODIC]: [],
    [MemoryType.SEMANTIC]: [],
    [MemoryType.WORKING]: [],
    [MemoryType.PROCEDURAL]: [],
  }
}

function createStorageSetup(
  options: MemoryStoreOptions,
  enableVectorSearch: boolean,
  embeddingDimension: number,
): StorageSetup {
  if (options.storage && options.persistence) {
    throw new MemoryStoreConfigurationError('storage and persistence cannot be configured together')
  }

  let storage = options.storage ?? new SimpleStorage()
  let sqliteStorage: SQLiteStorage | undefined
  let persistenceEnabled = false
  if (options.persistence?.writeEnabled === true) {
    const path = options.persistence.path
    if (typeof path !== 'string' || !path.trim() || path.includes('\0')) {
      throw new MemoryStoreConfigurationError('persistence.path must be a non-empty filesystem path')
    }
    sqliteStorage = new SQLiteStorage({ dbPath: resolve(path), writeEnabled: true })
    storage = sqliteStorage
    persistenceEnabled = true
  }
  if (enableVectorSearch && !storage.supportsSemanticSearch()) {
    storage = new RAGStorage(storage, new HashEmbedder(embeddingDimension))
  }
  return { persistenceEnabled, sqliteStorage, storage }
}

function isLongTermType(memoryType: MemoryType): boolean {
  return memoryType === MemoryType.LONG_TERM
    || memoryType === MemoryType.SEMANTIC
    || memoryType === MemoryType.PROCEDURAL
}

function isMemoryType(value: unknown): value is MemoryType {
  return typeof value === 'string' && (LEGACY_MEMORY_TYPES as readonly string[]).includes(value)
}

function isRecord(value: unknown): value is MemoryMetadata {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function normalizeEntryMetadata(
  metadata: MemoryMetadata | undefined,
  context: Readonly<Record<string, unknown>> | undefined,
  tags: readonly string[] | undefined,
  importanceScore: number | undefined,
): MemoryMetadata {
  const normalized = normalizeStoredMetadata(metadata ?? {})
  normalized.context = copyContext(context)
  normalized.tags = copyTags(tags)
  normalized.importance = requireImportance(importanceScore ?? 0.5)
  return normalized
}

function normalizeStoredMetadata(metadata: MemoryMetadata): MemoryMetadata {
  if (!isRecord(metadata)) throw new MemoryCompatibilityError('metadata must be an object')
  const context = isRecord(metadata.context) ? { ...metadata.context } : {}
  const tags = Array.isArray(metadata.tags) && metadata.tags.every(tag => typeof tag === 'string')
    ? [...metadata.tags]
    : []
  const importance = typeof metadata.importance === 'number' && Number.isFinite(metadata.importance)
    && metadata.importance >= 0 && metadata.importance <= 1
    ? metadata.importance
    : 0.5
  return { ...metadata, context, tags, importance }
}

function persistedMemoryType(item: MemoryItem): MemoryType | undefined {
  if (item.metadata[STORE_MARKER] !== true) return undefined
  const memoryType = item.metadata[STORE_MEMORY_TYPE]
  return isMemoryType(memoryType) ? memoryType : undefined
}

function requireDate(value: Date, name: string): Date {
  if (!(value instanceof Date) || Number.isNaN(value.valueOf())) {
    throw new MemoryCompatibilityError(`${name} must be a valid Date`)
  }
  return value
}

function requireImportance(value: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new MemoryCompatibilityError('importanceScore must be a finite number from zero through one')
  }
  return value
}

function requireLimit(value: number): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new MemoryCompatibilityError('limit must be a non-negative integer')
  }
  return value
}

function requireMemoryType(value: unknown): MemoryType {
  if (!isMemoryType(value)) {
    throw new MemoryCompatibilityError(`unknown memory type: ${String(value)}`)
  }
  return value
}

function requireNonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new MemoryCompatibilityError(`${name} must be a non-negative integer`)
  }
  return value
}

function requirePositiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new MemoryStoreConfigurationError(`${name} must be a positive integer`)
  }
  return value
}

function saveOptionsFor(entry: MemoryEntry): MemorySaveOptions {
  return {
    importance: entry.importanceScore,
    ...(entry.agentId ? { agentId: entry.agentId } : {}),
    ...(entry.conversationId ? { conversationId: entry.conversationId } : {}),
    ...(entry.taskId ? { taskId: entry.taskId } : {}),
    ...(entry.userId ? { userId: entry.userId } : {}),
  }
}
