// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  Memory,
  MemoryItem,
  type MemoryFilters,
  type MemoryMetadata,
  type MemorySaveOptions,
  type MemorySearchOptions,
  type MemoryUpdate,
} from './base.js'
import { RAGStorage, SQLiteStorage, type MemoryStorage } from './storage.js'

export interface LongTermMemoryOptions {
  /**
   * Number of access-state touches batched before one persistence flush.
   * Higher values reduce write amplification at the cost of losing recent
   * access counts on an unflushed crash. Call `flushAccessState()` on a
   * graceful shutdown to make every touch durable.
   */
  readonly accessFlushThreshold?: number
  readonly dbPath?: string
  readonly enableEmbeddings?: boolean
  readonly maxItems?: number
  /**
   * Tenant identity for tiers over a shared backend. When set, hydration and
   * search only admit records owned by this id (matching `agent_id`,
   * `user_id`, or the stamped `metadata.owner_id`), so two instances over one
   * database cannot read each other's memories. Saved records are stamped
   * with `metadata.owner_id` so a later instance with the same owner
   * re-hydrates them.
   */
  readonly ownerId?: string
  readonly retentionDays?: number
  readonly storage?: MemoryStorage
}

/** Durable memory tier scored by lexical relevance, recency, and importance. */
export class LongTermMemory extends Memory {
  readonly ownerId: string | undefined
  readonly retentionDays: number
  private readonly accessDirty = new Map<MemoryItem, number>()
  private readonly accessFlushThreshold: number
  private pendingAccessWrites = 0

  constructor(options: LongTermMemoryOptions = {}) {
    const enableEmbeddings = options.enableEmbeddings ?? true
    const storage = options.storage ?? (enableEmbeddings
      ? new RAGStorage(new SQLiteStorage({ ...(options.dbPath ? { dbPath: options.dbPath } : {}) }))
      : new SQLiteStorage({ ...(options.dbPath ? { dbPath: options.dbPath } : {}) }))
    super(storage, options.maxItems ?? 10_000, enableEmbeddings)
    this.retentionDays = options.retentionDays ?? 365
    this.ownerId = options.ownerId?.trim() || undefined
    const threshold = options.accessFlushThreshold ?? 25
    this.accessFlushThreshold = Number.isInteger(threshold) && threshold >= 1 ? threshold : 25
    this.hydrate()
  }

  clear(): void {
    this.deletePersisted(this.items, 'clear long-term memory')
    this.items.length = 0
    this.index.clear()
    this.accessDirty.clear()
    this.pendingAccessWrites = 0
  }

  consolidate(mergeSimilar = true, similarityThreshold = 0.8): string {
    if (this.items.length === 0) return 'No long-term memories available.'
    if (mergeSimilar) this.mergeSimilar(similarityThreshold)
    const groups = new Map<string, MemoryItem[]>()
    for (const item of this.items) {
      const key = item.conversationId ?? item.agentId ?? 'general'
      const values = groups.get(key) ?? []
      values.push(item)
      groups.set(key, values)
    }
    const lines = ['Long-term memory summary:']
    for (const [key, items] of groups) {
      lines.push(`\n${titleCase(key)}:`)
      for (const item of items
        .slice()
        .sort((left, right) => importance(right) - importance(left) || right.timestamp.valueOf() - left.timestamp.valueOf())
        .slice(0, 5)) {
        lines.push(`  - ${item.content.slice(0, 150)} (importance: ${importance(item).toFixed(1)}, accessed: ${item.accessCount}x)`)
      }
    }
    return lines.join('\n')
  }

  delete(memoryId?: string, filters?: MemoryFilters): number {
    const targets = memoryId
      ? this.index.get(memoryId) ? [this.index.get(memoryId) as MemoryItem] : []
      : filters ? this.items.filter(item => this.matchesFilters(item, filters)) : []
    this.deletePersisted(targets, 'delete long-term memory')
    for (const item of targets) {
      this.remove(item)
      this.accessDirty.delete(item)
    }
    return targets.length
  }

  /**
   * Persist every batched access-state update. Touches are debounced to
   * avoid one storage rewrite per retrieve/search hit; call this on a
   * graceful shutdown when recent access counts must survive a restart.
   */
  flushAccessState(): void {
    if (this.accessDirty.size === 0) {
      this.pendingAccessWrites = 0
      return
    }
    for (const [item, increment] of this.accessDirty) {
      const lastAccessed = item.lastAccessed?.toISOString()
      if (!lastAccessed) continue
      const result = this.storage?.updateAccessState?.(storageKey(item.memoryId), increment, lastAccessed)
      if (result === 'updated' || result === 'missing') continue
      this.mergeAccessState(item, increment, lastAccessed)
    }
    this.accessDirty.clear()
    this.pendingAccessWrites = 0
  }

  retrieve(memoryId?: string, filters?: MemoryFilters, limit = 10): MemoryItem | MemoryItem[] | undefined {
    if (memoryId) {
      const item = this.index.get(memoryId)
      if (item) this.touchAndTrack(item)
      return item
    }
    const matches = this.items.filter(item => this.matchesFilters(item, filters)).slice(0, limit)
    for (const item of matches) this.touchAndTrack(item)
    return matches
  }

  save(content: string, metadata: MemoryMetadata = {}, options: MemorySaveOptions = {}): MemoryItem {
    const item = new MemoryItem({
      content,
      memoryType: 'long_term',
      metadata: {
        ...metadata,
        importance: options.importance ?? 0.5,
        // The configured tenant is authoritative: caller metadata must not
        // stamp a record as belonging to another owner.
        ...(this.ownerId ? { owner_id: this.ownerId } : {}),
      },
      ...(options.agentId ? { agentId: options.agentId } : {}),
      ...(options.taskId ? { taskId: options.taskId } : {}),
      ...(options.userId ? { userId: options.userId } : {}),
      ...(options.conversationId ? { conversationId: options.conversationId } : {}),
    })
    this.persist(item)
    try {
      if (this.maxItems !== undefined && this.items.length >= this.maxItems) this.cleanupOldMemories()
    } catch (error) {
      this.storage?.delete(storageKey(item.memoryId))
      throw error
    }
    this.append(item)
    return item
  }

  search(query: string, limit = 10, filters?: MemoryFilters, options: MemorySearchOptions = {}): MemoryItem[] {
    if (options.useSemantic !== false && this.storage?.supportsSemanticSearch()) {
      const semantic = this.storage.semanticSearch(query, limit * 2)
      const matches: MemoryItem[] = []
      for (const result of semantic) {
        if (!result.key.startsWith('ltm_') || !isRecord(result.data)) continue
        const decoded = MemoryItem.fromRecord(result.data)
        // Only items this instance owns may match: admitting raw decoded
        // records here would leak every other tenant's memories from a
        // shared backend.
        const item = this.index.get(decoded.memoryId)
        if (!item || !this.matchesFilters(item, filters)) continue
        item.relevanceScore = result.similarity
        this.touchAndTrack(item)
        matches.push(item)
        if (matches.length >= limit) break
      }
      return matches
    }
    const normalizedQuery = query.toLowerCase()
    const matches: MemoryItem[] = []
    for (const item of this.items) {
      if (!this.matchesFilters(item, filters)) continue
      const relevance = lexicalRelevance(item.content, normalizedQuery)
      const ageDays = Math.max(0, (Date.now() - item.timestamp.valueOf()) / 86_400_000)
      const recency = Math.max(0, 1 - ageDays / this.retentionDays)
      item.relevanceScore = relevance * 0.5 + recency * 0.3 + importance(item) * 0.2
      if (item.relevanceScore <= 0) continue
      this.touchAndTrack(item)
      matches.push(item)
    }
    return matches.sort((left, right) => right.relevanceScore - left.relevanceScore).slice(0, limit)
  }

  update(memoryId: string, updates: MemoryUpdate): boolean {
    const item = this.index.get(memoryId)
    if (!item) return false
    const updated = MemoryItem.fromRecord(item.toRecord())
    this.updateItem(updated, updates)
    if (this.ownerId) updated.metadata = { ...updated.metadata, owner_id: this.ownerId }
    this.persist(updated)
    this.updateItem(item, updates)
    if (this.ownerId) item.metadata = { ...item.metadata, owner_id: this.ownerId }
    this.accessDirty.delete(item)
    return true
  }

  private cleanupOldMemories(): void {
    const cutoff = Date.now() - this.retentionDays * 86_400_000
    let targets = this.items.filter(item => item.timestamp.valueOf() < cutoff || (importance(item) < 0.3 && item.accessCount < 2))
    const minimumToRemove = Math.max(1, Math.floor(this.items.length * 0.2))
    if (targets.length < minimumToRemove) {
      targets = this.items
        .slice()
        .sort((left, right) => valueScore(left, this.retentionDays) - valueScore(right, this.retentionDays))
        .slice(0, minimumToRemove)
    }
    const uniqueTargets = [...new Set(targets)]
    this.deletePersisted(uniqueTargets, 'evict long-term memory')
    for (const item of uniqueTargets) {
      this.remove(item)
      this.accessDirty.delete(item)
    }
  }

  private deletePersisted(targets: readonly MemoryItem[], operation: string): void {
    if (!this.storage) return
    const deleted: MemoryItem[] = []
    for (const item of targets) {
      if (!this.storage.delete(storageKey(item.memoryId))) {
        for (const removed of deleted) this.storage.save(storageKey(removed.memoryId), removed.toRecord())
        throw new Error(`Failed to ${operation} ${item.memoryId}`)
      }
      deleted.push(item)
    }
  }

  private hydrate(): void {
    const records: MemoryItem[] = []
    for (const key of this.storage?.listKeys('ltm_') ?? []) {
      if (!key.startsWith('ltm_')) continue
      let record: unknown
      try {
        record = this.storage?.load(key)
      } catch (error) {
        console.warn(`Skipping corrupt long-term memory record ${key}:`, error)
        continue
      }
      if (!isRecord(record)) continue
      const item = MemoryItem.fromRecord(record)
      // Tenant filter: over a shared backend, only restore this owner's
      // records instead of every `ltm_*` row in the database.
      if (this.ownerId && !this.owns(item)) continue
      records.push(item)
    }
    records.sort((left, right) => left.timestamp.valueOf() - right.timestamp.valueOf())
    const retained = this.maxItems === undefined ? records : records.slice(-this.maxItems)
    for (const item of retained) this.append(item)
  }

  /**
   * Read-only retrieval for summaries and boot hydration: returns the most
   * important items without touching access state or re-persisting anything.
   */
  mostImportant(limit = 10): MemoryItem[] {
    return this.items
      .slice()
      .sort((left, right) => importance(right) - importance(left) || right.timestamp.valueOf() - left.timestamp.valueOf())
      .slice(0, Math.max(0, limit))
  }

  private mergeSimilar(threshold: number): void {
    const candidates = this.items.slice()
    const merged = new Map<MemoryItem, MemoryItem>()
    const removed = new Set<MemoryItem>()
    for (let index = 0; index < candidates.length; index += 1) {
      const current = candidates[index]
      if (!current || removed.has(current)) continue
      const updated = MemoryItem.fromRecord(current.toRecord())
      const sourceTerms = terms(current.content)
      for (const other of candidates.slice(index + 1)) {
        if (removed.has(other)) continue
        const overlap = overlapRatio(sourceTerms, terms(other.content))
        if (overlap < threshold) continue
        updated.content = `${updated.content}\n${other.content}`
        updated.metadata = { ...updated.metadata, merged: true }
        removed.add(other)
      }
      merged.set(current, updated)
    }

    if (this.storage) {
      const originals = candidates.map(item => [storageKey(item.memoryId), item.toRecord()] as const)
      try {
        for (const updated of merged.values()) this.persist(updated)
        for (const item of removed) {
          if (!this.storage.delete(storageKey(item.memoryId))) {
            throw new Error(`Failed to merge long-term memory ${item.memoryId}`)
          }
        }
      } catch (error) {
        const failed = originals.filter(([key, record]) => !this.storage?.save(key, record))
        if (failed.length > 0) {
          throw new Error('Failed to roll back long-term memory consolidation', { cause: error })
        }
        throw error
      }
    }

    for (const [current, updated] of merged) {
      current.content = updated.content
      current.metadata = { ...updated.metadata }
      this.accessDirty.delete(current)
    }
    for (const item of removed) {
      this.remove(item)
      this.accessDirty.delete(item)
    }
  }

  /** Return whether a hydrated record belongs to this instance's tenant. */
  private owns(item: MemoryItem): boolean {
    if (!this.ownerId) return true
    return item.agentId === this.ownerId
      || item.userId === this.ownerId
      || item.metadata.owner_id === this.ownerId
  }

  private persist(item: MemoryItem): void {
    if (this.storage && !this.storage.save(storageKey(item.memoryId), item.toRecord())) {
      throw new Error(`Failed to persist long-term memory ${item.memoryId}`)
    }
  }

  /**
   * Compatibility fallback for custom backends without a field-update primitive.
   * It merges against the latest row before saving, which prevents stale-field
   * clobbering but cannot promise atomic increments across concurrent processes.
   */
  private mergeAccessState(item: MemoryItem, increment: number, lastAccessed: string): void {
    const key = storageKey(item.memoryId)
    const current = this.storage?.load(key)
    if (current === undefined) return
    if (!isRecord(current)) throw new Error(`Failed to persist long-term memory ${item.memoryId}`)
    const accessCount = typeof current.access_count === 'number' && Number.isInteger(current.access_count)
      ? current.access_count
      : 0
    if (!this.storage?.save(key, {
      ...current,
      access_count: accessCount + increment,
      last_accessed: lastAccessed,
    })) {
      throw new Error(`Failed to persist long-term memory ${item.memoryId}`)
    }
  }

  /**
   * Record one access hit and batch its persistence. Immediate per-hit
   * rewrites caused heavy write amplification and last-writer-wins clobbering
   * between instances sharing a backend; hits flush every
   * `accessFlushThreshold` touches or on `flushAccessState()`.
   */
  private touchAndTrack(item: MemoryItem): void {
    item.touch()
    if (!this.storage) return
    this.accessDirty.set(item, (this.accessDirty.get(item) ?? 0) + 1)
    this.pendingAccessWrites += 1
    if (this.pendingAccessWrites >= this.accessFlushThreshold) this.flushAccessState()
  }
}

function importance(item: MemoryItem): number {
  const value = item.metadata.importance
  return typeof value === 'number' && Number.isFinite(value) ? value : 0.5
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function lexicalRelevance(content: string, query: string): number {
  const normalized = content.toLowerCase()
  if (normalized.includes(query)) return 1
  const queryTerms = terms(query)
  return queryTerms.length === 0 ? 0 : queryTerms.filter(term => normalized.includes(term)).length / queryTerms.length
}

function overlapRatio(left: readonly string[], right: readonly string[]): number {
  if (left.length === 0 || right.length === 0) return 0
  const rightSet = new Set(right)
  return left.filter(term => rightSet.has(term)).length / Math.max(left.length, right.length)
}

function storageKey(memoryId: string): string {
  return `ltm_${memoryId}`
}

function terms(value: string): string[] {
  return value.toLowerCase().split(/\s+/).filter(Boolean)
}

function titleCase(value: string): string {
  return value.slice(0, 1).toUpperCase() + value.slice(1)
}

function valueScore(item: MemoryItem, retentionDays: number): number {
  const ageDays = Math.max(0, (Date.now() - item.timestamp.valueOf()) / 86_400_000)
  return importance(item) * 0.3 + item.accessCount / 100 * 0.3 + (1 - ageDays / retentionDays) * 0.4
}
