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
  readonly dbPath?: string
  readonly enableEmbeddings?: boolean
  readonly maxItems?: number
  readonly retentionDays?: number
  readonly storage?: MemoryStorage
}

/** Durable memory tier scored by lexical relevance, recency, and importance. */
export class LongTermMemory extends Memory {
  readonly retentionDays: number

  constructor(options: LongTermMemoryOptions = {}) {
    const enableEmbeddings = options.enableEmbeddings ?? true
    const storage = options.storage ?? (enableEmbeddings
      ? new RAGStorage(new SQLiteStorage({ ...(options.dbPath ? { dbPath: options.dbPath } : {}) }))
      : new SQLiteStorage({ ...(options.dbPath ? { dbPath: options.dbPath } : {}) }))
    super(storage, options.maxItems ?? 10_000, enableEmbeddings)
    this.retentionDays = options.retentionDays ?? 365
    this.hydrate()
  }

  clear(): void {
    for (const item of this.items) this.storage?.delete(storageKey(item.memoryId))
    this.items.length = 0
    this.index.clear()
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
    for (const item of targets) {
      this.remove(item)
      this.storage?.delete(storageKey(item.memoryId))
    }
    return targets.length
  }

  retrieve(memoryId?: string, filters?: MemoryFilters, limit = 10): MemoryItem | MemoryItem[] | undefined {
    if (memoryId) {
      const item = this.index.get(memoryId)
      if (item) {
        item.touch()
        this.persist(item)
      }
      return item
    }
    const matches = this.items.filter(item => this.matchesFilters(item, filters)).slice(0, limit)
    for (const item of matches) {
      item.touch()
      this.persist(item)
    }
    return matches
  }

  save(content: string, metadata: MemoryMetadata = {}, options: MemorySaveOptions = {}): MemoryItem {
    if (this.maxItems !== undefined && this.items.length >= this.maxItems) this.cleanupOldMemories()
    const item = new MemoryItem({
      content,
      memoryType: 'long_term',
      metadata: { ...metadata, importance: options.importance ?? 0.5 },
      ...(options.agentId ? { agentId: options.agentId } : {}),
      ...(options.taskId ? { taskId: options.taskId } : {}),
      ...(options.userId ? { userId: options.userId } : {}),
      ...(options.conversationId ? { conversationId: options.conversationId } : {}),
    })
    this.append(item)
    this.persist(item)
    return item
  }

  search(query: string, limit = 10, filters?: MemoryFilters, options: MemorySearchOptions = {}): MemoryItem[] {
    if (options.useSemantic !== false && this.storage?.supportsSemanticSearch()) {
      const semantic = this.storage.semanticSearch(query, limit * 2)
      const matches: MemoryItem[] = []
      for (const result of semantic) {
        if (!result.key.startsWith('ltm_') || !isRecord(result.data)) continue
        const decoded = MemoryItem.fromRecord(result.data)
        const item = this.index.get(decoded.memoryId) ?? decoded
        if (!this.matchesFilters(item, filters)) continue
        item.relevanceScore = result.similarity
        item.touch()
        this.persist(item)
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
      item.touch()
      this.persist(item)
      matches.push(item)
    }
    return matches.sort((left, right) => right.relevanceScore - left.relevanceScore).slice(0, limit)
  }

  update(memoryId: string, updates: MemoryUpdate): boolean {
    const item = this.index.get(memoryId)
    if (!item) return false
    this.updateItem(item, updates)
    this.persist(item)
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
    for (const item of new Set(targets)) {
      this.remove(item)
      this.storage?.delete(storageKey(item.memoryId))
    }
  }

  private hydrate(): void {
    for (const key of this.storage?.listKeys('ltm_') ?? []) {
      const record = this.storage?.load(key)
      if (isRecord(record)) this.append(MemoryItem.fromRecord(record))
    }
  }

  private mergeSimilar(threshold: number): void {
    const candidates = this.items.slice()
    for (let index = 0; index < candidates.length; index += 1) {
      const current = candidates[index]
      if (!current || !this.index.has(current.memoryId)) continue
      const sourceTerms = terms(current.content)
      for (const other of candidates.slice(index + 1)) {
        if (!this.index.has(other.memoryId)) continue
        const overlap = overlapRatio(sourceTerms, terms(other.content))
        if (overlap < threshold) continue
        current.content = `${current.content}\n${other.content}`
        current.metadata = { ...current.metadata, merged: true }
        this.remove(other)
        this.storage?.delete(storageKey(other.memoryId))
      }
      this.persist(current)
    }
  }

  private persist(item: MemoryItem): void {
    this.storage?.save(storageKey(item.memoryId), item.toRecord())
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
