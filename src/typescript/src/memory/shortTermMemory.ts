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
import type { MemoryStorage } from './storage.js'

export interface ShortTermMemoryOptions {
  readonly capacity?: number
  readonly enableEmbeddings?: boolean
  readonly storage?: MemoryStorage
}

/** Bounded, recency-ordered working memory tier. */
export class ShortTermMemory extends Memory {
  constructor(options: ShortTermMemoryOptions = {}) {
    const capacity = options.capacity ?? 20
    if (!Number.isInteger(capacity) || capacity < 1) throw new Error('ShortTermMemory capacity must be at least one')
    super(options.storage, capacity, options.enableEmbeddings ?? false)
  }

  clear(): void {
    for (const item of this.items) this.storage?.delete(storageKey(item.memoryId))
    this.items.length = 0
    this.index.clear()
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

  getRecent(count = 5): MemoryItem[] {
    return this.items.slice(-Math.max(0, count))
  }

  retrieve(memoryId?: string, filters?: MemoryFilters, limit = 10): MemoryItem | MemoryItem[] | undefined {
    if (memoryId) {
      const item = this.index.get(memoryId)
      if (item) item.touch()
      return item
    }
    const matches = this.items.slice().reverse().filter(item => this.matchesFilters(item, filters)).slice(0, limit)
    for (const item of matches) item.touch()
    return matches
  }

  save(content: string, metadata: MemoryMetadata = {}, options: MemorySaveOptions = {}): MemoryItem {
    const item = new MemoryItem({
      content,
      memoryType: 'short_term',
      metadata: { ...metadata },
      ...(options.agentId ? { agentId: options.agentId } : {}),
      ...(options.taskId ? { taskId: options.taskId } : {}),
      ...(options.userId ? { userId: options.userId } : {}),
      ...(options.conversationId ? { conversationId: options.conversationId } : {}),
    })
    if (this.items.length >= (this.maxItems ?? Infinity)) {
      const evicted = this.items.shift()
      if (evicted) {
        this.index.delete(evicted.memoryId)
        this.storage?.delete(storageKey(evicted.memoryId))
      }
    }
    this.append(item)
    this.storage?.save(storageKey(item.memoryId), item.toRecord())
    return item
  }

  search(query: string, limit = 10, filters?: MemoryFilters, options: MemorySearchOptions = {}): MemoryItem[] {
    const normalizedQuery = query.toLowerCase()
    const words = normalizedQuery.split(/\s+/).filter(Boolean)
    const minimum = options.minRelevance ?? 0
    const matches: MemoryItem[] = []
    for (const item of this.items.slice().reverse()) {
      if (!this.matchesFilters(item, filters)) continue
      const content = item.content.toLowerCase()
      const relevance = normalizedQuery && content.includes(normalizedQuery)
        ? 1
        : words.length === 0 ? 1 : words.filter(word => content.includes(word)).length / words.length
      if (relevance < minimum) continue
      item.relevanceScore = relevance
      item.touch()
      matches.push(item)
      if (matches.length >= limit) break
    }
    return matches.sort((left, right) => right.relevanceScore - left.relevanceScore || right.timestamp.valueOf() - left.timestamp.valueOf())
  }

  summarize(): string {
    if (this.items.length === 0) return 'No recent memories.'
    const conversations = new Map<string, MemoryItem[]>()
    for (const item of this.items) {
      const key = item.conversationId ?? 'default'
      const values = conversations.get(key) ?? []
      values.push(item)
      conversations.set(key, values)
    }
    const lines = ['Recent activity:']
    for (const [conversationId, items] of conversations) {
      if (conversationId !== 'default') lines.push(`\nConversation ${conversationId}:`)
      for (const item of items.slice(-3)) lines.push(`  [${item.agentId ?? 'System'}]: ${item.content.slice(0, 100)}`)
    }
    return lines.join('\n')
  }

  update(memoryId: string, updates: MemoryUpdate): boolean {
    const item = this.index.get(memoryId)
    if (!item) return false
    this.updateItem(item, updates)
    this.storage?.save(storageKey(memoryId), item.toRecord())
    return true
  }
}

function storageKey(memoryId: string): string {
  return `stm_${memoryId}`
}
