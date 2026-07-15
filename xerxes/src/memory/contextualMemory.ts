// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  Memory,
  type MemoryFilters,
  type MemoryItem,
  type MemoryMetadata,
  type MemorySaveOptions,
  type MemorySearchOptions,
  type MemoryUpdate,
} from './base.js'
import { LongTermMemory, type LongTermMemoryOptions } from './longTermMemory.js'
import { ShortTermMemory } from './shortTermMemory.js'
import type { MemoryStorage } from './storage.js'

export interface ContextFrame {
  readonly data: Readonly<Record<string, unknown>>
  readonly timestamp: Date
  readonly type: string
}

export interface ContextualMemoryOptions {
  readonly importanceThreshold?: number
  readonly longTerm?: LongTermMemory
  readonly longTermOptions?: LongTermMemoryOptions
  readonly longTermStorage?: MemoryStorage
  readonly promotionThreshold?: number
  readonly shortTermCapacity?: number
}

export interface ContextualSaveOptions extends MemorySaveOptions {
  readonly toLongTerm?: boolean
}

/** Composite short/long-term store with context-aware reranking and promotion. */
export class ContextualMemory extends Memory {
  readonly contextStack: ContextFrame[] = []
  readonly importanceThreshold: number
  readonly longTerm: LongTermMemory
  readonly promotionThreshold: number
  readonly shortTerm: ShortTermMemory

  constructor(options: ContextualMemoryOptions = {}) {
    super()
    this.shortTerm = new ShortTermMemory({ capacity: options.shortTermCapacity ?? 20 })
    this.longTerm = options.longTerm ?? new LongTermMemory({
      ...(options.longTermOptions ?? {}),
      ...(options.longTermStorage ? { storage: options.longTermStorage } : {}),
    })
    this.promotionThreshold = options.promotionThreshold ?? 3
    this.importanceThreshold = options.importanceThreshold ?? 0.7
  }

  clear(): void {
    this.shortTerm.clear()
    this.longTerm.clear()
    this.contextStack.length = 0
  }

  delete(memoryId?: string, filters?: MemoryFilters): number {
    return this.shortTerm.delete(memoryId, filters) + this.longTerm.delete(memoryId, filters)
  }

  getCurrentContext(): ContextFrame | undefined {
    return this.contextStack.at(-1)
  }

  getContextSummary(): string {
    const lines: string[] = []
    if (this.contextStack.length > 0) {
      lines.push('Current context:')
      for (const context of this.contextStack.slice(-3)) lines.push(`  - ${context.type}: ${JSON.stringify(context.data).slice(0, 100)}`)
    }
    const recent = this.shortTerm.getRecent(5)
    if (recent.length > 0) {
      lines.push('\nRecent activity:')
      for (const item of recent) lines.push(`  - ${item.content.slice(0, 100)}`)
    }
    const important = this.longTerm.search('', 20).filter(item => importance(item) >= 0.8).slice(0, 3)
    if (important.length > 0) {
      lines.push('\nImportant memories:')
      for (const item of important) lines.push(`  - ${item.content.slice(0, 100)}`)
    }
    return lines.length === 0 ? 'No context available.' : lines.join('\n')
  }

  popContext(): ContextFrame | undefined {
    return this.contextStack.pop()
  }

  pushContext(type: string, data: Readonly<Record<string, unknown>>): void {
    this.contextStack.push({ type, data: { ...data }, timestamp: new Date() })
  }

  retrieve(memoryId?: string, filters?: MemoryFilters, limit = 10): MemoryItem | MemoryItem[] | undefined {
    if (memoryId) {
      const shortTerm = this.shortTerm.retrieve(memoryId)
      if (shortTerm && !Array.isArray(shortTerm)) {
        this.checkPromotion(shortTerm)
        return shortTerm
      }
      return this.longTerm.retrieve(memoryId)
    }
    const shortTerm = this.shortTerm.retrieve(undefined, filters, limit)
    const results = Array.isArray(shortTerm) ? [...shortTerm] : shortTerm ? [shortTerm] : []
    const longTerm = this.longTerm.retrieve(undefined, filters, Math.max(0, limit - results.length))
    if (Array.isArray(longTerm)) results.push(...longTerm)
    else if (longTerm) results.push(longTerm)
    return results.slice(0, limit)
  }

  save(content: string, metadata: MemoryMetadata = {}, options: ContextualSaveOptions = {}): MemoryItem {
    const currentContext = this.getCurrentContext()
    const contextMetadata = currentContext
      ? { ...metadata, context: { type: currentContext.type, data: currentContext.data, timestamp: currentContext.timestamp.toISOString() } }
      : { ...metadata }
    const importance = options.importance ?? 0.5
    if (options.toLongTerm || importance >= this.importanceThreshold) {
      return this.longTerm.save(content, contextMetadata, options)
    }
    const item = this.shortTerm.save(content, contextMetadata, options)
    item.metadata.importance = importance
    this.checkPromotion(item)
    return item
  }

  search(query: string, limit = 10, filters?: MemoryFilters, options: MemorySearchOptions = {}): MemoryItem[] {
    const results = this.shortTerm.search(query, limit, filters, options)
    for (const item of results) item.metadata.source = 'short_term'
    const longTerm = this.longTerm.search(query, limit, filters, options)
    for (const item of longTerm) item.metadata.source = 'long_term'
    const merged = [...results, ...longTerm]
    if (this.contextStack.length > 0) this.rerankByContext(merged)
    return merged.sort((left, right) => right.relevanceScore - left.relevanceScore).slice(0, limit)
  }

  update(memoryId: string, updates: MemoryUpdate): boolean {
    return this.shortTerm.update(memoryId, updates) || this.longTerm.update(memoryId, updates)
  }

  private checkPromotion(item: MemoryItem): void {
    if (item.accessCount < this.promotionThreshold || item.metadata.promoted === true) return
    this.longTerm.save(item.content, item.metadata, {
      ...(item.agentId ? { agentId: item.agentId } : {}),
      ...(item.taskId ? { taskId: item.taskId } : {}),
      ...(item.userId ? { userId: item.userId } : {}),
      ...(item.conversationId ? { conversationId: item.conversationId } : {}),
      importance: importance(item) || 0.6,
    })
    item.metadata.promoted = true
  }

  private rerankByContext(items: MemoryItem[]): void {
    const current = this.getCurrentContext()
    if (!current) return
    for (const item of items) {
      const value = item.metadata.context
      if (!isContextMetadata(value)) continue
      let contextMatch = value.type === current.type ? 0.5 : 0
      const shared = sharedWords(JSON.stringify(value.data), JSON.stringify(current.data))
      if (shared > 0) contextMatch += 0.5 * shared
      item.relevanceScore = item.relevanceScore * 0.7 + contextMatch * 0.3
    }
  }
}

function importance(item: MemoryItem): number {
  return typeof item.metadata.importance === 'number' ? item.metadata.importance : 0.5
}

function isContextMetadata(value: unknown): value is { readonly data: unknown; readonly type: string } {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    && typeof (value as Record<string, unknown>).type === 'string'
}

function sharedWords(left: string, right: string): number {
  const leftWords = new Set(left.toLowerCase().split(/\s+/).filter(Boolean))
  const rightWords = new Set(right.toLowerCase().split(/\s+/).filter(Boolean))
  const overlap = [...leftWords].filter(word => rightWords.has(word)).length
  return overlap / Math.max(leftWords.size, rightWords.size, 1)
}
