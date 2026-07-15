// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { MemoryStorage } from './storage.js'

export type MemoryMetadata = Record<string, unknown>
export type MemoryFilter = unknown | ((value: unknown) => boolean)
export type MemoryFilters = Readonly<Record<string, MemoryFilter>>

export interface MemoryItemOptions {
  readonly accessCount?: number
  readonly agentId?: string
  readonly content: string
  readonly conversationId?: string
  readonly embedding?: readonly number[]
  readonly lastAccessed?: Date
  readonly memoryId?: string
  readonly memoryType?: string
  readonly metadata?: MemoryMetadata
  readonly relevanceScore?: number
  readonly taskId?: string
  readonly timestamp?: Date
  readonly userId?: string
}

/** JSON-safe persistent shape shared by all memory storage backends. */
export interface MemoryItemRecord extends Record<string, unknown> {
  readonly access_count: number
  readonly agent_id: string | null
  readonly content: string
  readonly conversation_id: string | null
  readonly last_accessed: string | null
  readonly memory_id: string
  readonly memory_type: string
  readonly metadata: MemoryMetadata
  readonly relevance_score: number
  readonly task_id: string | null
  readonly timestamp: string
  readonly user_id: string | null
}

/** A recallable value plus ownership, access, and retrieval metadata. */
export class MemoryItem {
  accessCount: number
  agentId: string | undefined
  content: string
  conversationId: string | undefined
  embedding: number[] | undefined
  lastAccessed: Date | undefined
  readonly memoryId: string
  memoryType: string
  metadata: MemoryMetadata
  relevanceScore: number
  taskId: string | undefined
  timestamp: Date
  userId: string | undefined

  constructor(options: MemoryItemOptions) {
    this.content = options.content
    this.memoryType = options.memoryType ?? 'general'
    this.timestamp = options.timestamp ?? new Date()
    this.metadata = { ...(options.metadata ?? {}) }
    this.agentId = options.agentId
    this.taskId = options.taskId
    this.conversationId = options.conversationId
    this.userId = options.userId
    this.relevanceScore = options.relevanceScore ?? 1
    this.accessCount = options.accessCount ?? 0
    this.lastAccessed = options.lastAccessed
    this.embedding = options.embedding ? [...options.embedding] : undefined
    this.memoryId = options.memoryId ?? crypto.randomUUID()
  }

  touch(now = new Date()): void {
    this.accessCount += 1
    this.lastAccessed = now
  }

  toRecord(): MemoryItemRecord {
    return {
      memory_id: this.memoryId,
      content: this.content,
      memory_type: this.memoryType,
      timestamp: this.timestamp.toISOString(),
      metadata: { ...this.metadata },
      agent_id: this.agentId ?? null,
      task_id: this.taskId ?? null,
      conversation_id: this.conversationId ?? null,
      user_id: this.userId ?? null,
      relevance_score: this.relevanceScore,
      access_count: this.accessCount,
      last_accessed: this.lastAccessed?.toISOString() ?? null,
    }
  }

  static fromRecord(value: Record<string, unknown>): MemoryItem {
    const metadata = isRecord(value.metadata) ? value.metadata : {}
    const embedding = Array.isArray(value.embedding) && value.embedding.every(item => typeof item === 'number')
      ? value.embedding
      : undefined
    const agentId = nullableString(value.agent_id)
    const taskId = nullableString(value.task_id)
    const conversationId = nullableString(value.conversation_id)
    const userId = nullableString(value.user_id)
    const lastAccessed = dateValue(value.last_accessed)
    return new MemoryItem({
      content: stringValue(value.content),
      memoryType: stringValue(value.memory_type) || 'general',
      timestamp: dateValue(value.timestamp) ?? new Date(),
      metadata,
      relevanceScore: numberValue(value.relevance_score, 1),
      accessCount: integerValue(value.access_count),
      ...(agentId ? { agentId } : {}),
      ...(taskId ? { taskId } : {}),
      ...(conversationId ? { conversationId } : {}),
      ...(userId ? { userId } : {}),
      ...(lastAccessed ? { lastAccessed } : {}),
      ...(embedding ? { embedding } : {}),
      memoryId: stringValue(value.memory_id) || crypto.randomUUID(),
    })
  }
}

export interface MemoryStatistics {
  readonly maxItems: number | undefined
  readonly memoryTypes: Readonly<Record<string, number>>
  readonly totalItems: number
  readonly uniqueAgents: number
  readonly uniqueConversations: number
  readonly uniqueUsers: number
}

export abstract class Memory {
  protected readonly index = new Map<string, MemoryItem>()
  protected items: MemoryItem[] = []

  protected constructor(
    protected readonly storage: MemoryStorage | undefined = undefined,
    readonly maxItems: number | undefined = undefined,
    readonly enableEmbeddings = false,
  ) {}

  abstract clear(): void
  abstract delete(memoryId?: string, filters?: MemoryFilters): number
  abstract retrieve(memoryId?: string, filters?: MemoryFilters, limit?: number): MemoryItem | MemoryItem[] | undefined
  abstract save(content: string, metadata?: MemoryMetadata, options?: MemorySaveOptions): MemoryItem
  abstract search(query: string, limit?: number, filters?: MemoryFilters, options?: MemorySearchOptions): MemoryItem[]
  abstract update(memoryId: string, updates: MemoryUpdate): boolean

  get size(): number {
    return this.items.length
  }

  getContext(limit = 10, format: 'json' | 'markdown' | 'text' = 'text'): string {
    const items = this.items.slice(-limit)
    if (format === 'json') {
      return JSON.stringify(items.map(item => item.toRecord()), null, 2)
    }
    if (format === 'markdown') {
      return items
        .map(item => `- [${formatDate(item.timestamp)}] ${item.agentId ? `**${item.agentId}**` : '**System**'}: ${item.content}`)
        .join('\n')
    }
    return items.map(item => item.agentId ? `[${item.agentId}]: ${item.content}` : item.content).join('\n')
  }

  getStatistics(): MemoryStatistics {
    const memoryTypes: Record<string, number> = {}
    const agents = new Set<string>()
    const users = new Set<string>()
    const conversations = new Set<string>()
    for (const item of this.items) {
      memoryTypes[item.memoryType] = (memoryTypes[item.memoryType] ?? 0) + 1
      if (item.agentId) agents.add(item.agentId)
      if (item.userId) users.add(item.userId)
      if (item.conversationId) conversations.add(item.conversationId)
    }
    return {
      totalItems: this.items.length,
      maxItems: this.maxItems,
      memoryTypes,
      uniqueAgents: agents.size,
      uniqueUsers: users.size,
      uniqueConversations: conversations.size,
    }
  }

  protected append(item: MemoryItem): void {
    this.items.push(item)
    this.index.set(item.memoryId, item)
  }

  protected matchesFilters(item: MemoryItem, filters: MemoryFilters | undefined): boolean {
    if (!filters) return true
    for (const [key, expected] of Object.entries(filters)) {
      const actual = itemField(item, key)
      if (actual === MISSING || (typeof expected === 'function' ? !expected(actual) : actual !== expected)) {
        return false
      }
    }
    return true
  }

  protected updateItem(item: MemoryItem, updates: MemoryUpdate): void {
    if (updates.content !== undefined) item.content = updates.content
    if (updates.memoryType !== undefined) item.memoryType = updates.memoryType
    if (updates.timestamp !== undefined) item.timestamp = updates.timestamp
    if (updates.metadata !== undefined) item.metadata = { ...updates.metadata }
    if (updates.agentId !== undefined) item.agentId = updates.agentId
    if (updates.taskId !== undefined) item.taskId = updates.taskId
    if (updates.conversationId !== undefined) item.conversationId = updates.conversationId
    if (updates.userId !== undefined) item.userId = updates.userId
    if (updates.relevanceScore !== undefined) item.relevanceScore = updates.relevanceScore
    if (updates.accessCount !== undefined) item.accessCount = updates.accessCount
    if (updates.lastAccessed !== undefined) item.lastAccessed = updates.lastAccessed
    if (updates.embedding !== undefined) item.embedding = [...updates.embedding]
  }

  protected remove(item: MemoryItem): void {
    const position = this.items.indexOf(item)
    if (position >= 0) this.items.splice(position, 1)
    this.index.delete(item.memoryId)
  }
}

export interface MemorySaveOptions {
  readonly agentId?: string
  readonly conversationId?: string
  readonly importance?: number
  readonly memoryType?: string
  readonly taskId?: string
  readonly userId?: string
}

export interface MemorySearchOptions {
  readonly minRelevance?: number
  readonly useSemantic?: boolean
}

export interface MemoryUpdate {
  readonly accessCount?: number
  readonly agentId?: string
  readonly content?: string
  readonly conversationId?: string
  readonly embedding?: readonly number[]
  readonly lastAccessed?: Date
  readonly memoryType?: string
  readonly metadata?: MemoryMetadata
  readonly relevanceScore?: number
  readonly taskId?: string
  readonly timestamp?: Date
  readonly userId?: string
}

const MISSING = Symbol('missing')

function itemField(item: MemoryItem, key: string): unknown | typeof MISSING {
  const fields: Record<string, unknown> = {
    memory_id: item.memoryId,
    memoryId: item.memoryId,
    content: item.content,
    memory_type: item.memoryType,
    memoryType: item.memoryType,
    timestamp: item.timestamp,
    metadata: item.metadata,
    agent_id: item.agentId,
    agentId: item.agentId,
    task_id: item.taskId,
    taskId: item.taskId,
    conversation_id: item.conversationId,
    conversationId: item.conversationId,
    user_id: item.userId,
    userId: item.userId,
    relevance_score: item.relevanceScore,
    relevanceScore: item.relevanceScore,
    access_count: item.accessCount,
    accessCount: item.accessCount,
    last_accessed: item.lastAccessed,
    lastAccessed: item.lastAccessed,
  }
  if (key in fields) return fields[key]
  return key in item.metadata ? item.metadata[key] : MISSING
}

function formatDate(value: Date): string {
  return value.toISOString().replace('T', ' ').slice(0, 19)
}

function dateValue(value: unknown): Date | undefined {
  if (value instanceof Date) return value
  if (typeof value !== 'string' || !value) return undefined
  const parsed = new Date(value)
  return Number.isNaN(parsed.valueOf()) ? undefined : parsed
}

function integerValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.trunc(value) : 0
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function nullableString(value: unknown): string | undefined {
  return typeof value === 'string' && value ? value : undefined
}

function numberValue(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}
