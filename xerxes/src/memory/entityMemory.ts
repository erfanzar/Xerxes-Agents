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

const COMMON_ENTITY_WORDS = new Set(['The', 'This', 'That', 'These', 'Those'])
const ENTITY_PHRASE_PATTERN = /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g
const QUOTED_ENTITY_PATTERN = /"([^"]*)"/g
const MAX_ENTITY_CONTEXTS = 20

const RELATIONSHIP_PATTERNS: ReadonlyArray<readonly [RegExp, string]> = [
  [/(\w+)\s+is\s+(?:a|an|the)?\s*(\w+)\s+of\s+(\w+)/gi, 'relation_of'],
  [/(\w+)\s+works\s+(?:at|for|with)\s+(\w+)/gi, 'works_with'],
  [/(\w+)\s+knows\s+(\w+)/gi, 'knows'],
  [/(\w+)\s+created\s+(\w+)/gi, 'created'],
]

export interface EntityMemoryOptions {
  readonly enableEmbeddings?: boolean
  readonly maxItems?: number
  readonly storage?: MemoryStorage
}

export interface EntitySaveOptions extends MemorySaveOptions {
  readonly entities?: readonly string[]
}

export interface EntitySearchOptions extends MemorySearchOptions {
  readonly entityFilter?: readonly string[]
}

export interface EntityRecord {
  readonly contexts: string[]
  readonly firstSeen: Date
  frequency: number
  lastSeen: Date
}

export interface EntityRelationship {
  readonly relation: string
  readonly target: string
}

export interface EntityInfo {
  readonly contexts?: readonly string[]
  readonly firstSeen?: Date
  readonly frequency?: number
  readonly lastSeen?: Date
  readonly mentions: readonly string[]
  readonly relationships: readonly EntityRelationship[]
}

type EntityRelationPair = readonly [source: string, target: string]

/**
 * A memory tier indexed by lightweight named-entity and relationship heuristics.
 *
 * It intentionally keeps the extraction rule deterministic and dependency-free:
 * capitalized phrases and quoted text become entities, while a small set of
 * relation phrases build a traversable graph.
 */
export class EntityMemory extends Memory {
  readonly entities: Record<string, EntityRecord> = {}
  readonly entityMentions: Record<string, string[]> = {}
  readonly relationships: Record<string, EntityRelationPair[]> = {}

  constructor(options: EntityMemoryOptions = {}) {
    super(options.storage, options.maxItems ?? 5_000, options.enableEmbeddings ?? false)
    this.hydrate()
  }

  clear(): void {
    const retained = this.items.slice()
    this.items.length = 0
    this.index.clear()
    this.rebuildGraph()
    try {
      this.persistSnapshots()
      this.deletePersisted(retained, 'clear entity memory')
    } catch (error) {
      for (const item of retained) this.append(item)
      this.rebuildGraph()
      try {
        this.persistSnapshots()
      } catch {
        // Preserve the original storage failure; item rows remain authoritative
        // and hydration rebuilds snapshots from them.
      }
      throw error
    }
  }

  delete(memoryId?: string, _filters?: MemoryFilters): number {
    if (!memoryId) return 0
    const item = this.index.get(memoryId)
    if (!item) return 0

    const position = this.items.indexOf(item)
    this.remove(item)
    this.rebuildGraph()
    try {
      this.persistSnapshots()
      this.deletePersisted([item], 'delete entity memory')
    } catch (error) {
      this.items.splice(position, 0, item)
      this.index.set(item.memoryId, item)
      this.rebuildGraph()
      try {
        this.persistSnapshots()
      } catch {
        // Preserve the original storage failure; item rows remain authoritative.
      }
      throw error
    }
    return 1
  }

  /** Return entities inferred from capitalized phrases and double-quoted text. */
  extractEntities(text: string): string[] {
    const entities = new Set<string>()
    for (const match of text.matchAll(ENTITY_PHRASE_PATTERN)) {
      const entity = match[0]
      if (entity && !COMMON_ENTITY_WORDS.has(entity)) entities.add(entity)
    }
    for (const match of text.matchAll(QUOTED_ENTITY_PATTERN)) {
      const entity = match[1]
      if (entity && !COMMON_ENTITY_WORDS.has(entity)) entities.add(entity)
    }
    return [...entities]
  }

  /** Infer directed relationship triples where both ends are known entities. */
  extractRelationships(text: string, entities: readonly string[]): Array<readonly [string, string, string]> {
    const known = new Set(entities)
    const relationships: Array<readonly [string, string, string]> = []
    for (const [pattern, relation] of RELATIONSHIP_PATTERNS) {
      pattern.lastIndex = 0
      for (const match of text.matchAll(pattern)) {
        const source = match[1]
        const target = match.at(-1)
        if (source && target && known.has(source) && known.has(target)) {
          relationships.push([source, relation, target])
        }
      }
    }
    return relationships
  }

  /** Return tracked metadata, mentions, and incoming/outgoing relation edges. */
  getEntityInfo(entity: string): EntityInfo {
    const record = this.entities[entity]
    const relationships: EntityRelationship[] = []
    for (const [relation, pairs] of Object.entries(this.relationships)) {
      for (const [source, target] of pairs) {
        if (source === entity) relationships.push({ relation, target })
        else if (target === entity) relationships.push({ relation: `inverse_${relation}`, target: source })
      }
    }
    return {
      ...(record ? {
        contexts: [...record.contexts],
        firstSeen: record.firstSeen,
        frequency: record.frequency,
        lastSeen: record.lastSeen,
      } : {}),
      mentions: [...(this.entityMentions[entity] ?? [])],
      relationships,
    }
  }

  /** Walk relationship edges in either direction, excluding the source entity. */
  getRelatedEntities(entity: string, maxDepth = 2): Set<string> {
    const related = new Set<string>()
    const toExplore: Array<readonly [string, number]> = [[entity, 0]]
    const explored = new Set<string>()

    while (toExplore.length > 0) {
      const next = toExplore.shift()
      if (!next) continue
      const [current, depth] = next
      if (explored.has(current) || depth > maxDepth) continue
      explored.add(current)

      for (const pairs of Object.values(this.relationships)) {
        for (const [source, target] of pairs) {
          if (source === current) {
            related.add(target)
            if (depth < maxDepth) toExplore.push([target, depth + 1])
          } else if (target === current) {
            related.add(source)
            if (depth < maxDepth) toExplore.push([source, depth + 1])
          }
        }
      }
    }
    related.delete(entity)
    return related
  }

  retrieve(memoryId?: string, filters?: MemoryFilters, limit = 10): MemoryItem | MemoryItem[] | undefined {
    if (memoryId) return this.index.get(memoryId)
    return this.items.filter(item => this.matchesFilters(item, filters)).slice(0, limit)
  }

  save(content: string, metadata: MemoryMetadata = {}, options: EntitySaveOptions = {}): MemoryItem {
    const entities = options.entities?.length ? [...options.entities] : this.extractEntities(content)
    const item = new MemoryItem({
      content,
      memoryType: 'entity',
      metadata: { ...metadata, entities },
      ...(options.agentId ? { agentId: options.agentId } : {}),
      ...(options.taskId ? { taskId: options.taskId } : {}),
      ...(options.userId ? { userId: options.userId } : {}),
      ...(options.conversationId ? { conversationId: options.conversationId } : {}),
    })

    const evicted: MemoryItem[] = []
    if (this.maxItems !== undefined) {
      while (this.items.length >= this.maxItems) {
        const oldest = this.items[0]
        if (!oldest) break
        evicted.push(oldest)
        this.remove(oldest)
      }
    }
    this.append(item)
    this.rebuildGraph()
    try {
      this.persist(item)
      for (const removed of evicted) this.storage?.delete(entityStorageKey(removed.memoryId))
      return item
    } catch (error) {
      this.remove(item)
      for (const removed of evicted.reverse()) {
        this.items.unshift(removed)
        this.index.set(removed.memoryId, removed)
      }
      this.rebuildGraph()
      this.storage?.delete(entityStorageKey(item.memoryId))
      throw error
    }
  }

  search(query: string, limit = 10, filters?: MemoryFilters, options: EntitySearchOptions = {}): MemoryItem[] {
    const queryEntities = this.extractEntities(query)
    const targetEntities = options.entityFilter?.length ? options.entityFilter : queryEntities
    const target = new Set(targetEntities)
    const matches: MemoryItem[] = []

    for (const item of this.items) {
      if (!this.matchesFilters(item, filters)) continue
      const itemEntities = entityNames(item)
      const overlap = itemEntities.filter(entity => target.has(entity))
      if (target.size > 0 && overlap.length === 0) continue
      item.relevanceScore = target.size > 0
        ? overlap.length / target.size
        : item.content.toLowerCase().includes(query.toLowerCase()) ? 1 : 0.5
      matches.push(item)
    }
    return matches.sort((left, right) => right.relevanceScore - left.relevanceScore).slice(0, limit)
  }

  update(memoryId: string, updates: MemoryUpdate): boolean {
    const item = this.index.get(memoryId)
    if (!item) return false
    const original = MemoryItem.fromRecord(item.toRecord())
    const applied = updates.content === undefined
      ? updates
      : {
          ...updates,
          metadata: { ...item.metadata, ...(updates.metadata ?? {}), entities: this.extractEntities(updates.content) },
        }
    this.updateItem(item, applied)
    this.rebuildGraph()
    try {
      this.persist(item)
      return true
    } catch (error) {
      restoreItem(item, original)
      this.rebuildGraph()
      this.storage?.save(entityStorageKey(memoryId), original.toRecord())
      throw error
    }
  }

  private rebuildGraph(): void {
    clearRecord(this.entities)
    clearRecord(this.relationships)
    clearRecord(this.entityMentions)
    for (const item of this.items) {
      const entities = entityNames(item)
      for (const entity of entities) this.updateEntity(entity, item)
      for (const [source, relation, target] of this.extractRelationships(item.content, entities)) {
        const pairs = this.relationships[relation] ?? []
        pairs.push([source, target])
        this.relationships[relation] = pairs
      }
    }
  }

  private deletePersisted(targets: readonly MemoryItem[], operation: string): void {
    if (!this.storage) return
    const deleted: MemoryItem[] = []
    for (const item of targets) {
      if (!this.storage.delete(entityStorageKey(item.memoryId))) {
        for (const removed of deleted) this.storage.save(entityStorageKey(removed.memoryId), removed.toRecord())
        throw new Error(`Failed to ${operation} ${item.memoryId}`)
      }
      deleted.push(item)
    }
  }

  private hydrate(): void {
    if (!this.storage) return
    const records: MemoryItem[] = []
    for (const key of this.storage.listKeys('entity_')) {
      if (!key.startsWith('entity_')) continue
      try {
        const record = this.storage.load(key)
        if (isRecord(record)) records.push(MemoryItem.fromRecord(record))
      } catch (error) {
        console.warn(`Skipping corrupt entity memory record ${key}:`, error)
      }
    }
    records.sort((left, right) => left.timestamp.valueOf() - right.timestamp.valueOf())
    const overflow = this.maxItems === undefined ? 0 : Math.max(0, records.length - this.maxItems)
    for (const item of records.slice(overflow)) this.append(item)

    // Overflow rows beyond the cap are pruned from the backend, not merely
    // dropped in memory: otherwise every restart pays full reload cost for
    // records that can never be admitted again, and the row set grows
    // without bound (mirroring the eviction deletes save() performs).
    for (const item of records.slice(0, overflow)) {
      try {
        if (!this.storage?.delete(entityStorageKey(item.memoryId))) {
          console.warn(`Could not prune evicted entity memory ${item.memoryId}`)
        }
      } catch (error) {
        console.warn(`Could not prune evicted entity memory ${item.memoryId}:`, error)
      }
    }

    // The item rows are authoritative. Rebuilding prevents stale or partially
    // persisted snapshots from resurrecting evicted or deleted graph nodes.
    this.rebuildGraph()
  }

  private persist(item: MemoryItem): void {
    if (!this.storage) return
    if (!this.storage.save(entityStorageKey(item.memoryId), item.toRecord())) {
      throw new Error(`Failed to persist entity memory ${item.memoryId}`)
    }
    this.persistSnapshots()
  }

  private persistSnapshots(): void {
    if (!this.storage) return
    if (!this.storage.save('_entity_entities', this.entities)
      || !this.storage.save('_entity_relationships', this.relationships)
      || !this.storage.save('_entity_mentions', this.entityMentions)) {
      throw new Error('Failed to persist entity memory snapshots')
    }
  }

  private updateEntity(entity: string, item: MemoryItem): void {
    const existing = this.entities[entity]
    if (existing) {
      existing.frequency += 1
      existing.lastSeen = item.timestamp
      existing.contexts.push(item.content.slice(0, 100))
      if (existing.contexts.length > MAX_ENTITY_CONTEXTS) {
        existing.contexts.splice(0, existing.contexts.length - MAX_ENTITY_CONTEXTS)
      }
    } else {
      this.entities[entity] = {
        firstSeen: item.timestamp,
        frequency: 1,
        lastSeen: item.timestamp,
        contexts: [item.content.slice(0, 100)],
      }
    }
    const mentions = this.entityMentions[entity] ?? []
    mentions.push(item.memoryId)
    this.entityMentions[entity] = mentions
  }
}

function clearRecord(record: Record<string, unknown>): void {
  for (const key of Object.keys(record)) delete record[key]
}

function entityNames(item: MemoryItem): string[] {
  const entities = item.metadata.entities
  return Array.isArray(entities) ? entities.filter((entity): entity is string => typeof entity === 'string') : []
}

function entityStorageKey(memoryId: string): string {
  return `entity_${memoryId}`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function restoreItem(target: MemoryItem, source: MemoryItem): void {
  target.accessCount = source.accessCount
  target.agentId = source.agentId
  target.content = source.content
  target.conversationId = source.conversationId
  target.embedding = source.embedding ? [...source.embedding] : undefined
  target.lastAccessed = source.lastAccessed
  target.memoryType = source.memoryType
  target.metadata = { ...source.metadata }
  target.relevanceScore = source.relevanceScore
  target.taskId = source.taskId
  target.timestamp = source.timestamp
  target.userId = source.userId
}
