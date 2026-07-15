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
  }

  clear(): void {
    this.items.length = 0
    this.index.clear()
    clearRecord(this.entities)
    clearRecord(this.relationships)
    clearRecord(this.entityMentions)

    for (const key of this.storage?.listKeys('entity_') ?? []) this.storage?.delete(key)
  }

  delete(memoryId?: string, _filters?: MemoryFilters): number {
    if (!memoryId) return 0
    const item = this.index.get(memoryId)
    if (!item) return 0

    for (const entity of entityNames(item)) removeMention(this.entityMentions, entity, memoryId)
    this.remove(item)
    this.storage?.delete(entityStorageKey(memoryId))
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

    for (const entity of entities) this.updateEntity(entity, item)
    for (const [source, relation, target] of this.extractRelationships(content, entities)) {
      const pairs = this.relationships[relation] ?? []
      pairs.push([source, target])
      this.relationships[relation] = pairs
    }

    this.append(item)
    this.persist(item)
    return item
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

    if (updates.content !== undefined) {
      const oldEntities = entityNames(item)
      const newEntities = this.extractEntities(updates.content)
      for (const entity of oldEntities) removeMention(this.entityMentions, entity, memoryId)
      for (const entity of newEntities) {
        const mentions = this.entityMentions[entity] ?? []
        mentions.push(memoryId)
        this.entityMentions[entity] = mentions
      }
      this.updateItem(item, {
        ...updates,
        metadata: { ...item.metadata, ...(updates.metadata ?? {}), entities: newEntities },
      })
    } else {
      this.updateItem(item, updates)
    }

    this.persist(item)
    return true
  }

  private persist(item: MemoryItem): void {
    if (!this.storage) return
    this.storage.save(entityStorageKey(item.memoryId), item.toRecord())
    this.storage.save('_entity_entities', this.entities)
    this.storage.save('_entity_relationships', this.relationships)
    this.storage.save('_entity_mentions', this.entityMentions)
  }

  private updateEntity(entity: string, item: MemoryItem): void {
    const existing = this.entities[entity]
    if (existing) {
      existing.frequency += 1
      existing.lastSeen = item.timestamp
      existing.contexts.push(item.content.slice(0, 100))
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

function removeMention(mentions: Record<string, string[]>, entity: string, memoryId: string): void {
  const values = mentions[entity]
  if (!values) return
  const position = values.indexOf(memoryId)
  if (position >= 0) values.splice(position, 1)
}
