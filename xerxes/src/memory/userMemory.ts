// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  ContextualMemory,
  type ContextualSaveOptions,
} from './contextualMemory.js'
import { EntityMemory, type EntitySaveOptions } from './entityMemory.js'
import type { MemoryItem, MemoryMetadata, MemorySearchOptions } from './base.js'
import { NamespacedStorage, type MemoryStorage } from './storage.js'

const USER_PREFERENCES_KEY = '_user_preferences'

export type UserPreferences = Record<string, unknown>

export interface UserMemorySaveOptions extends ContextualSaveOptions, EntitySaveOptions {}

export interface UserMemoryStatistics {
  readonly entitiesKnown: number
  readonly longTermMemories?: number
  readonly preferences: UserPreferences
  readonly relationships?: number
  readonly shortTermMemories?: number
  readonly totalMemories: number
  readonly userId: string
}

/**
 * User-scoped facade that joins contextual recall, entity recall, and simple
 * persisted preferences without sharing an in-memory conversation between users.
 */
export class UserMemory {
  readonly userEntities = new Map<string, EntityMemory>()
  readonly userMemories = new Map<string, ContextualMemory>()
  readonly userPreferences = new Map<string, UserPreferences>()

  constructor(readonly storage?: MemoryStorage) {
    this.loadUsers()
  }

  clearUserMemory(userId: string): void {
    const memory = this.userMemories.get(userId)
    if (memory) {
      memory.clear()
      this.userMemories.delete(userId)
    }

    const entityMemory = this.userEntities.get(userId)
    if (entityMemory) {
      entityMemory.clear()
      this.userEntities.delete(userId)
    }

    if (this.userPreferences.delete(userId)) this.savePreferences()
  }

  /** Return a user-local contextual store, creating its entity and preference state if needed. */
  getOrCreateUserMemory(userId: string): ContextualMemory {
    const existing = this.userMemories.get(userId)
    if (existing) return existing

    // Persist under a per-user key namespace so tiers over a shared backend
    // only hydrate, search, and clear their own user's records.
    const storage = this.storage ? new NamespacedStorage(this.storage, `user_${userId}_`) : undefined
    const memory = new ContextualMemory({
      ...(storage ? { longTermStorage: storage } : {}),
    })
    const entityMemory = new EntityMemory({
      ...(storage ? { storage } : {}),
    })
    this.userMemories.set(userId, memory)
    this.userEntities.set(userId, entityMemory)
    if (!this.userPreferences.has(userId)) this.userPreferences.set(userId, defaultPreferences())
    this.savePreferences()
    return memory
  }

  getUserContext(userId: string): string {
    const memory = this.getOrCreateUserMemory(userId)
    const entityMemory = this.userEntities.get(userId)
    const parts: string[] = []
    const preferences = this.getUserPreferences(userId)
    if (Object.keys(preferences).length > 0) parts.push(`User preferences: ${JSON.stringify(preferences)}`)
    parts.push(memory.getContextSummary())
    if (entityMemory && Object.keys(entityMemory.entities).length > 0) {
      parts.push(`Known entities: ${Object.keys(entityMemory.entities).slice(0, 10).join(', ')}`)
    }
    return parts.join('\n\n')
  }

  getUserPreferences(userId: string): UserPreferences {
    return { ...(this.userPreferences.get(userId) ?? defaultPreferences()) }
  }

  getUserStatistics(userId: string): UserMemoryStatistics {
    const memory = this.userMemories.get(userId)
    const entityMemory = this.userEntities.get(userId)
    const statistics: UserMemoryStatistics = {
      userId,
      totalMemories: 0,
      entitiesKnown: 0,
      preferences: this.getUserPreferences(userId),
    }
    if (memory) {
      const shortTermMemories = memory.shortTerm.size
      const longTermMemories = memory.longTerm.size
      return {
        ...statistics,
        totalMemories: shortTermMemories + longTermMemories,
        shortTermMemories,
        longTermMemories,
        ...(entityMemory ? {
          entitiesKnown: Object.keys(entityMemory.entities).length,
          relationships: relationshipCount(entityMemory),
        } : {}),
      }
    }
    if (!entityMemory) return statistics
    return {
      ...statistics,
      entitiesKnown: Object.keys(entityMemory.entities).length,
      relationships: relationshipCount(entityMemory),
    }
  }

  saveMemory(
    userId: string,
    content: string,
    metadata: MemoryMetadata = {},
    options: UserMemorySaveOptions = {},
  ): MemoryItem {
    const memory = this.getOrCreateUserMemory(userId)
    const scopedMetadata = { ...metadata, user_id: userId }
    const scopedOptions: UserMemorySaveOptions = { ...options, userId }
    const item = memory.save(content, scopedMetadata, scopedOptions)
    this.userEntities.get(userId)?.save(content, scopedMetadata, scopedOptions)
    return item
  }

  searchUserMemory(userId: string, query: string, limit = 10, options: MemorySearchOptions = {}): MemoryItem[] {
    return this.getOrCreateUserMemory(userId).search(query, limit, undefined, options)
  }

  updateUserPreferences(userId: string, preferences: Readonly<UserPreferences>): void {
    this.userPreferences.set(userId, {
      ...(this.userPreferences.get(userId) ?? defaultPreferences()),
      ...preferences,
    })
    this.savePreferences()
  }

  private loadUsers(): void {
    if (!this.storage?.exists(USER_PREFERENCES_KEY)) return
    const stored = this.storage.load(USER_PREFERENCES_KEY)
    if (!isRecord(stored)) return
    for (const [userId, preferences] of Object.entries(stored)) {
      if (!isRecord(preferences)) continue
      this.userPreferences.set(userId, { ...preferences })
    }
  }

  private savePreferences(): void {
    if (!this.storage) return
    const serialized: Record<string, UserPreferences> = {}
    for (const [userId, preferences] of this.userPreferences) serialized[userId] = { ...preferences }
    this.storage.save(USER_PREFERENCES_KEY, serialized)
  }
}

function defaultPreferences(): UserPreferences {
  return {
    response_style: 'balanced',
    verbosity: 'normal',
    technical_level: 'intermediate',
    language: 'en',
    timezone: 'UTC',
    memory_enabled: true,
    max_context_items: 10,
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function relationshipCount(memory: EntityMemory): number {
  return Object.values(memory.relationships).reduce((count, pairs) => count + pairs.length, 0)
}
