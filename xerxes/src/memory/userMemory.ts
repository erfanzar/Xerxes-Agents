// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'

import { scanContextContent } from '../security/promptScanner.js'
import {
  ContextualMemory,
  type ContextualSaveOptions,
} from './contextualMemory.js'
import { EntityMemory, type EntitySaveOptions } from './entityMemory.js'
import type { MemoryItem, MemoryMetadata, MemorySearchOptions } from './base.js'
import { NamespacedStorage, type MemoryStorage } from './storage.js'

const LEGACY_USER_PREFERENCES_KEY = '_user_preferences'
const USER_PREFERENCES_KEY_SUFFIX = 'user_preferences'

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

    if (this.userPreferences.delete(userId)) {
      // The per-user preference record goes with the user, and any
      // not-yet-migrated copy inside the legacy wholesale blob is scrubbed
      // too: otherwise a restart after clearing an inactive user would
      // resurrect stale preferences straight out of that blob.
      this.storage?.delete(preferenceStorageKey(userId))
      this.scrubLegacyPreferences(userId)
    }
  }

  /** Return a user-local contextual store, creating its entity and preference state if needed. */
  getOrCreateUserMemory(userId: string): ContextualMemory {
    const existing = this.userMemories.get(userId)
    if (existing) return existing

    // Persist under a per-user key namespace so tiers over a shared backend
    // only hydrate, search, and clear their own user's records. The namespace
    // segment is a hash of the user id, never the raw id: with raw ids the
    // prefix `user_a_` also matches user `a_b`'s keys, letting one user's
    // clear()/listKeys() delete or expose another user's records. The raw id
    // stays on the records themselves (metadata.user_id / options.userId).
    const storage = this.storage ? new NamespacedStorage(this.storage, userNamespace(userId)) : undefined
    const memory = new ContextualMemory({
      ...(storage ? { longTermStorage: storage } : {}),
      // Low-importance memories land in the short-term tier; without a durable
      // backend here they would vanish across restarts while only the
      // long-term side persisted.
      ...(storage ? { shortTermStorage: storage } : {}),
    })
    const entityMemory = new EntityMemory({
      ...(storage ? { storage } : {}),
    })
    this.userMemories.set(userId, memory)
    this.userEntities.set(userId, entityMemory)
    if (!this.userPreferences.has(userId)) this.userPreferences.set(userId, defaultPreferences())
    this.saveUserPreferences(userId)
    return memory
  }

  getUserContext(userId: string): string {
    const memory = this.getOrCreateUserMemory(userId)
    const entityMemory = this.userEntities.get(userId)
    const parts: string[] = []
    const preferences = this.getUserPreferences(userId)
    // Preference values and entity names are user- or tool-influenced data
    // recalled into prompts; neutralise embedded hostile instructions.
    if (Object.keys(preferences).length > 0) {
      parts.push(`User preferences: ${scanContextContent(JSON.stringify(preferences), `user preferences for ${userId}`)}`)
    }
    parts.push(memory.getContextSummary())
    if (entityMemory && Object.keys(entityMemory.entities).length > 0) {
      const names = Object.keys(entityMemory.entities).slice(0, 10).join(', ')
      parts.push(`Known entities: ${scanContextContent(names, `known entities for ${userId}`)}`)
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
    this.saveUserPreferences(userId)
  }

  private loadUsers(): void {
    if (!this.storage) return
    // Per-user namespaced records are authoritative once written.
    for (const key of this.storage.listKeys(USER_PREFERENCES_KEY_SUFFIX)) {
      if (!key.endsWith(`_${USER_PREFERENCES_KEY_SUFFIX}`)) continue
      const stored = this.storage.load(key)
      if (!isPreferenceRecord(stored)) continue
      this.userPreferences.set(stored.user_id, { ...stored.preferences })
    }
    // Legacy wholesale blob, migrated on read: distribute every entry to its
    // namespaced key once, then rewrite the blob without the entries whose
    // namespaced records now carry them (including copies that already did).
    // A blob left intact would resurrect users this facade cleared — its
    // stale copy outliving their deletion — so after migration it only ever
    // holds entries migration could not distribute. Never deleted outright:
    // older readers keep a valid, if aging, view.
    if (!this.storage.exists(LEGACY_USER_PREFERENCES_KEY)) return
    const stored = this.storage.load(LEGACY_USER_PREFERENCES_KEY)
    if (!isRecord(stored)) return
    let migratedAny = false
    const remaining: Record<string, unknown> = {}
    for (const [userId, preferences] of Object.entries(stored)) {
      const key = preferenceStorageKey(userId)
      if (isRecord(preferences) && !this.storage.exists(key)) {
        this.userPreferences.set(userId, { ...preferences })
        this.storage.save(key, preferenceRecord(userId, preferences))
        migratedAny = true
        continue
      }
      // Namespaced record already authoritative, or entry too malformed to
      // distribute; malformed shapes stay in the blob rather than being
      // silently discarded.
      if (isRecord(preferences)) migratedAny = true
      else remaining[userId] = preferences
    }
    if (migratedAny) this.storage.save(LEGACY_USER_PREFERENCES_KEY, remaining)
  }

  /** Remove one user's entry from the legacy wholesale blob, when present. */
  private scrubLegacyPreferences(userId: string): void {
    if (!this.storage?.exists(LEGACY_USER_PREFERENCES_KEY)) return
    const stored = this.storage.load(LEGACY_USER_PREFERENCES_KEY)
    if (!isRecord(stored) || !(userId in stored)) return
    const remaining = { ...stored }
    delete remaining[userId]
    this.storage.save(LEGACY_USER_PREFERENCES_KEY, remaining)
  }

  /**
   * Persist one user's preferences under their own namespaced key.
   *
   * A single shared blob made every writer's save a whole-map last-write-wins
   * across processes; distinct keys mean concurrent facades updating
   * different users never touch each other's records, and the read-merge
   * keeps same-user writers from dropping fields they never saw.
   */
  private saveUserPreferences(userId: string): void {
    if (!this.storage) return
    const stored = this.storage.load(preferenceStorageKey(userId))
    const merged: UserPreferences = {
      ...(isPreferenceRecord(stored) ? stored.preferences : {}),
      ...(this.userPreferences.get(userId) ?? {}),
    }
    this.storage.save(preferenceStorageKey(userId), preferenceRecord(userId, merged))
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

/**
 * Collision-proof per-user storage namespace. Hashing the user id removes
 * prefix ambiguity between ids such as `a` and `a_b` and keeps raw ids out
 * of backend keys.
 */
function userNamespace(userId: string): string {
  const digest = createHash('sha256').update(userId, 'utf8').digest('hex').slice(0, 16)
  return `user_${digest}_`
}

/** Per-user preference record key: the namespaced prefix plus a fixed suffix. */
function preferenceStorageKey(userId: string): string {
  return `${userNamespace(userId)}${USER_PREFERENCES_KEY_SUFFIX}`
}

function preferenceRecord(userId: string, preferences: Readonly<UserPreferences>): { readonly preferences: UserPreferences; readonly user_id: string } {
  return { preferences: { ...preferences }, user_id: userId }
}

function isPreferenceRecord(value: unknown): value is { readonly preferences: UserPreferences; readonly user_id: string } {
  return isRecord(value)
    && typeof value.user_id === 'string'
    && value.user_id.length > 0
    && isRecord(value.preferences)
}

function relationshipCount(memory: EntityMemory): number {
  return Object.values(memory.relationships).reduce((count, pairs) => count + pairs.length, 0)
}
