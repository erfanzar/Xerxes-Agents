// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { MemoryStorage } from '../memory/storage.js'

export const IDENTITY_KEY_PREFIX = '_identity_'

export interface IdentityRecord {
  readonly channel: string
  readonly channelUserId: string
  readonly displayName: string
  readonly firstSeen: string
  readonly userId: string
}

export interface IdentityResolverOptions {
  /** Optional synchronous Xerxes storage used to retain aliases across restarts. */
  readonly storage?: MemoryStorage
  /** Injectable clock for deterministic hosts and tests. */
  readonly clock?: () => Date
  /** Injectable global-id factory; production defaults to crypto.randomUUID. */
  readonly userIdFactory?: () => string
}

/**
 * Resolves platform-local identities into stable global Xerxes user IDs.
 *
 * Persistence is intentionally best effort: an unavailable optional memory
 * store must not make an inbound chat delivery fail. Stored records use the
 * Python-readable snake_case shape so a cutover can hydrate existing aliases.
 */
export class IdentityResolver {
  private readonly clock: () => Date
  private readonly records = new Map<string, IdentityRecord>()
  private readonly storage: MemoryStorage | undefined
  private readonly userIdFactory: () => string

  constructor(options: IdentityResolverOptions = {}) {
    this.clock = options.clock ?? (() => new Date())
    this.storage = options.storage
    this.userIdFactory = options.userIdFactory ?? (() => crypto.randomUUID())
    this.hydrate()
  }

  /** Return a snapshot of every known channel-side alias for a global user. */
  channelsFor(userId: string): IdentityRecord[] {
    return [...this.records.values()]
      .filter(record => record.userId === userId)
      .map(copyRecord)
  }

  /** Return a snapshot of every record in creation/hydration order. */
  all(): IdentityRecord[] {
    return [...this.records.values()].map(copyRecord)
  }

  /** Look up an alias without creating it. */
  get(channel: string, channelUserId: string): IdentityRecord | undefined {
    return copyOptional(this.records.get(identityKey(channel, channelUserId)))
  }

  /**
   * Force a channel identity to reference a supplied global ID.
   *
   * This lets a trusted operator merge accounts from separate platforms.
   */
  link(userId: string, channel: string, channelUserId: string): IdentityRecord {
    const normalizedUserId = requiredValue(userId, 'userId')
    const normalizedChannel = requiredValue(channel, 'channel')
    const normalizedChannelUserId = requiredValue(channelUserId, 'channelUserId')
    const key = identityKey(normalizedChannel, normalizedChannelUserId)
    const existing = this.records.get(key)
    if (existing?.userId === normalizedUserId) return copyRecord(existing)
    const record: IdentityRecord = existing === undefined
      ? {
        userId: normalizedUserId,
        channel: normalizedChannel,
        channelUserId: normalizedChannelUserId,
        displayName: '',
        firstSeen: validTimestamp(this.clock()),
      }
      : { ...existing, userId: normalizedUserId }
    this.records.set(key, record)
    this.persist(key, record)
    return copyRecord(record)
  }

  /**
   * Return an existing identity or mint a globally unique ID on first sight.
   *
   * A non-empty display name fills an older blank name but never replaces an
   * already-known one, preserving the original user-selected attribution.
   */
  resolve(channel: string, channelUserId: string, displayName = ''): IdentityRecord {
    const normalizedChannel = requiredValue(channel, 'channel')
    const normalizedChannelUserId = requiredValue(channelUserId, 'channelUserId')
    const key = identityKey(normalizedChannel, normalizedChannelUserId)
    const existing = this.records.get(key)
    if (existing !== undefined) {
      if (!existing.displayName && displayName) {
        const updated = { ...existing, displayName }
        this.records.set(key, updated)
        this.persist(key, updated)
        return copyRecord(updated)
      }
      return copyRecord(existing)
    }
    const record: IdentityRecord = {
      userId: requiredValue(this.userIdFactory(), 'userIdFactory result'),
      channel: normalizedChannel,
      channelUserId: normalizedChannelUserId,
      displayName,
      firstSeen: validTimestamp(this.clock()),
    }
    this.records.set(key, record)
    this.persist(key, record)
    return copyRecord(record)
  }

  private hydrate(): void {
    if (this.storage === undefined) return
    let keys: string[]
    try {
      keys = this.storage.listKeys(IDENTITY_KEY_PREFIX)
    } catch {
      return
    }
    for (const key of keys) {
      if (!key.startsWith(IDENTITY_KEY_PREFIX)) continue
      try {
        const record = parseStoredRecord(this.storage.load(key))
        if (record !== undefined) this.records.set(key, record)
      } catch {
        // Optional historical records must not block startup when malformed.
      }
    }
  }

  private persist(key: string, record: IdentityRecord): void {
    if (this.storage === undefined) return
    try {
      this.storage.save(key, storedRecord(record))
    } catch {
      // Identity lookup remains available when optional persistence is down.
    }
  }
}

/** Build the canonical storage key for a platform-local identity. */
export function identityKey(channel: string, channelUserId: string): string {
  return IDENTITY_KEY_PREFIX + requiredValue(channel, 'channel') + '::' + requiredValue(channelUserId, 'channelUserId')
}

function copyOptional(record: IdentityRecord | undefined): IdentityRecord | undefined {
  return record === undefined ? undefined : copyRecord(record)
}

function copyRecord(record: IdentityRecord): IdentityRecord {
  return { ...record }
}

function parseStoredRecord(value: unknown): IdentityRecord | undefined {
  if (!isRecord(value)) return undefined
  const userId = stringField(value, 'user_id') ?? stringField(value, 'userId')
  const channel = stringField(value, 'channel')
  const channelUserId = stringField(value, 'channel_user_id') ?? stringField(value, 'channelUserId')
  if (!userId || !channel || !channelUserId) return undefined
  return {
    userId,
    channel,
    channelUserId,
    displayName: stringField(value, 'display_name') ?? stringField(value, 'displayName') ?? '',
    firstSeen: stringField(value, 'first_seen') ?? stringField(value, 'firstSeen') ?? '',
  }
}

function storedRecord(record: IdentityRecord): Record<string, string> {
  return {
    user_id: record.userId,
    channel: record.channel,
    channel_user_id: record.channelUserId,
    display_name: record.displayName,
    first_seen: record.firstSeen,
  }
}

function requiredValue(value: string, name: string): string {
  const normalized = value.trim()
  if (!normalized) throw new TypeError(name + ' must not be blank')
  return normalized
}

function validTimestamp(value: Date): string {
  if (!Number.isFinite(value.getTime())) throw new TypeError('clock returned an invalid date')
  return value.toISOString()
}

function stringField(value: Record<string, unknown>, key: string): string | undefined {
  const field = value[key]
  return typeof field === 'string' ? field : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
