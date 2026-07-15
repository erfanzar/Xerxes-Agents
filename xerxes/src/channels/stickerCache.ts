// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

export interface StickerRecord {
  readonly fetchedAt: number
  readonly localPath: string
  readonly platform: string
  readonly stickerId: string
}

export interface StickerCacheOptions {
  readonly clock?: () => number
  readonly lruSize?: number
}

/**
 * Bounded persistent LRU index for downloaded platform stickers and custom emoji.
 *
 * Cache entries refer to host-managed files; evicting an index entry never
 * deletes that file. Index writes are atomic so a failed write keeps the
 * previous on-disk view intact.
 */
export class StickerCache {
  readonly baseDirectory: string
  readonly indexPath: string

  private readonly clock: () => number
  private entries = new Map<string, StickerRecord>()
  private readonly lruSize: number

  constructor(baseDirectory: string, options: StickerCacheOptions = {}) {
    this.baseDirectory = resolve(requiredValue(baseDirectory, 'baseDirectory'))
    this.indexPath = join(this.baseDirectory, '_index.json')
    this.clock = options.clock ?? (() => Date.now() / 1_000)
    this.lruSize = positiveInteger(options.lruSize ?? 256, 'lruSize')
    mkdirSync(this.baseDirectory, { recursive: true })
    this.loadIndex()
  }

  /** Drop entries and rewrite the index without deleting downloaded media files. */
  clear(): void {
    const next = new Map<string, StickerRecord>()
    this.saveIndex(next)
    this.entries = next
  }

  /** Look up an entry and promote it to most-recently-used. */
  get(platform: string, stickerId: string): StickerRecord | undefined {
    const key = stickerKey(platform, stickerId)
    const record = this.entries.get(key)
    if (record === undefined) return undefined
    this.entries.delete(key)
    this.entries.set(key, record)
    return copyRecord(record)
  }

  /** Insert an entry, evicting the least-recently-used records over the cap. */
  put(platform: string, stickerId: string, localPath: string): StickerRecord {
    const record: StickerRecord = {
      platform: requiredValue(platform, 'platform'),
      stickerId: requiredValue(stickerId, 'stickerId'),
      localPath: requiredValue(localPath, 'localPath'),
      fetchedAt: timestamp(this.clock()),
    }
    const key = stickerKey(record.platform, record.stickerId)
    const next = new Map(this.entries)
    next.delete(key)
    next.set(key, record)
    while (next.size > this.lruSize) {
      const oldest = next.keys().next().value
      if (oldest === undefined) break
      next.delete(oldest)
    }
    this.saveIndex(next)
    this.entries = next
    return copyRecord(record)
  }

  /** Return the number of indexed records. */
  size(): number {
    return this.entries.size
  }

  private loadIndex(): void {
    if (!existsSync(this.indexPath)) return
    let parsed: unknown
    try {
      parsed = JSON.parse(readFileSync(this.indexPath, 'utf8')) as unknown
    } catch {
      return
    }
    if (!Array.isArray(parsed)) return
    for (const value of parsed) {
      const record = parseRecord(value)
      if (record === undefined) continue
      const key = stickerKey(record.platform, record.stickerId)
      this.entries.delete(key)
      this.entries.set(key, record)
      while (this.entries.size > this.lruSize) {
        const oldest = this.entries.keys().next().value
        if (oldest === undefined) break
        this.entries.delete(oldest)
      }
    }
  }

  private saveIndex(entries: ReadonlyMap<string, StickerRecord>): void {
    const body = JSON.stringify([...entries.values()], null, 2) + '\n'
    const temporary = this.indexPath + '.' + randomUUID() + '.tmp'
    try {
      writeFileSync(temporary, body, 'utf8')
      renameSync(temporary, this.indexPath)
    } catch (error) {
      rmSync(temporary, { force: true })
      throw error
    }
  }
}

function copyRecord(record: StickerRecord): StickerRecord {
  return { ...record }
}

function parseRecord(value: unknown): StickerRecord | undefined {
  if (!isRecord(value)) return undefined
  const fetchedAt = value.fetched_at ?? value.fetchedAt
  const localPath = value.local_path ?? value.localPath
  const snakeCaseStickerId = value.sticker_id
  const camelCaseStickerId = value.stickerId
  const stickerId = typeof snakeCaseStickerId === 'string' ? snakeCaseStickerId : camelCaseStickerId
  if (
    typeof value.platform !== 'string'
    || typeof stickerId !== 'string'
    || typeof localPath !== 'string'
    || typeof fetchedAt !== 'number'
    || !Number.isFinite(fetchedAt)
  ) {
    return undefined
  }
  return {
    platform: value.platform,
    stickerId,
    localPath,
    fetchedAt,
  }
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(name + ' must be a positive safe integer')
  }
  return value
}

function requiredValue(value: string, name: string): string {
  const normalized = value.trim()
  if (!normalized) throw new TypeError(name + ' must not be blank')
  return normalized
}

function stickerKey(platform: string, stickerId: string): string {
  return requiredValue(platform, 'platform') + '\u0000' + requiredValue(stickerId, 'stickerId')
}

function timestamp(value: number): number {
  if (!Number.isFinite(value)) throw new TypeError('clock must return a finite Unix timestamp')
  return value
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
