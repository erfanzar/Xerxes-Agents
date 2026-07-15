// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'

import type { ToolExecutionContext, ToolExecutor } from '../executors/toolRegistry.js'
import type { ToolCall } from '../types/toolCalls.js'

export const JitterMode = {
  NONE: 'none',
  FULL: 'full',
  EQUAL: 'equal',
  DECORRELATED: 'decorrelated',
} as const

export type JitterMode = (typeof JitterMode)[keyof typeof JitterMode]

export interface BackoffPolicyOptions {
  readonly baseDelay?: number
  readonly maxDelay?: number
  readonly mode?: JitterMode
  readonly multiplier?: number
  readonly random?: () => number
}

/** Exponential backoff schedule with deterministic injection points for tests. */
export class BackoffPolicy {
  readonly baseDelay: number
  readonly maxDelay: number
  readonly mode: JitterMode
  readonly multiplier: number
  private readonly random: () => number

  constructor(options: BackoffPolicyOptions = {}) {
    this.baseDelay = validatePositive(options.baseDelay ?? 0.5, 'baseDelay')
    this.maxDelay = validatePositive(options.maxDelay ?? 30, 'maxDelay')
    this.multiplier = validatePositive(options.multiplier ?? 2, 'multiplier')
    this.mode = options.mode ?? JitterMode.FULL
    this.random = options.random ?? Math.random
    if (!Object.values(JitterMode).includes(this.mode)) {
      throw new RangeError('Unknown jitter mode: ' + this.mode)
    }
  }

  delay(attempt: number, lastDelay = 0): number {
    const normalizedAttempt = Math.max(0, Math.floor(attempt))
    const cap = Math.max(this.baseDelay, this.maxDelay)
    const exponential = Math.min(cap, this.baseDelay * this.multiplier ** normalizedAttempt)
    switch (this.mode) {
      case JitterMode.NONE:
        return exponential
      case JitterMode.FULL:
        return this.random() * exponential
      case JitterMode.EQUAL:
        return exponential / 2 + this.random() * exponential / 2
      case JitterMode.DECORRELATED: {
        const minimum = this.baseDelay
        const maximum = Math.max(minimum, Math.max(minimum, lastDelay) * this.multiplier)
        return Math.min(cap, minimum + this.random() * (maximum - minimum))
      }
    }
  }

  *delays(maxAttempts: number): IterableIterator<number> {
    let lastDelay = 0
    for (let attempt = 0; attempt < Math.max(0, Math.floor(maxAttempts)); attempt += 1) {
      const delay = this.delay(attempt, lastDelay)
      lastDelay = delay
      yield delay
    }
  }
}

export interface ToolResultCacheOptions {
  readonly maxEntries?: number
  readonly now?: () => number
  readonly ttl?: number
}

interface CacheEntry<T> {
  readonly insertedAt: number
  readonly value: T
}

/** Small deterministic LRU + TTL cache for explicitly idempotent tool calls. */
export class ToolResultCache<T = string> {
  private readonly entries = new Map<string, CacheEntry<T>>()
  private readonly maxEntries: number
  private readonly now: () => number
  private readonly ttl: number
  private hits = 0
  private misses = 0

  constructor(options: ToolResultCacheOptions = {}) {
    this.maxEntries = validateInteger(options.maxEntries ?? 256, 'maxEntries', 1)
    this.ttl = validatePositive(options.ttl ?? 30, 'ttl')
    this.now = options.now ?? (() => performance.now() / 1_000)
  }

  get size(): number {
    return this.entries.size
  }

  get statistics(): Readonly<{ hits: number; misses: number }> {
    return Object.freeze({ hits: this.hits, misses: this.misses })
  }

  get(toolName: string, arguments_: unknown): T | undefined {
    const key = cacheKey(toolName, arguments_)
    const entry = this.entries.get(key)
    if (!entry || this.now() - entry.insertedAt > this.ttl) {
      if (entry) this.entries.delete(key)
      this.misses += 1
      return undefined
    }
    this.entries.delete(key)
    this.entries.set(key, entry)
    this.hits += 1
    return entry.value
  }

  set(toolName: string, arguments_: unknown, value: T): void {
    const key = cacheKey(toolName, arguments_)
    this.entries.delete(key)
    this.entries.set(key, { value, insertedAt: this.now() })
    while (this.entries.size > this.maxEntries) {
      const oldest = this.entries.keys().next().value
      if (oldest === undefined) break
      this.entries.delete(oldest)
    }
  }

  getOrSet(toolName: string, arguments_: unknown, producer: () => T): { readonly hit: boolean; readonly value: T } {
    const cached = this.get(toolName, arguments_)
    if (cached !== undefined) return { hit: true, value: cached }
    const value = producer()
    this.set(toolName, arguments_, value)
    return { hit: false, value }
  }

  async getOrSetAsync(
    toolName: string,
    arguments_: unknown,
    producer: () => Promise<T>,
  ): Promise<{ readonly hit: boolean; readonly value: T }> {
    const cached = this.get(toolName, arguments_)
    if (cached !== undefined) return { hit: true, value: cached }
    const value = await producer()
    this.set(toolName, arguments_, value)
    return { hit: false, value }
  }

  invalidate(toolName?: string): number {
    if (toolName === undefined) {
      const count = this.entries.size
      this.entries.clear()
      return count
    }
    let removed = 0
    for (const key of [...this.entries.keys()]) {
      if (key.startsWith(toolName + ':')) {
        this.entries.delete(key)
        removed += 1
      }
    }
    return removed
  }
}

export interface CachingToolExecutorOptions {
  /** Only listed idempotent tool names can be cached. The default caches nothing. */
  readonly cacheableTools?: Iterable<string>
  readonly cache?: ToolResultCache<string>
}

/** ToolExecutor decorator that cannot cache mutations unless callers opt in by exact name. */
export class CachingToolExecutor implements ToolExecutor {
  readonly cache: ToolResultCache<string>
  private readonly cacheableTools: ReadonlySet<string>

  constructor(
    private readonly delegate: ToolExecutor,
    options: CachingToolExecutorOptions = {},
  ) {
    this.cache = options.cache ?? new ToolResultCache<string>()
    this.cacheableTools = new Set(options.cacheableTools ?? [])
  }

  execute(call: ToolCall, context: ToolExecutionContext, signal?: AbortSignal): Promise<string> {
    if (!this.cacheableTools.has(call.function.name)) {
      return this.delegate.execute(call, context, signal)
    }
    return this.cache.getOrSetAsync(
      call.function.name,
      call.function.arguments,
      () => this.delegate.execute(call, context, signal),
    ).then(result => result.value)
  }
}

/** Stable MD5 digest retained for Python-compatible cache-key values. */
export function hashArgs(value: unknown): string {
  return createHash('md5').update(stableJson(value), 'utf8').digest('hex')
}

function cacheKey(toolName: string, arguments_: unknown): string {
  return toolName + ':' + hashArgs(arguments_)
}

function stableJson(value: unknown): string {
  if (value === null) return 'null'
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'boolean' || typeof value === 'number') return JSON.stringify(value)
  if (Array.isArray(value)) return '[' + value.map(stableJson).join(',') + ']'
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>
    return '{' + Object.keys(record).sort().map(key => JSON.stringify(key) + ':' + stableJson(record[key])).join(',') + '}'
  }
  return JSON.stringify(String(value))
}

function validatePositive(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(name + ' must be positive')
  return value
}

function validateInteger(value: number, name: string, minimum: number): number {
  if (!Number.isInteger(value) || value < minimum) throw new RangeError(name + ' must be an integer of at least ' + minimum)
  return value
}
