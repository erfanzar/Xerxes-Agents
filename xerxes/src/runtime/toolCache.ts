// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { statSync } from 'node:fs'
import { dirname } from 'node:path'

/** The fixed read-only allowlist inherited from Xerxes' tool-output cache. */
export const CACHEABLE_TOOL_NAMES = Object.freeze([
  'ReadFile',
  'GlobTool',
  'GrepTool',
  'ListDir',
] as const)

export type ToolCacheInput = Readonly<Record<string, unknown>>
export type ToolCacheExecutor = (toolName: string, toolInput: ToolCacheInput) => string | Promise<string>
export type CachedToolCacheExecutor = (toolName: string, toolInput: ToolCacheInput) => Promise<string>

export interface ToolCacheFileMetadata {
  readonly mtimeMs: number
  readonly size: number
}

export type ToolCacheFileStat = (path: string) => ToolCacheFileMetadata | undefined

export interface ToolOutputCacheOptions {
  /** Injectable file metadata source, useful for virtual filesystems and deterministic tests. */
  readonly fileStat?: ToolCacheFileStat
  /**
   * Maximum UTF-8 byte size of one cached result. Oversized results are
   * executed but never stored, so a single unbounded ReadFile/GrepTool output
   * cannot pin hundreds of megabytes in the cache (200 entries × unbounded
   * results would otherwise grow the heap without limit).
   */
  readonly maxEntryBytes?: number
  readonly maxEntries?: number
  /** Injected monotonic seconds clock; wall time is deliberately never used. */
  readonly monotonicNow?: () => number
  readonly ttlSeconds?: number
}

export interface ToolCacheInvalidationOptions {
  readonly filePath?: string
  readonly toolName?: string
}

export interface ToolOutputCacheStats {
  readonly hitRate: number
  readonly hits: number
  readonly misses: number
  readonly size: number
}

/** Raised when cache configuration or a cache request is invalid. */
export class ToolOutputCacheError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ToolOutputCacheError'
  }
}

interface CacheEntry {
  readonly filePaths: readonly string[]
  readonly insertedAt: number
  readonly result: string
  readonly toolName: string
}

const CACHEABLE_TOOLS = new Set<string>(CACHEABLE_TOOL_NAMES)
const DEFAULT_MAX_ENTRIES = 200
const DEFAULT_MAX_ENTRY_BYTES = 256 * 1_024
const DEFAULT_TTL_SECONDS = 300

/**
 * Bounded LRU cache for output from explicitly allowlisted read-only tools.
 *
 * Keys combine a stable content digest with mtime/size signatures for path
 * arguments. The cache does not configure logging, inspect environment, or
 * make writes; callers control invalidation after mutations.
 *
 * Known freshness caveat: invalidation relies solely on mtime + size. A
 * rewrite that preserves both values (e.g. via `utimes` after a same-length
 * edit) is served stale until the entry's TTL expires or it is explicitly
 * invalidated. Callers that mutate files must invalidate affected paths.
 */
export class ToolOutputCache {
  private readonly entries = new Map<string, CacheEntry>()
  private readonly fileStat: ToolCacheFileStat
  private readonly maxEntries: number
  private readonly maxEntryBytes: number
  private readonly monotonicNow: () => number
  private readonly ttlSeconds: number
  private hits = 0
  private misses = 0

  constructor(options: ToolOutputCacheOptions = {}) {
    this.maxEntries = requirePositiveInteger(options.maxEntries ?? DEFAULT_MAX_ENTRIES, 'maxEntries')
    this.maxEntryBytes = requirePositiveInteger(options.maxEntryBytes ?? DEFAULT_MAX_ENTRY_BYTES, 'maxEntryBytes')
    this.ttlSeconds = requireNonNegativeFinite(options.ttlSeconds ?? DEFAULT_TTL_SECONDS, 'ttlSeconds')
    this.monotonicNow = options.monotonicNow ?? (() => performance.now() / 1_000)
    this.fileStat = options.fileStat ?? defaultFileStat
  }

  get hitRate(): number {
    const total = this.hits + this.misses
    return total === 0 ? 0 : this.hits / total
  }

  get size(): number {
    return this.entries.size
  }

  get stats(): ToolOutputCacheStats {
    return Object.freeze({
      hitRate: this.hitRate,
      hits: this.hits,
      misses: this.misses,
      size: this.size,
    })
  }

  /** Return a cached output or undefined after a miss, TTL expiry, or non-read-only call. */
  get(toolName: string, toolInput: ToolCacheInput): string | undefined {
    if (!isCacheableTool(toolName)) return undefined
    const key = this.makeKey(toolName, toolInput).key
    const entry = this.entries.get(key)
    if (!entry) {
      this.misses += 1
      return undefined
    }
    if (this.elapsed(entry.insertedAt) > this.ttlSeconds) {
      this.entries.delete(key)
      this.misses += 1
      return undefined
    }
    this.entries.delete(key)
    this.entries.set(key, entry)
    this.hits += 1
    return entry.result
  }

  /**
   * Store one read-only output and evict least-recently-used entries over
   * capacity. Results larger than `maxEntryBytes` are deliberately skipped
   * rather than truncated so the cache never serves partial tool output.
   */
  put(toolName: string, toolInput: ToolCacheInput, result: string): void {
    if (!isCacheableTool(toolName)) return
    if (typeof result !== 'string') throw new ToolOutputCacheError('tool cache result must be a string')
    if (Buffer.byteLength(result, 'utf8') > this.maxEntryBytes) return
    const cacheKey = this.makeKey(toolName, toolInput)
    this.entries.delete(cacheKey.key)
    this.entries.set(cacheKey.key, {
      filePaths: cacheKey.filePaths,
      insertedAt: this.now(),
      result,
      toolName,
    })
    while (this.entries.size > this.maxEntries) {
      const oldest = this.entries.keys().next().value
      if (oldest === undefined) return
      this.entries.delete(oldest)
    }
  }

  /**
   * Remove cache entries by tool, path, both, or all entries when no filters are supplied.
   *
   * A path filter matches the file/directory dependencies extracted when the
   * entry was cached, including the parent directory of a slash-containing glob.
   */
  invalidate(options: ToolCacheInvalidationOptions = {}): number {
    if (options.toolName === undefined && options.filePath === undefined) {
      const count = this.entries.size
      this.entries.clear()
      return count
    }
    if (options.filePath !== undefined && (typeof options.filePath !== 'string' || !options.filePath)) {
      throw new ToolOutputCacheError('filePath must be a non-empty string')
    }
    if (options.toolName !== undefined && (typeof options.toolName !== 'string' || !options.toolName)) {
      throw new ToolOutputCacheError('toolName must be a non-empty string')
    }
    let removed = 0
    for (const [key, entry] of this.entries) {
      if (options.toolName !== undefined && entry.toolName !== options.toolName) continue
      if (options.filePath !== undefined && !entry.filePaths.includes(options.filePath)) continue
      this.entries.delete(key)
      removed += 1
    }
    return removed
  }

  /** Wrap synchronous or asynchronous executors in one awaitable cache boundary. */
  wrap(executor: ToolCacheExecutor): CachedToolCacheExecutor {
    if (typeof executor !== 'function') throw new ToolOutputCacheError('tool cache executor must be a function')
    return async (toolName, toolInput) => {
      const cached = this.get(toolName, toolInput)
      if (cached !== undefined) return cached
      const result = await executor(toolName, toolInput)
      this.put(toolName, toolInput, result)
      return result
    }
  }

  private elapsed(insertedAt: number): number {
    return this.now() - insertedAt
  }

  private makeKey(
    toolName: string,
    toolInput: ToolCacheInput,
  ): { readonly filePaths: readonly string[]; readonly key: string } {
    const filePaths = extractToolFilePaths(toolInput)
    const signature = filePaths.length === 0 ? 'no-files' : this.fileSignature(filePaths)
    return { filePaths, key: toolCacheContentKey(toolName, toolInput) + ':' + signature }
  }

  private fileSignature(paths: readonly string[]): string {
    return paths.map(path => {
      try {
        const metadata = this.fileStat(path)
        if (!metadata) return 'missing'
        const mtimeMs = requireNonNegativeFinite(metadata.mtimeMs, 'file mtimeMs')
        const size = requireNonNegativeFinite(metadata.size, 'file size')
        return `${mtimeMs}:${size}`
      } catch {
        return 'missing'
      }
    }).join('|')
  }

  private now(): number {
    return requireNonNegativeFinite(this.monotonicNow(), 'monotonicNow result')
  }
}

/** Extract dependency paths used for mtime/size cache invalidation. */
export function extractToolFilePaths(toolInput: ToolCacheInput): string[] {
  requireInput(toolInput)
  const paths: string[] = []
  for (const key of ['file_path', 'path', 'directory', 'dir']) {
    const value = toolInput[key]
    if (typeof value === 'string') paths.push(value)
  }
  const pattern = toolInput.pattern
  if (typeof pattern === 'string' && pattern.includes('/')) paths.push(dirname(pattern))
  return paths
}

/** Deterministic 16-character SHA-256 digest for a tool name and JSON-like input. */
export function toolCacheContentKey(toolName: string, toolInput: ToolCacheInput): string {
  if (typeof toolName !== 'string' || !toolName) {
    throw new ToolOutputCacheError('toolName must be a non-empty string')
  }
  requireInput(toolInput)
  return createHash('sha256').update(stableJson({ input: toolInput, tool: toolName }), 'utf8').digest('hex').slice(0, 16)
}

function defaultFileStat(path: string): ToolCacheFileMetadata | undefined {
  try {
    const details = statSync(path)
    return { mtimeMs: details.mtimeMs, size: details.size }
  } catch {
    return undefined
  }
}

function isCacheableTool(toolName: string): boolean {
  return typeof toolName === 'string' && CACHEABLE_TOOLS.has(toolName)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function requireInput(value: unknown): asserts value is ToolCacheInput {
  if (!isRecord(value)) throw new ToolOutputCacheError('toolInput must be an object')
}

function requireNonNegativeFinite(value: number, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new ToolOutputCacheError(name + ' must be a non-negative finite number')
  }
  return value
}

function requirePositiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new ToolOutputCacheError(name + ' must be a positive integer')
  }
  return value
}

function stableJson(value: unknown): string {
  if (value === null) return 'null'
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') return Number.isFinite(value) ? JSON.stringify(value) : JSON.stringify(String(value))
  if (typeof value === 'bigint') return JSON.stringify(value.toString())
  if (Array.isArray(value)) return '[' + value.map(stableJson).join(',') + ']'
  if (isRecord(value)) {
    return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + stableJson(value[key])).join(',') + '}'
  }
  return JSON.stringify(String(value))
}
