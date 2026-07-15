// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, unlinkSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

export const DEFAULT_TOOL_RESULT_INLINE_LIMIT_CHARS = 16_000
export const DEFAULT_TOOL_RESULT_LRU_SIZE = 32
export const TOOL_RESULT_REFERENCE_PREFIX = '[tool-result-ref:'
export const TOOL_RESULT_REFERENCE_SUFFIX = ']'

export interface ToolResultStoredFile {
  readonly modifiedAt: number
  readonly name: string
}

/** Filesystem boundary for oversized tool-result retention. */
export interface ToolResultStorageFileSystem {
  exists(path: string): boolean
  list(directory: string): readonly ToolResultStoredFile[]
  makeDirectory(directory: string): void
  read(path: string): string
  remove(path: string): void
  write(path: string, content: string): void
}

export interface ToolResultStorageOptions {
  readonly fileSystem?: ToolResultStorageFileSystem
  readonly inlineLimit?: number
  readonly lruSize?: number
  readonly sessionId?: string
}

/**
 * LRU cache plus per-session disk storage for tool outputs too large for a transcript.
 *
 * References use the Python-compatible `[tool-result-ref:<id>:<chars>:<sha>]`
 * layout. The size field deliberately counts Unicode code points, matching
 * Python's `len(str)` behavior even though the historical label says `bytes`.
 */
export class ToolResultStorage {
  static readonly REF_PREFIX = TOOL_RESULT_REFERENCE_PREFIX
  static readonly REF_SUFFIX = TOOL_RESULT_REFERENCE_SUFFIX

  readonly directory: string
  readonly inlineLimit: number
  private readonly cache = new Map<string, unknown>()
  private readonly fileSystem: ToolResultStorageFileSystem
  private readonly lruSize: number

  constructor(baseDirectory: string, options: ToolResultStorageOptions = {}) {
    if (!baseDirectory) throw new TypeError('baseDirectory must be non-empty')
    const sessionId = options.sessionId ?? 'default'
    if (!sessionId) throw new TypeError('sessionId must be non-empty')
    this.inlineLimit = nonNegativeInteger(
      options.inlineLimit ?? DEFAULT_TOOL_RESULT_INLINE_LIMIT_CHARS,
      'inlineLimit',
    )
    this.lruSize = positiveInteger(options.lruSize ?? DEFAULT_TOOL_RESULT_LRU_SIZE, 'lruSize')
    this.fileSystem = options.fileSystem ?? nodeFileSystem
    this.directory = join(baseDirectory, sessionId)
    this.fileSystem.makeDirectory(this.directory)
  }

  /** Return the input unchanged unless its serialized representation exceeds the inline limit. */
  maybeStore(toolName: string, content: unknown): unknown {
    if (!toolName) throw new TypeError('toolName must be non-empty')
    const payload = serializeToolResultPayload(content)
    if (unicodeLength(payload) <= this.inlineLimit) return content

    const digest = sha1Short(payload)
    const referenceId = toolName + '_' + digest
    const path = this.pathFor(referenceId)
    if (!this.fileSystem.exists(path)) {
      this.fileSystem.write(path, payload)
    }
    this.remember(referenceId, content)
    return this.reference(referenceId, unicodeLength(payload), digest)
  }

  /** Resolve a reference placeholder or raw reference ID, checking memory before disk. */
  fetch(referenceOrId: string): unknown | undefined {
    if (typeof referenceOrId !== 'string') return undefined
    const referenceId = ToolResultStorage.parseRef(referenceOrId) ?? referenceOrId
    const cached = this.cache.get(referenceId)
    if (cached !== undefined || this.cache.has(referenceId)) {
      this.remember(referenceId, cached)
      return cached
    }

    const path = this.pathFor(referenceId)
    if (!this.fileSystem.exists(path)) return undefined
    const raw = this.fileSystem.read(path)
    const value = parseStoredPayload(raw)
    this.remember(referenceId, value)
    return value
  }

  /** Return stored IDs for the current session in lexical order. */
  listRefs(): string[] {
    return this.fileSystem.list(this.directory)
      .filter(file => file.name.endsWith('.json'))
      .map(file => file.name.slice(0, -'.json'.length))
      .sort()
  }

  /** Remove all but the most recently modified stored results. */
  prune(keep = 100): number {
    const boundedKeep = nonNegativeInteger(keep, 'keep')
    const files = this.fileSystem.list(this.directory)
      .filter(file => file.name.endsWith('.json'))
      .sort((left, right) => right.modifiedAt - left.modifiedAt || left.name.localeCompare(right.name))
    let removed = 0
    for (const file of files.slice(boundedKeep)) {
      try {
        this.fileSystem.remove(join(this.directory, file.name))
        this.cache.delete(file.name.slice(0, -'.json'.length))
        removed += 1
      } catch (error) {
        if (!isMissingFileError(error)) throw error
      }
    }
    return removed
  }

  static isRef(value: unknown): value is string {
    return (
      typeof value === 'string'
      && value.startsWith(TOOL_RESULT_REFERENCE_PREFIX)
      && value.endsWith(TOOL_RESULT_REFERENCE_SUFFIX)
    )
  }

  /** Extract a reference ID from a placeholder, or return undefined for ordinary text. */
  static parseRef(value: string): string | undefined {
    if (!ToolResultStorage.isRef(value)) return undefined
    const body = value.slice(TOOL_RESULT_REFERENCE_PREFIX.length, -TOOL_RESULT_REFERENCE_SUFFIX.length)
    return body.split(':', 1)[0]
  }

  private pathFor(referenceId: string): string {
    if (!referenceId || referenceId.includes('/') || referenceId.includes('\\')) {
      throw new TypeError('tool-result reference ID must not contain a path separator')
    }
    return join(this.directory, referenceId + '.json')
  }

  private reference(referenceId: string, size: number, digest: string): string {
    return TOOL_RESULT_REFERENCE_PREFIX + referenceId + ':' + size + ':' + digest + TOOL_RESULT_REFERENCE_SUFFIX
  }

  private remember(referenceId: string, value: unknown): void {
    this.cache.delete(referenceId)
    this.cache.set(referenceId, value)
    while (this.cache.size > this.lruSize) {
      const oldest = this.cache.keys().next().value
      if (oldest === undefined) return
      this.cache.delete(oldest)
    }
  }
}

/** Serialize a result exactly as the on-disk overflow store does. */
export function serializeToolResultPayload(content: unknown): string {
  if (typeof content === 'string') return content
  try {
    return JSON.stringify(content) ?? String(content)
  } catch {
    return String(content)
  }
}

/** First twelve hexadecimal characters of the UTF-8 SHA-1 content digest. */
export function sha1Short(content: string): string {
  return createHash('sha1').update(content, 'utf8').digest('hex').slice(0, 12)
}

function parseStoredPayload(raw: string): unknown {
  try {
    return JSON.parse(raw) as unknown
  } catch {
    return raw
  }
}

function unicodeLength(value: string): number {
  return Array.from(value).length
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) throw new TypeError(name + ' must be a non-negative integer')
  return value
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) throw new TypeError(name + ' must be a positive integer')
  return value
}

function isMissingFileError(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

const nodeFileSystem: ToolResultStorageFileSystem = {
  exists: existsSync,
  list(directory): readonly ToolResultStoredFile[] {
    if (!existsSync(directory)) return []
    return readdirSync(directory, { withFileTypes: true })
      .filter(entry => entry.isFile())
      .map(entry => {
        const path = join(directory, entry.name)
        return { name: entry.name, modifiedAt: statSync(path).mtimeMs }
      })
  },
  makeDirectory(directory): void {
    mkdirSync(directory, { recursive: true })
  },
  read: path => readFileSync(path, 'utf8'),
  remove: unlinkSync,
  write: (path, content) => writeFileSync(path, content, 'utf8'),
}
