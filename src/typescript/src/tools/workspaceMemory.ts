// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, rename, rm, stat } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { optionalInteger, requiredString } from './inputs.js'
import { WorkspacePathResolver } from './pathSafety.js'

export type WorkspaceMemoryKind = 'memory' | 'user'

export interface WorkspaceMemoryStoreOptions {
  readonly paths?: WorkspacePathResolver
  readonly workspaceRoot?: string
}

export interface WorkspaceMemoryEntry {
  readonly content: string
  readonly id: number
}

/**
 * Line-oriented workspace notes compatible with the Python MEMORY.md/USER.md
 * CRUD helpers. Each mutation uses an in-process path lock and an atomic
 * rename, so concurrent tool calls cannot silently lose entries.
 */
export class WorkspaceMemoryStore {
  readonly paths: WorkspacePathResolver
  private readonly locks = new Map<WorkspaceMemoryKind, Promise<void>>()

  constructor(options: WorkspaceMemoryStoreOptions = {}) {
    this.paths = options.paths ?? new WorkspacePathResolver(options.workspaceRoot)
  }

  async add(kind: WorkspaceMemoryKind, content: string): Promise<{ readonly content: string; readonly id: number; readonly ok: true } | Failure> {
    const cleaned = content.trim()
    if (!cleaned) return failure('content required')
    return this.withLock(kind, async () => {
      const lines = await this.readLines(kind)
      lines.push(cleaned)
      await this.writeLines(kind, lines)
      return { ok: true, id: lines.length, content: cleaned }
    })
  }

  async list(kind: WorkspaceMemoryKind, limit?: number): Promise<{ readonly items: readonly WorkspaceMemoryEntry[]; readonly ok: true }> {
    const lines = await this.readLines(kind)
    const entries = lines.map((content, index) => ({ id: index + 1, content }))
    const selected = limit === undefined ? entries : entries.slice(-validateLimit(limit))
    return { ok: true, items: selected }
  }

  async remove(kind: WorkspaceMemoryKind, entryId: number): Promise<{ readonly content: string; readonly id: number; readonly ok: true } | Failure> {
    return this.withLock(kind, async () => {
      const lines = await this.readLines(kind)
      if (!hasEntry(lines, entryId)) return failure('id ' + entryId + ' not found')
      const [content] = lines.splice(entryId - 1, 1)
      await this.writeLines(kind, lines)
      return { ok: true, id: entryId, content: content ?? '' }
    })
  }

  async replace(
    kind: WorkspaceMemoryKind,
    entryId: number,
    content: string,
  ): Promise<{ readonly content: string; readonly id: number; readonly ok: true } | Failure> {
    const cleaned = content.trim()
    if (!cleaned) return failure('content required')
    return this.withLock(kind, async () => {
      const lines = await this.readLines(kind)
      if (!hasEntry(lines, entryId)) return failure('id ' + entryId + ' not found')
      lines[entryId - 1] = cleaned
      await this.writeLines(kind, lines)
      return { ok: true, id: entryId, content: cleaned }
    })
  }

  private async readLines(kind: WorkspaceMemoryKind): Promise<string[]> {
    const target = await this.target(kind)
    try {
      await stat(target)
    } catch (error) {
      if (isMissing(error)) return []
      throw error
    }
    return (await Bun.file(target).text()).split(/\r?\n/).filter(line => Boolean(line.trim()))
  }

  private async writeLines(kind: WorkspaceMemoryKind, lines: readonly string[]): Promise<void> {
    const target = await this.target(kind)
    await mkdir(dirname(target), { recursive: true })
    const temporary = join(dirname(target), '.' + kind + '.' + crypto.randomUUID() + '.tmp')
    try {
      await Bun.write(temporary, lines.length ? lines.join('\n') + '\n' : '')
      await rename(temporary, target)
    } finally {
      await rm(temporary, { force: true })
    }
  }

  private async target(kind: WorkspaceMemoryKind): Promise<string> {
    return this.paths.resolve(kind === 'memory' ? 'MEMORY.md' : 'USER.md')
  }

  private async withLock<T>(kind: WorkspaceMemoryKind, operation: () => Promise<T>): Promise<T> {
    const previous = this.locks.get(kind) ?? Promise.resolve()
    let release: (() => void) | undefined
    const current = new Promise<void>(resolveLock => {
      release = resolveLock
    })
    this.locks.set(kind, current)
    await previous
    try {
      return await operation()
    } finally {
      release?.()
      if (this.locks.get(kind) === current) this.locks.delete(kind)
    }
  }
}

export interface WorkspaceMemoryToolsOptions extends WorkspaceMemoryStoreOptions {
  readonly store?: WorkspaceMemoryStore
}

export const MEMORY_ADD_DEFINITION = memoryDefinition('memory_add', 'Add one line to workspace MEMORY.md.', {
  content: { type: 'string' },
}, ['content'])
export const MEMORY_LIST_DEFINITION = memoryDefinition('memory_list', 'List line entries in workspace MEMORY.md.', {
  limit: { type: 'integer', minimum: 1 },
})
export const MEMORY_REPLACE_DEFINITION = memoryDefinition('memory_replace', 'Replace one 1-based line in workspace MEMORY.md.', {
  entry_id: { type: 'integer', minimum: 1 },
  content: { type: 'string' },
}, ['entry_id', 'content'])
export const MEMORY_REMOVE_DEFINITION = memoryDefinition('memory_remove', 'Remove one 1-based line in workspace MEMORY.md.', {
  entry_id: { type: 'integer', minimum: 1 },
}, ['entry_id'])
export const USER_ADD_DEFINITION = memoryDefinition('user_add', 'Add one line to workspace USER.md.', {
  content: { type: 'string' },
}, ['content'])
export const USER_LIST_DEFINITION = memoryDefinition('user_list', 'List line entries in workspace USER.md.', {
  limit: { type: 'integer', minimum: 1 },
})
export const USER_REPLACE_DEFINITION = memoryDefinition('user_replace', 'Replace one 1-based line in workspace USER.md.', {
  entry_id: { type: 'integer', minimum: 1 },
  content: { type: 'string' },
}, ['entry_id', 'content'])
export const USER_REMOVE_DEFINITION = memoryDefinition('user_remove', 'Remove one 1-based line in workspace USER.md.', {
  entry_id: { type: 'integer', minimum: 1 },
}, ['entry_id'])

export const WORKSPACE_MEMORY_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  MEMORY_ADD_DEFINITION,
  MEMORY_LIST_DEFINITION,
  MEMORY_REPLACE_DEFINITION,
  MEMORY_REMOVE_DEFINITION,
  USER_ADD_DEFINITION,
  USER_LIST_DEFINITION,
  USER_REPLACE_DEFINITION,
  USER_REMOVE_DEFINITION,
]

/** Register the Python-compatible workspace-memory CRUD tool names. */
export function registerWorkspaceMemoryTools(registry: ToolRegistry, options: WorkspaceMemoryToolsOptions = {}): WorkspaceMemoryStore {
  const store = options.store ?? new WorkspaceMemoryStore(options)
  registerKind(registry, store, 'memory', MEMORY_ADD_DEFINITION, MEMORY_LIST_DEFINITION, MEMORY_REPLACE_DEFINITION, MEMORY_REMOVE_DEFINITION)
  registerKind(registry, store, 'user', USER_ADD_DEFINITION, USER_LIST_DEFINITION, USER_REPLACE_DEFINITION, USER_REMOVE_DEFINITION)
  return store
}

function registerKind(
  registry: ToolRegistry,
  store: WorkspaceMemoryStore,
  kind: WorkspaceMemoryKind,
  add: ToolDefinition,
  list: ToolDefinition,
  replace: ToolDefinition,
  remove: ToolDefinition,
): void {
  registry.register(add, inputs => store.add(kind, requiredString(inputs, 'content')))
  registry.register(list, inputs => store.list(kind, optionalLimit(inputs)))
  registry.register(replace, inputs => store.replace(kind, positiveEntryId(inputs), requiredString(inputs, 'content')))
  registry.register(remove, inputs => store.remove(kind, positiveEntryId(inputs)))
}

function memoryDefinition(
  name: string,
  description: string,
  properties: Record<string, unknown>,
  required: readonly string[] = [],
): ToolDefinition {
  return {
    type: 'function',
    function: {
      name,
      description,
      parameters: {
        type: 'object',
        additionalProperties: false,
        properties,
        ...(required.length ? { required } : {}),
      },
    },
  }
}

function optionalLimit(inputs: JsonObject): number | undefined {
  if (inputs.limit === undefined) return undefined
  return validateLimit(optionalInteger(inputs, 'limit', 1))
}

function positiveEntryId(inputs: JsonObject): number {
  return validateLimit(optionalInteger(inputs, 'entry_id', 0))
}

function validateLimit(value: number): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error('entry_id and limit must be positive integers')
  }
  return value
}

function hasEntry(lines: readonly string[], entryId: number): boolean {
  return Number.isInteger(entryId) && entryId >= 1 && entryId <= lines.length
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

interface Failure {
  readonly error: string
  readonly ok: false
}

function failure(error: string): Failure {
  return { ok: false, error }
}
