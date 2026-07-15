// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ExecutionRegistry, RegistryEntry } from './executionRegistry.js'

export interface ToolPoolOptions {
  readonly categories?: readonly string[]
  readonly deniedTools?: Iterable<string>
  readonly safeOnly?: boolean
  readonly tools?: readonly RegistryEntry[]
}

export interface AssembleToolPoolOptions {
  /** Restrict the pool to matching tool categories. An empty list keeps all categories. */
  readonly categories?: readonly string[]
  /** Tool names to exclude exactly. */
  readonly denyTools?: Iterable<string>
  /** Exclude tool names beginning with any supplied prefix. */
  readonly denyPrefixes?: readonly string[]
  /** Keep only tools which the registry marks safe. */
  readonly safeOnly?: boolean
  /** Omit tools whose source hint contains mcp, case-insensitively. */
  readonly includeMcp?: boolean
}

/**
 * Immutable schema-ready view of tools exposed for one turn or sub-agent.
 *
 * Entries are snapshotted at construction time, so later registry mutations
 * and caller-owned filter collection mutations do not alter this pool.
 */
export class ToolPool {
  readonly categories: readonly string[]
  readonly deniedTools: ReadonlySet<string>
  readonly safeOnly: boolean
  readonly tools: readonly RegistryEntry[]

  constructor(options: ToolPoolOptions = {}) {
    this.tools = Object.freeze((options.tools ?? []).map(snapshotEntry))
    this.deniedTools = immutableSet(options.deniedTools ?? [])
    this.categories = Object.freeze([...(options.categories ?? [])])
    this.safeOnly = options.safeOnly ?? false
    Object.freeze(this)
  }

  get toolCount(): number {
    return this.tools.length
  }

  get toolNames(): readonly string[] {
    return Object.freeze(this.tools.map(tool => tool.name))
  }

  /** Look up a tool by its exact advertised name. */
  getTool(name: string): RegistryEntry | undefined {
    return this.tools.find(tool => tool.name === name)
  }

  /** Return immutable tool schemas suitable for an LLM request. */
  toSchemas(): ReadonlyArray<Readonly<Record<string, unknown>>> {
    return Object.freeze(this.tools.map(toolSchema))
  }

  /** Render the pool's filters and surviving tools as a Markdown overview. */
  asMarkdown(): string {
    const lines = [
      '# Tool Pool',
      '',
      'Tools: ' + this.toolCount,
      'Safe only: ' + this.safeOnly,
      'Categories: ' + (this.categories.join(', ') || 'all'),
      'Denied: ' + ([...this.deniedTools].sort().join(', ') || 'none'),
      '',
    ]
    for (const tool of this.tools) {
      const safeTag = tool.safe ? ' [safe]' : ''
      const categoryTag = tool.category ? ' (' + tool.category + ')' : ''
      lines.push('- **' + tool.name + '**' + safeTag + categoryTag + ' — ' + tool.description)
    }
    return lines.join('\n')
  }
}

/**
 * Build an immutable filtered tool pool from a registry.
 *
 * Filtering follows the Python runtime's ordering: safety, categories, exact
 * deny-list names, denied prefixes, and then MCP source hints.
 */
export function assembleToolPool(
  registry: ExecutionRegistry | null | undefined = undefined,
  options: AssembleToolPoolOptions = {},
): ToolPool {
  if (registry === undefined || registry === null) return new ToolPool()

  const denied = new Set(options.denyTools ?? [])
  const prefixes = options.denyPrefixes ?? []
  const categories = options.categories ?? []
  const safeOnly = options.safeOnly ?? false
  const includeMcp = options.includeMcp ?? true

  let tools = registry.listTools({ safeOnly })
  if (categories.length > 0) tools = tools.filter(tool => categories.includes(tool.category))
  if (denied.size > 0) tools = tools.filter(tool => !denied.has(tool.name))
  if (prefixes.length > 0) tools = tools.filter(tool => !prefixes.some(prefix => tool.name.startsWith(prefix)))
  if (!includeMcp) tools = tools.filter(tool => !tool.sourceHint.toLowerCase().includes('mcp'))

  return new ToolPool({
    tools,
    deniedTools: denied,
    categories,
    safeOnly,
  })
}

function toolSchema(entry: RegistryEntry): Readonly<Record<string, unknown>> {
  if (hasSchema(entry.schema)) return entry.schema
  return Object.freeze({
    name: entry.name,
    description: entry.description || 'Execute ' + entry.name,
    input_schema: Object.freeze({
      type: 'object',
      properties: Object.freeze({}),
    }),
  })
}

function hasSchema(schema: Readonly<Record<string, unknown>> | undefined): schema is Readonly<Record<string, unknown>> {
  return schema !== undefined && Object.keys(schema).length > 0
}

function snapshotEntry(entry: RegistryEntry): RegistryEntry {
  return Object.freeze({
    name: entry.name,
    kind: entry.kind,
    description: entry.description,
    handler: entry.handler,
    category: entry.category,
    safe: entry.safe,
    sourceHint: entry.sourceHint,
    schema: entry.schema === undefined ? undefined : freezeRecord(entry.schema),
  })
}

function freezeRecord(record: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> {
  const snapshot: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(record)) snapshot[key] = freezeValue(value)
  return Object.freeze(snapshot)
}

function freezeValue(value: unknown): unknown {
  if (Array.isArray(value)) return Object.freeze(value.map(freezeValue))
  if (value instanceof Date) return value.toISOString()
  if (value !== null && typeof value === 'object') return freezeRecord(value as Readonly<Record<string, unknown>>)
  return value
}

function immutableSet(values: Iterable<string>): ReadonlySet<string> {
  const valuesSnapshot = new Set(values)
  const snapshot: ReadonlySet<string> = {
    get size(): number {
      return valuesSnapshot.size
    },
    has: value => valuesSnapshot.has(value),
    entries: () => valuesSnapshot.entries(),
    keys: () => valuesSnapshot.keys(),
    values: () => valuesSnapshot.values(),
    forEach: (callback, thisArg) => {
      valuesSnapshot.forEach(value => callback.call(thisArg, value, value, snapshot))
    },
    [Symbol.iterator]: () => valuesSnapshot[Symbol.iterator](),
    union: <Value>(other: ReadonlySetLike<Value>): Set<string | Value> => {
      const result = new Set<string | Value>(valuesSnapshot)
      forEachSetValue(other, value => result.add(value))
      return result
    },
    intersection: <Value>(other: ReadonlySetLike<Value>): Set<string & Value> => {
      const result = new Set<string & Value>()
      for (const value of valuesSnapshot) {
        if (other.has(value as unknown as Value)) result.add(value as unknown as string & Value)
      }
      return result
    },
    difference: <Value>(other: ReadonlySetLike<Value>): Set<string> => {
      const result = new Set<string>()
      for (const value of valuesSnapshot) {
        if (!other.has(value as unknown as Value)) result.add(value)
      }
      return result
    },
    symmetricDifference: <Value>(other: ReadonlySetLike<Value>): Set<string | Value> => {
      const result = new Set<string | Value>(valuesSnapshot)
      forEachSetValue(other, value => {
        if (valuesSnapshot.has(value as unknown as string)) result.delete(value)
        else result.add(value)
      })
      return result
    },
    isSubsetOf: (other: ReadonlySetLike<unknown>): boolean => {
      for (const value of valuesSnapshot) {
        if (!other.has(value)) return false
      }
      return true
    },
    isSupersetOf: (other: ReadonlySetLike<unknown>): boolean => {
      let result = true
      forEachSetValue(other, value => {
        if (!valuesSnapshot.has(value as string)) result = false
      })
      return result
    },
    isDisjointFrom: (other: ReadonlySetLike<unknown>): boolean => {
      let result = true
      forEachSetValue(other, value => {
        if (valuesSnapshot.has(value as string)) result = false
      })
      return result
    },
  }
  return Object.freeze(snapshot)
}

function forEachSetValue<Value>(set: ReadonlySetLike<Value>, callback: (value: Value) => void): void {
  const iterator = set.keys()
  for (;;) {
    const next = iterator.next()
    if (next.done) return
    callback(next.value)
  }
}
