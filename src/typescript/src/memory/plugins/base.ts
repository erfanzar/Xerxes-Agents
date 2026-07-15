// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject, JsonValue } from '../../types/toolCalls.js'
import type { MemoryProvider, MemoryProviderToolSchema, MemoryToolCall } from '../provider.js'

/** The shared external-memory operations exposed to a model. */
export const EXTERNAL_MEMORY_ACTIONS = ['add', 'search', 'list', 'remove'] as const

/** One of the model-facing operations every external memory plugin understands. */
export type ExternalMemoryAction = typeof EXTERNAL_MEMORY_ACTIONS[number]

/** Explicit environment boundary for provider configuration. */
export interface MemoryPluginEnvironment {
  get(name: string): string | undefined
}

/** Host-owned implementation for a configured external memory backend. */
export interface ExternalMemoryUpstream {
  call(action: ExternalMemoryAction, arguments_: JsonObject): JsonValue | Promise<JsonValue>
  initialize?(): void | Promise<void>
  isAvailable?(): boolean | Promise<boolean>
  shutdown?(): void | Promise<void>
}

/** Construction inputs shared by every concrete external-memory provider. */
export interface ExternalMemoryProviderBaseOptions {
  readonly clock?: () => number
  readonly environment: MemoryPluginEnvironment
  readonly name: string
  readonly namespaceLabel?: string
  readonly requiredEnvironment?: readonly string[]
  readonly upstream: ExternalMemoryUpstream
}

/** Construction inputs for a deterministic in-memory provider used by tests and embedders. */
export interface SimpleMemoryProviderOptions {
  readonly clock?: () => number
  readonly environment: MemoryPluginEnvironment
  readonly namespaceLabel?: string
  readonly requiredEnvironment?: readonly string[]
  readonly upstream?: ExternalMemoryUpstream
}

/** Raised when a provider is constructed with an invalid static configuration. */
export class ExternalMemoryPluginConfigurationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ExternalMemoryPluginConfigurationError'
  }
}

/**
 * Common lifecycle, schema, availability, and error handling for memory plugins.
 *
 * This class deliberately has no ambient environment, network, SDK, or storage
 * dependencies. Embedders provide an environment reader and an upstream port,
 * making provider capability and all external effects visible at the boundary.
 */
export class ExternalMemoryProviderBase implements MemoryProvider {
  readonly name: string
  readonly namespaceLabel: string
  readonly requiredEnvironment: readonly string[]

  private initialized = false
  private readonly clock: () => number
  private readonly environment: MemoryPluginEnvironment
  private readonly upstream: ExternalMemoryUpstream

  constructor(options: ExternalMemoryProviderBaseOptions) {
    this.name = requireLabel(options.name, 'provider name')
    this.namespaceLabel = requireLabel(options.namespaceLabel ?? options.name, 'provider namespace label')
    const requiredEnvironment = options.requiredEnvironment ?? []
    this.requiredEnvironment = Object.freeze(requiredEnvironment.map(name => requireLabel(name, 'environment name')))
    this.clock = options.clock ?? systemClock
    this.environment = options.environment
    this.upstream = options.upstream
  }

  /** Return the four standard add/search/list/remove schemas for this provider namespace. */
  getToolSchemas(): readonly MemoryProviderToolSchema[] {
    return standardMemoryToolSchemas(this.namespaceLabel)
  }

  /** Initialize the host-owned upstream once, only after availability is confirmed. */
  async initialize(): Promise<void> {
    if (this.initialized) return
    await this.upstream.initialize?.()
    this.initialized = true
  }

  /** Return whether required configuration exists and the upstream has declared itself usable. */
  async isAvailable(): Promise<boolean> {
    if (this.requiredEnvironment.some(name => !hasEnvironmentValue(this.environment, name))) return false
    return this.upstream.isAvailable === undefined ? true : await this.upstream.isAvailable()
  }

  /** Execute one standard tool call, returning a JSON-safe success or explicit failure result. */
  async handleToolCall(call: MemoryToolCall): Promise<JsonObject> {
    let available: boolean
    try {
      available = await this.isAvailable()
    } catch (error) {
      return failure(error)
    }
    if (!available) return { ok: false, error: `${this.name} backend not available` }

    const action = actionForCall(call.name, this.namespaceLabel)
    if (action === undefined) return { ok: false, error: `unknown action: ${call.name}` }

    const argumentError = validateArguments(action, call.arguments)
    if (argumentError !== undefined) return { ok: false, error: argumentError, action }

    try {
      await this.initialize()
      const result = await this.upstream.call(action, call.arguments)
      return { ok: true, action, result, ts: validTimestamp(this.clock()) }
    } catch (error) {
      return { ...failure(error), action }
    }
  }

  /** Release the configured upstream and allow a subsequent explicit initialization. */
  async shutdown(): Promise<void> {
    await this.upstream.shutdown?.()
    this.initialized = false
  }
}

/**
 * In-process implementation of the four standard memory actions.
 *
 * This is an explicit test/development upstream, not a fallback for a
 * configured external backend. Embedders can pass it to
 * {@link makeSimpleMemoryProvider} without introducing network or credential
 * side effects.
 */
export class InMemoryExternalMemoryUpstream implements ExternalMemoryUpstream {
  private readonly entries = new Map<string, JsonObject>()
  private nextEntryNumber = 1

  call(action: ExternalMemoryAction, arguments_: JsonObject): JsonValue {
    if (action === 'add') {
      const id = `mem_${String(this.nextEntryNumber).padStart(4, '0')}`
      this.nextEntryNumber += 1
      const content = requiredMemoryPluginArgument(arguments_, 'content')
      const tags = Array.isArray(arguments_.tags)
        ? arguments_.tags.filter((tag): tag is string => typeof tag === 'string')
        : []
      const entry: JsonObject = { id, content, tags }
      this.entries.set(id, entry)
      return entry
    }
    if (action === 'list') {
      const limit = memoryPluginLimit(arguments_, 20)
      return [...this.entries.values()].slice(-limit)
    }
    if (action === 'search') {
      const query = requiredMemoryPluginArgument(arguments_, 'query').toLowerCase()
      const limit = memoryPluginLimit(arguments_, 20)
      return [...this.entries.values()]
        .filter(entry => typeof entry.content === 'string' && entry.content.toLowerCase().includes(query))
        .slice(-limit)
    }
    const entryId = requiredMemoryPluginArgument(arguments_, 'entry_id')
    return { removed: this.entries.delete(entryId) }
  }
}

/** Build a self-contained standard provider without ambient SDK, network, or credential access. */
export function makeSimpleMemoryProvider(
  name: string,
  options: SimpleMemoryProviderOptions,
): ExternalMemoryProviderBase {
  return new ExternalMemoryProviderBase({
    name,
    namespaceLabel: options.namespaceLabel ?? name,
    requiredEnvironment: options.requiredEnvironment ?? [],
    environment: options.environment,
    upstream: options.upstream ?? new InMemoryExternalMemoryUpstream(),
    ...(options.clock === undefined ? {} : { clock: options.clock }),
  })
}

/** Build immutable model tool schemas shared by every external-memory provider. */
export function standardMemoryToolSchemas(namespaceLabel: string): readonly MemoryProviderToolSchema[] {
  const label = requireLabel(namespaceLabel, 'provider namespace label')
  return Object.freeze([
    Object.freeze({
      name: `${label}_add`,
      description: `Add a memory entry to the ${label} backend.`,
      input_schema: Object.freeze({
        type: 'object',
        required: Object.freeze(['content']),
        properties: Object.freeze({
          content: Object.freeze({ type: 'string' }),
          tags: Object.freeze({ type: 'array', items: Object.freeze({ type: 'string' }) }),
        }),
      }),
    }),
    Object.freeze({
      name: `${label}_search`,
      description: `Search ${label} memory for relevant entries.`,
      input_schema: Object.freeze({
        type: 'object',
        required: Object.freeze(['query']),
        properties: Object.freeze({
          query: Object.freeze({ type: 'string' }),
          limit: Object.freeze({ type: 'integer' }),
        }),
      }),
    }),
    Object.freeze({
      name: `${label}_list`,
      description: `List recent ${label} memory entries.`,
      input_schema: Object.freeze({
        type: 'object',
        properties: Object.freeze({ limit: Object.freeze({ type: 'integer' }) }),
      }),
    }),
    Object.freeze({
      name: `${label}_remove`,
      description: `Remove a ${label} memory entry by id.`,
      input_schema: Object.freeze({
        type: 'object',
        required: Object.freeze(['entry_id']),
        properties: Object.freeze({ entry_id: Object.freeze({ type: 'string' }) }),
      }),
    }),
  ])
}

/** Read a non-empty configured value or make the missing configuration explicit to the caller. */
export function requiredMemoryPluginEnvironment(environment: MemoryPluginEnvironment, name: string): string {
  const value = environment.get(name)
  if (typeof value !== 'string' || !value.trim()) {
    throw new ExternalMemoryPluginConfigurationError(`missing required environment variable: ${name}`)
  }
  return value
}

/** Return a configured value or a caller-owned fallback without reading process state. */
export function memoryPluginEnvironmentValue(
  environment: MemoryPluginEnvironment,
  name: string,
  fallback: string,
): string {
  const value = environment.get(name)
  return typeof value === 'string' && value.trim() ? value : fallback
}

/** Convert an optional standard tool limit to a concrete request value. */
export function memoryPluginLimit(arguments_: JsonObject, fallback: number): number {
  const limit = arguments_.limit
  return typeof limit === 'number' && Number.isSafeInteger(limit) && limit >= 0 ? limit : fallback
}

/** Read a required string from a standard tool payload before building an upstream request. */
export function requiredMemoryPluginArgument(arguments_: JsonObject, name: string): string {
  const value = arguments_[name]
  if (typeof value !== 'string') throw new Error(`memory tool argument must be a string: ${name}`)
  return value
}

/** Read an optional string from a standard tool payload without coercing a caller-owned value. */
export function optionalMemoryPluginArgument(arguments_: JsonObject, name: string): string | undefined {
  const value = arguments_[name]
  if (value === undefined) return undefined
  if (typeof value !== 'string') throw new Error(`memory tool argument must be a string: ${name}`)
  return value
}

function actionForCall(name: string, namespaceLabel: string): ExternalMemoryAction | undefined {
  if (typeof name !== 'string') return undefined
  const prefix = `${namespaceLabel}_`
  const action = name.startsWith(prefix) ? name.slice(prefix.length) : name
  return isExternalMemoryAction(action) ? action : undefined
}

function failure(error: unknown): JsonObject {
  return { ok: false, error: errorMessage(error) }
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message
  return typeof error === 'string' && error ? error : 'external memory provider failed'
}

function hasEnvironmentValue(environment: MemoryPluginEnvironment, name: string): boolean {
  const value = environment.get(name)
  return typeof value === 'string' && value.trim().length > 0
}

function isExternalMemoryAction(value: string): value is ExternalMemoryAction {
  return (EXTERNAL_MEMORY_ACTIONS as readonly string[]).includes(value)
}

function requireLabel(value: string, description: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new ExternalMemoryPluginConfigurationError(`${description} must be a non-empty string`)
  }
  return value.trim()
}

function systemClock(): number {
  return Date.now() / 1000
}

function validTimestamp(value: number): number {
  if (!Number.isFinite(value)) {
    throw new ExternalMemoryPluginConfigurationError('memory plugin clock must return a finite timestamp')
  }
  return value
}

function validateArguments(action: ExternalMemoryAction, arguments_: JsonObject): string | undefined {
  if (!isRecord(arguments_)) return 'memory tool arguments must be an object'
  if (action === 'add') {
    if (typeof arguments_.content !== 'string') return 'add requires a string content argument'
    const tags = arguments_.tags
    if (tags !== undefined && (!Array.isArray(tags) || !tags.every(tag => typeof tag === 'string'))) {
      return 'add tags must be an array of strings'
    }
  }
  if (action === 'search' && typeof arguments_.query !== 'string') return 'search requires a string query argument'
  if (action === 'remove' && typeof arguments_.entry_id !== 'string') return 'remove requires a string entry_id argument'
  const limit = arguments_.limit
  if (limit !== undefined && (typeof limit !== 'number' || !Number.isSafeInteger(limit) || limit < 0)) {
    return 'limit must be a non-negative safe integer'
  }
  return undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
