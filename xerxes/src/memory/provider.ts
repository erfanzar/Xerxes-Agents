// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject, JsonSchema } from '../types/toolCalls.js'

/** A JSON-decoded, model-invoked call directed at an external memory provider. */
export interface MemoryToolCall {
  readonly arguments: JsonObject
  readonly name: string
}

/** A provider-owned model tool schema. This module does not define any concrete tools. */
export type MemoryProviderToolSchema = JsonSchema

/** Optional hooks that a provider may implement to observe runtime lifecycle events. */
export interface MemoryProviderLifecycleHooks {
  onDelegation?(parentSessionId: string, childSessionId: string): void | Promise<void>
  onMemoryWrite?(reference: string, content: string): void | Promise<void>
  onPreCompress?(state: unknown): void | Promise<void>
  onSessionEnd?(state: unknown): void | Promise<void>
  onTurnStart?(state: unknown): void | Promise<void>
  shutdown?(): void | Promise<void>
}

/**
 * Host boundary for one external memory backend.
 *
 * Implementations are supplied by the embedding application. This module does
 * not make network requests or ship a concrete external-memory backend.
 */
export interface MemoryProvider extends MemoryProviderLifecycleHooks {
  readonly name: string
  getToolSchemas(): readonly MemoryProviderToolSchema[] | Promise<readonly MemoryProviderToolSchema[]>
  handleToolCall(call: MemoryToolCall): JsonObject | Promise<JsonObject>
  initialize(): void | Promise<void>
  isAvailable(): boolean | Promise<boolean>
}

/** Raised when a provider cannot be registered or selected safely. */
export class MemoryProviderRegistryError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'MemoryProviderRegistryError'
  }
}

/** Raised when activation names a provider that has not been registered. */
export class UnknownMemoryProviderError extends MemoryProviderRegistryError {
  constructor(name: string) {
    super(`unknown memory provider: ${name}`)
    this.name = 'UnknownMemoryProviderError'
  }
}

/**
 * In-process external-memory provider registry with one active provider slot.
 *
 * Built-in memory tiers are intentionally independent of this registry. Bun's
 * JavaScript execution model makes the synchronous map mutations atomic within
 * a runtime isolate; provider lifecycle work is never invoked by this class.
 */
export class MemoryProviderRegistry {
  private activeName: string | undefined
  private readonly providers = new Map<string, MemoryProvider>()

  /** Return the selected external provider, if any. */
  active(): MemoryProvider | undefined {
    return this.activeName === undefined ? undefined : this.providers.get(this.activeName)
  }

  /** Return a provider by name without changing the active selection. */
  get(name: string): MemoryProvider | undefined {
    return this.providers.get(normalizeProviderName(name))
  }

  /** Return registered provider names in deterministic lexical order. */
  listNames(): string[] {
    return [...this.providers.keys()].sort(compareStrings)
  }

  /** Register or replace a provider under its normalized name. */
  register(provider: MemoryProvider): void {
    const name = providerNameFrom(provider)
    this.providers.set(name, provider)
  }

  /** Select one registered provider, or clear the active selection. */
  setActive(name: string | null | undefined): MemoryProvider | undefined {
    if (name === null || name === undefined) {
      this.activeName = undefined
      return undefined
    }
    const normalized = normalizeProviderName(name)
    const provider = this.providers.get(normalized)
    if (!provider) {
      throw new UnknownMemoryProviderError(normalized)
    }
    this.activeName = normalized
    return provider
  }

  /** Remove a provider and clear selection when it was active. */
  unregister(name: string): boolean {
    const normalized = normalizeProviderName(name)
    if (!normalized) {
      return false
    }
    const removed = this.providers.delete(normalized)
    if (this.activeName === normalized) {
      this.activeName = undefined
    }
    return removed
  }
}

const DEFAULT_MEMORY_PROVIDER_REGISTRY = new MemoryProviderRegistry()

/** Build a tool-call value with an empty argument object by default. */
export function createMemoryToolCall(name: string, arguments_: JsonObject = {}): MemoryToolCall {
  const normalizedName = normalizeToolName(name)
  return Object.freeze({
    arguments: Object.freeze({ ...arguments_ }),
    name: normalizedName,
  })
}

/** Return the process-wide external-memory provider registry. */
export function memoryProviderRegistry(): MemoryProviderRegistry {
  return DEFAULT_MEMORY_PROVIDER_REGISTRY
}

/** Register a provider in the process-wide external-memory registry. */
export function registerMemoryProvider(provider: MemoryProvider): void {
  DEFAULT_MEMORY_PROVIDER_REGISTRY.register(provider)
}

/** Return the selected process-wide external-memory provider, if any. */
export function activeMemoryProvider(): MemoryProvider | undefined {
  return DEFAULT_MEMORY_PROVIDER_REGISTRY.active()
}

/** Select a process-wide provider by name, or clear the active selection. */
export function setActiveMemoryProvider(name: string | null | undefined): MemoryProvider | undefined {
  return DEFAULT_MEMORY_PROVIDER_REGISTRY.setActive(name)
}

function providerNameFrom(provider: MemoryProvider): string {
  if (typeof provider !== 'object' || provider === null) {
    throw new MemoryProviderRegistryError('memory provider must be an object')
  }
  const name = normalizeProviderName(provider.name)
  if (!name) {
    throw new MemoryProviderRegistryError('memory provider name must not be empty')
  }
  return name
}

function normalizeProviderName(name: string): string {
  if (typeof name !== 'string') {
    throw new MemoryProviderRegistryError('memory provider name must be a string')
  }
  return name.trim()
}

function normalizeToolName(name: string): string {
  if (typeof name !== 'string' || !name.trim()) {
    throw new MemoryProviderRegistryError('memory tool call name must not be empty')
  }
  return name.trim()
}

function compareStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}
