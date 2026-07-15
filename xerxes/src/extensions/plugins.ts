// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'

import { DependencyResolver, parseDependency, VersionConstraint } from './dependency.js'
import type { HookCallback, HookPoint } from './hooks.js'
import type { FetchImplementation, LlmClient } from '../llms/client.js'
import type { ProviderOverrides } from '../llms/providerRegistry.js'

export const PluginType = {
  TOOL: 'tool',
  HOOK: 'hook',
  PROVIDER: 'provider',
  CHANNEL: 'channel',
  SEARCH: 'search',
  SPEECH: 'speech',
} as const

export type PluginType = (typeof PluginType)[keyof typeof PluginType]
export type PluginTool = (...args: unknown[]) => unknown | Promise<unknown>

/** Settings that the host passes through when it asks a plugin for an LLM client. */
export interface PluginLlmProviderOptions {
  readonly apiKey?: string
  readonly baseUrl?: string
  readonly fetchImplementation?: FetchImplementation
  readonly promptCaching?: boolean
  readonly responsesApi?: boolean
}

/** Input supplied to a plugin-owned LLM client factory. */
export interface PluginLlmProviderRequest {
  /** Model name with an optional `provider/` prefix removed. */
  readonly model: string
  readonly options: PluginLlmProviderOptions
  readonly overrides: ProviderOverrides
  readonly providerName: string
  /** Original model value passed to `createLlmClient`, retained for provider-specific routing. */
  readonly requestedModel: string
}

/** Factory a provider plugin must expose to participate in native LLM selection. */
export interface PluginLlmProviderFactory {
  createClient(request: PluginLlmProviderRequest): LlmClient
}

/** Minimal registry port consumed by the native LLM client factory. */
export interface PluginLlmProviderRegistry {
  getProvider(name: string): PluginLlmProviderFactory | undefined
}

/** Validate the runtime boundary used by dynamically loaded provider plugins. */
export function isPluginLlmProviderFactory(value: unknown): value is PluginLlmProviderFactory {
  return typeof value === 'object'
    && value !== null
    && typeof (value as { createClient?: unknown }).createClient === 'function'
}

export interface PluginMeta {
  readonly author?: string
  readonly dependencies?: readonly string[]
  readonly description?: string
  readonly name: string
  readonly pluginType?: PluginType
  readonly version?: string
  readonly versionConstraints?: Readonly<Record<string, string>>
}

export interface RegisteredPlugin {
  readonly channels: Map<string, unknown>
  readonly hooks: Map<string, HookCallback>
  readonly meta: RequiredPluginMeta
  provider: PluginLlmProviderFactory | undefined
  readonly tools: Map<string, PluginTool>
}

interface RequiredPluginMeta {
  readonly author: string
  readonly dependencies: readonly string[]
  readonly description: string
  readonly name: string
  readonly pluginType: PluginType
  readonly version: string
  readonly versionConstraints: Readonly<Record<string, string>>
}

export class PluginConflictError extends Error {
  constructor(readonly resource: string, readonly existing: string) {
    super(`Plugin '${resource}' conflicts with existing plugin '${existing}'`)
    this.name = 'PluginConflictError'
  }
}

export interface XerxesPluginModule {
  readonly register?: (registry: PluginRegistry) => void | Promise<void>
}

/** Owns plugin capability indexes and validates dependency-compatible load order. */
export class PluginRegistry {
  private readonly channels = new Map<string, { readonly owner: string; readonly value: unknown }>()
  private readonly hooks = new Map<string, Array<{ readonly callback: HookCallback; readonly owner: string }>>()
  private readonly plugins = new Map<string, RegisteredPlugin>()
  private readonly providers = new Map<string, { readonly owner: string; readonly value: PluginLlmProviderFactory }>()
  private readonly tools = new Map<string, { readonly owner: string; readonly value: PluginTool }>()

  get pluginNames(): string[] {
    return [...this.plugins.keys()]
  }

  async discover(directory: string): Promise<string[]> {
    if (!existsSync(directory)) return []
    const discovered: string[] = []
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      if (!entry.isFile() || entry.name.startsWith('_') || !/\.(?:[cm]?js|ts)$/.test(entry.name)) continue
      const before = new Set(this.plugins.keys())
      try {
        const path = join(directory, entry.name)
        const module = await import(`${pathToFileURL(path).href}?xerxes=${Date.now()}`) as XerxesPluginModule
        await module.register?.(this)
        for (const name of this.plugins.keys()) if (!before.has(name)) discovered.push(name)
      } catch {
        // One third-party plugin must not prevent discovery of the next.
      }
    }
    return discovered
  }

  getAllChannels(): Record<string, unknown> {
    return Object.fromEntries([...this.channels].map(([name, entry]) => [name, entry.value]))
  }

  getAllTools(): Record<string, PluginTool> {
    return Object.fromEntries([...this.tools].map(([name, entry]) => [name, entry.value]))
  }

  getChannel(name: string): unknown | undefined {
    return this.channels.get(name)?.value
  }

  getHooks(name: string): HookCallback[] {
    return (this.hooks.get(name) ?? []).map(entry => entry.callback)
  }

  getLoadOrder(): string[] {
    const graph: Record<string, string[]> = {}
    for (const [name, plugin] of this.plugins) {
      const dependencies = plugin.meta.dependencies.map(parseDependency).map(spec => spec.name)
      for (const dependency of Object.keys(plugin.meta.versionConstraints)) if (!dependencies.includes(dependency)) dependencies.push(dependency)
      graph[name] = dependencies
    }
    return new DependencyResolver().topologicalSort(graph)
  }

  getPlugin(name: string): RegisteredPlugin | undefined {
    return this.plugins.get(name)
  }

  getProvider(name: string): PluginLlmProviderFactory | undefined {
    return this.providers.get(name)?.value
  }

  getTool(name: string): PluginTool | undefined {
    return this.tools.get(name)?.value
  }

  registerChannel(name: string, channel: unknown, meta?: PluginMeta, pluginName?: string): void {
    const owner = this.resolveOwner(meta, pluginName)
    this.registerUnique(this.channels, `channel:${name}`, name, channel, owner)
    this.plugins.get(owner)?.channels.set(name, channel)
  }

  registerHook(name: HookPoint | string, callback: HookCallback, meta?: PluginMeta, pluginName?: string): void {
    const owner = this.resolveOwner(meta, pluginName)
    const values = this.hooks.get(name) ?? []
    values.push({ callback, owner })
    this.hooks.set(name, values)
    this.plugins.get(owner)?.hooks.set(name, callback)
  }

  registerPlugin(meta: PluginMeta): RegisteredPlugin {
    if (this.plugins.has(meta.name)) throw new PluginConflictError(meta.name, meta.name)
    const plugin: RegisteredPlugin = {
      meta: normalizeMeta(meta), tools: new Map(), hooks: new Map(), channels: new Map(), provider: undefined,
    }
    this.plugins.set(meta.name, plugin)
    return plugin
  }

  registerProvider(name: string, provider: PluginLlmProviderFactory, meta?: PluginMeta, pluginName?: string): void {
    if (!isPluginLlmProviderFactory(provider)) {
      throw new TypeError(`Provider '${name}' must expose createClient(request)`)
    }
    const owner = this.resolveOwner(meta, pluginName)
    this.registerUnique(this.providers, `provider:${name}`, name, provider, owner)
    const plugin = this.plugins.get(owner)
    if (plugin) plugin.provider = provider
  }

  registerTool(name: string, tool: PluginTool, meta?: PluginMeta, pluginName?: string): void {
    const owner = this.resolveOwner(meta, pluginName)
    this.registerUnique(this.tools, `tool:${name}`, name, tool, owner)
    this.plugins.get(owner)?.tools.set(name, tool)
  }

  unregisterPlugin(name: string): void {
    if (!this.plugins.delete(name)) return
    for (const [tool, entry] of this.tools) if (entry.owner === name) this.tools.delete(tool)
    for (const [provider, entry] of this.providers) if (entry.owner === name) this.providers.delete(provider)
    for (const [channel, entry] of this.channels) if (entry.owner === name) this.channels.delete(channel)
    for (const [hook, entries] of this.hooks) {
      const retained = entries.filter(entry => entry.owner !== name)
      if (retained.length) this.hooks.set(hook, retained)
      else this.hooks.delete(hook)
    }
  }

  validateDependencies(): string[] {
    const available = Object.fromEntries([...this.plugins].map(([name, plugin]) => [name, plugin.meta.version]))
    const resolver = new DependencyResolver()
    const errors: string[] = []
    for (const [name, plugin] of this.plugins) {
      const requirements = [
        ...plugin.meta.dependencies.map(parseDependency),
        ...Object.entries(plugin.meta.versionConstraints).map(([dependency, constraint]) => parseDependency(`${dependency}${constraint}`)),
      ]
      const result = resolver.resolve(available, requirements)
      errors.push(...result.missing.map(dependency => `Plugin '${name}' requires missing dependency '${dependency}'`))
      errors.push(...result.conflicts.map(conflict => `Plugin '${name}' has version conflict: ${conflict}`))
    }
    return errors
  }

  versionConflicts(name: string, version: string): string[] {
    const conflicts: string[] = []
    for (const [pluginName, plugin] of this.plugins) {
      const constraints = [
        ...plugin.meta.dependencies.map(parseDependency).filter(spec => spec.name === name),
        ...Object.entries(plugin.meta.versionConstraints).filter(([dependency]) => dependency === name)
          .map(([dependency, constraint]) => ({ name: dependency, versionConstraint: constraint })),
      ]
      for (const constraint of constraints) {
        if (constraint.versionConstraint && !new VersionConstraint(constraint.versionConstraint).satisfies(version)) {
          conflicts.push(`Plugin '${pluginName}' requires ${name}${constraint.versionConstraint}, but version ${version} would be registered`)
        }
      }
    }
    return conflicts
  }

  private registerUnique<T>(
    index: Map<string, { readonly owner: string; readonly value: T }>,
    resource: string,
    name: string,
    value: T,
    owner: string,
  ): void {
    const existing = index.get(name)
    if (existing) throw new PluginConflictError(resource, existing.owner)
    index.set(name, { value, owner })
  }

  private resolveOwner(meta: PluginMeta | undefined, explicitOwner: string | undefined): string {
    const owner = explicitOwner ?? meta?.name ?? '__standalone__'
    if (meta && !this.plugins.has(owner)) this.registerPlugin(meta)
    return owner
  }
}

function normalizeMeta(meta: PluginMeta): RequiredPluginMeta {
  return {
    name: meta.name,
    version: meta.version ?? '0.1.0',
    pluginType: meta.pluginType ?? PluginType.TOOL,
    description: meta.description ?? '',
    author: meta.author ?? '',
    dependencies: [...(meta.dependencies ?? [])],
    versionConstraints: { ...(meta.versionConstraints ?? {}) },
  }
}
