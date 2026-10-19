// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, readdirSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

import { isWindows } from '../core/hostPlatform.js'
import { DependencyResolver, parseDependency, VersionConstraint } from './dependency.js'
import type { HookCallback, HookPoint } from './hooks.js'
import type { SlashHandler, SlashPluginOptions } from './slashPlugins.js'
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

/** What a structured plugin tool receives with each call, beside its arguments. */
export interface PluginToolContext {
  /** Aborts when the turn is cancelled; long-running tools should honour it. */
  readonly signal?: AbortSignal
  readonly sessionId?: string
  readonly agentId?: string
  /**
   * Values of the secrets the plugin declared in `permissions.secrets` and the
   * user has set (managed plugins only). Undeclared or unset names are absent.
   */
  readonly secrets?: Readonly<Record<string, string>>
}

/**
 * What a plugin says it needs, declared through `registerPlugin({ permissions })`.
 *
 * For managed plugins, which run in their own Bun process, `env` is enforced
 * (the process starts with an empty environment and receives only these
 * variables) and `network` is a guard rail on `fetch` only. `filesystem` is
 * declared and displayed, never enforced: plugin code can still use `node:fs`,
 * `node:net`, `Bun.spawn` and so on with the user's privileges. This is not a
 * security sandbox.
 */
export interface PluginPermissions {
  /** Hostnames `fetch` may reach; `*.example.com` admits subdomains. */
  readonly network?: readonly string[]
  /** Environment variable names copied into the plugin process. */
  readonly env?: readonly string[]
  /** Absolute path prefixes the plugin says it reads or writes (displayed, not enforced). */
  readonly filesystem?: readonly string[]
  /** Secret names (UPPER_SNAKE) the plugin receives as `context.secrets`. */
  readonly secrets?: readonly string[]
}

export interface NormalizedPluginPermissions {
  readonly network: readonly string[]
  readonly env: readonly string[]
  readonly filesystem: readonly string[]
  readonly secrets: readonly string[]
}

export const PLUGIN_SECRET_NAME = /^[A-Z][A-Z0-9_]{0,63}$/
const PLUGIN_ENV_NAME = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/
const PLUGIN_HOST_NAME = /^(?:\*\.)?[a-z0-9](?:[a-z0-9-]{0,62})(?:\.[a-z0-9](?:[a-z0-9-]{0,62}))*$/
const MAX_PERMISSION_ENTRIES = 64
const MAX_TOOL_TIMEOUT_MS = 3_600_000

/** Validate a declared permissions object at the registration boundary. */
export function normalizePluginPermissions(value: unknown, plugin: string): NormalizedPluginPermissions {
  if (value === undefined || value === null) return Object.freeze({ network: [], env: [], filesystem: [], secrets: [] })
  if (typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`Plugin '${plugin}' permissions must be an object`)
  const record = value as Record<string, unknown>
  for (const key of Object.keys(record)) {
    if (!['network', 'env', 'filesystem', 'secrets'].includes(key)) throw new TypeError(`Plugin '${plugin}' permissions has unknown key '${key}' (use network, env, filesystem or secrets)`)
  }
  const list = (key: string, check: (entry: string) => string | undefined): readonly string[] => {
    const raw = record[key]
    if (raw === undefined) return Object.freeze([])
    if (!Array.isArray(raw) || raw.length > MAX_PERMISSION_ENTRIES) throw new TypeError(`Plugin '${plugin}' permissions.${key} must be an array of at most ${MAX_PERMISSION_ENTRIES} strings`)
    const entries: string[] = []
    for (const entry of raw) {
      if (typeof entry !== 'string') throw new TypeError(`Plugin '${plugin}' permissions.${key} must contain only strings`)
      const problem = check(entry)
      if (problem) throw new TypeError(`Plugin '${plugin}' permissions.${key}: ${problem}`)
      const normalized = key === 'network' ? entry.toLowerCase() : entry
      if (!entries.includes(normalized)) entries.push(normalized)
    }
    return Object.freeze(entries)
  }
  return Object.freeze({
    network: list('network', entry => PLUGIN_HOST_NAME.test(entry.toLowerCase()) ? undefined : `'${entry}' is not a hostname (no scheme, port or path; '*.example.com' admits subdomains)`),
    env: list('env', entry => PLUGIN_ENV_NAME.test(entry) ? undefined : `'${entry}' is not an environment variable name`),
    filesystem: list('filesystem', entry => entry.startsWith('/') || /^[A-Za-z]:[\\/]/.test(entry) ? undefined : `'${entry}' must be an absolute path prefix`),
    secrets: list('secrets', entry => PLUGIN_SECRET_NAME.test(entry) ? undefined : `'${entry}' must be UPPER_SNAKE_CASE (A-Z, 0-9, _; at most 64 characters)`),
  })
}

/**
 * A plugin tool described the way the model sees it: its own description and a
 * JSON-schema `parameters` object. `execute` receives the arguments object the
 * model sent and the call context. Preferred over a bare positional function,
 * whose arguments the model can only guess.
 */
export interface PluginToolDefinition {
  readonly description: string
  readonly parameters: Readonly<Record<string, unknown>>
  /** Per-call time limit for managed plugins (default 60 000 ms, at most one hour). */
  readonly timeoutMs?: number
  execute(input: Record<string, unknown>, context: PluginToolContext): unknown | Promise<unknown>
}

/** Model-facing description of a structured tool, as validated at registration. */
export interface PluginToolSchema {
  readonly description: string
  readonly parameters: Readonly<Record<string, unknown>>
  readonly timeoutMs?: number
}

/** A plugin-owned slash command and its registration options. */
export interface PluginCommand {
  readonly handler: SlashHandler
  readonly options: SlashPluginOptions
}

const MAX_TOOL_DESCRIPTION = 1_024
const PLUGIN_COMMAND_TOKEN = /^[a-z0-9][a-z0-9_-]*$/
/** Schemas of structured tools, keyed by the callable the registry stores. */
const TOOL_SCHEMAS = new WeakMap<PluginTool, PluginToolSchema>()

/** The schema a structured tool was registered with; undefined for positional tools. */
export function pluginToolSchema(tool: PluginTool | undefined): PluginToolSchema | undefined {
  return tool ? TOOL_SCHEMAS.get(tool) : undefined
}

/** Validate a structured definition at the registration boundary and adapt it to the registry's callable form. */
function pluginToolCallable(name: string, value: PluginTool | PluginToolDefinition): PluginTool {
  if (typeof value === 'function') return value
  if (!value || typeof value !== 'object') throw new TypeError(`Plugin tool '${name}' must be a function or { description, parameters, execute }`)
  const description = typeof value.description === 'string' ? value.description.trim() : ''
  if (!description || description.length > MAX_TOOL_DESCRIPTION) {
    throw new TypeError(`Plugin tool '${name}' needs a description of 1–${MAX_TOOL_DESCRIPTION} characters`)
  }
  if (typeof value.execute !== 'function') throw new TypeError(`Plugin tool '${name}' must define execute(input, context)`)
  const raw = value.parameters as unknown
  if (!raw || typeof raw !== 'object' || Array.isArray(raw) || (raw as { type?: unknown }).type !== 'object') {
    throw new TypeError(`Plugin tool '${name}' parameters must be a JSON schema with type "object"`)
  }
  let parameters: Readonly<Record<string, unknown>>
  try { parameters = JSON.parse(JSON.stringify(raw)) as Record<string, unknown> }
  catch { throw new TypeError(`Plugin tool '${name}' parameters must be JSON-serializable`) }
  const timeoutMs = value.timeoutMs as unknown
  if (timeoutMs !== undefined && (typeof timeoutMs !== 'number' || !Number.isInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > MAX_TOOL_TIMEOUT_MS)) {
    throw new TypeError(`Plugin tool '${name}' timeoutMs must be an integer from 1 to ${MAX_TOOL_TIMEOUT_MS}`)
  }
  const definition = value
  const tool: PluginTool = (input, context) => definition.execute(
    input && typeof input === 'object' && !Array.isArray(input) ? input as Record<string, unknown> : {},
    context && typeof context === 'object' ? context as PluginToolContext : {},
  )
  TOOL_SCHEMAS.set(tool, Object.freeze({ description, parameters: Object.freeze(parameters), ...(timeoutMs === undefined ? {} : { timeoutMs }) }))
  return tool
}

function pluginCommandToken(name: string): string {
  const token = typeof name === 'string' ? name.trim().replace(/^\/+/, '').toLowerCase() : ''
  if (!PLUGIN_COMMAND_TOKEN.test(token)) throw new TypeError('Plugin command names may contain only letters, numbers, hyphens, or underscores')
  return token
}

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
  /** What the plugin needs; see {@link PluginPermissions} for what is and is not enforced. */
  readonly permissions?: PluginPermissions
}

export interface RegisteredPlugin {
  readonly sourcePath?: string
  readonly channels: Map<string, unknown>
  readonly commands: Map<string, PluginCommand>
  readonly hooks: Map<string, HookCallback>
  readonly meta: RequiredPluginMeta
  provider: PluginLlmProviderFactory | undefined
  readonly tools: Map<string, PluginTool>
}

export interface PluginInventoryEntry {
  readonly name: string
  readonly description: string
  readonly version: string
  readonly source: { readonly kind: 'module'; readonly path: string } | { readonly kind: 'host-registration' }
  readonly tools: readonly string[]
  readonly hooks: readonly string[]
  readonly channels: readonly string[]
  readonly providers: readonly string[]
  readonly dependencies: readonly string[]
  /** Slash commands the plugin contributes, without the leading slash. */
  readonly commands: readonly string[]
  /** Descriptions of the plugin's structured tools (positional tools have none). */
  readonly toolDescriptions: Readonly<Record<string, string>>
  /** Declared permissions (empty lists when none were declared). */
  readonly permissions: NormalizedPluginPermissions
}

interface RequiredPluginMeta {
  readonly author: string
  readonly dependencies: readonly string[]
  readonly description: string
  readonly name: string
  readonly pluginType: PluginType
  readonly version: string
  readonly versionConstraints: Readonly<Record<string, string>>
  readonly permissions: NormalizedPluginPermissions
}

export class PluginConflictError extends Error {
  constructor(readonly resource: string, readonly existing: string) {
    super(`Plugin '${resource}' conflicts with existing plugin '${existing}'`)
    this.name = 'PluginConflictError'
  }
}

/**
 * Registration uses an isolated capability view and publishes only after success.
 * Await all capability registrations before resolving. Retained read methods see
 * the live registry after commit; later mutations of this view are rejected.
 * Arbitrary module side effects outside the registry are not transactional.
 */
export interface XerxesPluginModule {
  readonly register?: (registry: PluginRegistry) => void | Promise<void>
}

/** Explicit host trust decisions for dynamic plugin execution. */
export interface PluginDiscoveryOptions {
  /**
   * Explicit host opt-in per module: file names relative to the plugin directory or absolute
   * module paths that are permitted to execute. When supplied, every other module is skipped
   * with a warning instead of being imported.
   */
  readonly allowedModules?: readonly string[]
}

/** Point-in-time view of every capability index, used to roll back a failed plugin registration. */
interface PluginRegistrationSnapshot {
  readonly channels: ReadonlySet<string>
  readonly commands: ReadonlySet<string>
  readonly hooks: ReadonlyMap<string, number>
  readonly plugins: ReadonlySet<string>
  readonly providers: ReadonlySet<string>
  readonly tools: ReadonlySet<string>
}

/** Owns plugin capability indexes and validates dependency-compatible load order. */
export class PluginRegistry {
  private readonly channels = new Map<string, { readonly owner: string; readonly value: unknown }>()
  private readonly commands = new Map<string, { readonly owner: string; readonly value: PluginCommand }>()
  private readonly hooks = new Map<string, Array<{ readonly callback: HookCallback; readonly owner: string }>>()
  private readonly plugins = new Map<string, RegisteredPlugin>()
  private readonly providers = new Map<string, { readonly owner: string; readonly value: PluginLlmProviderFactory }>()
  private readonly tools = new Map<string, { readonly owner: string; readonly value: PluginTool }>()
  private readonly failures: string[] = []
  private readonly loadedModules = new Set<string>()
  private discoveryQueue: Promise<void> = Promise.resolve()
  private registrationClosed = false
  private committedRegistry: PluginRegistry | undefined
  /** Module path currently executing register(), plus the plugin names it has registered. */
  private activeDiscovery: { readonly names: Set<string>; readonly path: string } | undefined

  get pluginNames(): string[] {
    if (this.committedRegistry) return this.committedRegistry.pluginNames
    return [...this.plugins.keys()]
  }

  /** Describe current registrations without invoking any plugin capability. */
  inventory(): PluginInventoryEntry[] {
    if (this.committedRegistry) return this.committedRegistry.inventory()
    return [...this.plugins.values()].map(plugin => ({
      name: plugin.meta.name, description: plugin.meta.description, version: plugin.meta.version,
      source: plugin.sourcePath ? { kind: 'module' as const, path: plugin.sourcePath } : { kind: 'host-registration' as const },
      tools: [...plugin.tools.keys()], hooks: [...plugin.hooks.keys()], channels: [...plugin.channels.keys()],
      providers: [...this.providers].filter(([, entry]) => entry.owner === plugin.meta.name).map(([name]) => name),
      dependencies: [...plugin.meta.dependencies],
      commands: [...plugin.commands.keys()],
      toolDescriptions: Object.fromEntries([...plugin.tools].flatMap(([name, tool]) => {
        const schema = TOOL_SCHEMAS.get(tool)
        return schema ? [[name, schema.description]] : []
      })),
      permissions: plugin.meta.permissions,
    }))
  }

  /** Formatted per-module errors captured during discovery, in discovery order. */
  get loadErrors(): readonly string[] {
    if (this.committedRegistry) return this.committedRegistry.loadErrors
    return [...this.failures]
  }

  async discover(directory: string, options: PluginDiscoveryOptions = {}): Promise<string[]> {
    this.assertRegistrationOpen()
    if (this.activeDiscovery) throw new Error("Nested plugin discovery during registration is unsupported")
    const operation = this.discoveryQueue.then(
      () => this.discoverPass(directory, options),
      () => this.discoverPass(directory, options),
    )
    this.discoveryQueue = operation.then(() => undefined, () => undefined)
    return operation
  }

  private async discoverPass(directory: string, options: PluginDiscoveryOptions): Promise<string[]> {
    if (!existsSync(directory)) return []
    if (isWorldWritable(directory)) {
      // A world-writable plugin directory lets any local user swap in arbitrary code; refuse to execute it.
      console.warn(`Plugin discovery skipped: directory is world-writable: ${directory}`)
      return []
    }
    const allowedModules = options.allowedModules === undefined
      ? undefined
      : new Set(options.allowedModules.map(entry => resolve(directory, entry)))
    const discovered: string[] = []
    // Discovery order must not depend on filesystem readdir order, or
    // cross-plugin mutation hooks would run in a platform-dependent sequence.
    const entries = readdirSync(directory, { withFileTypes: true })
      .sort((left, right) => left.name < right.name ? -1 : left.name > right.name ? 1 : 0)
    for (const entry of entries) {
      if (!entry.isFile() || entry.name.startsWith('_') || !/\.(?:[cm]?js|ts)$/.test(entry.name)) continue
      const path = join(directory, entry.name)
      if (allowedModules !== undefined && !allowedModules.has(resolve(path))) {
        console.warn(`Plugin discovery skipped module without explicit host opt-in: ${path}`)
        continue
      }
      if (isWorldWritable(path)) {
        // A world-writable module file lets any local user inject arbitrary
        // code into a trusted directory; refuse to execute it.
        console.warn(`Plugin discovery skipped: module is world-writable: ${path}`)
        continue
      }
      const href = pathToFileURL(path).href
      // An already-loaded module must not re-execute; its registrations would conflict with themselves.
      if (this.loadedModules.has(href)) continue
      let staged: PluginRegistry | undefined
      try {
        const module = await import(href) as XerxesPluginModule
        staged = this.registrationStage(path)
        const snapshot = staged.registrationSnapshot()
        await module.register?.(staged)
        staged.registrationClosed = true
        const added = this.commitRegistration(staged, snapshot)
        staged.committedRegistry = this
        staged.activeDiscovery = undefined
        staged.releaseRegistrationStage()
        this.loadedModules.add(href)
        discovered.push(...added)
      } catch (error) {
        if (staged) { staged.registrationClosed = true; staged.activeDiscovery = undefined; staged.releaseRegistrationStage() }
        const message = `${path}: ${errorMessage(error)}`
        this.failures.push(message)
        console.error(`Plugin discovery failed: ${message}`)
      }
    }
    return discovered
  }

  /**
   * Run a synchronous registration as if the module at `path` performed it,
   * with the same staging, ownership and conflict rules as discovery. Used to
   * rebuild registrations a plugin made in its own process; nothing is
   * imported here. Throws (leaving this registry untouched) on any failure.
   */
  registerFromModule(path: string, register: (registry: PluginRegistry) => void): string[] {
    this.assertRegistrationOpen()
    if (this.activeDiscovery) throw new Error('Nested plugin registration is unsupported')
    const staged = this.registrationStage(path)
    try {
      const snapshot = staged.registrationSnapshot()
      register(staged)
      staged.registrationClosed = true
      const added = this.commitRegistration(staged, snapshot)
      staged.committedRegistry = this
      return added
    } finally {
      staged.registrationClosed = true
      staged.activeDiscovery = undefined
      staged.releaseRegistrationStage()
    }
  }

  getAllChannels(): Record<string, unknown> {
    if (this.committedRegistry) return this.committedRegistry.getAllChannels()
    return Object.fromEntries([...this.channels].map(([name, entry]) => [name, entry.value]))
  }

  getAllCommands(): Record<string, PluginCommand> {
    if (this.committedRegistry) return this.committedRegistry.getAllCommands()
    return Object.fromEntries([...this.commands].map(([name, entry]) => [name, entry.value]))
  }

  /** The plugin that owns a tool, for model-facing descriptions. */
  toolOwner(name: string): string | undefined {
    if (this.committedRegistry) return this.committedRegistry.toolOwner(name)
    return this.tools.get(name)?.owner
  }

  getAllTools(): Record<string, PluginTool> {
    if (this.committedRegistry) return this.committedRegistry.getAllTools()
    return Object.fromEntries([...this.tools].map(([name, entry]) => [name, entry.value]))
  }

  getChannel(name: string): unknown | undefined {
    if (this.committedRegistry) return this.committedRegistry.getChannel(name)
    return this.channels.get(name)?.value
  }

  getHooks(name: string): HookCallback[] {
    if (this.committedRegistry) return this.committedRegistry.getHooks(name)
    return (this.hooks.get(name) ?? []).map(entry => entry.callback)
  }

  getLoadOrder(): string[] {
    if (this.committedRegistry) return this.committedRegistry.getLoadOrder()
    const graph: Record<string, string[]> = {}
    for (const [name, plugin] of this.plugins) {
      const dependencies = plugin.meta.dependencies.map(parseDependency).map(spec => spec.name)
      for (const dependency of Object.keys(plugin.meta.versionConstraints)) if (!dependencies.includes(dependency)) dependencies.push(dependency)
      graph[name] = dependencies
    }
    return new DependencyResolver().topologicalSort(graph)
  }

  getPlugin(name: string): RegisteredPlugin | undefined {
    if (this.committedRegistry) return this.committedRegistry.getPlugin(name)
    return this.plugins.get(name)
  }

  getProvider(name: string): PluginLlmProviderFactory | undefined {
    if (this.committedRegistry) return this.committedRegistry.getProvider(name)
    return this.providers.get(name)?.value
  }

  getTool(name: string): PluginTool | undefined {
    if (this.committedRegistry) return this.committedRegistry.getTool(name)
    return this.tools.get(name)?.value
  }

  registerChannel(name: string, channel: unknown, meta?: PluginMeta, pluginName?: string): void {
    this.assertRegistrationOpen()
    const owner = this.resolveOwner(meta, pluginName)
    this.registerUnique(this.channels, `channel:${name}`, name, channel, owner)
    this.plugins.get(owner)?.channels.set(name, channel)
  }

  registerHook(name: HookPoint | string, callback: HookCallback, meta?: PluginMeta, pluginName?: string): void {
    this.assertRegistrationOpen()
    const owner = this.resolveOwner(meta, pluginName)
    const values = this.hooks.get(name) ?? []
    values.push({ callback, owner })
    this.hooks.set(name, values)
    this.plugins.get(owner)?.hooks.set(name, callback)
  }

  registerPlugin(meta: PluginMeta): RegisteredPlugin {
    this.assertRegistrationOpen()
    if (this.plugins.has(meta.name)) throw new PluginConflictError(meta.name, meta.name)
    this.activeDiscovery?.names.add(meta.name)
    const plugin: RegisteredPlugin = {
      ...(this.activeDiscovery ? { sourcePath: resolve(this.activeDiscovery.path) } : {}),
      meta: normalizeMeta(meta), tools: new Map(), hooks: new Map(), channels: new Map(), commands: new Map(), provider: undefined,
    }
    this.plugins.set(meta.name, plugin)
    return plugin
  }

  registerProvider(name: string, provider: PluginLlmProviderFactory, meta?: PluginMeta, pluginName?: string): void {
    this.assertRegistrationOpen()
    if (!isPluginLlmProviderFactory(provider)) {
      throw new TypeError(`Provider '${name}' must expose createClient(request)`)
    }
    const owner = this.resolveOwner(meta, pluginName)
    this.registerUnique(this.providers, `provider:${name}`, name, provider, owner)
    const plugin = this.plugins.get(owner)
    if (plugin) plugin.provider = provider
  }

  /**
   * Register a tool: either a positional function (legacy; the model passes
   * `{ args: [...] }`) or a {@link PluginToolDefinition} with its own
   * description and JSON-schema parameters, validated here.
   */
  registerTool(name: string, tool: PluginTool | PluginToolDefinition, meta?: PluginMeta, pluginName?: string): void {
    this.assertRegistrationOpen()
    const callable = pluginToolCallable(name, tool)
    const owner = this.resolveOwner(meta, pluginName)
    this.registerUnique(this.tools, `tool:${name}`, name, callable, owner)
    this.plugins.get(owner)?.tools.set(name, callable)
  }

  /**
   * Register a slash command. The host decides whether to expose it; the
   * handler receives the argument text and the invoking session's context.
   */
  registerCommand(name: string, handler: SlashHandler, options: SlashPluginOptions = {}, meta?: PluginMeta, pluginName?: string): void {
    this.assertRegistrationOpen()
    if (typeof handler !== 'function') throw new TypeError(`Plugin command '${name}' handler must be a function`)
    const token = pluginCommandToken(name)
    const owner = this.resolveOwner(meta, pluginName)
    const command: PluginCommand = Object.freeze({ handler, options: Object.freeze({ ...options }) })
    this.registerUnique(this.commands, `command:${token}`, token, command, owner)
    this.plugins.get(owner)?.commands.set(token, command)
  }

  unregisterPlugin(name: string): void {
    this.assertRegistrationOpen()
    const discovery = this.activeDiscovery
    if (discovery && this.plugins.has(name) && !discovery.names.has(name)) {
      throw new Error(`Plugin module ${discovery.path} cannot unregister existing plugin '${name}' during registration`)
    }
    if (!this.plugins.delete(name)) return
    for (const [tool, entry] of this.tools) if (entry.owner === name) this.tools.delete(tool)
    for (const [provider, entry] of this.providers) if (entry.owner === name) this.providers.delete(provider)
    for (const [channel, entry] of this.channels) if (entry.owner === name) this.channels.delete(channel)
    for (const [command, entry] of this.commands) if (entry.owner === name) this.commands.delete(command)
    for (const [hook, entries] of this.hooks) {
      const retained = entries.filter(entry => entry.owner !== name)
      if (retained.length) this.hooks.set(hook, retained)
      else this.hooks.delete(hook)
    }
  }

  validateDependencies(): string[] {
    if (this.committedRegistry) return this.committedRegistry.validateDependencies()
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
    if (this.committedRegistry) return this.committedRegistry.versionConflicts(name, version)
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

  private registrationSnapshot(): PluginRegistrationSnapshot {
    return {
      plugins: new Set(this.plugins.keys()),
      tools: new Set(this.tools.keys()),
      providers: new Set(this.providers.keys()),
      channels: new Set(this.channels.keys()),
      commands: new Set(this.commands.keys()),
      hooks: new Map([...this.hooks].map(([name, entries]) => [name, entries.length])),
    }
  }

  private releaseRegistrationStage(): void {
    this.plugins.clear()
    this.tools.clear()
    this.providers.clear()
    this.channels.clear()
    this.commands.clear()
    this.hooks.clear()
  }

  private assertRegistrationOpen(): void {
    if (this.registrationClosed) throw new Error('Plugin registration is closed; capabilities must be registered before register() settles')
  }

  /** A module receives an isolated registry; live consumers never see partial registration. */
  private registrationStage(path: string): PluginRegistry {
    const stage = new PluginRegistry()
    for (const [name, plugin] of this.plugins) stage.plugins.set(name, {
      ...plugin, meta: normalizeMeta(plugin.meta), tools: new Map(plugin.tools),
      hooks: new Map(plugin.hooks), channels: new Map(plugin.channels), commands: new Map(plugin.commands),
    })
    for (const [name, entry] of this.tools) stage.tools.set(name, entry)
    for (const [name, entry] of this.providers) stage.providers.set(name, entry)
    for (const [name, entry] of this.channels) stage.channels.set(name, entry)
    for (const [name, entry] of this.commands) stage.commands.set(name, entry)
    for (const [name, entries] of this.hooks) stage.hooks.set(name, [...entries])
    stage.activeDiscovery = { names: new Set(), path }
    return stage
  }

  /** Recheck live conflicts after awaits, then publish all deltas synchronously. */
  private commitRegistration(stage: PluginRegistry, before: PluginRegistrationSnapshot): string[] {
    const plugins = [...stage.plugins].filter(([name]) => !before.plugins.has(name))
    const tools = [...stage.tools].filter(([name]) => !before.tools.has(name))
    const providers = [...stage.providers].filter(([name]) => !before.providers.has(name))
    const channels = [...stage.channels].filter(([name]) => !before.channels.has(name))
    const commands = [...stage.commands].filter(([name]) => !before.commands.has(name))
    for (const [name] of plugins) if (this.plugins.has(name)) throw new PluginConflictError(name, name)
    for (const [name] of commands) { const existing = this.commands.get(name); if (existing) throw new PluginConflictError('command:' + name, existing.owner) }
    for (const [name] of tools) { const existing = this.tools.get(name); if (existing) throw new PluginConflictError('tool:' + name, existing.owner) }
    for (const [name] of providers) { const existing = this.providers.get(name); if (existing) throw new PluginConflictError('provider:' + name, existing.owner) }
    for (const [name] of channels) { const existing = this.channels.get(name); if (existing) throw new PluginConflictError('channel:' + name, existing.owner) }
    for (const [name, plugin] of plugins) this.plugins.set(name, plugin)
    for (const [name, entry] of tools) this.tools.set(name, entry)
    for (const [name, entry] of providers) this.providers.set(name, entry)
    for (const [name, entry] of channels) this.channels.set(name, entry)
    for (const [name, entry] of commands) this.commands.set(name, entry)
    for (const [name, entries] of stage.hooks) {
      const added = entries.slice(before.hooks.get(name) ?? 0)
      if (added.length) this.hooks.set(name, [...(this.hooks.get(name) ?? []), ...added])
    }
    return plugins.map(([name]) => name)
  }

  private resolveOwner(meta: PluginMeta | undefined, explicitOwner: string | undefined): string {
    if (meta && !this.plugins.has(meta.name)) this.registerPlugin(meta)
    let owner = explicitOwner ?? meta?.name ?? '__standalone__'
    const discovery = this.activeDiscovery
    if (discovery !== undefined && owner !== '__standalone__' && !discovery.names.has(owner)) {
      // A module must not attribute its capabilities to a plugin it did not register itself.
      const rebound = discovery.names.size === 1 ? [...discovery.names][0]! : '__standalone__'
      console.warn(
        `Plugin module ${discovery.path} cannot register under foreign plugin '${owner}'; attributing to '${rebound}' instead`,
      )
      owner = rebound
    }
    return owner
  }
}

/**
 * World-writable paths must never be a source of executable plugin code —
 * this covers both the discovery directory and each module file imported
 * from it.
 *
 * The POSIX mode bits this reads do not exist on Windows, where access is an ACL
 * and `stat().mode` is a synthesized value: Node reports 0o666/0o777-style modes
 * derived from the read-only attribute alone, so the other-write bit is set for
 * essentially every path. Evaluating it there would reject every plugin
 * directory on the machine as unsafe. Windows containment is enforced by the
 * discovery roots (per-user profile directories) instead, so this check reports
 * "not world-writable" rather than pretending to an answer it cannot compute.
 */
export function isWorldWritable(path: string, platform: NodeJS.Platform = process.platform): boolean {
  if (isWindows(platform)) {
    return false
  }
  try {
    return (statSync(path).mode & 0o002) !== 0
  } catch {
    // Dynamic imports execute arbitrary code. If permissions cannot be inspected, withhold the
    // path rather than treating an unknown ownership boundary as safe.
    return true
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
    permissions: normalizePluginPermissions(meta.permissions, meta.name),
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
