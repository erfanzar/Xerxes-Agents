// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { readFile, realpath, stat } from 'node:fs/promises'
import { dirname, sep } from 'node:path'

import { resolveCommand } from '../bridge/commands.js'
import type { ToolRegistry } from '../executors/toolRegistry.js'
import { PluginProcess, type PluginInvokeOptions } from './pluginHost.js'
import type { PluginSecretStore } from './pluginSecrets.js'
import {
  isWorldWritable,
  normalizePluginPermissions,
  PluginRegistry,
  pluginToolSchema,
  PluginType,
  type PluginInventoryEntry,
  type PluginMeta,
  type PluginPermissions,
  type PluginTool,
  type PluginToolContext,
  type PluginToolDefinition,
} from './plugins.js'
import type { SlashPluginOptions } from './slashPlugins.js'

/** One plugin as `/plugins`, PluginManageTool and the desktop see it. */
export type ManagedPluginInventoryEntry = PluginInventoryEntry & {
  readonly enabled: boolean
  readonly module: string
  readonly loadError?: string
  /** The file on disk differs from the code in memory; `/plugins reload` loads it. */
  readonly stale?: boolean
  /** Installed for this run of Xerxes only: never saved, gone after a restart. */
  readonly temporary?: boolean
  /** Calls, failures and timing of each tool since the daemon started (in memory only). */
  readonly usage: Readonly<Record<string, PluginToolUsage>>
  /** Whether the plugin's own process is running right now (it starts on demand). */
  readonly running: boolean
  /** `installed`: the user's managed list. `project`: the workspace's `.xerxes/plugins`. */
  readonly scope: 'installed' | 'project'
  /** Project plugins only: whether this exact content is trusted. */
  readonly trusted?: boolean
}

/** The fields of a loaded module that its inventory is built from. */
export interface PluginModuleView {
  readonly path: string
  readonly enabled: boolean
  readonly names: readonly string[]
  readonly registry: PluginRegistry
  readonly runtime?: PluginModuleRuntime
  readonly error?: string
  readonly stale?: boolean
  readonly temporary?: boolean
}

const EMPTY_PERMISSIONS = normalizePluginPermissions(undefined, '')

/** Inventory rows for one module: its registered plugins, or a stub per name when nothing is loaded. */
export function moduleInventory(item: PluginModuleView, scope: 'installed' | 'project', usage: PluginUsageLedger, trusted?: boolean): ManagedPluginInventoryEntry[] {
  const running = item.runtime?.running === true
  const extra = scope === 'project' && trusted !== undefined ? { trusted } : {}
  const inventory = item.registry.inventory()
  if (inventory.length) {
    return inventory.map(plugin => ({
      ...plugin, enabled: item.enabled, module: item.path, usage: usage.forTools(plugin.tools), running, scope,
      ...(item.stale ? { stale: true } : {}), ...(item.temporary ? { temporary: true } : {}), ...extra,
    }))
  }
  return item.names.map(name => ({
    name, version: '', description: item.error ? `Failed to load: ${item.error}` : 'Disabled module',
    source: { kind: 'module' as const, path: item.path },
    tools: [], hooks: [], channels: [], providers: [], dependencies: [], commands: [], toolDescriptions: {},
    permissions: EMPTY_PERMISSIONS, enabled: false, module: item.path, usage: {}, running: false, scope,
    ...(item.error ? { loadError: item.error } : {}), ...extra,
  }))
}

/** Shared per-daemon services every published plugin tool uses. */
export interface PluginToolServices {
  readonly usage: PluginUsageLedger
  readonly secrets?: PluginSecretStore
}

/** The tool to run for one call and the registry that owns it (it changes on reload). */
export interface LivePluginTool {
  readonly tool: PluginTool
  readonly registry: PluginRegistry
}

/**
 * Register one plugin tool as `plugin_<name>`. A structured tool reaches the
 * model with its own description and schema and receives the turn's signal and
 * its declared secrets; a positional tool takes `{ args: [...] }`. Every call
 * is counted in the usage ledger. Returns the registered tool name.
 */
export function publishPluginTool(
  tools: ToolRegistry,
  registry: PluginRegistry,
  name: string,
  callback: PluginTool,
  live: () => Promise<LivePluginTool>,
  services: PluginToolServices,
  origin = 'plugin',
): string {
  const toolName = `plugin_${name}`
  if (!/^[a-zA-Z0-9_-]{1,64}$/.test(toolName)) throw new Error(`Invalid plugin tool name: ${name}`)
  const owner = registry.toolOwner(name)
  const meta = owner ? registry.getPlugin(owner)?.meta : undefined
  const from = meta ? `${origin} ${meta.name}` : `a native ${origin}`
  const schema = pluginToolSchema(callback)
  const definition = schema
    ? { type: 'function' as const, function: { name: toolName, description: `${schema.description} (from ${from})`, parameters: schema.parameters } }
    : {
      type: 'function' as const,
      function: {
        name: toolName,
        description: [meta?.description, `Positional tool ${name} from ${from}: pass the arguments it expects, in order, as {"args": [...]}.`].filter(Boolean).join(' — '),
        parameters: { type: 'object', properties: { args: { type: 'array', items: {} } }, required: ['args'], additionalProperties: false },
      },
    }
  tools.register(definition, async (inputs, context, signal) => {
    const current = await live()
    const started = performance.now()
    try {
      let value: unknown
      if (pluginToolSchema(current.tool)) {
        const currentOwner = current.registry.toolOwner(name)
        const declared = currentOwner ? current.registry.getPlugin(currentOwner)?.meta.permissions.secrets ?? [] : []
        const secrets = currentOwner && declared.length && services.secrets ? await services.secrets.values(currentOwner, declared) : undefined
        value = await current.tool(inputs, {
          ...(signal ? { signal } : {}),
          ...(context.sessionId ? { sessionId: context.sessionId } : {}),
          ...(context.agentId ? { agentId: context.agentId } : {}),
          ...(secrets ? { secrets } : {}),
        } satisfies PluginToolContext)
      } else {
        const args = inputs.args
        if (!Array.isArray(args)) throw new Error('args must be an array')
        value = await current.tool(...args)
      }
      services.usage.record(name, performance.now() - started)
      return value
    } catch (error) {
      services.usage.record(name, performance.now() - started, error)
      throw error
    }
    // Never deferred: a plugin installed mid-turn must be callable at the
    // next step without a tool search first.
  }, 'default', { defer: false })
  return toolName
}

/** Per-call limit for plugin tools and commands unless a tool sets `timeoutMs`. */
export const DEFAULT_PLUGIN_CALL_TIMEOUT_MS = 60_000
const MAX_LAST_ERROR_CHARS = 300

export type PluginInvoke = (
  kind: 'call' | 'command',
  payload: Readonly<Record<string, unknown>>,
  options: PluginInvokeOptions,
) => Promise<unknown>

/** In-memory usage of one plugin tool since the daemon started. */
export interface PluginToolUsage {
  readonly calls: number
  readonly failures: number
  readonly totalMs: number
  readonly meanMs: number
  readonly lastError?: string
  /** ISO time of the most recent call. */
  readonly lastUsedAt?: string
}

/** Counts calls, failures and time per plugin tool. Never persisted. */
export class PluginUsageLedger {
  private readonly stats = new Map<string, { calls: number; failures: number; totalMs: number; lastError?: string; lastUsedAt?: number }>()

  constructor(private readonly now: () => number = Date.now) {}

  record(tool: string, durationMs: number, error?: unknown): void {
    const entry = this.stats.get(tool) ?? { calls: 0, failures: 0, totalMs: 0 }
    entry.calls += 1
    entry.totalMs += Math.max(0, Math.round(durationMs))
    entry.lastUsedAt = this.now()
    if (error !== undefined) {
      entry.failures += 1
      const text = error instanceof Error ? error.message : String(error)
      entry.lastError = text.length > MAX_LAST_ERROR_CHARS ? `${text.slice(0, MAX_LAST_ERROR_CHARS - 1)}…` : text
    }
    this.stats.set(tool, entry)
  }

  get(tool: string): PluginToolUsage {
    const entry = this.stats.get(tool)
    if (!entry) return { calls: 0, failures: 0, totalMs: 0, meanMs: 0 }
    return {
      calls: entry.calls,
      failures: entry.failures,
      totalMs: entry.totalMs,
      meanMs: entry.calls ? Math.round(entry.totalMs / entry.calls) : 0,
      ...(entry.lastError === undefined ? {} : { lastError: entry.lastError }),
      ...(entry.lastUsedAt === undefined ? {} : { lastUsedAt: new Date(entry.lastUsedAt).toISOString() }),
    }
  }

  forTools(tools: readonly string[]): Record<string, PluginToolUsage> {
    return Object.fromEntries(tools.map(tool => [tool, this.get(tool)]))
  }
}

interface RuntimeSpec {
  readonly path: string
  readonly folders: readonly string[]
  readonly env: Readonly<Record<string, string>>
  readonly digest: string
  readonly signature: string
  readonly executable?: string
  readonly loadTimeoutMs?: number
}

/**
 * The live process behind one loaded plugin module. A process that crashed,
 * timed out or was stopped is started again on the next call — but only when
 * the module's files and registration are byte-for-byte what was loaded, so a
 * restart never silently swaps in edited code.
 */
export class PluginModuleRuntime {
  private starting: Promise<PluginProcess> | undefined
  private generation = 0

  constructor(private readonly spec: RuntimeSpec, private current: PluginProcess | undefined) {}

  get path(): string { return this.spec.path }

  get running(): boolean { return this.current?.alive === true }

  async invoke(kind: 'call' | 'command', payload: Readonly<Record<string, unknown>>, options: PluginInvokeOptions): Promise<unknown> {
    const child = await this.process()
    return child.invoke(kind, payload, options)
  }

  /** Stop the process. A later call starts a fresh one. */
  terminate(): void {
    this.generation += 1
    this.current?.terminate()
    this.current = undefined
  }

  private process(): Promise<PluginProcess> {
    if (this.current?.alive) return Promise.resolve(this.current)
    this.starting ??= this.respawn().finally(() => { this.starting = undefined })
    return this.starting
  }

  private async respawn(): Promise<PluginProcess> {
    const generation = this.generation
    const child = new PluginProcess({
      path: this.spec.path, folders: this.spec.folders, env: this.spec.env,
      ...(this.spec.executable ? { executable: this.spec.executable } : {}),
      ...(this.spec.loadTimeoutMs ? { loadTimeoutMs: this.spec.loadTimeoutMs } : {}),
    })
    try {
      const loaded = await child.load()
      const files = await loadedFiles(this.spec.path, loaded.files)
      if (registrationSignature(loaded.calls) !== this.spec.signature || await filesDigest(files) !== this.spec.digest) {
        throw new Error('the plugin source changed since it was loaded; run /plugins reload <name> to load the new code')
      }
      if (generation !== this.generation) throw new Error('the plugin was stopped')
    } catch (error) {
      child.terminate()
      throw new Error(`Could not restart the plugin process: ${error instanceof Error ? error.message : String(error)}`)
    }
    this.current = child
    return child
  }
}

export interface LoadedPluginModule {
  readonly registry: PluginRegistry
  readonly runtime: PluginModuleRuntime
  /** Absolute paths of the module files the plugin process loaded from its folder. */
  readonly files: readonly string[]
}

export interface PluginModuleLoadOptions {
  /** Where declared `permissions.env` values come from. Defaults to the daemon's environment. */
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly executable?: string
  readonly loadTimeoutMs?: number
}

/**
 * Load one plugin module in its own process and rebuild its registration in a
 * daemon-side {@link PluginRegistry} whose tools and commands call into that
 * process. The module's code never runs in the daemon.
 */
export async function loadPluginModule(path: string, options: PluginModuleLoadOptions = {}): Promise<LoadedPluginModule> {
  if (!/\.(?:[cm]?js|ts)$/.test(path)) throw new Error('Install expects a local native .ts/.js module exporting register(registry)')
  const info = await stat(path)
  if (!info.isFile()) throw new Error('Install expects a local native .ts/.js module exporting register(registry)')
  if (isWorldWritable(dirname(path)) || isWorldWritable(path)) throw new Error('The plugin module or its folder is world-writable; refusing to run it')
  const canonical = await realpath(path)
  const folders = [...new Set([dirname(path), dirname(canonical)])].map(folder => folder.endsWith(sep) ? folder : folder + sep)
  const exec = {
    ...(options.executable ? { executable: options.executable } : {}),
    ...(options.loadTimeoutMs ? { loadTimeoutMs: options.loadTimeoutMs } : {}),
  }
  const holder: { runtime?: PluginModuleRuntime } = {}
  const invoke: PluginInvoke = (kind, payload, invokeOptions) => holder.runtime
    ? holder.runtime.invoke(kind, payload, invokeOptions)
    : Promise.reject(new Error('Plugin is not running'))
  // First pass with an empty environment: what the plugin declares decides
  // which variables it may see, so it cannot read any before declaring them.
  let child = new PluginProcess({ path, folders, ...exec })
  try {
    let loaded = await child.load()
    const registry = new PluginRegistry()
    replayRegistration(registry, path, loaded.calls, invoke)
    validateManagedRegistry(registry)
    const env = declaredEnvironment(registry, options.environment ?? process.env)
    if (Object.keys(env).length) {
      child.terminate('was restarted with its declared environment')
      child = new PluginProcess({ path, folders, env, ...exec })
      const second = await child.load()
      if (registrationSignature(second.calls) !== registrationSignature(loaded.calls)) {
        throw new Error('The plugin registered different capabilities once given its declared environment; register() must not depend on environment variables')
      }
      loaded = second
    }
    const files = await loadedFiles(path, loaded.files)
    holder.runtime = new PluginModuleRuntime({
      path, folders, env, digest: await filesDigest(files), signature: registrationSignature(loaded.calls), ...exec,
    }, child)
    return { registry, runtime: holder.runtime, files }
  } catch (error) {
    child.terminate()
    throw error
  }
}

/** The rules every managed (installed or project) plugin must satisfy. */
export function validateManagedRegistry(registry: PluginRegistry): void {
  const inventory = registry.inventory()
  if (!inventory.length) throw new Error('Module did not register a plugin')
  if (inventory.flatMap(plugin => plugin.tools).some(name => !/^[a-zA-Z0-9_-]{1,57}$/.test(name))) throw new Error('Plugin tool names must use at most 57 letters, numbers, underscores or hyphens')
  if (inventory.some(plugin => plugin.hooks.length || plugin.providers.length || plugin.channels.length)) throw new Error('Managed installation supports tool and slash-command plugins. Provider, hook, and channel plugins require an embedding host.')
  const builtIn = inventory.flatMap(plugin => plugin.commands).find(name => resolveCommand(name) !== undefined)
  if (builtIn) throw new Error(`Plugin command /${builtIn} conflicts with a built-in command; rename it`)
  const dependencyErrors = registry.validateDependencies()
  if (dependencyErrors.length) throw new Error(dependencyErrors.join('\n'))
}

function declaredEnvironment(registry: PluginRegistry, source: Readonly<Record<string, string | undefined>>): Record<string, string> {
  const env: Record<string, string> = {}
  for (const plugin of registry.inventory()) {
    for (const name of plugin.permissions.env) {
      const value = source[name]
      if (typeof value === 'string') env[name] = value
    }
  }
  return env
}

function registrationSignature(calls: readonly unknown[]): string {
  return JSON.stringify(calls)
}

async function loadedFiles(path: string, reported: readonly string[]): Promise<string[]> {
  return [...new Set([await realpath(path), ...reported])].sort()
}

async function filesDigest(files: readonly string[]): Promise<string> {
  const hash = createHash('sha256')
  for (const file of files) {
    hash.update(file).update('\0')
    try { hash.update(await readFile(file)) } catch { hash.update('\0missing\0') }
    hash.update('\0')
  }
  return hash.digest('hex')
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Rebuild a typed plugin meta from the child's JSON copy, rejecting malformed fields. */
function metaOf(value: unknown): PluginMeta | undefined {
  if (value === undefined || value === null) return undefined
  if (!isRecord(value) || typeof value.name !== 'string' || !value.name.trim()) throw new TypeError('Plugin meta needs a non-empty string name')
  const text = (key: string): string | undefined => {
    const field = value[key]
    if (field === undefined) return undefined
    if (typeof field !== 'string') throw new TypeError(`Plugin '${String(value.name)}' meta.${key} must be a string`)
    return field
  }
  const dependencies = value.dependencies
  if (dependencies !== undefined && (!Array.isArray(dependencies) || !dependencies.every(item => typeof item === 'string'))) throw new TypeError(`Plugin '${value.name}' meta.dependencies must be an array of strings`)
  const constraints = value.versionConstraints
  if (constraints !== undefined && (!isRecord(constraints) || !Object.values(constraints).every(item => typeof item === 'string'))) throw new TypeError(`Plugin '${value.name}' meta.versionConstraints must map names to strings`)
  const pluginType = text('pluginType')
  if (pluginType !== undefined && !(Object.values(PluginType) as string[]).includes(pluginType)) throw new TypeError(`Plugin '${value.name}' meta.pluginType is not a known plugin type`)
  const description = text('description')
  const version = text('version')
  const author = text('author')
  return {
    name: value.name,
    ...(description === undefined ? {} : { description }),
    ...(version === undefined ? {} : { version }),
    ...(author === undefined ? {} : { author }),
    ...(pluginType === undefined ? {} : { pluginType: pluginType as PluginType }),
    ...(dependencies === undefined ? {} : { dependencies: dependencies as string[] }),
    ...(constraints === undefined ? {} : { versionConstraints: constraints as Record<string, string> }),
    ...(value.permissions === undefined ? {} : { permissions: value.permissions as PluginPermissions }),
  }
}

function callContext(context: PluginToolContext): Record<string, unknown> {
  return {
    ...(context.sessionId ? { sessionId: context.sessionId } : {}),
    ...(context.agentId ? { agentId: context.agentId } : {}),
    ...(context.secrets ? { secrets: { ...context.secrets } } : {}),
  }
}

function toolEntry(name: string, raw: unknown, invoke: PluginInvoke): PluginTool | PluginToolDefinition {
  // An unusable entry is handed to the registry unchanged in kind, so its own
  // validation produces the usual error message.
  if (!isRecord(raw) || raw.kind === 'invalid') return null as unknown as PluginTool
  const label = `Plugin tool ${name}`
  if (raw.kind === 'positional') {
    return (...args: unknown[]) => invoke('call', { name, args }, { timeoutMs: DEFAULT_PLUGIN_CALL_TIMEOUT_MS, label })
  }
  if (raw.parametersError === true) throw new TypeError(`Plugin tool '${name}' parameters must be JSON-serializable`)
  const timeoutMs = raw.timeoutMs
  return {
    description: raw.description as string,
    parameters: raw.parameters as Record<string, unknown>,
    ...(timeoutMs === undefined || timeoutMs === null ? {} : { timeoutMs: timeoutMs as number }),
    execute: (raw.hasExecute === true
      ? (input: Record<string, unknown>, context: PluginToolContext) => invoke('call', { name, input, context: callContext(context) }, {
        timeoutMs: typeof timeoutMs === 'number' ? timeoutMs : DEFAULT_PLUGIN_CALL_TIMEOUT_MS,
        label,
        ...(context.signal ? { signal: context.signal } : {}),
      })
      : undefined) as PluginToolDefinition['execute'],
  }
}

/** Replay what the child recorded through the registry's normal validation and staging. */
function replayRegistration(registry: PluginRegistry, path: string, calls: readonly unknown[], invoke: PluginInvoke): void {
  registry.registerFromModule(path, staged => {
    for (const raw of calls) {
      if (!isRecord(raw)) throw new Error('Malformed plugin registration')
      const meta = metaOf(raw.meta)
      const owner = typeof raw.pluginName === 'string' ? raw.pluginName : undefined
      const name = typeof raw.name === 'string' ? raw.name : ''
      if (raw.op === 'plugin') {
        if (!meta) throw new TypeError('registerPlugin needs { name }')
        staged.registerPlugin(meta)
      } else if (raw.op === 'tool') {
        staged.registerTool(name, toolEntry(name, raw.tool, invoke), meta, owner)
      } else if (raw.op === 'command') {
        if (raw.options !== undefined && raw.options !== null && !isRecord(raw.options)) throw new TypeError(`Plugin command '${name}' options must be an object`)
        const handler = raw.handler === true
          ? (args: string, context: unknown) => invoke('command', { name, args, context: isRecord(context) ? { ...context } : {} }, { timeoutMs: DEFAULT_PLUGIN_CALL_TIMEOUT_MS, label: `Plugin command /${name}` })
          : undefined
        staged.registerCommand(name, handler as never, (raw.options ?? {}) as SlashPluginOptions, meta, owner)
      } else if (raw.op === 'hook') {
        staged.registerHook(name, () => undefined, meta, owner)
      } else if (raw.op === 'provider') {
        staged.registerProvider(name, { createClient: () => { throw new Error('Provider plugins require an embedding host') } }, meta, owner)
      } else if (raw.op === 'channel') {
        staged.registerChannel(name, {}, meta, owner)
      } else if (raw.op === 'unregister') {
        staged.unregisterPlugin(name)
      } else {
        throw new Error('Malformed plugin registration')
      }
    }
  })
}
