// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { createHash } from 'node:crypto'
import { PluginRegistry, type PluginInventoryEntry, type PluginTool } from './plugins.js'
import type { SlashPluginContext, SlashPluginRegistry } from './slashPlugins.js'
import { withFileLock } from '../session/daemonTranscript.js'
import type { ToolRegistry } from '../executors/toolRegistry.js'
import {
  loadPluginModule,
  moduleInventory,
  publishPluginTool,
  PluginUsageLedger,
  type LivePluginTool,
  type ManagedPluginInventoryEntry,
  type PluginModuleLoadOptions,
  type PluginModuleRuntime,
  type PluginToolServices,
} from './pluginRuntime.js'
import { PluginSecretStore } from './pluginSecrets.js'
import { exportPluginSources, looksLikeGitSource, PluginSourceCache, renderGitPreview, type GitPluginPreview, type GitRunner } from './pluginShare.js'
import { ProjectPlugins } from './projectPlugins.js'

export type { ManagedPluginInventoryEntry } from './pluginRuntime.js'

/** What `/plugins` can do to an installed module. */
export type ManagedPluginAction = 'install' | 'enable' | 'disable' | 'uninstall' | 'reload'
export const MANAGED_PLUGIN_ACTIONS: readonly ManagedPluginAction[] = Object.freeze(['install', 'enable', 'disable', 'uninstall', 'reload'])

export interface ManagedPluginChangeOptions {
  /** `install` only: keep the plugin in memory and out of the saved manifest. */
  readonly temporary?: boolean
}

export interface ManagedPluginsOptions {
  /** Secret values for plugins; defaults to `plugin-secrets.json` beside the manifest. */
  readonly secrets?: PluginSecretStore
  readonly usage?: PluginUsageLedger
  /** Where `/plugins install <git-url>` clones; defaults to `plugins/sources` beside the manifest. */
  readonly sourcesDirectory?: string
  readonly git?: GitRunner
  /** Process options for every plugin module (environment source, Bun executable, load timeout). */
  readonly load?: PluginModuleLoadOptions
  /** Trusted project-plugin hashes; defaults to `project-plugin-trust.json` beside the manifest. */
  readonly projectTrustFile?: string
}

interface InstalledModule { path: string; enabled: boolean; names: string[] }
// `error` marks an enabled entry whose module could not be loaded at startup.
// It stays enabled in the manifest, so fixing the file and reloading brings it
// back, but it registers nothing until then.
interface LoadedModule extends InstalledModule {
  registry: PluginRegistry
  /** The plugin's own process (absent when nothing is loaded). */
  runtime?: PluginModuleRuntime
  fingerprint?: string
  error?: string
  /** Module files the plugin process loaded from the plugin's folder. */
  files?: readonly string[]
  stale?: boolean
  /** Lives only in this process; never written to the manifest. */
  temporary?: boolean
}

/** Whether a secret is declared by the plugin and whether a value is stored. */
export interface PluginSecretStatus {
  readonly plugin: string
  readonly declared: readonly string[]
  readonly set: readonly string[]
}

/**
 * Explicit local module opt-in, persisted separately from automatic workspace
 * discovery. Every module runs in its own Bun process (see pluginHost.ts); the
 * daemon only holds a validated description of what it registered.
 */
export class ManagedPlugins {
  private modules: LoadedModule[] = []
  private queue: Promise<unknown> = Promise.resolve()
  private synced: { readonly registry: SlashPluginRegistry; readonly names: Set<string> } | undefined
  /**
   * Every tool registry this manager filled, held weakly: the daemon rebuilds
   * registries on each settings change, and a running turn keeps using the
   * one it started with. A plugin change is pushed into all that are alive,
   * so it reaches running turns at their next step.
   */
  private readonly registries: WeakRef<ToolRegistry>[] = []
  private readonly published = new WeakMap<ToolRegistry, readonly string[]>()
  private readonly listeners = new Set<() => void>()
  /** Processes started during the current operation; stopped if it fails. */
  private created: PluginModuleRuntime[] = []
  readonly usage: PluginUsageLedger
  readonly secrets: PluginSecretStore
  readonly sources: PluginSourceCache
  readonly project: ProjectPlugins
  private readonly services: PluginToolServices

  constructor(private readonly manifest: string, private readonly options: ManagedPluginsOptions = {}) {
    const home = dirname(manifest)
    this.usage = options.usage ?? new PluginUsageLedger()
    this.secrets = options.secrets ?? new PluginSecretStore(join(home, 'plugin-secrets.json'))
    this.sources = new PluginSourceCache(options.sourcesDirectory ?? join(home, 'plugins', 'sources'), options.git ? { git: options.git } : {})
    this.services = { usage: this.usage, secrets: this.secrets }
    this.project = new ProjectPlugins({
      trustFile: options.projectTrustFile ?? join(home, 'project-plugin-trust.json'),
      services: this.services,
      ...(options.load ? { load: options.load } : {}),
      reservedTools: () => new Set(this.modules.filter(item => item.enabled && !item.error).flatMap(item => Object.keys(item.registry.getAllTools()))),
    })
  }

  /** Called after every committed change (install, reload, uninstall, ...). Returns an unsubscribe. */
  onChange(listener: () => void): () => void {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  private async readManifest(): Promise<InstalledModule[]> {
    let raw: unknown
    try { raw = JSON.parse(await readFile(this.manifest, 'utf8')) }
    catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []; throw error }
    if (!Array.isArray(raw) || raw.length > 100) throw new Error('Invalid managed plugin manifest')
    const parsed: InstalledModule[] = []
    for (const item of raw) {
      if (!item || typeof item.path !== 'string' || !isAbsolute(item.path) || resolve(item.path) !== item.path || typeof item.enabled !== 'boolean' || !Array.isArray(item.names) || !item.names.length || !item.names.every((name: unknown) => typeof name === 'string' && name.length > 0)) throw new Error('Invalid managed plugin entry')
      parsed.push({ path: item.path, enabled: item.enabled, names: item.names })
    }
    if (new Set(parsed.map(item => item.path)).size !== parsed.length || new Set(parsed.flatMap(item => item.names)).size !== parsed.flatMap(item => item.names).length) throw new Error('Duplicate managed plugin paths or names')
    return parsed
  }

  private async writeManifest(): Promise<void> {
    await mkdir(dirname(this.manifest), { recursive: true })
    const temporary = `${this.manifest}.${crypto.randomUUID()}.tmp`
    try {
      const saved = this.modules.filter(module => !module.temporary)
      await writeFile(temporary, JSON.stringify(saved.map(({ path, enabled, names }) => ({ path, enabled, names })), null, 2), { mode: 0o600 })
      await rename(temporary, this.manifest)
    } finally { await rm(temporary, { force: true }) }
  }

  private async fingerprint(path: string): Promise<string> { return createHash('sha256').update(await readFile(path)).digest('hex') }

  /** Stop processes that the new module list no longer references. */
  private retire(from: readonly LoadedModule[], to: readonly LoadedModule[]): void {
    const kept = new Set(to.flatMap(item => item.runtime ? [item.runtime] : []))
    for (const item of from) if (item.runtime && !kept.has(item.runtime)) item.runtime.terminate()
    for (const runtime of this.created) if (!kept.has(runtime)) runtime.terminate()
    this.created = []
  }

  // Startup is fault tolerant per module: one moved, broken or renamed module
  // must not throw out of the daemon's boot, or `/plugins disable` could not
  // reach it either.
  async load(): Promise<void> {
    const before = this.modules
    this.created = []
    try { await this.hydrate() }
    catch (error) { this.retire([], before); throw error }
    this.retire(before, this.modules)
    this.publish()
  }

  /**
   * Rebuild the in-memory view from the manifest. Code already loaded keeps
   * serving (an edited file is only marked stale until it is reloaded); a
   * module that cannot load is recorded with its error instead of throwing.
   * `disable` treats one entry as disabled; `skip` leaves one entry unloaded
   * because the caller is about to replace or remove it.
   */
  private async hydrate(target: { readonly disable?: string; readonly skip?: string } = {}): Promise<void> {
    const matches = (stored: InstalledModule, value: string | undefined) => value !== undefined && (stored.path === value || stored.names.includes(value))
    const loaded: LoadedModule[] = []
    for (const stored of await this.readManifest()) {
      const item = matches(stored, target.disable) ? { ...stored, enabled: false } : stored
      const previous = this.modules.find(entry => entry.path === item.path)
      const carried = {
        ...(previous?.fingerprint ? { fingerprint: previous.fingerprint } : {}),
        ...(previous?.files ? { files: previous.files } : {}),
      }
      const kept = previous?.runtime && !previous.error ? { runtime: previous.runtime } : {}
      if (!item.enabled || matches(stored, target.skip)) {
        loaded.push({ ...item, registry: previous?.registry ?? new PluginRegistry(), ...carried, ...kept, ...(item.enabled && previous?.error ? { error: previous.error } : {}) })
        continue
      }
      // A module that failed stays failed until it is reloaded or re-enabled,
      // so it cannot block changes to the other modules.
      if (previous?.error) { loaded.push({ ...item, registry: previous.registry, ...carried, error: previous.error }); continue }
      try {
        const sameNames = (registry: PluginRegistry) => registry.pluginNames.slice().sort().join('\0') === item.names.slice().sort().join('\0')
        if (previous?.registry.pluginNames.length && sameNames(previous.registry)) {
          const current = await this.fingerprint(item.path).catch(() => undefined)
          loaded.push({ ...item, registry: previous.registry, ...carried, ...kept, ...(previous.fingerprint && current !== previous.fingerprint ? { stale: true } : {}) })
          continue
        }
        const fingerprint = await this.fingerprint(item.path)
        const module = await this.readModule(item.path)
        if (!sameNames(module.registry)) throw new Error('Plugin registration names changed; run /plugins reload to accept the new names')
        loaded.push({ ...item, registry: module.registry, runtime: module.runtime, fingerprint, files: module.files })
      } catch (error) {
        loaded.push({ ...item, registry: new PluginRegistry(), ...carried, error: error instanceof Error ? error.message : String(error) })
      }
    }
    // Temporary installs are not in the manifest; they stay as they are.
    for (const kept of this.modules) {
      if (kept.temporary && !loaded.some(entry => entry.path === kept.path)) loaded.push(kept)
    }
    this.validate(loaded)
    this.modules = loaded
  }

  private validate(modules: readonly LoadedModule[]): void {
    const names = modules.flatMap(item => item.names)
    if (new Set(names).size !== names.length) throw new Error('Plugin name conflicts with another installed module')
    const active = modules.filter(item => item.enabled && !item.error)
    const toolNames = active.flatMap(item => Object.keys(item.registry.getAllTools()))
    if (new Set(toolNames).size !== toolNames.length) throw new Error('Plugin tool names conflict')
    const commandNames = active.flatMap(item => Object.keys(item.registry.getAllCommands()))
    if (new Set(commandNames).size !== commandNames.length) throw new Error('Plugin command names conflict with another installed module')
  }

  inventory(): ManagedPluginInventoryEntry[] {
    return this.modules.flatMap(item => moduleInventory(item, 'installed', this.usage))
  }

  /** Load one module in a fresh process. Its process is stopped if the surrounding operation fails. */
  private async readModule(path: string) {
    const module = await loadPluginModule(path, this.options.load ?? {})
    this.created.push(module.runtime)
    return module
  }

  private find(value: string): LoadedModule | undefined {
    return this.modules.find(entry => entry.path === value || entry.names.includes(value))
  }

  async change(action: ManagedPluginAction, value: string, options: ManagedPluginChangeOptions = {}): Promise<string> {
    const operation = this.queue.then(() => withFileLock(`${this.manifest}.lock`, async () => {
      const previous = [...this.modules]
      this.created = []
      let message: string
      try {
        await this.hydrate(action === 'disable' ? { disable: value } : action === 'uninstall' || action === 'reload' ? { skip: value } : {})
        if (action === 'install') {
          const path = resolve(value)
          const existing = this.modules.find(item => item.path === path)
          if (existing?.temporary && !options.temporary) {
            // Keeping a temporary install: the same code, now saved.
            const { temporary: _temporary, ...kept } = existing
            this.modules = this.modules.map(entry => entry === existing ? kept : entry)
            message = `Kept ${existing.names.join(', ')} permanently: it is saved and loads every time Xerxes starts.`
          } else {
            if (existing) throw new Error('Module is already installed; use enable, disable or reload')
            const { registry, runtime, files } = await this.readModule(path)
            const fingerprint = await this.fingerprint(path)
            this.modules = [...this.modules, { path, registry, runtime, enabled: true, names: registry.pluginNames, fingerprint, files, ...(options.temporary ? { temporary: true } : {}) }]
            const name = registry.pluginNames[0] ?? value
            message = options.temporary
              ? `Installed ${value} for this run of Xerxes only (not saved; gone after a restart). Its tools can be called from the next step. Install it again without the temporary option to keep it.`
              : `Installed and enabled ${value}. Its tools can be called from the next step. The module file stays where it is; after editing it run /plugins reload ${name}.`
            const declared = permissionSummary(registry)
            if (declared) message += `\n${declared}`
          }
        } else {
          const item = this.find(value)
          if (!item) throw new Error('Managed plugin not found; use /plugins list and its name or module path')
          if (action === 'uninstall') {
            this.modules = this.modules.filter(entry => entry !== item)
            message = `Uninstalled ${value}. Its tools and commands are gone from the next turn; the module file at ${item.path} was not touched.`
          } else if (action === 'disable') {
            this.modules = this.modules.map(entry => entry === item ? { ...entry, enabled: false } : entry)
            message = `Disabled ${value}. Calls are refused from now on; /plugins enable ${value} turns it back on.`
          } else {
            const fingerprint = await this.fingerprint(item.path)
            const fresh = action === 'reload' || !item.registry.pluginNames.length || item.error !== undefined || item.fingerprint !== fingerprint || !item.runtime
            const module = fresh ? await this.readModule(item.path) : { registry: item.registry, runtime: item.runtime, files: [...(item.files ?? [])] }
            const names = module.registry.pluginNames
            if (action === 'enable' && names.slice().sort().join('\0') !== item.names.slice().sort().join('\0')) {
              throw new Error('Plugin registration names changed; run /plugins reload to accept the new names')
            }
            const enabled = action === 'enable' ? true : item.enabled
            this.modules = this.modules.map(entry => entry === item ? {
              path: item.path, enabled, names, registry: module.registry, fingerprint, files: module.files,
              ...(module.runtime ? { runtime: module.runtime } : {}), ...(item.temporary ? { temporary: true } : {}),
            } : entry)
            const renamed = names.slice().sort().join(', ') !== item.names.slice().sort().join(', ')
            message = action === 'reload'
              ? `Reloaded ${value} from ${item.path} in a fresh plugin process${renamed ? ` (now provides ${names.join(', ')})` : ''}.${enabled ? ' The new code serves calls from now on.' : ' It is still disabled.'}`
              : `Enabled ${value}${fresh ? ' with the current code on disk' : ''}. Its tools are available from the next turn.`
          }
        }
        this.validate(this.modules)
        await this.writeManifest()
      } catch (error) {
        this.retire([], previous)
        this.modules = previous
        throw error
      }
      this.retire(previous, this.modules)
      // A disabled plugin keeps its description but not an idle process.
      for (const item of this.modules) if (!item.enabled) item.runtime?.terminate()
      this.publish()
      for (const listener of this.listeners) {
        try { listener() } catch (error) { console.error(`Plugin change listener failed: ${error instanceof Error ? error.message : String(error)}`) }
      }
      return message
    }, { waitMs: 30_000, staleMs: 60_000, label: 'plugin settings' }))
    this.queue = operation.catch(() => undefined)
    return operation
  }

  /** Declared permissions in one line (see {@link describePermissions}). */
  static describePermissions(permissions: ManagedPluginInventoryEntry['permissions']): string {
    return describePermissions(permissions)
  }

  /** Narrow a host or managed inventory row to a managed (installed or project) one. */
  static isManagedEntry(entry: PluginInventoryEntry): entry is ManagedPluginInventoryEntry {
    return 'scope' in entry && 'usage' in entry && 'module' in entry
  }

  /** Whether `/plugins install` text names a git URL rather than a local module. */
  static isGitSource(text: string): boolean {
    return looksLikeGitSource(text)
  }

  /** Clone a git source for review (nothing is loaded) and render its preview with a one-time install token. */
  async previewSource(url: string): Promise<{ readonly text: string; readonly preview: GitPluginPreview }> {
    const preview = await this.sources.preview(url)
    return { text: renderGitPreview(preview), preview }
  }

  /** Install a git source previewed by {@link PluginSourceCache.preview}; the token is single-use. */
  async installFromSource(token: string, options: ManagedPluginChangeOptions = {}): Promise<string> {
    const source = this.sources.consume(token)
    const message = await this.change('install', source.entry, options)
    return `${message}\nSource: ${source.url} (cloned to ${source.directory}).`
  }

  /** Copy an installed plugin's sources (never its secrets) to a new directory. */
  async exportPlugin(nameOrPath: string, destination: string): Promise<string[]> {
    const item = this.find(nameOrPath)
    if (!item) throw new Error('Managed plugin not found; use /plugins list and its name or module path')
    return exportPluginSources(item.path, item.files ?? [], destination)
  }

  /** Declared and stored secret names for a plugin (installed or a loaded project plugin). Never values. */
  async secretStatus(plugin: string, workspace?: string): Promise<PluginSecretStatus> {
    const entry = this.pluginEntry(plugin, workspace)
    return { plugin, declared: entry ? [...entry.permissions.secrets] : [], set: await this.secrets.names(plugin) }
  }

  /** Store a secret value. The plugin must exist and declare the name. The value is never echoed. */
  async setSecret(plugin: string, name: string, value: string, workspace?: string): Promise<void> {
    const entry = this.pluginEntry(plugin, workspace)
    if (!entry) throw new Error(`No plugin named ${plugin} is installed`)
    if (!entry.permissions.secrets.includes(name)) {
      throw new Error(`Plugin ${plugin} does not declare secret ${name}${entry.permissions.secrets.length ? ` (it declares ${entry.permissions.secrets.join(', ')})` : ''}`)
    }
    await this.secrets.set(plugin, name, value)
  }

  async unsetSecret(plugin: string, name: string): Promise<boolean> {
    return this.secrets.unset(plugin, name)
  }

  private pluginEntry(plugin: string, workspace?: string): ManagedPluginInventoryEntry | undefined {
    return this.inventory().find(entry => entry.name === plugin)
      ?? (workspace ? this.project.inventory(workspace).find(entry => entry.name === plugin) : undefined)
  }

  /** The tool to run for one call: refused when disabled anywhere, the reloaded code when there is one. */
  private async isLive(path: string): Promise<boolean> {
    const module = this.modules.find(entry => entry.path === path)
    if (module?.temporary) return module.enabled && !module.error
    return (await this.readManifest()).some(current => current.path === path && current.enabled)
  }

  private async liveTool(path: string, name: string, fallback: PluginTool, registry: PluginRegistry): Promise<LivePluginTool> {
    if (!(await this.isLive(path))) throw new Error('Plugin was disabled or uninstalled')
    const module = this.modules.find(entry => entry.path === path)
    if (!module || module.error || !module.registry.pluginNames.length) return { tool: fallback, registry }
    const tool = module.registry.getTool(name)
    if (!tool) throw new Error(`Plugin no longer provides tool ${name}; it was reloaded without it`)
    return { tool, registry: module.registry }
  }

  /**
   * Expose enabled plugin tools as `plugin_<name>`. A structured tool reaches
   * the model with its own description and parameter schema and receives the
   * turn's abort signal and its declared secrets; a positional tool takes
   * `{ args: [...] }`.
   */
  registerTools(tools: ToolRegistry): void {
    if (!this.registries.some(reference => reference.deref() === tools)) this.registries.push(new WeakRef(tools))
    this.populate(tools)
  }

  /** Push the current plugin set into every live registry this manager filled. */
  private publish(): void {
    for (let index = this.registries.length - 1; index >= 0; index -= 1) {
      const tools = this.registries[index]!.deref()
      if (tools) this.populate(tools)
      else this.registries.splice(index, 1)
    }
  }

  /** Replace this manager's plugin tools in one registry with the current ones. */
  private populate(tools: ToolRegistry): void {
    for (const name of this.published.get(tools) ?? []) tools.unregister(name)
    const names: string[] = []
    this.published.set(tools, names)
    for (const item of this.modules) {
      if (!item.enabled || item.error) continue
      const registry = item.registry
      for (const [name, callback] of Object.entries(registry.getAllTools())) {
        names.push(publishPluginTool(tools, registry, name, callback, () => this.liveTool(item.path, name, callback, registry), this.services))
      }
    }
  }

  /**
   * Publish enabled plugins' slash commands into a host registry, replacing
   * whatever this manager published before. Conflicts are returned as
   * messages, never thrown, so one plugin cannot stop the others or the host.
   */
  syncCommands(registry: SlashPluginRegistry): string[] {
    this.clearCommands()
    const names = new Set<string>()
    this.synced = { registry, names }
    const problems: string[] = []
    for (const item of this.modules) {
      if (!item.enabled || item.error) continue
      for (const [name, command] of Object.entries(item.registry.getAllCommands())) {
        try {
          registry.register(name, (args, context) => this.runCommand(item.path, name, args, context), command.options)
          names.add(name)
        } catch (error) {
          problems.push(`/${name} from ${item.path}: ${error instanceof Error ? error.message : String(error)}`)
        }
      }
    }
    return problems
  }

  /** Withdraw every command this manager published. */
  clearCommands(): void {
    if (!this.synced) return
    for (const name of this.synced.names) this.synced.registry.unregister(name)
    this.synced = undefined
  }

  /** Stop every plugin process (installed and project). Calls start them again on demand. */
  close(): void {
    for (const item of this.modules) item.runtime?.terminate()
    this.project.close()
  }

  private async runCommand(path: string, name: string, args: string, context: SlashPluginContext): Promise<unknown> {
    if (!(await this.isLive(path))) throw new Error('Plugin was disabled or uninstalled')
    const command = this.modules.find(entry => entry.path === path)?.registry.getAllCommands()[name]
    if (!command) throw new Error(`Plugin no longer provides /${name}`)
    return command.handler(args, context)
  }
}

/** One line per plugin naming what it declared, or '' when nothing was declared. */
export function permissionSummary(registry: PluginRegistry): string {
  return registry.inventory().map(plugin => {
    const text = describePermissions(plugin.permissions)
    return text ? `${plugin.name} declares: ${text}` : ''
  }).filter(Boolean).join('\n')
}

/** Declared permissions in one line, with what each kind actually means. */
export function describePermissions(permissions: ManagedPluginInventoryEntry['permissions']): string {
  return [
    permissions.network.length ? `network ${permissions.network.join(', ')} (fetch guard rail)` : '',
    permissions.env.length ? `env ${permissions.env.join(', ')} (only these reach its process)` : '',
    permissions.filesystem.length ? `filesystem ${permissions.filesystem.join(', ')} (declared, not enforced)` : '',
    permissions.secrets.length ? `secrets ${permissions.secrets.join(', ')}` : '',
  ].filter(Boolean).join('; ')
}
