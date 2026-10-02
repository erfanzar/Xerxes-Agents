// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, resolve } from 'node:path'
import { PluginRegistry, type PluginInventoryEntry } from './plugins.js'
import { createHash } from 'node:crypto'
import { withFileLock } from '../session/daemonTranscript.js'
import type { ToolRegistry } from '../executors/toolRegistry.js'

interface InstalledModule { path: string; enabled: boolean; names: string[] }
// `error` marks an enabled entry whose module could not be loaded at startup.
// It stays enabled in the manifest, so fixing the file and restarting brings it
// back, but it registers nothing until then.
interface LoadedModule extends InstalledModule { registry: PluginRegistry; fingerprint?: string; error?: string }
/** Explicit local module opt-in, persisted separately from automatic workspace discovery. */
export class ManagedPlugins {
  private modules: LoadedModule[] = []
  private queue: Promise<unknown> = Promise.resolve()
  constructor(private readonly manifest: string) {}
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
  private async fingerprint(path: string): Promise<string> { return createHash('sha256').update(await readFile(path)).digest('hex') }
  // Startup is fault tolerant per module: one moved, broken or renamed module
  // used to throw out of the daemon's boot, and with no daemon running
  // `/plugins disable` could not reach it either.
  async load(): Promise<void> { return this.hydrate(undefined, true) }
  private async hydrate(disableValue?: string, tolerateBroken = false): Promise<void> {
    const loaded: LoadedModule[] = []
    for (const stored of await this.readManifest()) {
      const item = disableValue && (stored.path === disableValue || stored.names.includes(disableValue)) ? { ...stored, enabled: false } : stored
      const previous = this.modules.find(entry => entry.path === item.path)
      if (!item.enabled) { loaded.push({ ...item, registry: previous?.registry ?? new PluginRegistry(), ...(previous?.fingerprint ? { fingerprint: previous.fingerprint } : {}) }); continue }
      // A module that failed at startup stays failed until it is re-enabled or
      // the daemon restarts, so it cannot block changes to the other modules.
      if (previous?.error) { loaded.push({ ...item, registry: previous.registry, error: previous.error }); continue }
      try {
        const fingerprint = await this.fingerprint(item.path)
        if (previous?.fingerprint && previous.fingerprint !== fingerprint) throw new Error('Plugin source changed. Restart Xerxes to load the updated module.')
        const registry = previous?.registry.pluginNames.length ? previous.registry : await this.readModule(item.path)
        if (registry.pluginNames.slice().sort().join('\0') !== item.names.slice().sort().join('\0')) throw new Error('Plugin registration names changed; restore the module or remove its manifest entry before reinstalling')
        loaded.push({ ...item, registry, fingerprint })
      } catch (error) {
        if (!tolerateBroken) throw error
        loaded.push({ ...item, registry: new PluginRegistry(), error: error instanceof Error ? error.message : String(error) })
      }
    }
    this.validate(loaded)
    this.modules = loaded
  }
  private validate(modules: LoadedModule[]): void {
    const names = modules.flatMap(item => item.names)
    if (new Set(names).size !== names.length) throw new Error('Plugin name conflicts with another installed module')
    const toolNames = modules.filter(item => item.enabled && !item.error).flatMap(item => Object.keys(item.registry.getAllTools()))
    if (new Set(toolNames).size !== toolNames.length) throw new Error('Plugin tool names conflict')
  }
  inventory(): Array<PluginInventoryEntry & { enabled: boolean; module: string; loadError?: string }> {
    return this.modules.flatMap(item => item.registry.inventory().length
      ? item.registry.inventory().map(plugin => ({ ...plugin, enabled: item.enabled, module: item.path }))
      : item.names.map(name => ({ name, version: '', description: item.error ? `Failed to load: ${item.error}` : 'Disabled module', source: { kind: 'module' as const, path: item.path }, tools: [], hooks: [], channels: [], providers: [], dependencies: [], enabled: false, module: item.path, ...(item.error ? { loadError: item.error } : {}) })))
  }
  private async readModule(path: string): Promise<PluginRegistry> {
    if (!/\.(?:[cm]?js|ts)$/.test(path) || !(await stat(path)).isFile()) throw new Error('Install expects a local native .ts/.js module exporting register(registry)')
    const registry = new PluginRegistry()
    await registry.discover(dirname(path), { allowedModules: [path] })
    if (registry.loadErrors.length) throw new Error(registry.loadErrors.join('\n'))
    const inventory = registry.inventory()
    if (!inventory.length) throw new Error('Module did not register a plugin')
    if (inventory.flatMap(p => p.tools).some(name => !/^[a-zA-Z0-9_-]{1,57}$/.test(name))) throw new Error('Plugin tool names must use at most 57 letters, numbers, underscores or hyphens')
    if (inventory.some(p => p.hooks.length || p.providers.length || p.channels.length)) throw new Error('This TUI host supports native tool plugins. Provider, hook, and channel plugins require an embedding host.')
    if (registry.validateDependencies().length) throw new Error(registry.validateDependencies().join('\n'))
    return registry
  }
  async change(action: 'install' | 'enable' | 'disable', value: string): Promise<string> {
    const operation = this.queue.then(() => withFileLock(`${this.manifest}.lock`, async () => {
      const previous = [...this.modules]
      await this.hydrate(action === 'disable' ? value : undefined);
      if (action === 'install') {
        const path = resolve(value)
        if (this.modules.some(item => item.path === path)) throw new Error('Module is already installed; use enable or disable')
        const registry = await this.readModule(path)
        this.modules = [...this.modules, { path, registry, enabled: true, names: registry.pluginNames, fingerprint: await this.fingerprint(path) }]
      } else {
        const item = this.modules.find(entry => entry.path === value || entry.names.includes(value))
        if (!item) throw new Error('Managed plugin not found; use /plugins list and its name or module path')
        if (action === 'enable' && item.fingerprint && item.fingerprint !== await this.fingerprint(item.path)) throw new Error('Plugin source changed. Restart Xerxes to load the updated module.')
        const registry = action === 'enable' && !item.registry.pluginNames.length ? await this.readModule(item.path) : item.registry
        if (action === 'enable' && registry.pluginNames.slice().sort().join('\0') !== item.names.slice().sort().join('\0')) throw new Error('Plugin registration names changed; restore the original module')
        const fingerprint = action === 'enable' ? await this.fingerprint(item.path) : item.fingerprint
        this.modules = this.modules.map(entry => entry === item ? { path: entry.path, names: entry.names, registry, ...(fingerprint ? { fingerprint } : {}), enabled: action === 'enable' } : entry)
      }
      try {
        this.validate(this.modules)
        await mkdir(dirname(this.manifest), { recursive: true })
        const temporary = `${this.manifest}.${crypto.randomUUID()}.tmp`
        try {
          await writeFile(temporary, JSON.stringify(this.modules.map(({ path, enabled, names }) => ({ path, enabled, names })), null, 2), { mode: 0o600 })
          await rename(temporary, this.manifest)
        } finally { await rm(temporary, { force: true }) }
      } catch (error) { this.modules = previous; throw error }
      return `${action === 'install' ? 'Installed and enabled' : action === 'enable' ? 'Enabled' : 'Disabled'} ${value}. Changes apply to subsequent turns. Local module files remain at their original path. Restart Xerxes after editing module code.`
    }, { waitMs: 30_000, staleMs: 60_000, label: 'plugin settings' }))
    this.queue = operation.catch(() => undefined)
    return operation
  }
  registerTools(tools: ToolRegistry): void {
    for (const item of this.modules) {
      if (!item.enabled || item.error) continue
      for (const [name, callback] of Object.entries(item.registry.getAllTools())) {
        const toolName = `plugin_${name}`
        if (!/^[a-zA-Z0-9_-]{1,64}$/.test(toolName)) throw new Error(`Invalid plugin tool name: ${name}`)
        tools.register({ type: 'function', function: { name: toolName, description: `Native plugin tool ${name}. Pass the positional arguments expected by the plugin.`, parameters: { type: 'object', properties: { args: { type: 'array', items: {} } }, required: ['args'], additionalProperties: false } } }, async inputs => {
          if (!(await this.readManifest()).some(current => current.path === item.path && current.enabled)) throw new Error('Plugin was disabled')
          const args = inputs.args
          if (!Array.isArray(args)) throw new Error('args must be an array')
          return callback(...args)
        })
      }
    }
  }
}
