// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomBytes } from 'node:crypto'
import { lstat, mkdir, open, readdir, readFile, realpath, rename, unlink } from 'node:fs/promises'
import { basename, dirname, join, relative, resolve, sep } from 'node:path'

import type { ToolRegistry } from '../executors/toolRegistry.js'
import {
  loadPluginModule,
  moduleInventory,
  publishPluginTool,
  type ManagedPluginInventoryEntry,
  type PluginModuleLoadOptions,
  type PluginModuleRuntime,
  type PluginToolServices,
} from './pluginRuntime.js'
import { normalizePluginPermissions, PluginRegistry } from './plugins.js'

const PROJECT_PLUGIN_NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/
const ENTRY_NAMES = ['index.ts', 'index.js', 'index.mjs']
const HASH_SKIP = new Set(['.git', 'node_modules'])
const MAX_HASHED_FILES = 2_000

/** One `<workspace>/.xerxes/plugins/<name>/` folder and what became of it. */
interface ProjectPlugin {
  readonly name: string
  readonly folder: string
  readonly entry: string
  readonly hash: string
  readonly trusted: boolean
  registry: PluginRegistry
  runtime?: PluginModuleRuntime
  error?: string
}

interface WorkspaceState {
  readonly root: string
  readonly plugins: ProjectPlugin[]
}

export interface ProjectPluginsOptions {
  /** Trusted content hashes, keyed by absolute plugin folder. Mode 0600. */
  readonly trustFile: string
  readonly services: PluginToolServices
  readonly load?: PluginModuleLoadOptions
  /** Tool names already owned by installed plugins; a project plugin may not shadow them. */
  readonly reservedTools?: () => ReadonlySet<string>
}

/**
 * Plugins that live inside a workspace. They load only for sessions in that
 * workspace and only when the folder's exact content hash was trusted with
 * `/plugins trust <name>` — the same rule as workspace skills. Editing any file
 * in the folder withdraws trust until it is trusted again.
 */
export class ProjectPlugins {
  private readonly workspaces = new Map<string, WorkspaceState>()
  private readonly loading = new Map<string, Promise<WorkspaceState>>()
  private readonly registries = new Map<string, WeakRef<ToolRegistry>[]>()
  private readonly published = new WeakMap<ToolRegistry, readonly string[]>()

  constructor(private readonly options: ProjectPluginsOptions) {}

  /** The folder project plugins are discovered in. */
  static directory(workspace: string): string {
    return join(resolve(workspace), '.xerxes', 'plugins')
  }

  /**
   * Discover the workspace's plugin folders, load the trusted ones in their own
   * processes and republish their tools. Folders whose content is unchanged
   * keep their running process.
   */
  refresh(workspace: string): Promise<WorkspaceState> {
    const root = resolve(workspace)
    const pending = this.loading.get(root)
    const next = (pending ?? Promise.resolve()).then(() => this.scan(root))
    const tracked = next.finally(() => { if (this.loading.get(root) === tracked) this.loading.delete(root) })
    this.loading.set(root, tracked)
    return tracked
  }

  private async scan(root: string): Promise<WorkspaceState> {
    const previous = this.workspaces.get(root)
    const trust = await this.readTrust()
    const plugins: ProjectPlugin[] = []
    const directory = ProjectPlugins.directory(root)
    const entries = await readdir(directory, { withFileTypes: true }).catch(() => [])
    for (const entry of entries.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0)) {
      if (!entry.isDirectory() || !PROJECT_PLUGIN_NAME.test(entry.name)) continue
      const folder = join(directory, entry.name)
      const file = await entryFile(folder)
      if (!file) continue
      let hash: string
      try { hash = await hashFolder(folder) } catch (error) {
        plugins.push({ name: entry.name, folder, entry: file, hash: '', trusted: false, registry: new PluginRegistry(), error: message(error) })
        continue
      }
      const trusted = trust[folder] === hash
      const old = previous?.plugins.find(plugin => plugin.folder === folder)
      if (old && old.hash === hash && old.trusted === trusted && !old.error) { plugins.push(old); continue }
      const plugin: ProjectPlugin = { name: entry.name, folder, entry: file, hash, trusted, registry: new PluginRegistry() }
      if (trusted) {
        try {
          const loaded = await loadPluginModule(file, this.options.load ?? {})
          plugin.registry = loaded.registry
          plugin.runtime = loaded.runtime
          const reserved = this.options.reservedTools?.() ?? new Set<string>()
          const taken = [...plugins.flatMap(other => Object.keys(other.registry.getAllTools())), ...reserved]
          const clash = Object.keys(loaded.registry.getAllTools()).find(name => taken.includes(name))
          if (clash) throw new Error(`tool ${clash} is already provided by another plugin`)
          if (Object.keys(loaded.registry.getAllCommands()).length) {
            // Slash commands are daemon-wide; a workspace must not add them.
            throw new Error('project plugins may register tools only; slash commands need an installed plugin')
          }
        } catch (error) {
          plugin.runtime?.terminate()
          delete plugin.runtime
          plugin.registry = new PluginRegistry()
          plugin.error = message(error)
        }
      }
      plugins.push(plugin)
    }
    const kept = new Set(plugins.flatMap(plugin => plugin.runtime ? [plugin.runtime] : []))
    for (const old of previous?.plugins ?? []) if (old.runtime && !kept.has(old.runtime)) old.runtime.terminate()
    const state = { root, plugins }
    this.workspaces.set(root, state)
    this.publish(root)
    return state
  }

  /** Trust the current content of one project plugin, then load it. */
  async trust(workspace: string, name: string): Promise<string> {
    const root = resolve(workspace)
    if (!PROJECT_PLUGIN_NAME.test(name)) throw new Error('Project plugin names are folder names under .xerxes/plugins')
    const folder = join(ProjectPlugins.directory(root), name)
    const info = await lstat(folder).catch(() => undefined)
    if (!info?.isDirectory()) throw new Error(`No project plugin folder ${folder}`)
    if (!(await entryFile(folder))) throw new Error(`${folder} has no index.ts, index.js or index.mjs`)
    const hash = await hashFolder(folder)
    const trust = await this.readTrust()
    trust[folder] = hash
    await writePrivate(this.options.trustFile, `${JSON.stringify(trust, null, 2)}\n`)
    const state = await this.refresh(root)
    const plugin = state.plugins.find(entry => entry.folder === folder)
    if (plugin?.error) return `Trusted ${name} (content ${hash.slice(0, 12)}), but it failed to load: ${plugin.error}`
    const tools = Object.keys(plugin?.registry.getAllTools() ?? {}).map(tool => `plugin_${tool}`)
    return `Trusted ${name} (content ${hash.slice(0, 12)}) for ${root}. ${tools.length ? `Tools: ${tools.join(', ')}.` : 'It registered no tools.'} Editing any file in ${folder} withdraws trust until you run /plugins trust ${name} again.`
  }

  /** Last known state of a workspace (empty until refresh() has run). */
  inventory(workspace: string): ManagedPluginInventoryEntry[] {
    const state = this.workspaces.get(resolve(workspace))
    if (!state) return []
    return state.plugins.flatMap(plugin => {
      if (!plugin.trusted || plugin.error) {
        return [{
          name: plugin.name, version: '',
          description: plugin.error ? `Failed to load: ${plugin.error}` : `Untrusted project plugin; run /plugins trust ${plugin.name} after reviewing ${plugin.folder}`,
          source: { kind: 'module' as const, path: plugin.entry },
          tools: [], hooks: [], channels: [], providers: [], dependencies: [], commands: [], toolDescriptions: {},
          permissions: normalizePluginPermissions(undefined, plugin.name), enabled: false, module: plugin.entry,
          usage: {}, running: false, scope: 'project' as const, trusted: plugin.trusted,
          ...(plugin.error ? { loadError: plugin.error } : {}),
        }]
      }
      return moduleInventory({
        path: plugin.entry, enabled: true, names: plugin.registry.pluginNames, registry: plugin.registry,
        ...(plugin.runtime ? { runtime: plugin.runtime } : {}),
      }, 'project', this.options.services.usage, true)
    })
  }

  /** Register this workspace's trusted plugin tools into a registry; kept current on later refreshes. */
  registerTools(tools: ToolRegistry, workspace: string): void {
    const root = resolve(workspace)
    const references = this.registries.get(root) ?? []
    if (!references.some(reference => reference.deref() === tools)) references.push(new WeakRef(tools))
    this.registries.set(root, references)
    this.populate(tools, root)
  }

  /** Stop every project plugin process. Calls start them again on demand. */
  close(): void {
    for (const state of this.workspaces.values()) for (const plugin of state.plugins) plugin.runtime?.terminate()
  }

  private publish(root: string): void {
    const references = this.registries.get(root) ?? []
    for (let index = references.length - 1; index >= 0; index -= 1) {
      const tools = references[index]!.deref()
      if (tools) this.populate(tools, root)
      else references.splice(index, 1)
    }
  }

  private populate(tools: ToolRegistry, root: string): void {
    for (const name of this.published.get(tools) ?? []) tools.unregister(name)
    const names: string[] = []
    this.published.set(tools, names)
    for (const plugin of this.workspaces.get(root)?.plugins ?? []) {
      if (!plugin.trusted || plugin.error) continue
      const registry = plugin.registry
      for (const [name, callback] of Object.entries(registry.getAllTools())) {
        if (tools.get(name === '' ? '' : `plugin_${name}`)) continue
        names.push(publishPluginTool(tools, registry, name, callback, async () => {
          const current = this.workspaces.get(root)?.plugins.find(entry => entry.folder === plugin.folder)
          if (!current || !current.trusted || current.error) throw new Error('Project plugin is no longer trusted or loaded')
          const tool = current.registry.getTool(name)
          if (!tool) throw new Error(`Project plugin no longer provides tool ${name}`)
          return { tool, registry: current.registry }
        }, this.options.services, 'project plugin'))
      }
    }
  }

  private async readTrust(): Promise<Record<string, string>> {
    let raw: string
    try { raw = await readFile(this.options.trustFile, 'utf8') } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return {}
      throw error
    }
    const parsed: unknown = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('Project plugin trust file is malformed')
    return Object.fromEntries(Object.entries(parsed as Record<string, unknown>).filter((entry): entry is [string, string] => typeof entry[1] === 'string'))
  }
}

async function entryFile(folder: string): Promise<string | undefined> {
  for (const name of ENTRY_NAMES) {
    const path = join(folder, name)
    const info = await lstat(path).catch(() => undefined)
    if (info?.isFile()) return path
  }
  return undefined
}

/**
 * SHA-256 over every regular file below a plugin folder (relative path and
 * bytes, lexical order), skipping `.git` and `node_modules`. A symlink makes the
 * folder unhashable, so it can never be trusted.
 */
export async function hashFolder(folder: string): Promise<string> {
  const root = await realpath(folder)
  const hash = createHash('sha256')
  let count = 0
  const walk = async (directory: string): Promise<void> => {
    const entries = (await readdir(directory, { withFileTypes: true })).sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0)
    for (const entry of entries) {
      if (HASH_SKIP.has(entry.name)) continue
      const path = join(directory, entry.name)
      if (entry.isSymbolicLink()) throw new Error(`${path} is a symlink; project plugins must not contain symlinks`)
      if (entry.isDirectory()) { await walk(path); continue }
      if (!entry.isFile()) continue
      if (++count > MAX_HASHED_FILES) throw new Error(`more than ${MAX_HASHED_FILES} files under ${folder}`)
      hash.update(relative(root, path).split(sep).join('/')).update('\0').update(await readFile(path)).update('\0')
    }
  }
  await walk(root)
  return hash.digest('hex')
}

async function writePrivate(path: string, data: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 })
  const temporary = join(dirname(path), `.${basename(path)}.${process.pid}.${randomBytes(8).toString('hex')}.tmp`)
  const handle = await open(temporary, 'wx', 0o600)
  try {
    await handle.writeFile(data, 'utf8')
    await handle.close()
    await rename(temporary, path)
  } catch (error) {
    await handle.close().catch(() => undefined)
    await unlink(temporary).catch(() => undefined)
    throw error
  }
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
