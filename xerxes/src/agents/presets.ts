// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  chmodSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { basename, dirname, extname, join, resolve } from 'node:path'

import { AgentSpecError } from '../core/errors.js'
import { xerxesHome } from '../daemon/paths.js'
import {
  BUILTIN_AGENT_DIRECTORY,
  loadAgentDefinitions,
  parseAgentMarkdown,
  type AgentDefinition,
} from './definitions.js'
import { loadAgentSpec } from './agentSpec.js'

const PRESET_ID = /^[a-z0-9][a-z0-9-]*$/
const SETTINGS_VERSION = 1
const METADATA_FILE = 'preset.json'
const COMPOSITION_FILE = 'agent.yaml'

export type AgentPresetTrust = 'project' | 'system' | 'user'

export interface AgentPresetEntry {
  readonly broken?: string
  readonly description: string
  readonly id: string
  readonly isDefault: boolean
  readonly manageable: boolean
  readonly name: string
  readonly path?: string
  readonly trust: AgentPresetTrust
}

export interface AgentPresetReadResult extends AgentPresetEntry {
  readonly content: string
}

export interface AgentPresetRosterOptions {
  readonly builtinDirectory?: string
  readonly home?: string
  readonly projectDirectory?: string
  readonly settingsPath?: string
  readonly userDirectory?: string
}

interface PresetSettings {
  readonly default: string
  readonly version: typeof SETTINGS_VERSION
}

interface PresetMetadata {
  readonly description?: string
  readonly name?: string
  readonly source?: string
  readonly version: typeof SETTINGS_VERSION
}

/**
 * DSH-style live roster over built-in, user, and project agent compositions.
 *
 * Discovery is deliberately uncached: an agent edited on disk appears on the
 * next list/read without restarting the daemon. Authoring is narrower than a
 * general filesystem primitive — it can only create or mutate one directory
 * under the user preset root, and every written composition must pass the same
 * strict loader used by the runtime before the atomic rename commits it.
 */
export class AgentPresetRoster {
  readonly builtinDirectory: string
  readonly projectDirectory: string
  readonly settingsPath: string
  readonly userDirectory: string

  constructor(options: AgentPresetRosterOptions = {}) {
    const home = resolve(options.home ?? xerxesHome())
    this.builtinDirectory = resolve(options.builtinDirectory ?? BUILTIN_AGENT_DIRECTORY)
    this.projectDirectory = resolve(options.projectDirectory ?? process.cwd())
    this.userDirectory = resolve(options.userDirectory ?? join(home, 'agents'))
    this.settingsPath = resolve(options.settingsPath ?? join(home, 'agent-presets.json'))
  }

  get defaultId(): string {
    return this.readSettings().default
  }

  list(cwd = this.projectDirectory): AgentPresetEntry[] {
    const root = resolve(cwd)
    const definitions = loadAgentDefinitions({
      cwd: root,
      userDirectory: this.userDirectory,
      projectDirectory: join(root, '.xerxes', 'agents'),
    })
    const paths = this.definitionPaths(root)
    const generated = this.generatedMetadata()
    const rows = new Map<string, AgentPresetEntry>()
    for (const [id, definition] of definitions) {
      const path = paths.get(id)
      const trust = definitionTrust(definition)
      const metadata = generated.get(id)?.metadata
      rows.set(id, {
        id,
        name: metadata?.name?.trim() || id,
        description: metadata?.description?.trim() || definition.description,
        trust,
        isDefault: id === this.defaultId,
        manageable: trust === 'user' && generated.get(id)?.path === path,
        ...(path ? { path } : {}),
      })
    }
    for (const [id, record] of generated) {
      if (rows.has(id)) continue
      rows.set(id, {
        id,
        name: record.metadata.name?.trim() || id,
        description: record.metadata.description?.trim() || '',
        trust: 'user',
        isDefault: id === this.defaultId,
        manageable: true,
        path: record.path,
        broken: record.error || 'composition is missing or failed to load',
      })
    }
    return [...rows.values()].sort((left, right) => (
      trustRank(left.trust) - trustRank(right.trust) || left.name.localeCompare(right.name)
    ))
  }

  definition(id: string, cwd = this.projectDirectory): AgentDefinition | undefined {
    return loadAgentDefinitions({
      cwd: resolve(cwd),
      userDirectory: this.userDirectory,
      projectDirectory: join(resolve(cwd), '.xerxes', 'agents'),
    }).get(id)
  }

  resolve(id: string, cwd = this.projectDirectory): AgentPresetEntry {
    const wanted = cleanPresetId(id)
    const found = this.list(cwd).find(row => row.id === wanted)
    if (!found) {
      const available = this.list(cwd).map(row => row.id).join(', ')
      throw new AgentSpecError(`Unknown agent preset '${wanted}'. Available presets: ${available}`)
    }
    return found
  }

  read(id: string, cwd = this.projectDirectory): AgentPresetReadResult {
    const preset = this.resolve(id, cwd)
    if (!preset.path || !existsSync(preset.path)) {
      throw new AgentSpecError(`Agent preset '${preset.id}' has no readable composition file`)
    }
    return { ...preset, content: readFileSync(preset.path, 'utf8') }
  }

  setDefault(id: string, cwd = this.projectDirectory): AgentPresetEntry {
    const preset = this.resolve(id, cwd)
    if (preset.broken) throw new AgentSpecError(`Agent preset '${id}' cannot be the default: ${preset.broken}`)
    this.writeSettings({ version: SETTINGS_VERSION, default: preset.id })
    return { ...preset, isDefault: true }
  }

  copy(from: string, id: string, name?: string, cwd = this.projectDirectory): AgentPresetEntry {
    const source = this.resolve(from, cwd)
    if (source.broken) throw new AgentSpecError(`Cannot copy broken agent preset '${source.id}': ${source.broken}`)
    const targetId = cleanPresetId(id)
    if (this.list(cwd).some(row => row.id === targetId)) {
      throw new AgentSpecError(`Agent preset '${targetId}' already exists`)
    }
    const definitions = loadAgentDefinitions({
      cwd,
      userDirectory: this.userDirectory,
      projectDirectory: join(resolve(cwd), '.xerxes', 'agents'),
    })
    const definition = definitions.get(source.id)
    if (!definition) throw new AgentSpecError(`Agent preset '${source.id}' is not loadable`)

    const directory = this.userPresetDirectory(targetId)
    mkdirSync(join(directory, 'subagents'), { recursive: true, mode: 0o700 })
    const emitted = new Set<string>()
    this.writeDefinitionTree(directory, targetId, definition, definitions, emitted, true)
    const metadata: PresetMetadata = {
      version: SETTINGS_VERSION,
      name: name?.trim() || targetId,
      description: definition.description,
      source: source.id,
    }
    writePrivateJson(join(directory, METADATA_FILE), metadata)
    return this.resolve(targetId, cwd)
  }

  write(id: string, content: string, cwd = this.projectDirectory): AgentPresetEntry {
    const preset = this.resolve(id, cwd)
    if (!preset.manageable || preset.trust !== 'user' || !preset.path) {
      throw new AgentSpecError(`Agent preset '${preset.id}' is not user-writable`)
    }
    if (!content.trim()) throw new AgentSpecError('Agent preset composition cannot be empty')
    const temporary = join(dirname(preset.path), `.agent.${process.pid}.${Date.now()}.yaml`)
    writeFileSync(temporary, content, { encoding: 'utf8', mode: 0o600 })
    try {
      const loaded = loadAgentSpec(temporary, { defaultAgentSpecPath: join(this.builtinDirectory, COMPOSITION_FILE) })
      if (loaded.name !== preset.id) {
        throw new AgentSpecError(`Agent preset id is '${preset.id}', but its composition declares '${loaded.name}'`)
      }
      renameSync(temporary, preset.path)
      chmodSync(preset.path, 0o600)
    } catch (error) {
      rmSync(temporary, { force: true })
      throw error
    }
    return this.resolve(preset.id, cwd)
  }

  validate(id: string, cwd = this.projectDirectory): AgentPresetEntry {
    const preset = this.resolve(id, cwd)
    if (preset.broken) throw new AgentSpecError(`Agent preset '${preset.id}' failed validation: ${preset.broken}`)
    if (!preset.path) throw new AgentSpecError(`Agent preset '${preset.id}' has no composition file`)
    const loaded = loadAgentSpec(preset.path, { defaultAgentSpecPath: join(this.builtinDirectory, COMPOSITION_FILE) })
    if (loaded.name !== preset.id) {
      throw new AgentSpecError(`Agent preset id is '${preset.id}', but its composition declares '${loaded.name}'`)
    }
    return preset
  }

  remove(id: string, cwd = this.projectDirectory): void {
    const preset = this.resolve(id, cwd)
    if (!preset.manageable || preset.trust !== 'user' || !preset.path) {
      throw new AgentSpecError(`Agent preset '${preset.id}' is not user-removable`)
    }
    const directory = dirname(preset.path)
    const expected = this.userPresetDirectory(preset.id)
    if (resolve(directory) !== expected) {
      throw new AgentSpecError(`Refusing to remove unmanaged preset path: ${directory}`)
    }
    rmSync(directory, { recursive: true, force: false })
    if (this.defaultId === preset.id) {
      this.writeSettings({ version: SETTINGS_VERSION, default: 'default' })
    }
  }

  private writeDefinitionTree(
    root: string,
    name: string,
    definition: AgentDefinition,
    definitions: ReadonlyMap<string, AgentDefinition>,
    emitted: Set<string>,
    includeChildren: boolean,
  ): void {
    if (emitted.has(name)) return
    emitted.add(name)
    const childRows: Array<{ alias: string; description: string; file: string }> = []
    if (includeChildren) for (const [alias, childSpec] of Object.entries(definition.subagents ?? {})) {
      const child = definitions.get(childSpec.resolvedProfile ?? alias)
      if (!child) continue
      const childName = `${name}-${slug(alias)}`
      const relative = join('subagents', `${slug(alias)}.yaml`)
      childRows.push({ alias, description: childSpec.description, file: `./${relative}` })
      this.writeDefinitionTree(join(root, 'subagents'), childName, child, definitions, emitted, false)
    }
    const output = renderAgentDefinition(name, definition, childRows)
    const target = root.endsWith('subagents')
      ? join(root, `${name.slice(name.lastIndexOf('-') + 1)}.yaml`)
      : join(root, COMPOSITION_FILE)
    mkdirSync(dirname(target), { recursive: true, mode: 0o700 })
    writeFileSync(target, output, { encoding: 'utf8', mode: 0o600 })
  }

  private definitionPaths(cwd: string): Map<string, string> {
    const paths = new Map<string, string>()
    for (const [directory, trust] of [
      [this.builtinDirectory, 'system'],
      [this.userDirectory, 'user'],
      [join(cwd, '.xerxes', 'agents'), 'project'],
    ] as const) {
      for (const path of definitionFiles(directory)) {
        try {
          const definition = extname(path) === '.md'
            ? parseAgentMarkdown(path, trust)
            : definitionFromResolved(path, trust, this.builtinDirectory)
          paths.set(definition.name, path)
        } catch {
          // Broken managed directories are added separately with their error.
        }
      }
    }
    return paths
  }

  private generatedMetadata(): Map<string, { error?: string; metadata: PresetMetadata; path: string }> {
    const result = new Map<string, { error?: string; metadata: PresetMetadata; path: string }>()
    if (!existsSync(this.userDirectory)) return result
    for (const entry of readdirSync(this.userDirectory, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue
      const directory = join(this.userDirectory, entry.name)
      const metadataPath = join(directory, METADATA_FILE)
      if (!existsSync(metadataPath)) continue
      let metadata: PresetMetadata = { version: SETTINGS_VERSION }
      let error: string | undefined
      try {
        const parsed = JSON.parse(readFileSync(metadataPath, 'utf8')) as Record<string, unknown>
        metadata = {
          version: SETTINGS_VERSION,
          ...(typeof parsed.name === 'string' ? { name: parsed.name } : {}),
          ...(typeof parsed.description === 'string' ? { description: parsed.description } : {}),
          ...(typeof parsed.source === 'string' ? { source: parsed.source } : {}),
        }
      } catch (caught) {
        error = `invalid ${METADATA_FILE}: ${errorMessage(caught)}`
      }
      const path = join(directory, COMPOSITION_FILE)
      if (!existsSync(path)) error = `${COMPOSITION_FILE} is missing`
      else {
        try {
          const loaded = loadAgentSpec(path, { defaultAgentSpecPath: join(this.builtinDirectory, COMPOSITION_FILE) })
          if (loaded.name !== entry.name) error = `composition declares '${loaded.name}', expected '${entry.name}'`
        } catch (caught) {
          error = errorMessage(caught)
        }
      }
      result.set(entry.name, { metadata, path, ...(error ? { error } : {}) })
    }
    return result
  }

  private userPresetDirectory(id: string): string {
    return resolve(this.userDirectory, cleanPresetId(id))
  }

  private readSettings(): PresetSettings {
    try {
      const parsed = JSON.parse(readFileSync(this.settingsPath, 'utf8')) as Record<string, unknown>
      if (parsed.version === SETTINGS_VERSION && typeof parsed.default === 'string' && PRESET_ID.test(parsed.default)) {
        return { version: SETTINGS_VERSION, default: parsed.default }
      }
    } catch {
      // Missing or malformed user settings expose the deployment default.
    }
    return { version: SETTINGS_VERSION, default: 'default' }
  }

  private writeSettings(settings: PresetSettings): void {
    mkdirSync(dirname(this.settingsPath), { recursive: true, mode: 0o700 })
    const temporary = `${this.settingsPath}.${process.pid}.${Date.now()}.tmp`
    writePrivateJson(temporary, settings)
    renameSync(temporary, this.settingsPath)
    chmodSync(this.settingsPath, 0o600)
  }
}

function definitionFromResolved(path: string, source: string, builtinDirectory: string): AgentDefinition {
  const spec = loadAgentSpec(path, { defaultAgentSpecPath: join(builtinDirectory, COMPOSITION_FILE) })
  return {
    allowedTools: spec.allowedTools,
    description: spec.whenToUse,
    excludeTools: spec.excludeTools,
    isolation: spec.isolation,
    maxDepth: spec.maxDepth,
    model: spec.model ?? '',
    name: spec.name,
    source,
    systemPrompt: spec.systemPrompt,
    tools: spec.tools,
  }
}

function definitionFiles(directory: string): string[] {
  if (!existsSync(directory)) return []
  const output: string[] = []
  const visit = (current: string): void => {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const path = join(current, entry.name)
      if (entry.isDirectory()) visit(path)
      else if (entry.isFile() && ['.yaml', '.yml', '.md'].includes(extname(entry.name))) output.push(path)
    }
  }
  visit(directory)
  return output.sort()
}

function renderAgentDefinition(
  name: string,
  definition: AgentDefinition,
  children: readonly { alias: string; description: string; file: string }[],
): string {
  const lines = ['version: 1', 'agent:', `  name: ${quoted(name)}`]
  if (definition.description) pushBlock(lines, 'when_to_use', definition.description, 2)
  pushBlock(lines, 'system_prompt', definition.systemPrompt, 2)
  if (definition.model) lines.push(`  model: ${quoted(definition.model)}`)
  if (definition.isolation) lines.push(`  isolation: ${quoted(definition.isolation)}`)
  if (Number.isFinite(definition.maxDepth)) lines.push(`  max_depth: ${definition.maxDepth}`)
  pushList(lines, 'tools', definition.tools, 2)
  if (definition.allowedTools !== null) pushList(lines, 'allowed_tools', definition.allowedTools, 2)
  pushList(lines, 'exclude_tools', definition.excludeTools, 2)
  if (children.length) {
    lines.push('  subagents:')
    for (const child of children) {
      lines.push(`    ${quoted(child.alias)}:`)
      lines.push(`      path: ${quoted(child.file)}`)
      lines.push(`      description: ${quoted(child.description)}`)
    }
  }
  return `${lines.join('\n')}\n`
}

function pushBlock(lines: string[], key: string, value: string, indent: number): void {
  const prefix = ' '.repeat(indent)
  lines.push(`${prefix}${key}: |-`)
  const bodyPrefix = ' '.repeat(indent + 2)
  for (const line of value.replace(/\r\n/g, '\n').split('\n')) lines.push(`${bodyPrefix}${line}`)
}

function pushList(lines: string[], key: string, values: readonly string[], indent: number): void {
  if (!values.length) return
  const prefix = ' '.repeat(indent)
  lines.push(`${prefix}${key}:`)
  for (const value of values) lines.push(`${prefix}  - ${quoted(value)}`)
}

function quoted(value: string): string {
  return JSON.stringify(value)
}

function cleanPresetId(value: string): string {
  const clean = value.trim()
  if (!PRESET_ID.test(clean)) {
    throw new AgentSpecError('Agent preset id must use lowercase letters, digits, and hyphens, starting with a letter or digit')
  }
  return clean
}

function slug(value: string): string {
  const clean = value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
  return clean || 'agent'
}

function definitionTrust(definition: AgentDefinition): AgentPresetTrust {
  return definition.source === 'built-in' ? 'system' : definition.source === 'project' ? 'project' : 'user'
}

function trustRank(trust: AgentPresetTrust): number {
  return trust === 'system' ? 0 : trust === 'user' ? 1 : 2
}

function writePrivateJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 })
  chmodSync(path, 0o600)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
