// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { dirname, extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { xerxesHome } from '../daemon/paths.js'
import { AgentSpecError } from '../core/errors.js'
import {
  agentSpecIsolationError,
  drainAgentSpecDiagnostics,
  loadAgentSpec,
  loadAgentSpecData,
  type AgentSpecLoadOptions,
  type ResolvedAgentSpec,
  type SubagentSpec,
} from './agentSpec.js'
import { parseYaml, yamlMap, type YamlMap, type YamlValue } from './yaml.js'

export interface AgentDefinition {
  readonly allowedTools: readonly string[] | null
  readonly description: string
  readonly excludeTools: readonly string[]
  readonly isolation: string
  readonly maxDepth: number
  readonly model: string
  readonly name: string
  readonly source: string
  /** Named child profiles declared by this agent's resolved YAML specification. */
  readonly subagents?: Readonly<Record<string, AgentSubagentSpec>>
  readonly systemPrompt: string
  readonly tools: readonly string[]
}

/** A creator-local catalog entry optionally bound to an exact loaded profile. */
export interface AgentSubagentSpec extends SubagentSpec {
  /** Internal definition-map key used when an alias would collide globally. */
  readonly resolvedProfile?: string
}

export interface AgentDefinitionLoadOptions extends AgentSpecLoadOptions {
  /** Definitions used before user and project overrides. */
  readonly builtinDefinitions?: ReadonlyMap<string, AgentDefinition>
  /** Working directory used to discover project-local files. */
  readonly cwd?: string
  /** Project agent directory; defaults to `<cwd>/.xerxes/agents`. */
  readonly projectDirectory?: string
  /** User agent directory; defaults to `$XERXES_HOME/agents`. */
  readonly userDirectory?: string
}

export const BUILTIN_AGENT_DIRECTORY = resolve(dirname(fileURLToPath(import.meta.url)), 'default')
let lastLoadErrors: string[] = []

/** Built-ins available in a source checkout; a hardcoded set survives an asset-less bundle. */
export const BUILTIN_AGENTS: ReadonlyMap<string, AgentDefinition> = loadBuiltinAgentDefinitions()

/** Load YAML built-ins, falling back only if every built-in file fails. */
export function loadBuiltinAgentDefinitions(directory = BUILTIN_AGENT_DIRECTORY): ReadonlyMap<string, AgentDefinition> {
  const definitions = new Map<string, AgentDefinition>()
  if (existsSync(directory)) {
    for (const path of agentFiles(directory, '.yaml')) {
      try {
        const spec = loadAgentSpec(path, { defaultAgentSpecPath: join(directory, 'agent.yaml') })
        definitions.set(spec.name, definitionFromSpec(spec, 'built-in'))
      } catch {
        // A single invalid bundled definition must not hide its healthy siblings.
      }
    }
  }
  return definitions.size ? definitions : hardcodedBuiltinDefinitions()
}

/** Merge built-in, user, and project agents. Later sources override earlier names. */
export function loadAgentDefinitions(options: AgentDefinitionLoadOptions = {}): Map<string, AgentDefinition> {
  lastLoadErrors = []
  const definitions = new Map(options.builtinDefinitions ?? BUILTIN_AGENTS)
  const cwd = resolve(options.cwd ?? process.cwd())
  const userDirectory = resolve(options.userDirectory ?? join(xerxesHome(), 'agents'))
  const projectDirectory = resolve(options.projectDirectory ?? join(cwd, '.xerxes', 'agents'))

  loadDefinitionDirectory(definitions, userDirectory, 'user', options)
  loadDefinitionDirectory(definitions, projectDirectory, 'project', options)
  for (const candidate of projectAgentCandidates(cwd)) {
    try {
      for (const definition of parseProjectAgentFile(candidate, 'project', options)) {
        definitions.set(definition.name, definition)
      }
      recordSpecDiagnostics()
    } catch (error) {
      recordLoadError(candidate, error)
    }
  }
  loadReferencedSubagentDefinitions(definitions, options)
  emitCollectedLoadErrors()
  return definitions
}

export function getAgentDefinition(name: string, options: AgentDefinitionLoadOptions = {}): AgentDefinition | undefined {
  return loadAgentDefinitions(options).get(name)
}

export function listAgentDefinitions(options: AgentDefinitionLoadOptions = {}): AgentDefinition[] {
  return [...loadAgentDefinitions(options).values()].sort((left, right) => left.name.localeCompare(right.name))
}

/** Formatted per-file errors captured by the last definition load. */
export function listAgentDefinitionLoadErrors(): string[] {
  if (!lastLoadErrors.length) {
    loadAgentDefinitions()
  }
  return [...lastLoadErrors]
}

/**
 * Formatted summary of the last definition-load issues, or undefined when the
 * catalog loaded cleanly. Doctor-style surfaces can render this directly; the
 * lines name the file and the validation failure for each skipped spec.
 */
export function formatAgentDefinitionLoadErrors(): string | undefined {
  const errors = listAgentDefinitionLoadErrors()
  return errors.length ? errors.map(error => `- ${error}`).join('\n') : undefined
}

let lastEmittedLoadErrors: string | undefined

/**
 * Emit collected load errors to stderr, at most once per loadAgentDefinitions
 * run.
 *
 * Strict spec rejection used to be invisible: a spec that stopped loading
 * silently vanished from the catalog and `xerxes --agent foo` only reported
 * "Unknown agent". Identical consecutive error sets are suppressed so hot
 * paths that re-resolve the catalog (per-spawn definition lookup) cannot flood
 * stderr with the same warning; any changed set is announced again.
 */
function emitCollectedLoadErrors(): void {
  if (!lastLoadErrors.length) {
    return
  }
  const signature = lastLoadErrors.join('\n')
  if (signature === lastEmittedLoadErrors) {
    return
  }
  lastEmittedLoadErrors = signature
  console.error(
    `[xerxes] ${lastLoadErrors.length} agent definition issue(s); affected specs are skipped until fixed:\n  ${
      lastLoadErrors.join('\n  ')
    }`,
  )
}

/**
 * Append notes collected by the spec parser while loading one file (deprecated
 * spellings such as a YAML `description` next to `when_to_use`) to the shared
 * load-error channel so adjustments ride the same surface as hard failures.
 */
function recordSpecDiagnostics(): void {
  for (const diagnostic of drainAgentSpecDiagnostics()) {
    lastLoadErrors.push(diagnostic)
  }
}

/**
 * Resolve a CLI agent reference to a definition.
 *
 * Exact names from the merged built-in/user/project catalog win first, so
 * `--agent researcher` keeps working even when a same-named file exists. A
 * reference matching no name is treated as a YAML or Markdown agent file path
 * relative to the working directory. Unknown references fail with the list of
 * available names so a typo is self-correcting.
 */
export function resolveAgentDefinition(
  reference: string,
  options: AgentDefinitionLoadOptions = {},
): AgentDefinition {
  const definitions = loadAgentDefinitions(options)
  const named = definitions.get(reference)
  if (named) {
    return named
  }
  const candidate = resolve(resolve(options.cwd ?? process.cwd()), reference)
  if (existsSync(candidate)) {
    const extension = extname(candidate)
    if (extension === '.yaml' || extension === '.yml') {
      const definition = definitionFromSpec(loadAgentSpec(candidate, options), 'cli')
      recordSpecDiagnostics()
      return definition
    }
    if (extension === '.md') {
      const definition = parseAgentMarkdown(candidate, 'cli')
      recordSpecDiagnostics()
      return definition
    }
    throw new AgentSpecError(`Unsupported agent file extension '${extension}': ${candidate}`)
  }
  const available = [...definitions.keys()].sort().join(', ')
  throw new AgentSpecError(`Unknown agent '${reference}'. Available agents: ${available}`)
}

/** Recognized fields in a Markdown definition's YAML frontmatter. */
const RECOGNIZED_MARKDOWN_FRONTMATTER_FIELDS: ReadonlySet<string> = new Set([
  'description',
  'isolation',
  'max_depth',
  'model',
  'tools',
])

/** Parse a Markdown definition with optional YAML frontmatter. */
export function parseAgentMarkdown(path: string, source = 'user'): AgentDefinition {
  const content = readFileSync(path, 'utf8')
  const name = basenameWithoutExtension(path)
  const frontmatter = markdownFrontmatter(content)
  const fields = frontmatter ? yamlMap(parseYaml(frontmatter.fields, path), `${path} frontmatter`) : {}
  // Markdown definitions previously bypassed spec validation entirely, so an
  // `isolation: worktrees` typo silently downgraded children to shared-FS
  // writes and a `depth_limit: 9` typo kept the default depth. Apply the same
  // strictness as the YAML loader: unknown keys and invalid modes fail the file.
  rejectUnknownFrontmatterFields(fields, path)
  const tools = stringList(fields.tools, `${path} frontmatter.tools`)
  const maxDepth = fields.max_depth === undefined ? 5 : integer(fields.max_depth, `${path} frontmatter.max_depth`)
  const declaredIsolation = fields.isolation === undefined || fields.isolation === null
    ? ''
    : String(fields.isolation)
  const isolationError = agentSpecIsolationError(declaredIsolation, `${path} frontmatter.isolation`)
  if (isolationError !== undefined) {
    throw new AgentSpecError(isolationError)
  }
  return freezeDefinition({
    name,
    description: stringValue(fields.description),
    systemPrompt: (frontmatter?.body ?? content).trim(),
    model: stringValue(fields.model),
    tools,
    allowedTools: null,
    excludeTools: [],
    source,
    maxDepth,
    isolation: declaredIsolation,
  })
}

/** Reject unrecognized frontmatter keys so misspelled settings cannot be ignored. */
function rejectUnknownFrontmatterFields(fields: YamlMap, path: string): void {
  for (const key of Object.keys(fields)) {
    if (RECOGNIZED_MARKDOWN_FRONTMATTER_FIELDS.has(key)) {
      continue
    }
    throw new AgentSpecError(
      `${path} frontmatter contains unknown field '${key}' ` +
      `(recognized: ${[...RECOGNIZED_MARKDOWN_FRONTMATTER_FIELDS].sort().join(', ')})`,
    )
  }
}

function loadDefinitionDirectory(
  definitions: Map<string, AgentDefinition>,
  directory: string,
  source: string,
  options: AgentDefinitionLoadOptions,
): void {
  for (const path of agentFiles(directory, '.yaml')) {
    try {
      const definition = definitionFromSpec(loadAgentSpec(path, options), source)
      definitions.set(definition.name, definition)
      recordSpecDiagnostics()
    } catch (error) {
      recordLoadError(path, error)
    }
  }
  for (const path of agentFiles(directory, '.md')) {
    try {
      const definition = parseAgentMarkdown(path, source)
      definitions.set(definition.name, definition)
      recordSpecDiagnostics()
    } catch (error) {
      recordLoadError(path, error)
    }
  }
}

/** Bind every catalog path to the exact profile it declares, even when its alias collides globally. */
function loadReferencedSubagentDefinitions(
  definitions: Map<string, AgentDefinition>,
  options: AgentDefinitionLoadOptions,
): void {
  const queue = [...definitions.entries()]
  const loadedPaths = new Map<string, string>()
  while (queue.length) {
    const entry = queue.shift()
    if (!entry) continue
    const [creatorKey, creator] = entry
    const boundSubagents: Record<string, AgentSubagentSpec> = {}
    for (const [alias, reference] of Object.entries(creator.subagents ?? {})) {
      try {
        const referenceKey = `${alias}\u0000${reference.path}`
        let profileKey = loadedPaths.get(referenceKey)
        if (!profileKey) {
          const loaded = definitionFromSpec(loadAgentSpec(reference.path, options), creator.source)
          recordSpecDiagnostics()
          const definition = loaded.name === alias ? loaded : freezeDefinition({ ...loaded, name: alias })
          const existing = definitions.get(alias)
          // Referenced-only profiles never claim the plain alias: that would make a
          // nested-only agent globally spawnable and let map iteration order decide
          // which creator wins the alias. Reuse the plain alias only when an
          // identical definition already owns it.
          profileKey = existing !== undefined && sameDefinition(existing, definition)
            ? alias
            : catalogDefinitionKey(alias, reference.path)
          if (profileKey !== alias) definitions.set(profileKey, definition)
          loadedPaths.set(referenceKey, profileKey)
          queue.push([profileKey, definition])
        }
        boundSubagents[alias] = Object.freeze({ ...reference, resolvedProfile: profileKey })
      } catch (error) {
        recordLoadError(reference.path, error)
      }
    }
    definitions.set(creatorKey, freezeDefinition({ ...creator, subagents: boundSubagents }))
  }
}

function catalogDefinitionKey(alias: string, path: string): string {
  return `@catalog:${alias}:${path}`
}

function sameDefinition(left: AgentDefinition, right: AgentDefinition): boolean {
  return left.name === right.name
    && left.description === right.description
    && left.systemPrompt === right.systemPrompt
    && left.model === right.model
    && left.isolation === right.isolation
    && left.maxDepth === right.maxDepth
    && JSON.stringify(left.tools) === JSON.stringify(right.tools)
    && JSON.stringify(left.allowedTools) === JSON.stringify(right.allowedTools)
    && JSON.stringify(left.excludeTools) === JSON.stringify(right.excludeTools)
    && JSON.stringify(left.subagents ?? {}) === JSON.stringify(right.subagents ?? {})
}

function parseProjectAgentFile(
  path: string,
  source: string,
  options: AgentDefinitionLoadOptions,
): AgentDefinition[] {
  if (basename(path) !== 'agents.yaml') {
    return [definitionFromSpec(loadAgentSpec(path, options), source)]
  }
  const raw = yamlMap(parseYaml(readFileSync(path, 'utf8'), path), path)
  if (!Object.hasOwn(raw, 'agents')) {
    return [definitionFromSpec(loadAgentSpecData(path, raw, options), source)]
  }
  const entries = yamlMap(raw.agents ?? null, `${path}.agents`)
  const definitions: AgentDefinition[] = []
  for (const [name, body] of Object.entries(entries)) {
    if (body === null || Array.isArray(body) || typeof body !== 'object') {
      throw new AgentSpecError(`agents.${name} must be a mapping`)
    }
    const normalized: YamlMap = {
      version: raw.version ?? '1',
      agent: { name, ...(body as YamlMap) },
    }
    definitions.push(definitionFromSpec(loadAgentSpecData(path, normalized, options), source))
  }
  return definitions
}

function definitionFromSpec(spec: ResolvedAgentSpec, source: string): AgentDefinition {
  return freezeDefinition({
    name: spec.name,
    description: spec.whenToUse,
    systemPrompt: spec.systemPrompt,
    model: spec.model ?? '',
    tools: spec.tools,
    allowedTools: spec.allowedTools,
    excludeTools: spec.excludeTools,
    subagents: spec.subagents,
    source,
    maxDepth: spec.maxDepth,
    isolation: spec.isolation,
  })
}

function freezeDefinition(definition: AgentDefinition): AgentDefinition {
  return Object.freeze({
    ...definition,
    tools: Object.freeze([...definition.tools]),
    allowedTools: definition.allowedTools === null ? null : Object.freeze([...definition.allowedTools]),
    excludeTools: Object.freeze([...definition.excludeTools]),
    subagents: freezeSubagents(definition.subagents),
  })
}

function freezeSubagents(
  subagents: Readonly<Record<string, AgentSubagentSpec>> | undefined,
): Readonly<Record<string, AgentSubagentSpec>> {
  return Object.freeze(Object.fromEntries(
    Object.entries(subagents ?? {}).map(([name, spec]) => [name, Object.freeze({ ...spec })]),
  ))
}

function markdownFrontmatter(content: string): { readonly body: string; readonly fields: string } | undefined {
  if (!content.startsWith('---')) {
    return undefined
  }
  const lines = content.replace(/\r\n/g, '\n').split('\n')
  if (lines[0] !== '---') {
    return undefined
  }
  const closing = lines.findIndex((line, index) => index > 0 && line === '---')
  if (closing < 0) {
    return undefined
  }
  return { fields: lines.slice(1, closing).join('\n'), body: lines.slice(closing + 1).join('\n') }
}

function projectAgentCandidates(cwd: string): string[] {
  return [
    join(cwd, '.kimi', 'agent.yaml'),
    join(cwd, '.kimi', 'agents.yaml'),
    join(cwd, 'agent.yaml'),
    join(cwd, 'agents.yaml'),
  ].filter(path => existsSync(path))
}

function agentFiles(directory: string, extension: '.md' | '.yaml'): string[] {
  if (!existsSync(directory)) {
    return []
  }
  const entries = readdirSync(directory, { withFileTypes: true })
  const direct = entries
    .filter(entry => entry.isFile() && extname(entry.name) === extension)
    .map(entry => join(directory, entry.name))
  // DSH-style authored presets live one directory per preset. Only the root
  // composition is a roster entry; referenced files below `subagents/` are
  // loaded through that composition and never leak into the global picker.
  const nested = extension === '.yaml'
    ? entries
      .filter(entry => entry.isDirectory() && existsSync(join(directory, entry.name, 'agent.yaml')))
      .map(entry => join(directory, entry.name, 'agent.yaml'))
    : []
  return [...direct, ...nested].sort()
}

function recordLoadError(path: string, error: unknown): void {
  const name = error instanceof Error ? error.constructor.name : typeof error
  const message = error instanceof Error ? error.message : String(error)
  lastLoadErrors.push(`${path}: ${name}: ${message}`)
}

function stringValue(value: YamlValue | undefined): string {
  return value === undefined || value === null ? '' : String(value)
}

function stringList(value: YamlValue | undefined, source: string): string[] {
  if (value === undefined || value === null) {
    return []
  }
  if (Array.isArray(value)) {
    return value.map(item => String(item))
  }
  if (typeof value === 'object') {
    throw new AgentSpecError(`${source} must be a list or scalar`)
  }
  return [String(value)]
}

function integer(value: YamlValue, source: string): number {
  const numeric = typeof value === 'number' ? value : Number(value)
  if (!Number.isSafeInteger(numeric) || numeric < 0) {
    throw new AgentSpecError(`${source} must be a non-negative safe integer`)
  }
  return numeric
}

function basename(path: string): string {
  const segments = path.split(/[\\/]/)
  return segments.at(-1) ?? path
}

function basenameWithoutExtension(path: string): string {
  const name = basename(path)
  const extension = extname(name)
  return extension ? name.slice(0, -extension.length) : name
}

function hardcodedBuiltinDefinitions(): ReadonlyMap<string, AgentDefinition> {
  const standardTools = [
    'ReadFile', 'WriteFile', 'FileEditTool', 'GlobTool', 'GrepTool', 'ListDir',
    'exec_command', 'write_stdin', 'list_terminal_sessions', 'close_terminal_session',
    'DuckDuckGoSearch', 'computer_use', 'AgentTool', 'SpawnAgents', 'SendMessageTool',
    'TaskCreateTool', 'TaskGetTool', 'TaskListTool', 'TaskOutputTool', 'TaskStopTool',
    'TaskUpdateTool', 'AwaitAgents', 'CheckAgentMessages', 'PeekAgent', 'ResetAgent',
    'HandoffTool', 'AskUserQuestionTool', 'SetInteractionModeTool', 'get_goal',
    'create_goal', 'update_goal', 'SkillTool', 'TodoWriteTool', 'ToolSearchTool',
  ]
  const definitions: AgentDefinition[] = [
    {
      name: 'default',
      description: 'Full Xerxes coding agent with filesystem, shell, research, planning, goals, and subagents.',
      systemPrompt: 'You are Xerxes, an interactive coding and research agent. Use the capabilities supplied by the host and never claim work you did not perform.',
      model: '',
      tools: standardTools,
      allowedTools: null,
      excludeTools: [],
      source: 'built-in',
      maxDepth: 5,
      isolation: '',
    },
    {
      name: 'creator',
      description: 'DSH-style Creator mode for inspecting, duplicating, authoring, and validating agent presets.',
      systemPrompt: 'You are running in Xerxes Creator mode. Inspect the live roster and tool catalog, duplicate a shipped preset, edit only the user copy, validate it, and explain that running sessions keep their original preset.',
      model: '',
      tools: [...standardTools, 'AgentPresetInspectTool', 'AgentPresetTool', 'CreatorRuntimeTool'],
      allowedTools: null,
      excludeTools: [],
      source: 'built-in',
      maxDepth: 5,
      isolation: '',
    },
    {
      name: 'general-purpose',
      description: 'General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks.',
      systemPrompt: '',
      model: '',
      tools: [],
      allowedTools: null,
      excludeTools: [],
      source: 'built-in',
      maxDepth: 5,
      isolation: '',
    },
    {
      name: 'coder',
      description: 'Specialized coding agent for writing, reading, and modifying code.',
      systemPrompt: 'You are a specialized coding assistant. Focus on clean, idiomatic, minimal changes.',
      model: '',
      tools: [],
      allowedTools: null,
      excludeTools: [],
      source: 'built-in',
      maxDepth: 5,
      isolation: '',
    },
    {
      name: 'reviewer',
      description: 'Code review agent analyzing quality, security, and correctness.',
      systemPrompt: 'You are a code reviewer. Categorize findings as Critical, Warning, or Suggestion.',
      model: '',
      tools: [],
      allowedTools: ['ReadFile', 'GlobTool', 'GrepTool', 'ListDir'],
      excludeTools: [],
      source: 'built-in',
      maxDepth: 5,
      isolation: '',
    },
    {
      name: 'researcher',
      description: 'Research agent for exploring codebases and answering questions.',
      systemPrompt: 'You are a research assistant focused on understanding codebases and providing evidence-based answers.',
      model: '',
      tools: [],
      allowedTools: ['ReadFile', 'GlobTool', 'GrepTool', 'ListDir', 'DuckDuckGoSearch', 'SetInteractionModeTool'],
      excludeTools: [],
      source: 'built-in',
      maxDepth: 5,
      isolation: '',
    },
    {
      name: 'tester',
      description: 'Testing agent that writes and runs tests.',
      systemPrompt: 'You are a testing specialist. Focus on edge cases and fast, readable tests.',
      model: '',
      tools: [],
      allowedTools: null,
      excludeTools: [],
      source: 'built-in',
      maxDepth: 5,
      isolation: '',
    },
    {
      name: 'planner',
      description: 'Planning agent that designs implementation strategies and task breakdowns.',
      systemPrompt: 'You are an expert software architect and planner. Produce structured plans, not code.',
      model: '',
      tools: [],
      // Read-only by design, but a planner still has to be able to record the
      // plan, ask whether it is right, and leave. Without the last three the
      // mode was a dead end: the model produced a plan it could not write
      // anywhere, had no approval tool, and no exit — so it asked in prose and
      // the session stayed in plan mode refusing every write that followed.
      // ToolSearchTool stays so discovery keeps working under deferred loading.
      allowedTools: [
        'AskUserQuestionTool',
        'ExitPlanModeTool',
        'GlobTool',
        'GrepTool',
        'ListDir',
        'ReadFile',
        'SetInteractionModeTool',
        'TodoWriteTool',
        'ToolSearchTool',
      ],
      excludeTools: [],
      source: 'built-in',
      maxDepth: 5,
      isolation: '',
    },
    {
      name: 'objective',
      description: 'Objective runner that iterates until explicit acceptance criteria pass.',
      systemPrompt: 'You are an objective runner. Do not claim completion until verification satisfies the acceptance criteria.',
      model: '',
      tools: [],
      allowedTools: null,
      excludeTools: ['AskUserQuestionTool', 'SkillTool'],
      source: 'built-in',
      maxDepth: 5,
      isolation: '',
    },
    {
      name: 'data-analyst',
      description: 'Data analysis agent for processing and analyzing data.',
      systemPrompt: 'You are a data analysis specialist. Present findings clearly with summaries.',
      model: '',
      tools: [],
      allowedTools: null,
      excludeTools: [],
      source: 'built-in',
      maxDepth: 5,
      isolation: '',
    },
  ]
  return new Map(definitions.map(definition => [definition.name, freezeDefinition(definition)]))
}
