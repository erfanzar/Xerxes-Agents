// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { dirname, extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { xerxesHome } from '../daemon/paths.js'
import { AgentSpecError } from '../core/errors.js'
import { loadAgentSpec, loadAgentSpecData, type AgentSpecLoadOptions, type ResolvedAgentSpec } from './agentSpec.js'
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
  readonly systemPrompt: string
  readonly tools: readonly string[]
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

const BUILTIN_AGENT_DIRECTORY = resolve(dirname(fileURLToPath(import.meta.url)), 'default')
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
    } catch (error) {
      recordLoadError(candidate, error)
    }
  }
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

/** Parse a Markdown definition with optional YAML frontmatter. */
export function parseAgentMarkdown(path: string, source = 'user'): AgentDefinition {
  const content = readFileSync(path, 'utf8')
  const name = basenameWithoutExtension(path)
  const frontmatter = markdownFrontmatter(content)
  const fields = frontmatter ? yamlMap(parseYaml(frontmatter.fields, path), `${path} frontmatter`) : {}
  const tools = stringList(fields.tools, `${path} frontmatter.tools`)
  const maxDepth = fields.max_depth === undefined ? 5 : integer(fields.max_depth, `${path} frontmatter.max_depth`)
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
    isolation: stringValue(fields.isolation),
  })
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
    } catch (error) {
      recordLoadError(path, error)
    }
  }
  for (const path of agentFiles(directory, '.md')) {
    try {
      const definition = parseAgentMarkdown(path, source)
      definitions.set(definition.name, definition)
    } catch (error) {
      recordLoadError(path, error)
    }
  }
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
  })
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
  return readdirSync(directory, { withFileTypes: true })
    .filter(entry => entry.isFile() && extname(entry.name) === extension)
    .map(entry => join(directory, entry.name))
    .sort()
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
  if (!Number.isInteger(numeric)) {
    throw new AgentSpecError(`${source} must be an integer`)
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
  const definitions: AgentDefinition[] = [
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
      allowedTools: ['ReadFile', 'GlobTool', 'GrepTool', 'ListDir', 'DuckDuckGoSearch'],
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
      allowedTools: ['ReadFile', 'GlobTool', 'GrepTool', 'ListDir'],
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
