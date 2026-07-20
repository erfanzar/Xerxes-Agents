// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { AgentSpecError } from '../core/errors.js'
import { parseYaml, yamlMap, type YamlMap, type YamlValue } from './yaml.js'

export const DEFAULT_AGENT_SPEC_VERSION = '1'
export const SUPPORTED_AGENT_SPEC_VERSIONS = [DEFAULT_AGENT_SPEC_VERSION] as const

const INHERIT = Symbol('agent-spec-inherit')
type Inherit = typeof INHERIT

export interface SubagentSpec {
  readonly description: string
  readonly path: string
}

export interface ResolvedAgentSpec {
  readonly allowedTools: readonly string[] | null
  readonly excludeTools: readonly string[]
  readonly isolation: string
  readonly maxDepth: number
  readonly model: string | null
  readonly name: string
  readonly source: 'yaml'
  readonly subagents: Readonly<Record<string, SubagentSpec>>
  readonly systemPrompt: string
  readonly tools: readonly string[]
  readonly whenToUse: string
}

export interface AgentSpecLoadOptions {
  /** Path used when a spec declares `extend: default`. */
  readonly defaultAgentSpecPath?: string
}

interface RawAgentSpec {
  readonly allowedTools: Inherit | readonly string[] | null
  readonly excludeTools: Inherit | readonly string[] | null
  readonly extend: string | undefined
  readonly isolation: Inherit | string
  readonly maxDepth: Inherit | number
  readonly model: Inherit | null | string
  readonly name: Inherit | string
  readonly sourcePath: string
  readonly subagents: Inherit | Readonly<Record<string, SubagentSpec>>
  readonly systemPrompt: Inherit | string
  readonly systemPromptArgs: Readonly<Record<string, string>>
  readonly systemPromptPath: Inherit | string
  readonly tools: Inherit | readonly string[] | null
  readonly version: string
  readonly whenToUse: Inherit | null | string
}

/** Load, inherit, and resolve a YAML agent definition. */
export function loadAgentSpec(path: string, options: AgentSpecLoadOptions = {}): ResolvedAgentSpec {
  const absolutePath = resolve(path)
  const raw = loadAgentSpecRecursive(absolutePath, options, new Set())
  if (raw.name === INHERIT) {
    throw new AgentSpecError(`Agent name is required: ${absolutePath}`)
  }
  if (raw.systemPrompt === INHERIT && raw.systemPromptPath === INHERIT) {
    throw new AgentSpecError(`system_prompt or system_prompt_path is required: ${absolutePath}`)
  }

  const systemPrompt = raw.systemPrompt === INHERIT
    ? resolveSystemPrompt(raw.systemPromptPath === INHERIT ? undefined : raw.systemPromptPath, raw.systemPromptArgs)
    : raw.systemPrompt
  return Object.freeze({
    name: raw.name,
    systemPrompt,
    model: raw.model === INHERIT ? null : raw.model,
    whenToUse: raw.whenToUse === INHERIT || raw.whenToUse === null ? '' : raw.whenToUse,
    tools: Object.freeze(raw.tools === INHERIT || raw.tools === null ? [] : [...raw.tools]),
    allowedTools: raw.allowedTools === INHERIT || raw.allowedTools === null ? null : Object.freeze([...raw.allowedTools]),
    excludeTools: Object.freeze(raw.excludeTools === INHERIT || raw.excludeTools === null ? [] : [...raw.excludeTools]),
    subagents: Object.freeze({ ...(raw.subagents === INHERIT ? {} : raw.subagents) }),
    maxDepth: raw.maxDepth === INHERIT ? Number.POSITIVE_INFINITY : raw.maxDepth,
    isolation: raw.isolation === INHERIT ? '' : raw.isolation,
    source: 'yaml' as const,
  })
}

/** Resolve an in-memory YAML mapping using `path` as its file-system origin. */
export function loadAgentSpecData(
  path: string,
  data: YamlValue,
  options: AgentSpecLoadOptions = {},
): ResolvedAgentSpec {
  const absolutePath = resolve(path)
  const raw = loadAgentSpecDataRecursive(absolutePath, data, options, new Set([absolutePath]))
  if (raw.name === INHERIT) {
    throw new AgentSpecError(`Agent name is required: ${absolutePath}`)
  }
  if (raw.systemPrompt === INHERIT && raw.systemPromptPath === INHERIT) {
    throw new AgentSpecError(`system_prompt or system_prompt_path is required: ${absolutePath}`)
  }
  const systemPrompt = raw.systemPrompt === INHERIT
    ? resolveSystemPrompt(raw.systemPromptPath === INHERIT ? undefined : raw.systemPromptPath, raw.systemPromptArgs)
    : raw.systemPrompt
  return Object.freeze({
    name: raw.name,
    systemPrompt,
    model: raw.model === INHERIT ? null : raw.model,
    whenToUse: raw.whenToUse === INHERIT || raw.whenToUse === null ? '' : raw.whenToUse,
    tools: Object.freeze(raw.tools === INHERIT || raw.tools === null ? [] : [...raw.tools]),
    allowedTools: raw.allowedTools === INHERIT || raw.allowedTools === null ? null : Object.freeze([...raw.allowedTools]),
    excludeTools: Object.freeze(raw.excludeTools === INHERIT || raw.excludeTools === null ? [] : [...raw.excludeTools]),
    subagents: Object.freeze({ ...(raw.subagents === INHERIT ? {} : raw.subagents) }),
    maxDepth: raw.maxDepth === INHERIT ? Number.POSITIVE_INFINITY : raw.maxDepth,
    isolation: raw.isolation === INHERIT ? '' : raw.isolation,
    source: 'yaml' as const,
  })
}

/** Resolve a prompt file and apply `${name}` and `${name:-fallback}` substitutions. */
export function resolveSystemPrompt(path: string | undefined, args: Readonly<Record<string, string>>): string {
  if (!path) {
    return ''
  }
  if (!existsSync(path)) {
    throw new AgentSpecError(`System prompt file not found: ${path}`)
  }
  return readFileSync(path, 'utf8').replace(/\$\{([^}]+)\}/g, (match, expression: string) => {
    const fallbackIndex = expression.indexOf(':-')
    if (fallbackIndex >= 0) {
      const name = expression.slice(0, fallbackIndex)
      const fallback = expression.slice(fallbackIndex + 2)
      return args[name] ?? fallback
    }
    return args[expression] ?? match
  })
}

function loadAgentSpecRecursive(
  path: string,
  options: AgentSpecLoadOptions,
  ancestors: ReadonlySet<string>,
): RawAgentSpec {
  if (ancestors.has(path)) {
    throw new AgentSpecError(`Circular agent spec inheritance: ${[...ancestors, path].join(' -> ')}`)
  }
  if (!existsSync(path)) {
    throw new AgentSpecError(`Agent spec file not found: ${path}`)
  }
  let data: YamlValue
  try {
    data = parseYaml(readFileSync(path, 'utf8'), path)
  } catch (error) {
    if (error instanceof AgentSpecError) {
      throw error
    }
    throw new AgentSpecError(`Invalid YAML in agent spec file ${path}: ${errorMessage(error)}`)
  }
  return loadAgentSpecDataRecursive(path, data, options, new Set([...ancestors, path]))
}

function loadAgentSpecDataRecursive(
  path: string,
  data: YamlValue,
  options: AgentSpecLoadOptions,
  ancestors: ReadonlySet<string>,
): RawAgentSpec {
  const spec = parseRawAgentSpec(path, data)
  if (!spec.extend) {
    return spec
  }
  const basePath = spec.extend === 'default'
    ? resolve(options.defaultAgentSpecPath ?? defaultAgentSpecPath())
    : resolve(dirname(path), spec.extend)
  const base = loadAgentSpecRecursive(basePath, options, ancestors)
  return mergeRawAgentSpecs(base, spec)
}

function parseRawAgentSpec(path: string, input: YamlValue): RawAgentSpec {
  const data = yamlMap(input, path)
  const version = String(data.version ?? DEFAULT_AGENT_SPEC_VERSION)
  if (!SUPPORTED_AGENT_SPEC_VERSIONS.includes(version as typeof DEFAULT_AGENT_SPEC_VERSION)) {
    throw new AgentSpecError(`Unsupported agent spec version: ${version}`)
  }
  const agentValue = data.agent ?? {}
  const agent = yamlMap(agentValue, `${path}.agent`)
  const scalar = (key: string): string | undefined => {
    if (!Object.hasOwn(agent, key)) return undefined
    const value = agent[key]
    if (value === undefined || value === null) return undefined
    if (typeof value === 'object') {
      throw new AgentSpecError(`${path}.agent.${key} must be a scalar, not a mapping or list`)
    }
    return String(value)
  }
  const raw: RawAgentSpec = {
    version,
    sourcePath: path,
    extend: scalar('extend'),
    name: scalar('name') ?? INHERIT,
    systemPrompt: scalar('system_prompt') ?? INHERIT,
    systemPromptPath: Object.hasOwn(agent, 'system_prompt_path')
      ? resolve(dirname(path), String(agent.system_prompt_path))
      : INHERIT,
    systemPromptArgs: stringMap(agent.system_prompt_args, `${path}.agent.system_prompt_args`),
    model: Object.hasOwn(agent, 'model') ? agent.model === null ? null : String(agent.model) : INHERIT,
    whenToUse: Object.hasOwn(agent, 'when_to_use') ? agent.when_to_use === null ? null : String(agent.when_to_use) : INHERIT,
    tools: stringListField(agent, 'tools', path),
    allowedTools: stringListField(agent, 'allowed_tools', path),
    excludeTools: stringListField(agent, 'exclude_tools', path),
    subagents: subagents(agent, path),
    maxDepth: Object.hasOwn(agent, 'max_depth') ? integer(agent.max_depth, `${path}.agent.max_depth`) : INHERIT,
    isolation: scalar('isolation') ?? INHERIT,
  }
  return raw
}

function mergeRawAgentSpecs(base: RawAgentSpec, child: RawAgentSpec): RawAgentSpec {
  return {
    version: child.version,
    sourcePath: child.sourcePath,
    extend: undefined,
    name: inherited(base.name, child.name),
    systemPrompt: inherited(base.systemPrompt, child.systemPrompt),
    systemPromptPath: inherited(base.systemPromptPath, child.systemPromptPath),
    systemPromptArgs: { ...base.systemPromptArgs, ...child.systemPromptArgs },
    model: inherited(base.model, child.model),
    whenToUse: inherited(base.whenToUse, child.whenToUse),
    tools: inherited(base.tools, child.tools),
    allowedTools: inherited(base.allowedTools, child.allowedTools),
    excludeTools: inherited(base.excludeTools, child.excludeTools),
    subagents: mergeSubagents(base.subagents, child.subagents),
    maxDepth: inherited(base.maxDepth, child.maxDepth),
    isolation: inherited(base.isolation, child.isolation),
  }
}

function inherited<T>(base: T, child: T): T {
  return child === INHERIT ? base : child
}

function mergeSubagents(
  base: RawAgentSpec['subagents'],
  child: RawAgentSpec['subagents'],
): RawAgentSpec['subagents'] {
  if (child === INHERIT) {
    return base
  }
  return { ...(base === INHERIT ? {} : base), ...child }
}

function stringMap(value: YamlValue | undefined, source: string): Readonly<Record<string, string>> {
  if (value === undefined || value === null) {
    return {}
  }
  const map = yamlMap(value, source)
  return Object.fromEntries(Object.entries(map).map(([key, item]) => [key, String(item)]))
}

function stringListField(
  agent: YamlMap,
  field: string,
  source: string,
): Inherit | readonly string[] | null {
  if (!Object.hasOwn(agent, field)) {
    return INHERIT
  }
  const value = agent[field]
  if (value === null) {
    return null
  }
  if (Array.isArray(value)) {
    return value.map(item => String(item))
  }
  if (typeof value === 'object') {
    throw new AgentSpecError(`${source}.agent.${field} must be a list, scalar, or null`)
  }
  return [String(value)]
}

function subagents(agent: YamlMap, path: string): Inherit | Readonly<Record<string, SubagentSpec>> {
  const rawSubagents = agent.subagents
  if (!Object.hasOwn(agent, 'subagents') || rawSubagents === undefined || rawSubagents === null) {
    return INHERIT
  }
  const value = yamlMap(rawSubagents, `${path}.agent.subagents`)
  const entries: Record<string, SubagentSpec> = {}
  for (const [name, entry] of Object.entries(value)) {
    const entryMap = entry !== null && !Array.isArray(entry) && typeof entry === 'object' ? entry as YamlMap : undefined
    const relativePath = entryMap ? String(entryMap.path ?? '') : String(entry)
    if (!relativePath) {
      continue
    }
    entries[name] = Object.freeze({
      path: resolve(dirname(path), relativePath),
      description: entryMap ? String(entryMap.description ?? '') : '',
    })
  }
  return entries
}

function integer(value: YamlValue | undefined, source: string): number {
  const numeric = typeof value === 'number' ? value : Number(value)
  if (!Number.isInteger(numeric)) {
    throw new AgentSpecError(`${source} must be an integer`)
  }
  return numeric
}

function defaultAgentSpecPath(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), 'default', 'agent.yaml')
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
