// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { AgentSpecError } from '../core/errors.js'
import { parseYaml, yamlMap, type YamlMap, type YamlValue } from './yaml.js'

export const DEFAULT_AGENT_SPEC_VERSION = '1'
export const SUPPORTED_AGENT_SPEC_VERSIONS = [DEFAULT_AGENT_SPEC_VERSION] as const

/**
 * Isolation modes a resolved spec may declare.
 *
 * `''` follows the runtime default (children share the caller's filesystem),
 * `'shared'` explicitly opts into shared-filesystem writes, and `'worktree'`
 * gives each spawned child its own git worktree. Anything else used to slip
 * through parsing and silently downgrade children to shared-FS writes because
 * consumers treat every non-`worktree` value as "no isolation".
 */
export const AGENT_SPEC_ISOLATION_MODES = ['', 'shared', 'worktree'] as const

const INHERIT = Symbol('agent-spec-inherit')
type Inherit = typeof INHERIT

/**
 * Non-fatal notes collected while parsing the most recently loaded spec
 * (deprecated field spellings and similar "accepted but adjusted" cases).
 *
 * Every top-level load resets the sink, so drained notes always describe that
 * one load including its `extend:` bases. Definition loading drains the notes
 * into the shared load-error channel; direct callers read them via
 * {@link drainAgentSpecDiagnostics}.
 */
let pendingDiagnostics: readonly string[] = []

/** Drain and reset notes collected during the most recent spec load. */
export function drainAgentSpecDiagnostics(): readonly string[] {
  const drained = pendingDiagnostics
  pendingDiagnostics = []
  return drained
}

function noteAgentSpecDiagnostic(message: string): void {
  pendingDiagnostics = [...pendingDiagnostics, message]
}

/**
 * Return the spec-validation error for an isolation value, or undefined when
 * the value is one of {@link AGENT_SPEC_ISOLATION_MODES}. Shared by every spec
 * surface (YAML sections, Markdown frontmatter) so rejection stays identical.
 */
export function agentSpecIsolationError(value: string, source: string): string | undefined {
  if ((AGENT_SPEC_ISOLATION_MODES as readonly string[]).includes(value)) {
    return undefined
  }
  return `${source} must be one of: ${AGENT_SPEC_ISOLATION_MODES
    .filter(mode => mode !== '')
    .join(', ')}, or an empty string for no isolation; got '${value}'`
}

/** Recognized top-level keys of an agent-spec document. */
const RECOGNIZED_AGENT_SPEC_KEYS: ReadonlySet<string> = new Set(['agent', 'version'])

/**
 * Recognized fields inside an agent spec's `agent:` section.
 *
 * `description` is accepted as a deprecated alias of `when_to_use` so specs
 * written in the Markdown definition's spelling still resolve; when both are
 * present `when_to_use` wins and a note is emitted.
 */
const RECOGNIZED_AGENT_SECTION_FIELDS: ReadonlySet<string> = new Set([
  'allowed_tools',
  'description',
  'exclude_tools',
  'extend',
  'isolation',
  'max_depth',
  'model',
  'name',
  'subagents',
  'system_prompt',
  'system_prompt_args',
  'system_prompt_path',
  'tools',
  'when_to_use',
])

/** Recognized fields inside one named entry of a spec's `subagents:` mapping. */
const RECOGNIZED_SUBAGENT_ENTRY_FIELDS: ReadonlySet<string> = new Set(['description', 'path'])

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
  pendingDiagnostics = []
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
    // Infinity marks "unset": consumers must treat it as absent and fall back
    // to (never widen) the manager-level depth ceiling.
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
  pendingDiagnostics = []
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
  rejectUnknownKeys(data, path, RECOGNIZED_AGENT_SPEC_KEYS, 'section')
  const version = String(data.version ?? DEFAULT_AGENT_SPEC_VERSION)
  if (!SUPPORTED_AGENT_SPEC_VERSIONS.includes(version as typeof DEFAULT_AGENT_SPEC_VERSION)) {
    throw new AgentSpecError(`Unsupported agent spec version: ${version}`)
  }
  const agentValue = data.agent ?? {}
  const agent = yamlMap(agentValue, `${path}.agent`)
  rejectUnknownKeys(agent, `${path}.agent`, RECOGNIZED_AGENT_SECTION_FIELDS, 'field')
  const scalar = (key: string): string | undefined => {
    if (!Object.hasOwn(agent, key)) return undefined
    const value = agent[key]
    if (value === undefined || value === null) return undefined
    if (typeof value === 'object') {
      throw new AgentSpecError(`${path}.agent.${key} must be a scalar, not a mapping or list`)
    }
    return String(value)
  }
  // `description` is the Markdown definition's spelling of `when_to_use`.
  // Accepting it keeps the two formats interchangeable; when both are present
  // `when_to_use` wins and a deprecation-style note rides the load-error channel.
  const declaresWhenToUse = Object.hasOwn(agent, 'when_to_use')
  const declaresDescription = Object.hasOwn(agent, 'description')
  if (declaresWhenToUse && declaresDescription) {
    noteAgentSpecDiagnostic(
      `${path}.agent: 'description' is deprecated for YAML agent specs; using 'when_to_use' and ignoring 'description'`,
    )
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
    whenToUse: declaresWhenToUse
      ? agent.when_to_use === null ? null : String(agent.when_to_use)
      : declaresDescription
        ? agent.description === null ? null : String(agent.description)
        : INHERIT,
    tools: stringListField(agent, 'tools', path),
    allowedTools: stringListField(agent, 'allowed_tools', path),
    excludeTools: stringListField(agent, 'exclude_tools', path),
    subagents: subagents(agent, path),
    maxDepth: Object.hasOwn(agent, 'max_depth') ? integer(agent.max_depth, `${path}.agent.max_depth`) : INHERIT,
    isolation: isolationMode(scalar('isolation'), path),
  }
  return raw
}

/**
 * Validate a declared `isolation` value, keeping the inherit sentinel for
 * absent/null fields so `extend:` chains keep inheriting the base mode.
 *
 * The loader throws the same AgentSpecError used for version and max_depth
 * violations: definition loading records the failure per file (the existing
 * lastLoadErrors channel in agents/definitions.ts) instead of silently
 * downgrading children to shared-filesystem writes at spawn time.
 */
function isolationMode(value: string | undefined, sourcePath: string): Inherit | string {
  if (value === undefined) {
    return INHERIT
  }
  const error = agentSpecIsolationError(value, `${sourcePath}.agent.isolation`)
  if (error !== undefined) {
    throw new AgentSpecError(error)
  }
  return value
}

/**
 * Reject unrecognized keys so misspelled settings (for example `isolaton`
 * instead of `isolation`) surface as load errors naming the file and field,
 * mirroring core/config.ts's strictness about unknown settings.
 */
function rejectUnknownKeys(map: YamlMap, source: string, recognized: ReadonlySet<string>, kind: 'field' | 'section'): void {
  for (const key of Object.keys(map)) {
    if (recognized.has(key)) {
      continue
    }
    throw new AgentSpecError(
      `${source} contains unknown agent-spec ${kind} '${key}' (recognized: ${[...recognized].sort().join(', ')})`,
    )
  }
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
    const entrySource = `${path}.agent.subagents.${name}`
    if (entry !== null && !Array.isArray(entry) && typeof entry === 'object') {
      const entryMap = entry as YamlMap
      rejectUnknownKeys(entryMap, entrySource, RECOGNIZED_SUBAGENT_ENTRY_FIELDS, 'field')
      // A missing, empty, or mistyped path used to be silently dropped, so the
      // parent only discovered the loss at spawn time; fail the spec instead.
      const declaredPath = entryMap.path
      if (!Object.hasOwn(entryMap, 'path') || typeof declaredPath !== 'string' || !declaredPath.trim()) {
        throw new AgentSpecError(`${entrySource} must declare a non-empty string 'path'`)
      }
      entries[name] = Object.freeze({
        path: resolve(dirname(path), declaredPath),
        description: String(entryMap.description ?? ''),
      })
      continue
    }
    if (typeof entry === 'string') {
      if (!entry.trim()) {
        throw new AgentSpecError(`${entrySource} must declare a non-empty string 'path'`)
      }
      entries[name] = Object.freeze({ path: resolve(dirname(path), entry), description: '' })
      continue
    }
    throw new AgentSpecError(
      `${entrySource} must be a mapping with a 'path' field or a relative path string`,
    )
  }
  return entries
}

function integer(value: YamlValue | undefined, source: string): number {
  const numeric = typeof value === 'number' ? value : Number(value)
  if (!Number.isSafeInteger(numeric) || numeric < 0) {
    throw new AgentSpecError(`${source} must be a non-negative safe integer`)
  }
  return numeric
}

function defaultAgentSpecPath(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), 'default', 'agent.yaml')
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
