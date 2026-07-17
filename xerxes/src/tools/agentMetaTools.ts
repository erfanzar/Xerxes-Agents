// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readdir, readFile, rename, rm, stat } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { ToolRegistry, type ToolExecutionContext } from '../executors/toolRegistry.js'
import { parseSkillMarkdown, type Skill, type SkillRegistry } from '../extensions/skills.js'
import type { SearchHit, SearchOptions } from '../session/search.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { optionalBoolean, optionalInteger, optionalString, optionalStringArray, requiredString } from './inputs.js'

type Awaitable<T> = Promise<T> | T

export type MixtureMember = (prompt: string) => Awaitable<unknown>

export interface MixtureOfAgentsConfig {
  readonly members: Readonly<Record<string, MixtureMember>>
  readonly synthesizer?: MixtureMember
  readonly voting: boolean
}

export interface ConfigureMixtureOfAgentsOptions {
  readonly synthesizer?: MixtureMember
  readonly voting?: boolean
}

export interface MixtureOfAgentsInvocation {
  readonly members?: readonly string[]
  readonly prompt: string
  readonly synthesise?: boolean
}

let configuredMixture: MixtureOfAgentsConfig = { members: {}, voting: false }

/**
 * Replace the process-local mixture configuration.
 *
 * Hosts that need session-specific mixtures should pass mixture to
 * registerAgentMetaTools instead of relying on this convenience state.
 */
export function configureMixtureOfAgents(
  members: Readonly<Record<string, MixtureMember>> | undefined = undefined,
  options: ConfigureMixtureOfAgentsOptions = {},
): void {
  configuredMixture = {
    members: { ...(members ?? {}) },
    voting: options.voting ?? false,
    ...(options.synthesizer === undefined ? {} : { synthesizer: options.synthesizer }),
  }
}

/** Return an isolated snapshot so callers cannot mutate the active roster. */
export function getMixtureOfAgentsConfig(): MixtureOfAgentsConfig {
  return {
    members: { ...configuredMixture.members },
    voting: configuredMixture.voting,
    ...(configuredMixture.synthesizer === undefined ? {} : { synthesizer: configuredMixture.synthesizer }),
  }
}

/** Fan one prompt out to the configured members and optionally vote or synthesise. */
export async function mixtureOfAgents(
  invocation: MixtureOfAgentsInvocation,
  config: MixtureOfAgentsConfig = getMixtureOfAgentsConfig(),
): Promise<JsonObject> {
  if (Object.keys(config.members).length === 0) {
    return { error: 'no MoA members configured', members: [], answers: {} }
  }

  const names = invocation.members === undefined ? Object.keys(config.members) : [...invocation.members]
  const answers: Record<string, string> = {}
  for (const name of names) {
    const member = config.members[name]
    if (member === undefined) {
      answers[name] = "[unknown member '" + name.replaceAll("'", String.fromCharCode(92) + "'") + "']"
      continue
    }
    try {
      answers[name] = String(await member(invocation.prompt))
    } catch (error) {
      answers[name] = '[error: ' + errorMessage(error) + ']'
    }
  }

  const result: JsonObject = { members: names, answers }
  if (config.voting) {
    const voted = majorityAnswer(answers)
    if (voted !== undefined) result.voted = voted
  }
  if ((invocation.synthesise ?? true) && config.synthesizer !== undefined && Object.keys(answers).length > 0) {
    try {
      const joined = Object.entries(answers).map(([name, answer]) => '[' + name + '] ' + answer).join('\n')
      result.final = String(await config.synthesizer('Combine these answers:\n' + joined))
    } catch (error) {
      result.final_error = errorMessage(error)
    }
  }
  return result
}

export interface SessionSearchRequest {
  readonly agentId?: string
  readonly limit: number
  readonly query: string
  readonly sessionId?: string
}

/** A host-owned search port. It must return its real session-search result. */
export interface SessionSearchPort {
  search(request: SessionSearchRequest): Awaitable<JsonObject>
}

/** Structural adapter for the existing SessionIndex and SessionStore search APIs. */
export interface IndexedSessionSearchPort {
  search(query: string, options?: SearchOptions): Awaitable<readonly SearchHit[]>
}

/** Structural adapter for the existing SearchHistoryTool API. */
export interface HistorySessionSearchPort {
  search(
    query: string,
    options?: { readonly agentId?: string; readonly limit?: number; readonly sessionId?: string },
  ): Awaitable<JsonObject>
}

let configuredSessionSearch: SessionSearchPort | undefined

export function setSessionSearchPort(port: SessionSearchPort | undefined): void {
  configuredSessionSearch = port
}

export function getSessionSearchPort(): SessionSearchPort | undefined {
  return configuredSessionSearch
}

/** Adapt an indexed searcher without copying or fabricating session records. */
export function sessionSearchPortFromIndex(index: IndexedSessionSearchPort): SessionSearchPort {
  return {
    async search(request: SessionSearchRequest): Promise<JsonObject> {
      const options: SearchOptions = {
        k: request.limit,
        ...(request.agentId === undefined ? {} : { agentId: request.agentId }),
        ...(request.sessionId === undefined ? {} : { sessionId: request.sessionId }),
      }
      const hits = await index.search(request.query, options)
      return {
        query: request.query,
        count: hits.length,
        hits: hits.map(sessionHitRecord),
      }
    },
  }
}

/** Adapt the public SearchHistoryTool shape while preserving its output. */
export function sessionSearchPortFromHistory(history: HistorySessionSearchPort): SessionSearchPort {
  return {
    search(request: SessionSearchRequest): Awaitable<JsonObject> {
      return history.search(request.query, {
        limit: request.limit,
        ...(request.agentId === undefined ? {} : { agentId: request.agentId }),
        ...(request.sessionId === undefined ? {} : { sessionId: request.sessionId }),
      })
    },
  }
}

export interface SkillCatalog {
  all(): Awaitable<readonly Skill[]>
  get(name: string): Awaitable<Skill | undefined>
}

export interface SkillManageRequest {
  readonly action: string
  readonly description: string
  readonly instructions: string
  readonly name: string
  readonly tags: readonly string[]
  readonly version: string
}

/** A writable skill source supplied explicitly by the host. */
export interface SkillManagementStore extends SkillCatalog {
  manage(request: SkillManageRequest): Awaitable<JsonObject>
}

export interface SkillBundleStoreOptions {
  /** The host-owned directory containing one subdirectory per skill bundle. */
  readonly directory: string
  /** Optional live registry to refresh after writes. Deletion stays explicit because the registry has no public remove API. */
  readonly registry?: SkillRegistry
}

/**
 * File-backed SKILL.md store for hosts that intentionally expose authoring.
 *
 * It never chooses a home directory itself. The host must provide directory,
 * preventing an LLM call from implicitly writing outside its configured scope.
 */
export class SkillBundleStore implements SkillManagementStore {
  readonly directory: string
  private readonly registry: SkillRegistry | undefined

  constructor(options: SkillBundleStoreOptions) {
    this.directory = resolve(options.directory)
    this.registry = options.registry
  }

  async all(): Promise<readonly Skill[]> {
    const paths = await skillMarkdownFiles(this.directory)
    const skills: Skill[] = []
    for (const path of paths) {
      try {
        skills.push(parseSkillMarkdown(await readFile(path, 'utf8'), path))
      } catch {
        // Discovery should not make a single malformed third-party file hide the rest.
      }
    }
    return skills.sort((left, right) => left.metadata.name.localeCompare(right.metadata.name))
  }

  async get(name: string): Promise<Skill | undefined> {
    if (!isValidSkillName(name)) return undefined
    const path = this.pathFor(name)
    if (!(await pathExists(path))) return undefined
    return parseSkillMarkdown(await readFile(path, 'utf8'), path)
  }

  async manage(request: SkillManageRequest): Promise<JsonObject> {
    assertValidSkillName(request.name)
    const path = this.pathFor(request.name)
    if (request.action === 'delete') {
      if (!(await pathExists(path))) {
        return { ok: false, name: request.name, error: 'not_found' }
      }
      await rm(path)
      const result: JsonObject = { ok: true, name: request.name, deleted: path, action: request.action }
      if (this.registry !== undefined) {
        result.registry_updated = false
        result.registry_error = 'SkillRegistry has no public removal API; rebuild or replace the host registry to evict this skill.'
      }
      return result
    }
    if (request.action !== 'create' && request.action !== 'update') {
      return { ok: false, error: 'unknown action ' + JSON.stringify(request.action) }
    }
    if (!request.instructions) {
      return { ok: false, error: 'instructions required for create/update' }
    }

    const content = skillMarkdown(request)
    await atomicWrite(path, content)
    const result: JsonObject = { ok: true, name: request.name, path, action: request.action }
    if (this.registry !== undefined) {
      try {
        this.registry.register(parseSkillMarkdown(content, path))
        result.registry_updated = true
      } catch (error) {
        result.registry_updated = false
        result.registry_error = errorMessage(error)
      }
    }
    return result
  }

  private pathFor(name: string): string {
    return join(this.directory, name, 'SKILL.md')
  }
}

let configuredSkillRegistry: SkillRegistry | undefined

export function setSkillRegistry(registry: SkillRegistry | undefined): void {
  configuredSkillRegistry = registry
}

export function getSkillRegistry(): SkillRegistry | undefined {
  return configuredSkillRegistry
}

/** Wrap the existing registry in the catalog port used by these tools. */
export function skillCatalogFromRegistry(registry: SkillRegistry): SkillCatalog {
  return {
    all: () => registry.all(),
    get: name => registry.get(name),
  }
}

export interface AgentMetaToolsOptions {
  /** Per-registry mixture configuration; this overrides the process-local convenience configuration. */
  readonly mixture?: MixtureOfAgentsConfig
  /** Session history is host-owned and is only queried when this real port is supplied. */
  readonly sessionSearch?: SessionSearchPort
  /** Read-only active registry for skills_list and skill_view. */
  readonly skillRegistry?: SkillRegistry
  /** Explicit writable/catalog store for skill_manage and, when no registry is supplied, listing and viewing. */
  readonly skillStore?: SkillManagementStore
}

export const MIXTURE_OF_AGENTS_DEFINITION: ToolDefinition = toolDefinition(
  'mixture_of_agents',
  'Fan one prompt out to configured agent members and optionally vote or synthesize the real responses.',
  {
    prompt: { type: 'string' },
    members: { type: 'array', items: { type: 'string' } },
    synthesise: { type: 'boolean', default: true },
  },
  ['prompt'],
)

export const SESSION_SEARCH_DEFINITION: ToolDefinition = toolDefinition(
  'session_search',
  'Search prior session transcripts through a host-provided session-search port.',
  {
    query: { type: 'string' },
    limit: { type: 'integer', minimum: 1, default: 5 },
    agent_id: { type: 'string' },
    session_id: { type: 'string' },
  },
  ['query'],
)

export const SKILLS_LIST_DEFINITION: ToolDefinition = toolDefinition(
  'skills_list',
  'List skills from the configured SkillRegistry or explicit skill store. Search uses transparent lexical matching.',
  { search: { type: 'string' } },
)

export const SKILL_VIEW_DEFINITION: ToolDefinition = toolDefinition(
  'skill_view',
  'Read one skill from the configured SkillRegistry or explicit skill store.',
  { name: { type: 'string' } },
  ['name'],
)

export const AGENT_META_SKILL_MANAGE_DEFINITION: ToolDefinition = toolDefinition(
  'skill_manage',
  'Create, update, or delete a skill only through a host-provided writable skill store.',
  {
    action: { type: 'string', enum: ['create', 'update', 'delete'] },
    name: { type: 'string' },
    instructions: { type: 'string' },
    description: { type: 'string' },
    version: { type: 'string', default: '0.1.0' },
    tags: { type: 'array', items: { type: 'string' } },
  },
  ['action', 'name'],
)

export const AGENT_META_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  MIXTURE_OF_AGENTS_DEFINITION,
  SESSION_SEARCH_DEFINITION,
  SKILLS_LIST_DEFINITION,
  SKILL_VIEW_DEFINITION,
  AGENT_META_SKILL_MANAGE_DEFINITION,
]

/**
 * Register the Python-compatible agent-meta tool names.
 *
 * Do not combine this registration with another skill_manage implementation
 * for the same ToolRegistry agent unless the host intentionally chooses one.
 */
export function registerAgentMetaTools(
  registry: ToolRegistry,
  options: AgentMetaToolsOptions = {},
  agentId = 'default',
): void {
  registry.register(MIXTURE_OF_AGENTS_DEFINITION, inputs => mixtureOfAgentsTool(inputs, options), agentId)
  registry.register(SESSION_SEARCH_DEFINITION, (inputs, context) => sessionSearchTool(inputs, context, options), agentId)
  registry.register(SKILLS_LIST_DEFINITION, inputs => skillsListTool(inputs, options), agentId)
  registry.register(SKILL_VIEW_DEFINITION, inputs => skillViewTool(inputs, options), agentId)
  registry.register(AGENT_META_SKILL_MANAGE_DEFINITION, inputs => skillManageTool(inputs, options), agentId)
}

export async function mixtureOfAgentsTool(inputs: JsonObject, options: AgentMetaToolsOptions = {}): Promise<JsonObject> {
  const configured = options.mixture ?? getMixtureOfAgentsConfig()
  return mixtureOfAgents({
    prompt: requiredString(inputs, 'prompt'),
    members: optionalStringArray(inputs, 'members'),
    synthesise: optionalBoolean(inputs, 'synthesise', true),
  }, configured)
}

export async function sessionSearchTool(
  inputs: JsonObject,
  _context: ToolExecutionContext,
  options: AgentMetaToolsOptions = {},
): Promise<JsonObject> {
  const port = options.sessionSearch ?? getSessionSearchPort()
  if (port === undefined) {
    return { error: 'no session searcher configured', hits: [] }
  }
  const limit = positiveLimit(optionalInteger(inputs, 'limit', 5))
  const agentId = optionalString(inputs, 'agent_id')
  const sessionId = optionalString(inputs, 'session_id')
  return port.search({
    query: requiredString(inputs, 'query'),
    limit,
    ...(agentId === undefined ? {} : { agentId }),
    ...(sessionId === undefined ? {} : { sessionId }),
  })
}

export async function skillsListTool(inputs: JsonObject, options: AgentMetaToolsOptions = {}): Promise<JsonObject> {
  const catalog = resolveSkillCatalog(options)
  if (catalog === undefined) {
    return { error: 'no skill registry configured', skills: [] }
  }
  const search = optionalString(inputs, 'search')
  const skills = await catalog.all()
  if (search === undefined || !search.trim()) {
    return { count: skills.length, skills: skills.map(skill => skillMetadataRecord(skill)) }
  }
  const matches = skills
    .map(skill => ({ skill, score: lexicalSkillScore(skill, search) }))
    .filter(match => match.score > 0)
    .sort((left, right) => right.score - left.score || left.skill.metadata.name.localeCompare(right.skill.metadata.name))
    .slice(0, 20)
  return {
    count: matches.length,
    skills: matches.map(match => skillMetadataRecord(match.skill, match.score)),
    query: search,
    match_strategy: 'lexical',
  }
}

export async function skillViewTool(inputs: JsonObject, options: AgentMetaToolsOptions = {}): Promise<JsonObject> {
  const name = requiredString(inputs, 'name')
  const catalog = resolveSkillCatalog(options)
  if (catalog === undefined) {
    return { error: 'no skill registry configured', name }
  }
  const exact = await catalog.get(name)
  if (exact !== undefined) return skillViewRecord(exact)

  // A miss must never return another skill's full body; report not-found plus candidate names.
  const candidates = (await catalog.all())
    .map(skill => ({ skill, score: lexicalSkillScore(skill, name) }))
    .filter(match => match.score > 0)
    .sort((left, right) => right.score - left.score || left.skill.metadata.name.localeCompare(right.skill.metadata.name))
    .slice(0, 5)
    .map(match => match.skill.metadata.name)
  return { candidates, error: 'not_found', name }
}

export async function skillManageTool(inputs: JsonObject, options: AgentMetaToolsOptions = {}): Promise<JsonObject> {
  if (options.skillStore === undefined) {
    return {
      ok: false,
      error: 'no writable skill store configured; the host must provide SkillManagementStore explicitly',
    }
  }
  const instructions = optionalString(inputs, 'instructions')
  const description = optionalString(inputs, 'description')
  const version = optionalString(inputs, 'version')
  return options.skillStore.manage({
    action: requiredString(inputs, 'action'),
    name: requiredString(inputs, 'name'),
    instructions: instructions ?? '',
    description: description ?? '',
    version: version ?? '0.1.0',
    tags: optionalStringArray(inputs, 'tags'),
  })
}

function resolveSkillCatalog(options: AgentMetaToolsOptions): SkillCatalog | undefined {
  if (options.skillRegistry !== undefined) return skillCatalogFromRegistry(options.skillRegistry)
  if (options.skillStore !== undefined) return options.skillStore
  const registry = getSkillRegistry()
  return registry === undefined ? undefined : skillCatalogFromRegistry(registry)
}

function majorityAnswer(answers: Readonly<Record<string, string>>): string | undefined {
  const counts = new Map<string, number>()
  for (const answer of Object.values(answers)) {
    const normalized = answer.split(/\s+/).join(' ').trim().toLocaleLowerCase()
    counts.set(normalized, (counts.get(normalized) ?? 0) + 1)
  }
  let voted: string | undefined
  let largest = 0
  for (const [answer, count] of counts) {
    if (count > largest) {
      voted = answer
      largest = count
    }
  }
  return voted
}

function sessionHitRecord(hit: SearchHit): JsonObject {
  return {
    session_id: hit.sessionId,
    turn_id: hit.turnId,
    agent_id: hit.agentId,
    prompt: hit.prompt,
    response: hit.response,
    score: Math.round(hit.score * 10_000) / 10_000,
    timestamp: hit.timestamp,
  }
}

function skillMetadataRecord(skill: Skill, score?: number): JsonObject {
  const result: JsonObject = {
    name: skill.metadata.name,
    version: skill.metadata.version,
    description: skill.metadata.description,
    tags: [...skill.metadata.tags],
  }
  if (score !== undefined) result.score = Math.round(score * 1000) / 1000
  return result
}

function skillViewRecord(skill: Skill): JsonObject {
  return {
    name: skill.metadata.name,
    version: skill.metadata.version,
    description: skill.metadata.description,
    tags: [...skill.metadata.tags],
    instructions: skill.instructions,
    source_path: skill.sourcePath,
  }
}

function lexicalSkillScore(skill: Skill, query: string): number {
  const normalizedQuery = normalizeText(query)
  if (!normalizedQuery) return 0
  const name = normalizeText(skill.metadata.name)
  const description = normalizeText(skill.metadata.description)
  const tags = skill.metadata.tags.map(normalizeText)
  if (name === normalizedQuery) return 1
  let score = 0
  let matches = 0
  for (const term of normalizedQuery.split(' ')) {
    if (!term) continue
    if (name.includes(term)) {
      score += 4
      matches += 1
    } else if (tags.some(tag => tag.includes(term))) {
      score += 2
      matches += 1
    } else if (description.includes(term) || normalizeText(skill.instructions).includes(term)) {
      score += 1
      matches += 1
    }
  }
  if (matches === 0) return 0
  return score / (4 * Math.max(1, normalizedQuery.split(' ').filter(Boolean).length))
}

function normalizeText(value: string): string {
  return value.toLocaleLowerCase().trim().replace(/\s+/g, ' ')
}

function skillMarkdown(request: SkillManageRequest): string {
  return [
    '---',
    'name: ' + JSON.stringify(request.name),
    'description: ' + JSON.stringify(request.description),
    'version: ' + JSON.stringify(request.version),
    'tags: ' + JSON.stringify(request.tags),
    '---',
    '',
    request.instructions,
  ].join('\n')
}

async function skillMarkdownFiles(directory: string): Promise<string[]> {
  let entries: Array<{ readonly name: string; isDirectory(): boolean; isFile(): boolean }>
  try {
    entries = await readdir(directory, { encoding: 'utf8', withFileTypes: true })
  } catch (error) {
    if (isMissing(error)) return []
    throw error
  }
  const paths: string[] = []
  for (const entry of entries) {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) {
      paths.push(...await skillMarkdownFiles(path))
    } else if (entry.isFile() && entry.name === 'SKILL.md') {
      paths.push(path)
    }
  }
  return paths
}

function assertValidSkillName(name: string): void {
  if (!isValidSkillName(name)) {
    throw new ValidationError('name', 'must be a non-empty skill name without path traversal', name)
  }
}

function isValidSkillName(name: string): boolean {
  return Boolean(name) && !name.includes('/') && !name.includes('\\') && !name.includes('..') && !name.startsWith('.')
}

async function atomicWrite(path: string, content: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  const temporary = join(dirname(path), '.' + crypto.randomUUID() + '.tmp')
  try {
    await Bun.write(temporary, content)
    await rename(temporary, path)
  } finally {
    await rm(temporary, { force: true })
  }
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path)
    return true
  } catch (error) {
    if (isMissing(error)) return false
    throw error
  }
}

function positiveLimit(limit: number): number {
  if (limit < 1) throw new ValidationError('limit', 'must be at least 1', limit)
  return limit
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function toolDefinition(
  name: string,
  description: string,
  properties: Record<string, unknown> = {},
  required: readonly string[] = [],
): ToolDefinition {
  return {
    type: 'function',
    function: {
      name,
      description,
      parameters: {
        type: 'object',
        additionalProperties: false,
        properties,
        ...(required.length === 0 ? {} : { required }),
      },
    },
  }
}
