// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import { ToolRegistry, type ToolExecutionContext } from '../executors/toolRegistry.js'
import { ContextualMemory } from '../memory/contextualMemory.js'
import {
  type Memory,
  type MemoryFilters,
  type MemoryItem,
  type MemoryMetadata,
} from '../memory/base.js'
import type { JsonObject, JsonValue, ToolDefinition } from '../types/toolCalls.js'
import { optionalInteger, optionalString, optionalStringArray } from './inputs.js'

const MEMORY_TYPES = [
  'short_term',
  'long_term',
  'working',
  'episodic',
  'semantic',
  'procedural',
] as const

const DURABLE_MEMORY_TYPES = new Set<ManagedMemoryType>(['long_term', 'semantic', 'procedural'])
const DEFAULT_AGENT_ID = 'default'
const DEFAULT_SEARCH_LIMIT = 10
const DEFAULT_CONSOLIDATION_LIMIT = 20
const MAX_TOOL_LIMIT = 10_000
const TAG_SCAN_LIMIT = 1_000

export type ManagedMemoryType = (typeof MEMORY_TYPES)[number]

/** Operation names retained for consumers of the Python memory-tool surface. */
export enum MemoryOperation {
  SAVE = 'save',
  SEARCH = 'search',
  RETRIEVE = 'retrieve',
  DELETE = 'delete',
  SUMMARIZE = 'summarize',
  CONSOLIDATE = 'consolidate',
}

/**
 * A host-owned contextual, short-term, or long-term memory tier exposed to
 * the legacy Python-compatible memory-management tools.
 *
 * The requested legacy type is retained in item metadata. Contextual memory
 * maps long-term, semantic, and procedural entries into its durable tier;
 * short-term, working, and episodic entries stay in its working tier.
 */
export interface MemoryToolContext {
  readonly agentId?: string
  readonly memory: Memory
  readonly now?: () => Date
}

/** A privileged host decision on whether a cross-agent memory operation may proceed. */
export interface CrossAgentMemoryRequest {
  /** The execution-context agent making the call, when known. */
  readonly callingAgentId: string | undefined
  /** The model-supplied target agent that differs from the calling agent. */
  readonly requestedAgentId: string
  /** The memory operation being attempted. */
  readonly operation: MemoryOperation
}

export type CrossAgentMemoryAccess = (request: CrossAgentMemoryRequest) => boolean

export interface MemoryToolsOptions {
  /** A shared memory tier, suitable for a single embedded runtime. */
  readonly context?: MemoryToolContext
  /** Resolve session- or agent-specific memory at tool execution time. */
  readonly resolveContext?: (
    context: ToolExecutionContext,
  ) => MemoryToolContext | undefined | Promise<MemoryToolContext | undefined>
  /**
   * Explicit privileged host port that authorizes a model-supplied `agent_id`
   * targeting another agent's memories. Without it, memory tools always scope
   * to the execution-context agent and reject cross-agent `agent_id` values.
   */
  readonly allowCrossAgent?: CrossAgentMemoryAccess
}

export interface SaveMemoryInput {
  readonly agentId?: string
  readonly content: string
  readonly memoryType?: string
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly tags?: readonly string[]
}

export interface SearchMemoryInput {
  readonly agentId?: string
  readonly limit?: number
  readonly memoryTypes?: readonly string[]
  readonly query: string
  readonly tags?: readonly string[]
  readonly timeRange?: MemoryTimeRange
}

export interface MemoryTimeRange {
  readonly end?: string
  readonly start?: string
}

export interface ConsolidateAgentMemoriesInput {
  readonly agentId: string
  readonly maxItems?: number
}

export interface DeleteMemoryInput {
  readonly agentId?: string
  readonly memoryId?: string
  readonly olderThan?: string
  readonly tags?: readonly string[]
}

export interface MemoryStatisticsInput {
  readonly agentId?: string
}

export const SAVE_MEMORY_DEFINITION = definition(
  'save_memory',
  'Save a persistent memory entry for later retrieval.',
  {
    content: { type: 'string' },
    memory_type: { type: 'string', enum: MEMORY_TYPES, default: 'short_term' },
    tags: { type: 'array', items: { type: 'string' } },
    metadata: { type: 'object', additionalProperties: true },
    agent_id: {
      type: 'string',
      description: 'Memory owner. Defaults to the calling agent; other agents require host-granted cross-agent access.',
    },
  },
  ['content'],
)

export const SEARCH_MEMORY_DEFINITION = definition(
  'search_memory',
  'Search persistent memories by query, type, tag, agent, and time range.',
  {
    query: { type: 'string' },
    memory_types: { type: 'array', items: { type: 'string', enum: MEMORY_TYPES } },
    tags: { type: 'array', items: { type: 'string' } },
    limit: { type: 'integer', minimum: 1, maximum: MAX_TOOL_LIMIT, default: DEFAULT_SEARCH_LIMIT },
    agent_id: {
      type: 'string',
      description: 'Memory owner. Defaults to the calling agent; other agents require host-granted cross-agent access.',
    },
    time_range: {
      type: 'object',
      additionalProperties: false,
      properties: {
        start: { type: 'string', format: 'date-time' },
        end: { type: 'string', format: 'date-time' },
      },
    },
  },
  ['query'],
)

export const CONSOLIDATE_AGENT_MEMORIES_DEFINITION = definition(
  'consolidate_agent_memories',
  'Summarize one agent memory collection by tag.',
  {
    agent_id: { type: 'string' },
    max_items: { type: 'integer', minimum: 1, maximum: MAX_TOOL_LIMIT, default: DEFAULT_CONSOLIDATION_LIMIT },
  },
  ['agent_id'],
)

export const DELETE_MEMORY_DEFINITION = definition(
  'delete_memory',
  'Delete memories by ID, tag, producing agent, or ISO timestamp.',
  {
    memory_id: { type: 'string' },
    tags: { type: 'array', items: { type: 'string' } },
    agent_id: {
      type: 'string',
      description: 'Memory owner. Defaults to the calling agent; other agents require host-granted cross-agent access.',
    },
    older_than: { type: 'string', format: 'date-time' },
  },
)

export const GET_MEMORY_STATISTICS_DEFINITION = definition(
  'get_memory_statistics',
  'Report memory counts, types, and ownership statistics.',
  { agent_id: { type: 'string' } },
)

export const GET_MEMORY_TAGS_AND_TERMS_DEFINITION = definition(
  'get_memory_tags_and_terms',
  'List persistent memory tags and their frequency for one agent.',
  { agent_id: { type: 'string' } },
)

export const MEMORY_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  SAVE_MEMORY_DEFINITION,
  SEARCH_MEMORY_DEFINITION,
  CONSOLIDATE_AGENT_MEMORIES_DEFINITION,
  DELETE_MEMORY_DEFINITION,
  GET_MEMORY_STATISTICS_DEFINITION,
  GET_MEMORY_TAGS_AND_TERMS_DEFINITION,
]

/** Register the Python memory_tool.py public surface against an injected store. */
export function registerMemoryTools(registry: ToolRegistry, options: MemoryToolsOptions = {}): void {
  registry.register(SAVE_MEMORY_DEFINITION, (inputs, execution) => executeWithContext(
    execution,
    options,
    memory => saveMemory(parseSaveMemoryInput(inputs), memory, options.allowCrossAgent),
  ))
  registry.register(SEARCH_MEMORY_DEFINITION, (inputs, execution) => executeWithContext(
    execution,
    options,
    memory => searchMemory(parseSearchMemoryInput(inputs), memory, options.allowCrossAgent),
  ))
  registry.register(CONSOLIDATE_AGENT_MEMORIES_DEFINITION, (inputs, execution) => executeWithContext(
    execution,
    options,
    memory => consolidateAgentMemories(parseConsolidateInput(inputs), memory, options.allowCrossAgent),
  ))
  registry.register(DELETE_MEMORY_DEFINITION, (inputs, execution) => executeWithContext(
    execution,
    options,
    memory => deleteMemory(parseDeleteMemoryInput(inputs), memory, options.allowCrossAgent),
  ))
  registry.register(GET_MEMORY_STATISTICS_DEFINITION, (inputs, execution) => executeWithContext(
    execution,
    options,
    memory => getMemoryStatistics(parseStatisticsInput(inputs), memory),
  ))
  registry.register(GET_MEMORY_TAGS_AND_TERMS_DEFINITION, (inputs, execution) => executeWithContext(
    execution,
    options,
    memory => getMemoryTagsAndTerms(parseStatisticsInput(inputs), memory, options.allowCrossAgent),
  ))
}

/** Store an entry while retaining its legacy memory category in metadata. */
export function saveMemory(
  input: SaveMemoryInput,
  context: MemoryToolContext | undefined,
  allowCrossAgent?: CrossAgentMemoryAccess,
): JsonObject {
  if (!context) return unavailable()
  try {
    const content = stringInput(input.content, 'content')
    const memoryType = normalizeMemoryType(input.memoryType ?? 'short_term')
    const agentId = agentIdFor(input.agentId, context, MemoryOperation.SAVE, allowCrossAgent)
    const tags = normalizeTags(input.tags)
    const now = nowFor(context)
    const metadata: MemoryMetadata = {
      ...copyMetadata(input.metadata),
      tags,
      timestamp: now.toISOString(),
      created_by: agentId,
      requested_memory_type: memoryType,
    }
    const item = saveIntoMemory(context.memory, content, metadata, memoryType, agentId)
    return {
      status: 'success',
      memory_id: item.memoryId,
      message: 'Memory saved successfully',
    }
  } catch (error) {
    return failure(error)
  }
}

/** Search durable and working memory with the legacy memory_tool.py result shape. */
export function searchMemory(
  input: SearchMemoryInput,
  context: MemoryToolContext | undefined,
  allowCrossAgent?: CrossAgentMemoryAccess,
): JsonObject {
  if (!context) return unavailable()
  try {
    const query = textInput(input.query, 'query')
    const limit = positiveLimit(input.limit ?? DEFAULT_SEARCH_LIMIT, 'limit')
    const agentId = agentIdFor(input.agentId, context, MemoryOperation.SEARCH, allowCrossAgent)
    const tags = normalizeTags(input.tags)
    const memoryTypes = normalizeMemoryTypes(input.memoryTypes)
    const timeRange = parseTimeRange(input.timeRange)
    const filters = basicFilters(agentId, tags)
    const candidateLimit = query || tags.length ? Math.min(MAX_TOOL_LIMIT, limit * 5) : limit
    const searched = context.memory.search(query, candidateLimit, filters)
    const recalled = retrieveItems(context.memory, filters, candidateLimit)
    const memories = uniqueItems([...searched, ...recalled])
      .filter(item => matchesMemoryType(item, memoryTypes))
      .filter(item => matchesQuery(item, query))
      .filter(item => matchesTags(item, tags))
      .filter(item => matchesTimeRange(item, timeRange))
      .slice(0, limit)
      .map(item => memoryResult(item))

    return {
      status: 'success',
      count: memories.length,
      memories,
      query,
    }
  } catch (error) {
    return failure(error)
  }
}

/** Render a concise, tag-organized view of one agent's recent memories. */
export function consolidateAgentMemories(
  input: ConsolidateAgentMemoriesInput,
  context: MemoryToolContext | undefined,
  allowCrossAgent?: CrossAgentMemoryAccess,
): JsonObject {
  if (!context) return unavailable()
  try {
    const agentId = agentIdFor(stringInput(input.agentId, 'agent_id'), context, MemoryOperation.CONSOLIDATE, allowCrossAgent)
    const maxItems = positiveLimit(input.maxItems ?? DEFAULT_CONSOLIDATION_LIMIT, 'max_items')
    const memories = retrieveItems(context.memory, { agent_id: agentId }, maxItems)
      .sort((left, right) => timestampFor(right).valueOf() - timestampFor(left).valueOf())
      .slice(0, maxItems)

    return {
      status: 'success',
      summary: consolidationSummary(memories),
      statistics: statisticsFor(context.memory),
    }
  } catch (error) {
    return failure(error)
  }
}

/** Delete entries selected by ID or one or more supplied criteria. */
export function deleteMemory(
  input: DeleteMemoryInput,
  context: MemoryToolContext | undefined,
  allowCrossAgent?: CrossAgentMemoryAccess,
): JsonObject {
  if (!context) return unavailable()
  try {
    const memoryId = optionalText(input.memoryId, 'memory_id')
    const tags = normalizeTags(input.tags)
    const requestedAgentId = optionalText(input.agentId, 'agent_id')
    const olderThan = parseTimestamp(input.olderThan, 'older_than')
    if (!memoryId && tags.length === 0 && !requestedAgentId && !olderThan) {
      throw new ValidationError('delete_memory', 'requires memory_id, tags, agent_id, or older_than')
    }
    // Deletion is always scoped to the calling agent — including bulk tag and
    // older_than criteria — unless a privileged host port explicitly
    // authorizes the cross-agent target.
    const agentId = agentIdFor(requestedAgentId, context, MemoryOperation.DELETE, allowCrossAgent)

    let deletedCount = 0
    if (memoryId) {
      const owned = allItems(context.memory).find(item => item.memoryId === memoryId)
      if (owned !== undefined && owned.agentId !== agentId && !allowCrossAgent?.({
        callingAgentId: context.agentId,
        operation: MemoryOperation.DELETE,
        requestedAgentId: owned.agentId ?? DEFAULT_AGENT_ID,
      })) {
        throw new ValidationError(
          'memory_id',
          'belongs to another agent; cross-agent memory deletion requires an explicit privileged host port',
          memoryId,
        )
      }
      deletedCount = context.memory.delete(memoryId)
    } else {
      const items = allItems(context.memory)
        .filter(item => item.agentId === agentId)
        .filter(item => matchesTags(item, tags))
        .filter(item => !olderThan || timestampFor(item).valueOf() < olderThan.valueOf())
      for (const item of items) deletedCount += context.memory.delete(item.memoryId)
    }

    return {
      status: 'success',
      message: 'Successfully deleted ' + deletedCount + ' memories',
      deleted_count: deletedCount,
    }
  } catch (error) {
    return failure(error)
  }
}

/** Return tag groups and descending tag frequencies for the current agent. */
export function getMemoryTagsAndTerms(
  input: MemoryStatisticsInput,
  context: MemoryToolContext | undefined,
  allowCrossAgent?: CrossAgentMemoryAccess,
): JsonObject {
  if (!context) return unavailable()
  try {
    const agentId = agentIdFor(input.agentId, context, MemoryOperation.RETRIEVE, allowCrossAgent)
    const tagsByType = Object.fromEntries(MEMORY_TYPES.map(type => [type, new Set<string>()])) as Record<
      ManagedMemoryType,
      Set<string>
    >
    const frequency = new Map<string, number>()
    const memories = retrieveItems(context.memory, { agent_id: agentId }, TAG_SCAN_LIMIT)

    for (const item of memories) {
      const type = requestedMemoryType(item)
      if (!type) continue
      for (const tag of tagsFor(item)) {
        tagsByType[type].add(tag)
        frequency.set(tag, (frequency.get(tag) ?? 0) + 1)
      }
    }

    const grouped: Record<string, JsonValue> = {}
    for (const type of MEMORY_TYPES) {
      const tags = [...tagsByType[type]].sort()
      if (tags.length) grouped[type] = tags
    }
    const tagFrequency: Record<string, number> = {}
    for (const [tag, count] of [...frequency.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))) {
      tagFrequency[tag] = count
    }
    const allTags = [...frequency.keys()].sort()

    return {
      status: 'success',
      tags_by_type: grouped,
      all_tags: allTags,
      tag_frequency: tagFrequency,
      total_unique_tags: allTags.length,
      agent_id: agentId,
    }
  } catch (error) {
    return failure(error)
  }
}

/** Return legacy snake_case statistics plus an optional agent-specific count. */
export function getMemoryStatistics(
  input: MemoryStatisticsInput,
  context: MemoryToolContext | undefined,
): JsonObject {
  if (!context) return unavailable()
  try {
    const agentId = optionalText(input.agentId, 'agent_id')
    const statistics = statisticsFor(context.memory)
    if (!agentId) return { status: 'success', statistics }
    const agentMemoryCount = allItems(context.memory).filter(item => item.agentId === agentId).length
    return {
      status: 'success',
      statistics: {
        ...statistics,
        agent_memory_count: agentMemoryCount,
        agent_id: agentId,
      },
    }
  } catch (error) {
    return failure(error)
  }
}

export function getMemoryToolDescriptions(): ReadonlyArray<{
  readonly category: 'Memory Management'
  readonly description: string
  readonly name: string
}> {
  return MEMORY_TOOL_DEFINITIONS.map(definition_ => ({
    name: definition_.function.name,
    description: definition_.function.description,
    category: 'Memory Management',
  }))
}

async function executeWithContext(
  execution: ToolExecutionContext,
  options: MemoryToolsOptions,
  operation: (context: MemoryToolContext | undefined) => JsonObject,
): Promise<JsonObject> {
  try {
    return operation(await resolveContext(execution, options))
  } catch (error) {
    return failure(error)
  }
}

async function resolveContext(
  execution: ToolExecutionContext,
  options: MemoryToolsOptions,
): Promise<MemoryToolContext | undefined> {
  const resolved = options.context ?? (options.resolveContext ? await options.resolveContext(execution) : undefined)
  if (!resolved || resolved.agentId || !execution.agentId) return resolved
  return { ...resolved, agentId: execution.agentId }
}

function parseSaveMemoryInput(inputs: JsonObject): SaveMemoryInput {
  const metadata = optionalObject(inputs, 'metadata')
  const memoryType = optionalString(inputs, 'memory_type')
  const tags = inputs.tags === undefined ? undefined : optionalStringArray(inputs, 'tags')
  const agentId = optionalString(inputs, 'agent_id')
  return {
    content: requiredInputString(inputs, 'content'),
    ...(memoryType !== undefined ? { memoryType } : {}),
    ...(tags !== undefined ? { tags } : {}),
    ...(metadata !== undefined ? { metadata } : {}),
    ...(agentId !== undefined ? { agentId } : {}),
  }
}

function parseSearchMemoryInput(inputs: JsonObject): SearchMemoryInput {
  const timeRange = optionalObject(inputs, 'time_range')
  const memoryTypes = inputs.memory_types === undefined ? undefined : optionalStringArray(inputs, 'memory_types')
  const tags = inputs.tags === undefined ? undefined : optionalStringArray(inputs, 'tags')
  const limit = inputs.limit === undefined ? undefined : optionalInteger(inputs, 'limit', DEFAULT_SEARCH_LIMIT)
  const agentId = optionalString(inputs, 'agent_id')
  const start = timeRange ? optionalString(timeRange, 'start') : undefined
  const end = timeRange ? optionalString(timeRange, 'end') : undefined
  return {
    query: requiredInputText(inputs, 'query'),
    ...(memoryTypes !== undefined ? { memoryTypes } : {}),
    ...(tags !== undefined ? { tags } : {}),
    ...(limit !== undefined ? { limit } : {}),
    ...(agentId !== undefined ? { agentId } : {}),
    ...(timeRange !== undefined ? {
      timeRange: {
        ...(start !== undefined ? { start } : {}),
        ...(end !== undefined ? { end } : {}),
      },
    } : {}),
  }
}

function parseConsolidateInput(inputs: JsonObject): ConsolidateAgentMemoriesInput {
  return {
    agentId: requiredInputString(inputs, 'agent_id'),
    ...(inputs.max_items !== undefined ? { maxItems: optionalInteger(inputs, 'max_items', DEFAULT_CONSOLIDATION_LIMIT) } : {}),
  }
}

function parseDeleteMemoryInput(inputs: JsonObject): DeleteMemoryInput {
  const memoryId = optionalString(inputs, 'memory_id')
  const tags = inputs.tags === undefined ? undefined : optionalStringArray(inputs, 'tags')
  const agentId = optionalString(inputs, 'agent_id')
  const olderThan = optionalString(inputs, 'older_than')
  return {
    ...(memoryId !== undefined ? { memoryId } : {}),
    ...(tags !== undefined ? { tags } : {}),
    ...(agentId !== undefined ? { agentId } : {}),
    ...(olderThan !== undefined ? { olderThan } : {}),
  }
}

function parseStatisticsInput(inputs: JsonObject): MemoryStatisticsInput {
  const agentId = optionalString(inputs, 'agent_id')
  return agentId === undefined ? {} : { agentId }
}

function saveIntoMemory(
  memory: Memory,
  content: string,
  metadata: MemoryMetadata,
  memoryType: ManagedMemoryType,
  agentId: string,
): MemoryItem {
  const options = { agentId, memoryType, importance: 0.5 }
  if (memory instanceof ContextualMemory) {
    return memory.save(content, metadata, {
      ...options,
      toLongTerm: DURABLE_MEMORY_TYPES.has(memoryType),
    })
  }
  return memory.save(content, metadata, options)
}

function basicFilters(agentId: string, tags: readonly string[]): MemoryFilters | undefined {
  const filters: Record<string, unknown> = { agent_id: agentId }
  if (tags.length) {
    filters.tags = (value: unknown) => tagValues(value).some(tag => tags.includes(tag))
  }
  return filters
}

function retrieveItems(memory: Memory, filters: MemoryFilters | undefined, limit: number): MemoryItem[] {
  if (memory instanceof ContextualMemory) {
    return uniqueItems([
      ...asItems(memory.shortTerm.retrieve(undefined, filters, limit)),
      ...asItems(memory.longTerm.retrieve(undefined, filters, limit)),
    ]).slice(0, limit)
  }
  return asItems(memory.retrieve(undefined, filters, limit))
}

function allItems(memory: Memory): MemoryItem[] {
  if (memory instanceof ContextualMemory) {
    return uniqueItems([
      ...asItems(memory.shortTerm.retrieve(undefined, undefined, memory.shortTerm.size)),
      ...asItems(memory.longTerm.retrieve(undefined, undefined, memory.longTerm.size)),
    ])
  }
  return asItems(memory.retrieve(undefined, undefined, memory.size))
}

function asItems(value: MemoryItem | MemoryItem[] | undefined): MemoryItem[] {
  if (!value) return []
  return Array.isArray(value) ? value : [value]
}

function uniqueItems(items: readonly MemoryItem[]): MemoryItem[] {
  const seen = new Set<string>()
  return items.filter(item => {
    if (seen.has(item.memoryId)) return false
    seen.add(item.memoryId)
    return true
  })
}

function matchesMemoryType(item: MemoryItem, memoryTypes: readonly ManagedMemoryType[]): boolean {
  if (memoryTypes.length === 0) return true
  const itemType = requestedMemoryType(item)
  return itemType !== undefined && memoryTypes.includes(itemType)
}

function matchesQuery(item: MemoryItem, query: string): boolean {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (terms.length === 0) return true
  const content = item.content.toLowerCase()
  const tags = tagsFor(item).map(tag => tag.toLowerCase())
  return terms.some(term => content.includes(term) || tags.some(tag => tag.includes(term)))
}

function matchesTags(item: MemoryItem, tags: readonly string[]): boolean {
  return tags.length === 0 || tagsFor(item).some(tag => tags.includes(tag))
}

function matchesTimeRange(item: MemoryItem, range: ParsedTimeRange | undefined): boolean {
  if (!range) return true
  const time = timestampFor(item).valueOf()
  return (!range.start || time >= range.start.valueOf()) && (!range.end || time <= range.end.valueOf())
}

function timestampFor(item: MemoryItem): Date {
  const metadataTimestamp = item.metadata.timestamp
  if (typeof metadataTimestamp === 'string') {
    const parsed = new Date(metadataTimestamp)
    if (!Number.isNaN(parsed.valueOf())) return parsed
  }
  return item.timestamp
}

function requestedMemoryType(item: MemoryItem): ManagedMemoryType | undefined {
  const requested = item.metadata.requested_memory_type
  if (typeof requested === 'string') {
    try {
      return normalizeMemoryType(requested)
    } catch {
      return undefined
    }
  }
  try {
    return normalizeMemoryType(item.memoryType)
  } catch {
    return undefined
  }
}

function tagsFor(item: MemoryItem): string[] {
  return tagValues(item.metadata.tags)
}

function tagValues(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((tag): tag is string => typeof tag === 'string') : []
}

function memoryResult(item: MemoryItem): JsonObject {
  const metadata = jsonObject(item.metadata)
  return {
    memory_id: item.memoryId,
    memory_type: requestedMemoryType(item) ?? item.memoryType,
    content: item.content,
    tags: tagsFor(item),
    timestamp: typeof item.metadata.timestamp === 'string' ? item.metadata.timestamp : item.timestamp.toISOString(),
    metadata,
  }
}

function consolidationSummary(memories: readonly MemoryItem[]): string {
  if (memories.length === 0) return 'No memories found for this agent.'
  const tagged = new Map<string, MemoryItem[]>()
  for (const memory of memories) {
    const rawTags = memory.metadata.tags
    const tags = rawTags === undefined ? ['untagged'] : tagValues(rawTags)
    for (const tag of tags) {
      const values = tagged.get(tag) ?? []
      values.push(memory)
      tagged.set(tag, values)
    }
  }
  const lines = ['Total memories: ' + memories.length, '\nMemories by category:']
  for (const tag of [...tagged.keys()].sort()) {
    const entries = tagged.get(tag) ?? []
    lines.push('\n' + tag.toUpperCase() + ':')
    for (const memory of entries.slice(0, 3)) lines.push('  - ' + memory.content)
    if (entries.length > 3) lines.push('  ... and ' + (entries.length - 3) + ' more')
  }
  return lines.join('\n')
}

function statisticsFor(memory: Memory): JsonObject {
  const items = allItems(memory)
  const memoryTypes: Record<string, number> = {}
  const agents = new Set<string>()
  const users = new Set<string>()
  const conversations = new Set<string>()
  for (const item of items) {
    const type = requestedMemoryType(item) ?? item.memoryType
    memoryTypes[type] = (memoryTypes[type] ?? 0) + 1
    if (item.agentId) agents.add(item.agentId)
    if (item.userId) users.add(item.userId)
    if (item.conversationId) conversations.add(item.conversationId)
  }
  return {
    total_items: items.length,
    total_memories: items.length,
    max_items: maxItemsFor(memory),
    memory_types: memoryTypes,
    unique_agents: agents.size,
    unique_users: users.size,
    unique_conversations: conversations.size,
    cache_hit_rate: 0,
  }
}

function maxItemsFor(memory: Memory): number | null {
  if (!(memory instanceof ContextualMemory)) return memory.maxItems ?? null
  if (memory.shortTerm.maxItems === undefined || memory.longTerm.maxItems === undefined) return null
  return memory.shortTerm.maxItems + memory.longTerm.maxItems
}

function normalizeMemoryTypes(value: readonly string[] | undefined): ManagedMemoryType[] {
  return (value ?? []).map(normalizeMemoryType)
}

function normalizeMemoryType(value: string): ManagedMemoryType {
  const normalized = stringInput(value, 'memory_type').toLowerCase()
  if ((MEMORY_TYPES as readonly string[]).includes(normalized)) return normalized as ManagedMemoryType
  throw new ValidationError('memory_type', 'must be one of ' + MEMORY_TYPES.join(', '), value)
}

function normalizeTags(value: readonly string[] | undefined): string[] {
  if (value === undefined) return []
  if (!Array.isArray(value) || value.some(tag => typeof tag !== 'string')) {
    throw new ValidationError('tags', 'must be an array of strings', value)
  }
  return [...value]
}

interface ParsedTimeRange {
  readonly end?: Date
  readonly start?: Date
}

function parseTimeRange(value: MemoryTimeRange | undefined): ParsedTimeRange | undefined {
  if (value === undefined) return undefined
  if (!isRecord(value)) throw new ValidationError('time_range', 'must be an object', value)
  const start = parseTimestamp(value.start, 'time_range.start')
  const end = parseTimestamp(value.end, 'time_range.end')
  if (start && end && start.valueOf() > end.valueOf()) {
    throw new ValidationError('time_range', 'start must not be after end', value)
  }
  return {
    ...(start ? { start } : {}),
    ...(end ? { end } : {}),
  }
}

function parseTimestamp(value: unknown, field: string): Date | undefined {
  if (value === undefined) return undefined
  const text = stringInput(value, field)
  const parsed = new Date(text)
  if (Number.isNaN(parsed.valueOf())) throw new ValidationError(field, 'must be a valid ISO timestamp', value)
  return parsed
}

function copyMetadata(value: Readonly<Record<string, unknown>> | undefined): MemoryMetadata {
  if (value === undefined) return {}
  if (!isRecord(value)) throw new ValidationError('metadata', 'must be an object', value)
  return { ...value }
}

function nowFor(context: MemoryToolContext): Date {
  const now = context.now ? context.now() : new Date()
  if (Number.isNaN(now.valueOf())) throw new ValidationError('now', 'must return a valid date')
  return now
}

/**
 * Resolve the memory owner for one operation. The execution-context agent
 * always wins over a model-supplied `agent_id`; a cross-agent target is only
 * honored when an explicit privileged host port authorizes it.
 */
function agentIdFor(
  value: unknown,
  context: MemoryToolContext,
  operation: MemoryOperation,
  allowCrossAgent: CrossAgentMemoryAccess | undefined,
): string {
  const callingAgentId = context.agentId ?? DEFAULT_AGENT_ID
  const requested = optionalText(value, 'agent_id')
  if (requested === undefined || requested === callingAgentId) return callingAgentId
  if (allowCrossAgent?.({ callingAgentId: context.agentId, operation, requestedAgentId: requested })) {
    return requested
  }
  throw new ValidationError(
    'agent_id',
    'targets memories owned by another agent; cross-agent memory access requires an explicit privileged host port',
    requested,
  )
}

function optionalText(value: unknown, field: string): string | undefined {
  if (value === undefined) return undefined
  return stringInput(value, field)
}

function stringInput(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value) throw new ValidationError(field, 'must be a non-empty string', value)
  return value
}

function textInput(value: unknown, field: string): string {
  if (typeof value !== 'string') throw new ValidationError(field, 'must be a string', value)
  return value
}

function requiredInputString(inputs: JsonObject, field: string): string {
  return stringInput(inputs[field], field)
}

function requiredInputText(inputs: JsonObject, field: string): string {
  return textInput(inputs[field], field)
}

function optionalObject(inputs: JsonObject, field: string): JsonObject | undefined {
  const value = inputs[field]
  if (value === undefined) return undefined
  if (!isRecord(value)) throw new ValidationError(field, 'must be an object', value)
  return value as JsonObject
}

function positiveLimit(value: number, field: string): number {
  if (!Number.isInteger(value) || value < 1 || value > MAX_TOOL_LIMIT) {
    throw new ValidationError(field, 'must be an integer between 1 and ' + MAX_TOOL_LIMIT, value)
  }
  return value
}

function unavailable(): JsonObject {
  return { status: 'error', message: 'Memory store not available' }
}

function failure(error: unknown): JsonObject {
  return { status: 'error', message: error instanceof Error ? error.message : String(error) }
}

function jsonObject(value: Readonly<Record<string, unknown>>): JsonObject {
  const result: JsonObject = {}
  for (const [key, entry] of Object.entries(value)) result[key] = jsonValue(entry)
  return result
}

function jsonValue(value: unknown): JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value
  if (typeof value === 'number') return Number.isFinite(value) ? value : String(value)
  if (value instanceof Date) return value.toISOString()
  if (Array.isArray(value)) return value.map(jsonValue)
  if (isRecord(value)) return jsonObject(value)
  return String(value)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function definition(
  name: string,
  description: string,
  properties: Record<string, unknown>,
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
        ...(required.length ? { required } : {}),
      },
    },
  }
}
