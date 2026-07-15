// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AgentMemory, normalizeScope } from '../memory/agentMemory.js'
import {
  AgentSelfMemory,
  getAgentSelfMemory,
  type AgentSelfMemoryLearningCategory,
} from '../memory/agentSelfMemory.js'
import { ToolRegistry, type ToolExecutionContext } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { optionalBoolean, optionalInteger, optionalString, requiredString } from './inputs.js'

export interface AgentMemoryToolsOptions {
  /** A shared memory instance, suitable for one-session or one-workspace hosts. */
  readonly memory?: AgentMemory
  /** Resolve memory at execution time when memory differs by session or agent. */
  readonly resolveMemory?: (context: ToolExecutionContext) => AgentMemory | undefined | Promise<AgentMemory | undefined>
  /** Optional self-knowledge store used by learn and project-context sync. */
  readonly selfMemory?: AgentSelfMemory
  /** Resolve self-knowledge at execution time when it differs by agent or session. */
  readonly resolveSelfMemory?: (
    context: ToolExecutionContext,
  ) => AgentSelfMemory | undefined | Promise<AgentSelfMemory | undefined>
}

export const AGENT_MEMORY_READ_DEFINITION: ToolDefinition = definition(
  'agent_memory_read',
  'Read a persistent agent-memory file from the global or project scope.',
  {
    scope: scopeSchema(),
    path: { type: 'string', description: 'Relative path inside the selected memory scope.' },
  },
  ['scope', 'path'],
)

export const AGENT_MEMORY_WRITE_DEFINITION: ToolDefinition = definition(
  'agent_memory_write',
  'Atomically replace a persistent agent-memory file in the global or project scope.',
  {
    scope: scopeSchema(),
    path: { type: 'string', description: 'Relative path inside the selected memory scope.' },
    body: { type: 'string', description: 'Complete UTF-8 text to persist.' },
  },
  ['scope', 'path', 'body'],
)

export const AGENT_MEMORY_APPEND_DEFINITION: ToolDefinition = definition(
  'agent_memory_append',
  'Append an entry to a persistent agent-memory file without losing concurrent entries.',
  {
    scope: scopeSchema(),
    path: { type: 'string', description: 'Relative path inside the selected memory scope.' },
    body: { type: 'string', description: 'Non-empty entry body.' },
    section: { type: 'string', description: 'Optional Markdown section heading.' },
    timestamp: { type: 'boolean', default: true, description: 'Prepend a UTC timestamp comment.' },
  },
  ['scope', 'path', 'body'],
)

export const AGENT_MEMORY_LIST_DEFINITION: ToolDefinition = definition(
  'agent_memory_list',
  'List persistent agent-memory files in one scope or both configured scopes.',
  { scope: scopeSchema() },
)

export const AGENT_MEMORY_SEARCH_DEFINITION: ToolDefinition = definition(
  'agent_memory_search',
  'Case-insensitively search persistent agent-memory files for a short query.',
  {
    query: { type: 'string' },
    scope: scopeSchema(),
    limit: { type: 'integer', minimum: 1, maximum: 1000000, default: 20 },
  },
  ['query'],
)

export const AGENT_MEMORY_JOURNAL_DEFINITION: ToolDefinition = definition(
  'agent_memory_journal',
  'Append one timestamped note to today’s project or global memory journal.',
  {
    scope: scopeSchema(),
    note: { type: 'string' },
  },
  ['scope', 'note'],
)

export const AGENT_MEMORY_STATUS_DEFINITION: ToolDefinition = definition(
  'agent_memory_status',
  'Report whether persistent agent memory is available and summarize configured files.',
)

export const AGENT_MEMORY_LEARN_DEFINITION: ToolDefinition = definition(
  'agent_memory_learn',
  'Record a durable observation about user preferences, tool patterns, skills, or self-reflection.',
  {
    observation: { type: 'string', description: 'The durable observation to record.' },
    category: {
      type: 'string',
      enum: ['user_taste', 'tool_pattern', 'skill_proposal', 'self_reflection'],
      description: 'The category of durable learning.',
    },
    importance: { type: 'string', enum: ['low', 'medium', 'high'], default: 'medium' },
  },
  ['observation', 'category'],
)

export const AGENT_MEMORY_SYNC_CONTEXT_DEFINITION: ToolDefinition = definition(
  'agent_memory_sync_context',
  'Read AGENTS.md, XERXES.md, USER.md, and SOUL.md from the project tree into self-knowledge.',
)

export const AGENT_MEMORY_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  AGENT_MEMORY_READ_DEFINITION,
  AGENT_MEMORY_WRITE_DEFINITION,
  AGENT_MEMORY_APPEND_DEFINITION,
  AGENT_MEMORY_LIST_DEFINITION,
  AGENT_MEMORY_SEARCH_DEFINITION,
  AGENT_MEMORY_JOURNAL_DEFINITION,
  AGENT_MEMORY_STATUS_DEFINITION,
  AGENT_MEMORY_LEARN_DEFINITION,
  AGENT_MEMORY_SYNC_CONTEXT_DEFINITION,
]

/** Register the persistent-memory tools against a host-owned memory resolver. */
export function registerAgentMemoryTools(registry: ToolRegistry, options: AgentMemoryToolsOptions): void {
  registry.register(AGENT_MEMORY_READ_DEFINITION, (inputs, context) => agentMemoryRead(inputs, context, options))
  registry.register(AGENT_MEMORY_WRITE_DEFINITION, (inputs, context) => agentMemoryWrite(inputs, context, options))
  registry.register(AGENT_MEMORY_APPEND_DEFINITION, (inputs, context) => agentMemoryAppend(inputs, context, options))
  registry.register(AGENT_MEMORY_LIST_DEFINITION, (inputs, context) => agentMemoryList(inputs, context, options))
  registry.register(AGENT_MEMORY_SEARCH_DEFINITION, (inputs, context) => agentMemorySearch(inputs, context, options))
  registry.register(AGENT_MEMORY_JOURNAL_DEFINITION, (inputs, context) => agentMemoryJournal(inputs, context, options))
  registry.register(AGENT_MEMORY_STATUS_DEFINITION, (inputs, context) => agentMemoryStatus(inputs, context, options))
  registry.register(AGENT_MEMORY_LEARN_DEFINITION, (inputs, context) => agentMemoryLearn(inputs, context, options))
  registry.register(AGENT_MEMORY_SYNC_CONTEXT_DEFINITION, (inputs, context) => agentMemorySyncContext(inputs, context, options))
}

export async function agentMemoryRead(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: AgentMemoryToolsOptions,
): Promise<JsonObject> {
  return withMemory(context, options, async memory => {
    const scope = normalizeScope(requiredString(inputs, 'scope'))
    const path = requiredString(inputs, 'path')
    const body = await memory.read(scope, path)
    return { ok: true, scope, path, body, bytes: Buffer.byteLength(body) }
  })
}

export async function agentMemoryWrite(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: AgentMemoryToolsOptions,
): Promise<JsonObject> {
  return withMemory(context, options, async memory => {
    const result = await memory.write(
      normalizeScope(requiredString(inputs, 'scope')),
      requiredString(inputs, 'path'),
      requiredString(inputs, 'body'),
    )
    return { ok: true, scope: result.scope, path: result.path, bytes: result.bytes }
  })
}

export async function agentMemoryAppend(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: AgentMemoryToolsOptions,
): Promise<JsonObject> {
  return withMemory(context, options, async memory => {
    const section = optionalString(inputs, 'section')
    const appendOptions: { section?: string; timestamp?: boolean } = {
      timestamp: optionalBoolean(inputs, 'timestamp', true),
    }
    if (section !== undefined) appendOptions.section = section
    const result = await memory.append(
      normalizeScope(requiredString(inputs, 'scope')),
      requiredString(inputs, 'path'),
      requiredString(inputs, 'body'),
      appendOptions,
    )
    return { ok: true, scope: result.scope, path: result.path, appended_bytes: result.appendedBytes }
  })
}

export async function agentMemoryList(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: AgentMemoryToolsOptions,
): Promise<JsonObject> {
  return withMemory(context, options, async memory => {
    const selected = optionalString(inputs, 'scope')
    const scope = selected === undefined ? undefined : normalizeScope(selected)
    const files = await memory.listFiles(scope)
    return {
      ok: true,
      scope: scope ?? 'all',
      count: files.length,
      files: files.map(file => ({ scope: file.scope, relative: file.path, bytes: file.bytes })),
    }
  })
}

export async function agentMemorySearch(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: AgentMemoryToolsOptions,
): Promise<JsonObject> {
  return withMemory(context, options, async memory => {
    const selected = optionalString(inputs, 'scope')
    const scope = selected === undefined ? undefined : normalizeScope(selected)
    const query = requiredString(inputs, 'query')
    const searchOptions: { limit?: number; scope?: string } = { limit: optionalInteger(inputs, 'limit', 20) }
    if (scope !== undefined) searchOptions.scope = scope
    const hits = await memory.search(query, searchOptions)
    return { ok: true, query, count: hits.length, hits }
  })
}

export async function agentMemoryJournal(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: AgentMemoryToolsOptions,
): Promise<JsonObject> {
  return withMemory(context, options, async memory => {
    const result = await memory.journal(normalizeScope(requiredString(inputs, 'scope')), requiredString(inputs, 'note'))
    return { ok: true, scope: result.scope, path: result.path, appended_bytes: result.appendedBytes }
  })
}

export async function agentMemoryStatus(
  _inputs: JsonObject,
  context: ToolExecutionContext,
  options: AgentMemoryToolsOptions,
): Promise<JsonObject> {
  const memory = await resolveMemory(context, options)
  if (!memory) return { ok: true, available: false }
  try {
    const status = await memory.status()
    return {
      ok: true,
      available: true,
      global_dir: status.globalDirectory,
      project_dir: status.projectDirectory ?? null,
      files_by_scope: status.filesByScope,
      total_files: status.totalFiles,
    }
  } catch (error) {
    return failure(error)
  }
}

export async function agentMemoryLearn(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: AgentMemoryToolsOptions,
): Promise<string> {
  const memory = await resolveSelfMemory(context, options)
  return memory.learn(
    requiredString(inputs, 'observation'),
    requiredString(inputs, 'category') as AgentSelfMemoryLearningCategory,
    optionalString(inputs, 'importance') ?? 'medium',
  )
}

export async function agentMemorySyncContext(
  _inputs: JsonObject,
  context: ToolExecutionContext,
  options: AgentMemoryToolsOptions,
): Promise<string> {
  const memory = await resolveSelfMemory(context, options)
  const projectRoot = typeof context.metadata.project_root === 'string'
    ? context.metadata.project_root
    : process.cwd()
  await memory.syncProjectContext(projectRoot)
  return 'Project context synced to agent memory.'
}

async function withMemory(
  context: ToolExecutionContext,
  options: AgentMemoryToolsOptions,
  operation: (memory: AgentMemory) => Promise<JsonObject>,
): Promise<JsonObject> {
  const memory = await resolveMemory(context, options)
  if (!memory) return { ok: false, error: 'agent memory not configured for this session' }
  try {
    return await operation(memory)
  } catch (error) {
    return failure(error)
  }
}

async function resolveMemory(
  context: ToolExecutionContext,
  options: AgentMemoryToolsOptions,
): Promise<AgentMemory | undefined> {
  return options.memory ?? (options.resolveMemory ? await options.resolveMemory(context) : undefined)
}

async function resolveSelfMemory(
  context: ToolExecutionContext,
  options: AgentMemoryToolsOptions,
): Promise<AgentSelfMemory> {
  const resolved = options.selfMemory
    ?? (options.resolveSelfMemory ? await options.resolveSelfMemory(context) : undefined)
  return resolved ?? getAgentSelfMemory(context.agentId ?? 'default')
}

function failure(error: unknown): JsonObject {
  return { ok: false, error: error instanceof Error ? error.message : String(error) }
}

function definition(name: string, description: string, properties: Record<string, unknown> = {}, required: string[] = []): ToolDefinition {
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

function scopeSchema(): Record<string, unknown> {
  return { type: 'string', enum: ['global', 'project'] }
}
