// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AgentMemory, memoryIndexBodyIssue, normalizeScope } from '../memory/agentMemory.js'
import {
  AgentSelfMemory,
  getAgentSelfMemory,
  type AgentSelfMemoryLearningCategory,
} from '../memory/agentSelfMemory.js'
import { ToolRegistry, type ToolExecutionContext } from '../executors/toolRegistry.js'
import { findCredentialPatterns } from '../security/promptScanner.js'
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
  'Read one persistent memory file verbatim from the global scope (follows the agent everywhere) or the project '
    + 'scope (this repository only, and unavailable when no project root is configured — agent_memory_status says '
    + 'which). `path` is relative to the chosen scope root; absolute paths, paths escaping the root, and symlinks '
    + 'leading out of it are all refused. This is the way to reach content the prompt only showed you as one index '
    + 'line: MEMORY.md lists topics, topics/<name>.md holds the body, and long canonical files are clipped on the '
    + 'way into the prompt. Nothing throws — a missing file returns {ok:false, error}, so confirm the path with '
    + 'agent_memory_list rather than retrying the same read.',
  {
    scope: scopeSchema(),
    path: { type: 'string', description: 'Relative path inside the selected memory scope.' },
  },
  ['scope', 'path'],
)

export const AGENT_MEMORY_WRITE_DEFINITION: ToolDefinition = definition(
  'agent_memory_write',
  'Atomically replace one memory file with `body` in full. This is a whole-file write, never a patch: everything '
    + 'not repeated in `body` is gone, so read the file first unless you are creating it, and use '
    + 'agent_memory_append for anything accumulative. Two classes of write are refused before disk is touched, and '
    + 'come back as {ok:false, error} rather than throwing. First, any body matching a credential pattern — memory '
    + 'is durable and re-injected every turn, so a secret written once is disclosed to every later provider call; '
    + 'record where the secret lives (env var name, vault path) and never its value. Second, prose in MEMORY.md, '
    + 'which is an index of headings and one-line entries only: put the body in topics/<name>.md with '
    + 'name/description/type frontmatter and link it from the index. That frontmatter is what makes a topic '
    + 'discoverable, because only canonical files are injected in full and every topic shows up as a single '
    + 'manifest line built from its description.',
  {
    scope: scopeSchema(),
    path: { type: 'string', description: 'Relative path inside the selected memory scope.' },
    body: { type: 'string', description: 'Complete UTF-8 text to persist. Never a credential value.' },
  },
  ['scope', 'path', 'body'],
)

export const AGENT_MEMORY_APPEND_DEFINITION: ToolDefinition = definition(
  'agent_memory_append',
  'Add an entry to a memory file without reading or rewriting it. The write goes through O_APPEND, so two writers '
    + 'sharing the directory — daemon and CLI, or parallel agents — cannot silently lose each other\'s entries the '
    + 'way a read-modify-write does; prefer this over agent_memory_write for anything that accumulates. The body is '
    + 'trimmed, prefixed with an ISO timestamp comment unless timestamp=false, optionally preceded by a fresh '
    + '"## <section>" heading (a new heading each call — it is never merged into an existing section of the same '
    + 'name), and separated from existing content by a blank line. An empty or whitespace-only body is an error; '
    + 'the file and its parent directories are created when missing. The same credential refusal as '
    + 'agent_memory_write applies, and failures return {ok:false, error} instead of throwing.',
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
  'Enumerate memory files recursively in one scope, or in both when `scope` is omitted (global alone if no project '
    + 'root is configured). Symlinked entries are skipped entirely. Each row carries the byte size plus the name, '
    + 'description, and type parsed from the file\'s frontmatter — an empty description is the signal that a topic '
    + 'file is not yet discoverable from the index, since the prompt manifest is built from exactly that field. '
    + 'count:0 means the scope directory exists but is empty, which is different from memory being unconfigured; '
    + 'agent_memory_status separates those two. Cheaper than guessing: list before reading.',
  { scope: scopeSchema() },
)

export const AGENT_MEMORY_SEARCH_DEFINITION: ToolDefinition = definition(
  'agent_memory_search',
  'Case-insensitive literal substring search across every memory file in scope. No regex, no stemming, no ranking: '
    + 'the query is matched as one contiguous string, so a multi-word query only hits where those words appear '
    + 'together in that order. Each hit is a roughly 120-character snippet around the match with newlines flattened '
    + 'to " / ", and at most three hits are taken from any single file, so a common term returns three per file '
    + 'rather than every occurrence. Hits arrive in file order and stop at `limit`. A blank query returns zero hits '
    + 'rather than everything. count:0 means the literal string is absent — try a shorter fragment before '
    + 'concluding nothing was recorded — and use agent_memory_read when a snippet needs its surrounding context.',
  {
    query: { type: 'string' },
    scope: scopeSchema(),
    limit: { type: 'integer', minimum: 1, maximum: 1000000, default: 20 },
  },
  ['query'],
)

export const AGENT_MEMORY_JOURNAL_DEFINITION: ToolDefinition = definition(
  'agent_memory_journal',
  'Append one bullet to journal/<today-in-UTC>.md in the chosen scope, creating that day file on first use. The '
    + 'date and the HH:MM:SS prefix come from the clock, so do not repeat them inside `note`. This is the running '
    + 'log of what happened, not durable knowledge: journal files are never injected into the prompt and are only '
    + 'reachable afterwards through agent_memory_search or agent_memory_read, so anything that should change future '
    + 'behaviour belongs in a topic file via agent_memory_write instead. The same credential refusal applies, and '
    + 'failures return {ok:false, error}.',
  {
    scope: scopeSchema(),
    note: { type: 'string' },
  },
  ['scope', 'note'],
)

export const AGENT_MEMORY_STATUS_DEFINITION: ToolDefinition = definition(
  'agent_memory_status',
  'Report whether persistent memory is wired up at all, plus the resolved global and project directories and the '
    + 'file count per scope. available:false means every other agent_memory_* tool in this session will answer '
    + '"agent memory not configured" — a configuration fact, not a transient error, so stop calling them instead of '
    + 'retrying. project_dir:null means only the global scope exists and any call with scope="project" will fail. '
    + 'Takes no arguments.',
)

export const AGENT_MEMORY_LEARN_DEFINITION: ToolDefinition = definition(
  'agent_memory_learn',
  'Record one durable observation in the agent\'s self-knowledge store, which is a separate store from the scope '
    + 'files the other agent_memory_* tools read and write. The category decides where the observation lands and '
    + 'the categories are not interchangeable: user_taste updates the user-preference note, tool_pattern and '
    + 'self_reflection each append a bullet to their own file, and skill_proposal files a proposed skill whose name '
    + 'is taken from the first sentence of the observation — so for that category, lead with a short name-like '
    + 'sentence. `importance` is accepted and currently ignored: it affects neither storage nor ordering nor '
    + 'retrieval, so do not try to encode urgency through it. Returns a one-line confirmation naming what was '
    + 'recorded.',
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
  'Refresh the stored project brief: reads AGENTS.md, XERXES.md, USER.md, and SOUL.md from the project root only — '
    + 'not from any subdirectory, and no other filename — clips each to its first 2000 characters, and overwrites '
    + 'the project-context note in self-knowledge with the result. Missing files are skipped in silence, so a '
    + 'successful call is not evidence that anything was found. It replaces the previous snapshot wholesale rather '
    + 'than merging, which makes it worth re-running only after those files actually change. Takes no arguments and '
    + 'returns a fixed confirmation string.',
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
    const path = requiredString(inputs, 'path')
    const body = requiredString(inputs, 'body')
    const rejection = memoryWriteRejection(path, body)
    if (rejection) return rejection
    const result = await memory.write(normalizeScope(requiredString(inputs, 'scope')), path, body)
    return { ok: true, scope: result.scope, path: result.path, bytes: result.bytes }
  })
}

/**
 * Refuse a memory write that would leak a credential or turn the index into a
 * container. Both checks run before the file is touched: memory is durable and
 * re-injected every turn, so a secret written once is a secret disclosed to
 * every later provider call.
 */
function memoryWriteRejection(path: string, body: string): JsonObject | undefined {
  const credentials = findCredentialPatterns(body)
  if (credentials.length > 0) {
    return {
      ok: false,
      error: 'refusing to persist credentials to memory: '
        + `${credentials.join(', ')} detected. Record where the secret lives (env var, vault path), never its value.`,
      credential_patterns: credentials,
    }
  }
  const indexIssue = memoryIndexBodyIssue(path, body)
  return indexIssue === undefined ? undefined : { ok: false, error: indexIssue }
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
    const path = requiredString(inputs, 'path')
    const body = requiredString(inputs, 'body')
    const rejection = memoryWriteRejection(path, body)
    if (rejection) return rejection
    const result = await memory.append(
      normalizeScope(requiredString(inputs, 'scope')),
      path,
      body,
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
      files: files.map(file => ({
        scope: file.scope,
        relative: file.path,
        bytes: file.bytes,
        description: file.description,
        type: file.type,
      })),
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
    const note = requiredString(inputs, 'note')
    const rejection = memoryWriteRejection('journal', note)
    if (rejection) return rejection
    const result = await memory.journal(normalizeScope(requiredString(inputs, 'scope')), note)
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
