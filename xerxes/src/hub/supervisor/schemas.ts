// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { validateToolArguments } from '../../runtime/argumentValidation.js'
import type { JsonObject, JsonSchema } from '../../types/toolCalls.js'

/**
 * Tool contract for the XERXES supervisor agent.
 *
 * These are Hub logical tools, not daemon methods (the daemon rejects a
 * `task.*` namespace). Each tool has a provider-facing name with underscores,
 * because several providers refuse dots in tool names, and a canonical dotted
 * name used in audit records and confirmation cards.
 *
 * Schemas are flat (`workspace_id` and `task_id` at the top level) so the
 * shared validator checks every field. `validateSupervisorArguments` adds the
 * length, range and item-count bounds that validator does not enforce.
 *
 * Gates:
 * - `read`: access-controlled read inside the thread's observation scope.
 * - `confirm`: creates an action intent; a human must tap Confirm before it runs.
 * - `audience`: delivery restricted to authorized thread participants.
 */

export type SupervisorGate = 'read' | 'confirm' | 'audience'

export interface SupervisorToolSpec {
  readonly name: SupervisorToolName
  readonly providerName: string
  readonly description: string
  readonly gate: SupervisorGate
  readonly parameters: JsonSchema
}

const ID = { type: 'string', minLength: 1, maxLength: 200 } as const
const WORKSPACE_ID = { ...ID, description: 'Workspace id from workspace_list.' } as const
const TASK_ID = { ...ID, description: 'Task (session) id from tasks_list.' } as const

function object(properties: Record<string, unknown>, required: readonly string[] = []): JsonSchema {
  return Object.freeze({ type: 'object', additionalProperties: false, properties, required: [...required] })
}

export const SUPERVISOR_TOOL_NAMES = [
  'workspace.list',
  'tasks.list',
  'task.digest',
  'task.tail',
  'task.search',
  'task.start',
  'task.steer',
  'task.stop',
  'agent.message',
  'goal.set',
  'model.set',
  'notify.user',
] as const
export type SupervisorToolName = (typeof SUPERVISOR_TOOL_NAMES)[number]

const SPECS: readonly SupervisorToolSpec[] = [
  {
    name: 'workspace.list',
    providerName: 'workspace_list',
    gate: 'read',
    description: 'List the workspaces you can observe, with connection health and capabilities.',
    parameters: object({}),
  },
  {
    name: 'tasks.list',
    providerName: 'tasks_list',
    gate: 'read',
    description: 'List tasks (sessions) across workspaces with status, goal, pending-human counts and agent tallies. Summaries only, never transcripts.',
    parameters: object({
      workspace_id: WORKSPACE_ID,
      cursor: { type: 'string', maxLength: 512 },
      limit: { type: 'integer', minimum: 1, maximum: 100, default: 20 },
    }),
  },
  {
    name: 'task.digest',
    providerName: 'task_digest',
    gate: 'read',
    description: 'Read a short digest of one task: objective, status, last outcome, blockers, waiting approvals, agents and next step. Status and counts are facts; the prose is a summary.',
    parameters: object(
      { workspace_id: WORKSPACE_ID, task_id: TASK_ID, max_age_seconds: { type: 'integer', minimum: 0, maximum: 300, default: 30 } },
      ['workspace_id', 'task_id'],
    ),
  },
  {
    name: 'task.tail',
    providerName: 'task_tail',
    gate: 'read',
    description: 'Read the last few user-visible steps of one task, bounded to 8 KiB.',
    parameters: object(
      {
        workspace_id: WORKSPACE_ID,
        task_id: TASK_ID,
        after: { type: 'integer', minimum: 0 },
        limit: { type: 'integer', minimum: 1, maximum: 20, default: 10 },
      },
      ['workspace_id', 'task_id'],
    ),
  },
  {
    name: 'task.search',
    providerName: 'task_search',
    gate: 'read',
    description: "Search one task's full history, including turns compaction archived.",
    parameters: object(
      {
        workspace_id: WORKSPACE_ID,
        task_id: TASK_ID,
        query: { type: 'string', minLength: 1, maxLength: 256 },
        limit: { type: 'integer', minimum: 1, maximum: 10, default: 5 },
      },
      ['workspace_id', 'task_id', 'query'],
    ),
  },
  {
    name: 'task.start',
    providerName: 'task_start',
    gate: 'confirm',
    description: 'Propose starting a new task (or a new turn in an existing idle task). A human must confirm before it runs.',
    parameters: object(
      {
        workspace_id: WORKSPACE_ID,
        task_id: TASK_ID,
        text: { type: 'string', minLength: 1, maxLength: 4000 },
        preset: ID,
      },
      ['workspace_id', 'text'],
    ),
  },
  {
    name: 'task.steer',
    providerName: 'task_steer',
    gate: 'confirm',
    description: 'Propose a steering message for a running task. A human must confirm. It is delivered at the next step and does not start an idle task.',
    parameters: object(
      { workspace_id: WORKSPACE_ID, task_id: TASK_ID, text: { type: 'string', minLength: 1, maxLength: 2000 } },
      ['workspace_id', 'task_id', 'text'],
    ),
  },
  {
    name: 'task.stop',
    providerName: 'task_stop',
    gate: 'confirm',
    description: "Propose stopping a task's current turn (and pausing its goal). With cascade, its running agents are interrupted too. A human must confirm.",
    parameters: object(
      { workspace_id: WORKSPACE_ID, task_id: TASK_ID, cascade: { type: 'boolean', default: false } },
      ['workspace_id', 'task_id'],
    ),
  },
  {
    name: 'agent.message',
    providerName: 'agent_message',
    gate: 'confirm',
    description: 'Propose a message to one subagent of a task. A human must confirm. A finished agent is reported, never restarted.',
    parameters: object(
      { workspace_id: WORKSPACE_ID, task_id: TASK_ID, agent_id: ID, text: { type: 'string', minLength: 1, maxLength: 2000 } },
      ['workspace_id', 'task_id', 'agent_id', 'text'],
    ),
  },
  {
    name: 'goal.set',
    providerName: 'goal_set',
    gate: 'confirm',
    description: "Propose setting or replacing a task's goal. expected_revision 0 means the task has no goal yet. A human must confirm. You can never mark a goal complete.",
    parameters: object(
      {
        workspace_id: WORKSPACE_ID,
        task_id: TASK_ID,
        expected_revision: { type: 'integer', minimum: 0 },
        objective: { type: 'string', minLength: 1, maxLength: 4000 },
        criteria: {
          type: 'array',
          maxItems: 32,
          items: {
            type: 'object',
            additionalProperties: false,
            required: ['id', 'description'],
            properties: {
              id: { type: 'string', minLength: 1, maxLength: 80 },
              description: { type: 'string', minLength: 1, maxLength: 1000 },
            },
          },
        },
      },
      ['workspace_id', 'task_id', 'expected_revision', 'objective'],
    ),
  },
  {
    name: 'model.set',
    providerName: 'model_set',
    gate: 'confirm',
    description: "Propose changing one task's model for that session only. Never changes defaults or credentials. A human must confirm.",
    parameters: object(
      { workspace_id: WORKSPACE_ID, task_id: TASK_ID, provider_profile: ID, model: { type: 'string', minLength: 1, maxLength: 512 } },
      ['workspace_id', 'task_id', 'provider_profile', 'model'],
    ),
  },
  {
    name: 'notify.user',
    providerName: 'notify_user',
    gate: 'audience',
    description: 'Send a short notice to people in this thread (inbox, and push when they enabled it).',
    parameters: object(
      {
        recipients: { type: 'array', minItems: 1, maxItems: 20, uniqueItems: true, items: { type: 'string' } },
        text: { type: 'string', minLength: 1, maxLength: 1000 },
        severity: { type: 'string', enum: ['info', 'attention'] },
      },
      ['recipients', 'text', 'severity'],
    ),
  },
]

export const SUPERVISOR_TOOLS: ReadonlyMap<SupervisorToolName, SupervisorToolSpec> = new Map(SPECS.map(spec => [spec.name, spec]))
const BY_PROVIDER_NAME: ReadonlyMap<string, SupervisorToolSpec> = new Map(SPECS.map(spec => [spec.providerName, spec]))

/**
 * Tool names that must never be registered for a supervisor, whatever a preset
 * or plugin says. Approval, review, permission-mode and generic passthrough
 * routes are human-only or unbounded.
 */
export const SUPERVISOR_FORBIDDEN_TOOLS: ReadonlySet<string> = new Set([
  'permission_response',
  'permission.respond',
  'question.respond',
  'learning.review',
  'lesson_review',
  'reflection_review',
  'goal.decision',
  'update_goal',
  'set_permission_mode',
  'slash',
  'rpc',
  'exec_command',
  'ExecuteShell',
  'Workflow',
])

export function supervisorToolByProviderName(name: string): SupervisorToolSpec | undefined {
  return BY_PROVIDER_NAME.get(name)
}

/** Throws when a registry would expose a forbidden or unknown tool to the supervisor. */
export function assertSupervisorToolCeiling(names: Iterable<string>): void {
  for (const name of names) {
    if (SUPERVISOR_FORBIDDEN_TOOLS.has(name)) throw new Error(`tool ${name} is human-only or unbounded and can never be given to the supervisor`)
    if (!BY_PROVIDER_NAME.has(name) && !SUPERVISOR_TOOLS.has(name as SupervisorToolName)) {
      throw new Error(`tool ${name} is outside the supervisor tool ceiling`)
    }
  }
}

export type SupervisorArgumentResult =
  | { readonly ok: true; readonly tool: SupervisorToolSpec; readonly args: JsonObject }
  | { readonly ok: false; readonly error: string }

/**
 * Validate model-emitted arguments for one supervisor tool by provider or
 * canonical name: the shared validator first, then the declared bounds.
 */
export function validateSupervisorArguments(name: string, args: unknown): SupervisorArgumentResult {
  const tool = BY_PROVIDER_NAME.get(name) ?? SUPERVISOR_TOOLS.get(name as SupervisorToolName)
  if (!tool) return { ok: false, error: `unknown supervisor tool ${JSON.stringify(name)}` }
  const base = validateToolArguments(tool.providerName, args, tool.parameters)
  if (!base.ok) return { ok: false, error: base.error }
  const coerced = base.coerced ?? {}
  const error = boundsError('', coerced, tool.parameters)
  if (error) return { ok: false, error }
  return { ok: true, tool, args: coerced }
}

/** Length, range and count bounds the shared validator leaves out. */
function boundsError(path: string, value: unknown, schema: unknown): string | undefined {
  if (!isRecord(schema)) return undefined
  const at = path || 'arguments'
  if (typeof value === 'string') {
    const length = [...value].length
    if (typeof schema.minLength === 'number' && length < schema.minLength) return `${at} must have at least ${schema.minLength} character(s)`
    if (typeof schema.maxLength === 'number' && length > schema.maxLength) return `${at} must have at most ${schema.maxLength} characters, got ${length}`
    return undefined
  }
  if (typeof value === 'number') {
    if (typeof schema.minimum === 'number' && value < schema.minimum) return `${at} must be at least ${schema.minimum}`
    if (typeof schema.maximum === 'number' && value > schema.maximum) return `${at} must be at most ${schema.maximum}`
    return undefined
  }
  if (Array.isArray(value)) {
    if (typeof schema.minItems === 'number' && value.length < schema.minItems) return `${at} needs at least ${schema.minItems} item(s)`
    if (typeof schema.maxItems === 'number' && value.length > schema.maxItems) return `${at} accepts at most ${schema.maxItems} items`
    if (schema.uniqueItems === true && new Set(value.map(item => JSON.stringify(item))).size !== value.length) return `${at} items must be unique`
    for (const [index, item] of value.entries()) {
      const nested = boundsError(`${at}[${index}]`, item, schema.items)
      if (nested) return nested
    }
    return undefined
  }
  if (isRecord(value) && isRecord(schema.properties)) {
    const properties = schema.properties
    for (const [key, item] of Object.entries(value)) {
      if (!(key in properties)) {
        if (schema.additionalProperties === false) return `${path ? `${path}.` : ''}${key} is not an accepted field`
        continue
      }
      if (isRecord(properties[key]) && typeof properties[key].type === 'string' && !matches(item, properties[key].type)) {
        return `${path ? `${path}.` : ''}${key} expected ${properties[key].type}`
      }
      const nested = boundsError(path ? `${path}.${key}` : key, item, properties[key])
      if (nested) return nested
    }
    for (const key of Array.isArray(schema.required) ? schema.required : []) {
      if (typeof key === 'string' && !(key in value)) return `${path ? `${path}.` : ''}${key} is required`
    }
  }
  return undefined
}

function matches(value: unknown, type: unknown): boolean {
  if (type === 'string') return typeof value === 'string'
  if (type === 'integer') return typeof value === 'number' && Number.isInteger(value)
  if (type === 'number') return typeof value === 'number' && Number.isFinite(value)
  if (type === 'boolean') return typeof value === 'boolean'
  if (type === 'array') return Array.isArray(value)
  if (type === 'object') return isRecord(value)
  return true
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
