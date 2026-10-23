// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ToolRegistry, type ToolExecutionContext, type ToolHandler } from '../../executors/toolRegistry.js'
import type { JsonObject } from '../../types/toolCalls.js'
import { resolveSessionKey, type CommandBackend } from '../commandRouter.js'
import { isRecord, utf8ByteLength } from '../protocol.js'
import type { HubWorkspaceView } from '../server.js'
import { ACTION_TOOLS, SupervisorDenied, isActionTool, type ActionService } from './actions.js'
import type { DigestService } from './digests.js'
import {
  SUPERVISOR_TOOLS, assertSupervisorToolCeiling, validateSupervisorArguments, type SupervisorToolSpec,
} from './schemas.js'
import {
  messageToWire, type ScopeReader, type SupervisorThreads, type UserPrincipal,
} from './threads.js'

/**
 * The supervisor's DEDICATED tool registry: exactly the twelve tools in
 * `schemas.ts` and nothing else. It never contains workspace plugins, MCP,
 * shell or filesystem tools, Workflow, review or ordinary goal tools.
 *
 * Reads are filtered by the requesting thread's ACL (the intersection of the
 * humans' grants and the agent ceiling) and use summary reads only. Mutations
 * never execute here: they create a confirmation card and return
 * `pending_confirmation`. `notify.user` reaches thread participants only.
 */

export const TURN_METADATA_KEY = 'xerxes_supervisor_turn'
export const TAIL_MAX_BYTES = 8 * 1024
export const SEARCH_MAX_BYTES = 8 * 1024

/** Per-turn state handed to every tool through the execution context. */
export interface SupervisorTurnState {
  readonly threadId: string
  readonly requester: UserPrincipal
  readonly reader: ScopeReader
  /** Set when a tool was refused for lack of authority. The runner stops the turn on it. */
  denial: string | undefined
}

function isTurnState(value: unknown): value is SupervisorTurnState {
  return typeof value === 'object' && value !== null && typeof (value as { threadId?: unknown }).threadId === 'string'
    && typeof (value as { reader?: unknown }).reader === 'object'
}

export interface SupervisorPushPort {
  (userId: string, message: { readonly title: string; readonly body: string; readonly url: string }): Promise<unknown>
}

export interface SupervisorToolDeps {
  readonly workspaces: Pick<HubWorkspaceView, 'list' | 'projection' | 'health'>
  readonly backend: Pick<CommandBackend, 'call'>
  readonly digests: Pick<DigestService, 'digest'>
  readonly actions: Pick<ActionService, 'createIntent'>
  readonly threads: Pick<SupervisorThreads, 'append' | 'emit' | 'get' | 'recipients' | 'participants'>
  readonly push?: SupervisorPushPort
  readonly onError?: (error: unknown, context: string) => void
}

function clip(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, Math.max(0, max - 1))}…`
}

function str(args: JsonObject, key: string): string | undefined {
  const value = args[key]
  return typeof value === 'string' ? value : undefined
}

function contentText(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content.map(part => (isRecord(part) && part.type === 'text' && typeof part.text === 'string' ? part.text : '')).join('')
}

/** Last user-visible steps of a history window, newest kept, within a byte ceiling. */
export function renderTail(history: unknown, maxBytes: number): { readonly lines: string[]; readonly truncated: boolean } {
  const lines: string[] = []
  const actions = isRecord(history) && Array.isArray(history.actions) ? history.actions : []
  for (const action of actions) {
    if (!isRecord(action)) continue
    for (const message of Array.isArray(action.messages) ? action.messages : []) {
      if (!isRecord(message)) continue
      const text = contentText(message.content).replace(/\s+/g, ' ').trim()
      if (text === '') continue
      if (message.role === 'user') lines.push(`user: ${clip(text, 500)}`)
      else if (message.role === 'assistant') lines.push(`assistant: ${clip(text, 500)}`)
    }
    for (const execution of Array.isArray(action.executions) ? action.executions : []) {
      if (!isRecord(execution)) continue
      const name = [execution.name, execution.toolName, execution.tool_name].find(value => typeof value === 'string') as string | undefined
      const failed = execution.is_error === true || execution.error !== undefined || execution.status === 'error'
      lines.push(`tool ${clip(name ?? 'tool', 60)}${failed ? ' (failed)' : ''}`)
    }
  }
  let bytes = 2
  let keep = lines.length
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const size = utf8ByteLength(lines[index]!) + 4
    if (bytes + size > maxBytes) { keep = lines.length - 1 - index; break }
    bytes += size
  }
  const kept = lines.slice(lines.length - keep)
  return { lines: kept, truncated: kept.length < lines.length }
}

const CURSOR_PREFIX = 'o1.'
function decodeCursor(cursor: string | undefined): number {
  if (cursor === undefined || !cursor.startsWith(CURSOR_PREFIX)) return 0
  const value = Number(cursor.slice(CURSOR_PREFIX.length))
  return Number.isSafeInteger(value) && value >= 0 ? value : 0
}

export function createSupervisorToolRegistry(deps: SupervisorToolDeps): ToolRegistry {
  const registry = new ToolRegistry({
    onDuplicateRegistration: name => { throw new Error(`duplicate supervisor tool ${name}`) },
  })

  const readable = (turn: SupervisorTurnState, workspaceId: string, sessionId?: string): void => {
    const known = deps.workspaces.list().some(workspace => workspace.id === workspaceId)
    if (!known || !turn.reader.canRead({ workspaceId, ...(sessionId === undefined ? {} : { sessionId }) })) {
      throw new Error('not found, or you cannot read it')
    }
  }

  const poolRead = async (workspaceId: string, method: string, params: Record<string, unknown>, signal?: AbortSignal): Promise<Readonly<Record<string, unknown>>> => {
    const outcome = await deps.backend.call(workspaceId, method, params, signal === undefined ? {} : { signal })
    if (outcome.state !== 'accepted') throw new Error(outcome.error.message)
    return outcome.result
  }

  const handlers: Readonly<Record<string, (args: JsonObject, turn: SupervisorTurnState, signal?: AbortSignal) => Promise<unknown> | unknown>> = {
    'workspace.list': (_args, turn) => ({
      workspaces: deps.workspaces.list()
        .filter(workspace => turn.reader.canRead({ workspaceId: workspace.id }))
        .map(workspace => {
          const health = deps.workspaces.health(workspace.id)
          return { id: workspace.id, label: workspace.label, health: health?.state ?? 'offline', degraded: health?.degraded ?? false }
        }),
    }),

    'tasks.list': (args, turn) => {
      const only = str(args, 'workspace_id')
      if (only !== undefined) readable(turn, only)
      const limit = typeof args.limit === 'number' ? args.limit : 20
      const offset = decodeCursor(str(args, 'cursor'))
      const rows = deps.workspaces.list()
        .filter(workspace => only === undefined || workspace.id === only)
        .flatMap(workspace => Object.values(deps.workspaces.projection(workspace.id)?.sessions ?? {})
          .filter(session => turn.reader.canRead({ workspaceId: workspace.id, sessionId: session.id }))
          .map(session => ({ workspace, session })))
        .sort((a, b) => Date.parse(b.session.updated_at) - Date.parse(a.session.updated_at) || a.session.id.localeCompare(b.session.id))
      const page = rows.slice(offset, offset + limit)
      return {
        tasks: page.map(({ workspace, session }) => ({
          workspace_id: workspace.id,
          task_id: session.id,
          title: clip(session.title, 120),
          status: session.status,
          pending_human_count: session.pending_interaction_count,
          agents: session.agent_counts,
          goal: clip(session.goal_summary, 200),
          updated_at: session.updated_at,
          stale: session.stale,
        })),
        ...(offset + limit < rows.length ? { next_cursor: `${CURSOR_PREFIX}${offset + limit}` } : {}),
      }
    },

    'task.digest': async (args, turn, signal) => {
      const workspaceId = str(args, 'workspace_id')!
      const taskId = str(args, 'task_id')!
      readable(turn, workspaceId, taskId)
      return await deps.digests.digest(workspaceId, taskId, {
        ...(typeof args.max_age_seconds === 'number' ? { maxAgeSeconds: args.max_age_seconds } : {}),
        ...(signal === undefined ? {} : { signal }),
      })
    },

    'task.tail': async (args, turn, signal) => {
      const workspaceId = str(args, 'workspace_id')!
      const taskId = str(args, 'task_id')!
      readable(turn, workspaceId, taskId)
      if (typeof args.after === 'number' && args.after > 0) throw new Error('"after" is not supported yet; omit it to read the latest steps')
      const key = await resolveSessionKey(deps.backend, workspaceId, taskId)
      if (key === undefined) throw new Error('that task is not active in its workspace, so it has no live tail')
      const limit = typeof args.limit === 'number' ? args.limit : 10
      const result = await poolRead(workspaceId, 'session.history', { session_key: key, history_limit: limit }, signal)
      const tail = renderTail(result.history, TAIL_MAX_BYTES)
      return { steps: tail.lines, truncated: tail.truncated }
    },

    'task.search': async (args, turn, signal) => {
      const workspaceId = str(args, 'workspace_id')!
      const taskId = str(args, 'task_id')!
      readable(turn, workspaceId, taskId)
      const limit = typeof args.limit === 'number' ? args.limit : 5
      const result = await poolRead(workspaceId, 'session.search', { query: str(args, 'query')!, session_id: taskId, limit }, signal)
      const hits = Array.isArray(result.hits) ? result.hits : []
      const kept: unknown[] = []
      let bytes = 2
      let truncated = false
      for (const hit of hits) {
        // Defence in depth: the daemon scopes the search, and anything from another session is dropped.
        if (isRecord(hit) && typeof hit.session_id === 'string' && hit.session_id !== taskId) continue
        const text = JSON.stringify(hit)
        const size = utf8ByteLength(text) + 1
        if (kept.length >= limit || bytes + size > SEARCH_MAX_BYTES) { truncated = true; break }
        kept.push(hit)
        bytes += size
      }
      return { hits: kept, truncated }
    },

    'notify.user': async (args, turn) => {
      const thread = deps.threads.get(turn.threadId)
      if (!thread) throw new Error('this thread no longer exists')
      const allowed = new Set(deps.threads.recipients(thread))
      const people = thread.kind === 'private'
        ? [{ id: turn.requester.id, name: turn.requester.displayName }]
        : deps.threads.participants(thread.id).map(person => ({ id: person.id, name: person.displayName }))
      const wanted = Array.isArray(args.recipients) ? args.recipients.filter((item): item is string => typeof item === 'string') : []
      const ids: string[] = []
      const skipped: string[] = []
      for (const name of wanted) {
        const match = people.find(person => person.id === name) ?? people.find(person => person.name.toLowerCase() === name.toLowerCase())
        if (match !== undefined && allowed.has(match.id)) { if (!ids.includes(match.id)) ids.push(match.id) } else skipped.push(name)
      }
      if (ids.length === 0) throw new Error('none of the recipients take part in this thread; notices only reach thread participants')
      const severity = str(args, 'severity') === 'attention' ? 'attention' : 'info'
      const { message } = deps.threads.append({
        threadId: thread.id, role: 'system', text: `XERXES notice (${severity}): ${str(args, 'text')!}`, audience: ids,
      })
      deps.threads.emit(thread.id, 'supervisor.thread_event', { message: messageToWire(message) }, ids)
      let pushed = 0
      if (severity === 'attention' && deps.push !== undefined) {
        for (const id of ids) {
          try {
            await deps.push(id, { title: 'XERXES', body: 'XERXES has a message for you.', url: '/xerxes' })
            pushed += 1
          } catch (error) {
            deps.onError?.(error, 'supervisor notify push')
          }
        }
      }
      return { delivered_to: ids, ...(skipped.length > 0 ? { skipped_not_in_thread: skipped } : {}), pushed }
    },
  }

  const mutation = (spec: SupervisorToolSpec): ((args: JsonObject, turn: SupervisorTurnState) => Promise<unknown>) => async (args, turn) => {
    if (!isActionTool(spec.name)) throw new Error(`${spec.name} is not an action tool`)
    try {
      const action = await deps.actions.createIntent({ threadId: turn.threadId, tool: spec.name, args, requester: turn.requester, reader: turn.reader })
      return {
        state: 'pending_confirmation',
        action_id: action.id,
        summary: action.summary,
        expires_at: new Date(action.expiresAt).toISOString(),
        note: 'Nothing has run. A person in this thread must tap Confirm before it does.',
      }
    } catch (error) {
      if (error instanceof SupervisorDenied) {
        turn.denial = error.message
        return { ok: false, denied: true, message: error.message }
      }
      throw error
    }
  }

  for (const spec of SUPERVISOR_TOOLS.values()) {
    const run = spec.gate === 'confirm' ? mutation(spec) : handlers[spec.name]
    if (run === undefined) throw new Error(`no implementation for supervisor tool ${spec.name}`)
    const handler: ToolHandler = async (inputs: JsonObject, context: ToolExecutionContext, signal?: AbortSignal) => {
      const turn = context.metadata[TURN_METADATA_KEY]
      if (!isTurnState(turn)) throw new Error('supervisor tools only run inside a supervisor turn')
      const checked = validateSupervisorArguments(spec.name, inputs)
      if (!checked.ok) throw new Error(checked.error)
      return await run(checked.args, turn, signal)
    }
    registry.register(
      { type: 'function', function: { name: spec.providerName, description: spec.description, parameters: spec.parameters } },
      handler,
      'default',
      {
        readOnly: spec.gate === 'read', concurrencySafe: spec.gate === 'read', defer: false, destructive: false, openWorld: false,
        interruptBehavior: 'cancel', maxResultBytes: 16 * 1024,
      },
    )
  }
  assertSupervisorToolCeiling(registry.definitions().map(definition => definition.function.name))
  if (registry.definitions().length !== SUPERVISOR_TOOLS.size) throw new Error('the supervisor registry must hold exactly the supervisor tools')
  void ACTION_TOOLS
  return registry
}
