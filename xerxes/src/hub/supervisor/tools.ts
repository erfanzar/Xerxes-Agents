// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ToolRegistry, type ToolExecutionContext, type ToolHandler } from '../../executors/toolRegistry.js'
import type { JsonObject } from '../../types/toolCalls.js'
import { resolveSessionKey, type CommandBackend } from '../commandRouter.js'
import { isRecord, utf8ByteLength } from '../protocol.js'
import type { HubWorkspaceView } from '../server.js'
import { ACTION_TOOLS, SupervisorDenied, isActionTool, type ActionService } from './actions.js'
import { displayOption, type PendingInteractions } from './autopilot.js'
import type { DigestService } from './digests.js'
import {
  SUPERVISOR_TOOLS, assertSupervisorToolCeiling, validateSupervisorArguments, type SupervisorToolSpec,
} from './schemas.js'
import {
  messageToWire, type ScopeReader, type SupervisorThreads, type UserPrincipal,
} from './threads.js'

/**
 * The supervisor's DEDICATED tool registry: exactly the fourteen tools in
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
  /** Present when the Hub started this turn by itself (XERXES Autopilot or Watch). */
  readonly trigger?: 'autopilot' | 'watch'
  /** The one task a Hub wake-up is about; only actions on it may run without a Confirm tap. */
  readonly target?: { readonly workspaceId: string; readonly sessionId: string }
  /** Whether a user message by anyone but the requester is in this turn's context. Absent counts as true. */
  readonly othersInContext?: boolean
}

function isTurnState(value: unknown): value is SupervisorTurnState {
  return typeof value === 'object' && value !== null && typeof (value as { threadId?: unknown }).threadId === 'string'
    && typeof (value as { reader?: unknown }).reader === 'object'
}

export interface SupervisorPushPort {
  (userId: string, message: { readonly title: string; readonly body: string; readonly url: string }): Promise<unknown>
}

/** XERXES Autopilot hooks for the `task.pending` and `question.answer` tools. Absent: both refuse. */
export interface SupervisorAutopilotPort {
  /** What the task waits on, from the Hub journal. Permission details are never included. */
  pending(workspaceId: string, sessionId: string): PendingInteractions | undefined
  /** Answer a waiting question on the owner's behalf; throws SupervisorDenied when Autopilot is not active. */
  answer(input: {
    readonly threadId: string
    readonly workspaceId: string
    readonly sessionId: string
    readonly requestId: string
    readonly answers: Readonly<Record<string, string>>
    readonly requester: UserPrincipal
    /** The thread's shared read scope (the same check createIntent applies). */
    readonly reader: ScopeReader
    readonly trigger: boolean
    readonly othersInContext: boolean
    /** The task a Hub wake-up is about; a wake may answer questions on that task only. */
    readonly target?: { readonly workspaceId: string; readonly sessionId: string }
  }): Promise<Record<string, unknown>>
}

export interface SupervisorToolDeps {
  readonly workspaces: Pick<HubWorkspaceView, 'list' | 'projection' | 'health'>
  readonly backend: Pick<CommandBackend, 'call'>
  readonly digests: Pick<DigestService, 'digest'>
  readonly actions: Pick<ActionService, 'createIntent'>
  readonly threads: Pick<SupervisorThreads, 'append' | 'emit' | 'get' | 'recipients' | 'participants'>
  readonly push?: SupervisorPushPort
  readonly autopilot?: SupervisorAutopilotPort
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

const ARGUMENT_FIELDS: ReadonlySet<string> = new Set(['arguments', 'input', 'inputs', 'tool_input', 'params'])

/** A copy of a search hit without argument-like fields, at any depth (bounded). */
export function stripArgumentFields(value: unknown, depth = 0): unknown {
  if (depth > 8) return undefined
  if (Array.isArray(value)) return value.map(item => stripArgumentFields(item, depth + 1))
  if (!isRecord(value)) return value
  const out: Record<string, unknown> = {}
  for (const [key, item] of Object.entries(value)) {
    if (ARGUMENT_FIELDS.has(key)) continue
    out[key] = stripArgumentFields(item, depth + 1)
  }
  return out
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
        // Tool or permission arguments never reach the model, even through search.
        const safe = stripArgumentFields(hit)
        const text = JSON.stringify(safe)
        const size = utf8ByteLength(text) + 1
        if (kept.length >= limit || bytes + size > SEARCH_MAX_BYTES) { truncated = true; break }
        kept.push(safe)
        bytes += size
      }
      return { hits: kept, truncated }
    },

    'task.pending': (args, turn) => {
      const workspaceId = str(args, 'workspace_id')!
      const taskId = str(args, 'task_id')!
      readable(turn, workspaceId, taskId)
      const count = deps.workspaces.projection(workspaceId)?.sessions[taskId]?.pending_interaction_count ?? 0
      const found = deps.autopilot?.pending(workspaceId, taskId)
      if (found === undefined) {
        return { questions: [], permission_requests_waiting: 0, waiting_total: count, note: 'Waiting items cannot be read on this Hub.' }
      }
      return {
        questions: found.questions.map(question => ({
          request_id: question.request_id,
          questions: question.questions.map(item => ({
            id: item.id,
            question: item.question,
            // Shown clipped; answer_question accepts the full text or exactly this shown form.
            options: item.options.map(displayOption),
            // The task marks its suggestion with "(Recommended)"; otherwise there is none.
            recommended: ((option: string | undefined) => (option === undefined ? null : displayOption(option)))(
              item.options.find(option => /\brecommended\b/i.test(option))),
            free_form_allowed: item.allow_free_form,
          })),
        })),
        // Count only: a permission's command and arguments never reach the supervisor.
        permission_requests_waiting: found.permissionIds.length,
        ...(found.permissionIds.length > 0 ? { permission_note: 'A permission request is waiting. Only a person can answer it; tell the owner.' } : {}),
        waiting_total: count,
        ...(found.complete ? {} : { note: 'The journal window is partial; this list may miss older items.' }),
      }
    },

    'question.answer': async (args, turn) => {
      const workspaceId = str(args, 'workspace_id')!
      const taskId = str(args, 'task_id')!
      readable(turn, workspaceId, taskId)
      const raw = isRecord(args.answers) ? args.answers : {}
      const entries = Object.entries(raw)
      if (entries.length === 0 || entries.length > 10) throw new Error('answers must map 1 to 10 question ids to answer text')
      const answers: Record<string, string> = {}
      for (const [key, value] of entries) {
        if (key === '' || key.length > 200 || typeof value !== 'string' || value.trim() === '' || [...value].length > 2000) {
          throw new Error('each answer must be non-empty text of at most 2000 characters under a question id')
        }
        answers[key] = value
      }
      if (deps.autopilot === undefined) {
        throw new SupervisorDenied('Autopilot is not available on this Hub, so XERXES cannot answer questions. Tell the owner.')
      }
      try {
        return await deps.autopilot.answer({
          threadId: turn.threadId, workspaceId, sessionId: taskId, requestId: str(args, 'request_id')!, answers, requester: turn.requester,
          // Any Hub-started turn (Autopilot or Watch) is a wake turn, never an "owner alone" conversation.
          reader: turn.reader, trigger: turn.trigger !== undefined, othersInContext: turn.othersInContext ?? true,
          ...(turn.target === undefined ? {} : { target: turn.target }),
        })
      } catch (error) {
        if (error instanceof SupervisorDenied) {
          turn.denial = error.message
          return { ok: false, denied: true, message: error.message }
        }
        throw error
      }
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
      const action = await deps.actions.createIntent({
        threadId: turn.threadId, tool: spec.name, args, requester: turn.requester, reader: turn.reader,
        turn: {
          ...(turn.trigger === undefined ? {} : { trigger: turn.trigger }), othersInContext: turn.othersInContext ?? true,
          ...(turn.target === undefined ? {} : { target: turn.target }),
        },
      })
      if (action.auto) {
        // XERXES Autopilot ran it at once for the owner: report what really happened.
        const state = action.state === 'accepted' || action.state === 'rejected' ? action.state : 'unknown'
        return {
          state,
          action_id: action.id,
          summary: action.summary,
          ...(action.confirmedByName === undefined ? {} : { on_behalf_of: action.confirmedByName }),
          ...(action.outcome === undefined ? {} : { outcome: action.outcome }),
          ...(action.errorMessage === undefined ? {} : { message: action.errorMessage }),
          note: state === 'accepted' ? 'Done by Autopilot. The owner was told.'
            : state === 'rejected' ? 'Not done. The owner was told. Do not retry blindly.'
            : 'The outcome is unknown. Do not retry it; the owner was told.',
        }
      }
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
