// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isRecord, type HubActor } from '../hub/protocol.js'
import { arr, clip, num, rec, str } from './parse.js'
import { NO_ACTIVITY, activityAfter, taskStatusOf, toolBaseName, toolFamily, type TaskActivity, type TaskStatus, type TodoItem } from './taskTypes.js'

/**
 * Folds daemon-vocabulary events (as relayed by the Hub) into the bounded block
 * list the phone renders. Pure and immutable: every call returns a new tail and
 * never mutates its input. Browser-safe.
 */

export const TAIL_BLOCK_LIMIT = 40
export const OLDER_BLOCK_LIMIT = 400
export const TEXT_LIMIT = 8_000
/** Tool arguments as shown: the shell command verbatim, otherwise pretty JSON. */
export const TOOL_ARGS_LIMIT = 8_000
export const TOOL_OUTPUT_LIMIT = 16_000
/** Each side of an edit diff. */
export const TOOL_DIFF_LIMIT = 8_000
const TODO_LIMIT = 100
const RESOLVED_LIMIT = 50

/** Stable daemon action identity for authoritative history-window reconciliation. */
interface HistoryIdentity { readonly historyActionId?: string }

/** What an edit/write tool call changes in one file. `before` is empty for a whole-file write. */
export interface ToolDiff {
  readonly path: string
  readonly before: string
  readonly after: string
}

export type Block = HistoryIdentity & (
  | { readonly kind: 'user'; readonly id: string; readonly text: string; readonly images: number; readonly author?: string; readonly steer: boolean }
  | { readonly kind: 'assistant'; readonly id: string; readonly text: string; readonly streaming: boolean }
  | { readonly kind: 'thinking'; readonly id: string; readonly text: string; readonly streaming: boolean }
  | {
    readonly kind: 'tool'; readonly id: string; readonly callId: string; readonly name: string; readonly summary: string; readonly state: 'working' | 'done' | 'error'
    readonly output?: string
    /** The full command for shell tools, otherwise pretty JSON of the arguments (diff fields left out when `diff` is set). */
    readonly args?: string
    readonly diff?: ToolDiff
    readonly durationMs?: number
  }
  | { readonly kind: 'notice'; readonly id: string; readonly text: string; readonly error: boolean }
)

export interface QuestionItem {
  readonly id: string
  readonly question: string
  readonly options: readonly string[]
  readonly allowFreeForm: boolean
}

export interface PendingInteraction {
  readonly id: string
  readonly kind: 'permission' | 'question'
  readonly toolName: string
  readonly title: string
  readonly description: string
  /** The real command, when the role may see it. */
  readonly command?: string
  readonly inputs: readonly (readonly [string, string])[]
  readonly workspace?: string
  /** True when the Hub withheld the arguments (viewer role). */
  readonly detailsHidden: boolean
  readonly questions: readonly QuestionItem[]
  readonly requestedBy?: string
}

export interface Resolution {
  readonly id: string
  readonly decision?: string
  readonly by?: string
}

export interface SessionTail {
  readonly blocks: readonly Block[]
  /** Older blocks were dropped from the tail; `hub.history` can fetch them. */
  readonly truncated: boolean
  readonly pending: readonly PendingInteraction[]
  readonly resolved: Readonly<Record<string, Resolution>>
  readonly activity: TaskActivity
  readonly status?: TaskStatus
  readonly title?: string
  readonly queuedSteers: number
  readonly effective: { readonly model?: string; readonly effort?: string; readonly goal?: string; readonly mode?: string }
  /** The latest TodoWriteTool / todo_write call's list, live or from history. */
  readonly todos?: readonly TodoItem[]
  /** The tool call that produced `todos`, so a newer live list is not replaced by an older history one. */
  readonly todosCallId?: string
}

export const EMPTY_TAIL: SessionTail = Object.freeze({
  blocks: [],
  truncated: false,
  pending: [],
  resolved: {},
  activity: NO_ACTIVITY,
  queuedSteers: 0,
  effective: {},
})

export interface FoldInput {
  readonly id: string
  readonly type: string
  readonly payload: Readonly<Record<string, unknown>>
  readonly actor?: HubActor
}

function actorName(actor: HubActor | undefined): string | undefined {
  if (!actor) return undefined
  if (actor.principal_kind === 'agent') return actor.requested_by ? `${actor.display_name} for ${actor.requested_by}` : actor.display_name
  return actor.display_name
}

/**
 * Drop the model-only `<turn-context>` block Xerxes puts in front of a prompt
 * (goal state, refreshed memory). The person sees only what they typed.
 */
export function withoutTurnContext(text: string): string {
  return text.replace(/^\s*<turn-context>[\s\S]*?<\/turn-context>\s*/, '')
}

/** Text and image count of a `turn_begin.user_input` (string or Part[]). */
export function userInputOf(value: unknown): { readonly text: string; readonly images: number } {
  if (typeof value === 'string') return { text: clip(withoutTurnContext(value), TEXT_LIMIT), images: 0 }
  let images = 0
  const parts: string[] = []
  for (const part of arr(value)) {
    const item = rec(part)
    if (item.type === 'text' && typeof item.text === 'string') parts.push(item.text)
    else if (item.type === 'image_url') images += 1
  }
  return { text: clip(withoutTurnContext(parts.join('\n')), TEXT_LIMIT), images }
}

const ARG_KEYS = ['command', 'cmd', 'path', 'file_path', 'pattern', 'query', 'url', 'text'] as const

/** One short line describing a tool call's arguments. */
export function toolSummary(rawArguments: unknown): string {
  let args: Readonly<Record<string, unknown>> = {}
  if (typeof rawArguments === 'string') {
    try { args = rec(JSON.parse(rawArguments) as unknown) } catch { return clip(rawArguments.replace(/\s+/g, ' ').trim(), 120) }
  } else args = rec(rawArguments)
  for (const key of ARG_KEYS) {
    const value = args[key]
    if (typeof value === 'string' && value.trim()) return clip(value.replace(/\s+/g, ' ').trim(), 120)
  }
  return ''
}

function argumentsOf(raw: unknown): { readonly record?: Readonly<Record<string, unknown>>; readonly text?: string } {
  if (typeof raw === 'string') {
    if (raw.trim() === '') return {}
    try {
      const parsed = JSON.parse(raw) as unknown
      return isRecord(parsed) ? { record: parsed } : { text: raw }
    } catch { return { text: raw } }
  }
  if (isRecord(raw)) return { record: raw }
  if (raw === undefined || raw === null) return {}
  return { text: JSON.stringify(raw) ?? '' }
}

function shellWord(word: string): string {
  return /^[\w@%+=:,./-]+$/.test(word) ? word : `'${word.replace(/'/g, `'\\''`)}'`
}

/** The command line of a shell tool: `command` verbatim, or `cmd` plus its argv (`exec_command`). */
function commandLine(args: Readonly<Record<string, unknown>>): string | undefined {
  const argv = arr(args.args).filter((item): item is string => typeof item === 'string')
  if (typeof args.command === 'string' && args.command.trim()) return args.command
  if (typeof args.cmd === 'string' && args.cmd.trim()) return [args.cmd, ...argv.map(shellWord)].join(' ')
  const list = Array.isArray(args.command) ? args.command : Array.isArray(args.cmd) ? args.cmd : undefined
  if (list && list.length > 0 && list.every(item => typeof item === 'string')) return (list as string[]).map(shellWord).join(' ')
  return undefined
}

const DIFF_FIELDS: Readonly<Record<string, readonly string[]>> = {
  edit: ['old_string', 'new_string'],
  file_edit: ['old_string', 'new_string'],
  multi_edit: ['edits'],
  write: ['content'],
  write_file: ['content'],
}

function diffOf(base: string, args: Readonly<Record<string, unknown>>): ToolDiff | undefined {
  const path = str(args.file_path) ?? str(args.path) ?? str(args.notebook_path)
  if (!path) return undefined
  const text = (value: unknown): string | undefined => (typeof value === 'string' ? value : undefined)
  if (base === 'write' || base === 'write_file') {
    const content = text(args.content)
    return content === undefined ? undefined : { path, before: '', after: clip(content, TOOL_DIFF_LIMIT) }
  }
  if (base === 'multi_edit') {
    const edits = arr(args.edits).map(rec).filter(edit => text(edit.old_string) !== undefined || text(edit.new_string) !== undefined)
    if (edits.length === 0) return undefined
    return {
      path,
      before: clip(edits.map(edit => text(edit.old_string) ?? '').join('\n⋯\n'), TOOL_DIFF_LIMIT),
      after: clip(edits.map(edit => text(edit.new_string) ?? '').join('\n⋯\n'), TOOL_DIFF_LIMIT),
    }
  }
  const before = text(args.old_string)
  const after = text(args.new_string)
  if (before === undefined && after === undefined) return undefined
  return { path, before: clip(before ?? '', TOOL_DIFF_LIMIT), after: clip(after ?? '', TOOL_DIFF_LIMIT) }
}

export interface ToolDetails {
  readonly args?: string
  readonly diff?: ToolDiff
}

/**
 * What a tool call did, for the expandable row: the shell command verbatim,
 * an edit as a diff, otherwise the arguments as pretty JSON. Bounded.
 */
export function toolDetails(name: string, rawArguments: unknown): ToolDetails {
  const { record, text } = argumentsOf(rawArguments)
  if (record === undefined) return text === undefined ? {} : { args: clip(text, TOOL_ARGS_LIMIT) }
  if (toolFamily(name) === 'command') {
    const command = commandLine(record)
    if (command !== undefined) return { args: clip(command, TOOL_ARGS_LIMIT) }
  }
  const base = toolBaseName(name)
  const fields = DIFF_FIELDS[base]
  const diff = fields ? diffOf(base, record) : undefined
  const shown = diff && fields ? Object.fromEntries(Object.entries(record).filter(([key]) => !fields.includes(key))) : record
  const json = Object.keys(shown).length === 0 ? undefined : JSON.stringify(shown, null, 2)
  return { ...(json ? { args: clip(json, TOOL_ARGS_LIMIT) } : {}), ...(diff ? { diff } : {}) }
}

const TODO_STATUSES: ReadonlySet<string> = new Set(['pending', 'in_progress', 'completed'])

/** The todo list a TodoWriteTool / todo_write call sets, or undefined for any other call or unreadable arguments. */
export function todosOf(name: string, rawArguments: unknown): readonly TodoItem[] | undefined {
  if (!/^todo_?write$/.test(toolBaseName(name))) return undefined
  const { record } = argumentsOf(rawArguments)
  if (record === undefined || !Array.isArray(record.todos)) return undefined
  return record.todos.slice(0, TODO_LIMIT).flatMap((raw): TodoItem[] => {
    const item = rec(raw)
    const content = str(item.content) ?? str(item.text) ?? str(item.title)
    if (!content) return []
    const status = typeof item.status === 'string' && TODO_STATUSES.has(item.status) ? (item.status as TodoItem['status']) : 'pending'
    return [{ content: clip(content, 500), status }]
  })
}

/** Tool output as shown: strings verbatim, anything else as JSON. Bounded. */
export function toolOutput(raw: unknown): string | undefined {
  if (raw === undefined || raw === null) return undefined
  return typeof raw === 'string' ? clip(raw, TOOL_OUTPUT_LIMIT) : clip(JSON.stringify(raw) ?? '', TOOL_OUTPUT_LIMIT)
}

function inputRows(args: Readonly<Record<string, unknown>>): (readonly [string, string])[] {
  const rows: (readonly [string, string])[] = []
  for (const [key, value] of Object.entries(args)) {
    if (rows.length >= 8) break
    const text = typeof value === 'string' ? value : JSON.stringify(value)
    if (text === undefined || key === 'command') continue
    rows.push([key, clip(text, 600)])
  }
  return rows
}

function parseArgs(value: unknown): Readonly<Record<string, unknown>> {
  if (typeof value === 'string') {
    try { return rec(JSON.parse(value) as unknown) } catch { return {} }
  }
  return rec(value)
}

const PERMISSION_REQUESTS: ReadonlySet<string> = new Set(['approval_request', 'permission_request'])
const RESOLUTIONS: ReadonlySet<string> = new Set([
  'approval_response', 'permission_response', 'permission_resolved', 'question_response', 'question_resolved', 'interaction_resolved',
])

function permissionOf(input: FoldInput): PendingInteraction | undefined {
  const p = input.payload
  const id = str(p.request_id) ?? str(p.id)
  if (!id) return undefined
  const args = parseArgs(p.arguments ?? p.inputs ?? p.input)
  const toolName = str(p.tool_name) ?? str(p.tool) ?? str(p.action) ?? str(p.name) ?? 'tool'
  const command = str(p.command) ?? (typeof args.command === 'string' ? args.command : undefined)
  const inputs = inputRows(args)
  const workspace = str(p.cwd) ?? str(p.workspace) ?? str(p.project_dir)
  const requestedBy = actorName(input.actor)
  const description = str(p.description) ?? ''
  return {
    id,
    kind: 'permission',
    toolName,
    title: str(p.title) ?? 'Allow this action?',
    description: clip(description, 600),
    ...(command ? { command: clip(command, 2000) } : {}),
    inputs,
    ...(workspace ? { workspace } : {}),
    // Only the Hub's own flags mean "hidden for your role". A real request may have `{}` inputs and
    // no description; guessing from missing fields hid Allow from the person who must answer it.
    detailsHidden: p.redacted === true || p.details_hidden === true,
    questions: [],
    ...(requestedBy ? { requestedBy } : {}),
  }
}

function questionOf(input: FoldInput): PendingInteraction | undefined {
  const p = input.payload
  const id = str(p.request_id) ?? str(p.id)
  if (!id) return undefined
  const questions: QuestionItem[] = []
  for (const raw of arr(p.questions)) {
    const q = rec(raw)
    const qid = str(q.id)
    const question = str(q.question)
    if (!qid || !question) continue
    questions.push({
      id: qid,
      question: clip(question, 1000),
      options: arr(q.options).filter((o): o is string => typeof o === 'string').slice(0, 12),
      allowFreeForm: q.allow_free_form === true,
    })
  }
  return {
    id,
    kind: 'question',
    toolName: 'question',
    title: 'Xerxes has a question',
    description: '',
    inputs: [],
    detailsHidden: false,
    questions,
    ...(actorName(input.actor) ? { requestedBy: actorName(input.actor)! } : {}),
  }
}

function pushBlock(blocks: readonly Block[], block: Block): { readonly blocks: readonly Block[]; readonly dropped: boolean } {
  const next = [...blocks, block]
  if (next.length <= TAIL_BLOCK_LIMIT) return { blocks: next, dropped: false }
  return { blocks: next.slice(next.length - TAIL_BLOCK_LIMIT), dropped: true }
}

function appendStreaming(tail: SessionTail, kind: 'assistant' | 'thinking', id: string, delta: string): SessionTail {
  const last = tail.blocks.at(-1)
  if (last && last.kind === kind && last.streaming) {
    const text = clip(last.text + delta, TEXT_LIMIT)
    return { ...tail, blocks: [...tail.blocks.slice(0, -1), { ...last, text }] }
  }
  // A new block of the other kind (thinking then text, or back) ends the one streaming before it.
  const added = pushBlock(settleStreaming(tail.blocks), { kind, id, text: clip(delta, TEXT_LIMIT), streaming: true })
  return { ...tail, blocks: added.blocks, truncated: tail.truncated || added.dropped }
}

function settleStreaming(blocks: readonly Block[]): readonly Block[] {
  if (!blocks.some(block => (block.kind === 'assistant' || block.kind === 'thinking') && block.streaming)) return blocks
  return blocks.map(block => ((block.kind === 'assistant' || block.kind === 'thinking') && block.streaming ? { ...block, streaming: false } : block))
}

function resolve(tail: SessionTail, id: string, decision: string | undefined, by: string | undefined): SessionTail {
  const resolved: Record<string, Resolution> = { ...tail.resolved, [id]: { id, ...(decision ? { decision } : {}), ...(by ? { by } : {}) } }
  const keys = Object.keys(resolved)
  for (const key of keys.slice(0, Math.max(0, keys.length - RESOLVED_LIMIT))) delete resolved[key]
  return { ...tail, pending: tail.pending.filter(item => item.id !== id), resolved }
}

/** Fold one event into the tail. Unknown event types leave the tail unchanged. */
export function foldSessionEvent(tail: SessionTail, input: FoldInput): SessionTail {
  const { type, payload } = input
  const activity = activityAfter(type, payload, tail.activity)
  const base: SessionTail = activity === tail.activity ? tail : { ...tail, activity }
  switch (type) {
    case 'turn_begin': {
      const { text, images } = userInputOf(payload.user_input)
      const author = actorName(input.actor)
      const added = pushBlock(settleStreaming(base.blocks), { kind: 'user', id: input.id, text, images, steer: false, ...(author ? { author } : {}) })
      return { ...base, blocks: added.blocks, truncated: base.truncated || added.dropped, status: 'working' }
    }
    case 'steer_input': {
      const added = pushBlock(base.blocks, { kind: 'user', id: input.id, text: clip(str(payload.content) ?? '', TEXT_LIMIT), images: 0, steer: true, ...(actorName(input.actor) ? { author: actorName(input.actor)! } : {}) })
      return { ...base, blocks: added.blocks, truncated: base.truncated || added.dropped, queuedSteers: base.queuedSteers + 1 }
    }
    case 'steer_applied': return { ...base, queuedSteers: 0 }
    case 'text_part': {
      const text = typeof payload.text === 'string' ? payload.text : ''
      return text ? appendStreaming(base, 'assistant', input.id, text) : base
    }
    case 'think_part': {
      const text = typeof payload.think === 'string' ? payload.think : ''
      return text ? appendStreaming(base, 'thinking', input.id, text) : base
    }
    case 'tool_call': {
      const callId = str(payload.tool_call_id) ?? str(payload.id) ?? input.id
      if (base.blocks.some(block => block.kind === 'tool' && block.callId === callId)) return base
      const name = str(payload.name) ?? 'tool'
      const rawArgs = payload.arguments ?? payload.inputs ?? payload.input
      const details = toolDetails(name, rawArgs)
      const added = pushBlock(settleStreaming(base.blocks), { kind: 'tool', id: input.id, callId, name, summary: toolSummary(rawArgs), state: 'working', ...details })
      const todos = todosOf(name, rawArgs)
      return { ...base, blocks: added.blocks, truncated: base.truncated || added.dropped, ...(todos ? { todos, todosCallId: callId } : {}) }
    }
    case 'tool_result': {
      const callId = str(payload.tool_call_id)
      const raw = payload.return_value ?? payload.result ?? payload.output ?? payload.error
      const failed = payload.is_error === true || payload.error != null || payload.permitted === false || (typeof raw === 'string' && raw.startsWith('Tool execution failed:'))
      const output = toolOutput(raw)
      const durationMs = num(payload.duration_ms)
      let changed = false
      const blocks = base.blocks.map(block => {
        if (changed || block.kind !== 'tool' || block.callId !== callId || block.state !== 'working') return block
        changed = true
        return { ...block, state: failed ? ('error' as const) : ('done' as const), ...(output ? { output } : {}), ...(durationMs === undefined ? {} : { durationMs }) }
      })
      return changed ? { ...base, blocks } : base
    }
    case 'turn_end': {
      const failed = payload.error !== undefined && payload.error !== null && payload.error !== false
      return { ...base, blocks: settleStreaming(base.blocks), pending: [], status: failed ? 'failed' : 'idle', queuedSteers: 0 }
    }
    case 'turn_error': case 'error': {
      const added = pushBlock(settleStreaming(base.blocks), { kind: 'notice', id: input.id, text: clip(str(payload.message) ?? str(payload.error) ?? 'Something went wrong.', 600), error: true })
      return { ...base, blocks: added.blocks, truncated: base.truncated || added.dropped, status: 'failed' }
    }
    case 'notification': {
      const text = str(payload.message) ?? str(payload.body) ?? str(payload.title)
      if (!text) return base
      const added = pushBlock(base.blocks, { kind: 'notice', id: input.id, text: clip(text, 600), error: payload.level === 'error' || payload.severity === 'error' })
      return { ...base, blocks: added.blocks, truncated: base.truncated || added.dropped }
    }
    case 'session_title': return str(payload.title) ? { ...base, title: str(payload.title)! } : base
    case 'status_update': case 'session_updated': case 'session_summary': case 'init_done': {
      const summary = type === 'status_update' || type === 'init_done' ? payload : rec(payload.session ?? payload)
      const goalRecord = rec(summary.goal)
      const goal = str(goalRecord.objective) ?? str(goalRecord.summary) ?? (typeof summary.goal === 'string' ? summary.goal : undefined)
      const model = str(summary.model)
      const effort = str(summary.reasoning_effort) ?? str(summary.effort)
      const mode = str(summary.mode)
      const title = str(summary.title)
      const next: SessionTail = {
        ...base,
        effective: {
          ...base.effective,
          ...(model ? { model } : {}),
          ...(effort ? { effort } : {}),
          ...(goal !== undefined ? { goal } : {}),
          ...(mode ? { mode } : {}),
        },
        ...(title ? { title } : {}),
        ...(summary.status !== undefined && type !== 'status_update' ? { status: taskStatusOf(summary.status) } : {}),
      }
      return next
    }
    default:
      break
  }
  if (PERMISSION_REQUESTS.has(type)) {
    const item = permissionOf(input)
    if (!item || base.pending.some(existing => existing.id === item.id) || base.resolved[item.id]) return base
    return { ...base, pending: [...base.pending, item], status: 'working' }
  }
  if (type === 'question_request') {
    const item = questionOf(input)
    if (!item || base.pending.some(existing => existing.id === item.id) || base.resolved[item.id]) return base
    return { ...base, pending: [...base.pending, item] }
  }
  if (RESOLUTIONS.has(type)) {
    const id = str(payload.request_id) ?? str(payload.id)
    if (!id) return base
    const resolvedBy = rec(payload.resolved_by)
    const by = actorName(input.actor) ?? str(resolvedBy.display_name) ?? (typeof payload.resolved_by === 'string' ? payload.resolved_by : undefined)
    const next = resolve(base, id, str(payload.response) ?? str(payload.decision), by)
    // The turn goes on once nothing waits on a person any more.
    return next.pending.length === 0 && next.status === 'needs_you' ? { ...next, status: 'working' } : next
  }
  return base
}

export function foldAll(tail: SessionTail, inputs: readonly FoldInput[]): SessionTail {
  let current = tail
  for (const input of inputs) current = foldSessionEvent(current, input)
  return current
}

/** Fold a page of history into standalone blocks (no pending, no live state). */
export function blocksFromHistory(inputs: readonly FoldInput[]): readonly Block[] {
  return foldAll(EMPTY_TAIL, inputs).blocks
}
