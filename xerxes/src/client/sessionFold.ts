// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubActor } from '../hub/protocol.js'
import { arr, clip, rec, str } from './parse.js'
import { NO_ACTIVITY, activityAfter, taskStatusOf, type TaskActivity, type TaskStatus } from './taskTypes.js'

/**
 * Folds daemon-vocabulary events (as relayed by the Hub) into the bounded block
 * list the phone renders. Pure and immutable: every call returns a new tail and
 * never mutates its input. Browser-safe.
 */

export const TAIL_BLOCK_LIMIT = 40
export const OLDER_BLOCK_LIMIT = 400
const TEXT_LIMIT = 8_000
const OUTPUT_LIMIT = 600
const RESOLVED_LIMIT = 50

/** Stable daemon action identity for authoritative history-window reconciliation. */
interface HistoryIdentity { readonly historyActionId?: string }

export type Block = HistoryIdentity & (
  | { readonly kind: 'user'; readonly id: string; readonly text: string; readonly images: number; readonly author?: string; readonly steer: boolean }
  | { readonly kind: 'assistant'; readonly id: string; readonly text: string; readonly streaming: boolean }
  | { readonly kind: 'thinking'; readonly id: string; readonly text: string; readonly streaming: boolean }
  | { readonly kind: 'tool'; readonly id: string; readonly callId: string; readonly name: string; readonly summary: string; readonly state: 'working' | 'done' | 'error'; readonly output?: string }
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
    detailsHidden: command === undefined && inputs.length === 0 && p.details_hidden !== false && (p.details_hidden === true || description === ''),
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
  const added = pushBlock(tail.blocks, { kind, id, text: clip(delta, TEXT_LIMIT), streaming: true })
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
      const added = pushBlock(settleStreaming(base.blocks), { kind: 'tool', id: input.id, callId, name, summary: toolSummary(payload.arguments), state: 'working' })
      return { ...base, blocks: added.blocks, truncated: base.truncated || added.dropped }
    }
    case 'tool_result': {
      const callId = str(payload.tool_call_id)
      const raw = payload.return_value ?? payload.result ?? payload.output ?? payload.error
      const failed = payload.is_error === true || payload.error != null || payload.permitted === false || (typeof raw === 'string' && raw.startsWith('Tool execution failed:'))
      const output = typeof raw === 'string' ? clip(raw, OUTPUT_LIMIT) : raw === undefined ? undefined : clip(JSON.stringify(raw) ?? '', OUTPUT_LIMIT)
      let changed = false
      const blocks = base.blocks.map(block => {
        if (changed || block.kind !== 'tool' || block.callId !== callId || block.state !== 'working') return block
        changed = true
        return { ...block, state: failed ? ('error' as const) : ('done' as const), ...(output ? { output } : {}) }
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
    return resolve(base, id, str(payload.response) ?? str(payload.decision), by)
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
