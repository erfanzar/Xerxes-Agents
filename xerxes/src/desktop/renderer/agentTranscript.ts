// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A subagent's conversation as main-transcript blocks.
 *
 * Two sources fold into one list: pages of the child's saved conversation
 * (`subagent.inspect` with `history_limit`, the same action shape as
 * `session.history`) and the live `subagent_event`s that arrived after the
 * latest page was read. The saved page is the source of truth; the live tail
 * only covers the moment until the next page arrives, and it never repeats a
 * tool call the page already holds.
 */

import { BlockBuilder } from './blocks.js'
import { historyBlocks, readHistoryPage, type HistoryPage } from './history.js'
import type { Block, ToolItem } from './types.js'

export type HistoryAction = HistoryPage['actions'][number]

/** Actions requested per page; the daemon may return fewer to fit its byte cap. */
export const AGENT_HISTORY_PAGE = 100
/** Live events kept between two saved pages. */
const MAX_LIVE_EVENTS = 400

const record = (value: unknown): Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
const text = (value: unknown): string => typeof value === 'string' ? value : ''

/** What one `subagent.inspect` reply says about the child's saved conversation. */
export type AgentHistoryRead =
  /** A daemon that predates the field: only the retained summary exists. */
  | { readonly kind: 'unsupported' }
  /** The daemon looked, and there is no readable conversation; `note` says why. */
  | { readonly kind: 'unavailable'; readonly note: string }
  | { readonly kind: 'page'; readonly page: HistoryPage; readonly clipped: boolean }

/** Validate the history part of an inspect reply. Throws on a malformed page. */
export function readAgentHistory(result: Readonly<Record<string, unknown>>): AgentHistoryRead {
  if (!('history' in result) || result.history === undefined) return { kind: 'unsupported' }
  if (result.history === null) return { kind: 'unavailable', note: text(result.history_note) || 'This agent did not save its conversation.' }
  const page = readHistoryPage(result.history)
  if (!page) return { kind: 'unsupported' }
  return { kind: 'page', page, clipped: record(result.history).clipped === true }
}

/** The loaded part of a child's conversation, oldest first, and how to page further back. */
export interface AgentHistoryState {
  readonly actions: readonly HistoryAction[]
  /** Cursor for the page before `actions[0]`; null when nothing older exists. */
  readonly before: string | null
  readonly more: boolean
  /** Some text in a loaded page was shortened by the daemon. */
  readonly clipped: boolean
}

/** Message position an action id starts with (`12:tool:0` → 12). */
function actionIndex(id: string): number {
  const value = Number.parseInt(id, 10)
  return Number.isFinite(value) ? value : -1
}

/**
 * Adopt a fresh newest page. Older actions already loaded (paged in, or
 * scrolled out of the newest window as the child kept working) stay in front
 * of it, so a long-running child never loses the steps between two reads.
 * Only the tail changes between reads, and the fresh page always covers it.
 */
export function refreshAgentHistory(previous: AgentHistoryState | null, page: HistoryPage, clipped: boolean): AgentHistoryState {
  const first = page.actions[0]
  const fresh: AgentHistoryState = { actions: [...page.actions], before: page.before, more: page.has_more, clipped }
  if (!previous || !page.has_more || !first) return fresh
  const at = previous.actions.findIndex(action => action.id === first.id)
  const start = actionIndex(first.id)
  const prefix = at >= 0 ? previous.actions.slice(0, at) : previous.actions.filter(action => actionIndex(action.id) < start)
  if (!prefix.length) return fresh
  return { actions: [...prefix, ...page.actions], before: previous.before, more: previous.more, clipped: previous.clipped || clipped }
}

/** Put an older page in front of what is loaded. */
export function prependAgentHistory(state: AgentHistoryState, page: HistoryPage, clipped: boolean): AgentHistoryState {
  const loaded = new Set(state.actions.map(action => action.id))
  return { actions: [...page.actions.filter(action => !loaded.has(action.id)), ...state.actions], before: page.before, more: page.has_more, clipped: state.clipped || clipped }
}

/** The fold's wording for a call whose result was never stored (blocks.ts). */
const UNANSWERED = 'Turn ended before a tool result was received'

/**
 * Saved actions as transcript blocks, with ids that stay stable across reads
 * (`ids` is kept by the caller). A call with no saved result is still running
 * while the agent works; once it stopped, the call says it never returned.
 */
export function agentHistoryBlocks(actions: readonly HistoryAction[], ids: Map<string, number>, active: boolean): Block[] {
  return historyBlocks({ actions: [...actions], before: null, has_more: false }, ids).map(block => {
    if (block.kind !== 'tools' || !block.items.some(item => item.error === UNANSWERED)) return block
    const items = block.items.map((item): ToolItem => {
      if (item.error !== UNANSWERED) return item
      if (!active) return { ...item, error: 'The agent stopped before this call returned a result.' }
      const { error: _error, ...rest } = item
      void _error
      return { ...rest, state: 'working' }
    })
    return { ...block, items, running: items.some(item => item.state === 'working') }
  })
}

/** One of the child's own stream events, unwrapped from its envelope. */
export interface AgentLiveEvent {
  readonly type: string
  readonly payload: Readonly<Record<string, unknown>>
}

const LIVE_TYPES = new Set(['turn_begin', 'turn_end', 'think_part', 'text_part', 'tool_call', 'tool_result'])

/**
 * The event a `subagent_event` payload carries for exactly this agent, or
 * null. Envelopes nest for grandchildren (as `foldAgentEvent` decodes them);
 * only the innermost envelope names the agent the event belongs to.
 */
export function agentLiveEventOf(envelope: Readonly<Record<string, unknown>>, agentId: string, depth = 0): AgentLiveEvent | null {
  if (depth > 32) return null
  const event = record(envelope.event)
  if (event.type === 'subagent_event') return agentLiveEventOf(record(event.payload), agentId, depth + 1)
  const type = text(event.type)
  if (!agentId || text(envelope.agent_id) !== agentId || !LIVE_TYPES.has(type)) return null
  return { type, payload: record(event.payload) }
}

/** Append one live event, keeping only the newest. */
export function appendLiveEvent<T extends AgentLiveEvent>(events: readonly T[], event: T): readonly T[] {
  return [...events, event].slice(-MAX_LIVE_EVENTS)
}

/**
 * Live events still needed after a fresh saved page arrives. Events received
 * after the read was sent stay. Older ones are dropped because the page holds
 * them, with one exception: the child publishes a tool event before it saves
 * it, so a tool result the page has not answered yet stays until a later
 * page has it.
 */
export function retainLiveEvents<T extends AgentLiveEvent & { readonly seq: number }>(events: readonly T[], sent: number, page: HistoryPage): readonly T[] {
  const answered = new Set<string>()
  for (const action of page.actions) {
    for (const message of action.messages) {
      const row = record(message)
      if (row.role === 'tool' && typeof row.tool_call_id === 'string') answered.add(row.tool_call_id)
    }
  }
  return events.filter(event => {
    if (event.seq > sent) return true
    if (event.type !== 'tool_call' && event.type !== 'tool_result') return false
    const id = text(event.payload.tool_call_id) || text(event.payload.id)
    return Boolean(id) && !answered.has(id)
  })
}

/** A live result frame as the payload the main fold's `tool_result` reads. */
function liveResultPayload(id: string, payload: Readonly<Record<string, unknown>>): Record<string, unknown> {
  return {
    tool_call_id: id,
    name: text(payload.name),
    return_value: typeof payload.result_full === 'string' ? payload.result_full : text(payload.return_value),
    ...(typeof payload.duration_ms === 'number' ? { duration_ms: payload.duration_ms } : {}),
    ...(payload.permitted === false ? { permitted: false } : {}),
    ...(text(payload.error) ? { error: text(payload.error) } : {}),
  }
}

/**
 * A call the saved page shows running can finish before the next page is
 * read: its live result settles that row in place instead of waiting.
 */
export function settleSavedTools(saved: readonly Block[], events: readonly AgentLiveEvent[]): Block[] {
  const results = new Map<string, Readonly<Record<string, unknown>>>()
  for (const { type, payload } of events) {
    const id = type === 'tool_result' ? text(payload.tool_call_id) || text(payload.id) : ''
    if (id) results.set(id, payload)
  }
  if (!results.size) return [...saved]
  return saved.map(block => {
    if (block.kind !== 'tools' || !block.items.some(item => item.state === 'working' && results.has(item.id))) return block
    const items = block.items.map((item): ToolItem => {
      const payload = item.state === 'working' ? results.get(item.id) : undefined
      if (!payload) return item
      // The same fold the main conversation uses decides state, output and failure.
      const builder = new BlockBuilder()
      builder.push('tool_call', { id: item.id, name: item.name, arguments: item.input })
      builder.push('tool_result', liveResultPayload(item.id, payload))
      const settled = builder.snapshot(true).flatMap(next => next.kind === 'tools' ? next.items : []).find(next => next.id === item.id)
      if (!settled) return item
      const { error: _error, ...rest } = item
      void _error
      return { ...rest, state: settled.state, output: settled.output, dur: settled.dur, ...(settled.error ? { error: settled.error } : {}) }
    })
    return { ...block, items, running: items.some(item => item.state === 'working') }
  })
}

/** Every tool call id a set of blocks holds. */
export function toolIdsOf(blocks: readonly Block[]): ReadonlySet<string> {
  return new Set(blocks.flatMap(block => block.kind === 'tools' ? block.items.map(item => item.id) : []))
}

/**
 * Live events as transcript blocks, through the same fold the main
 * conversation uses. Live frames carry the exact text, arguments and result
 * in `*_full` fields (the short fields are previews); a call the saved page
 * already shows is skipped rather than drawn twice.
 */
export function liveAgentBlocks(events: readonly AgentLiveEvent[], saved: ReadonlySet<string>, active: boolean): Block[] {
  const builder = new BlockBuilder()
  for (const { type, payload } of events) {
    if (type === 'text_part') {
      const value = text(payload.text_full) || text(payload.text)
      if (value) builder.push('text_part', { text: value })
    } else if (type === 'think_part') {
      const value = text(payload.think) || text(payload.text)
      if (value) builder.push('think_part', { think: value })
    } else if (type === 'tool_call') {
      const id = text(payload.id) || text(payload.tool_call_id)
      if (!id || saved.has(id)) continue
      builder.push('tool_call', { id, name: text(payload.name), arguments: text(payload.arguments_full) || text(payload.arguments) })
    } else if (type === 'tool_result') {
      const id = text(payload.tool_call_id) || text(payload.id)
      if (!id || saved.has(id)) continue
      builder.push('tool_result', liveResultPayload(id, payload))
    } else if (type === 'turn_end') builder.finalize()
  }
  return [...(active ? builder.snapshot(true) : builder.all())]
}

/**
 * The whole conversation: saved blocks, then the live tail. While the agent
 * works, the newest saved reply or reasoning (a crash checkpoint of text that
 * is still being written) carries the streaming caret until a live run takes
 * the tail.
 */
export function agentTranscriptBlocks(saved: readonly Block[], live: readonly Block[], active: boolean): Block[] {
  const blocks = [...saved, ...live]
  const last = blocks.at(-1)
  if (active && !live.length && last && (last.kind === 'agent' || last.kind === 'thinking') && !last.streaming) {
    blocks[blocks.length - 1] = { ...last, streaming: true }
  }
  return blocks
}
