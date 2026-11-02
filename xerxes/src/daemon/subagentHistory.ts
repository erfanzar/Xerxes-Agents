// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonRpcPayload } from '../protocol/jsonRpc.js'
import { DaemonTranscriptStore, looksLikeSessionId } from '../session/daemonTranscript.js'
import { RESUME_REPLAY_SENTINEL } from '../session/resumeRepair.js'
import { sessionHistoryPage } from './historyPage.js'
import type { DaemonSession } from './runtime.js'

type Message = DaemonSession['messages'][number]

/** Serialized ceiling for one page of a child's conversation; the page shrinks to fit. */
export const SUBAGENT_HISTORY_PAGE_BYTES = 512 * 1024
/** Longest single text a page carries (a tool result, a reply, call arguments). */
export const SUBAGENT_HISTORY_TEXT_CHARS = 32_000

/** Placeholder a crash checkpoint writes for a reply that was still only reasoning. */
const PARTIAL_REASONING_PLACEHOLDER = '[interrupted while reasoning]'

export interface SubagentHistoryRequest {
  /** Transcript directory the child's conversation was saved in. */
  readonly directory: string
  readonly projectDirectory: string
  /** The parent session the inspect request was scoped to. */
  readonly parentSessionId: string
  /** Exact runtime id of the child being inspected. */
  readonly task: string
  /** The parent's retained manifest row for that child. */
  readonly saved: Readonly<Record<string, unknown>>
  readonly limit: number
  readonly before?: unknown
  /** Wire projection for messages (inline image budgeting). */
  readonly project?: (messages: Message[]) => Message[]
}

/**
 * One page of a child agent's own saved conversation, in the same action shape
 * `session.history` returns, so a client renders it with its main transcript.
 *
 * Read-only: the child's transcript is loaded, never written, so a running
 * child keeps sole ownership of it. The page is bounded twice: every text is
 * clipped to SUBAGENT_HISTORY_TEXT_CHARS, then the action count halves until
 * the page fits SUBAGENT_HISTORY_PAGE_BYTES. The `before` cursor stays valid
 * because clipping happens after paging.
 *
 * A child whose conversation was not saved (or belongs to someone else)
 * answers `history: null` with a reason rather than an error, so the caller
 * still gets the rest of the inspect response.
 */
export async function subagentHistoryPayload(request: SubagentHistoryRequest): Promise<JsonRpcPayload> {
  const historySessionId = typeof request.saved.history_session_id === 'string' ? request.saved.history_session_id.trim() : ''
  if (!historySessionId || !looksLikeSessionId(historySessionId)) return unavailable('This agent did not save its own conversation.')
  const store = new DaemonTranscriptStore({ directory: request.directory, currentProjectDirectory: request.projectDirectory })
  let transcript: Awaited<ReturnType<DaemonTranscriptStore['load']>>
  try {
    transcript = await store.load(historySessionId)
  } catch (error) {
    // The identity, status and retained output above are still worth returning.
    return unavailable(`This agent's saved conversation could not be read: ${error instanceof Error ? error.message : String(error)}`, historySessionId)
  }
  if (!transcript) return unavailable('This agent has not saved its conversation yet.', historySessionId)
  const metadata = transcript.metadata
  const owner = typeof metadata.subagent_handle_id === 'string' ? metadata.subagent_handle_id : metadata.subagent_id
  const parent = metadata.parent_session_id
  // The manifest row names the file; the file must name this child and parent back.
  if (owner !== request.task || (typeof parent === 'string' && parent && parent !== request.parentSessionId)) {
    return unavailable('The saved conversation does not belong to this agent.', historySessionId)
  }
  // Resume repair answers every call still outstanding at the end with a
  // replay sentinel. For a running child that call is simply in flight; it is
  // left unanswered so the client shows it running. An earlier sentinel is a
  // real interruption and stays.
  let end = transcript.messages.length
  while (end > 0 && transcript.messages[end - 1]!.role === 'tool' && transcript.messages[end - 1]!.content === RESUME_REPLAY_SENTINEL) end -= 1
  const messages = transcript.messages.slice(0, end).flatMap(visibleMessage)
  const assistants = messages.filter(message => message.role === 'assistant').length
  const session = {
    id: historySessionId,
    messages,
    toolExecutions: [...transcript.toolExecutions],
    // Retained reasoning is a tail; only an exact one-per-reply list can be
    // matched to replies by position. Messages carry their own `thinking`.
    thinkingContent: transcript.thinkingContent.length === assistants ? [...transcript.thinkingContent] : [],
  }
  let limit = request.limit
  for (;;) {
    const page = sessionHistoryPage(session, limit, request.before)
    let clipped = false
    const clip = (value: string): string => {
      if (value.length <= SUBAGENT_HISTORY_TEXT_CHARS) return value
      clipped = true
      return `${value.slice(0, SUBAGENT_HISTORY_TEXT_CHARS)}\n… [${value.length - SUBAGENT_HISTORY_TEXT_CHARS} more characters not shown]`
    }
    const flat = page.actions.flatMap(action => action.messages.map(message => clipMessage(message, clip)))
    const projected = request.project ? request.project(flat) : flat
    let offset = 0
    const actions = page.actions.map(action => {
      const slice = projected.slice(offset, offset + action.messages.length)
      offset += action.messages.length
      return { ...action, messages: slice, executions: action.executions.map(withoutResult), thinking: action.thinking.map(value => typeof value === 'string' ? clip(value) : value) }
    })
    const history = { ...page, actions, ...(clipped ? { clipped: true } : {}) }
    if (limit <= 1 || Buffer.byteLength(JSON.stringify(history), 'utf8') <= SUBAGENT_HISTORY_PAGE_BYTES) {
      return { history_session_id: historySessionId, history }
    }
    limit = Math.max(1, Math.floor(limit / 2))
  }
}

function unavailable(note: string, historySessionId?: string): JsonRpcPayload {
  return { history: null, history_note: note, ...(historySessionId ? { history_session_id: historySessionId } : {}) }
}

/**
 * A crash checkpoint stands in a placeholder for a reply that was still only
 * reasoning. The agent never wrote that text, so only its reasoning is shown.
 */
function visibleMessage(message: Readonly<Record<string, unknown>>): Message[] {
  if (typeof message.role !== 'string') return []
  if (message.role === 'assistant' && message.checkpoint_partial === true && message.content === PARTIAL_REASONING_PLACEHOLDER) {
    return [{ ...message, role: message.role, content: '' }]
  }
  return [{ ...message, role: message.role }]
}

function clipMessage(message: Message, clip: (value: string) => string): Message {
  const content = typeof message.content === 'string' ? clip(message.content)
    : Array.isArray(message.content) ? message.content.map(part => clipPart(part, clip))
    : message.content
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls.map(call => clipCall(call, clip)) : undefined
  return { ...message, content, ...(calls ? { tool_calls: calls } : {}), ...(typeof message.thinking === 'string' ? { thinking: clip(message.thinking) } : {}) }
}

function clipPart(part: unknown, clip: (value: string) => string): unknown {
  if (!isRecord(part)) return part
  const next: Record<string, unknown> = { ...part }
  for (const key of ['text', 'thinking', 'think', 'content'] as const) {
    if (typeof next[key] === 'string') next[key] = clip(next[key])
  }
  return next
}

function clipCall(call: unknown, clip: (value: string) => string): unknown {
  if (!isRecord(call) || !isRecord(call.function) || typeof call.function.arguments !== 'string') return call
  return { ...call, function: { ...call.function, arguments: clip(call.function.arguments) } }
}

/** The tool message already carries the result; the execution keeps identity, timing and permission. */
function withoutResult(execution: unknown): unknown {
  if (!isRecord(execution)) return execution
  const { result: _result, ...rest } = execution
  void _result
  return rest
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
