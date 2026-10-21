// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SessionSearchPort, SessionSearchRequest } from '../tools/agentMetaTools.js'
import type { JsonObject } from '../types/toolCalls.js'
import { ArchiveHistory, searchableHistory } from './archiveHistory.js'
import type { DaemonSession } from './runtime.js'

type Message = DaemonSession['messages'][number]

/** Characters of context kept around each match in a hit's snippet. */
const SNIPPET_RADIUS = 240
/** A hit never returns more than this much text; ReadFile-style paging is not needed for search. */
const MAX_SNIPPET_CHARS = 1_200
const MAX_HITS = 25

export interface ArchiveSearchPortOptions {
  /** Live session messages, or undefined when the session is unknown. */
  readonly liveMessages: (sessionId: string) => readonly Message[] | undefined
  /** `<id>.precompact.jsonl`, or undefined when the session has no transcript path. */
  readonly archivePath: (sessionId: string) => string | undefined
  readonly archive?: ArchiveHistory
}

/**
 * `session_search` over one session's whole history: the verbatim
 * pre-compaction archive plus the live window. Only the calling turn's own
 * session is searchable; another session id is refused rather than read.
 */
export function archiveSessionSearchPort(options: ArchiveSearchPortOptions): SessionSearchPort {
  const archive = options.archive ?? new ArchiveHistory()
  return {
    async search(request: SessionSearchRequest): Promise<JsonObject> {
      const sessionId = request.sessionId ?? request.callerSessionId
      if (!sessionId) return { error: 'no session to search: this call has no session', hits: [] }
      if (request.callerSessionId && sessionId !== request.callerSessionId) {
        return { error: 'only the current session can be searched', hits: [] }
      }
      const live = options.liveMessages(sessionId)
      if (!live) return { error: `session ${sessionId} is not open`, hits: [] }
      const terms = queryTerms(request.query)
      if (!terms.length) return { query: request.query, count: 0, hits: [] }
      const archived = await archive.archived(options.archivePath(sessionId))
      const history = searchableHistory(archived?.messages ?? [], live)
      const scored: { index: number; score: number; text: string; first: number }[] = []
      history.messages.forEach((message, index) => {
        const text = messageText(message)
        if (!text) return
        const lower = text.toLowerCase()
        let score = 0
        let first = -1
        for (const term of terms) {
          let at = lower.indexOf(term)
          if (at < 0) continue
          if (first < 0 || at < first) first = at
          let count = 0
          while (at >= 0 && count < 20) { count++; at = lower.indexOf(term, at + term.length) }
          score += 1 + Math.log2(count)
        }
        // Every term must appear: a query is usually a value being recovered.
        if (score > 0 && terms.every(term => lower.includes(term))) scored.push({ index, score, text, first })
      })
      scored.sort((left, right) => right.score - left.score || right.index - left.index)
      const limit = Math.min(MAX_HITS, request.limit)
      return {
        query: request.query,
        session_id: sessionId,
        searched_messages: history.messages.length,
        archived_messages: archived?.messages.length ?? 0,
        count: scored.length,
        hits: scored.slice(0, limit).map(hit => {
          const message = history.messages[hit.index]!
          return {
            position: hit.index,
            role: message.role,
            ...(message.role === 'tool' && typeof message.tool_call_id === 'string' && message.tool_call_id ? { tool_call_id: message.tool_call_id } : {}),
            in_context: history.inContext[hit.index] ?? false,
            snippet: snippet(hit.text, hit.first),
            chars: hit.text.length,
          }
        }),
      }
    },
  }
}

function queryTerms(query: string): string[] {
  return [...new Set(query.toLowerCase().split(/\s+/).map(term => term.trim()).filter(Boolean))]
}

function snippet(text: string, at: number): string {
  if (text.length <= MAX_SNIPPET_CHARS) return text
  const start = Math.max(0, at - SNIPPET_RADIUS)
  const end = Math.min(text.length, start + MAX_SNIPPET_CHARS)
  return `${start > 0 ? '…' : ''}${text.slice(start, end)}${end < text.length ? '…' : ''}`
}

function messageText(message: Message): string {
  const parts: string[] = []
  const content = message.content as unknown
  if (typeof content === 'string') parts.push(content)
  else if (Array.isArray(content)) {
    for (const part of content) {
      if (typeof part === 'string') parts.push(part)
      else if (part && typeof part === 'object' && typeof (part as { text?: unknown }).text === 'string') parts.push((part as { text: string }).text)
    }
  }
  const calls = (message as { tool_calls?: unknown }).tool_calls
  if (Array.isArray(calls)) {
    for (const call of calls) {
      const fn = call && typeof call === 'object' ? (call as { function?: { name?: unknown; arguments?: unknown } }).function : undefined
      if (fn) parts.push(`${String(fn.name ?? '')}(${typeof fn.arguments === 'string' ? fn.arguments : JSON.stringify(fn.arguments ?? '')})`)
    }
  }
  return parts.join('\n')
}
