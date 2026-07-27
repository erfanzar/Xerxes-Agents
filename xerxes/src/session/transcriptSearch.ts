// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Cross-session search over the transcripts the daemon actually keeps.
 *
 * Deliberately a port and not a store: it holds no record of its own, is fed
 * from the same message arrays the daemon persists, and can be dropped and
 * rebuilt from them at any time. A second durable model of a conversation is
 * exactly the drift this codebase already pays for elsewhere.
 *
 * Honesty invariant: a message whose shape this indexer does not model is
 * indexed as EMPTY and counted, producing a visible under-count. The tempting
 * alternative — serializing the whole row and indexing that — matches on JSON
 * punctuation and field names, so a query "matches" a message with nothing in
 * it a human would recognize as the term.
 */

/** Hits are capped so one enormous session cannot fill an answer by itself. */
const DEFAULT_HIT_LIMIT = 20
const DEFAULT_HITS_PER_SESSION = 5
const EXCERPT_RADIUS = 80
/** Bounded per-message text: the index lives in daemon memory for its lifetime. */
const MAX_INDEXED_CHARS = 16_384
const TOKEN_PATTERN = /[\p{L}\p{N}_]+/gu

/** One transcript offered to the index, in the daemon's own message shape. */
export interface TranscriptSearchDocument {
  readonly messages: readonly unknown[]
  readonly sessionId: string
  readonly title?: string
  readonly updatedAt?: string
}

export interface TranscriptSearchHit {
  readonly excerpt: string
  readonly messageIndex: number
  readonly role: string
  readonly sessionId: string
  readonly title: string
  readonly updatedAt: string
}

export interface TranscriptSearchOptions {
  readonly limit?: number
  readonly perSession?: number
  readonly sessionId?: string
}

export interface TranscriptSearchStats {
  /** Every message the index holds a row for, searchable or not. */
  readonly indexedMessages: number
  /** Rows that produced at least one token. */
  readonly searchableMessages: number
  readonly sessions: number
  /** Rows whose text was cut at the per-message ceiling. */
  readonly truncatedMessages: number
  /** Rows indexed as empty because their shape is not modelled here. */
  readonly unrecognizedMessages: number
}

interface IndexedRow {
  readonly lowerText: string
  readonly messageIndex: number
  readonly role: string
  readonly text: string
}

interface IndexedSession {
  readonly rows: readonly IndexedRow[]
  readonly searchableMessages: number
  readonly title: string
  /** Token to row ordinals, so a query touches only the sessions that can match. */
  readonly tokens: ReadonlyMap<string, readonly number[]>
  readonly truncatedMessages: number
  readonly unrecognizedMessages: number
  readonly updatedAt: string
}

export class TranscriptSearchIndex {
  private readonly sessions = new Map<string, IndexedSession>()

  /** Replace everything held for one session. Re-indexing after a save is the normal path. */
  index(document: TranscriptSearchDocument): void {
    if (!document.sessionId) return
    const rows: IndexedRow[] = []
    const tokens = new Map<string, number[]>()
    let searchableMessages = 0
    let truncatedMessages = 0
    let unrecognizedMessages = 0

    document.messages.forEach((message, messageIndex) => {
      const extracted = extractSearchableText(message)
      if (!extracted.recognized) unrecognizedMessages += 1
      let text = extracted.text
      if (text.length > MAX_INDEXED_CHARS) {
        text = text.slice(0, MAX_INDEXED_CHARS)
        truncatedMessages += 1
      }
      const lowerText = text.toLowerCase()
      const ordinal = rows.push({ lowerText, messageIndex, role: roleOf(message), text }) - 1
      let tokenized = false
      for (const token of lowerText.matchAll(TOKEN_PATTERN)) {
        tokenized = true
        const postings = tokens.get(token[0])
        if (postings === undefined) tokens.set(token[0], [ordinal])
        else if (postings[postings.length - 1] !== ordinal) postings.push(ordinal)
      }
      if (tokenized) searchableMessages += 1
    })

    this.sessions.set(document.sessionId, {
      rows,
      searchableMessages,
      title: document.title ?? '',
      tokens,
      truncatedMessages,
      unrecognizedMessages,
      updatedAt: document.updatedAt ?? '',
    })
  }

  has(sessionId: string): boolean {
    return this.sessions.has(sessionId)
  }

  remove(sessionId: string): boolean {
    return this.sessions.delete(sessionId)
  }

  clear(): void {
    this.sessions.clear()
  }

  /** Every indexed term must appear in a row for it to match; newest sessions answer first. */
  search(query: string, options: TranscriptSearchOptions = {}): readonly TranscriptSearchHit[] {
    const terms = [...new Set(Array.from(query.toLowerCase().matchAll(TOKEN_PATTERN), match => match[0]))]
    if (terms.length === 0) return []
    const limit = boundedCount(options.limit, DEFAULT_HIT_LIMIT)
    const perSession = boundedCount(options.perSession, DEFAULT_HITS_PER_SESSION)
    const scoped = [...this.sessions.entries()]
      .filter(([sessionId]) => options.sessionId === undefined || sessionId === options.sessionId)
      .sort(([, left], [, right]) => timestampMillis(right.updatedAt) - timestampMillis(left.updatedAt))

    const hits: TranscriptSearchHit[] = []
    for (const [sessionId, session] of scoped) {
      if (hits.length >= limit) break
      const ordinals = intersectPostings(session.tokens, terms)
      if (ordinals === undefined) continue
      let taken = 0
      for (const ordinal of ordinals) {
        if (taken >= perSession || hits.length >= limit) break
        const row = session.rows[ordinal]
        if (row === undefined) continue
        hits.push({
          excerpt: excerptAround(row, terms),
          messageIndex: row.messageIndex,
          role: row.role,
          sessionId,
          title: session.title,
          updatedAt: session.updatedAt,
        })
        taken += 1
      }
    }
    return hits
  }

  stats(): TranscriptSearchStats {
    let indexedMessages = 0
    let searchableMessages = 0
    let truncatedMessages = 0
    let unrecognizedMessages = 0
    for (const session of this.sessions.values()) {
      indexedMessages += session.rows.length
      searchableMessages += session.searchableMessages
      truncatedMessages += session.truncatedMessages
      unrecognizedMessages += session.unrecognizedMessages
    }
    return {
      indexedMessages,
      searchableMessages,
      sessions: this.sessions.size,
      truncatedMessages,
      unrecognizedMessages,
    }
  }
}

interface ExtractedText {
  readonly recognized: boolean
  readonly text: string
}

/** Text a human would recognize as the message's content, or nothing at all. */
export function extractSearchableText(message: unknown): ExtractedText {
  if (!isRecord(message)) return { recognized: false, text: '' }
  const content = message.content
  if (typeof content === 'string') return { recognized: true, text: content }
  if (Array.isArray(content)) return extractFromBlocks(content)
  if (content === undefined || content === null) {
    // A tool-call-only assistant message legitimately carries no text; a
    // non-string `text` field is a shape this indexer does not model.
    if (message.text === undefined) return { recognized: true, text: '' }
    return typeof message.text === 'string'
      ? { recognized: true, text: message.text }
      : { recognized: false, text: '' }
  }
  return { recognized: false, text: '' }
}

function extractFromBlocks(blocks: readonly unknown[]): ExtractedText {
  const parts: string[] = []
  let recognized = true
  for (const block of blocks) {
    if (typeof block === 'string') {
      parts.push(block)
      continue
    }
    if (!isRecord(block)) {
      recognized = false
      continue
    }
    if (typeof block.text === 'string') {
      parts.push(block.text)
      continue
    }
    // A typed block without text (tool_use, image, thinking) contributes
    // nothing and is still understood; a block with neither is not.
    if (typeof block.type !== 'string') recognized = false
  }
  return { recognized, text: parts.join('\n') }
}

/** Row ordinals carrying every term, or undefined when one term is absent entirely. */
function intersectPostings(
  tokens: ReadonlyMap<string, readonly number[]>,
  terms: readonly string[],
): readonly number[] | undefined {
  let current: readonly number[] | undefined
  for (const term of terms) {
    const postings = tokens.get(term)
    if (postings === undefined) return undefined
    if (current === undefined) {
      current = postings
      continue
    }
    const known = new Set(postings)
    current = current.filter(ordinal => known.has(ordinal))
    if (current.length === 0) return current
  }
  return current
}

function excerptAround(row: IndexedRow, terms: readonly string[]): string {
  let earliest = -1
  for (const term of terms) {
    const found = row.lowerText.indexOf(term)
    if (found >= 0 && (earliest < 0 || found < earliest)) earliest = found
  }
  const start = Math.max(0, (earliest < 0 ? 0 : earliest) - EXCERPT_RADIUS)
  const end = Math.min(row.text.length, start + EXCERPT_RADIUS * 2)
  const slice = row.text.slice(start, end).replaceAll(/\s+/g, ' ').trim()
  return `${start > 0 ? '…' : ''}${slice}${end < row.text.length ? '…' : ''}`
}

function roleOf(message: unknown): string {
  return isRecord(message) && typeof message.role === 'string' ? message.role : ''
}

function boundedCount(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isInteger(value) && value > 0 ? value : fallback
}

/** Malformed timestamps sort as the epoch instead of producing NaN orderings. */
function timestampMillis(value: string): number {
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
