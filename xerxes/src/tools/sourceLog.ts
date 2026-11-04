// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Passive record of the web sources a session's research tools returned or fetched.
 *
 * Search and fetch tools write here as a side effect; nothing reads it back into
 * the model's context. A `/sources` command (or any host) renders it for the user.
 */

/** Session key used when a tool runs without a session id in its context. */
export const DEFAULT_SOURCE_SESSION = 'default'

/** Most sources kept per session; the oldest are dropped first. */
export const MAX_SOURCES_PER_SESSION = 500

/** Most sessions tracked at once; the least recently written are dropped first. */
export const MAX_SOURCE_SESSIONS = 200

export interface SourceInput {
  readonly sessionId?: string | undefined
  readonly url: string
  readonly title?: string | undefined
  /** Tool that produced the source, for example `DuckDuckGoSearch` or `WebScraper`. */
  readonly tool: string
  /** ISO timestamp; defaults to now. */
  readonly at?: string | undefined
}

export interface SourceEntry {
  readonly url: string
  readonly title?: string
  readonly tool: string
  readonly at: string
}

/** Bounded, deduplicated (by URL) per-session source list. */
export class SourceLog {
  private readonly sessions = new Map<string, SourceEntry[]>()

  /** Returns the stored entry, or `undefined` when the URL is not a usable http(s) URL. */
  record(input: SourceInput): SourceEntry | undefined {
    const url = normalizeSourceUrl(input.url)
    if (url === undefined) return undefined
    const sessionId = input.sessionId?.trim() || DEFAULT_SOURCE_SESSION
    const title = input.title?.replace(/\s+/g, ' ').trim()
    const existing = this.sessions.get(sessionId)
    const list = existing ?? []
    // Re-insert so Map order tracks the most recently written session.
    this.sessions.delete(sessionId)
    this.sessions.set(sessionId, list)
    const index = list.findIndex(entry => entry.url === url)
    if (index >= 0) {
      const current = list[index]!
      if (current.title === undefined && title) {
        const filled: SourceEntry = { ...current, title }
        list[index] = filled
        return filled
      }
      return current
    }
    const entry: SourceEntry = {
      at: input.at ?? new Date().toISOString(),
      tool: input.tool,
      url,
      ...(title ? { title } : {}),
    }
    list.push(entry)
    if (list.length > MAX_SOURCES_PER_SESSION) list.splice(0, list.length - MAX_SOURCES_PER_SESSION)
    while (this.sessions.size > MAX_SOURCE_SESSIONS) {
      const oldest = this.sessions.keys().next().value
      if (oldest === undefined) break
      this.sessions.delete(oldest)
    }
    return entry
  }

  list(sessionId: string = DEFAULT_SOURCE_SESSION): readonly SourceEntry[] {
    return [...(this.sessions.get(sessionId) ?? [])]
  }

  clear(sessionId?: string): void {
    if (sessionId === undefined) this.sessions.clear()
    else this.sessions.delete(sessionId)
  }
}

/** Markdown bullet list of sources, oldest first. */
export function renderSources(list: readonly SourceEntry[]): string {
  if (list.length === 0) return 'No sources recorded for this session.'
  const lines = list.map((entry, index) => {
    const label = (entry.title ?? entry.url).replace(/[[\]]/g, '')
    return `${index + 1}. [${label}](${entry.url}) — ${entry.tool}, ${entry.at}`
  })
  return [`Sources (${list.length}):`, '', ...lines].join('\n')
}

let defaultLog = new SourceLog()

/** The process-wide log the research tools write to. */
export function getSourceLog(): SourceLog {
  return defaultLog
}

/** Replace the process-wide log (tests); returns the previous one so it can be restored. */
export function setSourceLog(log: SourceLog): SourceLog {
  const previous = defaultLog
  defaultLog = log
  return previous
}

function normalizeSourceUrl(value: string): string | undefined {
  try {
    const parsed = new URL(value.trim())
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return undefined
    parsed.hash = ''
    return parsed.toString()
  } catch {
    return undefined
  }
}
