// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { lstat } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { isCompactionSummaryMessage } from '../context/compressor.js'
import type { DaemonSession } from './runtime.js'

type Message = DaemonSession['messages'][number]
const identity = (message: Message): string => {
  // Pruning can shorten tool output without changing which execution it is.
  if (message.role === 'tool' && message.tool_call_id) return `tool:${message.tool_call_id}`
  return createHash('sha256').update(JSON.stringify({ role: message.role, content: message.content, tool_calls: message.tool_calls })).digest('hex')
}

/** A read archive, ready to stitch the live window onto. */
export interface ArchivedHistory {
  stitch(current: readonly Message[]): Message[]
}

/** A stitched history with each message's identity, kept in step so no record rehashes it. */
interface Stitched { readonly messages: Message[]; readonly keys: string[] }

/**
 * Stitch successive compacted windows, retaining repeated human messages.
 *
 * `retained` is how many messages followed the summary when the history's
 * last window was compacted (absent for archives written before it was
 * recorded): exactly that many messages at the end of the history are the
 * tail the next window starts from.
 */
export function appendHistoryWindow(history: readonly Message[], window: readonly Message[], current = false, retained?: number): Message[] {
  const stitched = { messages: [...history], keys: history.map(identity) }
  stitchWindow(stitched, window, current, retained)
  return stitched.messages
}

function stitchWindow(history: Stitched, window: readonly Message[], current: boolean, retained: number | undefined): void {
  const { messages, keys } = history
  const h = keys.length
  const summary = window.findLastIndex(isCompactionSummaryMessage)
  const next = summary < 0 ? window : window.slice(summary + 1)
  const newKeys = next.map(identity)
  // Keep the first `end` history messages, then append next from `from`.
  const splice = (end: number, from: number, rows: readonly Message[] = next, rowKeys: readonly string[] = newKeys): void => {
    messages.length = end
    keys.length = end
    for (let index = from; index < rows.length; index++) {
      messages.push(rows[index]!)
      keys.push(rowKeys[index]!)
    }
  }
  if (current && !window.length) return splice(0, 0)
  if (!h) return splice(0, 0, window, window.map(identity))
  if (summary < 0) {
    let prefix = 0
    while (prefix < Math.min(h, next.length) && keys[prefix] === newKeys[prefix]) prefix++
    if (prefix === h || prefix === next.length) return current && prefix === next.length ? splice(0, 0) : splice(h, prefix)
  }
  const start = retained === undefined ? -1 : Math.max(0, h - retained)
  let kept = 0
  if (summary >= 0 && start >= 0) {
    while (kept < next.length && start + kept < h && newKeys[kept] === keys[start + kept]) kept++
    // The window starts exactly where the recorded tail does. Whatever of the
    // tail it lacks was undone; whatever follows the surviving part is new.
    if (kept) return splice(start + kept, kept)
  }
  if (!next.length && start < 0) return
  if (current && summary >= 0) {
    // Undo can shorten the retained tail. Respect its last surviving anchor
    // instead of resurrecting later archived turns on the next reopen.
    for (let end = h; end >= next.length && next.length; end--) {
      if (newKeys.every((key, i) => key === keys[end - next.length + i])) return splice(end, next.length)
    }
  }
  for (let count = Math.min(h, next.length); count > 0; count--) {
    if (newKeys.slice(0, count).every((key, i) => key === keys[h - count + i])) return splice(h, count)
  }
  if (summary >= 0) {
    // Undo took back the whole recorded tail (and maybe a turn ran since).
    // Undo always stops at a user message, so a new turn starts with one.
    if (start >= 0 && (!next.length || (next[0]?.role === 'user' && messages[start]?.role === 'user'))) return splice(start, 0)
    // Without a recorded tail, find where the window's surviving prefix
    // diverges from the history at a turn boundary; undo dropped what follows.
    if (start < 0 && next.length) {
      for (let at = h - 1; at >= 0; at--) {
        let run = 0
        while (run < next.length && at + run < h && newKeys[run] === keys[at + run]) run++
        if (run && run < next.length && at + run < h && next[run]?.role === 'user' && messages[at + run]?.role === 'user') return splice(at + run, run)
      }
    }
  }
  // A zero-retention compaction has no overlap; every subsequent row is new.
  splice(h, 0)
}

const RECORD_START = '{"archived_at"'

function retainedCount(record: object): number | undefined {
  const value = 'retained_messages' in record ? record.retained_messages : undefined
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined
}

/** Read-only archive projection. Never replace the provider's compact context. */
export class ArchiveHistory {
  private cache = new Map<string, { stamp: string; bytes: number; history: Stitched; retained: number | undefined }>()

  constructor(private readonly report: (warning: string) => void = warning => console.error(warning)) {}

  async messages(path: string | undefined, current: readonly Message[]): Promise<Message[]> {
    const archived = await this.archived(path)
    return archived ? archived.stitch(current) : [...current]
  }

  /** The archived history alone, so a caller can read its live tail after the
   * await rather than splicing in messages that went stale during the read. */
  async archived(path: string | undefined): Promise<ArchivedHistory | undefined> {
    if (!path) return undefined
    const info = await lstat(path).catch(error => {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
      throw error
    })
    if (!info) return undefined
    if (!info.isFile()) throw new Error('Conversation archive must be a regular file')
    if (info.size > 256 * 1024 * 1024) throw new Error('Conversation archive exceeds 256 MiB; export it before loading older history')
    const stamp = `${info.ino}:${info.size}:${info.mtimeMs}`
    let cached = this.cache.get(path)
    if (cached?.stamp !== stamp) {
      const history: Stitched = { messages: [], keys: [] }
      let pending = '', retained: number | undefined, skipped = 0
      const reader = Bun.file(path).stream().pipeThrough(new TextDecoderStream()).getReader()
      try {
        for (;;) {
          const chunk = await reader.read()
          if (chunk.done) break
          pending += chunk.value
          let boundary: number
          while ((boundary = pending.indexOf('\n')) >= 0) {
            const line = pending.slice(0, boundary)
            pending = pending.slice(boundary + 1)
            if (!line.trim()) continue
            const record = parseRecord(line)
            const rows = record && 'messages' in record ? record.messages : undefined
            if (!record || !Array.isArray(rows) || rows.some(row => !row || typeof row !== 'object' || typeof row.role !== 'string')) {
              // One bad record used to make the whole session unopenable. The
              // file stays untouched for recovery; the view skips the record.
              skipped++
              continue
            }
            stitchWindow(history, rows as Message[], false, retained)
            retained = retainedCount(record)
          }
        }
      } finally { await reader.cancel(); reader.releaseLock() }
      if (skipped) this.report(`Conversation archive ${path}: skipped ${skipped} unreadable record${skipped === 1 ? '' : 's'}; the file is preserved for recovery`)
      // An append in progress is not a complete archive record yet.
      cached = { stamp, bytes: info.size, history, retained }
      this.cache.delete(path)
      if (info.size <= 64 * 1024 * 1024) this.cache.set(path, cached)
      while (this.cache.size > 2 || [...this.cache.values()].reduce((sum, entry) => sum + entry.bytes, 0) > 64 * 1024 * 1024) this.cache.delete(this.cache.keys().next().value!)
    }
    const { history, retained } = cached
    return {
      stitch(current) {
        const view: Stitched = { messages: [...history.messages], keys: [...history.keys] }
        stitchWindow(view, current, true, retained)
        return view.messages
      },
    }
  }
}

/**
 * Parse one archive line. A crash or a full disk mid-append leaves a torn
 * record with no newline, and the next compaction's record is appended
 * straight after it; that record starts at the next record opener (inside a
 * JSON string its quotes would be escaped, so the opener cannot occur there).
 */
function parseRecord(line: string): object | undefined {
  for (let start = 0; start >= 0; start = line.indexOf(RECORD_START, start + 1)) {
    try {
      const record: unknown = JSON.parse(start ? line.slice(start) : line)
      return record && typeof record === 'object' ? record : undefined
    } catch { /* try the next record opener on this line */ }
  }
  return undefined
}
