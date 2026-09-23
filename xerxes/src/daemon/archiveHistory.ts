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

/** Stitch successive compacted windows, retaining repeated human messages. */
export function appendHistoryWindow(history: readonly Message[], window: readonly Message[], current = false): Message[] {
  if (current && !window.length) return []
  if (!history.length) return [...window]
  const summary = window.findLastIndex(isCompactionSummaryMessage)
  const next = summary < 0 ? window : window.slice(summary + 1)
  if (!next.length) return [...history]
  const oldKeys = history.map(identity), newKeys = next.map(identity)
  if (summary < 0) {
    let prefix = 0
    while (prefix < Math.min(history.length, next.length) && oldKeys[prefix] === newKeys[prefix]) prefix++
    if (prefix === history.length || prefix === next.length) return current && prefix === next.length ? [...next] : [...history, ...next.slice(prefix)]
  }
  if (current && summary >= 0) {
    // Undo can shorten the retained tail. Respect its last surviving anchor
    // instead of resurrecting later archived turns on the next reopen.
    for (let end = history.length; end >= next.length; end--) {
      if (newKeys.every((key, i) => key === oldKeys[end - next.length + i])) return history.slice(0, end)
    }
  }
  for (let count = Math.min(history.length, next.length); count > 0; count--) {
    if (newKeys.slice(0, count).every((key, i) => key === oldKeys[history.length - count + i])) {
      return [...history, ...next.slice(count)]
    }
  }
  // A zero-retention compaction has no overlap; every subsequent row is new.
  return [...history, ...next]
}

/** Read-only archive projection. Never replace the provider's compact context. */
export class ArchiveHistory {
  private cache = new Map<string, { stamp: string; bytes: number; messages: Message[] }>()
  async messages(path: string | undefined, current: readonly Message[]): Promise<Message[]> {
    if (!path) return [...current]
    const info = await lstat(path).catch(error => {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
      throw error
    })
    if (!info) return [...current]
    if (!info.isFile()) throw new Error('Conversation archive must be a regular file')
    if (info.size > 256 * 1024 * 1024) throw new Error('Conversation archive exceeds 256 MiB; export it before loading older history')
    const stamp = `${info.ino}:${info.size}:${info.mtimeMs}`
    let cached = this.cache.get(path)
    if (cached?.stamp !== stamp) {
      let messages: Message[] = [], pending = ''
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
            let record: unknown
            try { record = JSON.parse(line) } catch { throw new Error('Conversation archive contains an unreadable record; preserve it for recovery') }
            const rows = record && typeof record === 'object' && 'messages' in record ? record.messages : undefined
            if (!Array.isArray(rows) || rows.some(row => !row || typeof row !== 'object' || typeof row.role !== 'string')) throw new Error('Conversation archive contains invalid messages; preserve it for recovery')
            messages = appendHistoryWindow(messages, rows as Message[])
          }
        }
      } finally { await reader.cancel(); reader.releaseLock() }
      // An append in progress is not a complete archive record yet.
      cached = { stamp, bytes: info.size, messages }
      this.cache.delete(path)
      if (info.size <= 64 * 1024 * 1024) this.cache.set(path, cached)
      while (this.cache.size > 2 || [...this.cache.values()].reduce((sum, entry) => sum + entry.bytes, 0) > 64 * 1024 * 1024) this.cache.delete(this.cache.keys().next().value!)
    }
    return appendHistoryWindow(cached.messages, current, true)
  }
}
