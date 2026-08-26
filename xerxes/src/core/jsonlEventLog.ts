// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { open } from 'node:fs/promises'
import type { FileHandle } from 'node:fs/promises'

/**
 * Shared reader for append-only JSONL event logs.
 *
 * Three of these logs — durable tasks, the scheduler, and governed memory —
 * each threw on the first unparseable line, and each ran that parse inside
 * every mutation. So a crash midway through an `appendFile` left a partial
 * final line, and from then on every read AND every write of that log threw,
 * permanently: the failure mode was total loss of a log whose entire purpose is
 * surviving a crash.
 *
 * A torn tail is the one kind of damage a crash can actually produce, because
 * appends are sequential — so it is recoverable and everything before it is
 * intact. Damage anywhere earlier means something other than a crash wrote to
 * the file, which is not something to paper over.
 */

export interface JsonlEventLogResult<T> {
  readonly events: readonly T[]
  /** Bytes of a trailing partial record that was skipped, 0 when the log was clean. */
  readonly truncatedTailBytes: number
}

export interface JsonlEventLogOptions<T> {
  /** Names the log in the error message a mid-log corruption raises. */
  readonly label: string
  readonly isValid: (value: unknown) => value is T
}

/**
 * Parse an append-only JSONL log, tolerating only a torn final record.
 *
 * Throws on malformed or invalid records anywhere but the last line.
 */
export function parseJsonlEventLog<T>(
  text: string,
  options: JsonlEventLogOptions<T>,
): JsonlEventLogResult<T> {
  const lines = text.split('\n')
  const events: T[] = []
  let truncatedTailBytes = 0

  for (const [index, line] of lines.entries()) {
    if (!line.trim()) continue
    // Only a final line with no terminating newline can be a torn append; a
    // short line in the middle was fully written and then damaged.
    const isUnterminatedFinalLine = index === lines.length - 1

    let raw: unknown
    try {
      raw = JSON.parse(line) as unknown
    } catch {
      if (isUnterminatedFinalLine) {
        truncatedTailBytes = Buffer.byteLength(line, 'utf8')
        break
      }
      throw new Error(`malformed ${options.label}`)
    }

    if (!options.isValid(raw)) {
      if (isUnterminatedFinalLine) {
        truncatedTailBytes = Buffer.byteLength(line, 'utf8')
        break
      }
      throw new Error(`invalid ${options.label} record`)
    }
    events.push(raw)
  }

  return { events, truncatedTailBytes }
}

/**
 * Drop a partial trailing record so the next append starts on a clean line.
 *
 * Tolerating a torn tail on read is only half the repair. The next append
 * concatenates onto whatever bytes are already there, so a partial final record
 * fuses with the new one and produces a malformed line in the MIDDLE of the log
 * — which is genuine corruption, no longer recoverable as a torn tail, and
 * bricks the log exactly as before. Truncating first is what makes the recovery
 * durable rather than momentary.
 *
 * A no-op for a log that is absent, empty, or already newline-terminated.
 */
export async function truncateTornTail(path: string): Promise<number> {
  let handle: FileHandle
  try {
    handle = await open(path, 'r+')
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return 0
    throw error
  }
  try {
    const { size } = await handle.stat()
    if (size === 0) return 0
    const tail = Buffer.alloc(1)
    await handle.read(tail, 0, 1, size - 1)
    if (tail.toString('utf8') === '\n') return 0

    // Walk back to the last newline; everything after it was never completed.
    const window = Math.min(size, TORN_TAIL_SCAN_BYTES)
    const buffer = Buffer.alloc(window)
    await handle.read(buffer, 0, window, size - window)
    const lastNewline = buffer.lastIndexOf('\n'.charCodeAt(0))
    // No newline in the window means the whole log is one unterminated record.
    const keep = lastNewline === -1 ? (window === size ? 0 : size) : size - window + lastNewline + 1
    if (keep === size) return 0
    await handle.truncate(keep)
    return size - keep
  } finally {
    await handle.close()
  }
}

/** How far back to look for the newline that ends the last complete record. */
const TORN_TAIL_SCAN_BYTES = 1024 * 1024
