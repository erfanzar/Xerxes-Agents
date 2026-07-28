// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Output plumbing shared by foreground and background command execution.
//
// The reason this exists rather than `new Response(stream).text()`: that helper
// resolves only at EOF, and EOF on a child's stdout means *every* holder of the
// write end has closed it — not merely the process we spawned. A command that
// backgrounds anything (`cmd &`, `nohup cmd`) hands a copy of that pipe to a
// process which outlives the one we can kill, so the read never completes.
//
// The consequence was not a slow tool call, it was an unbounded one: the timeout
// fired, killed the direct child, set `timedOut`, and then the call sat forever
// on a read that could not finish. A single stray `&` stalled a whole turn.
//
// So output is drained through a reader we hold, into a bounded buffer, and the
// call is free to return whatever arrived without waiting for anyone to close
// anything.

/** Chars kept per stream before the oldest are dropped. */
const DEFAULT_CAPACITY = 1_000_000

/**
 * A bounded, incrementally drainable text buffer.
 *
 * Keeps the tail rather than the head: for a long-running command the recent
 * output is what a caller polling it needs, and the alternative — refusing to
 * read further once full — would block the child on a full pipe.
 */
export class BoundedOutputBuffer {
  private chunks: string[] = []
  private droppedChars = 0
  private length = 0

  constructor(private readonly capacity: number = DEFAULT_CAPACITY) {}

  /** Whether any output was discarded to stay within capacity. */
  get dropped(): boolean {
    return this.droppedChars > 0
  }

  get size(): number {
    return this.length
  }

  append(text: string): void {
    if (!text) return
    this.chunks.push(text)
    this.length += text.length
    while (this.length > this.capacity && this.chunks.length > 0) {
      const first = this.chunks[0]
      if (first === undefined) break
      const excess = this.length - this.capacity
      if (first.length <= excess) {
        this.chunks.shift()
        this.length -= first.length
        this.droppedChars += first.length
      } else {
        this.chunks[0] = first.slice(excess)
        this.length -= excess
        this.droppedChars += excess
      }
    }
  }

  /** Read and remove up to `maxChars`, so repeated polls see only new output. */
  take(maxChars: number): { readonly text: string; readonly truncated: boolean } {
    const joined = this.chunks.join('')
    this.chunks = []
    this.length = 0
    if (joined.length <= maxChars) {
      return { text: joined, truncated: false }
    }
    // Keep the remainder for the next poll rather than discarding it: a caller
    // reading a chatty process in pages must not lose the pages it has not read.
    const head = joined.slice(0, maxChars)
    const rest = joined.slice(maxChars)
    this.chunks = [rest]
    this.length = rest.length
    return { text: head, truncated: true }
  }

  /** Read without consuming, for a final snapshot. */
  peek(maxChars: number): { readonly text: string; readonly truncated: boolean } {
    const joined = this.chunks.join('')
    return joined.length <= maxChars
      ? { text: joined, truncated: false }
      : { text: joined.slice(0, maxChars), truncated: true }
  }
}

/** A drain in progress, which the owner can abandon without waiting for EOF. */
export interface StreamDrain {
  /**
   * Stop reading and release our end of the pipe.
   *
   * Deliberately does not wait for the writer: a surviving grandchild may hold
   * the write end open indefinitely, and that is precisely the case this whole
   * module exists to survive.
   */
  cancel(): void
  /** Resolves at EOF, or when cancelled. Never rejects. */
  readonly done: Promise<void>
}

/**
 * Drain a child stream into `buffer` until EOF or cancellation.
 *
 * Decoding is streaming, so a multi-byte character split across chunk boundaries
 * is not corrupted into replacement characters.
 */
export function drainStream(
  stream: ReadableStream<Uint8Array> | null | undefined,
  buffer: BoundedOutputBuffer,
): StreamDrain {
  if (!stream) {
    return { done: Promise.resolve(), cancel: () => {} }
  }
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let cancelled = false

  const done = (async () => {
    try {
      for (;;) {
        const { done: finished, value } = await reader.read()
        if (finished || cancelled) break
        if (value) buffer.append(decoder.decode(value, { stream: true }))
      }
      if (!cancelled) {
        const tail = decoder.decode()
        if (tail) buffer.append(tail)
      }
    } catch {
      // A closed or errored pipe is a normal end of output, not a tool failure.
    } finally {
      try {
        reader.releaseLock()
      } catch {
        // Already released by cancel(); nothing to do.
      }
    }
  })()

  return {
    done,
    cancel: () => {
      cancelled = true
      // cancel() rejects if the reader is mid-read on some streams; the read
      // loop above exits on the `cancelled` flag regardless.
      void reader.cancel().catch(() => {})
    },
  }
}

/** Cap text for a tool response, marking that it was cut. */
export function capOutput(output: string, maxChars: number): { readonly text: string; readonly truncated: boolean } {
  if (output.length <= maxChars) {
    return { text: output, truncated: false }
  }
  return { text: `${output.slice(0, maxChars)}\n…[truncated]…`, truncated: true }
}
