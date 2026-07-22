// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** A completed Server-Sent Events record. */
export interface SSEEvent {
  readonly data: string
  readonly event: string
  readonly id: string
  readonly retry: number | undefined
}

/**
 * Incrementally parse a Server-Sent Events text stream.
 *
 * Feed decoded text chunks as they arrive, then retrieve complete records with
 * {@link drain}. `lastEventId` is available to callers that reconnect with a
 * `Last-Event-ID` request header.
 */
export class SSEParser {
  private buffer = ''
  private completed: SSEEvent[] = []
  private currentData: string[] = []
  private currentEvent = 'message'
  private currentId = ''
  private currentRetry: number | undefined

  lastEventId = ''

  /** Buffer raw text and process every complete line. */
  feed(chunk: string): void {
    this.buffer += chunk
    for (;;) {
      const newline = this.buffer.indexOf('\n')
      const carriage = this.buffer.indexOf('\r')
      let end = -1
      let skip = 1
      if (newline >= 0 && (carriage < 0 || newline < carriage)) {
        end = newline
      } else if (carriage >= 0) {
        // A bare '\r' terminates a line, but a trailing '\r' may be half of a
        // '\r\n' pair split across chunks, so wait for the next chunk.
        if (carriage === this.buffer.length - 1) break
        end = carriage
        skip = this.buffer[carriage + 1] === '\n' ? 2 : 1
      } else {
        break
      }
      const line = this.buffer.slice(0, end)
      this.buffer = this.buffer.slice(end + skip)
      this.handleLine(line)
    }
  }

  /** Return and clear the records completed since the preceding drain. */
  drain(): SSEEvent[] {
    const completed = this.completed
    this.completed = []
    return completed
  }

  private handleLine(line: string): void {
    if (!line) {
      this.dispatch()
      return
    }
    if (line.startsWith(':')) {
      return
    }

    const separator = line.indexOf(':')
    const field = separator < 0 ? line : line.slice(0, separator)
    const rawValue = separator < 0 ? '' : line.slice(separator + 1)
    // The SSE spec removes exactly one leading space after the colon.
    const value = rawValue.startsWith(' ') ? rawValue.slice(1) : rawValue
    if (field === 'event') {
      this.currentEvent = value
    } else if (field === 'data') {
      this.currentData.push(value)
    } else if (field === 'id') {
      this.currentId = value
    } else if (field === 'retry') {
      this.currentRetry = parseRetry(value)
    }
  }

  private dispatch(): void {
    // Per the WHATWG SSE spec, an event is never dispatched with an empty
    // data buffer, regardless of the event type.
    if (!this.currentData.length) {
      this.resetCurrentRecord()
      return
    }

    const event: SSEEvent = {
      event: this.currentEvent,
      data: this.currentData.join('\n'),
      id: this.currentId,
      retry: this.currentRetry,
    }
    this.completed.push(event)
    if (event.id) {
      this.lastEventId = event.id
    }
    this.resetCurrentRecord()
  }

  private resetCurrentRecord(): void {
    this.currentData = []
    this.currentEvent = 'message'
    this.currentId = ''
    this.currentRetry = undefined
  }
}

/**
 * Parse a finite sequence of decoded SSE chunks.
 *
 * A final blank record delimiter flushes an otherwise unterminated final
 * event, matching the incremental parser's normal record boundary behavior.
 */
export function* parseSseStream(chunks: Iterable<string>): Generator<SSEEvent> {
  const parser = new SSEParser()
  for (const chunk of chunks) {
    parser.feed(chunk)
    yield* parser.drain()
  }
  parser.feed('\n\n')
  yield* parser.drain()
}

function parseRetry(value: string): number | undefined {
  const normalized = value.trim()
  if (!/^[+-]?\d+$/.test(normalized)) {
    return undefined
  }
  const retry = Number(normalized)
  return Number.isSafeInteger(retry) ? retry : undefined
}
