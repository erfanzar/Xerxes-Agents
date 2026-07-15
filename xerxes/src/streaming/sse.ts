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
    while (this.buffer.includes('\n')) {
      const newline = this.buffer.indexOf('\n')
      const line = this.buffer.slice(0, newline).replace(/\r+$/, '')
      this.buffer = this.buffer.slice(newline + 1)
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
    const value = (separator < 0 ? '' : line.slice(separator + 1)).replace(/^ +/, '')
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
    if (!this.currentData.length && this.currentEvent === 'message') {
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
