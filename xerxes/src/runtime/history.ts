// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface HistoryEventOptions {
  readonly detail?: string
  readonly kind: string
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly timestamp?: string
  readonly title: string
}

/** Immutable chronological runtime event. */
export class HistoryEvent {
  readonly detail: string
  readonly kind: string
  readonly metadata: Readonly<Record<string, unknown>>
  readonly timestamp: string
  readonly title: string

  constructor(options: HistoryEventOptions) {
    this.kind = textValue(options.kind, 'kind')
    this.title = textValue(options.title, 'title')
    this.detail = textValue(options.detail ?? '', 'detail')
    this.timestamp = timestampValue(options.timestamp ?? new Date().toISOString())
    this.metadata = freezeRecord(options.metadata ?? {})
    Object.freeze(this)
  }

  toRecord(): Readonly<Record<string, unknown>> {
    return Object.freeze({
      kind: this.kind,
      title: this.title,
      detail: this.detail,
      timestamp: this.timestamp,
      ...cloneRecord(this.metadata),
    })
  }
}

export interface HistoryLogOptions {
  readonly now?: () => Date
}

/** Ordered append-only session history suitable for display and persistence. */
export class HistoryLog {
  private readonly clock: () => Date
  private readonly ledger: HistoryEvent[] = []

  constructor(options: HistoryLogOptions = {}) {
    this.clock = options.now ?? (() => new Date())
  }

  get events(): readonly HistoryEvent[] {
    return this.ledger.map(event => copyEvent(event))
  }

  get eventCount(): number {
    return this.ledger.length
  }

  add(
    kind: string,
    title: string,
    detail = '',
    metadata: Readonly<Record<string, unknown>> = {},
  ): HistoryEvent {
    const event = new HistoryEvent({
      kind,
      title,
      detail,
      metadata,
      timestamp: this.nowTimestamp(),
    })
    this.ledger.push(event)
    return copyEvent(event)
  }

  addToolCall(name: string, resultPreview = '', durationMs = 0): HistoryEvent {
    return this.add('tool_call', name, resultPreview.slice(0, 200), { duration_ms: durationMs })
  }

  addError(message: string, source = ''): HistoryEvent {
    return this.add('error', message, source)
  }

  addTurn(model: string, inputTokens = 0, outputTokens = 0): HistoryEvent {
    return this.add(
      'turn',
      'Turn completed (' + model + ')',
      'in=' + inputTokens + ', out=' + outputTokens,
      { model, in_tokens: inputTokens, out_tokens: outputTokens },
    )
  }

  addPermission(toolName: string, granted: boolean): HistoryEvent {
    const status = granted ? 'granted' : 'denied'
    return this.add('permission_' + status, toolName + ': ' + status)
  }

  filterByKind(kind: string): HistoryEvent[] {
    return this.ledger.filter(event => event.kind === kind).map(copyEvent)
  }

  last(count = 10): HistoryEvent[] {
    if (!Number.isInteger(count)) throw new RangeError('count must be an integer')
    return this.ledger.slice(-Math.max(0, count)).map(copyEvent)
  }

  clear(): void {
    this.ledger.length = 0
  }

  asMarkdown(): string {
    const lines = ['# Session History', '', 'Events: ' + this.eventCount, '']
    for (const event of this.ledger) {
      const detail = event.detail ? ' — ' + event.detail : ''
      lines.push('- [' + event.timestamp.slice(0, 19) + '] **' + event.kind + '**: ' + event.title + detail)
    }
    return lines.join('\n')
  }

  asDicts(): ReadonlyArray<Readonly<Record<string, unknown>>> {
    return this.ledger.map(event => event.toRecord())
  }

  private nowTimestamp(): string {
    const now = this.clock()
    if (!(now instanceof Date) || Number.isNaN(now.valueOf())) throw new RangeError('now must return a valid Date')
    return now.toISOString()
  }
}

function copyEvent(event: HistoryEvent): HistoryEvent {
  return new HistoryEvent({
    kind: event.kind,
    title: event.title,
    detail: event.detail,
    timestamp: event.timestamp,
    metadata: event.metadata,
  })
}

function textValue(value: unknown, name: string): string {
  if (typeof value !== 'string') throw new TypeError(name + ' must be a string')
  return value
}

function timestampValue(value: unknown): string {
  if (typeof value !== 'string' || Number.isNaN(new Date(value).valueOf())) {
    throw new RangeError('timestamp must be a valid ISO timestamp')
  }
  return value
}

function cloneRecord(record: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> {
  return freezeRecord(record)
}

function freezeRecord(record: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> {
  const copy: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(record)) copy[key] = freezeValue(value)
  return Object.freeze(copy)
}

function freezeValue(value: unknown): unknown {
  if (Array.isArray(value)) return Object.freeze(value.map(freezeValue))
  if (value instanceof Date) return value.toISOString()
  if (value && typeof value === 'object') return freezeRecord(value as Record<string, unknown>)
  return value
}
