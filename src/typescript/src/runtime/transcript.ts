// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface TranscriptEntryOptions {
  readonly content: string
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly role: string
  readonly timestamp?: string
}

/** Immutable message record retained by the in-memory transcript store. */
export class TranscriptEntry {
  readonly content: string
  readonly metadata: Readonly<Record<string, unknown>>
  readonly role: string
  readonly timestamp: string

  constructor(options: TranscriptEntryOptions) {
    this.role = textValue(options.role, 'role')
    this.content = textValue(options.content, 'content')
    this.timestamp = timestampValue(options.timestamp ?? new Date().toISOString())
    this.metadata = freezeRecord(options.metadata ?? {})
    Object.freeze(this)
  }
}

export interface TranscriptStoreOptions {
  readonly now?: () => Date
}

/** Ordered transcript with replay, provider-message, and compact display helpers. */
export class TranscriptStore {
  private readonly clock: () => Date
  private readonly ledger: TranscriptEntry[] = []
  private persisted = false

  constructor(options: TranscriptStoreOptions = {}) {
    this.clock = options.now ?? (() => new Date())
  }

  get entries(): readonly TranscriptEntry[] {
    return this.ledger.map(copyEntry)
  }

  get flushed(): boolean {
    return this.persisted
  }

  get turnCount(): number {
    return this.ledger.filter(entry => entry.role === 'user').length
  }

  get messageCount(): number {
    return this.ledger.length
  }

  append(role: string, content: string, metadata: Readonly<Record<string, unknown>> = {}): void {
    this.ledger.push(new TranscriptEntry({
      role,
      content,
      metadata,
      timestamp: this.nowTimestamp(),
    }))
    this.persisted = false
  }

  appendEntry(entry: TranscriptEntry): void {
    if (!(entry instanceof TranscriptEntry)) throw new TypeError('entry must be a TranscriptEntry')
    this.ledger.push(copyEntry(entry))
    this.persisted = false
  }

  replay(): readonly TranscriptEntry[] {
    return Object.freeze(this.ledger.map(copyEntry))
  }

  /** Merge provider-specific metadata onto each basic role/content message. */
  toMessages(): ReadonlyArray<Readonly<Record<string, unknown>>> {
    return this.ledger.map(entry => Object.freeze({
      role: entry.role,
      content: entry.content,
      ...cloneRecord(entry.metadata),
    }))
  }

  flush(): void {
    this.persisted = true
  }

  clear(): void {
    this.ledger.length = 0
    this.persisted = false
  }

  asMarkdown(): string {
    const lines = ['# Transcript', '', 'Messages: ' + this.messageCount, '']
    for (const entry of this.ledger) {
      const preview = entry.content.length > 200 ? entry.content.slice(0, 200) + '...' : entry.content
      lines.push('- **' + entry.role + '**: ' + preview)
    }
    return lines.join('\n')
  }

  private nowTimestamp(): string {
    const now = this.clock()
    if (!(now instanceof Date) || Number.isNaN(now.valueOf())) throw new RangeError('now must return a valid Date')
    return now.toISOString()
  }
}

function copyEntry(entry: TranscriptEntry): TranscriptEntry {
  return new TranscriptEntry({
    role: entry.role,
    content: entry.content,
    timestamp: entry.timestamp,
    metadata: entry.metadata,
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
