// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFileSync, mkdirSync } from 'node:fs'
import { dirname } from 'node:path'

import type { AuditEvent } from './events.js'

/** Minimal synchronous contract implemented by every audit destination. */
export interface AuditCollector {
  emit(event: AuditEvent): void
  flush(): void
}

/** A caller-owned synchronous text sink, useful for embedding or tests. */
export interface AuditTextSink {
  flush?(): void
  write(value: string): void
}

export interface InMemoryCollectorOptions {
  readonly maxEvents?: number
}

/** Bounded in-process event collector intended for tests, session views, and fan-out. */
export class InMemoryCollector implements AuditCollector {
  static readonly MAX_IN_MEMORY_EVENTS = 10_000

  private events: AuditEvent[] = []
  readonly maxEvents: number

  constructor(options: InMemoryCollectorOptions = {}) {
    const maxEvents = options.maxEvents ?? InMemoryCollector.MAX_IN_MEMORY_EVENTS
    if (!Number.isInteger(maxEvents) || maxEvents <= 0) {
      throw new Error('InMemoryCollector maxEvents must be a positive integer')
    }
    this.maxEvents = maxEvents
  }

  emit(event: AuditEvent): void {
    this.events.push(event)
    const overflow = this.events.length - this.maxEvents
    if (overflow > 0) this.events.splice(0, overflow)
  }

  flush(): void {
    // The collector has no buffered I/O.
  }

  get size(): number {
    return this.events.length
  }

  /** Return an array copy without exposing the collector's internal ordering state. */
  getEvents(): AuditEvent[] {
    return [...this.events]
  }

  getEventsByType(eventType: string): AuditEvent[] {
    return this.events.filter(event => event.eventType === eventType)
  }

  clear(): void {
    this.events = []
  }
}

/** Appends one audit event JSON object per line to a file path or caller-owned text sink. */
export class JSONLSinkCollector implements AuditCollector {
  private closed = false
  private readonly filePath: string | undefined
  private readonly stream: AuditTextSink | undefined

  constructor(sink: string | AuditTextSink) {
    if (typeof sink === 'string') {
      mkdirSync(dirname(sink), { recursive: true })
      this.filePath = sink
      this.stream = undefined
      return
    }
    this.filePath = undefined
    this.stream = sink
  }

  emit(event: AuditEvent): void {
    if (this.closed) throw new Error('Cannot emit to a closed audit collector')
    const line = `${event.toJson()}\n`
    if (this.filePath !== undefined) {
      appendFileSync(this.filePath, line, 'utf8')
      return
    }
    this.stream?.write(line)
  }

  flush(): void {
    if (this.closed) return
    this.stream?.flush?.()
  }

  /** Mark this sink closed. Caller-owned streams are deliberately left open. */
  close(): void {
    if (this.closed) return
    this.flush()
    this.closed = true
  }
}

/** Fans each event out to child collectors in registration order. */
export class CompositeCollector implements AuditCollector {
  private readonly collectors: AuditCollector[]

  constructor(collectors: Iterable<AuditCollector> = []) {
    this.collectors = Array.from(collectors)
  }

  add(collector: AuditCollector): void {
    this.collectors.push(collector)
  }

  emit(event: AuditEvent): void {
    for (const collector of this.collectors) collector.emit(event)
  }

  flush(): void {
    for (const collector of this.collectors) collector.flush()
  }
}
