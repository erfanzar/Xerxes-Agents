// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdirSync, renameSync, statSync } from 'node:fs'
import { appendFile } from 'node:fs/promises'
import { dirname } from 'node:path'

import type { AuditEvent } from './events.js'

const AUDIT_FILE_MODE = 0o600
const AUDIT_DIRECTORY_MODE = 0o700

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

export interface JSONLSinkOptions {
  /** Rotate the audit file once a batch would push it past this size. */
  readonly maxBytes?: number
  /** Number of rotated `.N` files retained alongside the active file. */
  readonly maxFiles?: number
  /** Bounded in-memory queue; events beyond it are dropped and counted. */
  readonly maxQueueSize?: number
  /** Interval used to drain buffered events to disk. */
  readonly flushIntervalMs?: number
}

/**
 * Appends one audit event JSON object per line to a file path or caller-owned text sink.
 *
 * File-backed sinks never perform synchronous per-event I/O: events enter a bounded
 * in-memory queue that a background drain writes asynchronously (mode `0600`, parent
 * directory `0700`), rotating the file when it grows past `maxBytes`. When the queue
 * is full, new events are dropped and counted on {@link droppedEvents} instead of
 * stalling the agent loop.
 */
export class JSONLSinkCollector implements AuditCollector {
  static readonly DEFAULT_MAX_BYTES = 10 * 1024 * 1024
  static readonly DEFAULT_MAX_FILES = 5
  static readonly DEFAULT_MAX_QUEUE_SIZE = 10_000
  static readonly DEFAULT_FLUSH_INTERVAL_MS = 250

  private closed = false
  private readonly filePath: string | undefined
  private readonly stream: AuditTextSink | undefined
  private readonly maxBytes: number
  private readonly maxFiles: number
  private readonly maxQueueSize: number
  private readonly flushIntervalMs: number
  private queue: string[] = []
  private draining: Promise<void> | undefined
  private flushTimer: ReturnType<typeof setTimeout> | undefined
  private currentSize: number | undefined
  private dropped = 0
  private writeFailures = 0
  private lastWriteError: unknown

  constructor(sink: string | AuditTextSink, options: JSONLSinkOptions = {}) {
    this.maxBytes = positiveInteger(options.maxBytes, JSONLSinkCollector.DEFAULT_MAX_BYTES, 'maxBytes')
    this.maxFiles = positiveInteger(options.maxFiles, JSONLSinkCollector.DEFAULT_MAX_FILES, 'maxFiles')
    this.maxQueueSize = positiveInteger(options.maxQueueSize, JSONLSinkCollector.DEFAULT_MAX_QUEUE_SIZE, 'maxQueueSize')
    this.flushIntervalMs = positiveInteger(options.flushIntervalMs, JSONLSinkCollector.DEFAULT_FLUSH_INTERVAL_MS, 'flushIntervalMs')
    if (typeof sink === 'string') {
      mkdirSync(dirname(sink), { recursive: true, mode: AUDIT_DIRECTORY_MODE })
      this.filePath = sink
      this.stream = undefined
      return
    }
    this.filePath = undefined
    this.stream = sink
  }

  /** Events discarded because the bounded queue was full. */
  get droppedEvents(): number {
    return this.dropped
  }

  /** Background write batches that failed (writes are best effort and never throw into the loop). */
  get failedWriteBatches(): number {
    return this.writeFailures
  }

  get lastError(): unknown {
    return this.lastWriteError
  }

  get pendingCount(): number {
    return this.queue.length
  }

  emit(event: AuditEvent): void {
    if (this.closed) throw new Error('Cannot emit to a closed audit collector')
    const line = `${event.toJson()}\n`
    if (this.filePath !== undefined) {
      if (this.queue.length >= this.maxQueueSize) {
        this.dropped += 1
        return
      }
      this.queue.push(line)
      this.scheduleFlush()
      return
    }
    this.stream?.write(line)
  }

  flush(): void {
    if (this.filePath !== undefined) {
      void this.drain()
      return
    }
    if (this.closed) return
    this.stream?.flush?.()
  }

  /** Write every queued event to disk, awaiting any in-flight batch first. */
  drain(): Promise<void> {
    if (this.filePath === undefined) {
      if (!this.closed) this.stream?.flush?.()
      return Promise.resolve()
    }
    if (this.draining !== undefined) {
      return this.draining.then(() => this.drain())
    }
    const batch = this.queue
    if (batch.length === 0) return Promise.resolve()
    this.queue = []
    this.draining = this.writeBatch(batch.join(''))
      .catch((error: unknown) => {
        // Slow/full filesystems must surface as counters, never as agent-loop stalls.
        this.writeFailures += 1
        this.lastWriteError = error
      })
      .finally(() => {
        this.draining = undefined
      })
    return this.draining.then(() => this.drain())
  }

  /** Mark this sink closed and flush pending events. Caller-owned streams stay open. */
  async close(): Promise<void> {
    if (this.closed) return
    this.closed = true
    if (this.flushTimer !== undefined) {
      clearTimeout(this.flushTimer)
      this.flushTimer = undefined
    }
    if (this.filePath !== undefined) {
      await this.drain()
      return
    }
    this.stream?.flush?.()
  }

  private scheduleFlush(): void {
    if (this.flushTimer !== undefined) return
    this.flushTimer = setTimeout(() => {
      this.flushTimer = undefined
      void this.drain()
    }, this.flushIntervalMs)
    this.flushTimer.unref?.()
  }

  private async writeBatch(data: string): Promise<void> {
    const path = this.filePath as string
    const bytes = Buffer.byteLength(data, 'utf8')
    this.currentSize ??= fileSize(path)
    if (this.currentSize > 0 && this.currentSize + bytes > this.maxBytes) {
      rotateAuditFiles(path, this.maxFiles)
      this.currentSize = 0
    }
    await appendFile(path, data, { encoding: 'utf8', mode: AUDIT_FILE_MODE })
    this.currentSize += bytes
  }
}

function fileSize(path: string): number {
  try {
    return statSync(path).size
  } catch {
    return 0
  }
}

function rotateAuditFiles(path: string, maxFiles: number): void {
  for (let index = maxFiles - 1; index >= 1; index -= 1) {
    try {
      renameSync(`${path}.${index}`, `${path}.${index + 1}`)
    } catch {
      // A missing rotated segment is fine.
    }
  }
  try {
    renameSync(path, `${path}.1`)
  } catch {
    // A missing active file is fine.
  }
}

function positiveInteger(value: number | undefined, fallback: number, name: string): number {
  if (value === undefined) return fallback
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`JSONLSinkCollector ${name} must be a positive integer`)
  }
  return value
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
