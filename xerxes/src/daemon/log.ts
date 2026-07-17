// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile, mkdir } from 'node:fs/promises'

import type { JsonValue } from '../types/toolCalls.js'

/** Caller-owned destination for short daemon activity lines. */
export interface DaemonLogOutput {
  write(chunk: string): void | Promise<void>
}

/** File operations needed to append records to one daily JSONL file. */
export interface DaemonLogFiles {
  appendText(path: string, content: string): Promise<void>
  ensureDirectory(directory: string): Promise<void>
}

/** Bun filesystem implementation. Appends use O_APPEND so writes stay O(1) per record. */
export class BunDaemonLogFiles implements DaemonLogFiles {
  async appendText(path: string, content: string): Promise<void> {
    await appendFile(path, content, 'utf8')
  }

  async ensureDirectory(directory: string): Promise<void> {
    await mkdir(directory, { recursive: true })
  }
}

export type DaemonLogFields = Readonly<Record<string, JsonValue>>
export type DaemonLogRecord = Readonly<Record<string, JsonValue>>

export interface DaemonLoggerOptions {
  /** Required so log timestamps remain host-controlled and testable. */
  readonly clock: () => Date
  readonly directory: string
  readonly files?: DaemonLogFiles
  /** Required output mirror; the logger never implicitly reads process stderr. */
  readonly output: DaemonLogOutput
}

interface PendingDaemonLog {
  readonly date: string
  readonly event: string
  readonly level: string
  readonly record: DaemonLogRecord
}

/**
 * Daily-rotated daemon JSONL logger.
 *
 * Records are serialized through one in-process queue and appended to the
 * current daily file with an O_APPEND write, so each record costs one small
 * write instead of a full-file read/replace cycle.
 */
export class DaemonLogger {
  readonly #clock: () => Date
  readonly #directory: string
  readonly #files: DaemonLogFiles
  readonly #output: DaemonLogOutput
  #closed = false
  #currentPath: string | undefined
  #directoryReady = false
  #tail: Promise<void> = Promise.resolve()

  constructor(options: DaemonLoggerOptions) {
    this.#directory = requiredDirectory(options.directory)
    this.#clock = options.clock
    this.#output = options.output
    this.#files = options.files ?? new BunDaemonLogFiles()
  }

  /** Path selected for the most recently persisted UTC-day record. */
  get currentPath(): string | undefined {
    return this.#currentPath
  }

  async log(level: string, event: string, fields: DaemonLogFields = {}): Promise<DaemonLogRecord> {
    if (this.#closed) {
      throw new Error('daemon logger is closed')
    }
    const pending = pendingRecord(this.#clock, level, event, fields)
    const operation = this.#tail.then(async () => {
      await this.writePending(pending)
    })
    this.#tail = operation.catch(() => undefined)
    await operation
    return pending.record
  }

  async info(event: string, fields: DaemonLogFields = {}): Promise<DaemonLogRecord> {
    return this.log('info', event, fields)
  }

  async error(event: string, fields: DaemonLogFields = {}): Promise<DaemonLogRecord> {
    return this.log('error', event, fields)
  }

  /** Wait for already-queued writes and reject further log calls. */
  async close(): Promise<void> {
    this.#closed = true
    await this.#tail
  }

  private async writePending(pending: PendingDaemonLog): Promise<void> {
    if (!this.#directoryReady) {
      await this.#files.ensureDirectory(this.#directory)
      this.#directoryReady = true
    }
    const target = daemonLogPath(this.#directory, pending.date)
    await this.#files.appendText(target, `${JSON.stringify(pending.record)}\n`)
    this.#currentPath = target
    await this.#output.write(`[${pending.level}] ${pending.event}\n`)
  }
}

/** Return the Python-compatible UTC daily filename without touching the filesystem. */
export function daemonLogPath(directory: string, date: string): string {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    throw new TypeError('daemon log date must use YYYY-MM-DD format')
  }
  return joinPath(requiredDirectory(directory), `daemon-${date}.jsonl`)
}

function pendingRecord(
  clock: () => Date,
  level: string,
  event: string,
  fields: DaemonLogFields,
): PendingDaemonLog {
  const timestamp = daemonTimestamp(clock)
  const normalizedLevel = requiredLabel(level, 'daemon log level')
  const normalizedEvent = requiredLabel(event, 'daemon log event')
  return {
    date: timestamp.slice(0, 10),
    level: normalizedLevel,
    event: normalizedEvent,
    // Match Python's `{ts, level, event, **kwargs}` merge order.
    record: Object.freeze({
      ts: timestamp,
      level: normalizedLevel,
      event: normalizedEvent,
      ...copyJsonFields(fields),
    }),
  }
}

function daemonTimestamp(clock: () => Date): string {
  const now = clock()
  if (!(now instanceof Date) || !Number.isFinite(now.getTime())) {
    throw new TypeError('daemon logger clock must return a valid Date')
  }
  return now.toISOString()
}

function requiredDirectory(value: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError('daemon log directory must be a non-empty string')
  }
  return value.replace(/\/+$/, '') || '/'
}

function requiredLabel(value: string, name: string): string {
  if (typeof value !== 'string' || !value.trim() || /[\r\n]/.test(value)) {
    throw new TypeError(`${name} must be a non-empty single-line string`)
  }
  return value
}

function joinPath(directory: string, fileName: string): string {
  return directory === '/' ? `/${fileName}` : `${directory}/${fileName}`
}

function copyJsonFields(fields: DaemonLogFields): Record<string, JsonValue> {
  const copied: Record<string, JsonValue> = {}
  for (const [key, value] of Object.entries(fields)) {
    copied[key] = copyJsonValue(value)
  }
  return copied
}

function copyJsonValue(value: unknown): JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value
  if (typeof value === 'number') {
    if (Number.isFinite(value)) return value
    throw new TypeError('daemon log fields must contain JSON-serializable values')
  }
  if (Array.isArray(value)) {
    return Object.freeze(value.map(copyJsonValue)) as unknown as JsonValue
  }
  if (isPlainRecord(value)) {
    const copied: Record<string, JsonValue> = {}
    for (const [key, nested] of Object.entries(value)) {
      copied[key] = copyJsonValue(nested)
    }
    return Object.freeze(copied) as JsonValue
  }
  throw new TypeError('daemon log fields must contain JSON-serializable values')
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}
