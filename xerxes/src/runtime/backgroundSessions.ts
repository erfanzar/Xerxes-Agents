// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const BackgroundStatus = Object.freeze({
  PENDING: 'pending',
  RUNNING: 'running',
  SUCCEEDED: 'succeeded',
  FAILED: 'failed',
  CANCELLED: 'cancelled',
} as const)

export type BackgroundStatus = (typeof BackgroundStatus)[keyof typeof BackgroundStatus]

export interface BackgroundSession {
  readonly error: string
  readonly finishedAt: number
  readonly id: string
  readonly metadata: Readonly<Record<string, unknown>>
  readonly prompt: string
  readonly result: string
  readonly startedAt: number
  readonly status: BackgroundStatus
}

/**
 * Runner port for one detached session. The AbortSignal is cooperative: a
 * cancelled session stays cancelled even if a runner ultimately resolves.
 */
export type BackgroundRunFn = (
  session: BackgroundSession,
  signal: AbortSignal,
) => string | Promise<string>

/** Python-compatible name for the injected background-session runner contract. */
export type RunFn = BackgroundRunFn

export interface BackgroundSessionManagerOptions {
  /** Stable ID factory, useful for deterministic tests and externally-owned IDs. */
  readonly idFactory?: () => string
  /** Maximum completed, settled records retained in memory. */
  readonly maxCompleted?: number
  /** Maximum runners active at once; values below one clamp to one. */
  readonly maxConcurrent?: number
  /** Epoch-seconds wall clock used for lifecycle timestamps. */
  readonly now?: () => number
  readonly runner: BackgroundRunFn
}

export interface SubmitBackgroundSessionOptions {
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly sessionId?: string
}

export interface ShutdownBackgroundSessionsOptions {
  /** Abort active runners before waiting. Defaults to false for Python parity. */
  readonly cancelRunning?: boolean
  /** Maximum asynchronous wait for outstanding runner cleanup. */
  readonly timeoutMs?: number
}

export interface WaitForBackgroundSessionOptions {
  /** Wait for worker cleanup after terminal cancellation, not only terminal status. */
  readonly settled?: boolean
  /** Return the latest snapshot after this timeout instead of waiting forever. */
  readonly timeoutMs?: number
}

interface Deferred {
  readonly promise: Promise<void>
  resolve(): void
}

interface ActiveSession {
  readonly controller: AbortController
  readonly promise: Promise<void>
  readonly token: number
}

interface MutableBackgroundSession {
  readonly completion: Deferred
  error: string
  finishedAt: number
  readonly id: string
  readonly metadata: Readonly<Record<string, unknown>>
  readonly prompt: string
  result: string
  readonly sequence: number
  readonly settled: Deferred
  startedAt: number
  status: BackgroundStatus
}

const DEFAULT_MAX_COMPLETED = 100
const DEFAULT_MAX_CONCURRENT = 4
const DEFAULT_SHUTDOWN_TIMEOUT_MS = 5_000
const MAX_ID_ATTEMPTS = 100

/**
 * Detached background task queue with FIFO concurrency control.
 *
 * JavaScript's event loop makes map/queue transitions atomic between awaits.
 * The manager nevertheless keeps active controllers separate from terminal
 * status: cancelling a runner marks it terminal immediately, but it still
 * consumes a concurrency slot until its promise settles.
 */
export class BackgroundSessionManager {
  private readonly active = new Map<string, ActiveSession>()
  private readonly idFactory: () => string
  private readonly maxCompleted: number
  private readonly maxConcurrent: number
  private readonly now: () => number
  private readonly queue: string[] = []
  private readonly sessions = new Map<string, MutableBackgroundSession>()
  private readonly runner: BackgroundRunFn
  private sequence = 0
  private shuttingDown = false
  private token = 0

  constructor(options: BackgroundSessionManagerOptions) {
    this.runner = options.runner
    this.maxConcurrent = boundedInteger(options.maxConcurrent ?? DEFAULT_MAX_CONCURRENT, 'maxConcurrent', 1)
    this.maxCompleted = boundedInteger(options.maxCompleted ?? DEFAULT_MAX_COMPLETED, 'maxCompleted', 0)
    this.idFactory = options.idFactory ?? defaultSessionId
    this.now = options.now ?? (() => Date.now() / 1_000)
  }

  get size(): number {
    return this.sessions.size
  }

  get runningCount(): number {
    return this.active.size
  }

  /** Submit a detached prompt. It starts immediately when a concurrency slot exists. */
  submit(prompt: string, options: SubmitBackgroundSessionOptions = {}): BackgroundSession {
    if (this.shuttingDown) throw new Error('manager is shutting down')
    const normalizedPrompt = textValue(prompt, 'prompt')
    const id = options.sessionId === undefined ? this.nextId() : requiredText(options.sessionId, 'sessionId')
    if (this.sessions.has(id)) throw new Error('background session already exists: ' + id)

    const record: MutableBackgroundSession = {
      id,
      prompt: normalizedPrompt,
      status: BackgroundStatus.PENDING,
      result: '',
      error: '',
      startedAt: 0,
      finishedAt: 0,
      metadata: freezeMetadata(options.metadata ?? {}),
      completion: deferred(),
      settled: deferred(),
      sequence: this.sequence,
    }
    this.sequence += 1
    this.sessions.set(id, record)
    this.queue.push(id)
    this.drain()
    return snapshot(record)
  }

  /** Return one immutable status snapshot, or undefined after unknown-ID lookup or retention pruning. */
  get(sessionId: string): BackgroundSession | undefined {
    const record = this.sessions.get(sessionId)
    return record === undefined ? undefined : snapshot(record)
  }

  /** Return retained sessions in original submission order. */
  listSessions(): BackgroundSession[] {
    return [...this.sessions.values()].map(snapshot)
  }

  /** Python-compatible spelling retained at the runtime boundary. */
  list_sessions(): BackgroundSession[] {
    return this.listSessions()
  }

  /**
   * Cooperatively cancel a pending or running session.
   *
   * A running task is signalled and marked cancelled immediately, while its
   * physical slot remains occupied until the runner settles. This prevents a
   * non-cooperative runner from silently breaking the concurrency limit.
   */
  cancel(sessionId: string): boolean {
    const record = this.sessions.get(sessionId)
    if (!record || !isCancellable(record.status)) return false
    this.markCancelled(record)
    const active = this.active.get(sessionId)
    if (active) active.controller.abort(new Error('Background session cancelled'))
    if (!active) {
      record.settled.resolve()
      this.pruneCompleted()
      this.drain()
    }
    return true
  }

  /**
   * Wait for a terminal status by default, or for worker cleanup with
   * settled=true. Timeouts return the current snapshot without cancelling it.
   */
  async wait(
    sessionId: string,
    options: WaitForBackgroundSessionOptions = {},
  ): Promise<BackgroundSession | undefined> {
    const record = this.sessions.get(sessionId)
    if (!record) return undefined
    const target = options.settled ? record.settled.promise : record.completion.promise
    await waitWithTimeout(target, options.timeoutMs)
    return snapshot(record)
  }

  /**
   * Stop accepting work and wait for active runners.
   *
   * Pending work is cancelled because it can no longer be promoted after
   * shutdown. Running work is only aborted when cancelRunning is requested,
   * preserving the Python manager's cooperative shutdown default.
   */
  async shutdown(options: ShutdownBackgroundSessionsOptions = {}): Promise<void> {
    if (this.shuttingDown) return
    this.shuttingDown = true
    for (const record of [...this.sessions.values()]) {
      if (record.status === BackgroundStatus.PENDING) this.cancel(record.id)
    }
    if (options.cancelRunning) {
      for (const record of this.sessions.values()) {
        if (record.status === BackgroundStatus.RUNNING) this.cancel(record.id)
      }
    }
    const active = [...this.active.values()].map(entry => entry.promise)
    const timeoutMs = nonNegativeInteger(options.timeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS, 'timeoutMs')
    await waitWithTimeout(Promise.allSettled(active).then(() => undefined), timeoutMs)
    this.pruneCompleted()
  }

  private drain(): void {
    if (this.shuttingDown) return
    while (this.active.size < this.maxConcurrent) {
      const record = this.nextPending()
      if (!record) return
      this.start(record)
    }
  }

  private nextPending(): MutableBackgroundSession | undefined {
    while (this.queue.length) {
      const id = this.queue.shift()
      if (!id) continue
      const record = this.sessions.get(id)
      if (record?.status === BackgroundStatus.PENDING) return record
    }
    return undefined
  }

  private start(record: MutableBackgroundSession): void {
    record.status = BackgroundStatus.RUNNING
    record.startedAt = this.timestamp()
    const controller = new AbortController()
    const token = this.token + 1
    this.token = token
    const runnerSession = snapshot(record)
    const promise = Promise.resolve()
      .then(() => this.runner(runnerSession, controller.signal))
      .then(
        result => this.succeed(record.id, token, result),
        error => this.fail(record.id, token, error),
      )
      .catch(error => this.fail(record.id, token, error))
      .finally(() => this.settle(record.id, token))
    this.active.set(record.id, { controller, promise, token })
  }

  private succeed(id: string, token: number, result: unknown): void {
    const record = this.currentRecord(id, token)
    if (!record || record.status === BackgroundStatus.CANCELLED) return
    if (typeof result !== 'string') {
      this.fail(id, token, new TypeError('background runner must return a string'))
      return
    }
    record.result = result
    record.error = ''
    record.status = BackgroundStatus.SUCCEEDED
    record.finishedAt = this.timestamp()
    record.completion.resolve()
  }

  private fail(id: string, token: number, error: unknown): void {
    const record = this.currentRecord(id, token)
    if (!record || record.status === BackgroundStatus.CANCELLED) return
    record.result = ''
    record.error = errorText(error)
    record.status = BackgroundStatus.FAILED
    record.finishedAt = this.timestamp()
    record.completion.resolve()
  }

  private settle(id: string, token: number): void {
    const current = this.active.get(id)
    if (!current || current.token !== token) return
    this.active.delete(id)
    const record = this.sessions.get(id)
    if (record) {
      if (record.status === BackgroundStatus.RUNNING) {
        record.status = BackgroundStatus.FAILED
        record.error = 'Error: background runner settled without a result'
        record.finishedAt = this.timestamp()
        record.completion.resolve()
      }
      if (record.finishedAt === 0 && isTerminal(record.status)) record.finishedAt = this.timestamp()
      record.settled.resolve()
    }
    this.pruneCompleted()
    this.drain()
  }

  private currentRecord(id: string, token: number): MutableBackgroundSession | undefined {
    const active = this.active.get(id)
    if (!active || active.token !== token) return undefined
    return this.sessions.get(id)
  }

  private markCancelled(record: MutableBackgroundSession): void {
    record.status = BackgroundStatus.CANCELLED
    record.result = ''
    record.error = ''
    record.finishedAt = this.timestamp()
    record.completion.resolve()
  }

  private pruneCompleted(): void {
    const completed = [...this.sessions.values()]
      .filter(record => isTerminal(record.status) && !this.active.has(record.id))
      .sort((left, right) => left.finishedAt - right.finishedAt || left.sequence - right.sequence)
    const removeCount = Math.max(0, completed.length - this.maxCompleted)
    for (const record of completed.slice(0, removeCount)) this.sessions.delete(record.id)
  }

  private nextId(): string {
    for (let attempt = 0; attempt < MAX_ID_ATTEMPTS; attempt += 1) {
      const candidate = requiredText(this.idFactory(), 'background session id')
      if (!this.sessions.has(candidate)) return candidate
    }
    throw new Error('background session id factory produced too many duplicate identifiers')
  }

  private timestamp(): number {
    const value = this.now()
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError('background session clock must return a non-negative finite epoch timestamp')
    }
    return value
  }
}

function snapshot(record: MutableBackgroundSession): BackgroundSession {
  return Object.freeze({
    id: record.id,
    prompt: record.prompt,
    status: record.status,
    result: record.result,
    error: record.error,
    startedAt: record.startedAt,
    finishedAt: record.finishedAt,
    metadata: cloneMetadata(record.metadata),
  })
}

function deferred(): Deferred {
  let resolvePromise: (() => void) | undefined
  const promise = new Promise<void>(resolve => {
    resolvePromise = resolve
  })
  return {
    promise,
    resolve: () => resolvePromise?.(),
  }
}

function isCancellable(status: BackgroundStatus): boolean {
  return status === BackgroundStatus.PENDING || status === BackgroundStatus.RUNNING
}

function isTerminal(status: BackgroundStatus): boolean {
  return status === BackgroundStatus.SUCCEEDED
    || status === BackgroundStatus.FAILED
    || status === BackgroundStatus.CANCELLED
}

function defaultSessionId(): string {
  return crypto.randomUUID().replaceAll('-', '').slice(0, 12)
}

function boundedInteger(value: number, name: string, minimum: number): number {
  if (!Number.isFinite(value)) throw new RangeError(name + ' must be finite')
  return Math.max(minimum, Math.trunc(value))
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative integer')
  }
  return value
}

function requiredText(value: unknown, name: string): string {
  if (typeof value !== 'string' || !value) throw new TypeError(name + ' must be a non-empty string')
  return value
}

function textValue(value: unknown, name: string): string {
  if (typeof value !== 'string') throw new TypeError(name + ' must be a string')
  return value
}

function errorText(error: unknown): string {
  if (error instanceof Error) return error.name + ': ' + error.message
  return 'Error: ' + String(error)
}

async function waitWithTimeout(promise: Promise<void>, timeoutMs: number | undefined): Promise<void> {
  if (timeoutMs === undefined) {
    await promise
    return
  }
  const timeout = nonNegativeInteger(timeoutMs, 'timeoutMs')
  if (timeout === 0) return
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    await Promise.race([
      promise,
      new Promise<void>(resolve => {
        timer = setTimeout(resolve, timeout)
      }),
    ])
  } finally {
    if (timer !== undefined) clearTimeout(timer)
  }
}

function freezeMetadata(metadata: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> {
  return freezeRecord(metadata)
}

function cloneMetadata(metadata: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> {
  return freezeRecord(metadata)
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
