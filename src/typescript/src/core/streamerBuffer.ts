// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Legacy sentinel accepted by put() as a request to close the stream. */
export const KILL_TAG = '/<[KILL-LOOP]>/'

export interface StreamerBufferOptions<T> {
  /** Maximum queued values; zero means unbounded. */
  readonly maxSize?: number
  /** Identifies a final completion event so maybeFinish() can close the buffer. */
  readonly isCompletion?: (item: T) => boolean
}

export interface StreamerBufferGetOptions {
  readonly signal?: AbortSignal
  /** Returns undefined after this duration when no value is available. */
  readonly timeoutMs?: number
}

/** Raised when StreamerBuffer is configured with an invalid capacity or timeout. */
export class StreamerBufferError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'StreamerBufferError'
  }
}

interface PendingReader<T> {
  reject(reason: unknown): void
  resolve(value: IteratorResult<T | null>): void
}

/**
 * Async producer/consumer bridge for Bun-native streaming.
 *
 * Producers can use non-blocking put() or backpressured push(); consumers use
 * stream() or the AsyncIterable protocol. Closing is idempotent, drains
 * already-buffered values, and resolves every waiting consumer without a
 * process-global queue, debug flag, or thread hand-off.
 */
export class StreamerBuffer<T> implements AsyncIterable<T> {
  private readonly completionPredicate: (item: T) => boolean
  private readonly maxSize: number
  private readonly queue: Array<T | null> = []
  private readonly readers = new Set<PendingReader<T>>()
  private readonly spaceWaiters = new Set<() => void>()
  private completionSeen = false
  private isClosed = false

  constructor(options: StreamerBufferOptions<T> = {}) {
    this.maxSize = requireMaxSize(options.maxSize ?? 0)
    this.completionPredicate = options.isCompletion ?? (() => false)
  }

  get closed(): boolean {
    return this.isClosed
  }

  get finishHit(): boolean {
    return this.completionSeen
  }

  /**
   * Queue one value without waiting for capacity.
   *
   * Returns false when closed or when a bounded queue is full. Passing KILL_TAG
   * requests a graceful close and leaves already-buffered values drainable.
   */
  put(item: T | null): boolean {
    if (this.isKillTag(item)) {
      if (this.isClosed) return false
      this.close()
      return true
    }
    if (this.isClosed) return false
    const reader = this.firstReader()
    if (reader) {
      reader.resolve({ done: false, value: item })
      return true
    }
    if (this.maxSize > 0 && this.queue.length >= this.maxSize) return false
    this.queue.push(item)
    return true
  }

  /** Wait for bounded-queue capacity rather than dropping a producer value. */
  async push(item: T | null): Promise<boolean> {
    if (this.isKillTag(item)) return this.put(item)
    while (!this.isClosed) {
      if (this.put(item)) return true
      await this.waitForSpace()
    }
    return false
  }

  /**
   * Receive one queued value, heartbeat, or undefined after close/timeout.
   *
   * Heartbeats are represented by null for parity with the original buffer;
   * stream() ignores them before yielding to its caller.
   */
  async get(options: StreamerBufferGetOptions = {}): Promise<T | null | undefined> {
    const item = await this.take(options)
    return item === undefined || item.done ? undefined : item.value
  }

  /** Idempotently close the buffer and release all waiting consumers/producers. */
  close(): boolean {
    if (this.isClosed) return false
    this.isClosed = true
    const done: IteratorResult<T | null> = { done: true, value: undefined }
    const readers = [...this.readers]
    this.readers.clear()
    for (const reader of readers) reader.resolve(done)
    this.resolveSpaceWaiters()
    return true
  }

  /** Close after a completion event has been observed and the supplied value is nullish. */
  maybeFinish(value: unknown): void {
    if (value === null || value === undefined) {
      if (this.completionSeen) this.close()
    }
  }

  /** Yield non-null streaming values until the buffer closes and drains. */
  async *stream(): AsyncGenerator<T> {
    while (true) {
      const next = await this.take()
      if (next === undefined || next.done) return
      if (next.value === null) continue
      if (this.completionPredicate(next.value)) this.completionSeen = true
      yield next.value
    }
  }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return this.stream()
  }

  private firstReader(): PendingReader<T> | undefined {
    const reader = this.readers.values().next().value
    if (reader) this.readers.delete(reader)
    return reader
  }

  private isKillTag(item: T | null): boolean {
    return typeof item === 'string' && item === KILL_TAG
  }

  private resolveSpaceWaiters(): void {
    for (const resolve of this.spaceWaiters) resolve()
    this.spaceWaiters.clear()
  }

  private take(options: StreamerBufferGetOptions = {}): Promise<IteratorResult<T | null> | undefined> {
    const timeoutMs = requireTimeout(options.timeoutMs)
    if (this.queue.length > 0) {
      const value = this.queue.shift() as T | null
      this.resolveOneSpaceWaiter()
      return Promise.resolve({ done: false, value })
    }
    if (this.isClosed) return Promise.resolve({ done: true, value: undefined })
    return this.waitForItem(timeoutMs, options.signal)
  }

  private waitForItem(
    timeoutMs: number | undefined,
    signal: AbortSignal | undefined,
  ): Promise<IteratorResult<T | null> | undefined> {
    return new Promise((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | undefined
      let settled = false
      const finish = (callback: () => void): void => {
        if (settled) return
        settled = true
        this.readers.delete(reader)
        if (timer !== undefined) clearTimeout(timer)
        signal?.removeEventListener('abort', abort)
        callback()
      }
      const reader: PendingReader<T> = {
        reject: reason => finish(() => reject(reason)),
        resolve: value => finish(() => resolve(value)),
      }
      const abort = (): void => reader.reject(abortReason(signal))

      if (signal?.aborted) {
        abort()
        return
      }
      this.readers.add(reader)
      if (signal) signal.addEventListener('abort', abort, { once: true })
      if (timeoutMs !== undefined) {
        timer = setTimeout(() => finish(() => resolve(undefined)), timeoutMs)
      }
    })
  }

  private waitForSpace(): Promise<void> {
    if (this.isClosed || this.maxSize === 0 || this.queue.length < this.maxSize) return Promise.resolve()
    return new Promise(resolve => this.spaceWaiters.add(resolve))
  }

  private resolveOneSpaceWaiter(): void {
    const waiter = this.spaceWaiters.values().next().value
    if (!waiter) return
    this.spaceWaiters.delete(waiter)
    waiter()
  }
}

function abortReason(signal: AbortSignal | undefined): Error {
  if (signal?.reason instanceof Error) return signal.reason
  return new DOMException('StreamerBuffer read was aborted', 'AbortError')
}

function requireMaxSize(value: number): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new StreamerBufferError('maxSize must be a non-negative integer')
  }
  return value
}

function requireTimeout(value: number | undefined): number | undefined {
  if (value === undefined) return undefined
  if (!Number.isFinite(value) || value < 0) {
    throw new StreamerBufferError('timeoutMs must be a non-negative finite number')
  }
  return value
}
