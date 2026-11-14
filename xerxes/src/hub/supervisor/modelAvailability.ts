// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * What XERXES does when no machine offers its model: wait, never drop.
 *
 * The model port throws `ModelUnavailableError` when the configured machine is
 * offline and no other online machine offers the same profile/model. The
 * runner then keeps the turn (so the person's message or wake stays queued in
 * order behind it), says so once in the thread, and waits here until either a
 * machine comes back (`notify()`, called on a runtime-pool health change) or
 * the retry interval passes, whichever is first. Only an abort (the owner's
 * stop, pause/kill, or shutdown) ends the wait.
 */

export class ModelUnavailableError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ModelUnavailableError'
  }
}

/** Re-check this often even when no health change arrives. */
export const DEFAULT_MODEL_RETRY_MS = 30_000

export interface ModelAvailabilityTimers {
  set(run: () => void, ms: number): unknown
  clear(handle: unknown): void
}

const REAL_TIMERS: ModelAvailabilityTimers = {
  set: (run, ms) => setTimeout(run, ms),
  clear: handle => clearTimeout(handle as ReturnType<typeof setTimeout>),
}

export class ModelAvailability {
  private readonly waiters = new Set<() => void>()
  private readonly retryMs: number
  private readonly timers: ModelAvailabilityTimers

  constructor(options: { readonly retryMs?: number; readonly timers?: ModelAvailabilityTimers } = {}) {
    this.retryMs = Math.max(1, options.retryMs ?? DEFAULT_MODEL_RETRY_MS)
    this.timers = options.timers ?? REAL_TIMERS
  }

  /** How many turns are waiting for a model right now. */
  get waiting(): number { return this.waiters.size }

  /** A machine may have come back: wake every waiting turn so it retries now. */
  notify(): void {
    for (const wake of [...this.waiters]) wake()
  }

  /** Resolves on `notify()` or after the retry interval; rejects with the abort reason. */
  wait(signal?: AbortSignal): Promise<void> {
    if (signal?.aborted) return Promise.reject(signal.reason)
    return new Promise<void>((resolve, reject) => {
      let handle: unknown
      const finish = (): void => {
        this.waiters.delete(wake)
        this.timers.clear(handle)
        signal?.removeEventListener('abort', onAbort)
      }
      const wake = (): void => { finish(); resolve() }
      const onAbort = (): void => { finish(); reject(signal!.reason) }
      this.waiters.add(wake)
      handle = this.timers.set(wake, this.retryMs)
      signal?.addEventListener('abort', onAbort, { once: true })
    })
  }
}
