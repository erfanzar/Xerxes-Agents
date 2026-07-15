// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AsyncLocalStorage } from 'node:async_hooks'

/** Raised by cooperative checkpoints after a user-requested interruption. */
export class InterruptRequestedError extends Error {
  constructor() {
    super('Tool interrupted by user')
    this.name = 'InterruptRequestedError'
  }
}

/**
 * Reusable cooperative cancellation token with a Bun-native AbortSignal view.
 *
 * Call set() from any callback, poll isSet() at work boundaries, or await
 * wait(). Calling clear() starts a fresh AbortSignal generation after an
 * interruption, so consumers retaining the old signal still observe its abort.
 */
export class InterruptToken {
  private controller = new AbortController()
  private interrupted = false
  private readonly waiters = new Set<() => void>()

  get signal(): AbortSignal {
    return this.controller.signal
  }

  set(): void {
    if (this.interrupted) return
    this.interrupted = true
    this.controller.abort(new InterruptRequestedError())
    for (const resolve of [...this.waiters]) {
      resolve()
    }
    this.waiters.clear()
  }

  clear(): void {
    if (!this.interrupted) return
    this.interrupted = false
    this.controller = new AbortController()
  }

  isSet(): boolean {
    return this.interrupted
  }

  /** Wait for interruption, returning false when timeoutMs elapses first. */
  wait(timeoutMs?: number): Promise<boolean> {
    if (timeoutMs !== undefined && (!Number.isFinite(timeoutMs) || timeoutMs < 0)) {
      throw new RangeError('timeoutMs must be a non-negative finite number')
    }
    if (this.interrupted) return Promise.resolve(true)

    return new Promise(resolve => {
      let timer: ReturnType<typeof setTimeout> | undefined
      const wake = () => {
        this.waiters.delete(wake)
        if (timer !== undefined) clearTimeout(timer)
        resolve(true)
      }
      this.waiters.add(wake)
      if (timeoutMs !== undefined) {
        timer = setTimeout(() => {
          if (this.waiters.delete(wake)) resolve(false)
        }, timeoutMs)
      }
    })
  }

  throwIfSet(): void {
    if (!this.interrupted) return
    const reason = this.controller.signal.reason
    throw reason instanceof Error ? reason : new InterruptRequestedError()
  }
}

/**
 * Explicit asynchronous scope port for interrupt tokens.
 *
 * AsyncLocalStorage propagates a scope through promises and timers created in
 * its callback, while concurrent sibling scopes remain isolated. It does not
 * cross Bun Worker boundaries; pass the token explicitly to worker code.
 */
export class AsyncInterruptScope {
  private readonly storage = new AsyncLocalStorage<InterruptToken | undefined>()

  current(): InterruptToken | undefined {
    return this.storage.getStore()
  }

  run<T>(token: InterruptToken | undefined, callback: () => T): T {
    return this.storage.run(token, callback)
  }

  /**
   * Attach a token to the current asynchronous execution chain.
   *
   * Prefer run() or interruptScope() for a bounded lifetime. enter() affects
   * only work created from the current context and does not alter sibling
   * asynchronous chains or Worker contexts.
   */
  enter(token: InterruptToken | undefined): void {
    this.storage.enterWith(token)
  }
}

const defaultInterruptScope = new AsyncInterruptScope()

/** Return the token bound to the current async execution chain, if any. */
export function currentToken(): InterruptToken | undefined {
  return defaultInterruptScope.current()
}

/**
 * Bind a token to the current async execution chain.
 *
 * Prefer interruptScope() for normal turn execution so the previous token is
 * restored automatically after nested work completes.
 */
export function setCurrentToken(token: InterruptToken | undefined): void {
  defaultInterruptScope.enter(token)
}

/** Clear the token from the current async execution chain. */
export function clearCurrentToken(): void {
  setCurrentToken(undefined)
}

/**
 * Run a callback with a token propagated across its asynchronous descendants.
 *
 * The callback may return either a direct value or a Promise. Nested scopes
 * restore their parent scope, and independent concurrent calls do not share a
 * token.
 */
export function interruptScope<T>(
  callback: (token: InterruptToken) => T,
  token: InterruptToken = new InterruptToken(),
): T {
  return defaultInterruptScope.run(token, () => callback(token))
}
