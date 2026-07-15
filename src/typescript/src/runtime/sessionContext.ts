// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AsyncLocalStorage } from 'node:async_hooks'

/**
 * Async-local active-session carrier for tools that need the live turn owner.
 *
 * It replaces Python's worker-thread-local pointer. A scope propagates through
 * promises and timers created during one turn, while concurrent turns retain
 * their own session objects. Workers must still receive session state
 * explicitly because AsyncLocalStorage does not cross Worker boundaries.
 */
export class AsyncSessionContext<Session> {
  private readonly storage = new AsyncLocalStorage<Session | undefined>()

  current(): Session | undefined {
    return this.storage.getStore()
  }

  enter(session: Session | undefined): void {
    this.storage.enterWith(session)
  }

  run<Value>(session: Session | undefined, operation: () => Value): Value {
    return this.storage.run(session, operation)
  }

  /**
   * Iterate an async source with its `next()` calls inside the supplied scope.
   *
   * Async generators begin executing on `next()`, not creation. Re-entering
   * before every next call keeps the active session available across yields.
   */
  async *iterate<Value>(session: Session | undefined, source: AsyncIterable<Value>): AsyncGenerator<Value> {
    const iterator = source[Symbol.asyncIterator]()
    try {
      while (true) {
        const next = await this.storage.run(session, () => iterator.next())
        if (next.done) return
        yield next.value
      }
    } finally {
      await this.storage.run(session, () => iterator.return?.())
    }
  }
}

const defaultSessionContext = new AsyncSessionContext<unknown>()

/** Return the live session bound to this asynchronous turn, if any. */
export function getActiveSession<Session = unknown>(): Session | undefined {
  return defaultSessionContext.current() as Session | undefined
}

/** Bind a session to descendants of the current asynchronous execution chain. */
export function setActiveSession<Session>(session: Session | undefined): void {
  defaultSessionContext.enter(session)
}

/** Clear the session binding for descendants of the current asynchronous chain. */
export function clearActiveSession(): void {
  defaultSessionContext.enter(undefined)
}

/** Run one synchronous or async operation with an isolated active session. */
export function runWithActiveSession<Session, Value>(session: Session | undefined, operation: () => Value): Value {
  return defaultSessionContext.run(session, operation)
}

/** Stream an async source while keeping its session scope available to deep tools. */
export function withActiveSession<Session, Value>(
  session: Session | undefined,
  source: AsyncIterable<Value>,
): AsyncGenerator<Value> {
  return defaultSessionContext.iterate(session, source)
}
