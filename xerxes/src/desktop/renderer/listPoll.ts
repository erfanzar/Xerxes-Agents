// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { pageIsInactive } from './pageVisibility.js'

/**
 * A polled list that never throws away its own answer. One request per key
 * runs at a time; a poll tick while it is still running is skipped instead of
 * cancelling it. (The Artifacts panel used to restart its effect every 5 s and
 * discard any list call that took longer, so during a busy turn the newest
 * artifacts never appeared.) A reply only lands if its key is still current.
 *
 * When `pauseWhenInactive` is enabled, periodic polls pause automatically while
 * the page is hidden, occluded, or unfocused (unless `force` is passed) and
 * can catch up immediately when focus/visibility returns via `resume()`.
 */
export interface ListPoll {
  /** Start a request for `key` unless one is already running or paused; false when skipped. */
  request(key: string, force?: boolean): boolean
  /** Trigger the deferred key (if any) once the page becomes active again. */
  resume(): boolean
}

export interface ListPollOptions {
  /** Skip non-forced polls while `pageIsInactive()` is true. */
  readonly pauseWhenInactive?: boolean
  /** Custom pause predicate (defaults to `pageIsInactive` when `pauseWhenInactive` is true). */
  readonly isPaused?: () => boolean
}

export function createListPoll<T>(
  fetch: (key: string) => Promise<T>,
  apply: (value: T) => void,
  fail: (error: unknown) => void,
  options?: ListPollOptions,
): ListPoll {
  let inflight: string | null = null
  let current = ''
  let deferredKey: string | null = null

  const paused = (): boolean => {
    if (options?.isPaused) return options.isPaused()
    if (options?.pauseWhenInactive) return pageIsInactive()
    return false
  }

  const start = (key: string, force = false): boolean => {
    current = key
    if (!force && paused()) {
      deferredKey = key
      return false
    }
    if (inflight === key) return false
    deferredKey = null
    inflight = key
    void fetch(key)
      .then(value => { if (current === key) apply(value) }, error => { if (current === key) fail(error) })
      .finally(() => { if (inflight === key) inflight = null })
    return true
  }

  return {
    request(key, force = false) {
      return start(key, force)
    },
    resume() {
      if (deferredKey === null || paused()) return false
      return start(deferredKey, false)
    },
  }
}
