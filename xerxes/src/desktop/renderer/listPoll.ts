// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A polled list that never throws away its own answer. One request per key
 * runs at a time; a poll tick while it is still running is skipped instead of
 * cancelling it. (The Artifacts panel used to restart its effect every 5 s and
 * discard any list call that took longer, so during a busy turn the newest
 * artifacts never appeared.) A reply only lands if its key is still current.
 */
export interface ListPoll {
  /** Start a request for `key` unless one is already running; false when skipped. */
  request(key: string): boolean
}

export function createListPoll<T>(
  fetch: (key: string) => Promise<T>,
  apply: (value: T) => void,
  fail: (error: unknown) => void,
): ListPoll {
  let inflight: string | null = null
  let current = ''
  return {
    request(key) {
      current = key
      if (inflight === key) return false
      inflight = key
      void fetch(key)
        .then(value => { if (current === key) apply(value) }, error => { if (current === key) fail(error) })
        .finally(() => { if (inflight === key) inflight = null })
      return true
    },
  }
}
