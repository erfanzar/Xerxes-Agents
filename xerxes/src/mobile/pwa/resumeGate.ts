// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** A foreground return resyncs at most once per this long. */
export const RESUME_MIN_INTERVAL_MS = 4_000

export interface ResumeGate {
  /** A foreground signal (visibilitychange, pageshow): runs unless a resume ran in the last interval. */
  trigger(): boolean
  /** The network came back: always runs (a dead link must reconnect now, not after the interval). */
  force(): void
}

/**
 * One iOS foreground fires visibilitychange and pageshow back to back; each
 * used to start a full resync (probe, session refresh, update check). The
 * gate lets the first through and drops the rest within the interval.
 */
export function createResumeGate(now: () => number, run: () => void, intervalMs = RESUME_MIN_INTERVAL_MS): ResumeGate {
  let last = Number.NEGATIVE_INFINITY
  const fire = (): void => {
    last = now()
    run()
  }
  return {
    trigger() {
      if (now() - last < intervalMs) return false
      fire()
      return true
    },
    force: fire,
  }
}
