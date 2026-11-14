// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Ends an XERXES turn only after real silence, never on wall-clock length.
 *
 * Every sign of progress (a model call starting or answering, a tool starting
 * or returning) calls `touch()`, which restarts the silence window. When no
 * activity arrives for `idleMs`, the watchdog aborts the turn's controller
 * with an `IdleTimeoutError`. `Infinity` (or a non-positive value) disables
 * it: then only the owner's stop ends the turn.
 */

/** A generous default: a slow reasoning call can sit silent for many minutes. */
export const DEFAULT_TURN_IDLE_MS = 30 * 60_000
/** A configured window below this is raised to it, so a typo cannot kill every turn. */
export const MIN_TURN_IDLE_MS = 60_000

export class IdleTimeoutError extends Error {
  constructor(readonly idleMs: number) {
    super(`no model or tool activity for ${Math.round(idleMs / 1000)} seconds`)
    this.name = 'IdleTimeoutError'
  }
}

/** The runner's idle window: unset is the default; non-finite or non-positive means no idle limit. */
export function normalizeIdleMs(value: number | undefined): number {
  if (value === undefined) return DEFAULT_TURN_IDLE_MS
  if (!Number.isFinite(value) || value <= 0) return Number.POSITIVE_INFINITY
  return value
}

/**
 * `XERXES_HUB_SUPERVISOR_IDLE_MS`: whole milliseconds (raised to at least one
 * minute), or `0`/`off` for no idle limit. Unset or malformed: undefined, so the
 * default applies rather than a guessed value.
 */
export function idleMsFromEnvironment(raw: string | undefined): number | undefined {
  const value = raw?.trim().toLowerCase()
  if (!value) return undefined
  if (value === 'off' || value === '0') return Number.POSITIVE_INFINITY
  if (!/^\d+$/.test(value)) return undefined
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) ? Math.max(MIN_TURN_IDLE_MS, parsed) : undefined
}

export interface IdleWatchdog {
  /** Progress happened: restart the silence window. */
  touch(): void
  /** Stop watching (the turn finished). */
  stop(): void
}

export interface IdleTimers {
  set(run: () => void, ms: number): unknown
  clear(handle: unknown): void
}

const REAL_TIMERS: IdleTimers = {
  set: (run, ms) => setTimeout(run, ms),
  clear: handle => clearTimeout(handle as ReturnType<typeof setTimeout>),
}

export function startIdleWatchdog(controller: AbortController, idleMs: number, timers: IdleTimers = REAL_TIMERS): IdleWatchdog {
  // setTimeout(Infinity) fires at once, so an unbounded window gets no timer at all.
  if (!Number.isFinite(idleMs)) return { touch() {}, stop() {} }
  let handle: unknown
  let stopped = false
  const arm = (): void => {
    if (handle !== undefined) timers.clear(handle)
    handle = timers.set(() => {
      handle = undefined
      if (!stopped && !controller.signal.aborted) controller.abort(new IdleTimeoutError(idleMs))
    }, idleMs)
  }
  arm()
  return {
    touch() { if (!stopped && !controller.signal.aborted) arm() },
    stop() {
      stopped = true
      if (handle !== undefined) timers.clear(handle)
      handle = undefined
    },
  }
}
