// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { DEFAULT_TURN_IDLE_MS, IdleTimeoutError, MIN_TURN_IDLE_MS, idleMsFromEnvironment, normalizeIdleMs, startIdleWatchdog, type IdleTimers } from '../src/hub/supervisor/idleWatchdog.js'

/** A manual clock: timers fire only when `advance` passes their due time. */
function fakeTimers() {
  let now = 0
  let next = 1
  const pending = new Map<number, { due: number; run: () => void }>()
  const timers: IdleTimers = {
    set: (run, ms) => { const id = next++; pending.set(id, { due: now + ms, run }); return id },
    clear: handle => { pending.delete(handle as number) },
  }
  const advance = (ms: number): void => {
    now += ms
    for (const [id, entry] of [...pending]) if (entry.due <= now) { pending.delete(id); entry.run() }
  }
  return { timers, advance, pending }
}

describe('XERXES turn idle watchdog', () => {
  test('a turn with steady activity runs far past the idle window', () => {
    const clock = fakeTimers()
    const controller = new AbortController()
    const watchdog = startIdleWatchdog(controller, 1000, clock.timers)
    // 100 activity bursts 900 ms apart: 90 s of work against a 1 s idle window.
    for (let i = 0; i < 100; i += 1) { clock.advance(900); watchdog.touch() }
    expect(controller.signal.aborted).toBe(false)
    watchdog.stop()
  })

  test('real silence ends the turn with an IdleTimeoutError', () => {
    const clock = fakeTimers()
    const controller = new AbortController()
    startIdleWatchdog(controller, 1000, clock.timers)
    clock.advance(999)
    expect(controller.signal.aborted).toBe(false)
    clock.advance(1)
    expect(controller.signal.reason).toBeInstanceOf(IdleTimeoutError)
    expect(String((controller.signal.reason as Error).message)).toContain('no model or tool activity for 1 seconds')
  })

  test('stop() leaves no timer behind, and an owner stop is never overwritten', () => {
    const clock = fakeTimers()
    const finished = new AbortController()
    startIdleWatchdog(finished, 1000, clock.timers).stop()
    expect(clock.pending.size).toBe(0)
    clock.advance(5000)
    expect(finished.signal.aborted).toBe(false)

    const stopped = new AbortController()
    startIdleWatchdog(stopped, 1000, clock.timers)
    stopped.abort(new Error('owner stopped'))
    clock.advance(5000)
    expect((stopped.signal.reason as Error).message).toBe('owner stopped')
  })

  test('an unbounded window sets no timer at all', () => {
    const clock = fakeTimers()
    const controller = new AbortController()
    const watchdog = startIdleWatchdog(controller, Number.POSITIVE_INFINITY, clock.timers)
    watchdog.touch()
    expect(clock.pending.size).toBe(0)
  })

  test('configuration: default, off, floor, malformed', () => {
    expect(normalizeIdleMs(undefined)).toBe(DEFAULT_TURN_IDLE_MS)
    expect(normalizeIdleMs(0)).toBe(Number.POSITIVE_INFINITY)
    expect(normalizeIdleMs(Number.POSITIVE_INFINITY)).toBe(Number.POSITIVE_INFINITY)
    expect(normalizeIdleMs(30)).toBe(30)
    expect(idleMsFromEnvironment(undefined)).toBeUndefined()
    expect(idleMsFromEnvironment('soon')).toBeUndefined()
    expect(idleMsFromEnvironment('off')).toBe(Number.POSITIVE_INFINITY)
    expect(idleMsFromEnvironment('0')).toBe(Number.POSITIVE_INFINITY)
    expect(idleMsFromEnvironment('5')).toBe(MIN_TURN_IDLE_MS)
    expect(idleMsFromEnvironment('3600000')).toBe(3_600_000)
  })
})
