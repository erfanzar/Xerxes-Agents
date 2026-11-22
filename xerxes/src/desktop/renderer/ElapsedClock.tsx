// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { memo, useEffect, useState, type ReactElement } from 'react'
import { pageIsInactive, subscribePerfPaused } from './pageVisibility.js'

export interface ElapsedClockProps {
  /** Whether the turn/clock is actively advancing. */
  readonly active: boolean
  /** Epoch ms when the active run started; omit for static snapshot renders. */
  readonly since?: number | null | undefined
  /** Fallback/base elapsed seconds (used when `since` is omitted or settled). */
  readonly baseSeconds: number
  /** Formatter converting seconds to display text. */
  readonly format: (seconds: number) => string
  readonly className?: string | undefined
}

function computeElapsedSeconds(active: boolean, since: number | null | undefined, baseSeconds: number, now: number): number {
  if (!active || since == null) return baseSeconds
  return Math.max(baseSeconds, Math.floor((now - since) / 1000))
}

/**
 * Isolated leaf elapsed-seconds indicator.
 *
 * Replaces the old global `store.ts` 1-second `setInterval` that forced a full
 * `<App>` React tree re-render every second for the entire duration of a turn.
 * Only ticks while `active && since != null` and pauses automatically while
 * the page is hidden, occluded, or unfocused.
 */
export const ElapsedClock = memo(function ElapsedClock({
  active,
  since,
  baseSeconds,
  format,
  className,
}: ElapsedClockProps): ReactElement {
  const [now, setNow] = useState<number>(() => Date.now())

  useEffect(() => {
    if (!active || since == null) return
    setNow(Date.now())
    const timer = setInterval(() => {
      if (!pageIsInactive()) setNow(Date.now())
    }, 1_000)
    const unsub = subscribePerfPaused(() => {
      if (!pageIsInactive()) setNow(Date.now())
    })
    return () => {
      clearInterval(timer)
      unsub()
    }
  }, [active, since])

  const seconds = computeElapsedSeconds(active, since, baseSeconds, now)
  return <span className={className}>{format(seconds)}</span>
})
