// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** Compact context-window utilization meters for status surfaces. */

export const FILLED_BLOCK = '█'
export const HALF_BLOCK = '▌'
export const EMPTY_BLOCK = '░'

export interface ContextBarOptions {
  empty?: string
  filled?: string
  half?: string
  used: number
  width?: number
  window: number
}

export interface ContextBarWithPctOptions extends ContextBarOptions {
  showPct?: boolean
}

const integer = (value: number, label: string): number => {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite`)
  }

  return Math.trunc(value)
}

/**
 * Renders a fixed-cell meter for used context tokens.
 *
 * A half cell makes low utilization visible while clamping any overfull
 * context to the provided width.
 */
export const contextBar = ({
  empty = EMPTY_BLOCK,
  filled = FILLED_BLOCK,
  half = HALF_BLOCK,
  used,
  width = 24,
  window
}: ContextBarOptions): string => {
  const safeWidth = integer(width, 'width')

  if (safeWidth <= 0) {
    return ''
  }

  const safeWindow = Math.max(1, integer(window, 'window'))
  const safeUsed = Math.max(0, integer(used, 'used'))
  const cells = safeUsed < safeWindow ? (safeUsed / safeWindow) * safeWidth : safeWidth
  let full = Math.trunc(cells)
  let hasHalf = cells - full >= 0.5

  if (full > safeWidth) {
    full = safeWidth
    hasHalf = false
  }

  const extra = hasHalf ? 1 : 0

  return filled.repeat(full) + (hasHalf ? half : '') + empty.repeat(safeWidth - full - extra)
}

/** Renders a context meter with the raw usage percentage beside it. */
export const contextBarWithPct = ({ showPct = true, used, window, ...options }: ContextBarWithPctOptions): string => {
  const bar = contextBar({ used, window, ...options })

  if (!showPct) {
    return bar
  }

  const pct = window > 0 ? (used / window) * 100 : 0

  return `${bar} ${pct.toFixed(1).padStart(5)}%`
}
