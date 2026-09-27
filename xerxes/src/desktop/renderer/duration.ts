// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * One elapsed-time format for every clock in the app: 43 → "43s",
 * 255 → "4m 15s", 7_500 → "2h 5m", 90_000 → "1d 1h". Each step keeps only
 * the next smaller unit. `compact` keeps just the largest ("4m", "2h",
 * "1d") for tight spots such as the session list.
 */
export function elapsedOf(totalSeconds: number, compact = false): string {
  const seconds = Math.max(0, Math.floor(Number.isFinite(totalSeconds) ? totalSeconds : 0))
  const units: readonly [string, number][] = [['d', 86_400], ['h', 3_600], ['m', 60], ['s', 1]]
  for (let index = 0; index < units.length; index += 1) {
    const [label, size] = units[index]!
    if (seconds < size && label !== 's') continue
    const whole = Math.floor(seconds / size)
    const next = units[index + 1]
    const rest = next ? Math.floor((seconds % size) / next[1]) : 0
    return compact || !next || rest === 0 ? `${whole}${label}` : `${whole}${label} ${rest}${next[0]}`
  }
  return '0s'
}

/** A duration in milliseconds, in the same format. */
export const durationOf = (milliseconds: number): string => elapsedOf(Math.round(milliseconds / 1_000))
