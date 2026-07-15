// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Pure /details visibility logic. Controls how much tool/thinking machinery the
// transcript shows: expanded (everything) → collapsed (tool one-liners, no
// verbose blocks, no live thinking) → hidden (no tool rows at all).

import type { TranscriptRow } from '../app/gatewayState.js'

export type DetailsMode = 'expanded' | 'collapsed' | 'hidden'

export const DETAILS_ORDER: readonly DetailsMode[] = ['expanded', 'collapsed', 'hidden']

const isMode = (s: string): s is DetailsMode => (DETAILS_ORDER as readonly string[]).includes(s)

/** Next mode in the cycle. */
export function cycleDetails(mode: DetailsMode): DetailsMode {
  const i = DETAILS_ORDER.indexOf(mode)
  return DETAILS_ORDER[(i + 1) % DETAILS_ORDER.length]!
}

/** Resolve a /details argument ('cycle' | a mode | empty) against the current. */
export function resolveDetails(arg: string, current: DetailsMode): DetailsMode {
  const a = arg.trim().toLowerCase()
  if (!a || a === 'cycle') {
    return cycleDetails(current)
  }
  return isMode(a) ? a : current
}

/** The live reasoning row is only shown when fully expanded. */
export function showThinking(mode: DetailsMode): boolean {
  return mode === 'expanded'
}

/** Filter/transform transcript rows for the current details mode. */
export function filterTranscript(rows: readonly TranscriptRow[], mode: DetailsMode): TranscriptRow[] {
  if (mode === 'expanded') {
    return rows as TranscriptRow[]
  }
  const out: TranscriptRow[] = []
  for (const row of rows) {
    if (row.role !== 'tool') {
      out.push(row)
      continue
    }
    if (mode === 'hidden') {
      continue
    }
    // collapsed: keep a one-liner, drop the verbose result blocks.
    out.push({ id: row.id, role: row.role, text: row.text || 'result ⋯' })
  }
  return out
}
