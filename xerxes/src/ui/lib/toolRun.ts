// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// One row per tool call is right at three calls and wrong at thirty: a long
// turn becomes a wall of near-identical lines, and the answer you were
// actually waiting for gets pushed off the top of the screen by its own
// scaffolding.
//
// A consecutive run of *settled, successful* calls collapses to a single
// summary. Anything that still needs your attention — a failure, a call
// still in flight — is never folded away.

import { isToolTrailResultLine, parseToolTrailResultLine, toolTrailParts } from './text.js'

/** Below this a run is not worth summarizing; the rows are cheaper to read. */
export const TOOL_RUN_MIN = 4

export interface ToolRunSummary {
  /** Total seconds across the run, when the lines carried durations. */
  duration: number
  /** `bun test …` — the call worth looking at if you look at any of them. */
  slowest: string
  slowestDuration: number
  /** `read ×2 · edit ×1` in call order. */
  tally: string
  total: number
}

export type ToolRunGroup =
  | { kind: 'row'; line: string }
  | { kind: 'run'; lines: string[]; summary: ToolRunSummary }

const succeeded = (line: string): boolean => {
  if (!isToolTrailResultLine(line)) {
    return false
  }

  return parseToolTrailResultLine(line)?.mark === '✓'
}

/**
 * Seconds one settled tool line reports, or 0 when it carries no duration.
 *
 * Exported because the turn ledger totals the same numbers this module folds
 * a run with: two different answers to "how long did that turn take" on one
 * screen is worse than either answer alone.
 */
export const toolTrailSeconds = (line: string): number => {
  const parsed = parseToolTrailResultLine(line)

  if (!parsed) {
    return 0
  }

  const value = Number.parseFloat(toolTrailParts(parsed.call).duration)

  return Number.isFinite(value) ? value : 0
}

/** `Read File` → `read`. The verb is the part worth counting. */
const verb = (line: string): string => {
  const parsed = parseToolTrailResultLine(line)
  const name = parsed ? toolTrailParts(parsed.call).name : line

  return (name.split(/\s+/)[0] ?? name).toLowerCase()
}

const summarize = (lines: string[]): ToolRunSummary => {
  const counts = new Map<string, number>()
  let duration = 0
  let slowest = ''
  let slowestDuration = -1

  for (const line of lines) {
    const key = verb(line)
    counts.set(key, (counts.get(key) ?? 0) + 1)

    const took = toolTrailSeconds(line)
    duration += took

    if (took > slowestDuration) {
      slowestDuration = took
      const parsed = parseToolTrailResultLine(line)
      const parts = parsed ? toolTrailParts(parsed.call) : null
      slowest = parts ? [parts.name, parts.args].filter(Boolean).join(' ') : line
    }
  }

  return {
    duration,
    slowest,
    slowestDuration: Math.max(0, slowestDuration),
    tally: [...counts].map(([name, count]) => `${name} ×${count}`).join(' · '),
    total: lines.length
  }
}

/**
 * Split a trail into rows that stay visible and runs that fold.
 *
 * Order is preserved exactly — collapsing must never reorder a trail, because
 * the sequence is the only record of what the turn actually did.
 */
export function groupToolRun(lines: readonly string[], min = TOOL_RUN_MIN): ToolRunGroup[] {
  const groups: ToolRunGroup[] = []
  let run: string[] = []

  const flush = () => {
    if (run.length >= min) {
      groups.push({ kind: 'run', lines: run, summary: summarize(run) })
    } else {
      run.forEach(line => groups.push({ kind: 'row', line }))
    }

    run = []
  }

  for (const line of lines) {
    if (succeeded(line)) {
      run.push(line)
      continue
    }

    flush()
    groups.push({ kind: 'row', line })
  }

  flush()

  return groups
}

/** Rows a group occupies when collapsed — the height estimator needs this. */
export const collapsedRunHeight = (group: ToolRunGroup): number =>
  group.kind === 'row' ? 1 : group.summary.slowestDuration > 0 ? 2 : 1
