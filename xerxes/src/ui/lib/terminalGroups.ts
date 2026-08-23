// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// The terminals panel listed newest-first regardless of whether a command
// still needed you. With forty tracked shells that buries the one that is
// still running and the three that failed under thirty-six that succeeded
// and will never matter again.
//
// Order by what you can still act on: running, then failed, then the rest.

import type { NocturneState } from '../domain/nocturne.js'

import type { TerminalSummary } from './terminals.js'

export type TerminalGroup = 'failed' | 'interactive' | 'running' | 'succeeded'

export const TERMINAL_GROUP_LABEL: Record<TerminalGroup, string> = {
  failed: 'FAILED',
  interactive: 'INTERACTIVE',
  running: 'RUNNING',
  succeeded: 'SUCCEEDED'
}

const RANK: Record<TerminalGroup, number> = { running: 0, failed: 1, interactive: 2, succeeded: 3 }

/**
 * The Nocturne state each group wears.
 *
 * `interactive` is amber, and it means what amber always means: this one is
 * waiting on a human. A live shell that accepts input is, by construction,
 * waiting for someone to type into it — and a root shell in /etc that nobody
 * remembers opening is exactly the case the amber exists to catch.
 */
export const TERMINAL_GROUP_STATE: Record<TerminalGroup, NocturneState> = {
  failed: 'failed',
  interactive: 'needsInput',
  running: 'working',
  succeeded: 'done'
}

export const terminalGroup = (entry: TerminalSummary): TerminalGroup => {
  if (entry.running) {
    // A PTY the user can type into is a shell someone has to come back to;
    // a background command is one the agent is waiting on. Both are alive,
    // but only one of them is waiting on YOU.
    return entry.kind === 'pty' && entry.canWrite ? 'interactive' : 'running'
  }

  // A null exit code on a settled shell means it was killed or never
  // reported — closer to a failure than a success, and worth surfacing.
  return entry.exitCode === 0 ? 'succeeded' : 'failed'
}

/** Most recent activity first inside a group. */
const recency = (entry: TerminalSummary): number => entry.endedAt ?? entry.startedAt ?? 0

/**
 * Stable ordering: group rank first, recency within a group. Sorting the
 * array the panel already holds means selection, arrow keys and the detail
 * view all follow automatically — no parallel display list to drift.
 */
export function orderTerminals(entries: readonly TerminalSummary[]): TerminalSummary[] {
  return [...entries].sort((a, b) => {
    const byGroup = RANK[terminalGroup(a)] - RANK[terminalGroup(b)]

    return byGroup !== 0 ? byGroup : recency(b) - recency(a)
  })
}

/** The label to print above `index`, or '' when the group has not changed. */
export function terminalHeading(entries: readonly TerminalSummary[], index: number): string {
  const entry = entries[index]

  if (!entry) {
    return ''
  }

  const group = terminalGroup(entry)
  const previous = entries[index - 1]

  if (previous && terminalGroup(previous) === group) {
    return ''
  }

  const size = entries.filter(candidate => terminalGroup(candidate) === group).length

  return `${TERMINAL_GROUP_LABEL[group]} · ${size}`
}
