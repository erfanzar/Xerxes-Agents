// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
//
// The agents panel groups by what the user can DO, not by wire status: the
// design's action order is unblock → monitor → review. A failed agent and a
// finished one must never read as peers, and a running agent must not outrank
// something that is literally waiting on you.
//
// (v2 named these running/failed/done; the redesign renames them to the
// action vocabulary while keeping the same sort contract.)

import type { NocturneState } from '../domain/nocturne.js'
import type { SubagentStatus } from '../types.js'

export type AgentGroup = 'failed' | 'input' | 'review' | 'working'

export const AGENT_GROUP_LABEL: Record<AgentGroup, string> = {
  failed: 'FAILED',
  input: 'NEEDS INPUT',
  review: 'READY TO REVIEW',
  working: 'WORKING'
}

/**
 * Four groups, ranked by what you have to do about them.
 *
 * `failed` was folded into `input` before the design system landed, which put
 * a dead agent above three live ones and made the amber caption lie: a run
 * that blew its token budget is not waiting on your answer, it is over.
 * It sorts last and renders as one dim line — it has already spent its money,
 * it does not get to spend your attention too.
 */
const RANK: Record<AgentGroup, number> = { input: 0, working: 1, review: 2, failed: 3 }

/** The Nocturne state each group wears; the colour follows from there. */
export const AGENT_GROUP_STATE: Record<AgentGroup, NocturneState> = {
  failed: 'failed',
  input: 'needsInput',
  review: 'done',
  working: 'working'
}

const FAILED_STATUSES = new Set<SubagentStatus>(['error', 'failed', 'timeout'])

export const agentGroup = (status: SubagentStatus): AgentGroup => {
  if (status === 'running' || status === 'queued') {
    return 'working'
  }

  if (status === 'completed') {
    return 'review'
  }

  // `interrupted` stays in `input`: you stopped it, so what happens next is
  // your decision, which is the definition of this group.
  return FAILED_STATUSES.has(status) ? 'failed' : 'input'
}

interface Groupable {
  item: { index?: number; startedAt?: number; status: SubagentStatus }
}

/**
 * Group rank first, then most recently started. Sorting the array the panel
 * already holds keeps selection, arrow keys and the inspector following the
 * same order the eye does — a separate display list would drift from what
 * the keyboard is moving through.
 */
export function orderAgentRecords<T extends Groupable>(records: readonly T[]): T[] {
  return [...records].sort((a, b) => {
    const byGroup = RANK[agentGroup(a.item.status)] - RANK[agentGroup(b.item.status)]

    if (byGroup !== 0) {
      return byGroup
    }

    const byRecency = (b.item.startedAt ?? 0) - (a.item.startedAt ?? 0)

    // Spawn index is the stable tiebreak: agents dispatched in one batch
    // share a timestamp, and a wobbling order between renders is worse than
    // an arbitrary but fixed one.
    return byRecency !== 0 ? byRecency : (a.item.index ?? 0) - (b.item.index ?? 0)
  })
}

/** The caption to print above `index`, or '' when the group has not changed. */
export function agentHeading<T extends Groupable>(records: readonly T[], index: number): string {
  const record = records[index]

  if (!record) {
    return ''
  }

  const group = agentGroup(record.item.status)

  if (records[index - 1] && agentGroup(records[index - 1]!.item.status) === group) {
    return ''
  }

  const size = records.filter(candidate => agentGroup(candidate.item.status) === group).length

  return `${AGENT_GROUP_LABEL[group]} · ${size}`
}
