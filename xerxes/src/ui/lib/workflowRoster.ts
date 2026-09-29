// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SubagentProgress } from '../types.js'

export type WorkflowAgentState = 'working' | 'done' | 'failed'

export interface WorkflowPhaseSummary {
  readonly name: string
  readonly total: number
  readonly working: number
  readonly done: number
  readonly failed: number
}

export interface WorkflowRosterSummary {
  readonly total: number
  readonly working: number
  readonly done: number
  readonly failed: number
  readonly phases: readonly WorkflowPhaseSummary[]
  /** Agents still running, then failed ones — the rows worth a line each. */
  readonly running: readonly SubagentProgress[]
  readonly failures: readonly SubagentProgress[]
}

export function workflowAgentState(agent: SubagentProgress): WorkflowAgentState {
  if (agent.status === 'running' || agent.status === 'queued') return 'working'
  if (agent.status === 'completed') return 'done'
  return 'failed'
}

/**
 * One workflow run's agents, found by the run name its tool row shows. Later
 * entries win for a repeated id (live over archived). Phases keep the order
 * the run first reached them.
 */
export function summarizeWorkflowRoster(agents: readonly SubagentProgress[], label: string): WorkflowRosterSummary {
  const byId = new Map<string, SubagentProgress>()
  for (const agent of agents) if (agent.group?.label === label) byId.set(agent.id, agent)
  const members = [...byId.values()]
  const phases = new Map<string, { total: number; working: number; done: number; failed: number }>()
  let working = 0, done = 0, failed = 0
  for (const agent of members) {
    const state = workflowAgentState(agent)
    const name = agent.group?.phase ?? ''
    const phase = phases.get(name) ?? { total: 0, working: 0, done: 0, failed: 0 }
    phase.total += 1
    phase[state === 'working' ? 'working' : state === 'done' ? 'done' : 'failed'] += 1
    phases.set(name, phase)
    if (state === 'working') working += 1
    else if (state === 'done') done += 1
    else failed += 1
  }
  return {
    total: members.length,
    working,
    done,
    failed,
    phases: [...phases.entries()].map(([name, counts]) => ({ name, ...counts })),
    running: members.filter(agent => workflowAgentState(agent) === 'working'),
    failures: members.filter(agent => workflowAgentState(agent) === 'failed'),
  }
}
