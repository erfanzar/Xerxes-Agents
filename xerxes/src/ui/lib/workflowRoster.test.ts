// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import type { SubagentProgress } from '../types.js'

import { summarizeToolStartDisplay, workflowRunFromLine } from './toolStartDisplay.js'
import { summarizeWorkflowRoster } from './workflowRoster.js'

const agent = (id: string, status: SubagentProgress['status'], phase: string, label = 'Review vnext'): SubagentProgress => ({
  id, status, goal: id, depth: 0, index: 0, notes: [], parentId: null, taskCount: 0, thinking: [], toolCount: 0, tools: [],
  group: { id: 'wf_1', label, phase },
} as unknown as SubagentProgress)

describe('workflow roster', () => {
  it('shows a Workflow call by its run name and never its script', () => {
    const display = summarizeToolStartDisplay('Workflow', '', JSON.stringify({ name: 'Review vnext', script: 'const secret = 1; return agent("x")' }))
    expect(display.context).toBe('run: Review vnext')
    expect(workflowRunFromLine('Workflow("run: Review vnext") ✓')).toBe('Review vnext')
    expect(workflowRunFromLine('Spawn Agents("2 agents: a, b")')).toBeNull()
  })

  it('tallies one run by phase and keeps rows only for running and failed agents', () => {
    const summary = summarizeWorkflowRoster([
      agent('a', 'completed', 'Slice'),
      agent('b', 'running', 'Review'),
      agent('c', 'failed', 'Review'),
      agent('d', 'completed', 'Review'),
      agent('x', 'running', 'Other', 'Another run'),
      agent('b', 'completed', 'Review'),
    ], 'Review vnext')
    expect(summary).toMatchObject({ total: 4, working: 0, done: 3, failed: 1 })
    expect(summary.phases).toEqual([
      { name: 'Slice', total: 1, working: 0, done: 1, failed: 0 },
      { name: 'Review', total: 3, working: 0, done: 2, failed: 1 },
    ])
    expect(summary.failures.map(item => item.id)).toEqual(['c'])
  })
})
