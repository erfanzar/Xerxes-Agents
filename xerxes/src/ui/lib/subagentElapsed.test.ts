// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import type { SubagentProgress } from '../types.js'
import { subagentElapsedSeconds } from './subagentElapsed.js'

const baseAgent: SubagentProgress = {
  depth: 0,
  goal: 'runtime',
  id: 'a1',
  index: 0,
  notes: [],
  parentId: null,
  startedAt: 1000,
  status: 'running',
  taskCount: 1,
  thinking: [],
  toolCount: 0,
  tools: []
}

describe('subagentElapsedSeconds', () => {
  it('keeps running agents live', () => {
    expect(subagentElapsedSeconds(baseAgent, 3500)).toBe(2.5)
  })

  it('freezes completed agents with recorded duration', () => {
    expect(subagentElapsedSeconds({ ...baseAgent, durationSeconds: 4, status: 'completed' }, 100_000)).toBe(4)
  })

  it('does not keep aging terminal agents without duration', () => {
    expect(subagentElapsedSeconds({ ...baseAgent, status: 'completed' }, 100_000)).toBeNull()
  })
})
