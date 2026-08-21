// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import type { SubagentProgress } from '../types.js'
import { fmtDuration, fmtTokens, subagentElapsedSeconds } from './subagentElapsed.js'

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

describe('fmtTokens', () => {
  it('formats compact token counts', () => {
    expect(fmtTokens(0)).toBe('0')
    expect(fmtTokens(542)).toBe('542')
    expect(fmtTokens(1200)).toBe('1.2k')
    expect(fmtTokens(12_000)).toBe('12k')
    expect(fmtTokens(Number.NaN)).toBe('0')
  })
})

describe('fmtDuration', () => {
  it('formats seconds, minutes, and compound durations', () => {
    expect(fmtDuration(0)).toBe('0s')
    expect(fmtDuration(42.4)).toBe('42s')
    expect(fmtDuration(60)).toBe('1m')
    expect(fmtDuration(95)).toBe('1m 35s')
  })
})
