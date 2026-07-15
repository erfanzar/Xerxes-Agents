// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  CostEvent,
  CostTracker,
  UNSCOPED_COST_SCOPE,
} from '../src/runtime/costTracker.js'

test('cost tracker uses shared provider pricing and Python-compatible cache multipliers', () => {
  const tracker = new CostTracker({
    agentId: 'planner',
    sessionId: 'session-a',
    now: () => new Date('2026-07-13T10:00:00.000Z'),
  })
  const event = tracker.recordTurn('openai/gpt-4o', 1_000, 500, 'turn_1', {
    cacheReadTokens: 100,
    cacheCreationTokens: 20,
  })

  expect(event).toMatchObject({
    model: 'openai/gpt-4o',
    inputTokens: 1_000,
    outputTokens: 500,
    label: 'turn_1',
    timestamp: '2026-07-13T10:00:00.000Z',
    cacheReadTokens: 100,
    cacheCreationTokens: 20,
    sessionId: 'session-a',
    agentId: 'planner',
  })
  expect(event.costUsd).toBeCloseTo(0.0075875, 12)
  expect(tracker.totalCostUsd).toBeCloseTo(0.0075875, 12)
  expect(tracker.totalInputTokens).toBe(1_000)
  expect(tracker.totalOutputTokens).toBe(500)
  expect(tracker.totalTokens).toBe(1_500)
  expect(tracker.totalCacheReadTokens).toBe(100)
  expect(tracker.totalCacheCreationTokens).toBe(20)
  expect(tracker.cacheHitRate()).toBeCloseTo(100 / 1_100, 12)
  expect(tracker.summary()).toContain('Total cost: $0.0076')
  expect(tracker.summary()).toContain('**openai/gpt-4o**: $0.0076 (1 turns, 1,500 tokens)')

  expect(tracker.asDicts()).toEqual([{
    model: 'openai/gpt-4o',
    in_tokens: 1_000,
    out_tokens: 500,
    cost_usd: event.costUsd,
    label: 'turn_1',
    timestamp: '2026-07-13T10:00:00.000Z',
  }])
  expect(tracker.asRecords()[0]).toMatchObject({
    cache_read_tokens: 100,
    cache_creation_tokens: 20,
    session_id: 'session-a',
    agent_id: 'planner',
  })
})

test('cost tracker groups a shared ledger by model, session, and agent without losing raw costs', () => {
  const tracker = new CostTracker({
    costCalculator: (model, inputTokens, outputTokens) => {
      if (model !== 'priced') return 0
      return (inputTokens + outputTokens * 2) / 1_000
    },
  })

  tracker.recordTurn('priced', 100, 50, 's1-a1', { sessionId: 'session-1', agentId: 'agent-1' })
  tracker.recordTurn('priced', 40, 10, 's1-a2', { sessionId: 'session-1', agentId: 'agent-2' })
  tracker.recordRaw('image', 0.03, 'image-model', { sessionId: 'session-2', agentId: 'agent-1' })
  tracker.recordRaw('unattributed', 0.01)

  expect(tracker.byModel()).toMatchObject({
    priced: { turns: 2, inputTokens: 140, outputTokens: 60, costUsd: 0.26 },
    'image-model': { turns: 1, costUsd: 0.03 },
  })
  expect(tracker.bySession()).toMatchObject({
    'session-1': { turns: 2, inputTokens: 140, outputTokens: 60, costUsd: 0.26 },
    'session-2': { turns: 1, costUsd: 0.03 },
    [UNSCOPED_COST_SCOPE]: { turns: 1, costUsd: 0.01 },
  })
  expect(tracker.byAgent()).toMatchObject({
    'agent-1': { turns: 2, costUsd: 0.23 },
    'agent-2': { turns: 1, costUsd: 0.06 },
    [UNSCOPED_COST_SCOPE]: { turns: 1, costUsd: 0.01 },
  })
  expect(tracker.forSession('session-1')).toMatchObject({ turns: 2, tokens: 200, costUsd: 0.26 })
  expect(tracker.forAgent('agent-1')).toMatchObject({ turns: 2, costUsd: 0.23 })

  const snapshot = tracker.events
  expect(snapshot).toHaveLength(4)
  expect(tracker.eventCount).toBe(4)
  expect(tracker.forSession('missing')).toMatchObject({ turns: 0, costUsd: 0, cacheHitRate: 0 })
})

test('cost events are immutable, unknown models cost zero, and invalid accounting values fail fast', () => {
  const tracker = new CostTracker({ now: () => new Date('2026-07-13T10:00:00.000Z') })
  const unknown = tracker.recordTurn('unknown-model', 200, 100, 'unknown', { cacheReadTokens: 50 })
  expect(unknown.costUsd).toBe(0)
  expect(Object.isFrozen(unknown)).toBe(true)
  expect(() => tracker.recordTurn('gpt-4o', -1, 0)).toThrow('inputTokens')
  expect(() => tracker.recordRaw('bad', Number.NaN)).toThrow('costUsd')
  expect(() => new CostEvent({
    model: 'gpt-4o',
    inputTokens: 0,
    outputTokens: 0,
    costUsd: 0,
    timestamp: 'not-a-date',
  })).toThrow('timestamp')

  const calls: Array<readonly [number, number]> = []
  const probeTracker = new CostTracker({
    costCalculator: (_model, inputTokens, outputTokens) => {
      calls.push([inputTokens, outputTokens])
      return 1
    },
  })
  probeTracker.recordTurn('fixture', 1, 2)
  expect(calls).toEqual([[1, 2]])
  probeTracker.recordTurn('fixture', 1, 2, '', { cacheReadTokens: 1 })
  expect(calls).toEqual([[1, 2], [1, 2], [1_000, 0]])

  tracker.clear()
  expect(tracker.eventCount).toBe(0)
  expect(tracker.totalCost).toBe(0)
})
