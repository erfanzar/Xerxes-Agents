// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { contextCalibrationRatio, estimateContextTokens } from '../src/context/windowUsage.js'
import { compactMessagesIfNeeded } from '../src/daemon/compactionRunner.js'
import { compactionBudget } from '../src/daemon/server.js'

/**
 * The context meter multiplies its estimate by the task's measured
 * calibration; compaction counts the plain estimate. Handed the raw budget,
 * compaction of a task calibrated at 3.8x judged 437K tokens to fit while the
 * meter read 206%, shed nothing, and every turn stopped.
 */

const model = 'claude-code/opus'
const metadata = { context_calibration: { model, ratio: 3.8 } }
const session = { metadata, messages: [] } as unknown as Parameters<typeof compactionBudget>[1]

function giantRound(calls: number, resultChars: number): Array<Record<string, unknown>> {
  const ids = Array.from({ length: calls }, (_, index) => `call_${index}`)
  return [
    { role: 'user', content: 'Review the source.' },
    { role: 'assistant', content: '', tool_calls: ids.map((id, index) => ({ id, type: 'function', function: { name: 'ReadFile', arguments: JSON.stringify({ offset: index }) } })) },
    ...ids.map(id => ({ role: 'tool', tool_call_id: id, content: 'x'.repeat(resultChars) })),
    { role: 'user', content: '?' },
  ]
}

test('the budget handed to compaction is in its own units: divided by the calibration', () => {
  expect(contextCalibrationRatio(metadata, model)).toBeCloseTo(3.8)
  expect(compactionBudget(968_000, session, model)).toBe(Math.floor(968_000 / 3.8))
  expect(compactionBudget(968_000, { metadata: {}, messages: [] } as never, model)).toBe(968_000)
})

test('a task the meter reads as over budget is compacted until the meter reads it as fitting', async () => {
  const budget = 968_000
  const messages = giantRound(300, 8_000)
  expect(estimateContextTokens(messages, { model })).toBeLessThan(budget)
  const meter = (list: readonly Record<string, unknown>[]) => 3.8 * estimateContextTokens(list, { model })
  expect(meter(messages)).toBeGreaterThan(budget)
  // With the raw budget compaction sees the plain estimate as fitting and does nothing.
  const raw = await compactMessagesIfNeeded({ model, messages, reason: 'test', maxContextTokens: budget, completion: async () => ({ content: 'summary' }) })
  expect(raw.compacted).toBe(false)
  const outcome = await compactMessagesIfNeeded({ model, messages, reason: 'test', maxContextTokens: compactionBudget(budget, session, model), completion: async () => ({ content: 'summary' }) })
  expect(outcome.compacted).toBe(true)
  if (outcome.compacted) expect(meter(outcome.messages)).toBeLessThanOrEqual(budget)
})
