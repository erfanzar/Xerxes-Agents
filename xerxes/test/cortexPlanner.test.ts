// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { executeExecutionPlan, PlanExecutionAbortedError } from '../src/cortex/planner.js'

test('execution plans propagate cancellation and cap parallel step concurrency', async () => {
  const plan = {
    id: 'bounded-plan',
    objective: 'bounded cancellable work',
    complexity: 'low' as const,
    estimatedMinutes: 0,
    steps: Array.from({ length: 8 }, (_value, index) => ({
      id: String(index),
      action: 'run',
      description: `Step ${index}`,
      arguments: {},
      dependencies: [],
    })),
  }
  const tracked = trackConcurrency()
  await executeExecutionPlan(plan, tracked.executor, { parallel: true, maxParallel: 2 })
  expect(tracked.maxActive).toBe(2)

  const controller = new AbortController()
  let started = 0
  const pending = executeExecutionPlan(plan, async () => {
    started += 1
    controller.abort(new Error('stop plan'))
    await delay(2)
    return 'late'
  }, { parallel: true, maxParallel: 1, signal: controller.signal })
  await expect(pending).rejects.toBeInstanceOf(PlanExecutionAbortedError)
  expect(started).toBe(1)
})

function delay(milliseconds: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, milliseconds))
}

function trackConcurrency(): { readonly executor: () => Promise<string>; readonly maxActive: number } {
  const state = { active: 0, maxActive: 0 }
  return {
    get maxActive() {
      return state.maxActive
    },
    executor: async () => {
      state.active += 1
      state.maxActive = Math.max(state.maxActive, state.active)
      await delay(5)
      state.active -= 1
      return 'ok'
    },
  }
}
