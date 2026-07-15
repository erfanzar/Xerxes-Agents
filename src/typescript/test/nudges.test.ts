// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  MemoryNudge,
  NudgeContext,
  NudgeManager,
  type NudgeRule,
  SkillNudge,
} from '../src/runtime/nudges.js'

test('memory nudges fire only at their interval when durable information was not persisted', () => {
  const nudge = new MemoryNudge({ interval: 4 })

  expect(nudge.shouldFire(new NudgeContext({
    turnIndex: 3,
    lastUserMessage: 'Please remember my deadline',
  }))).toBe(true)
  expect(nudge.shouldFire(new NudgeContext({
    turnIndex: 3,
    lastUserMessage: 'What time is it?',
  }))).toBe(false)
  expect(nudge.shouldFire(new NudgeContext({
    turnIndex: 2,
    lastAssistantMessage: 'I will remember this.',
  }))).toBe(false)
  expect(nudge.shouldFire(new NudgeContext({
    turnIndex: 3,
    lastUserMessage: 'Remember this preference.',
    memoryWritesSinceLastFire: 1,
  }))).toBe(false)
  expect(nudge.message(new NudgeContext({ turnIndex: 0 }))).toContain('save_memory')
})

test('skill nudges use the successful tool-call threshold', () => {
  const nudge = new SkillNudge({ threshold: 6 })

  expect(nudge.shouldFire(new NudgeContext({ turnIndex: 0, successfulToolCallsThisTurn: 6 }))).toBe(true)
  expect(nudge.shouldFire(new NudgeContext({ turnIndex: 0, successfulToolCallsThisTurn: 5 }))).toBe(false)
  expect(nudge.message(new NudgeContext({ turnIndex: 0 }))).toContain('skill_manage')
})

test('nudge manager keeps default rules, supports per-rule disablement, and tracks fires', () => {
  const defaults = new NudgeManager([])
  expect(new Set(defaults.rules.map(rule => rule.name))).toEqual(new Set(['memory', 'skill']))

  defaults.disable('memory')
  expect(defaults.check(new NudgeContext({
    turnIndex: 7,
    lastUserMessage: 'Please remember my preferred format.',
  }))).toEqual([])
  expect(defaults.disabled()).toEqual(new Set(['memory']))
  defaults.enable('memory')
  expect(defaults.disabled()).toEqual(new Set())

  const manager = new NudgeManager([new SkillNudge({ threshold: 1 })])
  const context = new NudgeContext({ turnIndex: 0, successfulToolCallsThisTurn: 1 })
  expect(manager.check(context)).toHaveLength(1)
  manager.check(context)
  manager.check(context)
  expect(manager.firedCount('skill')).toBe(3)
})

test('nudge manager accepts custom rules and context applies dataclass-equivalent defaults', () => {
  const always: NudgeRule = {
    name: 'always',
    shouldFire: () => true,
    message: () => 'always',
  }
  const context = new NudgeContext({ turnIndex: 0 })
  expect(context).toMatchObject({
    toolCallsThisTurn: 0,
    successfulToolCallsThisTurn: 0,
    memoryWritesThisTurn: 0,
    memoryWritesSinceLastFire: 0,
    lastUserMessage: '',
    lastAssistantMessage: '',
    metadata: {},
  })
  expect(new NudgeManager([always]).check(context)).toEqual([['always', 'always']])
})
