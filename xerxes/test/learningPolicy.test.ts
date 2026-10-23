// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { assembleContextLayers } from '../src/context/assembly.js'
import { evaluateBudget, measureLayers } from '../src/context/promptBudget.js'
import { learningPolicyPrompt } from '../src/runtime/learningPolicy.js'

const ALL = new Set(['lesson_record', 'lesson_recall', 'session_search'])

test('the learning policy says when to recall, record and search, and separates lessons from memory', () => {
  const text = learningPolicyPrompt({ tools: ALL, reflection: 'all' })
  expect(text.startsWith('[Learning policy]')).toBe(true)
  expect(text).toContain('call lesson_recall')
  expect(text).toContain('call lesson_record')
  expect(text).toContain('call session_search')
  expect(text).toContain('Lessons versus memory')
  expect(text).toContain('after a long turn (25 tool calls or 20 minutes)')
  expect(text).toContain('Never record an unverified guess')
})

test('the policy covers exactly the visible tools and the reflection setting', () => {
  expect(learningPolicyPrompt({ tools: new Set(['ReadFile']), reflection: 'all' })).toBe('')
  const searchOnly = learningPolicyPrompt({ tools: new Set(['session_search']), reflection: 'off' })
  expect(searchOnly).toContain('session_search')
  expect(searchOnly).not.toContain('lesson_record')
  expect(searchOnly).not.toContain('Reflection:')
  const goals = learningPolicyPrompt({ tools: ALL, reflection: 'goals' })
  expect(goals).toContain('Reflection: after a goal completes or is blocked, the host')
  expect(goals).not.toContain('long turn')
})

test('it is its own system layer right after the goal policy, and stays within the prompt budget', () => {
  const learningPolicy = learningPolicyPrompt({ tools: ALL, reflection: 'all' })
  const layers = assembleContextLayers({ bootstrap: 'b', agentPrompt: 'a', toolGuidance: 't', goalPolicy: 'g', learningPolicy, modeHint: '', subagentJoin: '' })
  expect(layers.map(layer => layer.name)).toEqual(['bootstrap', 'agent', 'tool_guidance', 'goal_policy', 'learning_policy'])
  expect(learningPolicy.length).toBeLessThan(3_000)
  expect(evaluateBudget(measureLayers(layers)).ok).toBe(true)
})
