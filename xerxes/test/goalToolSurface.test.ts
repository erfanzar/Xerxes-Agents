// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The goal tools must survive the agent tool filter, not merely be registered.
 *
 * This is the exact bug a live run caught after every unit test passed: the
 * tools were registered, present in the 79-schema registry snapshot, and named
 * in the composition root — and the model still reported them unavailable,
 * because the default agent declares an explicit `tools:` allow-list and no
 * entry had been added to it. The model spent an entire 24-round budget saying
 * "objective complete, the update_goal tool is unavailable".
 *
 * Registration and exposure are two different things, and only the second one
 * is what the model sees.
 */

import { expect, test } from 'bun:test'

import { BUILTIN_AGENTS } from '../src/agents/definitions.js'
import { GOAL_TOOL_DEFINITIONS } from '../src/runtime/goalTools.js'

const GOAL_TOOL_NAMES = GOAL_TOOL_DEFINITIONS.map(definition => definition.function.name)

/**
 * Which built-in agent surfaces must carry which goal tools.
 *
 * Modes that may change the world get the whole surface. Modes that may not —
 * plan and research — still get the read, because a mode that cannot see the
 * session's objective cannot reason about it either.
 */
const EXPECTED: ReadonlyArray<{ readonly agent: string; readonly tools: readonly string[] }> = [
  { agent: 'default', tools: GOAL_TOOL_NAMES },
  { agent: 'objective', tools: GOAL_TOOL_NAMES },
  { agent: 'planner', tools: ['get_goal'] },
  { agent: 'researcher', tools: ['get_goal'] },
]

test('goal tools are exposed on the agent surfaces that need them', () => {
  expect(GOAL_TOOL_NAMES).toEqual(['get_goal', 'create_goal', 'update_goal'])

  for (const expectation of EXPECTED) {
    const definition = BUILTIN_AGENTS.get(expectation.agent)
    expect(definition, `built-in agent "${expectation.agent}" must exist`).toBeDefined()
    // An empty `tools` list means "no restriction", so it is a pass; a
    // non-empty one is an allow-list and every goal tool must be named in it.
    if (definition!.tools.length === 0) continue
    for (const name of expectation.tools) {
      expect(
        { agent: expectation.agent, tool: name, exposed: definition!.tools.includes(name) },
        `${name} must be on the ${expectation.agent} surface`,
      ).toEqual({ agent: expectation.agent, tool: name, exposed: true })
    }
    // And nothing may exclude them back out again.
    for (const name of expectation.tools) {
      expect(definition!.excludeTools).not.toContain(name)
    }
    if (expectation.tools.length < GOAL_TOOL_NAMES.length) {
      for (const name of GOAL_TOOL_NAMES.filter(candidate => !expectation.tools.includes(candidate))) {
        expect(definition!.excludeTools, `${name} must be excluded from ${expectation.agent}`)
          .toContain(name)
      }
    }
  }
})

test('a read-only mode is never handed the mutating goal tools', () => {
  for (const agent of ['planner', 'researcher']) {
    const definition = BUILTIN_AGENTS.get(agent)!
    // These modes inherit the default agent's declared surface, so the goal
    // tools arrive here whether or not this file names them; what keeps the
    // mutating pair out is the exclusion, exactly as it keeps WriteFile out.
    expect(definition.excludeTools).toContain('create_goal')
    expect(definition.excludeTools).toContain('update_goal')
    // The read stays: a mode that cannot see the session's objective cannot
    // reason about it either.
    expect(definition.excludeTools).not.toContain('get_goal')
  }
})
