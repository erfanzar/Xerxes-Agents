// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { JsonObject } from '../src/types/toolCalls.js'
import { resetGoalActivations } from '../src/runtime/goalDomain.js'
import { goalPolicyPrompt, registerGoalTools, type GoalToolHost } from '../src/runtime/goalTools.js'
import { admitGoalRound } from '../src/runtime/goalDomain.js'

afterEach(() => resetGoalActivations())

interface Harness {
  registry: ToolRegistry
  metadata: Record<string, unknown>
  call: (name: string, inputs?: JsonObject) => Promise<any>
  setTurn: (turn: { human?: boolean; round?: number; evidence?: boolean }) => void
  /** The live compare-and-set ref, as the tools require it. */
  ref: () => Promise<{ goal_id: string; revision: number }>
  /** Admit real rounds, then open the turn as the latest one — as production does. */
  enterRound: (count?: number) => void
}

function harness(options: { blockedAfter?: number } = {}): Harness {
  const metadata: Record<string, unknown> = {}
  let turn = { human: true, round: undefined as number | undefined, evidence: true }
  const host: GoalToolHost = {
    sessionId: () => 'session-1',
    metadata: () => metadata,
    isHumanTurn: () => turn.human,
    currentRound: () => turn.round,
    now: () => 1_000,
  }
  const registry = new ToolRegistry()
  registerGoalTools(registry, host, options.blockedAfter === undefined ? {} : { blockedAfterConsecutiveRounds: options.blockedAfter })
  return {
    registry,
    metadata,
    call: async (name, inputs: JsonObject = {}) => JSON.parse(await registry.execute(
      { id: `c-${name}`, type: 'function', function: { name, arguments: inputs } },
      { agentId: 'default', metadata, sessionId: 'session-1' },
    )),
    setTurn: patch => { turn = { ...turn, ...patch } },
    ref: async () => {
      const current = JSON.parse(await registry.execute(
        { id: 'c-ref', type: 'function', function: { name: 'get_goal', arguments: {} } },
        { agentId: 'default', metadata, sessionId: 'session-1' },
      ))
      return { goal_id: current.goal.id, revision: current.goal.revision }
    },
    enterRound: (count = 1) => {
      let source
      for (let index = 0; index < count; index += 1) {
        source = admitGoalRound(metadata, 'session-1', 1_000)
      }
      turn = { ...turn, human: false, round: source?.round }
    },
  }
}

test('get_goal reports null before a goal exists, then the CAS ref', async () => {
  const h = harness()
  expect(await h.call('get_goal')).toEqual({ goal: null })

  await h.call('create_goal', { objective: 'make the loop cancel-safe' })
  const read = await h.call('get_goal')
  expect(read.goal).toMatchObject({ revision: 1, phase: 'active', objective: 'make the loop cancel-safe' })
  expect(read.activation).toBe('armed')
})

test('lifecycle is a typed transition, not a phrase in the prose', async () => {
  const h = harness()
  const created = await h.call('create_goal', { objective: 'ship' })
  const ref = { goal_id: created.goal.id, revision: created.goal.revision }

  // The old guard could only end a goal if the model wrote an English marker.
  // Here the model says what it means and gets a definite answer.
  const done = await h.call('update_goal', { ...ref, action: 'complete' })
  expect(done.goal).toMatchObject({ phase: 'complete' })
})

test('a stale revision is returned as a retryable result, not a thrown turn', async () => {
  const h = harness()
  const created = await h.call('create_goal', { objective: 'ship' })

  const stale = await h.call('update_goal', { goal_id: created.goal.id, revision: 99, action: 'pause' })
  expect(stale).toMatchObject({ ok: false, code: 'GOAL_STALE_REVISION' })
  expect(String(stale.error)).toContain('current is 1')
})

test('creating, editing, pausing and resuming require a direct human turn', async () => {
  const h = harness()
  const created = await h.call('create_goal', { objective: 'ship' })
  const ref = { goal_id: created.goal.id, revision: created.goal.revision }

  // An automatic continuation round has no authority to redefine its own goal.
  h.setTurn({ human: false, round: 4 })
  await expect(h.call('update_goal', { ...ref, action: 'edit', objective: 'something else' }))
    .rejects.toThrow('requires a direct human turn')
  await expect(h.call('update_goal', { ...ref, action: 'pause' })).rejects.toThrow('requires a direct human turn')
  await expect(h.call('create_goal', { objective: 'a second goal' })).rejects.toThrow('requires a direct human turn')
})

test('completion is not gated on mechanically detected evidence', async () => {
  const h = harness()
  await h.call('create_goal', { objective: 'ship' })
  h.enterRound(5)

  // A gate here used to require a command from a hardcoded list of
  // "verification" names. A live run failed on it: the model proved its work
  // with `cmp` (exit 0), was refused because `cmp` is not on the list, and
  // deleted its own correct output to start over. A whitelist cannot enumerate
  // how a thing is checked, so the requirement is stated in the policy prompt
  // and in the closing brief instead — where it can be about the work rather
  // than about which binary was invoked.
  const completed = await h.call('update_goal', { ...(await h.ref()), action: 'complete' })
  expect(completed.goal).toMatchObject({ phase: 'complete' })
  expect(completed.wrapup).toContain('how it was verified')
})

test('the policy tells the model to run the check in the turn that completes', () => {
  const policy = goalPolicyPrompt(3)
  expect(policy).toContain('THIS turn ran the check that proves it')
  expect(policy).toContain('run it')
  expect(policy).toContain('at least 3 consecutive goal rounds')
})

test('a round that is not the goal\'s current one carries no concluding authority', async () => {
  const h = harness()
  await h.call('create_goal', { objective: 'ship' })
  h.enterRound(3)
  // The turn believes it is round 2 while the goal has admitted 3: stale
  // authority, which is exactly when a completion claim is least trustworthy.
  h.setTurn({ round: 2 })
  await expect(h.call('update_goal', { ...(await h.ref()), action: 'complete' }))
    .rejects.toThrow('current continuation round')
})

test('self-blocking is mechanically rejected before the configured round', async () => {
  const h = harness({ blockedAfter: 3 })
  await h.call('create_goal', { objective: 'ship' })

  h.enterRound(1)
  const early = await h.call('update_goal', {
    ...(await h.ref()),
    action: 'blocked',
    blocked_reason: 'this is hard',
  })
  expect(early).toMatchObject({ ok: false })
  expect(String(early.error)).toContain('rejected before round 3')

  h.enterRound(2)
  const blocked = await h.call('update_goal', {
    ...(await h.ref()),
    action: 'blocked',
    blocked_reason: 'no credentials on this host',
  })
  expect(blocked.goal).toMatchObject({ phase: 'blocked', blockedReason: { code: 'model-reported' } })
})

test('an autonomous conclusion is briefed to address the user; a human one is not', async () => {
  const h = harness()
  const created = await h.call('create_goal', { objective: 'ship' })
  const ref = { goal_id: created.goal.id, revision: created.goal.revision }

  // Human-driven: the person is right there, so no closing brief is attached.
  const byHuman = await h.call('update_goal', { ...ref, action: 'complete' })
  expect(byHuman.wrapup).toBeUndefined()

  await h.call('create_goal', { objective: 'again' })
  h.enterRound(2)
  const autonomous = await h.call('update_goal', { ...(await h.ref()), action: 'complete' })
  // The run must not end on a silent tool call: the model gets one more
  // inference with an explicit instruction to report the outcome.
  expect(autonomous.wrapup).toContain('<goal_complete>')
  expect(autonomous.wrapup).toContain('"again"')
  expect(autonomous.wrapup).toContain('Address the user directly')
})

test('an autonomous block briefs the model to name the blocker to the user', async () => {
  const h = harness({ blockedAfter: 1 })
  await h.call('create_goal', { objective: 'reach the API' })
  h.enterRound(3)
  const blocked = await h.call('update_goal', {
    ...(await h.ref()),
    action: 'blocked',
    blocked_reason: 'no credentials on this host',
  })
  expect(blocked.wrapup).toContain('<goal_blocked>')
  expect(blocked.wrapup).toContain('no credentials on this host')
})

test('subagents are refused every goal operation', async () => {
  const h = harness()
  h.metadata.session_kind = 'subagent'
  for (const [name, inputs] of [
    ['get_goal', {}],
    ['create_goal', { objective: 'x' }],
    ['update_goal', { goal_id: 'g', revision: 1, action: 'pause' }],
  ] as const) {
    await expect(h.call(name, inputs)).rejects.toThrow('only the main agent')
  }
})

test('empty-string and zero fillers count as omitted', async () => {
  const h = harness()
  const created = await h.call('create_goal', { objective: 'ship', max_goal_rounds: 0 })
  // A strict-schema model emits 0 for an integer it means to omit; taking it
  // literally would create a goal that can never run a round.
  expect(created.goal.maxGoalRounds).toBeGreaterThan(0)

  const ref = { goal_id: created.goal.id, revision: created.goal.revision }
  const edited = await h.call('update_goal', { ...ref, action: 'edit', objective: '', max_goal_rounds: 9 })
  expect(edited.goal).toMatchObject({ objective: 'ship', maxGoalRounds: 9 })
})
