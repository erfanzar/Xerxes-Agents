// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { JsonObject } from '../src/types/toolCalls.js'
import { resetGoalActivations, getGoal, recordGoalEvidence } from '../src/runtime/goalDomain.js'
import { goalPolicyPrompt, registerGoalTools, type GoalToolHost } from '../src/runtime/goalTools.js'
import { admitGoalRound } from '../src/runtime/goalDomain.js'

afterEach(() => resetGoalActivations())

test('human tools configure goal wall-time limits and autonomous rounds cannot raise them', async () => {
  const h = harness()
  const created = await h.call('create_goal', { objective: 'bounded work', max_duration_ms: 60_000 })
  expect(created.goal.maxDurationMs).toBe(60_000)
  expect(created.goal.deadlineAt).toBe(61_000)
  h.enterRound()
  await expect(h.call('update_goal', { ...await h.ref(), action: 'edit', max_duration_ms: 120_000 })).rejects.toThrow()
  h.setTurn({ human: true })
  const edited = await h.call('update_goal', { ...await h.ref(), action: 'edit', max_duration_ms: 120_000 })
  expect(edited.goal.deadlineAt).toBe(121_000)
})

test('goal tools expose token caps and restrict edits to human turns', async () => {
  const h = harness()
  const created = await h.call('create_goal', { objective: 'bounded work', max_total_tokens: 1000 })
  expect(created.goal.maxTotalTokens).toBe(1000)
  h.enterRound()
  await expect(h.call('update_goal', { ...await h.ref(), action: 'edit', max_total_tokens: 2000 })).rejects.toThrow()
  h.setTurn({ human: true })
  expect((await h.call('update_goal', { ...await h.ref(), action: 'edit', max_total_tokens: 2000 })).goal.maxTotalTokens).toBe(2000)
})

interface Harness {
  registry: ToolRegistry
  metadata: Record<string, unknown>
  call: (name: string, inputs?: JsonObject) => Promise<any>
  setTurn: (turn: { human?: boolean; round?: number | undefined; evidence?: boolean }) => void
  /** The live compare-and-set ref, as the tools require it. */
  ref: () => Promise<{ goal_id: string; revision: number }>
  /** Admit real rounds, then open the turn as the latest one — as production does. */
  enterRound: (count?: number) => void
}

test('milestones are visible and can advance during the current goal round without changing the objective', async () => {
  const h = harness()
  const initial = await h.call('create_goal', { objective: 'ship with evidence', current_milestone: 'Map the runtime',
    criteria: [{ id: 'verified', description: 'Run the required checks' }] })
  expect(initial.goal.currentMilestone).toBe('Map the runtime')
  h.enterRound()
  const before = await h.ref()
  const updated = await h.call('update_goal', { ...before, action: 'milestone', current_milestone: 'Verify restart recovery' })
  expect(updated.goal).toMatchObject({ objective: 'ship with evidence', phase: 'active', roundsStarted: 1,
    currentMilestone: 'Verify restart recovery', criteria: [{ id: 'verified', description: 'Run the required checks' }] })
  expect(updated.goal.revision).toBe(before.revision + 1)
  expect((await h.call('update_goal', { ...before, action: 'milestone', current_milestone: 'stale' })).ok).toBe(false)
  expect((await h.call('update_goal', { ...await h.ref(), action: 'complete' })).ok).toBe(false)
  const cleared = await h.call('update_goal', { ...await h.ref(), action: 'milestone', current_milestone: null })
  expect(cleared.goal.currentMilestone).toBeUndefined()
})

test('milestone tool updates cannot smuggle objective edits or use background authority', async () => {
  const h = harness()
  await h.call('create_goal', { objective: 'keep the requirement' })
  const before = await h.ref()
  await expect(h.call('update_goal', { ...before, action: 'milestone', current_milestone: 'next', objective: 'replace it' })).rejects.toThrow()
  await expect(h.call('update_goal', { ...before, action: 'milestone' })).rejects.toThrow('current_milestone')
  h.setTurn({ human: false, round: undefined })
  await expect(h.call('update_goal', { ...before, action: 'milestone', current_milestone: 'unrelated scheduled job' })).rejects.toThrow()
  h.enterRound()
  h.setTurn({ round: 99 })
  await expect(h.call('update_goal', { ...await h.ref(), action: 'milestone', current_milestone: 'stale worker' })).rejects.toThrow()
  h.setTurn({ human: true, round: undefined })
  expect((await h.call('update_goal', { ...await h.ref(), action: 'milestone', current_milestone: 'x'.repeat(1001) })).ok).toBe(false)
  expect((await h.call('get_goal')).goal.currentMilestone).toBeUndefined()
})

function harness(options: { blockedAfter?: number; executions?: readonly unknown[]; validateResume?: GoalToolHost['validateResume'] } = {}): Harness {
  const metadata: Record<string, unknown> = {}
  let turn = { human: true, round: undefined as number | undefined, evidence: true }
  const host: GoalToolHost = {
    sessionId: () => 'session-1',
    metadata: () => metadata,
    isHumanTurn: () => turn.human,
    currentRound: () => turn.round,
    evidenceExecution: (_context, id) => options.executions?.find(record => !!record && typeof record === 'object' && (record as Record<string, unknown>).toolCallId === id),
    now: () => 1_000,
    ...(options.validateResume ? { validateResume: options.validateResume } : {}),
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

test('declared criteria need host-resolved successful evidence without a command whitelist', async () => {
  const executions = [
    { toolCallId: 'cmp', name: 'ExecCommand', permitted: true, result: '{"exit_code":0}' },
    { toolCallId: 'failed', name: 'ExecCommand', permitted: true, result: '{"exit_code":1}' },
    { toolCallId: 'pending', name: 'ExecCommand', permitted: true, result: '{"session_id":123,"exit_code":null}' },
    { toolCallId: 'denied', name: 'ReadFile', permitted: false, result: 'denied' },
  ]
  const h = harness({ executions })
  await h.call('create_goal', { objective: 'ship', criteria: [{ id: 'matches', description: 'Output matches the expected artifact' }] })
  expect((await h.call('update_goal', { ...(await h.ref()), action: 'complete' })).ok).toBe(false)
  for (const id of ['missing', 'failed', 'pending', 'denied']) {
    const before = await h.ref()
    expect((await h.call('update_goal', { ...before, action: 'record_evidence', criterion_id: 'matches', tool_call_id: id, evidence_summary: 'These match' })).ok).toBe(false)
    expect(await h.ref()).toEqual(before)
  }
  const recorded = await h.call('update_goal', { ...(await h.ref()), action: 'record_evidence', criterion_id: 'matches', tool_call_id: 'cmp', evidence_summary: 'cmp returned zero for the two artifacts' })
  expect(recorded.goal.criteria[0].evidence.toolCallId).toBe('cmp')
  expect((await h.call('update_goal', { ...(await h.ref()), action: 'complete' })).goal.phase).toBe('complete')
})

test('model tools can read a human decision but cannot create one', async () => {
  const h = harness()
  await h.call('create_goal', { objective: 'ship', criteria: [{ id: 'visual', description: 'User accepts the layout' }] })
  const before = await h.ref()
  await expect(h.call('update_goal', { ...before, action: 'user-decision', criterion_id: 'visual',
    evidence_summary: 'The user accepted it' })).rejects.toThrow()
  expect(await h.ref()).toEqual(before)
  const goal = getGoal(h.metadata, 'session-1')!
  recordGoalEvidence(h.metadata, 'session-1', goal, 'visual', {
    kind: 'user-decision', decisionId: 'trusted-host-decision', summary: 'I accept the layout.', recordedAt: 1000,
  }, 1000)
  expect((await h.call('get_goal')).goal.criteria[0].evidence).toMatchObject({ kind: 'user-decision', decisionId: 'trusted-host-decision' })
  expect((await h.call('update_goal', { ...await h.ref(), action: 'complete' })).goal.phase).toBe('complete')
})

test('legacy goals without declared criteria are not gated on mechanically detected evidence', async () => {
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

test('screenshot resume payload ignores unrelated fields while retaining milestone, criteria and caps', async () => {
  const h = harness()
  await h.call('create_goal', { objective: 'run ten scans', current_milestone: 'scan four', max_total_tokens: 2_000_000,
    max_duration_ms: 60_000, max_goal_rounds: 10, criteria: [{ id: 'verified', description: 'Run all scans' }] })
  await h.call('update_goal', { ...await h.ref(), action: 'pause' })
  const result = await h.call('update_goal', { ...await h.ref(), action: 'resume', objective: '', current_milestone: null,
    criteria: [], criterion_id: '', tool_call_id: '', evidence_summary: '', max_goal_rounds: 24,
    max_duration_ms: 86400000, max_total_tokens: 4000000, blocked_reason: '' })
  expect(result.goal).toMatchObject({ phase: 'active', objective: 'run ten scans', currentMilestone: 'scan four',
    maxTotalTokens: 2_000_000, maxDurationMs: 60_000, maxGoalRounds: 10,
    criteria: [{ id: 'verified', description: 'Run all scans' }] })
  expect(result.ignored_fields).toContain('max_total_tokens')
  expect(result.notice).toContain('Resume never changes limits')
})

test('exhausted resume returns recovery and unlimited explicitly clears caps without losing the goal', async () => {
  const h = harness({ validateResume: (_context, goal) => {
    if (goal.maxTotalTokens !== undefined) throw new Error('Goal token budget exhausted (2090726/2000000)')
  } })
  const created = await h.call('create_goal', { objective: 'run ten scans', max_total_tokens: 2_000_000, max_duration_ms: 60000, max_goal_rounds: 24 })
  await h.call('update_goal', { ...await h.ref(), action: 'blocked', blocked_reason: 'Token budget exhausted' })
  const before = await h.ref()
  const denied = await h.call('update_goal', { ...before, action: 'resume', current_milestone: null, max_total_tokens: 4_000_000 })
  expect(denied).toMatchObject({ ok: false, code: 'GOAL_RESUME_DENIED', goal: { phase: 'blocked' } })
  expect(denied.recovery).toContain('do not retry this unchanged resume')
  expect(await h.ref()).toEqual(before)
  h.setTurn({ human: false })
  await expect(h.call('update_goal', { ...before, action: 'unlimited' })).rejects.toThrow('direct human turn')
  h.setTurn({ human: true })
  const uncapped = await h.call('update_goal', { ...before, action: 'unlimited' })
  expect(uncapped.goal).toMatchObject({ id: created.goal.id, objective: 'run ten scans', phase: 'blocked', maxGoalRounds: Number.MAX_SAFE_INTEGER })
  expect(uncapped.goal.maxTotalTokens).toBeUndefined()
  expect(uncapped.goal.maxDurationMs).toBeUndefined()
  expect((await h.call('update_goal', { ...await h.ref(), action: 'resume' })).goal.phase).toBe('active')
})

test('milestone actions tolerate empty provider placeholders but reject meaningful edits', async () => {
  const h = harness()
  await h.call('create_goal', { objective: 'ship', current_milestone: null })
  const result = await h.call('update_goal', { ...await h.ref(), action: 'milestone', current_milestone: 'test', objective: '', criteria: [], blocked_reason: '' })
  expect(result.goal.currentMilestone).toBe('test')
  await expect(h.call('update_goal', { ...await h.ref(), action: 'milestone', current_milestone: null, objective: 'replacement' })).rejects.toThrow()
})

test('milestones ignore numeric provider fillers without changing goal budgets', async () => {
  const h = harness()
  const before = await h.call('create_goal', { objective: 'ship', max_goal_rounds: 10, max_duration_ms: 60000, max_total_tokens: 1000 })
  h.enterRound()
  const result = await h.call('update_goal', { ...await h.ref(), action: 'milestone', current_milestone: 'verify dashboard', objective: '', criteria: [], criterion_id: '', tool_call_id: '', evidence_summary: '', blocked_reason: '', max_goal_rounds: Number.MAX_SAFE_INTEGER, max_duration_ms: 1, max_total_tokens: 1 })
  expect(result.goal.currentMilestone).toBe('verify dashboard')
  for (const key of ['maxGoalRounds', 'maxDurationMs', 'maxTotalTokens']) expect(result.goal[key]).toBe(before.goal[key])
  expect(result.ignored_fields).toEqual(['max_goal_rounds', 'max_duration_ms', 'max_total_tokens'])
})

test('a goal cannot be created with placeholder limits that end it before its first round', async () => {
  // Captured from a live gpt-6-sol turn: every optional limit filled with its
  // schema minimum (1 ms, 1 token, 1 round). The goal blocked the instant it
  // was created, before a single agent ran.
  const h = harness()
  await expect(h.call('create_goal', { objective: 'bug-bounty campaign', max_duration_ms: 1, max_total_tokens: 1, max_goal_rounds: 1 }))
    .rejects.toThrow(/max_duration_ms: must be at least 60000 ms.*omit max_duration_ms or pass null/)
  await expect(h.call('create_goal', { objective: 'bug-bounty campaign', max_total_tokens: 1 })).rejects.toThrow(/max_total_tokens: must be at least 1000/)
  // null is the honest "no limit" for a model that must fill every field.
  const created = await h.call('create_goal', { objective: 'bug-bounty campaign', max_duration_ms: null, max_total_tokens: null, max_goal_rounds: null })
  const unlimited = await harness().call('create_goal', { objective: 'bug-bounty campaign' })
  for (const limit of ['maxDurationMs', 'maxTotalTokens', 'maxGoalRounds'] as const) expect(created.goal[limit]).toEqual(unlimited.goal[limit])
})

test('a status change with placeholder limits is not refused: only edit reads limits', async () => {
  // Captured from gpt-6-sol: a strict schema made it fill max_duration_ms on
  // "blocked" and "complete", both were refused as edit-only, and the model
  // had no call left that could close its goal.
  const h = harness()
  await h.call('create_goal', { objective: 'ship it' })
  const blocked = await h.call('update_goal', { ...await h.ref(), action: 'blocked', blocked_reason: 'CI is down', max_duration_ms: 60_000, max_total_tokens: 1_000, max_goal_rounds: 1 })
  expect(blocked.goal.phase).toBe('blocked')
  expect(blocked.ignored_fields).toEqual(['max_goal_rounds', 'max_duration_ms', 'max_total_tokens'])
  // The limits were not applied.
  const unlimited = await harness().call('create_goal', { objective: 'ship it' })
  for (const limit of ['maxDurationMs', 'maxTotalTokens', 'maxGoalRounds'] as const) expect(blocked.goal[limit]).toEqual(unlimited.goal[limit])
})
