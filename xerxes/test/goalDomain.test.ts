// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'

import {
  DEFAULT_MAX_GOAL_ROUNDS,
  GoalError,
  admitGoalRound,
  blockGoal,
  clearGoal,
  completeGoal,
  createGoal,
  disarmGoal,
  editGoal,
  foldGoalChanges,
  getGoal,
  pauseGoal,
  readGoalChanges,
  resetGoalActivations,
  resumeGoal,
} from '../src/runtime/goalDomain.js'

afterEach(() => resetGoalActivations())

const session = 'session-1'
const fresh = () => ({}) as Record<string, unknown>

test('a created goal is active, armed, and at revision 1', () => {
  const metadata = fresh()
  const goal = createGoal(metadata, session, { objective: 'make the loop cancel-safe' }, 1_000)

  expect(goal).toMatchObject({
    objective: 'make the loop cancel-safe',
    phase: 'active',
    activation: 'armed',
    revision: 1,
    roundsStarted: 0,
    maxGoalRounds: DEFAULT_MAX_GOAL_ROUNDS,
  })
  expect(getGoal(metadata, session)).toMatchObject({ id: goal.id, revision: 1 })
})

test('mutations are compare-and-set on the exact revision', () => {
  const metadata = fresh()
  const goal = createGoal(metadata, session, { objective: 'ship it' }, 1_000)

  // A writer that observed older state is refused rather than clobbering.
  expect(() => pauseGoal(metadata, session, { id: goal.id, revision: 99 }, 2_000))
    .toThrow('stale revision 99')
  expect(() => pauseGoal(metadata, session, { id: 'goal_other', revision: 1 }, 2_000))
    .toThrow('is not the current goal')

  const paused = pauseGoal(metadata, session, goal, 2_000)
  expect(paused).toMatchObject({ phase: 'paused', revision: 2, activation: 'disarmed' })
})

test('phase transitions are constrained, and complete/block disarm', () => {
  const metadata = fresh()
  const goal = createGoal(metadata, session, { objective: 'ship it' }, 1_000)

  // Blocking is only reachable from active — a paused goal is not stuck, it is held.
  const paused = pauseGoal(metadata, session, goal, 2_000)
  expect(() => blockGoal(metadata, session, paused, { code: 'x', message: 'y' }, 3_000))
    .toThrow('cannot block')

  const resumed = resumeGoal(metadata, session, paused, 3_000)
  expect(resumed).toMatchObject({ phase: 'active', activation: 'armed' })

  const blocked = blockGoal(metadata, session, resumed, { code: 'missing-cred', message: 'no API key' }, 4_000)
  expect(blocked).toMatchObject({
    phase: 'blocked',
    activation: 'disarmed',
    blockedReason: { code: 'missing-cred', message: 'no API key' },
  })

  const done = completeGoal(metadata, session, blocked, 5_000)
  expect(done).toMatchObject({ phase: 'complete', activation: 'disarmed' })
  // A phase that is no longer blocked must not keep carrying its reason.
  expect(done.blockedReason).toBeUndefined()
})

test('a second create is refused unless the current goal is complete', () => {
  const metadata = fresh()
  const goal = createGoal(metadata, session, { objective: 'first' }, 1_000)
  expect(() => createGoal(metadata, session, { objective: 'second' }, 2_000))
    .toThrow('already exists with phase "active"')

  const done = completeGoal(metadata, session, goal, 3_000)
  expect(createGoal(metadata, session, { objective: 'second' }, 4_000)).toMatchObject({
    objective: 'second',
    phase: 'active',
  })
  expect(done.phase).toBe('complete')
})

test('activation is process-local and never survives a reload', () => {
  const metadata = fresh()
  const goal = createGoal(metadata, session, { objective: 'keep going' }, 1_000)
  expect(getGoal(metadata, session)?.activation).toBe('armed')

  // A resumed or forked session must not silently continue autonomous work.
  // The durable phase is untouched; only this process loses authority.
  disarmGoal(session)
  expect(getGoal(metadata, session)).toMatchObject({ phase: 'active', activation: 'disarmed' })

  // A fresh process has no activation at all, and reads as disarmed.
  resetGoalActivations()
  expect(getGoal(metadata, session)).toMatchObject({ phase: 'active', activation: 'disarmed' })

  // Only an explicit resume rearms it.
  const current = getGoal(metadata, session)!
  expect(resumeGoal(metadata, session, current, 2_000).activation).toBe('armed')
  expect(goal.id).toBe(current.id)
})

test('rounds are admitted only while active, armed, and under the cap', () => {
  const metadata = fresh()
  createGoal(metadata, session, { objective: 'iterate', maxGoalRounds: 2 }, 1_000)

  const first = admitGoalRound(metadata, session, 2_000)
  expect(first).toMatchObject({ kind: 'goal', round: 1 })
  const second = admitGoalRound(metadata, session, 3_000)
  expect(second).toMatchObject({ round: 2 })

  // The cap is a real ceiling, not a suggestion.
  expect(admitGoalRound(metadata, session, 4_000)).toBeUndefined()
  expect(getGoal(metadata, session)?.roundsStarted).toBe(2)

  // And an exhausted goal cannot be resumed without raising the cap.
  const current = getGoal(metadata, session)!
  disarmGoal(session)
  expect(() => resumeGoal(metadata, session, current, 5_000)).toThrow('exhausted 2 goal rounds')
})

test('a disarmed or non-active goal admits no rounds', () => {
  const metadata = fresh()
  const goal = createGoal(metadata, session, { objective: 'iterate' }, 1_000)

  disarmGoal(session)
  expect(admitGoalRound(metadata, session, 2_000)).toBeUndefined()

  resumeGoal(metadata, session, getGoal(metadata, session)!, 3_000)
  expect(admitGoalRound(metadata, session, 4_000)).toMatchObject({ round: 1 })

  pauseGoal(metadata, session, getGoal(metadata, session)!, 5_000)
  expect(admitGoalRound(metadata, session, 6_000)).toBeUndefined()
  expect(goal.phase).toBe('active')
})

test('state is a strict fold over whole-value changes, and clear leaves a tombstone', () => {
  const metadata = fresh()
  const goal = createGoal(metadata, session, { objective: 'ship' }, 1_000)
  admitGoalRound(metadata, session, 2_000)
  editGoal(metadata, session, getGoal(metadata, session)!, { objective: 'ship carefully' }, 3_000)

  const changes = readGoalChanges(metadata)
  expect(changes.map(c => c.operation)).toEqual(['create', 'round', 'edit'])
  // Replaying the log from scratch reproduces exactly the live view.
  expect(foldGoalChanges(changes).goal).toMatchObject({ objective: 'ship carefully' })

  const tombstone = clearGoal(metadata, session, getGoal(metadata, session)!, 4_000)
  expect(tombstone.id).toBe(goal.id)
  expect(getGoal(metadata, session)).toBeUndefined()
  // History is retained rather than truncated, so the clear is auditable.
  expect(readGoalChanges(metadata).at(-1)).toMatchObject({ operation: 'clear' })
})

test('invalid input is refused with stable codes', () => {
  const metadata = fresh()
  const attempt = (fn: () => unknown): string => {
    try { fn(); return 'no-throw' } catch (error) { return (error as GoalError).code }
  }

  expect(attempt(() => createGoal(metadata, session, { objective: '   ' }, 1_000))).toBe('GOAL_INVALID_OBJECTIVE')
  expect(attempt(() => createGoal(metadata, session, { objective: 'x', maxGoalRounds: 0 }, 1_000)))
    .toBe('GOAL_INVALID_MAX_ROUNDS')

  const goal = createGoal(metadata, session, { objective: 'x' }, 1_000)
  expect(attempt(() => editGoal(metadata, session, goal, {}, 2_000))).toBe('GOAL_INVALID_EDIT')
  expect(attempt(() => blockGoal(metadata, session, goal, { code: 'c', message: '  ' }, 2_000)))
    .toBe('GOAL_INVALID_BLOCK_REASON')
})

test('the fold refuses a change log this module could not have written', () => {
  const metadata: Record<string, unknown> = {}
  const session = 'strict'
  createGoal(metadata, session, { objective: 'ship', maxGoalRounds: 4 }, 1_000)
  admitGoalRound(metadata, session, 2_000)
  const valid = readGoalChanges(metadata)

  // Every case below describes a log that is internally inconsistent, which is
  // what tampering, a partial write, or a merge of two histories looks like.
  const corruptions: ReadonlyArray<{ readonly label: string; readonly changes: unknown[] }> = [
    {
      label: 'a revision gap',
      changes: [valid[0], { ...valid[1], goal: { ...(valid[1] as any).goal, revision: 9 } }],
    },
    {
      label: 'a round counter that skips',
      changes: [valid[0], { ...valid[1], roundsStarted: 3 }],
    },
    {
      label: 'a round past the declared cap',
      changes: [
        { ...valid[0], goal: { ...(valid[0] as any).goal, maxGoalRounds: 1 } },
        { ...valid[1], goal: { ...(valid[1] as any).goal, maxGoalRounds: 1 } },
        {
          ...valid[1],
          goal: { ...(valid[1] as any).goal, maxGoalRounds: 1, revision: 3 },
          roundsStarted: 2,
        },
      ],
    },
    {
      label: 'a second create over live work',
      changes: [valid[0], valid[0]],
    },
    {
      label: 'a mutation belonging to a different goal',
      changes: [valid[0], { ...valid[1], goal: { ...(valid[1] as any).goal, id: 'goal_other' } }],
    },
  ]

  for (const corruption of corruptions) {
    expect(() => foldGoalChanges(corruption.changes as never), corruption.label).toThrow(GoalError)
  }
  // The honest log still folds, and carries no process-local activation into
  // the durable record.
  expect(foldGoalChanges(valid).roundsStarted).toBe(1)
  for (const change of valid) {
    expect(Object.keys((change as any).goal ?? {})).not.toContain('activation')
  }
})

test('a long-running goal compacts its log instead of losing its head', () => {
  const metadata: Record<string, unknown> = {}
  const session = 'compacting'
  createGoal(metadata, session, { objective: 'a very long haul', maxGoalRounds: 5_000 }, 1_000)
  for (let round = 0; round < 400; round += 1) {
    admitGoalRound(metadata, session, 2_000 + round)
  }

  const changes = readGoalChanges(metadata)
  // Bounded: session metadata cannot grow without limit.
  expect(changes.length).toBeLessThanOrEqual(256)
  // Intact: the surviving log still folds strictly, and to the true state —
  // truncating the head would have produced a log whose first entry is a round
  // with no create behind it, indistinguishable from tampering.
  const folded = foldGoalChanges(changes)
  expect(folded.roundsStarted).toBe(400)
  expect(folded.goal?.objective).toBe('a very long haul')
  expect(getGoal(metadata, session)?.roundsStarted).toBe(400)
})
