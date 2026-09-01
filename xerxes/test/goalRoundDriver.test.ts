// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'

import {
  createGoal,
  disarmGoal,
  getGoal,
  pauseGoal,
  resetGoalActivations,
} from '../src/runtime/goalDomain.js'
import { nextGoalRound } from '../src/runtime/goalRoundDriver.js'

afterEach(() => resetGoalActivations())

const session = 'session-1'

test('an active armed goal admits sequential attributed rounds', () => {
  const metadata: Record<string, unknown> = {}
  createGoal(metadata, session, { objective: 'make the loop cancel-safe', maxGoalRounds: 3 }, 1_000)

  const first = nextGoalRound(metadata, session, { now: 2_000 })
  expect(first).toHaveProperty('admitted')
  if (!('admitted' in first)) throw new Error('expected admission')
  expect(first.admitted.source).toMatchObject({ kind: 'goal', round: 1 })
  expect(first.admitted.prompt).toContain('Round 1 of 3')
  // The objective is quoted so tag-shaped text arrives as data.
  expect(first.admitted.prompt).toContain('"make the loop cancel-safe"')

  const second = nextGoalRound(metadata, session, { now: 3_000 })
  if (!('admitted' in second)) throw new Error('expected admission')
  expect(second.admitted.source.round).toBe(2)
})

test('the cap is a ceiling and refuses with a reason', () => {
  const metadata: Record<string, unknown> = {}
  createGoal(metadata, session, { objective: 'iterate', maxGoalRounds: 1 }, 1_000)

  expect(nextGoalRound(metadata, session, { now: 2_000 })).toHaveProperty('admitted')
  expect(nextGoalRound(metadata, session, { now: 3_000 })).toMatchObject({ refused: 'rounds-exhausted' })
})

test('a disarmed or paused goal never continues on its own', () => {
  const metadata: Record<string, unknown> = {}
  createGoal(metadata, session, { objective: 'iterate' }, 1_000)

  // This is the resumed-session case: the goal is still active, but this
  // process has no authority to act on it unattended.
  disarmGoal(session)
  expect(nextGoalRound(metadata, session, { now: 2_000 })).toMatchObject({ refused: 'disarmed' })

  resetGoalActivations()
  createGoal({} as Record<string, unknown>, 'other', { objective: 'x' }, 1_000)
  const paused: Record<string, unknown> = {}
  const goal = createGoal(paused, 'paused-session', { objective: 'hold' }, 1_000)
  pauseGoal(paused, 'paused-session', goal, 2_000)
  expect(nextGoalRound(paused, 'paused-session', { now: 3_000 })).toMatchObject({ refused: 'not-active' })
})

test('automatic work yields to a waiting human message', () => {
  const metadata: Record<string, unknown> = {}
  createGoal(metadata, session, { objective: 'iterate' }, 1_000)

  const refused = nextGoalRound(metadata, session, { humanWorkPending: true, now: 2_000 })
  expect(refused).toMatchObject({ refused: 'human-work-pending' })
  // Crucially it did not consume a round on the way to refusing.
  expect(getGoal(metadata, session)?.roundsStarted).toBe(0)
})

test('no goal means no continuation at all', () => {
  expect(nextGoalRound({}, session, { now: 1_000 })).toMatchObject({ refused: 'no-goal' })
})
