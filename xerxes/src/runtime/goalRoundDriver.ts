// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Same-session goal continuation.
 *
 * When a session goes idle holding an active, armed goal with capacity left,
 * this admits the next round and produces the prompt that opens it. Modelled on
 * DeepSeek Harness's goal-round driver
 * (github.com/deepseek-ai/deepseek-harness, MIT); written against Xerxes's own
 * turn loop, with no source reproduced.
 *
 * The important property is that a round is a real user turn, not an extra lap
 * inside one. The guard this replaces pushed a reminder into the running turn's
 * message array and continued, so an objective run was a single physical turn
 * that grew without bound, could not be interleaved with a human message, and
 * lost everything if it died. Each round now enters as its own attributed
 * message: the transcript stays readable, a human can steer between rounds, and
 * a crash costs one round rather than the whole objective.
 *
 * Idle is the only checkpoint. Provider errors and token limits are not goal
 * outcomes — the driver does not try to classify the preceding turn, it only
 * asks whether the durable goal still wants another round.
 */

import {
  admitGoalRound,
  blockGoal,
  getGoal,
  type GoalMessageSource,
  type GoalView,
} from './goalDomain.js'

/** One admitted round: the prompt to enqueue and the attribution it carries. */
export interface AdmittedGoalRound {
  readonly prompt: string
  /**
   * One line naming the round, for the transcript.
   *
   * The provider needs the full brief every round; a reader does not — a
   * transcript in which every other entry is the same forty-line block is one
   * nobody can follow, which defeats the point of rounds being real turns.
   */
  readonly displayText: string
  readonly source: GoalMessageSource
}

/** Why no round was admitted, for callers that surface it. */
export type GoalRoundRefusal =
  | 'no-goal'
  | 'not-active'
  | 'disarmed'
  | 'rounds-exhausted'
  | 'human-work-pending'

export type GoalRoundOutcome =
  | { readonly admitted: AdmittedGoalRound }
  | { readonly refused: GoalRoundRefusal; readonly goal?: GoalView }

export interface GoalRoundOptions {
  /**
   * Whether a human message is already waiting.
   *
   * Automatic work yields to it: a person who has typed something should not
   * wait behind a round the machine queued for itself.
   */
  readonly humanWorkPending?: boolean
  readonly now?: number
}

/**
 * Decide whether to open another goal round, and admit it if so.
 *
 * Admission mutates the durable log — the round number is reserved before the
 * prompt exists, so a crash between the two leaves a consumed round rather than
 * a round that runs twice.
 */
export function nextGoalRound(
  metadata: Record<string, unknown>,
  sessionId: string,
  options: GoalRoundOptions = {},
): GoalRoundOutcome {
  const goal = getGoal(metadata, sessionId)
  if (!goal) return { refused: 'no-goal' }
  if (goal.phase !== 'active') return { refused: 'not-active', goal }
  if (goal.activation !== 'armed') return { refused: 'disarmed', goal }
  if (options.humanWorkPending) return { refused: 'human-work-pending', goal }
  if (goal.roundsStarted >= goal.maxGoalRounds) {
    // Exhaustion is a durable outcome, not a silent stop. A goal that simply
    // stopped producing rounds is indistinguishable, from every surface a
    // person can see, from one that was never armed — so the run would end
    // with the goal still reading "active" and nothing saying why nothing is
    // happening. Recording it as blocked names the cause and points at the
    // fix, which is to raise the cap and resume.
    return {
      refused: 'rounds-exhausted',
      goal: blockGoal(
        metadata,
        sessionId,
        { id: goal.id, revision: goal.revision },
        {
          code: 'round-limit',
          message: `Goal reached its configured limit of ${goal.maxGoalRounds} rounds.`,
        },
        options.now ?? Date.now(),
      ),
    }
  }

  const source = admitGoalRound(metadata, sessionId, options.now ?? Date.now())
  if (!source) return { refused: 'not-active', goal }
  return {
    admitted: {
      prompt: goalRoundPrompt(goal, source.round),
      displayText: goalRoundLabel(goal, source.round),
      source,
    },
  }
}

/** The one-line transcript label for an admitted round. */
export function goalRoundLabel(goal: GoalView, round: number): string {
  return `Goal round ${round}/${goal.maxGoalRounds} — ${goal.objective}`
}

/**
 * The retained prompt one admitted round enters as.
 *
 * The objective is JSON-quoted so multiline or tag-shaped text arrives as data
 * rather than as instructions that could restructure the prompt around it.
 */
export function goalRoundPrompt(goal: GoalView, round: number): string {
  return [
    '<goal_round>',
    `Objective: ${JSON.stringify(goal.objective)}`,
    `Round ${round} of ${goal.maxGoalRounds}.`,
    '',
    'The current workspace, this session\'s tool results, and the durable goal state are authoritative —',
    'not your recollection of them. Continue the objective.',
    '',
    'When the objective is met, run the check that proves it and then call update_goal with action',
    'complete. If work remains, keep going and leave the goal active; do not summarise and stop. Call',
    'update_goal with action blocked only once the same concrete condition has genuinely persisted across',
    'rounds, and name that condition.',
    '</goal_round>',
  ].join('\n')
}
