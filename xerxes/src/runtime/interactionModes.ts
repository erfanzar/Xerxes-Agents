// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Canonical session interaction modes shared by runtime, daemon, and tools. */
export const INTERACTION_MODES = Object.freeze(['code', 'researcher', 'plan', 'objective'] as const)

export type InteractionMode = (typeof INTERACTION_MODES)[number]
export type InteractionModeAgentName = 'coder' | 'objective' | 'planner' | 'researcher'

/** Accepted user and model spellings mapped to their canonical interaction mode. */
export const MODE_ALIASES: Readonly<Record<string, InteractionMode>> = Object.freeze({
  '': 'code',
  coding: 'code',
  coder: 'code',
  code: 'code',
  research: 'researcher',
  researcher: 'researcher',
  plan: 'plan',
  planner: 'plan',
  goal: 'objective',
  goals: 'objective',
  'goal-runner': 'objective',
  goal_runner: 'objective',
  objective: 'objective',
  objectives: 'objective',
  iterate: 'objective',
  autonomous: 'objective',
})

/** Return an alias target, leaving unknown labels unresolved for strict callers. */
export function resolveInteractionMode(mode: unknown): InteractionMode | undefined {
  return MODE_ALIASES[modeKey(mode)]
}

/** Coerce a user or model mode label to the safe canonical mode. */
export function normalizeInteractionMode(mode: unknown, planMode = false): InteractionMode {
  if (planMode) return 'plan'
  return resolveInteractionMode(mode) ?? 'code'
}

/** Map an interaction mode to the matching built-in agent definition name. */
export function agentNameForMode(mode: unknown): InteractionModeAgentName {
  switch (normalizeInteractionMode(mode)) {
    case 'plan': return 'planner'
    case 'researcher': return 'researcher'
    case 'objective': return 'objective'
    case 'code': return 'coder'
  }
}

/** Return model-facing mode guidance without advertising an unavailable switch tool. */
export function modeSwitchHint(mode: unknown, canSwitch = true): string {
  const transitionHint = canSwitch
    ? ' As the main agent, you may use SetInteractionModeTool to schedule code, researcher, plan, or objective mode for the next user turn. Finish this turn under the current mode and its current tool policy.'
    : ''
  switch (normalizeInteractionMode(mode)) {
    case 'plan':
      return '[Mode control]\n'
        + 'You are in plan mode. Produce a plan only.'
        + transitionHint
    case 'researcher':
      return '[Mode control]\n'
        + 'You are in researcher mode. Gather evidence and answer with citations.'
        + transitionHint
    case 'objective':
      return '[Mode control]\n'
        + "You are in objective mode. Treat the user's requested outcome as a hard objective with acceptance "
        + 'criteria. Maintain a compact task ledger, choose one hypothesis at a time, edit/build/test/benchmark, '
        + 'compare results to the acceptance criteria, keep or revert based on evidence, and continue. Do not '
        + 'final-answer with a narrative status while the acceptance criteria are unmet. Leave objective mode '
        + 'only after verification proves the objective is met, the user changes modes, or you are concretely '
        + 'blocked and can name the blocker plus the exact evidence.'
        + (canSwitch
          ? ' When a leave condition is met, use SetInteractionModeTool to schedule code mode or another appropriate mode for the next user turn. The user or session host may also change modes.'
          : '')
    case 'code':
      return canSwitch
        ? '[Mode control]\n'
          + 'Use code mode for normal implementation.'
          + transitionHint
        : ''
  }
}

function modeKey(mode: unknown): string {
  return String(mode || 'code').trim().toLowerCase()
}
