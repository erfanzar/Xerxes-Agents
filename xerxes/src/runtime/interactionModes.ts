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
  switch (normalizeInteractionMode(mode)) {
    case 'plan':
      return '[Mode control]\n'
        + 'You are in plan mode. Produce a plan only.'
        + (canSwitch
          ? ' Do not use SetInteractionModeTool to leave this mode; only the user or session host may do that.'
          : '')
    case 'researcher':
      return '[Mode control]\n'
        + 'You are in researcher mode. Gather evidence and answer with citations.'
        + (canSwitch
          ? ' Do not use SetInteractionModeTool to leave this mode; only the user or session host may do that.'
          : '')
    case 'objective':
      return '[Mode control]\n'
        + "You are in objective mode. Treat the user's requested outcome as a hard objective with acceptance "
        + 'criteria. Maintain a compact task ledger, choose one hypothesis at a time, edit/build/test/benchmark, '
        + 'compare results to the acceptance criteria, keep or revert based on evidence, and continue. Do not '
        + 'final-answer with a narrative status while the acceptance criteria are unmet. Leave objective mode '
        + 'only after verification proves the objective is met, the user changes modes, or you are concretely '
        + 'blocked and can name the blocker plus the exact evidence.'
        + (canSwitch
          ? ' Do not use SetInteractionModeTool to leave this mode; report verified completion or a concrete blocker and let the user or session host switch modes.'
          : '')
    case 'code':
      return canSwitch
        ? '[Mode control]\n'
          + 'Use code mode for normal implementation. To schedule a different policy for the next user turn, call '
          + 'SetInteractionModeTool(mode="researcher") or SetInteractionModeTool(mode="plan"). If the user asks for '
          + 'a measurable outcome that requires repeated attempts until tests, benchmarks, or checks pass, call '
          + 'SetInteractionModeTool(mode="objective").'
        : ''
  }
}

function modeKey(mode: unknown): string {
  return String(mode || 'code').trim().toLowerCase()
}
