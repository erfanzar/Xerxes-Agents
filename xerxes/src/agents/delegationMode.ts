// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * How eagerly a conversation delegates to agents and workflows: the user's
 * choice from the composer chip or `/delegate`, carried to the model as a
 * short instruction each turn. Behaviour stays prompt-driven — nothing here
 * forces or forbids a tool; the model is told the user's preference.
 */

export const DELEGATION_MODES = ['off', 'auto', 'eager'] as const
export type DelegationMode = typeof DELEGATION_MODES[number]

/** Session metadata key; absent means the default, auto. */
export const DELEGATION_MODE_METADATA_KEY = 'delegation_mode'

export function isDelegationMode(value: unknown): value is DelegationMode {
  return typeof value === 'string' && (DELEGATION_MODES as readonly string[]).includes(value)
}

export function delegationModeOf(metadata: Readonly<Record<string, unknown>>): DelegationMode {
  const value = metadata[DELEGATION_MODE_METADATA_KEY]
  return isDelegationMode(value) ? value : 'auto'
}

/** The per-turn instruction for a mode. */
export function delegationModePrompt(mode: DelegationMode): string {
  switch (mode) {
    case 'off':
      return [
        '# Delegation: off',
        'The user turned agents off for this conversation. Do the work yourself: do not start subagents or workflows unless the user explicitly asks for them in a message.',
      ].join('\n')
    case 'eager':
      return [
        '# Delegation: eager',
        'The user asked for maximum delegation. Whenever work can split at all — reading several files, checking more than one hypothesis, reviewing, testing, verifying your own conclusions — hand the parts to agents, preferably in one Workflow with a find stage and an independent verify stage. Do not ask about budget first; the user already chose thoroughness. Keep only genuinely sequential, tightly coupled steps for yourself.',
      ].join('\n')
    case 'auto':
      return [
        '# Delegation: auto',
        'Delegate when the work splits (see the Workflow guidance); work alone when it does not. Before a workflow that would start more than about 10 agents — or one you cannot bound — ask the user once with AskUserQuestionTool, unless they already said: "Budget" (fast, cheap models, fewer agents, no verify stage, a token_budget) or "Thorough" (strong models, a verify stage). Follow their answer for the rest of the task.',
      ].join('\n')
  }
}
