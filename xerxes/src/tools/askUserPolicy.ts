// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * When to ask the user, shared by every tool that pauses for an answer
 * (AskUserQuestionTool in the daemon and workflow surfaces, clarify, and the
 * operator ask_user) so the rule reads the same wherever the tool appears
 * and costs nothing when it is hidden.
 */
export const ASK_USER_POLICY =
  'Ask only when a decision is the user\'s and no file, tool, or earlier message settles it: ambiguous '
  + 'requirements with different outcomes, a preference between valid approaches, or approval for something '
  + 'irreversible. Otherwise decide, and state the assumption. Never ask what a tool can find out, to re-confirm '
  + 'an approved plan, or "should I continue?". Ask one specific question. When there are distinct choices, put '
  + 'each in `options` (the user clicks one, or types their own answer instead) and lead with your '
  + 'recommendation, marked "(Recommended)"; keep the question itself to the context needed to decide, not a '
  + 'second copy of the choices.'

/** The `options` parameter, described the same on every surface that asks. */
export const ASK_USER_OPTIONS_SCHEMA = {
  type: 'array',
  items: { type: 'string' },
  description: 'The choices, 2 to 5, each a short self-contained answer the user can click (for example "Run the '
    + 'script, then the tests (Recommended)"). Recommended first. The user can always type something else, so '
    + 'do not add an "Other" choice. Omit for an open question.',
} as const
