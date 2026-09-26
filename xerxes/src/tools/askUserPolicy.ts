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
  + 'an approved plan, or "should I continue?". Ask one specific question that names the options and leads '
  + 'with your recommendation.'
