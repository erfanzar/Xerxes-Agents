// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * When and why the model should learn, as one always-on policy block. Per-tool
 * guidance says how to call a tool; nothing said when learning belongs in the
 * work, so the tools went unused. Goals are used because they carry exactly
 * this kind of block (`goalPolicyPrompt`).
 */

/** Tools whose presence turns the learning policy on. */
export const LEARNING_POLICY_TOOLS: ReadonlySet<string> = new Set(['lesson_record', 'lesson_recall', 'session_search'])

export interface LearningPolicyInput {
  /** Tool names visible to the model this session. */
  readonly tools: ReadonlySet<string>
  /** Host reflection policy: whether long turns and goal endings are reviewed after the turn. */
  readonly reflection: 'goals' | 'all' | 'off'
}

/** The policy text for exactly the visible tools; empty when none of them is visible. */
export function learningPolicyPrompt(input: LearningPolicyInput): string {
  const lessons = input.tools.has('lesson_record')
  const recall = input.tools.has('lesson_recall')
  const search = input.tools.has('session_search')
  if (!lessons && !recall && !search) return ''
  const lines = ['[Learning policy]', 'You improve across sessions only through what you record now. Treat learning as part of finishing work, not an extra.']
  if (recall) {
    lines.push('Before: when a task touches an area that went wrong before (a flaky test, a build or deploy step, a tool quirk, a',
      'command that needs special flags), call lesson_recall with a few words for the situation first. Approved lessons for',
      'the current message arrive automatically in the turn context; recall also finds pending ones and other cues.')
  }
  if (lessons) {
    lines.push('After: right after a tool call or test proves something reusable, call lesson_record so the host cites that call as',
      'evidence. Good lessons: a root cause that took more than one attempt to find, a command or flag that works where the',
      'obvious one fails, an environment or test quirk, a correction the user gave you. Write the general rule, not the',
      'task answer. Never record an unverified guess. A person approves lessons before they are injected, so record',
      'freely when the evidence is real; one or two per substantive task is normal, zero after a hard-won fix is a miss.',
      'Lessons versus memory: a lesson is an evidence-backed rule about how to work in this workspace; memory files hold',
      'facts, decisions and user preferences. Do not write the same thing to both.')
  }
  if (search) {
    lines.push('Context: after a compaction summary, or whenever an exact value, path, error, command or decision from earlier in',
      'this session is missing, call session_search before guessing, re-deriving, or asking the user to repeat it.')
  }
  if (input.reflection !== 'off') {
    lines.push(input.reflection === 'all'
      ? 'Reflection: after a goal completes or is blocked, and after a long turn (25 tool calls or 20 minutes), the host'
      : 'Reflection: after a goal completes or is blocked, the host',
    'reviews the transcript and may propose one skill or agent improvement for a person to approve. You do not call it;',
    'it can only cite what the transcript shows, so state failures, fixes and the evidence for them plainly.')
  }
  return lines.join('\n')
}
