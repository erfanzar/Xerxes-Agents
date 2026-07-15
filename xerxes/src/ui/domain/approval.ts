// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type ApprovalChoice = 'always' | 'deny' | 'once' | 'session'

export interface ApprovalKey {
  downArrow?: boolean
  escape?: boolean
  return?: boolean
  upArrow?: boolean
}

export type ApprovalAction =
  { choice: ApprovalChoice; kind: 'choose' } | { delta: -1 | 1; kind: 'move' } | { kind: 'noop' }

const DEFAULT_APPROVAL_CHOICES: readonly ApprovalChoice[] = ['once', 'session', 'always', 'deny']

/** Pure keyboard dispatch for approval prompts. */
export function approvalAction(
  input: string,
  key: ApprovalKey,
  selected: number,
  choices: readonly ApprovalChoice[] = DEFAULT_APPROVAL_CHOICES
): ApprovalAction {
  if (key.escape) {
    return { choice: 'deny', kind: 'choose' }
  }

  const quickChoice = Number.parseInt(input, 10)

  if (quickChoice >= 1 && quickChoice <= choices.length) {
    return { choice: choices[quickChoice - 1]!, kind: 'choose' }
  }

  if (key.return && choices[selected]) {
    return { choice: choices[selected]!, kind: 'choose' }
  }

  if (key.upArrow && selected > 0) {
    return { delta: -1, kind: 'move' }
  }

  if (key.downArrow && selected < choices.length - 1) {
    return { delta: 1, kind: 'move' }
  }

  return { kind: 'noop' }
}
