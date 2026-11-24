// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { ToolDiff } from '../types.js'

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

export function approvalChoiceForOptionKind(kind: string): ApprovalChoice {
  switch (kind) {
    case 'reject_once':
    case 'reject_always':
    case 'reject':
    case 'deny':
      return 'deny'
    case 'allow_always':
      return 'always'
    case 'allow_session':
    case 'approve_for_session':
      return 'session'
    default:
      return 'once'
  }
}

export function approvalConsequenceForOptionKind(kind: string): string {
  switch (kind) {
    case 'allow_once':
      return 'asks again next time'
    case 'allow_always':
      return 'remembers this permission'
    case 'reject_once':
      return 'denies this request'
    case 'reject_always':
      return 'always denies this action'
    default:
      return ''
  }
}

export interface DiffPreviewLine {
  kind: 'add' | 'del' | 'header' | 'meta'
  text: string
}

/** Build a bounded line preview for one or more ACP file diffs. */
export function formatApprovalDiffPreview(diffs: readonly ToolDiff[], maxLines = 10): DiffPreviewLine[] {
  const all: DiffPreviewLine[] = []
  for (const entry of diffs) {
    const label = entry.path || '(file diff)'
    all.push({ kind: 'header', text: label })
    const oldLines = typeof entry.old_text === 'string' && entry.old_text.length > 0 ? entry.old_text.split('\n') : []
    const newLines = typeof entry.new_text === 'string' && entry.new_text.length > 0 ? entry.new_text.split('\n') : []
    for (const line of oldLines) {
      all.push({ kind: 'del', text: `- ${line}` })
    }
    for (const line of newLines) {
      all.push({ kind: 'add', text: `+ ${line}` })
    }
  }
  if (all.length <= maxLines) {
    return all
  }
  const visible = all.slice(0, Math.max(1, maxLines - 1))
  const remaining = all.length - visible.length
  return [...visible, { kind: 'meta', text: `… +${remaining} more diff line${remaining === 1 ? '' : 's'}` }]
}

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
