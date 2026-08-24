// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Compact subagent cards for the transcript trail (redesign mockups
// 02/03 element ⑥): one ≤3-line, stateless card per archived agent.
//
//   ● researcher — Survey Token-Parse Patterns     ← status dot + name + task
//   └ reading src/session/tokenCache.ts            ← violet latest activity (running)
//   8.2k tok · 3 tools                             ← dim budget, when known
//
// The model here is the single source of truth for BOTH the painted rows
// (opentui/messageLine.tsx) and the virtualization height correction
// (app/useMainApp.ts): `subagentCardRows` counts exactly the lines the
// renderer will paint, so the estimator can never drift from the paint.

import type { Theme } from '../theme.js'
import type { SubagentProgress, SubagentStatus } from '../types.js'

import { stateSkin } from '../domain/nocturne.js'

import { AGENT_GROUP_STATE, agentGroup } from './agentGroups.js'
import { subagentFailed } from './agentRetry.js'
import { fmtTokens } from './subagentElapsed.js'

/** Same card-title ceiling as the agents panel, so both speak alike. */
export const SUBAGENT_CARD_TITLE_MAX = 24

const compactLine = (value: string, max: number): string => {
  const line = value.replace(/\s+/g, ' ').trim()

  return line.length > max ? `${line.slice(0, Math.max(1, max - 1)).trimEnd()}…` : line
}

const titleCase = (value: string): string => value.replace(/\b[a-z]/g, letter => letter.toUpperCase())

/**
 * Concise human TASK label for an agent — what the en-dash shows. Mirrors the
 * agents panel's normalization vocabulary (strip runtime suffixes, split
 * kebab/snake, title case) but sources from the task (explicit title, then
 * goal) rather than the identity: the transcript headline already carries the
 * name, and the mockup row reads "● researcher#2 — <task>". Reimplemented
 * here rather than imported so this pure lib does not pull the panel's
 * component module into the height-estimator import graph.
 */
export function subagentCardSummary(item: SubagentProgress): string {
  const source = item.title?.trim() || item.goal?.trim() || item.agentType?.trim() || item.model?.trim()
  const withoutRuntimeSuffix = (source ?? '').split('#', 1)[0] ?? ''
  const normalized = titleCase(
    withoutRuntimeSuffix
      .replace(/^\/?root\//i, '')
      .replace(/[-_]+/g, ' ')
      .trim()
  )

  return compactLine(normalized || 'Agent task', Math.max(8, SUBAGENT_CARD_TITLE_MAX))
}

/** Stable short name: the spawn name when present, else the role type. */
export function subagentDisplayName(item: SubagentProgress): string {
  return item.name?.trim() || item.agentType?.trim() || ''
}

/**
 * The dot colour for a status — one lookup through the Nocturne state table,
 * so a card in the transcript, a card in the rail and a row in the F6 overlay
 * cannot disagree about what colour "working" is.
 *
 * It reads `ds` rather than `color.toolName` deliberately: `toolName` is the
 * ramp step tool VERBS sit on, not a state, and borrowing it here is what
 * turned every running agent's dot grey the moment the ramp was assigned by
 * role.
 */
export function subagentCardAccent(status: SubagentStatus, t: Theme): string {
  if (subagentFailed(status)) {
    return t.ds.failed
  }

  return stateSkin(AGENT_GROUP_STATE[agentGroup(status)], t.ds).dot
}

export interface SubagentCardModel {
  /** Headline label after the status dot ('' renders nothing). */
  headline: string
  /** Task summary after the en-dash; '' when it would repeat the headline. */
  summary: string
  /** Latest activity while still working; '' otherwise. Rendered violet. */
  activity: string
  /** Result sentence once settled; '' while no outcome was reported. */
  result: string
  /** `8.2k tok · 3 tools` budget; '' when nothing is known yet. */
  budget: string
}

const TERMINAL_STATUSES = new Set<SubagentStatus>(['completed', 'error', 'failed', 'interrupted', 'timeout'])

/** Tool tally using the widest of the counters the wire provides. */
export const subagentToolCount = (item: SubagentProgress): number =>
  Math.max(item.toolCount, item.toolCalls?.length ?? 0, item.tools.length, item.outputTail?.length ?? 0)

/** Build every line the transcript card will paint for one agent. */
export function subagentCardModel(item: SubagentProgress): SubagentCardModel {
  const displayName = subagentDisplayName(item)
  const summary = subagentCardSummary(item)
  // The dash segment is the TASK; when there is no separate name the derived
  // title already says it and repeating it twice would be noise.
  const headline = displayName || summary
  const showSummary = Boolean(displayName) && summary.toLowerCase() !== displayName.toLowerCase()
  const tokens = (item.inputTokens ?? 0) + (item.outputTokens ?? 0)
  const tools = subagentToolCount(item)

  let activity = ''
  let result = ''

  if (TERMINAL_STATUSES.has(item.status)) {
    const settled = item.summary?.trim()
    const note = item.notes.at(-1)?.trim()

    if (settled) {
      result = `Result: ${compactLine(settled, 140)}`
    } else if (note) {
      result = compactLine(note, 140)
    }
  } else {
    const latest =
      item.notes.at(-1)?.trim() ||
      item.tools.at(-1)?.trim() ||
      item.thinking.at(-1)?.trim() ||
      item.outputTail?.at(-1)?.preview.trim() ||
      item.summary?.trim() ||
      (item.status === 'queued' ? 'Waiting to start' : 'Working')

    activity = compactLine(latest, 120)
  }

  return {
    activity,
    budget: tokens > 0 || tools > 0 ? `${fmtTokens(tokens)} tok · ${tools} tool${tools === 1 ? '' : 's'}` : '',
    headline,
    result,
    summary: showSummary ? summary : ''
  }
}

/** Rows ONE agent's card paints: headline + optional status/budget lines. */
export function subagentCardRowCount(item: SubagentProgress): number {
  const model = subagentCardModel(item)

  return 1 + (model.activity ? 1 : 0) + (model.result ? 1 : 0) + (model.budget ? 1 : 0)
}

/**
 * Rows ALL cards on one message paint. This is what the height estimator adds
 * on top of `estimatedMsgHeight` (which deliberately knows nothing about
 * subagents — see virtualHeights.ts) so cold-scroll offsets match the paint.
 */
export function subagentCardRows(items: readonly SubagentProgress[] | undefined): number {
  if (!items?.length) {
    return 0
  }

  let rows = 0

  for (const item of items) {
    rows += subagentCardRowCount(item)
  }

  return rows
}
