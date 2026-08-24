// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Nocturne's vocabulary, in one place.
//
// The design canvas closes with a claim the code has to make true: "four row
// patterns compose every screen — a new screen should be assemblable from
// leader rows, captions, cards and a footer without inventing a fifth shape."
// This module owns the pure half of that (glyph table, leader math, state →
// voice mapping, the order things are given up in as the terminal narrows);
// `opentui/nocturne.tsx` owns the rendered half.
//
// Before this existed the dotted-leader run was implemented three times, with
// three different safety margins, in three files. That is exactly the drift a
// design system is supposed to stop.

/**
 * One glyph, one job.
 *
 * Shape carries the *kind* of thing; colour carries its state. A screen that
 * needs a mark not in this table is a screen that has invented a fifth row
 * pattern, so the table is the review question, not a convenience.
 */
export const GLYPH = {
  /** State. Always coloured, never bare. */
  state: '●',
  /** Brand, and the cwd marker in the statusbar. */
  brand: '✦',
  /** Expandable, collapsed. */
  collapsed: '▸',
  /** Expandable, open. */
  expanded: '▾',
  /** A tool call. */
  tool: '⏺',
  /** You are typing. */
  prompt: '❯',
  /** The ledger line that closes a turn. */
  ledger: '└',
  /** Soft-wrap continuation, so a wrap is never read as a second line. */
  wrap: '↳',
  /** Separator between facts. */
  separator: '·',
  /** Statusbar section break — quieter than a separator. */
  sectionBreak: '│',
  /** Mode indicator. */
  mode: '◆'
} as const

/**
 * The four states a row can be in, product-wide.
 *
 * Screens name them differently — agents say "ready to review" where
 * terminals say "succeeded", terminals say "idle interactive" where agents
 * say "needs input" — but a row is only ever one of these four, and the
 * colour it gets is decided here rather than per screen.
 */
export type NocturneState = 'done' | 'failed' | 'needsInput' | 'working'

/**
 * Group order: by what you have to do, not by lifecycle.
 *
 * A blocked agent at the bottom of a status-sorted list is a stalled agent,
 * so `needsInput` leads and `failed` — which has already spent its money and
 * does not get to spend your attention too — trails.
 */
export const ATTENTION_ORDER: readonly NocturneState[] = ['needsInput', 'working', 'done', 'failed']

export const attentionRank = (state: NocturneState): number => {
  const rank = ATTENTION_ORDER.indexOf(state)

  return rank < 0 ? ATTENTION_ORDER.length : rank
}

/** Sort helper for grouped lists; ties keep their incoming order. */
export const byAttention = <T>(state: (item: T) => NocturneState) =>
  (a: T, b: T): number => attentionRank(state(a)) - attentionRank(state(b))

export interface StateSkin {
  /** The `●` colour, and the 1-cell rail on a selected card. */
  dot: string
  /** Tinted card ground. */
  ground: string
  /** Card edge. */
  border: string
  /**
   * Prose that has to carry the state itself — the question an agent is
   * blocked on, a failure's last line. A softened step of the voice colour,
   * because the voice colours are tuned for marks and chrome, not paragraphs.
   */
  text: string
}

/** The tokens a state contributes, resolved against a ground. */
export function stateSkin(
  state: NocturneState,
  ds: {
    done: string
    doneCardBg: string
    doneCardBorder: string
    failed: string
    failedCardBg: string
    failedCardBorder: string
    failedText: string
    hairline: string
    needsInput: string
    needsInputCardBg: string
    needsInputCardBorder: string
    needsInputText: string
    prose: string
    working: string
    workingCardBg: string
  }
): StateSkin {
  switch (state) {
    case 'needsInput':
      return {
        dot: ds.needsInput,
        ground: ds.needsInputCardBg,
        border: ds.needsInputCardBorder,
        text: ds.needsInputText
      }
    case 'done':
      return { dot: ds.done, ground: ds.doneCardBg, border: ds.doneCardBorder, text: ds.prose }
    case 'failed':
      return { dot: ds.failed, ground: ds.failedCardBg, border: ds.failedCardBorder, text: ds.failedText }
    default:
      return { dot: ds.working, ground: ds.workingCardBg, border: ds.hairline, text: ds.prose }
  }
}

/**
 * The safety margin every dotted leader stops short by.
 *
 * The leading separator space plus a four-column cushion absorbs
 * ambiguous-width glyphs (`→` measures one cell but some terminals ink two)
 * and the scrollbox's reserved scrollbar column, so the right-aligned
 * quantity is never pushed off the line. `truncate-end` remains the backstop.
 */
export const LEADER_SAFETY = 5

/**
 * The run of `·` between a row's label and its right-aligned quantity.
 *
 * Returns '' rather than a short run when there is no room: one or two stray
 * dots read as a rendering fault, and the row is perfectly legible without
 * them. Every screen's leader rows call this, so durations and token counts
 * stack into one readable column across the whole product.
 */
export function leaderRun(available: number, leftWidth: number, rightWidth: number, safety = LEADER_SAFETY): string {
  const count = Math.floor(available) - leftWidth - rightWidth - safety

  return count >= 2 ? ` ${'·'.repeat(count)}` : ''
}

/**
 * What a screen gives up as the terminal narrows, and in what order.
 *
 * Screen 09 fixes the order: secondary counts, then the source labels that
 * trail a group caption, then card goal lines, then side panels. Titles and
 * state dots never go, because they are the answer to "what is happening" —
 * and the composer and the approval card never degrade at all, so neither
 * appears here.
 *
 * Six one-line agents beat three legible ones when you are scanning for the
 * amber dot, so cards collapse to their single-line form rather than
 * reflowing into taller cards.
 */
export interface NocturneDensity {
  /** The second half of a card's budget (`4s · 1.5k tok` → `4s`). */
  cardBudget: boolean
  /** The source label trailing a caption (`built in`, `cleared on quit`). */
  captionSource: boolean
  /** The one-clause goal beside or under a card title. */
  goals: boolean
  /** Side panels: the diff file index, the agent inspector. */
  sidePanels: boolean
}

/**
 * Everything on. For surfaces whose narrowness is their design rather than a
 * degradation — the agents rail is 38-48 columns by construction and is a
 * summary at every terminal size, so measuring it against screen-scale
 * thresholds would leave it permanently in maximum-sacrifice mode.
 */
export const FULL_DENSITY: NocturneDensity = {
  cardBudget: true,
  captionSource: true,
  goals: true,
  sidePanels: true
}

/**
 * The agents rail: a summary column, 38-48 columns wide by construction.
 *
 * It keeps the budget — the numbers are the reason to glance at it — and
 * drops the goal, because a one-clause goal on a 40-column row does not
 * shorten the title, it eats it: `● Structur...p the repo`.
 */
export const RAIL_DENSITY: NocturneDensity = {
  cardBudget: true,
  captionSource: false,
  goals: false,
  sidePanels: false
}

export const densityFor = (columns: number): NocturneDensity => {
  const cols = Math.floor(columns)

  return {
    sidePanels: cols >= 100,
    goals: cols >= 88,
    captionSource: cols >= 76,
    cardBudget: cols >= 64
  }
}

/**
 * Wrap a command or a line of code, marking every continuation with `↳`.
 *
 * A soft wrap and a second command look identical in a monospace column, and
 * on an approval card that difference is the difference between running one
 * thing and running two. The glyph costs two columns and removes the
 * ambiguity entirely, so the wrap is explicit rather than inferred.
 *
 * Breaks on whitespace where it can and mid-token only when a single token is
 * longer than the column — a command is not prose and must never lose a
 * character to prettier wrapping.
 */
export function wrapWithContinuation(text: string, columns: number): string[] {
  const width = Math.floor(columns)

  if (width < 4 || !text) {
    return [text]
  }

  const lines: string[] = []
  // Continuations pay for the `↳ ` prefix out of their own width.
  let budget = width
  let current = ''

  const push = () => {
    lines.push(lines.length === 0 ? current : `${GLYPH.wrap} ${current}`)
    current = ''
    budget = width - 2
  }

  for (const token of text.split(/(\s+)/)) {
    if (!token) {
      continue
    }

    if (current.length + token.length <= budget) {
      current += token
      continue
    }

    if (current.trim()) {
      push()
    }

    let rest = token.trimStart()

    while (rest.length > budget) {
      current = rest.slice(0, budget)
      rest = rest.slice(budget)
      push()
    }

    current = rest
  }

  if (current.trim() || !lines.length) {
    push()
  }

  return lines
}

/**
 * Clip a path from the LEFT, not the right — the filename is the part you
 * are looking for. Returns the path unchanged when it already fits.
 */
export function clipPath(path: string, max: number): string {
  const limit = Math.floor(max)

  if (limit <= 1 || path.length <= limit) {
    return path
  }

  return `…${path.slice(path.length - (limit - 1))}`
}
