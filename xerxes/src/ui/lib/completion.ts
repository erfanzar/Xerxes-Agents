// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Pure completion helpers shared by the useCompletion hook. The daemon's
// `complete` RPC returns the candidates; this decides WHEN to ask and HOW to
// apply a chosen value over the active token.

export interface Completion {
  value: string
  label: string
  meta?: string
}

// ── Menu geometry ───────────────────────────────────────────────────────
//
// The completion menu renders in-flow, directly above the composer, so its
// height has to be a constant the layout can honor. It used to have none: a
// row-count cap of 10 over variable-height rows let word-wrapped skill
// descriptions grow the menu past the terminal and push the composer and
// footer off-screen. Rows are now clamped to one line each and the row count
// is derived from the terminal height.

/** Never show more than this many rows, however tall the terminal is. */
export const COMPLETION_MENU_MAX_ROWS = 8

/**
 * Rows the rest of the chrome needs: session header, tab strip, transcript
 * column padding, prompt zone, composer input, hint row, notice banner and
 * the workspace footer — plus a couple of transcript rows so the menu never
 * feels like it took over the screen.
 *
 * 16 → 19 when the menu became a bordered card with a query header: the frame
 * costs a row per edge and the header one more. They come out of the chrome
 * budget rather than the row cap, because the alternative is a menu that
 * claims eight rows, draws eleven, and pushes the composer off-screen.
 */
export const COMPLETION_MENU_RESERVED_ROWS = 19

/** How many completion rows fit, given the item count and terminal height. */
export function completionMenuRows(items: number, terminalRows: number): number {
  const budget = Math.floor(terminalRows) - COMPLETION_MENU_RESERVED_ROWS

  return Math.max(0, Math.min(COMPLETION_MENU_MAX_ROWS, Math.floor(items), budget))
}

export interface CompletionColumns {
  /** 0 when the terminal is too narrow to afford a description column. */
  metaWidth: number
  nameWidth: number
}

const NAME_COLUMN_MIN = 8
const NAME_COLUMN_MAX = 28
/** Below this, a description is too clipped to inform; drop it entirely. */
const META_COLUMN_MIN = 14
/** paddingX(2)*2 + marker(2) + gap(2). */
const COMPLETION_ROW_CHROME = 8

/**
 * Column widths for the menu's two-column rows.
 *
 * `nameWidth` is measured over ALL items, not just the visible window: a
 * window-relative width makes the description column jump sideways as the
 * selection moves, which reads as flicker.
 */
export function completionColumns(
  items: readonly { display: string }[],
  totalWidth: number
): CompletionColumns {
  const widest = items.reduce((max, item) => Math.max(max, item.display.length), 0)
  const nameWidth = Math.min(NAME_COLUMN_MAX, Math.max(NAME_COLUMN_MIN, widest))
  const metaWidth = Math.floor(totalWidth) - COMPLETION_ROW_CHROME - nameWidth

  if (metaWidth < META_COLUMN_MIN) {
    return { metaWidth: 0, nameWidth: Math.max(NAME_COLUMN_MIN, Math.floor(totalWidth) - COMPLETION_ROW_CHROME) }
  }

  return { metaWidth, nameWidth }
}

/**
 * Every bundled skill description opens with this, which wastes the most
 * valuable columns in the row on boilerplate identical across all of them.
 */
const SKILL_DESCRIPTION_BOILERPLATE = /^use this skill (?:whenever|when|to|for)\s+/i

/** Reduce a description — possibly multi-sentence prose — to one clamped line. */
export function completionMeta(meta: string | undefined, max: number): string {
  if (!meta || max <= 0) {
    return ''
  }

  const flat = meta.replace(/\s+/g, ' ').trim().replace(SKILL_DESCRIPTION_BOILERPLATE, '')
  // First sentence only: the rest is detail the /help panel still shows.
  const firstSentence = /^(.+?[.!?])(?:\s|$)/.exec(flat)?.[1] ?? flat

  return firstSentence.length > max ? `${firstSentence.slice(0, Math.max(1, max - 1)).trimEnd()}…` : firstSentence
}

// ── Menu ordering ───────────────────────────────────────────────────────

/**
 * Category order for the bare-slash menu, most-reached-for first.
 *
 * A plain "/" used to be an alphabetical wall of project skills, so the first
 * screen was `ai-voiceover-…`, `apple-notes`, `arxiv` — none of which anyone
 * opens the menu to find. Skills stay first-class through prefix matching
 * (typing `/deep` still surfaces `deepscan` at the top) and through the
 * skills hub; they just no longer own the default view.
 */
const GROUP_ORDER = [
  'session',
  'config',
  // `info` sits above `skills` deliberately: /help and /status are among the
  // most-reached commands, and a project with forty skills would otherwise
  // bury them — a milder version of the problem this ordering exists to fix.
  'info',
  'skills',
  'tools',
  'memory',
  'snapshots',
  'voice',
  'messaging',
  'feedback',
  'exit'
] as const

const groupRank = (group: string | undefined): number => {
  const i = GROUP_ORDER.indexOf((group ?? '').toLowerCase() as (typeof GROUP_ORDER)[number])

  return i < 0 ? GROUP_ORDER.length : i
}

const bare = (s: string) => s.replace(/^\/+/, '').toLowerCase()

/**
 * Rank by how well the item answers what was typed, then by category.
 *
 * The tier ladder is the one the mockup pins — "fuzzy: prefix → substring →
 * skill body": exact name, then name prefix, then a substring hit on the name
 * itself, and only then a hit on the item's description text (the `meta` the
 * menu row already displays). Items matching neither name nor description rank
 * last; nothing is dropped, the bounded window simply pushes them behind the
 * "+n more" row. Descriptions are never searched ahead of names: typing
 * `/scan` must not bury `deepscan` under every skill whose prose mentions a
 * scan.
 *
 * Must run AFTER `mergeCompletionItems`: that merge is a position-preserving
 * dedupe whose local-wins property protects a locally-known skill from a
 * daemon duplicate, and re-sorting before it would be undone by the
 * concatenation.
 */
export function rankCompletionItems<T extends { display: string; group?: string; meta?: string; text?: string }>(
  items: readonly T[],
  prefix: string
): T[] {
  const want = bare(prefix)

  const tier = (item: T): number => {
    const name = bare(item.display)

    if (!want) {
      return 1
    }

    if (name === want) {
      return 0
    }

    if (name.startsWith(want)) {
      return 1
    }

    if (name.includes(want)) {
      return 2
    }

    const body = (item.meta ?? '').toLowerCase()

    return body.includes(want) ? 3 : 4
  }

  return [...items]
    .map((item, index) => ({ index, item }))
    .sort((a, b) => {
      const byTier = tier(a.item) - tier(b.item)

      if (byTier !== 0) {
        return byTier
      }

      const byGroup = groupRank(a.item.group) - groupRank(b.item.group)

      if (byGroup !== 0) {
        return byGroup
      }

      // Shorter names first: '/help' should beat '/helper-skill' for '/help'.
      // Only when something was typed — with a bare '/' there is nothing for
      // length to be evidence of, and it would scramble each group into an
      // arbitrary short-to-long order instead of a scannable alphabetical one.
      const byLength = want ? a.item.display.length - b.item.display.length : 0

      if (byLength !== 0) {
        return byLength
      }

      const byName = a.item.display.localeCompare(b.item.display)

      // Stable final tie-break so equal items never reorder between renders.
      return byName !== 0 ? byName : a.index - b.index
    })
    .map(entry => entry.item)
}

/** True when the draft's active token is worth requesting completions for. */
export function shouldRequestCompletion(draft: string): boolean {
  if (!draft.trim()) {
    return false
  }
  // Slash command name (still typing it, no space yet).
  if (draft.startsWith('/') && !draft.includes(' ')) {
    return true
  }
  // Path-like last token: starts with @ / . ~ or contains a slash.
  const last = draft.split(/\s+/).at(-1) ?? ''
  return /^@?([./~]|[^\s]*\/)/.test(last) && last.length > 0
}

/** The token the completion menu is operating on (last whitespace token). */
export function activeToken(draft: string): string {
  if (draft.startsWith('/') && !draft.includes(' ')) {
    return draft
  }
  return draft.split(/\s+/).at(-1) ?? ''
}

/** Replace the active token in `draft` with `value`. */
export function applyCompletion(draft: string, value: string): string {
  // Slash command: the value is the whole new draft.
  if (draft.startsWith('/') && !draft.includes(' ')) {
    return value
  }
  // Otherwise replace the trailing non-space run (the active token).
  const idx = draft.search(/\S+$/)
  if (idx < 0) {
    return draft + value
  }
  return draft.slice(0, idx) + value
}

/** Cycle a selection index within [0, len) with wraparound. */
export function cycleIndex(current: number, len: number, delta: number): number {
  if (len <= 0) {
    return 0
  }
  return (((current + delta) % len) + len) % len
}
