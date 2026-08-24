// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
//
// The row patterns, rendered.
//
// Screen 10 of the design canvas is explicit that four shapes compose every
// screen: a leader row (glyph, verb, target, dotted leader, right-aligned
// quantity), a caption (label · count, then a rule to the edge), a card (dot,
// title, goal, right-aligned budget, at most one live line) and a footer
// (state left, lowercase keys right). Assembling a new screen out of those —
// rather than inventing a fifth shape — is the whole contract.
//
// The two STATELESS shapes live here, because there was nothing to them but
// arithmetic and colour and three files had each grown their own copy. The
// card and the footer do not: a card owns selection, depth, renderable ids
// and per-panel affordances (retry, kill-arming), and a footer owns the keys
// its own panel binds. Extracting those would have produced a component with
// a prop for every caller — which is a shared shape in name only. They stay
// with their panels and take their colours from `stateSkin`, which is where
// the actual duplication was.
//
// The pure half lives in `domain/nocturne.ts`; this file is presentation
// only, so a screen never has to know a hex or re-derive leader arithmetic.
import { stringWidth } from '../lib/terminalRuntime.opentui.js'

import { GLYPH, leaderRun } from '../domain/nocturne.js'
import type { Theme } from '../theme.js'

import { Box, Span, Text } from './primitives.js'

/**
 * Caption — `LABEL · n` then a rule to the edge.
 *
 * The count sits on the caption rather than inside the group so a
 * 312-model provider or a 31-file directory reads as a warning before you
 * open it. `tone` colours both the label and the count: a caption is the
 * first place amber can appear, and it has to be legible as amber from
 * across the screen.
 */
export function GroupCaption({
  count,
  label,
  source,
  t,
  tone,
  width
}: {
  count?: number | string
  label: string
  /** Where this group came from — `built in`, `.xerxes/skills · project`. */
  source?: string
  t: Theme
  /** Voice colour for the label and count; defaults to the caption grey. */
  tone?: string
  width: number
}) {
  const hue = tone ?? t.ds.caption
  const head = count === undefined ? label.toUpperCase() : `${label.toUpperCase()} ${GLYPH.separator} ${count}`
  const tail = source ? ` ${source}` : ''
  const rule = Math.max(0, Math.floor(width) - stringWidth(head) - stringWidth(tail) - 2)

  return (
    <Box flexShrink={0} height={1} width="100%">
      <Text wrap="truncate-end">
        <Span color={hue}>{label.toUpperCase()}</Span>
        {count === undefined ? null : (
          <>
            <Span color={t.ds.separator}>{` ${GLYPH.separator} `}</Span>
            <Span color={hue}>{String(count)}</Span>
          </>
        )}
        {rule > 0 ? <Span color={t.ds.divider}>{` ${'─'.repeat(rule)}`}</Span> : null}
        {tail ? <Span color={t.ds.separator}>{tail}</Span> : null}
      </Text>
    </Box>
  )
}

export interface LeaderRowProps {
  /** Leading mark. Defaults to the tool disc. */
  glyph?: string
  glyphColor?: string
  /** The verb — `read`, `grep`, `hunks seen`. Ramp step: secondary. */
  label?: string
  labelColor?: string
  /** What the verb acted on. Ramp step: title. */
  target?: string
  targetColor?: string
  /** Facts between the target and the leader, in source order. */
  notes?: readonly { color?: string; text: string }[]
  /** The right-aligned quantity. Ramp step: numeric. */
  right?: string
  rightColor?: string
  /** Quieter dots, for rows that are not the current selection. */
  quiet?: boolean
  t: Theme
  /** Columns the row may occupy. */
  width: number
}

/**
 * Leader row — glyph, verb, target, dotted leader, right-aligned quantity.
 *
 * Every quantity in the product is right-aligned through this component, so
 * durations and token counts stack into a column you can read vertically
 * without reading the rows.
 */
export function LeaderRow({
  glyph = GLYPH.tool,
  glyphColor,
  label,
  labelColor,
  notes,
  quiet,
  right,
  rightColor,
  t,
  target,
  targetColor,
  width
}: LeaderRowProps) {
  const noteText = (notes ?? []).map(note => `  ${note.text}`).join('')
  const left =
    (glyph ? `${glyph} ` : '') + (label ? `${label} ` : '') + (target ?? '') + noteText
  const dots = right ? leaderRun(width, stringWidth(left), stringWidth(right)) : ''

  return (
    <Box flexShrink={0} height={1} width="100%">
      <Text wrap="truncate-end">
        {glyph ? <Span color={glyphColor ?? t.color.accent}>{`${glyph} `}</Span> : null}
        {label ? <Span color={labelColor ?? t.color.toolName}>{`${label} `}</Span> : null}
        {target ? <Span color={targetColor ?? t.ds.title}>{target}</Span> : null}
        {(notes ?? []).map((note, index) => (
          <Span color={note.color ?? t.ds.meta} key={index}>{`  ${note.text}`}</Span>
        ))}
        {dots ? <Span color={quiet ? t.ds.leaderQuiet : t.ds.leader}>{dots}</Span> : null}
        {right ? <Span color={rightColor ?? t.ds.numeric}>{` ${right}`}</Span> : null}
      </Text>
    </Box>
  )
}
