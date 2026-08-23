// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** @jsxImportSource @opentui/react */
//
// The slash/path completion menu, rendered in-flow directly above the
// composer.
//
// It stays in-flow rather than becoming an overlay on purpose: the menu must
// read as attached to the input, and an absolutely-positioned overlay
// resolves against the layout root, so anchoring it would need a measured
// ref and would lag by a frame every time the textarea grows.
//
// What went wrong before was never the in-flow placement — it was that the
// menu had no bounded height. A row cap of 10 over variable-height rows let
// word-wrapped skill descriptions grow it past the terminal, and because
// every ancestor up to the footer is flexShrink={0}, Yoga had nothing to give
// and pushed the composer and footer off-screen. Fixed height, fixed rows,
// nothing to shrink.
//
// Raw <box>/<text>/<span> rather than the Box/Text primitives: `Text` accepts
// no width and hardcodes flexShrink={0}, which is exactly what stopped the
// name column from yielding and let descriptions wrap with a hanging indent
// that shifted per command name. Every picker drops to raw elements for the
// same reason.
import { useStore } from '@nanostores/react'
import { createTextAttributes } from '@opentui/core'
import { useTerminalDimensions } from '@opentui/react'

import type { CompletionItem } from '../app/interfaces.js'
import { $uiTheme } from '../app/uiStore.js'
import { GLYPH } from '../domain/nocturne.js'
import { completionColumns, completionMenuRows, completionMeta } from '../lib/completion.js'
import { compactPreview } from '../lib/text.js'

import { GroupCaption } from './nocturne.js'
import { windowItems } from './overlayLayout.js'

const BOLD = createTextAttributes({ bold: true })

/**
 * The group label to print above item `index`, or '' for none.
 *
 * The menu is already ranked by command group; without headers that ordering
 * is real but invisible, so it reads as an arbitrary sequence. A label prints
 * at the first item of each group, and also at the top of the window so a
 * mid-group scroll position still says what you are looking at.
 */
function headingFor(items: readonly CompletionItem[], index: number, windowStart = -1): string {
  const group = items[index]?.group

  if (!group) {
    return ''
  }

  if (index === 0 || index === windowStart) {
    return group
  }

  return items[index - 1]?.group === group ? '' : group
}

/** Headers a window starting at `start` would print, for the row budget. */
function headerCount(items: readonly CompletionItem[], start: number, count: number): number {
  let headers = 0

  for (let i = start; i < start + count && i < items.length; i++) {
    if (headingFor(items, i, start)) {
      headers++
    }
  }

  return headers
}

export interface CompletionMenuProps {
  compIdx: number
  completions: CompletionItem[]
  /**
   * The token being completed. Used to highlight the matched substring IN
   * PLACE — the canvas is explicit that the list must not re-order under your
   * fingers while you type, so the match is marked where it sits rather than
   * being floated to the top.
   */
  query?: string
  width: number
}

/** Split `display` around the first case-insensitive hit of `query`. */
function matchParts(display: string, query: string): [string, string, string] {
  const needle = query.replace(/^[/@]+/, '')

  if (!needle) {
    return [display, '', '']
  }

  const at = display.toLowerCase().indexOf(needle.toLowerCase())

  return at < 0 ? [display, '', ''] : [display.slice(0, at), display.slice(at, at + needle.length), display.slice(at + needle.length)]
}

/**
 * The completion menu — a view over your draft, never its owner.
 *
 * ⎋ closes the list and leaves what you typed sitting in the composer exactly
 * as typed, which is why the header says so out loud: the one thing people
 * fear about an autocomplete popup is that dismissing it eats the sentence.
 */
export function CompletionMenu({ compIdx, completions, query = '', width }: CompletionMenuProps) {
  const t = useStore($uiTheme)
  const { height } = useTerminalDimensions()
  const visible = completionMenuRows(completions.length, height)

  if (!completions.length || visible <= 0) {
    return null
  }

  // Headers occupy rows from the same budget as the items, so the item count
  // is solved for rather than assumed: window, count the headers that window
  // needs, shrink, and settle. Two passes is enough — the header count only
  // ever falls as the window shrinks.
  let budget = visible
  let windowed = windowItems(completions, compIdx, budget)

  for (let pass = 0; pass < 2; pass++) {
    const headers = headerCount(completions, windowed.offset, windowed.items.length)
    const next = Math.max(1, visible - headers)

    if (next === budget) {
      break
    }

    budget = next
    windowed = windowItems(completions, compIdx, budget)
  }

  const { items, offset } = windowed
  const { metaWidth, nameWidth } = completionColumns(completions, width)
  const above = offset
  const below = completions.length - offset - items.length
  const hidden = above + below
  const groups = new Set(completions.map(item => item.group).filter(Boolean)).size
  const token = query.replace(/^[/@]+/, '')
  const sigil = query.startsWith('@') ? '@' : '/'

  return (
    <box
      backgroundColor={t.color.completionBg}
      border
      borderColor={t.ds.focusEdge}
      borderStyle="rounded"
      flexDirection="column"
      flexShrink={0}
      width="100%"
    >
      {/* What you are looking at, and how much of it there is. A count on the
          header is the difference between "no matches yet" and "still
          typing". */}
      <box flexShrink={0} height={1} paddingLeft={1} paddingRight={1} width="100%">
        <text flexShrink={0} truncate width="100%" wrapMode="none">
          <span fg={t.color.accent}>{sigil}</span>
          <span fg={t.ds.secondary}>{token}</span>
          <span fg={t.ds.separator}>{`  ${GLYPH.separator} `}</span>
          <span fg={t.ds.caption}>
            {`${completions.length} match${completions.length === 1 ? '' : 'es'}`}
            {groups > 1 ? ` in ${groups} groups` : ''}
          </span>
        </text>
      </box>
      {items.map((item, i) => {
        const active = offset + i === compIdx
        const meta = metaWidth > 0 ? completionMeta(item.meta, metaWidth) : ''
        const heading = headingFor(completions, offset + i, offset)
        const name = compactPreview(item.display.replace(/^[/@]/, ''), nameWidth)
        const [before, hit, after] = matchParts(name, token)
        const pad = ' '.repeat(Math.max(0, nameWidth - (before.length + hit.length + after.length)))

        return (
          <box flexDirection="column" flexShrink={0} key={`${item.text}:${item.display}`}>
            {heading ? (
              <box flexShrink={0} height={1} paddingLeft={1} paddingRight={1} width="100%">
                <GroupCaption
                  count={completions.filter(candidate => candidate.group === item.group).length}
                  label={heading}
                  t={t}
                  width={width - 4}
                />
              </box>
            ) : null}
            {/* The selected row is a filled ground plus the accent on its own
                marker column — a frame around one row of a list reads as a
                button, and the marker reads as "you are here". */}
            <box
              backgroundColor={active ? t.color.completionCurrentBg : t.color.completionBg}
              flexDirection="row"
              flexShrink={0}
              height={1}
              paddingLeft={1}
              paddingRight={1}
              width="100%"
            >
              <text flexShrink={0} truncate width="100%" wrapMode="none">
                {/* The sigil IS the marker column: the name below it is
                    printed without its own leading `/`, so the row reads
                    `/ review` rather than `/ /review`. */}
                <span fg={active ? t.color.accent : t.ds.separator}>{`${sigil} `}</span>
                {/* Clamp as well as pad: a name longer than the column would
                    otherwise push its own description sideways and break the
                    alignment for every other row. */}
                <span fg={active ? t.ds.title : t.ds.secondary}>{before}</span>
                <span attributes={hit ? BOLD : undefined} fg={t.color.accent}>
                  {hit}
                </span>
                <span fg={active ? t.ds.title : t.ds.secondary}>{after + pad}</span>
                {meta ? <span fg={t.ds.meta}>{`  ${meta}`}</span> : null}
              </text>
            </box>
          </box>
        )
      })}
      {hidden > 0 ? (
        <box flexDirection="row" flexShrink={0} height={1} paddingLeft={1} paddingRight={1} width="100%">
          {/* Eight rows maximum, then "+n more". A completion list you have to
              scroll is a search result, and search belongs on its own screen.
              The key hints live on the composer's own hint row, so this line
              only has to answer "how much is off-screen". */}
          <text flexShrink={0} truncate width="100%" wrapMode="none">
            <span fg={t.ds.caption}>{`+${hidden} more`}</span>
            <span fg={t.ds.separator}>{`  ${GLYPH.separator} `}</span>
            <span fg={t.ds.caption}>{`${compIdx + 1}/${completions.length}`}</span>
          </text>
        </box>
      ) : null}
    </box>
  )
}
