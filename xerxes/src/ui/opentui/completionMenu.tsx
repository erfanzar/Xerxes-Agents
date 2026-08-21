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
import { completionColumns, completionMenuRows, completionMeta } from '../lib/completion.js'
import { compactPreview } from '../lib/text.js'

import { windowItems } from './overlayLayout.js'

const BOLD = createTextAttributes({ bold: true })

export interface CompletionMenuProps {
  compIdx: number
  completions: CompletionItem[]
  width: number
}

export function CompletionMenu({ compIdx, completions, width }: CompletionMenuProps) {
  const t = useStore($uiTheme)
  const { height } = useTerminalDimensions()
  const visible = completionMenuRows(completions.length, height)

  if (!completions.length || visible <= 0) {
    return null
  }

  const { items, offset } = windowItems(completions, compIdx, visible)
  const { metaWidth, nameWidth } = completionColumns(completions, width)
  const above = offset
  const below = completions.length - offset - items.length
  const hasOverflow = above > 0 || below > 0

  return (
    <box backgroundColor={t.color.completionBg} flexDirection="column" flexShrink={0} width="100%">
      {items.map((item, i) => {
        const active = offset + i === compIdx
        const meta = metaWidth > 0 ? completionMeta(item.meta, metaWidth) : ''

        return (
          <box
            backgroundColor={active ? t.color.selectionBg : t.color.completionBg}
            flexDirection="row"
            flexShrink={0}
            height={1}
            key={`${item.text}:${item.display}`}
            paddingLeft={2}
            paddingRight={2}
            width="100%"
          >
            <text flexShrink={0} truncate width="100%" wrapMode="none">
              <span fg={active ? t.color.brandGold : t.color.muted}>{active ? '▸ ' : '  '}</span>
              {/* Clamp as well as pad: a name longer than the column would
                  otherwise push its own description sideways and break the
                  alignment for every other row. */}
              <span attributes={active ? BOLD : undefined} fg={active ? t.color.brandGold : t.color.label}>
                {compactPreview(item.display, nameWidth).padEnd(nameWidth)}
              </span>
              {meta ? <span fg={t.color.muted}>{`  ${meta}`}</span> : null}
            </text>
          </box>
        )
      })}
      {hasOverflow ? (
        <box flexDirection="row" flexShrink={0} height={1} paddingLeft={2} paddingRight={2} width="100%">
          {/* The key hints (Tab/↑↓/Esc) already live in the composer hint row,
              so this single row only has to answer "how much is off-screen". */}
          <text fg={t.color.muted} flexShrink={0} truncate width="100%" wrapMode="none">
            {above > 0 ? `↑ ${above}` : '  '}
            {`   ${compIdx + 1}/${completions.length}`}
            {below > 0 ? `   ↓ ${below}` : ''}
          </text>
        </box>
      ) : null}
    </box>
  )
}
