// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { GLYPH } from '../domain/nocturne.js'
import { ctxBarColor, ctxMeterBar } from '../domain/statusFormat.js'
import { PANEL_INDENT, panelMeterCells, panelSectionGap, panelWidth } from '../lib/panelLayout.js'
import type { Theme } from '../theme.js'
import type { PanelData, PanelSection } from '../types.js'

import { GroupCaption, LeaderRow } from './nocturne.js'
import { Box, Span, Text } from './primitives.js'

/**
 * A slash command's structured answer (`/usage`, setup): captions, leader
 * rows, meters. Built only from the Nocturne row shapes, so it reads like the
 * rest of the transcript rather than a pasted report. Line arithmetic lives in
 * `panelLayout.ts`, which the virtual-height estimate shares.
 */
export function PanelMessage({ cols = 100, panel, t, width: fixedWidth, titled = true }: {
  cols?: number
  panel: PanelData
  t: Theme
  /** Exact columns, for a panel inside a dialog rather than the transcript. */
  width?: number
  /** Dialogs carry their own title; the transcript needs the caption. */
  titled?: boolean
}) {
  // A report reads down a column; past ~76 cells the leaders stop guiding the eye.
  const width = fixedWidth ?? panelWidth(cols)
  const inner = width - PANEL_INDENT
  const labelWidth = (section: PanelSection) => Math.max(0, ...(section.meters ?? []).map(meter => meter.label.length), ...(section.rows ?? []).map(([label]) => label.length))
  const noteWidth = (section: PanelSection) => Math.max(0, ...(section.meters ?? []).map(meter => meter.note?.length ?? 0))
  return (
    <Box flexDirection="column" flexShrink={0} paddingLeft={titled ? 2 : 0} marginBottom={titled ? 1 : 0}>
      {titled ? <GroupCaption label={panel.title} t={t} width={width} /> : null}
      {panel.sections.map((section, index) => {
        const pad = labelWidth(section)
        const notePad = noteWidth(section)
        const rightPad = Math.max(0, ...(section.meters ?? []).map(meter => meter.right?.length ?? 0))
        // The bar gives up cells before the numbers do; below that, the reset time goes.
        const cells = panelMeterCells(inner, pad, notePad, rightPad)
        const showRight = cells > 0
        const barCells = cells > 0 ? cells : panelMeterCells(inner, pad, notePad, 0)
        return (
          <Box flexDirection="column" flexShrink={0} key={index} marginTop={panelSectionGap(section, index)}>
            {section.title ? <GroupCaption count={section.count} label={section.title} t={t} width={width} /> : null}
            {section.heading ? (
              <LeaderRow
                glyph={section.heading.state === 'active' ? GLYPH.state : '○'}
                glyphColor={section.heading.state === 'failed' ? t.ds.failed : section.heading.state === 'active' ? t.ds.done : t.ds.caption}
                notes={(section.heading.notes ?? []).map(note => ({ text: note }))}
                right={section.heading.right}
                rightColor={section.heading.state === 'failed' ? t.ds.failedText : t.ds.numeric}
                t={t}
                target={section.heading.label}
                targetColor={t.ds.strong}
                width={width}
              />
            ) : null}
            <Box flexDirection="column" flexShrink={0} paddingLeft={PANEL_INDENT}>
              {(section.rows ?? []).map(([label, value]) => (
                <LeaderRow glyph="" key={label} label={label.padEnd(pad)} labelColor={t.ds.secondary} quiet right={value} t={t} width={inner} />
              ))}
              {(section.meters ?? []).map(meter => {
                const percent = Math.max(0, Math.min(100, meter.percent))
                return (
                  <Box flexShrink={0} height={1} key={meter.label} width="100%">
                    <Text wrap="truncate-end">
                      <Span color={t.ds.secondary}>{`${meter.label.padEnd(pad)}  `}</Span>
                      <Span color={ctxBarColor(percent, t)}>{ctxMeterBar(percent, Math.max(4, barCells))}</Span>
                      <Span color={t.ds.numeric}>{` ${String(Math.round(percent)).padStart(3)}%`}</Span>
                      {notePad ? <Span color={t.ds.meta}>{`  ${(meter.note ?? '').padEnd(notePad)}`}</Span> : null}
                      {meter.right && showRight ? <Span color={t.ds.caption}>{`  ${meter.right}`}</Span> : null}
                    </Text>
                  </Box>
                )
              })}
              {(section.items ?? []).map((item, itemIndex) => (
                <Text color={t.ds.prose} key={itemIndex} wrap="truncate-end">{`${GLYPH.separator} ${item}`}</Text>
              ))}
              {section.text ? <Text color={t.ds.meta} wrap="wrap">{section.text}</Text> : null}
            </Box>
          </Box>
        )
      })}
    </Box>
  )
}
