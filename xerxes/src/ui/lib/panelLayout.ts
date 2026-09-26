// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Line arithmetic for transcript panels (`/usage`, setup), shared by the
// renderer and the virtual-height estimate so a panel never gets clipped or
// padded by a guess that drifted from what actually paints.
import type { PanelData, PanelSection } from '../types.js'

import { stringWidth } from './terminalRuntime.opentui.js'

/** Indent of every panel line under its caption. */
export const PANEL_INDENT = 2
/** Cells in a panel meter bar. */
export const PANEL_METER_CELLS = 16
/** Widest a panel grows: past this the dotted leaders stop guiding the eye. */
export const PANEL_MAX_WIDTH = 76

/** Columns a panel occupies inside a transcript `cols` wide (its gutter and indent removed). */
export function panelWidth(cols: number): number {
  return Math.max(36, Math.min(PANEL_MAX_WIDTH, cols - 8))
}

function wrappedLines(text: string, width: number): number {
  const usable = Math.max(10, width)
  return text.split('\n').reduce((total, line) => total + Math.max(1, Math.ceil(stringWidth(line) / usable)), 0)
}

/** Painted rows for one section, excluding the gap before it. */
export function panelSectionLines(section: PanelSection, cols: number): number {
  const width = panelWidth(cols) - PANEL_INDENT
  return (section.title ? 1 : 0)
    + (section.heading ? 1 : 0)
    + (section.rows?.length ?? 0)
    + (section.meters?.length ?? 0)
    + (section.items?.length ?? 0)
    + (section.text ? wrappedLines(section.text, width) : 0)
}

/** A captioned section opens a new group and gets a blank line above it. */
export function panelSectionGap(section: PanelSection, index: number): number {
  return index > 0 && section.title ? 1 : 0
}

/**
 * Bar cells that fit a meter row `inner` wide: label, bar, ` 100%`, the note
 * column and the right text. 0 means even a 4-cell bar does not fit with the
 * right text, so the caller drops it.
 */
export function panelMeterCells(inner: number, label: number, note: number, right: number): number {
  const fixed = label + 2 + 5 + (note ? note + 2 : 0) + (right ? right + 2 : 0) + 1
  const cells = Math.min(PANEL_METER_CELLS, inner - fixed)
  return cells >= 4 ? cells : 0
}

export function panelLineCount(panel: PanelData, cols: number): number {
  // Title caption, then sections, then one line of breathing room.
  const body = panel.sections.reduce((total, section, index) => total + panelSectionGap(section, index) + panelSectionLines(section, cols), 0)
  return 1 + body + 1
}
