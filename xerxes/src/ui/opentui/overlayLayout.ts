// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Fit a modal panel inside the terminal while retaining its preferred gutter.
 * The minimum is aspirational: terminals narrower than it must win, otherwise
 * Yoga positions part of the panel offscreen and can hide every useful key.
 */
export function responsivePanelWidth(
  terminalWidth: number,
  { gutter = 4, max, min }: { gutter?: number; max: number; min: number }
): number {
  const width = Math.max(1, Math.floor(terminalWidth))
  const available = Math.max(1, width - 2)
  const fluid = Math.max(1, width - Math.max(2, gutter))

  return Math.min(available, max, Math.max(Math.min(min, available), fluid))
}

export interface OverlayPanelSize {
  height: number
  width: number
}

export interface OverlayPanelSpec {
  /**
   * Rows the panel needs for its own content, including chrome. When given,
   * the panel shrinks to fit rather than always filling its allowance —
   * otherwise a list with three rows renders as a 38-row empty box, which
   * looks more broken than a panel that is merely too small.
   */
  desiredHeight?: number
  gutterX?: number
  gutterY?: number
  maxHeight?: number
  maxWidth: number
  minHeight?: number
  minWidth: number
}

/**
 * Fit a full-screen overlay's panel box to the terminal.
 *
 * The width half delegates to {@link responsivePanelWidth}, which stays the
 * primitive; this is the single entry point the panels call, so the two
 * competing width helpers stop diverging.
 *
 * The height half is what was missing entirely. F6 computed its own
 * `marginY = min(30, floor((height - 20) / 2))`, which on a 42-row terminal
 * left the panel 20 rows — less than half the screen, for a surface whose
 * whole job is a long list. F7 and F8 said `height="80%"`, which cannot be
 * asserted from a pure function because it resolves inside Yoga.
 */
export function overlayPanelSize(
  terminal: { height: number; width: number },
  { desiredHeight, gutterX = 4, gutterY = 4, maxHeight, maxWidth, minHeight = 12, minWidth }: OverlayPanelSpec
): OverlayPanelSize {
  const rows = Math.max(1, Math.floor(terminal.height))
  const available = Math.max(1, rows - 2)
  const fluid = Math.max(1, rows - Math.max(2, gutterY))
  const allowance = Math.min(
    available,
    maxHeight ?? Number.POSITIVE_INFINITY,
    Math.max(Math.min(minHeight, available), fluid)
  )
  // Shrink-to-fit, but never below the aspirational minimum and never above
  // what the terminal can show.
  const height =
    desiredHeight === undefined
      ? allowance
      : Math.min(allowance, Math.max(Math.min(minHeight, available), Math.ceil(desiredHeight)))

  return { height, width: responsivePanelWidth(terminal.width, { gutter: gutterX, max: maxWidth, min: minWidth }) }
}

/**
 * The width half alone, for content-sized overlays (the pager, the info
 * sheet) that let their height follow their content. They still take their
 * numbers from {@link OVERLAY_PANEL_SPECS} so no width constants escape the
 * table.
 */
export function overlayPanelWidth(terminalWidth: number, spec: OverlayPanelSpec): number {
  return responsivePanelWidth(terminalWidth, { gutter: spec.gutterX ?? 4, max: spec.maxWidth, min: spec.minWidth })
}

/**
 * Per-panel intent, in one table.
 *
 * The max widths deliberately differ — agent rows are narrow, diffs want the
 * context, terminal output wants more still. Divergence in a named table is a
 * decision; the same divergence spread across three component files was an
 * accident nobody could see.
 */
export const OVERLAY_PANEL_SPECS = {
  // Mockup 04's agent view is a LARGE bounded surface, not a centered card:
  // full height minus the standard gutter and diff-width, so a long grouped
  // list and the inspector both get the room the design gives them.
  agents: { maxWidth: 120, minWidth: 48 },
  diff: { maxWidth: 120, minWidth: 60 },
  info: { maxWidth: 90, minWidth: 42 },
  pager: { maxWidth: 110, minWidth: 48 },
  terminals: { maxWidth: 140, minWidth: 60 }
} as const satisfies Record<string, OverlayPanelSpec>

/**
 * Slice a picker list to the visible window around the selection. Shared by
 * every picker overlay so scroll behavior stays identical across all of them.
 */
export function windowItems<T>(
  items: readonly T[],
  selected: number,
  visible: number
): { items: readonly T[]; offset: number } {
  if (visible <= 0) {
    return { items: [] as readonly T[], offset: 0 }
  }

  const offset = Math.max(0, Math.min(selected - Math.floor(visible / 2), items.length - visible))

  return { items: items.slice(offset, offset + visible), offset }
}
