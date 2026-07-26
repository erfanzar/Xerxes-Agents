// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { atom } from 'nanostores'

/**
 * Shared width adjustment for the F6 agents panel (sidebar and overlay) and
 * the F7 diff viewer. Cmd+Shift+←/→ grows/shrinks the panel where the
 * terminal forwards Super; Ctrl/Option+Shift+←/→ are the portable fallback
 * (macOS terminals usually intercept Cmd chords).
 *
 * The delta is session-scoped: it applies on top of each panel's responsive
 * default width and is clamped per panel at render time, so one store feeds
 * all three surfaces.
 */

export const PANEL_WIDTH_STEP = 4
export const PANEL_WIDTH_DELTA_MIN = -24
export const PANEL_WIDTH_DELTA_MAX = 60

const $panelWidthDelta = atom(0)

export const getPanelWidthDelta = () => $panelWidthDelta.get()

/** Positive grows the panel, negative shrinks it. Returns the new delta. */
export const adjustPanelWidth = (step: number): number => {
  const next = Math.min(PANEL_WIDTH_DELTA_MAX, Math.max(PANEL_WIDTH_DELTA_MIN, $panelWidthDelta.get() + step))
  $panelWidthDelta.set(next)
  return next
}

export const resetPanelWidth = () => $panelWidthDelta.set(0)

/** Apply the shared delta to a panel's default width: delta 0 keeps the
 *  default exactly; growth is capped at the terminal width minus the overlay
 *  gutter; shrinking bottoms out at a usable 24 columns. */
export const withPanelWidthDelta = (base: number, terminalWidth: number): number =>
  Math.max(24, Math.min(Math.max(24, terminalWidth - 2), base + $panelWidthDelta.get()))

export { $panelWidthDelta }
