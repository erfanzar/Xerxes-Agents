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
