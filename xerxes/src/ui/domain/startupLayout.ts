// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface StartupLayoutState {
  busy: boolean
  hasLiveTurn: boolean
  pendingInteraction: boolean
  transcriptEmpty: boolean
}

/**
 * The welcome art is only an idle-session surface. A daemon-owned turn (for
 * example a skill shorthand) can become busy or request approval before it
 * contributes a transcript row, so those signals must switch to the session
 * layout independently of transcript hydration.
 */
export const shouldShowStartupWelcome = ({
  busy,
  hasLiveTurn,
  pendingInteraction,
  transcriptEmpty
}: StartupLayoutState): boolean => transcriptEmpty && !busy && !hasLiveTurn && !pendingInteraction

/** Keep the normal 75-column composer, but use ultra-wide terminals better. */
export const startupComposerWidth = (columns: number): number => {
  const available = Math.max(1, Math.floor(columns) - 4)
  const preferred = columns >= 160 ? Math.min(104, Math.floor(columns * 0.55)) : 75

  return Math.min(available, preferred)
}
