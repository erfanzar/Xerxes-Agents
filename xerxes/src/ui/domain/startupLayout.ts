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

/**
 * The one reading column the app composes onto.
 *
 * Responsive by design: the full available session width minus a small
 * symmetric outer gutter (2 columns per side). There is deliberately no
 * desktop cap — on a 220-column terminal the transcript, user bands, tool
 * rows, and composer all belong to one grid that uses the space, rather than
 * a fixed 75/104-column strip floating in emptiness.
 *
 * Narrow terminals are naturally preserved: the gutter shrinks nothing that
 * matters and the caller's own flex containers clip before overflow.
 *
 * `columns` should be the *session* width — `useMainApp` already subtracts
 * the agent sidebar when it computes `composer.cols`, so every consumer of
 * this measure (transcript column, completion menu, input metrics, tool-row
 * leaders) narrows together when the sidebar mounts.
 */
export const contentColumnWidth = (columns: number): number => Math.max(1, Math.floor(columns) - 4)
