// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Thinking/reasoning blocks are collapsed by default (a one-line
// `▸ thinking` indicator) and expand only on an explicit toggle: Ctrl+T
// flips every block at once, clicking a thinking header flips that block.
// State lives outside the message components so it survives transcript
// virtualization (rows unmount/remount as they leave the mounted window).

import { atom } from 'nanostores'

export interface ThinkingVisibility {
  /** Global default set by the Ctrl+T chord. */
  allExpanded: boolean
  /** Per-row explicit choices; win over the global default. */
  overrides: Record<string, boolean>
}

const buildState = (): ThinkingVisibility => ({ allExpanded: false, overrides: {} })

export const $thinkingVisibility = atom<ThinkingVisibility>(buildState())

export const getThinkingVisibility = () => $thinkingVisibility.get()

/** Effective expansion for one transcript row: explicit override → global. */
export const thinkingRowExpanded = (visibility: ThinkingVisibility, rowId: string): boolean =>
  visibility.overrides[rowId] ?? visibility.allExpanded

/** Flip one row against its current effective state (click on the header). */
export const toggleThinkingRow = (rowId: string) => {
  const current = $thinkingVisibility.get()

  $thinkingVisibility.set({
    ...current,
    overrides: { ...current.overrides, [rowId]: !thinkingRowExpanded(current, rowId) }
  })
}

/** Flip the global default (Ctrl+T). Per-row overrides stay in force. */
export const toggleAllThinking = () => {
  const current = $thinkingVisibility.get()

  $thinkingVisibility.set({ ...current, allExpanded: !current.allExpanded })
}

export const resetThinkingVisibility = () => $thinkingVisibility.set(buildState())
