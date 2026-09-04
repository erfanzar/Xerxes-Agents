// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Expansion state for individual tool steps. Kept outside the message
// components for the same reason thinking blocks and tool runs are: transcript
// rows unmount and remount as they leave the virtualized window, and a toggle
// that lived in the row would silently reset when you scrolled past it.

import { atom } from 'nanostores'

export interface ToolStepVisibility {
  overrides: Record<string, boolean>
}

const buildState = (): ToolStepVisibility => ({ overrides: {} })

export const $toolStepVisibility = atom<ToolStepVisibility>(buildState())

export const getToolStepVisibility = () => $toolStepVisibility.get()

/** Steps collapse by default; only an explicit click opens one. */
export const toolStepExpanded = (visibility: ToolStepVisibility, stepId: string): boolean =>
  visibility.overrides[stepId] ?? false

export const toggleToolStep = (stepId: string) => {
  const current = $toolStepVisibility.get()

  $toolStepVisibility.set({
    overrides: { ...current.overrides, [stepId]: !toolStepExpanded(current, stepId) }
  })
}

export const resetToolStepVisibility = () => $toolStepVisibility.set(buildState())
