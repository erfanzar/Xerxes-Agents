// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Target host selection for VS Code commands (menu actions, sendSelection).
 * Commands mutate only the active or most recently visible Xerxes surface
 * instead of broadcasting to every open sidebar and editor panel at once.
 */

export interface TargetHostCandidate {
  readonly surface: 'view' | 'panel'
  readonly isVisible: () => boolean
  readonly isActive: () => boolean
  readonly lastActiveAt: number
}

function mostRecent<T extends TargetHostCandidate>(candidates: readonly T[]): T | undefined {
  let best: T | undefined
  for (const candidate of candidates) {
    if (!best || candidate.lastActiveAt >= best.lastActiveAt) best = candidate
  }
  return best
}

/**
 * Choose the single Xerxes webview host that should receive a user command:
 * 1. An active (focused) panel/view, preferring the most recently activated.
 * 2. Otherwise a visible panel/view, preferring the most recently activated.
 * 3. Otherwise the sidebar view (`surface === 'view'`).
 * 4. Otherwise the most recently used remaining host.
 */
export function pickTargetHost<T extends TargetHostCandidate>(entries: Iterable<T>): T | undefined {
  const all = [...entries]
  if (all.length === 0) return undefined

  const active = all.filter(entry => entry.isActive())
  if (active.length > 0) return mostRecent(active)

  const visible = all.filter(entry => entry.isVisible())
  if (visible.length > 0) return mostRecent(visible)

  const views = all.filter(entry => entry.surface === 'view')
  if (views.length > 0) return mostRecent(views)

  return mostRecent(all)
}
