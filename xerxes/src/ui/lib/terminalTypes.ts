// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Terminal mouse-reporting modes understood by Xerxes configuration. */
export type MouseTrackingMode = 'all' | 'buttons' | 'off' | 'wheel'

/** An async operation that temporarily owns the terminal. */
export type RunExternalProcess = () => Promise<void>

/**
 * Controller-facing scroll contract.
 *
 * OpenTUI's native ScrollBoxRenderable is adapted to this narrow interface by
 * the view, keeping the controller independent from renderer implementation
 * details.
 */
export interface ScrollBoxHandle {
  getFreshScrollHeight: () => number
  getLastManualScrollAt: () => number
  getPendingDelta: () => number
  getScrollHeight: () => number
  getScrollTop: () => number
  getViewportHeight: () => number
  getViewportTop: () => number
  isSticky: () => boolean
  scrollBy: (dy: number) => void
  scrollTo: (y: number) => void
  scrollToBottom: () => void
  scrollToElement: (element: unknown, offset?: number) => void
  setClampBounds: (min: number | undefined, max: number | undefined) => void
  subscribe: (listener: () => void) => () => void
}
