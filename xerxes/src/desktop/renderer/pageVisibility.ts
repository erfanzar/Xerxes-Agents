// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Whether this page is out of sight: hidden, minimized, or covered by another
 * workspace view in the same window (the main process marks the root
 * `data-occluded`). A window keeps one page per workspace it has opened, so
 * polling that ignores this runs once per open workspace, all the time.
 * Pollers skip their runtime call while this is true and keep their timer.
 */
export function pageIsBackground(): boolean {
  if (typeof document === 'undefined') return false
  return document.visibilityState === 'hidden' || document.documentElement.hasAttribute('data-occluded')
}
