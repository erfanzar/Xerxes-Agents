// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createRoot } from 'react-dom/client'

import { restoreAppearance } from './appearance.js'
import { applyBackdrop, loadBackdrop } from './backdrop.js'
import { restoreDisplayPrefs } from './displayPrefs.js'
import { ErrorBoundary } from './ErrorBoundary.js'
import { App } from './App.js'

// Theme: follow the system only while the user has not pinned an explicit
// choice (Settings writes data-user-theme alongside data-theme).
restoreAppearance()
const root = document.documentElement
// applyBackdrop owns data-theme: a background without a palette decides
// light or dark from its own colours; otherwise the system (or a palette's
// variant for it) does.
const applyTheme = (): void => {
  if (root.hasAttribute('data-user-theme')) return
  applyBackdrop(loadBackdrop())
}
applyBackdrop(loadBackdrop())
restoreDisplayPrefs()
// Covered by another workspace view in the same window: draw nothing, so a
// transparent view on top does not show this page through it.
const setOccluded = (occluded: boolean): void => {
  if (occluded) root.setAttribute('data-occluded', '')
  else root.removeAttribute('data-occluded')
}
// A push can arrive before this page listens (startup), so it also asks. A
// push that lands first is newer than the answer and wins.
let pushed = false
window.xerxes?.onOccluded?.(occluded => { pushed = true; setOccluded(occluded) })
void window.xerxes?.isOccluded?.().then(occluded => { if (!pushed) setOccluded(occluded) }).catch(() => {})
window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', applyTheme)

const container = document.getElementById('root')
if (container) {
  // Last line of defence: without this, any throw during render leaves an
  // empty black window with no message and no way back but the View menu.
  createRoot(container).render(<ErrorBoundary><App /></ErrorBoundary>)
}
