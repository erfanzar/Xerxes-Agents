// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { APPEARANCE_KEY, applyAppearance, applyStyle, loadAppearance, saveAppearance, type StyleChoice } from './appearance.js'
import { applyBackdrop, BACKDROP_KEY, loadBackdrop } from './backdrop.js'
import { PALETTE_KEY } from './palettes.js'

/**
 * Switch Classic ⇄ Neo in place: remember it for every view, stamp the page,
 * then repaint the background (Neo takes only its colours, solid; Classic shows
 * the picture again). No reload, no other setting touched.
 */
export function chooseStyle(style: StyleChoice, root: HTMLElement = document.documentElement): void {
  saveAppearance({ style })
  applyStyle(style, root)
  applyBackdrop(loadBackdrop(), root)
}

/** Storage keys that together decide how every page looks. */
export const SHARED_APPEARANCE_KEYS: ReadonlySet<string> = new Set([APPEARANCE_KEY, BACKDROP_KEY, PALETTE_KEY])

/** Re-apply theme, text size, background and palette from shared storage. */
export function applyStoredAppearance(): void {
  applyAppearance(loadAppearance())
  applyBackdrop(loadBackdrop())
}

type StorageTarget = {
  addEventListener(type: 'storage', listener: (event: StorageEvent) => void): void
  removeEventListener(type: 'storage', listener: (event: StorageEvent) => void): void
}

/**
 * Every workspace window and SSH view is its own page over one shared storage.
 * Pages read the look once at startup, so a theme picked in one view stayed
 * there and every other open view kept the old one until it was reloaded. The
 * browser fires `storage` in every other page when one writes, so follow it.
 * A `null` key means storage was cleared. Returns the stop function.
 */
export function followAppearanceChanges(target: StorageTarget = window, apply: () => void = applyStoredAppearance): () => void {
  const listener = (event: StorageEvent) => {
    if (event.key === null || SHARED_APPEARANCE_KEYS.has(event.key)) apply()
  }
  target.addEventListener('storage', listener)
  return () => target.removeEventListener('storage', listener)
}
