// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const APPEARANCE_KEY = 'xerxes.desktop.appearance.v1'
export const FONT_SIZES = ['11', '12', '13', '15', '17'] as const
export type FontSize = typeof FONT_SIZES[number]
export type ThemeChoice = 'system' | 'dark' | 'light'
export interface Appearance { readonly theme: ThemeChoice; readonly font: FontSize }
/** Sizes are px; the labels are what the control shows. */
export const FONT_LABELS: Record<FontSize, string> = { '11': 'XS', '12': 'S', '13': 'M', '15': 'L', '17': 'XL' }
/** A fresh install's interface text size (XS). */
export const DEFAULT_FONT_SIZE: FontSize = '11'

export function parseAppearance(value: unknown): Appearance {
  const record = value && typeof value === 'object' ? value as Record<string, unknown> : {}
  const font = FONT_SIZES.find(size => size === record.font)
  return { theme: record.theme === 'dark' || record.theme === 'light' ? record.theme : 'system', font: font ?? DEFAULT_FONT_SIZE }
}

/** The stored choice shared by every window and workspace view. */
export function loadAppearance(): Appearance {
  try { return parseAppearance(JSON.parse(localStorage.getItem(APPEARANCE_KEY) || '{}')) } catch { return parseAppearance(undefined) }
}

/**
 * Store only the fields that changed. Every workspace and SSH view shares this
 * storage; writing the whole record from one page's attributes let a page still
 * showing an older theme put that theme back when only the font was changed.
 */
export function saveAppearance(change: Partial<Appearance>): void {
  try { localStorage.setItem(APPEARANCE_KEY, JSON.stringify({ ...loadAppearance(), ...change })) } catch { /* Keep the current window usable when persistence is unavailable. */ }
}

/** Put a choice on the page. 'system' hands light/dark back to the OS (and the background). */
export function applyAppearance(value: Appearance, root: Pick<HTMLElement, 'setAttribute' | 'removeAttribute'> = document.documentElement): void {
  root.setAttribute('data-font', value.font)
  if (value.theme === 'system') {
    root.removeAttribute('data-user-theme')
  } else {
    root.setAttribute('data-user-theme', value.theme)
    root.setAttribute('data-theme', value.theme)
  }
}

export function restoreAppearance(): void {
  try { applyAppearance(loadAppearance()) } catch { /* System appearance remains the default for invalid storage. */ }
}
