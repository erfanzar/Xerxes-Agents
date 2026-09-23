// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const APPEARANCE_KEY = 'xerxes.desktop.appearance.v1'
export const FONT_SIZES = ['11', '12', '13', '15', '17'] as const
export type FontSize = typeof FONT_SIZES[number]
/** Sizes are px; the labels are what the control shows. */
export const FONT_LABELS: Record<FontSize, string> = { '11': 'XS', '12': 'S', '13': 'M', '15': 'L', '17': 'XL' }

export function parseAppearance(value: unknown): { theme: 'system' | 'dark' | 'light'; font: FontSize } {
  const record = value && typeof value === 'object' ? value as Record<string, unknown> : {}
  const font = FONT_SIZES.find(size => size === record.font)
  return { theme: record.theme === 'dark' || record.theme === 'light' ? record.theme : 'system', font: font ?? '12' }
}
export function saveAppearance(): void {
  try { localStorage.setItem(APPEARANCE_KEY, JSON.stringify({ theme: document.documentElement.getAttribute('data-user-theme') ?? 'system', font: document.documentElement.getAttribute('data-font') ?? '12' })) } catch { /* Keep the current window usable when persistence is unavailable. */ }
}
export function restoreAppearance(): void {
  try {
    const value = parseAppearance(JSON.parse(localStorage.getItem(APPEARANCE_KEY) || '{}'))
    document.documentElement.setAttribute('data-font', value.font)
    if (value.theme !== 'system') {
      document.documentElement.setAttribute('data-user-theme', value.theme)
      document.documentElement.setAttribute('data-theme', value.theme)
    }
  } catch { /* System appearance remains the default for invalid storage. */ }
}
