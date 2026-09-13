// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const APPEARANCE_KEY = 'xerxes.desktop.appearance.v1'
export function parseAppearance(value: unknown): { theme: 'system' | 'dark' | 'light'; font: '11' | '12' | '13' } {
  const record = value && typeof value === 'object' ? value as Record<string, unknown> : {}
  return { theme: record.theme === 'dark' || record.theme === 'light' ? record.theme : 'system', font: record.font === '11' || record.font === '13' ? record.font : '12' }
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
