// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Display preferences that are per machine, not per task: UI scale, the
 * terminal font, and how much each sidebar row shows. Each is stored in
 * local storage and re-applied when a window starts; a change in one window
 * reaches the others through the storage event.
 */

export const UI_SCALES = [90, 100, 110, 125, 150, 175] as const
export type SessionDensity = 'compact' | 'comfortable' | 'detailed'
export const SESSION_DENSITIES: readonly SessionDensity[] = ['compact', 'comfortable', 'detailed']

const SCALE_KEY = 'xerxes.desktop.ui-scale.v1'
const TERMINAL_FONT_KEY = 'xerxes.desktop.terminal-font.v1'
const DENSITY_KEY = 'xerxes.desktop.session-density.v1'

function read(key: string): string | null {
  try { return localStorage.getItem(key) } catch { return null }
}
function write(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key)
    else localStorage.setItem(key, value)
  } catch { /* Applies to this window even when it cannot be kept. */ }
}

/** A fresh install's UI scale. */
export const DEFAULT_UI_SCALE = 90

/** Percent, one of UI_SCALES; DEFAULT_UI_SCALE when unset or unreadable. */
export function loadUiScale(): number {
  const value = Number(read(SCALE_KEY))
  return (UI_SCALES as readonly number[]).includes(value) ? value : DEFAULT_UI_SCALE
}

/** Zoom the whole window. The View menu's Cmd +/−/0 still work on top. */
export function applyUiScale(percent: number, save = true): void {
  if (!(UI_SCALES as readonly number[]).includes(percent)) throw new Error(`Unsupported UI scale ${percent}%`)
  window.xerxes?.setZoomFactor?.(percent / 100)
  if (save) write(SCALE_KEY, String(percent))
}

/**
 * A font-family list for the terminals: names separated by commas, quoted
 * or not. Anything that could close the declaration is refused rather than
 * passed to CSS.
 */
export function parseTerminalFont(value: string): string | undefined {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  if (trimmed.length > 200 || /[;{}<>\\\n\r]/.test(trimmed)) throw new Error('Use font names separated by commas, e.g. MesloLGS NF, monospace')
  return trimmed
}

export function loadTerminalFont(): string | undefined {
  try { return parseTerminalFont(read(TERMINAL_FONT_KEY) ?? '') } catch { return undefined }
}

/** Set (or clear, with undefined) the terminal font; open terminals follow. */
export function applyTerminalFont(font: string | undefined, save = true): void {
  const root = document.documentElement
  if (font) root.style.setProperty('--x-terminal-font', font)
  else root.style.removeProperty('--x-terminal-font')
  // Terminals watch this attribute to pick the font up without remounting.
  root.setAttribute('data-terminal-font', font ? String(font.length) + ':' + font.slice(0, 24) : 'default')
  if (save) write(TERMINAL_FONT_KEY, font ?? null)
}

export function loadSessionDensity(): SessionDensity {
  const value = read(DENSITY_KEY)
  return SESSION_DENSITIES.includes(value as SessionDensity) ? value as SessionDensity : 'comfortable'
}

export function applySessionDensity(density: SessionDensity, save = true): void {
  document.documentElement.setAttribute('data-session-density', density)
  if (save) write(DENSITY_KEY, density)
}

/** Subscribe to density changes (Settings writes the root attribute). */
export function subscribeSessionDensity(listener: () => void): () => void {
  if (typeof MutationObserver === 'undefined' || typeof document === 'undefined') return () => {}
  const observer = new MutationObserver(listener)
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-session-density'] })
  return () => observer.disconnect()
}

export function currentSessionDensity(): SessionDensity {
  if (typeof document === 'undefined') return 'comfortable'
  const value = document.documentElement.getAttribute('data-session-density')
  return SESSION_DENSITIES.includes(value as SessionDensity) ? value as SessionDensity : 'comfortable'
}

/** Re-apply every stored preference; called once as each window starts. */
export function restoreDisplayPrefs(): void {
  applyUiScale(loadUiScale(), false)
  applyTerminalFont(loadTerminalFont(), false)
  applySessionDensity(loadSessionDensity(), false)
  window.addEventListener('storage', event => {
    if (event.key === SCALE_KEY) applyUiScale(loadUiScale(), false)
    else if (event.key === TERMINAL_FONT_KEY) applyTerminalFont(loadTerminalFont(), false)
    else if (event.key === DENSITY_KEY) applySessionDensity(loadSessionDensity(), false)
  })
}
