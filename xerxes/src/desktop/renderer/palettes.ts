// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Colour themes for the desktop.
 *
 * A palette sets the surfaces, text and accent. With no background they are
 * solid; with a background they become the tint of the glass over the
 * blurred picture, at the opacity the Conversation / Side panels sliders
 * set, so a theme and a background work together instead of replacing each
 * other. Light / Dark / System picks a two-mode palette's variant; a
 * single-mode palette (Cyberpunk, Midnight…) is always its own mode.
 *
 * The palettes are Hermes Agent's desktop themes (MIT, Copyright (c) 2025
 * Nous Research), several of them forks of their VS Code originals (GitHub,
 * Catppuccin, Everforest, Solarized). Values are copied, not re-derived; keep
 * them in step with the upstream rather than hand-editing hexes.
 */

export interface PaletteColors {
  readonly background: string
  readonly foreground: string
  readonly card: string
  readonly muted: string
  readonly mutedForeground: string
  readonly popover: string
  readonly primary: string
  readonly primaryForeground: string
  readonly accent: string
  readonly border: string
  readonly destructive: string
  readonly sidebarBackground?: string
  readonly userBubble?: string
}

export type Palette =
  | { readonly name: string; readonly label: string; readonly description: string; readonly light: PaletteColors; readonly dark: PaletteColors }
  | { readonly name: string; readonly label: string; readonly description: string; readonly only: PaletteColors }

export const PALETTES: readonly Palette[] = [
  { name: 'nous', label: 'Nous', description: 'GitHub chrome, Nous blue accent', light: { background: '#ffffff', foreground: '#1f2328', card: '#f6f8fa', muted: '#f6f6f6', mutedForeground: '#656d76', popover: '#ffffff', primary: '#0053fd', primaryForeground: '#ffffff', accent: '#e3edff', border: '#d0d7de', destructive: '#cf222e', sidebarBackground: '#f6f8fa', userBubble: '#dae7fd' }, dark: { background: '#0d1117', foreground: '#e6edf3', card: '#010409', muted: '#1a1e24', mutedForeground: '#7d8590', popover: '#161b22', primary: '#4a84fe', primaryForeground: '#161616', accent: '#17243a', border: '#30363d', destructive: '#f85149', sidebarBackground: '#010409', userBubble: '#07162c' } },
  { name: 'github', label: 'GitHub', description: 'GitHub Light Default and Dark Default', light: { background: '#ffffff', foreground: '#1f2328', card: '#f6f8fa', muted: '#f6f6f6', mutedForeground: '#656d76', popover: '#ffffff', primary: '#196d31', primaryForeground: '#ffffff', accent: '#e3ede6', border: '#d0d7de', destructive: '#cf222e', sidebarBackground: '#f6f8fa', userBubble: '#dbe7e2' }, dark: { background: '#0d1117', foreground: '#e6edf3', card: '#010409', muted: '#1a1e24', mutedForeground: '#7d8590', popover: '#161b22', primary: '#4f9e5e', primaryForeground: '#ffffff', accent: '#192a24', border: '#30363d', destructive: '#f85149', sidebarBackground: '#010409', userBubble: '#0f2018' } },
  { name: 'catppuccin', label: 'Catppuccin', description: 'Soothing pastels — Latte and Mocha', light: { background: '#eff1f5', foreground: '#4c4f69', card: '#e6e9ef', muted: '#e8ebef', mutedForeground: '#4c4f69', popover: '#e6e9ef', primary: '#6d2ebf', primaryForeground: '#ffffff', accent: '#dfdaef', border: '#acb0be', destructive: '#d20f39', sidebarBackground: '#e6e9ef', userBubble: '#d7d3e9' }, dark: { background: '#1e1e2e', foreground: '#cdd6f4', card: '#181825', muted: '#29293a', mutedForeground: '#cdd6f4', popover: '#181825', primary: '#cba6f7', primaryForeground: '#ffffff', accent: '#3d3652', border: '#585b70', destructive: '#f38ba8', sidebarBackground: '#181825', userBubble: '#38324b' } },
  { name: 'everforest', label: 'Everforest', description: 'Warm, low-contrast forest greens', light: { background: '#fdf6e3', foreground: '#5c6a72', card: '#fdf6e3', muted: '#f7f0de', mutedForeground: '#939f91', popover: '#fdf6e3', primary: '#586b35', primaryForeground: '#ffffff', accent: '#e9e5ce', border: '#fdf6e3', destructive: '#f1706f', sidebarBackground: '#fdf6e3', userBubble: '#e9e5ce' }, dark: { background: '#2d353b', foreground: '#d3c6aa', card: '#2d353b', muted: '#373e42', mutedForeground: '#859289', popover: '#2d353b', primary: '#a7c080', primaryForeground: '#ffffff', accent: '#434e47', border: '#2d353b', destructive: '#da6362', sidebarBackground: '#2d353b', userBubble: '#434e47' } },
  { name: 'solarized', label: 'Solarized', description: 'Fixed-contrast light and dark', light: { background: '#fdf6e3', foreground: '#1f1f1f', card: '#d3cbb7', muted: '#f4eddb', mutedForeground: '#9ca8a6', popover: '#eee8d5', primary: '#675e34', primaryForeground: '#ffffff', accent: '#ebe4ce', border: '#ddd6c1', destructive: '#e25563', sidebarBackground: '#eee8d5', userBubble: '#c6bea7' }, dark: { background: '#002b36', foreground: '#839496', card: '#002b36', muted: '#08313c', mutedForeground: '#586e75', popover: '#001f26', primary: '#6ea1c4', primaryForeground: '#ffffff', accent: '#144050', border: '#234751', destructive: '#e35957', sidebarBackground: '#001f26', userBubble: '#144050' } },
  { name: 'nous-alt', label: 'Nous Alt', description: 'Glass neutrals, cream on mission-blue', light: { background: '#F8FAFF', foreground: '#17171A', card: '#FFFFFF', muted: 'color-mix(in srgb, #0053FD 5%, #FFFFFF)', mutedForeground: '#666678', popover: '#FFFFFF', primary: '#0053FD', primaryForeground: '#FCFCFC', accent: 'color-mix(in srgb, #0053FD 10%, #FFFFFF)', border: 'color-mix(in srgb, #0053FD 22%, transparent)', destructive: '#C72E4D', sidebarBackground: '#F3F7FF', userBubble: 'color-mix(in srgb, #0053FD 6%, #FFFFFF)' }, dark: { background: '#0D2F86', foreground: '#FFE6CB', card: '#12378F', muted: '#183F9A', mutedForeground: '#B5C7F3', popover: '#123A96', primary: '#FFE6CB', primaryForeground: '#0D2F86', accent: '#1540B1', border: '#3158AD', destructive: '#C0473A', sidebarBackground: '#09286F', userBubble: '#143B91' } },
  { name: 'midnight', label: 'Midnight', description: 'Deep blue-violet with cool accents', only: { background: '#08081c', foreground: '#ddd6ff', card: '#0d0d28', muted: '#13133a', mutedForeground: '#7c7ab0', popover: '#0f0f2e', primary: '#ddd6ff', primaryForeground: '#08081c', accent: '#1a1a44', border: '#1e1e52', destructive: '#b03060', sidebarBackground: '#06061a', userBubble: '#14143a' } },
  { name: 'ember', label: 'Ember', description: 'Warm crimson and bronze — forge vibes', only: { background: '#160800', foreground: '#ffd8b0', card: '#1e0e04', muted: '#2a1408', mutedForeground: '#aa7a56', popover: '#221008', primary: '#ffd8b0', primaryForeground: '#160800', accent: '#301600', border: '#3a1c08', destructive: '#c43010', sidebarBackground: '#100600', userBubble: '#2a1000' } },
  { name: 'mono', label: 'Mono', description: 'Clean grayscale — minimal and focused', only: { background: '#0e0e0e', foreground: '#eaeaea', card: '#141414', muted: '#1e1e1e', mutedForeground: '#808080', popover: '#181818', primary: '#eaeaea', primaryForeground: '#0e0e0e', accent: '#222222', border: '#2a2a2a', destructive: '#a84040', sidebarBackground: '#0a0a0a', userBubble: '#1a1a1a' } },
  { name: 'slate', label: 'Slate', description: 'Cool slate blue — focused developer theme', only: { background: '#0d1117', foreground: '#c9d1d9', card: '#161b22', muted: '#21262d', mutedForeground: '#8b949e', popover: '#1c2128', primary: '#c9d1d9', primaryForeground: '#0d1117', accent: '#1e2530', border: '#30363d', destructive: '#cf4848', sidebarBackground: '#090d13', userBubble: '#1e2a38' } },
  { name: 'cyberpunk', label: 'Cyberpunk', description: 'Neon green on black — matrix terminal', only: { background: '#000a00', foreground: '#00ff41', card: '#001200', muted: '#001a00', mutedForeground: '#1a8a30', popover: '#001000', primary: '#00ff41', primaryForeground: '#000a00', accent: '#002000', border: '#003000', destructive: '#ff003c', sidebarBackground: '#000600', userBubble: '#001400' } },
]

export const PALETTE_KEY = 'xerxes.desktop.palette.v1'

/** A fresh install's colour theme. */
export const DEFAULT_PALETTE = 'nous'
/** Stored when the user picks Xerxes's own colours, so the default does not return. */
const XERXES_COLOURS = 'xerxes'

/** The stored palette, or undefined for Xerxes's own colours. */
export function loadPalette(): Palette | undefined {
  let name: string | null
  try { name = localStorage.getItem(PALETTE_KEY) } catch { name = null }
  if (name === XERXES_COLOURS) return undefined
  return PALETTES.find(palette => palette.name === (name ?? DEFAULT_PALETTE)) ?? PALETTES.find(palette => palette.name === DEFAULT_PALETTE)
}

/** Remember a palette; undefined means Xerxes's own colours (kept explicitly). */
export function savePalette(name: string | undefined): void {
  try { localStorage.setItem(PALETTE_KEY, name ?? XERXES_COLOURS) } catch { /* The choice still applies to this window. */ }
}

/** Relative luminance of a #rgb/#rrggbb colour; undefined for anything else. */
function luminanceOf(color: string): number | undefined {
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(color.trim())?.[1]
  if (!hex) return undefined
  const full = hex.length === 3 ? [...hex].map(c => c + c).join('') : hex
  const [r, g, b] = [0, 2, 4].map(i => parseInt(full.slice(i, i + 2), 16) / 255).map(c => c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!
}

/** Black or white, whichever reads better on `fill`; `fallback` for a non-hex fill. */
export function inkOn(fill: string, fallback: string): string {
  const l = luminanceOf(fill)
  if (l === undefined) return fallback
  const onBlack = (l + 0.05) / 0.05, onWhite = 1.05 / (l + 0.05)
  return onBlack >= onWhite ? '#0b0d11' : '#ffffff'
}

/** The variant to paint and whether it is light or dark. */
export function paletteVariant(palette: Palette, wanted: 'light' | 'dark'): { colors: PaletteColors; mode: 'light' | 'dark' } {
  if ('only' in palette) return { colors: palette.only, mode: (luminanceOf(palette.only.background) ?? 0) > 0.4 ? 'light' : 'dark' }
  return { colors: palette[wanted], mode: wanted }
}

/** `color` at `opacity` (0–1), for any CSS colour including color-mix(). */
function translucent(color: string, opacity: number): string {
  if (opacity >= 1) return color
  return `color-mix(in srgb, ${color} ${Math.round(opacity * 100)}%, transparent)`
}

/**
 * Every colour token a palette drives. `levels` are the background's
 * opacities; without them (no background) every surface is solid.
 */
export function paletteTokens(colors: PaletteColors, levels?: { readonly chat: number; readonly panels: number }): Record<string, string> {
  const chat = levels?.chat ?? 1
  const panels = levels?.panels ?? 1
  const cards = levels ? Math.min(1, panels + 0.08) : 1
  const nav = colors.sidebarBackground ?? colors.card
  return {
    '--x-screen': translucent(colors.background, chat),
    '--x-backdrop': translucent(colors.background, panels),
    '--x-sunken': translucent(colors.muted, Math.max(panels, 0.6)),
    '--x-chrome': translucent(nav, cards),
    '--mac-toolbar': translucent(nav, panels),
    '--glass-navigation': translucent(nav, panels),
    '--x-card': translucent(colors.card, cards),
    '--x-chip': translucent(colors.muted, cards),
    '--x-selected': translucent(colors.accent, Math.max(cards, 0.9)),
    '--mac-control': translucent(colors.muted, Math.max(cards, 0.9)),
    '--glass-overlay': translucent(colors.popover, 0.96),
    '--x-popover': colors.popover,
    '--x-seam': colors.border,
    '--x-hairline': translucent(colors.border, 0.85),
    '--x-divider': translucent(colors.border, 0.5),
    '--x-separator': colors.border,
    '--x-title': colors.foreground,
    '--x-strong': colors.foreground,
    '--x-prose': colors.foreground,
    '--x-secondary': `color-mix(in srgb, ${colors.foreground} 70%, ${colors.mutedForeground})`,
    '--x-meta': colors.mutedForeground,
    '--x-caption': colors.mutedForeground,
    '--x-numeric': colors.mutedForeground,
    '--x-accent': colors.primary,
    '--x-focus': colors.primary,
    '--x-working': colors.primary,
    '--x-accent-soft': translucent(colors.primary, 0.12),
    // Opaque ink on filled buttons, never a translucent surface. Picked by
    // contrast: some themes' own (white on Mocha's pale lavender) is unreadable.
    '--x-on-accent': inkOn(colors.primary, colors.primaryForeground),
    '--x-inverse': inkOn(colors.foreground, colors.background),
    // "Working" and the subagents chip have their own tokens (purple and a
    // fixed blue); a theme owns them too, or they stay off-palette.
    '--x-activity': colors.primary,
    '--x-fleet-blue': colors.primary,
    '--x-failed': colors.destructive,
    ...(colors.userBubble ? { '--x-user-bubble': colors.userBubble } : {}),
  }
}
