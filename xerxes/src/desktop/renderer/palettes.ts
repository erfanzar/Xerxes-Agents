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
 * `toxic-forest`, `electric-night`, `afterglow` and `mahogany` are built from palette cards the user
 * supplied: the background is the card's darkest swatch (or a deeper shade of it when that swatch is too
 * bright to carry text, as in Electric Night), its brightest readable swatch is the accent, and
 * the other surfaces are steps of the background. Swatches that fail AA as text or fill are not used verbatim.
 * Except those and `claude` (Xerxes's own), the palettes are Hermes Agent's desktop themes (MIT, Copyright (c) 2025
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
  // T3 Code built-in themes (matching pingdotgg/t3code themePalettes.ts)
  { name: 't3-code', label: 'T3 Code', description: 'Pure obsidian black, crisp light paper, and T3 cobalt blue', light: { background: '#fcfcfc', foreground: '#18181b', card: '#ffffff', muted: '#f4f4f5', mutedForeground: '#71717b', popover: '#ffffff', primary: '#1b4ed8', primaryForeground: '#ffffff', accent: '#f4f4f5', border: '#e4e4e7', destructive: '#ef4444', sidebarBackground: '#fafafa', userBubble: '#f4f4f5' }, dark: { background: '#000000', foreground: '#f5f5f5', card: '#111111', muted: '#141414', mutedForeground: '#818181', popover: '#111111', primary: '#346bf1', primaryForeground: '#ffffff', accent: '#1a1b1b', border: '#242424', destructive: '#fb414a', sidebarBackground: '#000000', userBubble: '#141414' } },
  { name: 't3-chat', label: 'T3 Chat', description: 'Soft rose vellum and deep plum ink with magenta accents', light: { background: '#fcf6fa', foreground: '#221426', card: '#f7ecf3', muted: '#f4e6f0', mutedForeground: '#765f75', popover: '#f5e8f1', primary: '#ca0277', primaryForeground: '#fffbfe', accent: '#ebd7e6', border: '#dfc8d8', destructive: '#e12d39', sidebarBackground: '#f4e6f0', userBubble: '#eed9e8' }, dark: { background: '#1c151d', foreground: '#fdfafd', card: '#281f29', muted: '#2e212e', mutedForeground: '#9e8c9d', popover: '#312533', primary: '#dc749e', primaryForeground: '#1c151d', accent: '#3e293d', border: '#4c384b', destructive: '#f05252', sidebarBackground: '#241a25', userBubble: '#382538' } },
  { name: 'grove', label: 'Grove', description: 'Botanical stone and deep pine greens', light: { background: '#f6f8f6', foreground: '#1e2420', card: '#eef2ef', muted: '#eaeff0', mutedForeground: '#5f6b63', popover: '#e9efea', primary: '#2d7a57', primaryForeground: '#ffffff', accent: '#dce8e0', border: '#ccd8cf', destructive: '#dc2626', sidebarBackground: '#edf2ee', userBubble: '#dfebe3' }, dark: { background: '#17231e', foreground: '#f6faf7', card: '#22312a', muted: '#23352c', mutedForeground: '#8ea396', popover: '#293b33', primary: '#63d297', primaryForeground: '#14211b', accent: '#2f493c', border: '#3c5a4b', destructive: '#ef4444', sidebarBackground: '#1d2b24', userBubble: '#2d473a' } },
  { name: 'ocean', label: 'Ocean', description: 'Misty slate and abyss marine blue', light: { background: '#f5f8fc', foreground: '#1b2230', card: '#edf2f8', muted: '#eaeff7', mutedForeground: '#5e6b7e', popover: '#e7eef6', primary: '#2563ab', primaryForeground: '#ffffff', accent: '#dce6f3', border: '#cad6e6', destructive: '#dc2626', sidebarBackground: '#ecf1f8', userBubble: '#dfeaf7' }, dark: { background: '#131d2b', foreground: '#f6f9fd', card: '#1e2c40', muted: '#1f3047', mutedForeground: '#8d9db5', popover: '#25364e', primary: '#69a9f5', primaryForeground: '#111a28', accent: '#29405e', border: '#385276', destructive: '#ef4444', sidebarBackground: '#182436', userBubble: '#263e5c' } },
  { name: 't3-ember', label: 'T3 Ember', description: 'Warm parchment and roasted copper with amber accents', light: { background: '#fbf7f3', foreground: '#261e1a', card: '#f4ece5', muted: '#f2e8e0', mutedForeground: '#76665d', popover: '#efe4db', primary: '#b4532a', primaryForeground: '#ffffff', accent: '#ead9cc', border: '#dcc9bb', destructive: '#dc2626', sidebarBackground: '#f4ebe3', userBubble: '#eddccf' }, dark: { background: '#221914', foreground: '#fdf9f6', card: '#31251e', muted: '#33251d', mutedForeground: '#a69287', popover: '#3b2c24', primary: '#e89b63', primaryForeground: '#211712', accent: '#473327', border: '#5c4233', destructive: '#ef4444', sidebarBackground: '#2a1f19', userBubble: '#443024' } },
  { name: 'iris', label: 'Iris', description: 'Soft lilac paper and deep amethyst violet', light: { background: '#f9f6fc', foreground: '#221b2c', card: '#f1ebf7', muted: '#eee7f6', mutedForeground: '#6b607a', popover: '#ece3f5', primary: '#6d44b8', primaryForeground: '#ffffff', accent: '#e3d8f0', border: '#d3c5e4', destructive: '#dc2626', sidebarBackground: '#f0e9f7', userBubble: '#e5dbf2' }, dark: { background: '#1a1527', foreground: '#fbf9fd', card: '#28203c', muted: '#2a2040', mutedForeground: '#998db0', popover: '#31274a', primary: '#a985f5', primaryForeground: '#181224', accent: '#3c2e5c', border: '#4e3c73', destructive: '#ef4444', sidebarBackground: '#211a33', userBubble: '#382a57' } },
  // Xerxes's own (not a Hermes theme): the look of Claude's desktop apps.
  // Dark is neutral grey, measured from the app itself — sidebar #111111,
  // conversation #151515, composer #1f1f1f, bubble #212121, selection
  // #363636 — not the warm near-black the first version guessed. Light
  // keeps Claude's warm cream. Claude orange in both.
  { name: 'claude', label: 'Claude', description: 'Claude desktop greys and Claude orange', light: { background: '#faf9f5', foreground: '#1f1e1d', card: '#ffffff', muted: '#f0eee6', mutedForeground: '#6b6a64', popover: '#ffffff', primary: '#c96442', primaryForeground: '#ffffff', accent: '#ece9df', border: '#e3e0d6', destructive: '#b53d31', sidebarBackground: '#f3f1ea', userBubble: '#efece3' }, dark: { background: '#151515', foreground: '#ececec', card: '#1f1f1f', muted: '#212121', mutedForeground: '#8f8f8f', popover: '#1f1f1f', primary: '#d97757', primaryForeground: '#151515', accent: '#363636', border: '#2f2f2f', destructive: '#e5534b', sidebarBackground: '#111111', userBubble: '#212121' } },
  { name: 'nous', label: 'Nous', description: 'GitHub chrome, Nous blue accent', light: { background: '#ffffff', foreground: '#1f2328', card: '#f6f8fa', muted: '#f6f6f6', mutedForeground: '#656d76', popover: '#ffffff', primary: '#0053fd', primaryForeground: '#ffffff', accent: '#e3edff', border: '#d0d7de', destructive: '#cf222e', sidebarBackground: '#f6f8fa', userBubble: '#dae7fd' }, dark: { background: '#0d1117', foreground: '#e6edf3', card: '#010409', muted: '#1a1e24', mutedForeground: '#7d8590', popover: '#161b22', primary: '#4a84fe', primaryForeground: '#161616', accent: '#17243a', border: '#30363d', destructive: '#f85149', sidebarBackground: '#010409', userBubble: '#07162c' } },
  { name: 'github', label: 'GitHub', description: 'GitHub Light Default and Dark Default', light: { background: '#ffffff', foreground: '#1f2328', card: '#f6f8fa', muted: '#f6f6f6', mutedForeground: '#656d76', popover: '#ffffff', primary: '#196d31', primaryForeground: '#ffffff', accent: '#e3ede6', border: '#d0d7de', destructive: '#cf222e', sidebarBackground: '#f6f8fa', userBubble: '#dbe7e2' }, dark: { background: '#0d1117', foreground: '#e6edf3', card: '#010409', muted: '#1a1e24', mutedForeground: '#7d8590', popover: '#161b22', primary: '#4f9e5e', primaryForeground: '#ffffff', accent: '#192a24', border: '#30363d', destructive: '#f85149', sidebarBackground: '#010409', userBubble: '#0f2018' } },
  { name: 'catppuccin', label: 'Catppuccin', description: 'Soothing pastels — Latte and Mocha', light: { background: '#eff1f5', foreground: '#4c4f69', card: '#e6e9ef', muted: '#e8ebef', mutedForeground: '#5a5d74', popover: '#e6e9ef', primary: '#6d2ebf', primaryForeground: '#ffffff', accent: '#dfdaef', border: '#acb0be', destructive: '#d20f39', sidebarBackground: '#e6e9ef', userBubble: '#d7d3e9' }, dark: { background: '#1e1e2e', foreground: '#cdd6f4', card: '#181825', muted: '#29293a', mutedForeground: '#a6adc8', popover: '#181825', primary: '#cba6f7', primaryForeground: '#ffffff', accent: '#3d3652', border: '#585b70', destructive: '#f38ba8', sidebarBackground: '#181825', userBubble: '#38324b' } },
  { name: 'everforest', label: 'Everforest', description: 'Warm, low-contrast forest greens', light: { background: '#fdf6e3', foreground: '#48565d', card: '#f4f0d9', muted: '#efebd4', mutedForeground: '#59665d', popover: '#f4f0d9', primary: '#586b35', primaryForeground: '#ffffff', accent: '#e9e5ce', border: '#d5c9ab', destructive: '#d94c4c', sidebarBackground: '#efebd4', userBubble: '#e9e5ce' }, dark: { background: '#2d353b', foreground: '#d3c6aa', card: '#343f44', muted: '#373e42', mutedForeground: '#b1beb4', popover: '#343f44', primary: '#a7c080', primaryForeground: '#232a2e', accent: '#434e47', border: '#4f585e', destructive: '#e67e80', sidebarBackground: '#272e33', userBubble: '#434e47' } },
  { name: 'solarized', label: 'Solarized', description: 'Fixed-contrast light and dark', light: { background: '#fdf6e3', foreground: '#1f1f1f', card: '#eee8d5', muted: '#f4eddb', mutedForeground: '#52676e', popover: '#f5efdc', primary: '#675e34', primaryForeground: '#ffffff', accent: '#ebe4ce', border: '#d5cdb6', destructive: '#dc322f', sidebarBackground: '#eee8d5', userBubble: '#e4ddc8' }, dark: { background: '#002b36', foreground: '#b9c7c7', card: '#073642', muted: '#08313c', mutedForeground: '#93a5a7', popover: '#073642', primary: '#6ea1c4', primaryForeground: '#002b36', accent: '#144050', border: '#234751', destructive: '#e35957', sidebarBackground: '#00212b', userBubble: '#144050' } },
  { name: 'nous-alt', label: 'Nous Alt', description: 'Glass neutrals, cream on mission-blue', light: { background: '#F8FAFF', foreground: '#17171A', card: '#FFFFFF', muted: 'color-mix(in srgb, #0053FD 5%, #FFFFFF)', mutedForeground: '#666678', popover: '#FFFFFF', primary: '#0053FD', primaryForeground: '#FCFCFC', accent: 'color-mix(in srgb, #0053FD 10%, #FFFFFF)', border: 'color-mix(in srgb, #0053FD 22%, transparent)', destructive: '#C72E4D', sidebarBackground: '#F3F7FF', userBubble: 'color-mix(in srgb, #0053FD 6%, #FFFFFF)' }, dark: { background: '#0D2F86', foreground: '#FFE6CB', card: '#12378F', muted: '#183F9A', mutedForeground: '#B5C7F3', popover: '#123A96', primary: '#FFE6CB', primaryForeground: '#0D2F86', accent: '#1540B1', border: '#3158AD', destructive: '#C0473A', sidebarBackground: '#09286F', userBubble: '#143B91' } },
  { name: 'midnight', label: 'Midnight', description: 'Deep blue-violet with cool accents', only: { background: '#08081c', foreground: '#ddd6ff', card: '#0d0d28', muted: '#13133a', mutedForeground: '#7c7ab0', popover: '#0f0f2e', primary: '#ddd6ff', primaryForeground: '#08081c', accent: '#1a1a44', border: '#1e1e52', destructive: '#b03060', sidebarBackground: '#06061a', userBubble: '#14143a' } },
  { name: 'ember', label: 'Ember', description: 'Warm crimson and bronze — forge vibes', only: { background: '#160800', foreground: '#ffd8b0', card: '#1e0e04', muted: '#2a1408', mutedForeground: '#aa7a56', popover: '#221008', primary: '#ffd8b0', primaryForeground: '#160800', accent: '#301600', border: '#3a1c08', destructive: '#c43010', sidebarBackground: '#100600', userBubble: '#2a1000' } },
  { name: 'mono', label: 'Mono', description: 'Clean grayscale — minimal and focused', only: { background: '#0e0e0e', foreground: '#eaeaea', card: '#141414', muted: '#1e1e1e', mutedForeground: '#808080', popover: '#181818', primary: '#eaeaea', primaryForeground: '#0e0e0e', accent: '#222222', border: '#2a2a2a', destructive: '#a84040', sidebarBackground: '#0a0a0a', userBubble: '#1a1a1a' } },
  { name: 'slate', label: 'Slate', description: 'Cool slate blue — focused developer theme', only: { background: '#0d1117', foreground: '#c9d1d9', card: '#161b22', muted: '#21262d', mutedForeground: '#8b949e', popover: '#1c2128', primary: '#c9d1d9', primaryForeground: '#0d1117', accent: '#1e2530', border: '#30363d', destructive: '#cf4848', sidebarBackground: '#090d13', userBubble: '#1e2a38' } },
  { name: 'cyberpunk', label: 'Cyberpunk', description: 'Neon green on black — matrix terminal', only: { background: '#000a00', foreground: '#00ff41', card: '#001200', muted: '#001a00', mutedForeground: '#229c3a', popover: '#001000', primary: '#00ff41', primaryForeground: '#000a00', accent: '#002000', border: '#003000', destructive: '#ff003c', sidebarBackground: '#000600', userBubble: '#001400' } },
  // From palette cards Erfan supplied (2026-11-20); see the header for how swatches map to roles.
  { name: 'toxic-forest', label: 'Toxic Forest', description: 'Deep teal with acid lime and yellow', only: { background: '#0b2027', foreground: '#eef3d6', card: '#0f2a33', muted: '#13323c', mutedForeground: '#a3b8a4', popover: '#0e2730', primary: '#eef800', primaryForeground: '#0b2027', accent: '#1f3d22', border: '#21424a', destructive: '#ff6b5c', sidebarBackground: '#081a20', userBubble: '#173a26' } },
  { name: 'electric-night', label: 'Electric Night', description: 'Ultraviolet with cyan and hot pink', only: { background: '#14083d', foreground: '#e8f7fd', card: '#1d0e57', muted: '#251266', mutedForeground: '#a6b8ec', popover: '#1b0d52', primary: '#4cc9f0', primaryForeground: '#14083d', accent: '#2e1580', border: '#3a0ca3', destructive: '#ff007f', sidebarBackground: '#100631', userBubble: '#2a1a86' } },
  { name: 'afterglow', label: 'Afterglow', description: 'Smoky black, plum and indigo, pale olive light', only: { background: '#24201a', foreground: '#ebeacb', card: '#2c2722', muted: '#352d35', mutedForeground: '#b0a3c8', popover: '#2a2520', primary: '#c8c793', primaryForeground: '#24201a', accent: '#3f2548', border: '#4a3e4e', destructive: '#e5736b', sidebarBackground: '#1c1914', userBubble: '#2d2b5e' } },
  { name: 'mahogany', label: 'Mahogany', description: 'Rich mahogany and espresso, lavender and amethyst', only: { background: '#3d1919', foreground: '#efeaf6', card: '#4a2020', muted: '#542727', mutedForeground: '#c6b2c8', popover: '#462020', primary: '#9798c5', primaryForeground: '#3d1919', accent: '#5e2c2c', border: '#6d4f48', destructive: '#ff8f8a', sidebarBackground: '#331414', userBubble: '#5a2f4f' } },
]

export const PALETTE_KEY = 'xerxes.desktop.palette.v1'

/** Stored when the user picks Xerxes's own colours — also what a fresh install uses. */
const XERXES_COLOURS = 'xerxes'

/**
 * The stored palette, or undefined for Xerxes's own colours: a fresh
 * install, an explicit Xerxes choice, and a stored name no longer shipped.
 */
export function loadPalette(): Palette | undefined {
  let name: string | null
  try { name = localStorage.getItem(PALETTE_KEY) } catch { name = null }
  if (name === null || name === XERXES_COLOURS) return undefined
  return PALETTES.find(palette => palette.name === name)
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
    '--x-dialog': colors.popover,
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
