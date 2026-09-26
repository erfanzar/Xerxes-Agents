// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { parseTerminalFont, UI_SCALES } from '../src/desktop/renderer/displayPrefs.js'
import { PALETTES, paletteTokens, paletteVariant } from '../src/desktop/renderer/palettes.js'

test('every theme has the colours the app paints, in both variants it offers', () => {
  expect(PALETTES.map(palette => palette.name)).toEqual(['nous', 'github', 'catppuccin', 'everforest', 'solarized', 'nous-alt', 'midnight', 'ember', 'mono', 'slate', 'cyberpunk'])
  for (const palette of PALETTES) {
    for (const wanted of ['light', 'dark'] as const) {
      const { colors } = paletteVariant(palette, wanted)
      for (const key of ['background', 'foreground', 'card', 'muted', 'mutedForeground', 'popover', 'primary', 'accent', 'border', 'destructive'] as const) {
        expect(colors[key]).toBeTruthy()
      }
    }
  }
})

test('a single-mode theme keeps its own mode; a two-mode theme follows Light / Dark', () => {
  const cyberpunk = PALETTES.find(palette => palette.name === 'cyberpunk')!
  expect(paletteVariant(cyberpunk, 'light').mode).toBe('dark')
  const github = PALETTES.find(palette => palette.name === 'github')!
  expect(paletteVariant(github, 'light')).toMatchObject({ mode: 'light', colors: { background: '#ffffff' } })
  expect(paletteVariant(github, 'dark')).toMatchObject({ mode: 'dark', colors: { background: '#0d1117' } })
})

test('without a background a theme is solid; with one it tints the glass at the slider opacities', () => {
  const { colors } = paletteVariant(PALETTES.find(palette => palette.name === 'catppuccin')!, 'dark')
  const solid = paletteTokens(colors)
  expect(solid['--x-screen']).toBe('#1e1e2e')
  expect(solid['--x-accent']).toBe(colors.primary)
  const glass = paletteTokens(colors, { chat: 0.8, panels: 0.85 })
  expect(glass['--x-screen']).toBe('color-mix(in srgb, #1e1e2e 80%, transparent)')
  expect(glass['--glass-navigation']).toBe(`color-mix(in srgb, ${colors.sidebarBackground} 85%, transparent)`)
  // Popovers stay opaque over the conversation either way.
  expect(glass['--x-popover']).toBe(colors.popover)
})

test('the terminal font accepts font lists and refuses anything that could escape the declaration', () => {
  expect(parseTerminalFont('  MesloLGS NF, monospace ')).toBe('MesloLGS NF, monospace')
  expect(parseTerminalFont('"JetBrainsMono Nerd Font"')).toBe('"JetBrainsMono Nerd Font"')
  expect(parseTerminalFont('')).toBeUndefined()
  for (const hostile of ['x; color: red', 'x } body { display:none', 'x\ny', 'x\\', 'a'.repeat(201)]) {
    expect(() => parseTerminalFont(hostile)).toThrow()
  }
  expect(UI_SCALES).toEqual([90, 100, 110, 125, 150, 175])
})

test('a fresh install uses Nous; picking Xerxes\'s own colours is remembered, not reset to the default', async () => {
  const { loadPalette, savePalette } = await import('../src/desktop/renderer/palettes.js')
  const store = new Map<string, string>()
  const original = globalThis.localStorage
  globalThis.localStorage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => { store.set(k, v) }, removeItem: (k: string) => { store.delete(k) } } as unknown as Storage
  try {
    expect(loadPalette()?.name).toBe('nous')
    savePalette(undefined)
    expect(loadPalette()).toBeUndefined()
    savePalette('cyberpunk')
    expect(loadPalette()?.name).toBe('cyberpunk')
  } finally { globalThis.localStorage = original }
})

test('button ink is chosen for contrast on the theme accent, and is always opaque', async () => {
  const { inkOn } = await import('../src/desktop/renderer/palettes.js')
  expect(inkOn('#cba6f7', '#ffffff')).toBe('#0b0d11')   // Mocha lavender: dark ink
  expect(inkOn('#196d31', '#ffffff')).toBe('#ffffff')   // GitHub green: white ink
  expect(inkOn('color-mix(in srgb, red 50%, blue)', '#fcfcfc')).toBe('#fcfcfc')
  for (const palette of PALETTES) {
    for (const wanted of ['light', 'dark'] as const) {
      const tokens = paletteTokens(paletteVariant(palette, wanted).colors, { chat: 0.3, panels: 0.3 })
      expect(tokens['--x-on-accent']).not.toContain('transparent')
      expect(tokens['--x-inverse']).not.toContain('transparent')
    }
  }
})
