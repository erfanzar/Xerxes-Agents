// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { BACKDROP_PRESETS, contrast, deriveTheme, dominantVivid, ensureContrast, parseBackdrop, parseHex } from '../src/desktop/renderer/backdrop.js'
import { coverRect } from '../src/desktop/renderer/backdropLayers.js'

test('light or dark follows the background itself, and the accent always clears AA on the conversation', () => {
  for (const { name, backdrop } of BACKDROP_PRESETS) {
    if (backdrop.kind === 'none' || backdrop.kind === 'transparent') continue
    const derived = deriveTheme(backdrop)!
    expect(derived.mode).toBe(['Dawn', 'Sea glass'].includes(name) ? 'light' : 'dark')
    const screen = parseHex(derived.tokens['--x-screen']!.slice(0, 7))!
    expect(contrast(parseHex(derived.tokens['--x-accent']!)!, screen)).toBeGreaterThanOrEqual(4.5)
    // Surfaces stay translucent so the background shows through.
    expect(derived.tokens['--x-screen']!).toMatch(/^#[0-9a-f]{8}$/)
  }
})

test('the accent keeps its hue: a deep blue stays blue rather than washing to grey', () => {
  const lifted = ensureContrast([11, 31, 77], [20, 20, 30], 4.5)
  expect(lifted[2]).toBeGreaterThan(lifted[0] + 60)
  expect(contrast(lifted, [20, 20, 30])).toBeGreaterThanOrEqual(4.5)
})

test('an image accent is its dominant vivid hue, never a blend of two hues', () => {
  const orange = Array.from({ length: 40 }, () => [255, 176, 0] as const)
  const grey = Array.from({ length: 400 }, () => [60, 60, 62] as const)
  const [r, g, b] = dominantVivid([...grey, ...orange])
  expect([Math.round(r), Math.round(g), Math.round(b)]).toEqual([255, 176, 0])
})

test('stored backgrounds are validated: bad colours or non-image data fall back to none', () => {
  expect(parseBackdrop({ kind: 'gradient', from: '#123', to: '#abcdef', angle: 90 })).toEqual({ kind: 'gradient', from: '#123', to: '#abcdef', angle: 90 })
  expect(parseBackdrop({ kind: 'gradient', from: 'red', to: '#abcdef' })).toEqual({ kind: 'none' })
  expect(parseBackdrop({ kind: 'image', src: 'https://evil.example/x.png', from: '#000', to: '#fff' })).toEqual({ kind: 'none' })
  expect(parseBackdrop({ kind: 'image', src: 'data:image/jpeg;base64,AAAA', from: '#000000', to: '#ffffff', accent: '#ff0000' })).toMatchObject({ kind: 'image', accent: '#ff0000' })
  expect(parseBackdrop(null)).toEqual({ kind: 'none' })
})

test('the conversation is more see-through than the side panels by default, and each level is tunable', async () => {
  const { DEFAULT_CHAT_OPACITY, DEFAULT_PANEL_OPACITY } = await import('../src/desktop/renderer/backdrop.js')
  const alphaOf = (hex: string) => parseInt(hex.slice(7, 9), 16) / 255
  const base = { kind: 'gradient' as const, from: '#0b1f4d', to: '#3b1a5a', angle: 135 }
  const stock = deriveTheme(base)!.tokens
  expect(DEFAULT_CHAT_OPACITY).toBeLessThan(DEFAULT_PANEL_OPACITY)
  expect(alphaOf(stock['--x-screen']!)).toBeLessThan(alphaOf(stock['--glass-navigation']!))
  const tuned = deriveTheme({ ...base, chat: 10, panels: 95 })!.tokens
  expect(alphaOf(tuned['--x-screen']!)).toBeCloseTo(0.1, 1)
  expect(alphaOf(tuned['--glass-navigation']!)).toBeCloseTo(0.95, 1)
  // Blur is its own lever, independent of how solid a surface is.
  expect(tuned['--x-chat-blur']).toBe(stock['--x-chat-blur'])
  expect(deriveTheme({ ...base, blur: 0 })!.tokens['--x-chat-blur']).toBe('none')
  expect(deriveTheme({ ...base, blur: 0 })!.tokens['--glass-blur']).toBe('none')
  // The default is macOS's own glass: 30px, colour kept at 180% saturation.
  expect(stock['--glass-blur']).toBe('blur(30px) saturate(180%)')
  expect(deriveTheme({ ...base, blur: 100 })!.tokens['--x-chat-blur']).toBe('blur(40px) saturate(180%)')
  expect(deriveTheme({ ...base, chat: 0 })!.tokens['--x-screen']!.slice(7)).toBe('00')
  expect(parseBackdrop({ ...base, chat: 250, panels: -5, blur: 140 })).toMatchObject({ chat: 100, panels: 0, blur: 100 })
})

test('every bundled background ships in the repo, is stored by path, and derives a readable theme', async () => {
  const { BUNDLED_BACKGROUNDS } = await import('../src/desktop/renderer/backdrop.js')
  expect(BUNDLED_BACKGROUNDS).toHaveLength(6)
  for (const { backdrop } of BUNDLED_BACKGROUNDS) {
    expect(await Bun.file(new URL(`../src/desktop/renderer/${backdrop.src}`, import.meta.url)).exists()).toBe(true)
    expect(parseBackdrop(backdrop)).toEqual(backdrop)
    const derived = deriveTheme(backdrop)!
    expect(contrast(parseHex(derived.tokens['--x-accent']!)!, parseHex(derived.tokens['--x-screen']!.slice(0, 7))!)).toBeGreaterThanOrEqual(4.5)
  }
  // Only the bundled folder is trusted by path.
  expect(parseBackdrop({ kind: 'image', src: '../../etc/passwd', from: '#000', to: '#fff' })).toEqual({ kind: 'none' })
})

test('a fresh install opens with no background at 70% / 75% levels; choosing None is remembered', async () => {
  const { loadBackdrop, saveBackdrop, BACKDROP_KEY, DEFAULT_CHAT_OPACITY, DEFAULT_PANEL_OPACITY } = await import('../src/desktop/renderer/backdrop.js')
  const store = new Map<string, string>()
  const original = globalThis.localStorage
  globalThis.localStorage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => { store.set(k, v) }, removeItem: (k: string) => { store.delete(k) } } as unknown as Storage
  try {
    expect([DEFAULT_CHAT_OPACITY, DEFAULT_PANEL_OPACITY]).toEqual([70, 75])
    expect(loadBackdrop()).toEqual({ kind: 'none' })
    saveBackdrop({ kind: 'none' })
    expect(store.get(BACKDROP_KEY)).toBe('{"kind":"none"}')
    expect(loadBackdrop()).toEqual({ kind: 'none' })
  } finally { globalThis.localStorage = original }
})

test('a surface\'s static blur layer lines up with the window background under cover', () => {
  // A wide image in a tall window: height fills, width overflows evenly both sides.
  expect(coverRect({ width: 2000, height: 1000 }, { width: 800, height: 600 })).toEqual({ x: -200, y: 0, width: 1200, height: 600 })
  // A tall image in a wide window: width fills, height overflows evenly.
  expect(coverRect({ width: 1000, height: 2000 }, { width: 1000, height: 500 })).toEqual({ x: 0, y: -750, width: 1000, height: 2000 })
})

test('persistent surfaces never use a live backdrop-filter over a background', async () => {
  // Live blur re-rasterized the whole surface on every repaint inside it
  // (spinners, streamed tokens) and kept the GPU busy for the whole turn.
  const css = await Bun.file(new URL('../src/desktop/renderer/atelier.css', import.meta.url)).text()
  expect(css).toContain(':root[data-backdrop] .atelier :is(.top,.side,.chat,.desktop-rail){backdrop-filter:none;')
  expect(css).not.toContain(':root[data-backdrop] .atelier .chat{backdrop-filter:var(')
  // The static layers overhang by the blur radius, which the Blur level sets.
  const tokens = deriveTheme({ kind: 'gradient', from: '#101820', to: '#203040', angle: 160, chat: 80, panels: 85, blur: 80 })!.tokens
  expect(tokens['--x-chat-r']).toBe('32px')
  expect(tokens['--x-panel-r']).toBe('32px')
})

test('a transparent background round-trips through storage with its levels', async () => {
  const { parseBackdrop } = await import('../src/desktop/renderer/backdrop.js')
  expect(parseBackdrop({ kind: 'transparent', chat: 40, panels: 90 })).toEqual({ kind: 'transparent', chat: 40, panels: 90 })
  expect(parseBackdrop({ kind: 'transparent', chat: 400 })).toEqual({ kind: 'transparent', chat: 100 })
  // Clear glass (system blur off) is remembered; anything else means on.
  expect(parseBackdrop({ kind: 'transparent', systemBlur: false, panels: 20 })).toEqual({ kind: 'transparent', systemBlur: false, panels: 20 })
  expect(parseBackdrop({ kind: 'transparent', systemBlur: 'no' })).toEqual({ kind: 'transparent' })
})
