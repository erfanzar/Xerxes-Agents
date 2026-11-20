// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { APPEARANCE_KEY, applyAppearance, applyStyle, loadAppearance, saveAppearance } from '../src/desktop/renderer/appearance.js'
import { followAppearanceChanges, SHARED_APPEARANCE_KEYS } from '../src/desktop/renderer/appearanceSync.js'
import { BACKDROP_KEY } from '../src/desktop/renderer/backdrop.js'
import { PALETTE_KEY } from '../src/desktop/renderer/palettes.js'

function withStorage<T>(run: (store: Map<string, string>) => T): T {
  const store = new Map<string, string>()
  const original = globalThis.localStorage
  globalThis.localStorage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => { store.set(k, v) }, removeItem: (k: string) => { store.delete(k) } } as unknown as Storage
  try { return run(store) } finally { globalThis.localStorage = original }
}

test('changing the text size in one view never puts back the theme another view chose', () => {
  withStorage(() => {
    saveAppearance({ theme: 'light' })
    // A second view, still showing dark, changes only the font.
    saveAppearance({ font: '15' })
    // Never chose a style: Vibes, the default.
    expect(loadAppearance()).toEqual({ theme: 'light', font: '15', style: 'vibes' })
    saveAppearance({ theme: 'system' })
    expect(loadAppearance()).toEqual({ theme: 'system', font: '15', style: 'vibes' })
    // The style is its own field: switching it keeps theme and font.
    saveAppearance({ style: 'neo' })
    expect(loadAppearance()).toEqual({ theme: 'system', font: '15', style: 'neo' })
  })
})

test('Vibes is the default, and a saved Classic or Shading opens as Vibes without changing theme or font', () => {
  withStorage(storage => {
    expect(loadAppearance().style).toBe('vibes')
    for (const retired of ['classic', 'shading']) {
      storage.set(APPEARANCE_KEY, JSON.stringify({ theme: 'light', font: '15', style: retired }))
      expect(loadAppearance()).toEqual({ theme: 'light', font: '15', style: 'vibes' })
    }
    storage.set(APPEARANCE_KEY, JSON.stringify({ theme: 'light', font: '15', style: 'neo' }))
    expect(loadAppearance()).toEqual({ theme: 'light', font: '15', style: 'neo' })
    saveAppearance({ font: '17' })
    expect(loadAppearance()).toEqual({ theme: 'light', font: '17', style: 'neo' })
  })
})

test('applying a stored choice sets the page attributes, and system hands the theme back', () => {
  const attributes = new Map<string, string>()
  const root = { setAttribute: (k: string, v: string) => { attributes.set(k, v) }, removeAttribute: (k: string) => { attributes.delete(k) } }
  applyAppearance({ theme: 'dark', font: '13', style: 'vibes' }, root)
  // Every style stamps data-style; only VS Code high contrast stamps none (applyStyle(null)).
  expect(Object.fromEntries(attributes)).toEqual({ 'data-style': 'vibes', 'data-font': '13', 'data-user-theme': 'dark', 'data-theme': 'dark' })
  applyAppearance({ theme: 'dark', font: '13', style: 'neo' }, root)
  expect(attributes.get('data-style')).toBe('neo')
  applyStyle(null, root)
  expect(attributes.has('data-style')).toBe(false)
  applyAppearance({ theme: 'system', font: '11', style: 'vibes' }, root)
  expect(attributes.get('data-style')).toBe('vibes')
  expect(attributes.has('data-user-theme')).toBe(false)
  expect(attributes.get('data-font')).toBe('11')
})

test('an open view re-applies the look when any other view changes theme, background or palette', () => {
  const listeners = new Set<(event: StorageEvent) => void>()
  const target = {
    addEventListener: (_: 'storage', listener: (event: StorageEvent) => void) => { listeners.add(listener) },
    removeEventListener: (_: 'storage', listener: (event: StorageEvent) => void) => { listeners.delete(listener) },
  }
  let applied = 0
  const stop = followAppearanceChanges(target, () => { applied++ })
  const fire = (key: string | null) => { for (const listener of listeners) listener({ key } as StorageEvent) }
  expect([...SHARED_APPEARANCE_KEYS].sort()).toEqual([APPEARANCE_KEY, BACKDROP_KEY, PALETTE_KEY].sort())
  for (const key of [APPEARANCE_KEY, BACKDROP_KEY, PALETTE_KEY]) fire(key)
  expect(applied).toBe(3)
  fire('xerxes.desktop.session.something-else')
  expect(applied).toBe(3)
  fire(null)
  expect(applied).toBe(4)
  stop()
  fire(APPEARANCE_KEY)
  expect(applied).toBe(4)
})
