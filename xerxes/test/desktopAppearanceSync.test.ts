// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { APPEARANCE_KEY, applyAppearance, loadAppearance, saveAppearance } from '../src/desktop/renderer/appearance.js'
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
    // Never chose a style: Neo, the default.
    expect(loadAppearance()).toEqual({ theme: 'light', font: '15', style: 'neo' })
    saveAppearance({ theme: 'system' })
    expect(loadAppearance()).toEqual({ theme: 'system', font: '15', style: 'neo' })
    // The style is its own field: switching it keeps theme and font.
    saveAppearance({ style: 'classic' })
    expect(loadAppearance()).toEqual({ theme: 'system', font: '15', style: 'classic' })
  })
})

test('applying a stored choice sets the page attributes, and system hands the theme back', () => {
  const attributes = new Map<string, string>()
  const root = { setAttribute: (k: string, v: string) => { attributes.set(k, v) }, removeAttribute: (k: string) => { attributes.delete(k) } }
  applyAppearance({ theme: 'dark', font: '13', style: 'classic' }, root)
  // Classic stamps no data-style, so every Classic rule sees the page it always did.
  expect(Object.fromEntries(attributes)).toEqual({ 'data-font': '13', 'data-user-theme': 'dark', 'data-theme': 'dark' })
  applyAppearance({ theme: 'dark', font: '13', style: 'neo' }, root)
  expect(attributes.get('data-style')).toBe('neo')
  applyAppearance({ theme: 'system', font: '11', style: 'classic' }, root)
  expect(attributes.has('data-style')).toBe(false)
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
