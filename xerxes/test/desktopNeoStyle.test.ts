// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { APPEARANCE_KEY, loadAppearance } from '../src/desktop/renderer/appearance.js'
import { chooseStyle } from '../src/desktop/renderer/appearanceSync.js'
import { applyBackdrop, contrast, parseHex } from '../src/desktop/renderer/backdrop.js'
import { PALETTES } from '../src/desktop/renderer/palettes.js'

const NEO = new URL('../src/desktop/renderer/neo.css', import.meta.url)

/** Top-level selectors of a stylesheet (comments stripped, @media bodies included). */
function selectorsOf(css: string): string[] {
  const clean = css.replace(/\/\*[\s\S]*?\*\//g, '')
  const out: string[] = []
  for (const match of clean.matchAll(/([^{}]+)\{/g)) {
    const head = match[1]!.trim()
    if (!head || head.startsWith('@') || /^(\d+%|from|to)(\s*,\s*(\d+%|from|to))*$/.test(head)) continue
    out.push(...head.split(/,(?![^()]*\))/).map(part => part.trim()).filter(Boolean))
  }
  return out
}

function block(css: string, selector: string): Record<string, string> {
  const start = css.indexOf(`${selector}{`)
  if (start < 0) throw new Error(`missing block ${selector}`)
  const body = css.slice(start + selector.length + 1, css.indexOf('}', start))
  const tokens: Record<string, string> = {}
  for (const declaration of body.replace(/\/\*[\s\S]*?\*\//g, '').split(';')) {
    const [name, ...value] = declaration.split(':')
    if (name?.trim().startsWith('--')) tokens[name.trim()] = value.join(':').trim()
  }
  return tokens
}

test('every Neo rule is scoped to data-style="neo", so Classic is untouched', async () => {
  const css = await Bun.file(NEO).text()
  const selectors = selectorsOf(css)
  expect(selectors.length).toBeGreaterThan(40)
  for (const selector of selectors) {
    expect(selector).toStartWith(":root[data-style='neo']")
    expect(selector).toContain(':not(#neo)')
  }
  // Keyframes are global names; Neo's never replace a Classic one.
  for (const match of css.matchAll(/@keyframes\s+([\w-]+)/g)) expect(match[1]).toStartWith('neo-')
})

test('Neo loads last, in the app, the build and the layout preview', async () => {
  const [html, build, preview] = await Promise.all([
    Bun.file(new URL('../src/desktop/renderer/index.html', import.meta.url)).text(),
    Bun.file(new URL('../scripts/buildDesktop.ts', import.meta.url)).text(),
    Bun.file(new URL('../scripts/previewDesktop.ts', import.meta.url)).text(),
  ])
  const sheets = [...html.matchAll(/href="\.\/([\w.-]+\.css)"/g)].map(match => match[1])
  expect(sheets.at(-1)).toBe('neo.css')
  expect(build).toContain("'neo.css'")
  expect(preview).toContain("'neo.css'")
})

test('Neo text clears WCAG AA on every surface, in dark and light', async () => {
  const css = await Bun.file(NEO).text()
  for (const selector of [":root[data-style='neo']:not(#neo)", ":root[data-style='neo'][data-theme='light']:not(#neo)"]) {
    const tokens = block(css, selector)
    const rgb = (name: string) => {
      const value = parseHex(tokens[name]!)
      if (!value) throw new Error(`${selector} ${name} is not a solid hex: ${tokens[name]}`)
      return value
    }
    for (const surface of ['--x-screen', '--x-card', '--x-chrome', '--x-selected', '--x-popover']) {
      for (const text of ['--x-title', '--x-prose', '--x-secondary', '--x-meta', '--x-numeric', '--x-accent', '--x-done', '--x-failed', '--x-needs']) {
        const ratio = contrast(rgb(text), rgb(surface))
        if (ratio < 4.5) throw new Error(`${selector}: ${text} on ${surface} is ${ratio.toFixed(2)}:1`)
      }
    }
    expect(contrast(rgb('--x-on-accent'), rgb('--x-accent'))).toBeGreaterThanOrEqual(4.5)
  }
})

function fakeRoot() {
  const attributes = new Map<string, string>()
  const styles = new Map<string, string>()
  return {
    attributes, styles,
    getAttribute: (k: string) => attributes.get(k) ?? null,
    hasAttribute: (k: string) => attributes.has(k),
    setAttribute: (k: string, v: string) => { attributes.set(k, v) },
    removeAttribute: (k: string) => { attributes.delete(k) },
    style: { setProperty: (k: string, v: string) => { styles.set(k, v) }, removeProperty: (k: string) => { styles.delete(k) } },
  }
}

/** A minimal page: storage, window.matchMedia and a document the background layers can touch. */
function withPage<T>(run: (root: ReturnType<typeof fakeRoot>, store: Map<string, string>) => T): T {
  const store = new Map<string, string>()
  const root = fakeRoot()
  const saved = { localStorage: globalThis.localStorage, window: (globalThis as { window?: unknown }).window, document: (globalThis as { document?: unknown }).document, raf: globalThis.requestAnimationFrame, Image: (globalThis as { Image?: unknown }).Image }
  globalThis.localStorage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => { store.set(k, v) }, removeItem: (k: string) => { store.delete(k) } } as unknown as Storage
  Object.assign(globalThis, {
    window: { matchMedia: () => ({ matches: false }) },
    document: { documentElement: root },
    requestAnimationFrame: () => 0,
    // Classic's image path measures the picture; nothing loads here.
    Image: class { onload: (() => void) | null = null; src = '' },
  })
  try { return run(root, store) } finally {
    globalThis.localStorage = saved.localStorage
    Object.assign(globalThis, { window: saved.window, document: saved.document, requestAnimationFrame: saved.raf, Image: saved.Image })
  }
}

/** Inline tokens that would show a picture, glass or blur. Neo must never paint any of them. */
const PICTURE_TOKENS = ['--x-backdrop-image', '--x-backdrop-gradient', '--x-backdrop-paint', '--glass-blur', '--x-chat-blur', '--x-chat-r', '--x-panel-r', '--glass-edge']

test('in Neo a background lends only its colours: solid surfaces, its accent, never the picture', () => {
  withPage((root, store) => {
    const image = { kind: 'image', src: 'backgrounds/background-1.jpg', from: '#0b1f4d', to: '#3b1a5a', accent: '#ff8800', chat: 20, panels: 10, blur: 90 }
    store.set('xerxes.desktop.backdrop.v1', JSON.stringify(image))
    store.set(APPEARANCE_KEY, JSON.stringify({ theme: 'system', font: '13' }))
    const page = root as unknown as HTMLElement

    applyBackdrop(image as never, page, undefined)
    expect(root.attributes.get('data-backdrop')).toBe('image')
    const classicAccent = root.styles.get('--x-accent')

    chooseStyle('neo', page)
    expect(loadAppearance()).toEqual({ theme: 'system', font: '13', style: 'neo' })
    expect(root.attributes.get('data-style')).toBe('neo')
    // No picture: the attribute every background rule keys off is gone.
    expect(root.attributes.has('data-backdrop')).toBe(false)
    expect(root.attributes.get('data-backdrop-colors')).toBe('image')
    for (const name of PICTURE_TOKENS) expect(root.styles.has(name)).toBe(false)
    // The colours stay: the same accent, and surfaces fully solid despite the 20%/10% sliders.
    expect(root.styles.get('--x-accent')).toBe(classicAccent!)
    for (const name of ['--x-screen', '--x-card', '--x-chrome', '--glass-navigation']) expect(root.styles.get(name)).toMatch(/^#[0-9a-f]{6}ff$/i)
    expect(root.styles.get('--glass-overlay')).toBe(root.styles.get('--x-popover')!)
    // The stored choice is untouched, so Classic shows the picture again.
    expect(JSON.parse(store.get('xerxes.desktop.backdrop.v1')!)).toEqual(image)

    chooseStyle('classic', page)
    expect(root.attributes.has('data-style')).toBe(false)
    expect(root.attributes.has('data-backdrop-colors')).toBe(false)
    expect(root.attributes.get('data-backdrop')).toBe('image')
    expect(root.styles.get('--x-backdrop-image')).toContain('background-1.jpg')
  })
})

test('a colour theme applies in Neo, over a background\'s colours, and the transparent window adds nothing', () => {
  withPage((root, store) => {
    const page = root as unknown as HTMLElement
    const catppuccin = PALETTES.find(p => p.name === 'catppuccin')!
    store.set('xerxes.desktop.palette.v1', 'catppuccin')
    root.setAttribute('data-style', 'neo')
    root.setAttribute('data-user-theme', 'dark')

    applyBackdrop({ kind: 'none' }, page, catppuccin)
    expect(root.attributes.get('data-palette')).toBe('catppuccin')
    expect(root.styles.get('--x-accent')).toBe('dark' in catppuccin ? catppuccin.dark.primary : '')

    applyBackdrop({ kind: 'gradient', from: '#0b1f4d', to: '#3b1a5a', angle: 135 }, page, catppuccin)
    expect(root.styles.get('--x-accent')).toBe('dark' in catppuccin ? catppuccin.dark.primary : '')
    for (const name of PICTURE_TOKENS) expect(root.styles.has(name)).toBe(false)

    // An explicit `undefined` palette falls back to the stored one, so clear it.
    store.delete('xerxes.desktop.palette.v1')
    applyBackdrop({ kind: 'transparent' }, page, undefined)
    expect(root.attributes.has('data-backdrop')).toBe(false)
    expect(root.attributes.has('data-backdrop-colors')).toBe(false)
    expect(root.styles.size).toBe(0)
  })
})

test('Settings tells a Neo user that a background shows only its colours', async () => {
  const { neoBackdropNote } = await import('../src/desktop/renderer/backdrop.js')
  expect(neoBackdropNote('image')).toContain('picture is not shown')
  expect(neoBackdropNote('image')).toContain('Only its colours are used')
  expect(neoBackdropNote('gradient')).toContain('Only its colours are used')
  expect(neoBackdropNote('transparent')).toContain('no effect')
  expect(neoBackdropNote('none')).toContain('never shows a background')
})

test('Neo still follows the Theme setting for light and dark', () => {
  withPage(root => {
    const page = root as unknown as HTMLElement
    root.setAttribute('data-style', 'neo')
    root.setAttribute('data-user-theme', 'light')
    applyBackdrop({ kind: 'none' }, page, undefined)
    expect(root.attributes.get('data-theme')).toBe('light')
    root.setAttribute('data-user-theme', 'dark')
    applyBackdrop({ kind: 'none' }, page, undefined)
    expect(root.attributes.get('data-theme')).toBe('dark')
  })
})
