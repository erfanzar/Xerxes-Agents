// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { APPEARANCE_KEY, applyStyle, isStyleChoice, loadAppearance, parseAppearance, STYLES } from '../src/desktop/renderer/appearance.js'
import { chooseStyle } from '../src/desktop/renderer/appearanceSync.js'
import { applyBackdrop, contrast, isSolidStyle, neoBackdropNote, parseHex } from '../src/desktop/renderer/backdrop.js'

const SHADING = new URL('../src/desktop/renderer/shading.css', import.meta.url)

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

test('Shading is a third style: parsed, labelled and stamped on the page', () => {
  expect(STYLES).toEqual(['classic', 'neo', 'shading'])
  expect(isStyleChoice('shading')).toBe(true)
  expect(isStyleChoice('retro')).toBe(false)
  expect(parseAppearance({ style: 'shading' }).style).toBe('shading')
  expect(parseAppearance({ style: 'retro' }).style).toBe('neo')
  const attributes = new Map<string, string>()
  const root = { setAttribute: (k: string, v: string) => { attributes.set(k, v) }, removeAttribute: (k: string) => { attributes.delete(k) } }
  applyStyle('shading', root)
  expect(attributes.get('data-style')).toBe('shading')
  applyStyle('classic', root)
  expect(attributes.has('data-style')).toBe(false)
})

test('every Shading rule is scoped to data-style="shading", so Classic and Neo are untouched', async () => {
  const css = await Bun.file(SHADING).text()
  const selectors = selectorsOf(css)
  expect(selectors.length).toBeGreaterThan(40)
  for (const selector of selectors) {
    expect(selector).toStartWith(":root[data-style='shading']")
    expect(selector).toContain(':not(#shading)')
  }
  for (const match of css.matchAll(/@keyframes\s+([\w-]+)/g)) expect(match[1]).toStartWith('shading-')
})

test('Shading ships in the app, the desktop build, the layout preview and the VS Code view', async () => {
  const [html, build, preview, extension] = await Promise.all([
    Bun.file(new URL('../src/desktop/renderer/index.html', import.meta.url)).text(),
    Bun.file(new URL('../scripts/buildDesktop.ts', import.meta.url)).text(),
    Bun.file(new URL('../scripts/previewDesktop.ts', import.meta.url)).text(),
    Bun.file(new URL('../src/vscode/extension.ts', import.meta.url)).text(),
  ])
  const sheets = [...html.matchAll(/href="\.\/([\w.-]+\.css)"/g)].map(match => match[1])
  expect(sheets).toContain('shading.css')
  expect(build).toContain("'shading.css'")
  expect(preview).toContain("'shading.css'")
  expect(extension).toContain("renderer/shading.css")
})

test('Shading has its own shape, not the Classic layout recoloured: inset cards, filled primary actions, outline chips', async () => {
  const css = await Bun.file(SHADING).text()
  const rule = (selector: string) => {
    const head = `:root[data-style='shading']:not(#shading) .atelier ${selector}{`
    const start = css.indexOf(head)
    if (start < 0) throw new Error(`missing Shading rule for ${selector}`)
    return css.slice(start + head.length, css.indexOf('}', start))
  }
  // The inset sidebar: the conversation and the inspector float as bordered, rounded cards.
  for (const surface of ['.chat', '.desktop-rail']) {
    expect(rule(surface)).toContain('margin:8px')
    expect(rule(surface)).toContain('border-radius:12px')
    expect(rule(surface)).toContain('box-shadow:var(--sh-shadow-sm)')
  }
  // Primary actions and your own turn are filled with the primary colour.
  expect(rule('.newchat')).toContain('background:var(--x-accent)')
  expect(rule('.msg--user')).toContain('background:var(--x-user-bubble)')
  expect(rule('.msg--user :is(.msg__text,.msg__text *)')).toContain('color:var(--x-on-accent)')
  // Toolbar chips are shadcn outline buttons.
  expect(rule('.composer__toolbar .cchip')).toContain('border:1px solid var(--sh-input)')
})

test('Shading text clears WCAG AA on every surface, in dark and light', async () => {
  const css = await Bun.file(SHADING).text()
  for (const selector of [":root[data-style='shading']:not(#shading)", ":root[data-style='shading'][data-theme='light']:not(#shading)"]) {
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

test('Settings names the solid style when explaining the background', () => {
  expect(isSolidStyle('shading')).toBe(true)
  expect(isSolidStyle('neo')).toBe(true)
  expect(isSolidStyle(null)).toBe(false)
  expect(neoBackdropNote('image', 'Shading')).toStartWith('Shading style: the picture is not shown')
  expect(neoBackdropNote('transparent', 'Shading')).toContain('Shading has no see-through window')
  expect(neoBackdropNote('image')).toStartWith('Neo style:')
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

test('in Shading a background lends only its colours, and Classic shows the picture again', () => {
  const store = new Map<string, string>()
  const root = fakeRoot()
  const saved = { localStorage: globalThis.localStorage, window: (globalThis as { window?: unknown }).window, document: (globalThis as { document?: unknown }).document, raf: globalThis.requestAnimationFrame, Image: (globalThis as { Image?: unknown }).Image }
  globalThis.localStorage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => { store.set(k, v) }, removeItem: (k: string) => { store.delete(k) } } as unknown as Storage
  Object.assign(globalThis, {
    window: { matchMedia: () => ({ matches: false }) },
    document: { documentElement: root },
    requestAnimationFrame: () => 0,
    Image: class { onload: (() => void) | null = null; src = '' },
  })
  try {
    const image = { kind: 'image', src: 'backgrounds/background-1.jpg', from: '#0b1f4d', to: '#3b1a5a', accent: '#ff8800', chat: 20, panels: 10, blur: 90 }
    store.set('xerxes.desktop.backdrop.v1', JSON.stringify(image))
    store.set(APPEARANCE_KEY, JSON.stringify({ theme: 'system', font: '13' }))
    const page = root as unknown as HTMLElement
    applyBackdrop(image as never, page, undefined)
    expect(root.attributes.get('data-backdrop')).toBe('image')

    chooseStyle('shading', page)
    expect(loadAppearance()).toEqual({ theme: 'system', font: '13', style: 'shading' })
    expect(root.attributes.get('data-style')).toBe('shading')
    expect(root.attributes.has('data-backdrop')).toBe(false)
    expect(root.attributes.get('data-backdrop-colors')).toBe('image')
    for (const name of ['--x-backdrop-image', '--glass-blur', '--x-chat-blur']) expect(root.styles.has(name)).toBe(false)

    chooseStyle('classic', page)
    expect(root.attributes.has('data-style')).toBe(false)
    expect(root.attributes.get('data-backdrop')).toBe('image')
  } finally {
    globalThis.localStorage = saved.localStorage
    Object.assign(globalThis, { window: saved.window, document: saved.document, requestAnimationFrame: saved.raf, Image: saved.Image })
  }
})
