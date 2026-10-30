// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

// The phone's "Shading, monochrome" stylesheet: neutral greys only, one documented error red,
// WCAG AA text contrast, iOS-safe sizing. These checks read mobile.css itself.

const MOBILE = join(import.meta.dirname, '..')
const CSS = readFileSync(join(MOBILE, 'mobile.css'), 'utf8')
const SOURCE = CSS.replace(/\/\*[\s\S]*?\*\//g, '')

interface Rule { readonly selector: string; readonly body: string; readonly media: string | undefined }

function parseRules(text: string, media: string | undefined, into: Rule[]): Rule[] {
  let index = 0
  while (index < text.length) {
    const open = text.indexOf('{', index)
    if (open < 0) break
    const prelude = text.slice(index, open).trim()
    let depth = 1
    let end = open + 1
    while (end < text.length && depth > 0) {
      if (text[end] === '{') depth++
      else if (text[end] === '}') depth--
      end++
    }
    const inner = text.slice(open + 1, end - 1)
    if (prelude.startsWith('@media')) parseRules(inner, prelude, into)
    else if (!prelude.startsWith('@')) into.push({ selector: prelude, body: inner, media })
    index = end
  }
  return into
}

function declarations(body: string): Map<string, string> {
  const out = new Map<string, string>()
  for (const part of body.split(';')) {
    const colon = part.indexOf(':')
    if (colon < 0) continue
    out.set(part.slice(0, colon).trim(), part.slice(colon + 1).trim())
  }
  return out
}

const RULES = parseRules(SOURCE, undefined, [])
const darkRule = RULES.find(rule => rule.selector === ':root' && rule.media === undefined && rule.body.includes('--x-screen'))
const lightRule = RULES.find(rule => rule.selector === ':root' && rule.media !== undefined && /prefers-color-scheme:\s*light/.test(rule.media))
const DARK = declarations(darkRule?.body ?? '')
const LIGHT = declarations(lightRule?.body ?? '')

/** The one non-grey colour: a low-saturation red for errors and destructive actions. */
const ERROR_RED: Readonly<Record<'dark' | 'light', string>> = { dark: 'e08e8e', light: 'a33a3a' }
const ERROR_TOKENS = new Set(['--x-danger', '--voice-danger'])

function rgb(hex: string): [number, number, number] {
  const digits = hex.replace(/^#/, '').toLowerCase()
  const full = digits.length <= 4 ? digits.slice(0, 3).split('').map(d => d + d).join('') : digits.slice(0, 6)
  return [0, 2, 4].map(i => Number.parseInt(full.slice(i, i + 2), 16)) as [number, number, number]
}

function luminance(hex: string): number {
  const [r, g, b] = rgb(hex).map(channel => {
    const c = channel / 255
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }) as [number, number, number]
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number]
  return (hi + 0.05) / (lo + 0.05)
}

/** Resolves a token to an opaque #rrggbb, using the dark value when light does not override it. */
function opaque(scheme: Map<string, string>, token: string): string {
  const value = scheme.get(token) ?? DARK.get(token)
  expect(value, `${token} is defined`).toBeDefined()
  expect(value, `${token} is an opaque hex colour`).toMatch(/^#[0-9a-f]{6}$/i)
  return value!
}

const REQUIRED_TOKENS = [
  '--x-backdrop', '--x-screen', '--x-card', '--x-chip', '--x-selected', '--x-sunken', '--x-hairline', '--x-input',
  '--x-title', '--x-prose', '--x-secondary', '--x-meta', '--x-faint', '--x-accent', '--x-on-accent',
  '--x-needs', '--x-needs-text', '--x-done', '--x-activity', '--x-danger', '--x-on-danger', '--x-focus', '--x-ring',
  '--sh-xs', '--sh-sm', '--sh-lg'
]
const TEXT_TOKENS = ['--x-title', '--x-prose', '--x-secondary', '--x-meta', '--x-needs-text', '--x-danger']
const SURFACE_TOKENS = ['--x-backdrop', '--x-screen', '--x-card', '--x-chip', '--x-selected']

describe('Shading monochrome theme tokens', () => {
  it('defines every token for dark (the default) and light', () => {
    expect(darkRule, 'a :root token block').toBeDefined()
    expect(lightRule, 'a prefers-color-scheme: light :root block').toBeDefined()
    for (const token of REQUIRED_TOKENS) {
      expect(DARK.has(token), `dark ${token}`).toBe(true)
      expect(LIGHT.has(token), `light ${token}`).toBe(true)
    }
    expect(DARK.get('color-scheme')).toBe('dark')
    expect(LIGHT.get('color-scheme')).toBe('light')
  })

  it('uses only neutral greys, except the documented error red', () => {
    const offenders: string[] = []
    for (const rule of RULES) {
      for (const [property, value] of declarations(rule.body)) {
        for (const match of value.matchAll(/#([0-9a-f]{3,8})\b/gi)) {
          const [r, g, b] = rgb(match[1]!)
          if (Math.max(r, g, b) - Math.min(r, g, b) <= 3) continue
          const red = match[1]!.toLowerCase().slice(0, 6)
          if (ERROR_TOKENS.has(property) && Object.values(ERROR_RED).includes(red)) continue
          offenders.push(`${rule.selector} { ${property}: ${value} }`)
        }
      }
    }
    expect(offenders).toEqual([])
    expect(DARK.get('--x-danger')).toBe(`#${ERROR_RED.dark}`)
    expect(LIGHT.get('--x-danger')).toBe(`#${ERROR_RED.light}`)
  })

  it('keeps the error red low in saturation', () => {
    for (const hex of Object.values(ERROR_RED)) {
      const [r, g, b] = rgb(hex).map(c => c / 255) as [number, number, number]
      const max = Math.max(r, g, b)
      const min = Math.min(r, g, b)
      const lightness = (max + min) / 2
      const saturation = (max - min) / (1 - Math.abs(2 * lightness - 1))
      expect(saturation, hex).toBeLessThanOrEqual(0.6)
    }
  })

  it('never introduces hue through colour functions or named colours', () => {
    expect(SOURCE).not.toMatch(/\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch)\(/i)
    for (const rule of RULES) {
      for (const [property, value] of declarations(rule.body)) {
        expect(value, `${rule.selector} ${property}`).not.toMatch(/\b(?:red|green|blue|yellow|orange|purple|violet|pink|cyan|magenta|teal|navy|gold)\b/i)
      }
    }
  })

  for (const [name, scheme] of [['dark', DARK], ['light', LIGHT]] as const) {
    it(`gives every ${name} text token at least 4.5:1 on the screen, card, chip and selected surfaces`, () => {
      for (const text of TEXT_TOKENS) {
        for (const surface of SURFACE_TOKENS) {
          const ratio = contrast(opaque(scheme, text), opaque(scheme, surface))
          expect(ratio, `${name} ${text} on ${surface}`).toBeGreaterThanOrEqual(4.5)
        }
      }
    })

    it(`keeps ${name} filled controls legible`, () => {
      expect(contrast(opaque(scheme, '--x-on-accent'), opaque(scheme, '--x-accent'))).toBeGreaterThanOrEqual(4.5)
      expect(contrast(opaque(scheme, '--x-on-danger'), opaque(scheme, '--x-danger'))).toBeGreaterThanOrEqual(4.5)
    })
  }
})

describe('Shading iOS correctness', () => {
  it('sizes the app with dynamic viewport units, never 100vh', () => {
    expect(SOURCE).not.toMatch(/(?<![\w-])\d+(?:\.\d+)?vh\b/)
    const app = RULES.find(rule => rule.selector === '.app')
    expect(declarations(app?.body ?? '').get('min-height')).toBe('100dvh')
  })

  it('keeps every input, select and textarea at 16px or more so iOS does not zoom on focus', () => {
    const sized = RULES.filter(rule => /\b(?:input|textarea|select)\b/.test(rule.selector))
      .flatMap(rule => {
        const found: { selector: string; px: number }[] = []
        for (const [property, value] of declarations(rule.body)) {
          if (property === 'font-size') found.push({ selector: rule.selector, px: Number.parseFloat(value) })
          if (property === 'font') {
            const size = /(\d+(?:\.\d+)?)px/.exec(value)
            if (size) found.push({ selector: rule.selector, px: Number.parseFloat(size[1]!) })
          }
        }
        return found
      })
    expect(sized.some(entry => /\binput\b/.test(entry.selector))).toBe(true)
    expect(sized.some(entry => /\btextarea\b/.test(entry.selector))).toBe(true)
    for (const entry of sized) expect(entry.px, entry.selector).toBeGreaterThanOrEqual(16)
  })

  it('pads the header, tab bar and sheets with the safe-area insets', () => {
    for (const selector of ['.app-head', '.tabs', '.sheet', '.voice-screen']) {
      const rule = RULES.find(candidate => candidate.selector === selector && candidate.media === undefined)
      expect(rule?.body ?? '', selector).toMatch(/--safe-(?:top|bottom)|env\(safe-area-inset/)
    }
    expect(DARK.get('--safe-top')).toContain('env(safe-area-inset-top')
    expect(DARK.get('--safe-bottom')).toContain('env(safe-area-inset-bottom')
  })

  it('turns off the tap flash and text inflation, and stops scroll chaining in sheets', () => {
    const html = declarations(RULES.find(rule => rule.selector === 'html')?.body ?? '')
    expect(html.get('-webkit-tap-highlight-color')).toBe('transparent')
    expect(html.get('-webkit-text-size-adjust')).toBe('100%')
    expect(html.get('overscroll-behavior')).toBe('none')
    const sheet = declarations(RULES.find(rule => rule.selector === '.sheet')?.body ?? '')
    expect(sheet.get('overflow-y')).toBe('auto')
    expect(sheet.get('overscroll-behavior')).toBe('contain')
    const composer = declarations(RULES.find(rule => rule.selector === '.composer')?.body ?? '')
    expect(composer.get('position')).toBe('sticky')
  })

  it('uses the system UI stack for text and a monospace stack for code', () => {
    expect(DARK.get('--sans')).toBe("-apple-system, BlinkMacSystemFont, 'SF Pro Text', Inter, system-ui, sans-serif")
    expect(DARK.get('--mono')).toBe("ui-monospace, 'SF Mono', Menlo, monospace")
  })
})

describe('Shading class coverage', () => {
  // Classes the components set that never had a rule; they are hooks, not styling.
  const UNSTYLED = new Set(['talk', 'block-md'])
  const DYNAMIC = [
    'status-busy', 'status-attention', 'status-calm', 'status-problem', 'banner-busy', 'banner-problem',
    'card-attention', 'card-unavailable', 'is-error', 'is-active', 'is-recording', 'is-muted', 'is-done',
    'chip-bad', 'chip-anomaly', 'sheet-tall', 'sheet-fullscreen', 'tool-working', 'tool-error', 'md-box', 'md-code', 'md-table', 'agent-orb'
  ]

  function sources(directory: string): string[] {
    return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
      const path = join(directory, entry.name)
      if (entry.isDirectory()) return entry.name === '__tests__' ? [] : sources(path)
      return entry.name.endsWith('.tsx') ? [readFileSync(path, 'utf8')] : []
    })
  }

  it('styles every class name the components use', () => {
    const used = new Set(DYNAMIC)
    for (const text of sources(MOBILE)) {
      for (const match of text.matchAll(/className="([^"]+)"/g)) for (const name of match[1]!.split(/\s+/)) if (name !== '') used.add(name)
    }
    const missing = [...used].filter(name => !UNSTYLED.has(name) && !new RegExp(`\\.${name}(?![\\w-])`).test(SOURCE))
    expect(missing).toEqual([])
  })
})
