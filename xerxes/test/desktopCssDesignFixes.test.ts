// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { PALETTES } from '../src/desktop/renderer/palettes.js'

const rendererDir = join(import.meta.dir, '../src/desktop/renderer')
const readCss = (name: string): string => readFileSync(join(rendererDir, name), 'utf8')

const appCss = readCss('app.css')
const atelierCss = readCss('atelier.css')
const layoutCss = readCss('layout.css')
const vibesCss = readCss('vibes.css')
const neoCss = readCss('neo.css')

function relativeLuminance(hex: string): number {
  const clean = hex.replace('#', '')
  const r = Number.parseInt(clean.slice(0, 2), 16) / 255
  const g = Number.parseInt(clean.slice(2, 4), 16) / 255
  const b = Number.parseInt(clean.slice(4, 6), 16) / 255
  const toLinear = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
  return 0.2126 * toLinear(r) + 0.7152 * toLinear(g) + 0.0722 * toLinear(b)
}

function contrastRatio(fgHex: string, bgHex: string): number {
  const l1 = relativeLuminance(fgHex)
  const l2 = relativeLuminance(bgHex)
  const lighter = Math.max(l1, l2)
  const darker = Math.min(l1, l2)
  return (lighter + 0.05) / (darker + 0.05)
}

function parseBlockTokens(css: string, header: string): Record<string, string> {
  const start = css.indexOf(header)
  expect(start).toBeGreaterThan(-1)
  const open = css.indexOf('{', start)
  const close = css.indexOf('}', open)
  const body = css.slice(open + 1, close)
  const tokens: Record<string, string> = {}
  for (const m of body.matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);?/gi)) {
    tokens[m[1]!] = m[2]!.trim()
  }
  return tokens
}

describe('Desktop CSS & design system fixes', () => {
  test('macOS traffic-light clearance works in focus mode and scales with --ui-zoom', () => {
    // Stray space between .atelier and .atelier--focus is fixed and uses zoom-aware 84px clearance.
    expect(vibesCss).toContain(
      ".atelier.atelier--focus:not(.app--no-traffic-lights) .chat__head.t3-topbar{padding-left:calc(84px / var(--ui-zoom, 1))}",
    )
    expect(vibesCss).not.toContain('.atelier .atelier--focus')
    // Sidebar header uses zoom-aware 84px clearance instead of hardcoded 74px.
    expect(vibesCss).toContain('.atelier .t3-side__head{')
    expect(vibesCss).toContain('padding:0 10px 0 calc(84px / var(--ui-zoom, 1))')
    expect(vibesCss).not.toContain('padding:0 10px 0 74px')
  })

  test('Vibes & Neo pane backgrounds yield to [data-backdrop] and inset corner notches are neutralized', () => {
    for (const selector of [
      ":root[data-style='vibes']:not([data-backdrop]):not([data-host='vscode']):not(#vibes) body{background:var(--x-backdrop)}",
      ":root:is([data-style='vibes'],[data-style='neo']):not([data-backdrop]):not([data-host='vscode']):not(#vibes) .atelier :is(.side,.top){background:var(--x-chrome)}",
      ":root:is([data-style='vibes'],[data-style='neo']):not([data-backdrop]):not([data-host='vscode']):not(#vibes) .atelier :is(.app__body,.chat){background:var(--x-screen)}",
      ":root:is([data-style='vibes'],[data-style='neo']):not([data-backdrop]):not([data-host='vscode']):not(#vibes) .atelier .chat__head.t3-topbar{background:var(--x-screen)}",
      ":root:is([data-style='vibes'],[data-style='neo']):not([data-backdrop]):not([data-host='vscode']):not(#vibes) .atelier .desktop-rail{background:var(--x-screen)}",
      ":root:is([data-style='vibes'],[data-style='neo']):not([data-backdrop]):not([data-host='vscode']):not(#vibes) .atelier .app__body>.desktop-rail:not(.t3-surface-launcher)>header{background:var(--x-screen)}",
      ":root:is([data-style='vibes'],[data-style='neo']):not([data-backdrop]):not([data-host='vscode']):not(#vibes) .atelier :is(.scm__side,.scm__diff,.file-browser,.file-browser__preview,.term){background:var(--x-screen)}",
    ]) {
      expect(vibesCss).toContain(selector)
    }

    for (const selector of [
      ":root[data-style='neo']:not([data-backdrop]):not([data-host='vscode']):not(#neo) .atelier .side{background:var(--x-chrome)}",
      ":root[data-style='neo']:not([data-backdrop]):not([data-host='vscode']):not(#neo) .atelier .top{background:var(--x-chrome)}",
      ":root[data-style='neo']:not([data-backdrop]):not([data-host='vscode']):not(#neo) .atelier .app__body,",
      ":root[data-style='neo']:not([data-backdrop]):not([data-host='vscode']):not(#neo) .atelier .chat{background:var(--x-screen)}",
      ":root[data-style='neo']:not([data-backdrop]):not([data-host='vscode']):not(#neo) .atelier .desktop-rail{background:var(--x-chrome)}",
    ]) {
      expect(neoCss).toContain(selector)
    }

    // Retired 12px corner-notch ::after renders nothing while preserving contract hook.
    expect(atelierCss).toContain(
      ":root[data-backdrop='transparent'][data-layout='claude'] .atelier .desktop-rail::after{content:none;border:solid var(--x-screen);border-width:0}",
    )
    expect(atelierCss).not.toContain('border-width:0 12px 12px 12px')
    // Backdrop diff pre yields cleanly.
    expect(vibesCss).toContain(
      ":root:is([data-style='vibes'],[data-style='neo'])[data-backdrop]:not(#vibes) .atelier .scm__diff>pre.change-review__source{--diff-bg:transparent;background:transparent}",
    )
  })

  test('checked .vq__radio works in both Vibes & Neo and uses var(--x-on-accent) for contrast', () => {
    expect(vibesCss).toContain(
      ":root:is([data-style='vibes'],[data-style='neo']):not(#vibes) .atelier :is(.vq__opt,.vq__other-toggle)[aria-checked='true']>.vq__radio{border:0;background:radial-gradient(circle,var(--x-on-accent) 0 3.5px,var(--x-accent) 4px)}",
    )
    expect(vibesCss).not.toContain('radial-gradient(circle,#ffffff')
    expect(neoCss).toContain(
      ":root[data-style='neo']:not(#neo) .atelier :is(.vq__opt,.vq__other-toggle)[aria-checked='true']>.vq__radio{border:0;background:var(--x-on-accent);box-shadow:inset 0 0 0 4px var(--x-accent)}",
    )
  })

  test('VS Code cascade protection gates desktop surfaces, composer card, and modal sizing', () => {
    expect(vibesCss).toContain(":root[data-style='vibes']:not([data-host='vscode']):not(#vibes){")
    expect(vibesCss).toContain(":root[data-style='vibes'][data-theme='light']:not([data-host='vscode']):not(#vibes){")
    expect(vibesCss).toContain(
      ":root[data-style='vibes']:not([data-host='vscode']):not(#vibes) .atelier .composer-wrap:not(.agent-dock) .composer__input{",
    )
    expect(vibesCss).toContain(
      ":root:is([data-style='vibes'],[data-style='neo']):not([data-host='vscode']):not(#vibes) .atelier .backdrop>.modal:has(>.modal__side){",
    )
    expect(layoutCss).toContain(":root:not([data-host='vscode']):not(#layout){--x-popover:var(--x-dialog)}")
    expect(layoutCss).toContain(
      ":root:not([data-host='vscode']):not(#layout) .atelier .composer-wrap{display:grid;grid-template-columns:minmax(0,1fr) auto;",
    )
    expect(layoutCss).toContain(
      ":root:not([data-host='vscode']):not(#layout) .atelier .modal:has(>.modal__side){width:min(980px,calc(100vw - 48px));",
    )
  })

  test('right-rail tabs expand labels for active/item/wide/compact tabs and place close inline', () => {
    expect(vibesCss).toContain(
      ".desktop-rail:is([data-fit='wide'],[data-fit='compact'])>header .desktop-rail__tab>button.desktop-rail__tabmain>span,",
    )
    expect(vibesCss).toContain(
      ".desktop-rail>header .desktop-rail__tab:is([data-selected],[data-item-tab],[data-has-label])>button.desktop-rail__tabmain>span{position:static;width:auto;height:auto;max-width:140px;",
    )
    expect(vibesCss).toContain(
      '.desktop-rail__tab>button.desktop-rail__tabclose{position:static;display:inline-grid;',
    )
    expect(vibesCss).not.toContain('top:-4px;right:-4px')
  })

  test('SwarmBoard & SideQuestion stay pinned flex at all viewport sizes without media gate', () => {
    expect(atelierCss).not.toContain('@media (min-width:960px) and (min-height:640px)')
    expect(atelierCss).not.toContain('@media(min-width:960px) and (min-height:640px)')
    expect(atelierCss).toContain('.atelier .desktop-rail>.studio-sheet-content:has(>.swarm-board),')
    expect(atelierCss).toContain(
      '.atelier .desktop-rail>.studio-sheet-content:has(>.side-question){display:flex;flex-direction:column;overflow:hidden}',
    )
  })

  test('two-column Settings modal sizing is scoped to .modal:has(>.modal__side) so single-column dialogs stay compact', () => {
    expect(vibesCss).not.toContain(':not(.modal--single)')
    expect(layoutCss).not.toContain(':not(.modal--single)')
    expect(vibesCss).toContain('.atelier .backdrop>.modal:has(>.modal__side){width:min(1022px,')
    expect(vibesCss).toContain(
      '.atelier .backdrop>.modal:not(:has(>.modal__side)){width:min(580px,calc(100vw - 48px));height:auto;max-height:calc(100dvh - 64px);',
    )
    expect(layoutCss).toContain(
      '.atelier .modal:not(:has(>.modal__side)){width:min(580px,calc(100vw - 48px));height:auto;max-height:calc(100dvh - 64px);',
    )
  })

  test('alien Material-3 Google Gemini ws-gemini overrides are completely removed from atelier.css', () => {
    for (const forbidden of ['--g-bg', '--g-primary', '#131314', '#a8c7fa', '#ws-gemini', '.ws-gemini']) {
      expect(atelierCss).not.toContain(forbidden)
    }
    expect(atelierCss).toContain('.workspace-sheet.studio-sheet{')
    expect(atelierCss).toContain('.workspace-panel .ws-card{')
  })

  test('dead animations, specificity, launcher header, flush rails, orb jump, SCM compact & perf regressions are fixed', () => {
    // 1. studio-spin replaces dead `animation: spin`.
    expect(atelierCss).toContain('.swarm-ov-spin>svg{animation:studio-spin 1.1s linear infinite}')
    expect(atelierCss).not.toMatch(/animation:\s*spin\b/)

    // 2. .artdoc* double-ID specificity in vibes.css & Neo square corners double-ID override.
    expect(vibesCss).toContain(":not(#vibes):not(#artdoc-vibes) .atelier .rsec--detail .artdoc__title{")
    expect(neoCss).toContain(":root[data-style='neo']:not(#neo):not(#neo-corners) .atelier :is(")

    // 3. .t3-surface-launcher excluded from 52px bordered rail header.
    expect(vibesCss).toContain('.app__body>.desktop-rail:not(.t3-surface-launcher)>header{')

    // 4. No rounded 10px corner wedges on .file-browser__document in flush Vibes/Neo rails.
    expect(atelierCss).toContain(
      ":root[data-layout='claude']:not([data-style='vibes']):not([data-style='neo']) .atelier .file-browser__document{border-radius:10px;overflow:hidden}",
    )
    expect(vibesCss).toContain('.desktop-rail :is(.file-browser__document,.file-preview){border-radius:0}')

    // 5. .railstatus__orb hidden consistently across all tones (no horizontal layout jump).
    expect(layoutCss).toContain(
      ':root:not(#layout):not(#tool-ux) .atelier .railstatus .railstatus__orb,:root:not(#layout):not(#tool-ux) .atelier .railstatus[data-tone=\'working\'] .railstatus__orb{display:none}',
    )

    // 6. SCM compact layout min-width 240px, 640px single-column collapse, auto-height commit textarea.
    expect(layoutCss).toContain(
      '.desktop-rail .scm.scm--compact{display:grid;grid-template-columns:minmax(240px,36%) minmax(0,1fr)}',
    )
    expect(layoutCss).toContain('@container (max-width:640px){')
    expect(layoutCss).toContain('.scm__commit textarea{min-height:56px;height:auto;')

    // 7. Zero live backdrop-filter on opaque composer & ThreadDetailsCard, and zero feTurbulence grain.
    expect(vibesCss).not.toContain('--t3-grain')
    expect(vibesCss).not.toContain('feTurbulence')
    for (const m of vibesCss.matchAll(/\.t3-details-card\{([^}]*)\}/g)) {
      expect(m[1]).not.toContain('backdrop-filter')
    }
    const composerBgRule = vibesCss.match(
      /:root\[data-style='vibes'\]:not\(\[data-host='vscode'\]\):not\(#vibes\) \.atelier \.composer-wrap:not\(\.agent-dock\) :is\(\.composer__images,\.composer__input,\.composer__toolbar\)\{([^}]*)\}/,
    )?.[1]
    expect(composerBgRule).toBeDefined()
    expect(composerBgRule).not.toContain('backdrop-filter')
  })

  test('z-index scale keeps Settings modal above findbar & decision-dock and separates toasts', () => {
    const zVal = (name: string): number => {
      const m = appCss.match(new RegExp(`${name}:\\s*(\\d+)`))
      expect(m).not.toBeNull()
      return Number(m![1])
    }
    expect(zVal('--z-dialog')).toBeGreaterThan(zVal('--z-picker'))
    expect(zVal('--z-picker')).toBeGreaterThan(zVal('--z-menu'))
    expect(zVal('--z-menu')).toBeGreaterThan(zVal('--z-modal'))
    expect(zVal('--z-modal')).toBeGreaterThan(zVal('--z-find'))
    expect(zVal('--z-find')).toBeGreaterThan(zVal('--z-dock'))
    expect(zVal('--z-dock')).toBeGreaterThan(zVal('--z-scrim'))
    expect(zVal('--z-scrim')).toBeGreaterThan(zVal('--z-sheet'))
    expect(zVal('--z-sheet')).toBeGreaterThan(zVal('--z-toast'))

    // .artifact-toast sits top-right (separate from bottom-right .notice-stack & composer).
    expect(layoutCss).toContain('.artifact-toast{position:fixed;top:60px;right:20px;bottom:auto;z-index:var(--z-toast,90);')
    expect(atelierCss).toContain('.atelier .notice-stack{position:fixed;z-index:var(--z-toast,90);right:16px;bottom:16px;')
  })

  test('palettes.ts and vibes.css tokens meet WCAG AA (>= 4.5:1) contrast and surface separation', () => {
    for (const p of PALETTES) {
      const variants = 'only' in p ? ([['only', p.only]] as const) : ([['light', p.light], ['dark', p.dark]] as const)
      for (const [mode, s] of variants) {
        expect(contrastRatio(s.foreground, s.background), `${p.name}.${mode} fg/bg`).toBeGreaterThanOrEqual(4.5)
        expect(contrastRatio(s.foreground, s.card), `${p.name}.${mode} fg/card`).toBeGreaterThanOrEqual(4.5)
        expect(contrastRatio(s.mutedForeground, s.background), `${p.name}.${mode} muted/bg`).toBeGreaterThanOrEqual(4.5)
        expect(contrastRatio(s.mutedForeground, s.card), `${p.name}.${mode} muted/card`).toBeGreaterThanOrEqual(4.5)
      }
    }

    const everforest = PALETTES.find((p) => p.name === 'everforest')!
    expect('light' in everforest && 'dark' in everforest).toBe(true)
    if ('light' in everforest) {
      expect(everforest.light.border).not.toBe(everforest.light.background)
      expect(everforest.light.card).not.toBe(everforest.light.background)
      expect(everforest.dark.border).not.toBe(everforest.dark.background)
      expect(everforest.dark.card).not.toBe(everforest.dark.background)
    }

    const catppuccin = PALETTES.find((p) => p.name === 'catppuccin')!
    expect('light' in catppuccin && 'dark' in catppuccin).toBe(true)
    if ('light' in catppuccin) {
      expect(catppuccin.light.mutedForeground).not.toBe(catppuccin.light.foreground)
      expect(catppuccin.dark.mutedForeground).not.toBe(catppuccin.dark.foreground)
    }

    const solarized = PALETTES.find((p) => p.name === 'solarized')!
    expect('dark' in solarized).toBe(true)
    if ('dark' in solarized) {
      expect(solarized.dark.card).not.toBe(solarized.dark.background)
    }

    const vibesDark = parseBlockTokens(vibesCss, ":root[data-style='vibes']:not([data-host='vscode']):not(#vibes)")
    for (const bg of [vibesDark['--x-screen']!, vibesDark['--x-card']!, vibesDark['--x-selected']!, vibesDark['--x-user-bubble']!]) {
      expect(contrastRatio(vibesDark['--x-meta']!, bg)).toBeGreaterThanOrEqual(4.5)
      expect(contrastRatio(vibesDark['--x-caption']!, bg)).toBeGreaterThanOrEqual(4.5)
      expect(contrastRatio(vibesDark['--x-numeric']!, bg)).toBeGreaterThanOrEqual(4.5)
      expect(contrastRatio(vibesDark['--x-secondary']!, bg)).toBeGreaterThanOrEqual(4.5)
    }

    const vibesLight = parseBlockTokens(
      vibesCss,
      ":root[data-style='vibes'][data-theme='light']:not([data-host='vscode']):not(#vibes)",
    )
    expect(contrastRatio(vibesLight['--x-meta']!, vibesLight['--x-screen']!)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(vibesLight['--x-meta']!, vibesLight['--x-chip']!)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(vibesLight['--x-needs']!, vibesLight['--x-needs-bg']!)).toBeGreaterThanOrEqual(4.5)
  })

  test('data-font (--ui-small / --ui-font / --ui-large) scaling and peer CSS hooks are wired', () => {
    expect(vibesCss).toContain('.t3-thread__title{font:500 var(--ui-large)/1.3 var(--sans)')
    expect(vibesCss).toContain('.t3-breadcrumb__project{')
    expect(vibesCss).toContain('font:400 var(--ui-large)/1 var(--sans)')
    expect(vibesCss).toContain('.menu .menu__item{min-height:28px;padding:0 10px;font:400 var(--ui-large)/1.2 var(--sans)}')
    expect(layoutCss).toContain('.modal__main :is(.modal__title,.hubphone>h3){font:600 calc(var(--ui-large) + 2px)/1.3 var(--sans)')

    // Peer contracts (#23, #24, #29)
    expect(appCss).toContain('.md ol { margin: 0 0 6px; padding-left: 22px; list-style: decimal; }')
    expect(appCss).toContain('.md ol > li::before { content: none; }')
    expect(atelierCss).toContain('.picker-layer>:is(.t3-scope-pop,.t3-add-pop,.t3-snooze-menu){pointer-events:auto;z-index:1}')
    expect(atelierCss).toContain('.atelier .learned-group__confirm{')
    expect(atelierCss).toContain('.atelier .learning-outcome{')
    expect(vibesCss).toContain('.t3-mpop__row.is-focused{background:var(--mac-control);color:var(--x-title)}')
    expect(vibesCss).toContain('.t3-breadcrumb__actions{width:24px;height:24px;min-width:24px}')
    expect(vibesCss).toContain('.vq__other-toggle{')
  })
})
