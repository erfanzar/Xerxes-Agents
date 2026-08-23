// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import {
  DARK_THEME,
  detectLightMode,
  fromSkin,
  LIGHT_THEME,
  NOCTURNE_DARK,
  NOCTURNE_LIGHT,
  themeForMode
} from '../theme.js'

describe('detectLightMode', () => {
  it('honors XERXES_TUI_LIGHT boolean first', () => {
    expect(detectLightMode({ XERXES_TUI_LIGHT: 'true' } as NodeJS.ProcessEnv)).toBe(true)
    expect(detectLightMode({ XERXES_TUI_LIGHT: 'off', XERXES_TUI_THEME: 'light' } as NodeJS.ProcessEnv)).toBe(false)
  })

  it('honors named XERXES_TUI_THEME', () => {
    expect(detectLightMode({ XERXES_TUI_THEME: 'light' } as NodeJS.ProcessEnv)).toBe(true)
    expect(detectLightMode({ XERXES_TUI_THEME: 'dark' } as NodeJS.ProcessEnv)).toBe(false)
  })

  it('reads COLORFGBG light slots 7/15', () => {
    expect(detectLightMode({ COLORFGBG: '0;15' } as NodeJS.ProcessEnv)).toBe(true)
    expect(detectLightMode({ COLORFGBG: '15;0' } as NodeJS.ProcessEnv)).toBe(false)
  })

  it('defaults to dark', () => {
    expect(detectLightMode({} as NodeJS.ProcessEnv)).toBe(false)
  })
})

describe('Nocturne palette', () => {
  it('derives every product role from a design-system token', () => {
    // The point of the rewrite: no role holds a hex that is not a step of
    // the ramp or one of the six voice colours. Spot-checking the roles is
    // how that stays true — a future edit that reaches for a nearby hex
    // instead of a token fails here.
    expect(DARK_THEME.color.primary).toBe(NOCTURNE_DARK.strong)
    expect(DARK_THEME.color.text).toBe(NOCTURNE_DARK.title)
    expect(DARK_THEME.color.muted).toBe(NOCTURNE_DARK.meta)
    expect(DARK_THEME.color.accent).toBe('#6ea8fe')
    expect(DARK_THEME.color.brandGold).toBe('#6ea8fe')
    expect(DARK_THEME.color.userBar).toBe('#6ea8fe')
    expect(DARK_THEME.color.border).toBe(NOCTURNE_DARK.hairline)
    expect(DARK_THEME.color.statusBg).toBe(NOCTURNE_DARK.screen)
    expect(DARK_THEME.color.completionBg).toBe(NOCTURNE_DARK.chrome)
    expect(DARK_THEME.color.completionCurrentBg).toBe(NOCTURNE_DARK.selected)
    expect(DARK_THEME.color.userBandBg).toBe(NOCTURNE_DARK.workingCardBg)
    expect(LIGHT_THEME.color.userBandBg).toBe(NOCTURNE_LIGHT.workingCardBg)
    expect(DARK_THEME.brand.name).toBe('XERXES')
    expect(DARK_THEME.brand.prompt).toBe('❯')
    expect(DARK_THEME.brand.welcome).toBe('Many agents, one terminal.')
  })

  it('gives each of the six voices exactly one meaning', () => {
    expect(DARK_THEME.color.accent).toBe(NOCTURNE_DARK.working)
    expect(DARK_THEME.color.ok).toBe('#57ca85')
    expect(DARK_THEME.color.error).toBe('#f47067')
    expect(DARK_THEME.color.warn).toBe('#d8ae58')
    expect(DARK_THEME.color.system).toBe('#b39cf0')
    expect(DARK_THEME.color.diffHunk).toBe('#56c8d8')

    // Amber means "a human is required" and nothing else, so it must not be
    // reachable through any other role. Cyan is hunk headers and nothing
    // else, for the same reason.
    const dark = DARK_THEME.color as Record<string, string>
    const amberRoles = Object.keys(dark).filter(key => dark[key] === NOCTURNE_DARK.needsInput)
    expect(amberRoles.sort()).toEqual(['statusWarn', 'warn'])
    const cyanRoles = Object.keys(dark).filter(key => dark[key] === NOCTURNE_DARK.structure)
    expect(cyanRoles).toEqual(['diffHunk'])
  })

  it('light theme keeps the same role shape with an inverted ramp', () => {
    expect(Object.keys(LIGHT_THEME.color).sort()).toEqual(Object.keys(DARK_THEME.color).sort())
    expect(Object.keys(NOCTURNE_LIGHT).sort()).toEqual(Object.keys(NOCTURNE_DARK).sort())
    expect(LIGHT_THEME.color.primary).toBe(NOCTURNE_LIGHT.strong)
    expect(LIGHT_THEME.color.text).toBe(NOCTURNE_LIGHT.title)
    expect(LIGHT_THEME.color.accent).toBe('#1f64b5')
  })
})

describe('interaction mode palettes', () => {
  it('borrows mode accents from the voice colours and leaves code un-hued', () => {
    const code = themeForMode(DARK_THEME, 'code')
    const researcher = themeForMode(DARK_THEME, 'researcher')
    const plan = themeForMode(DARK_THEME, 'plan')
    const objective = themeForMode(DARK_THEME, 'objective')

    // Between the ramp's title and secondary steps: present enough to mark a
    // chip, colourless enough that the default mode adds no hue to a screen.
    expect(code.color.accent).toBe('#b1b8c1')
    expect(code.color.statusBg).toBe(NOCTURNE_DARK.screen)
    expect(code.color.completionBg).toBe(NOCTURNE_DARK.chrome)
    expect(code.color.completionCurrentBg).toBe(NOCTURNE_DARK.selected)
    expect(researcher.color.accent).toBe(NOCTURNE_DARK.working)
    expect(plan.color.accent).toBe(NOCTURNE_DARK.structure)
    expect(objective.color.accent).toBe(NOCTURNE_DARK.activity)
  })

  it('preserves semantic colors, branding, and the semantic warn signal', () => {
    const objective = themeForMode(DARK_THEME, 'objective')

    expect(objective.color.ok).toBe(DARK_THEME.color.ok)
    expect(objective.color.warn).toBe(DARK_THEME.color.warn)
    expect(objective.color.error).toBe(DARK_THEME.color.error)
    expect(objective.brand).toBe(DARK_THEME.brand)
  })

  // Regression guard for the bug that made the whole transcript one gray: an
  // earlier `themeForMode` smeared the mode accent over `primary` and `label`
  // too, so in `code` mode the user band, markdown headings, and panel labels
  // all collapsed onto '#aeb4bb'. Voice and brand roles must survive every
  // mode; if a future edit adds one of these to the override list, this fails.
  it('never lets an interaction mode repaint the voice or brand roles', () => {
    for (const mode of ['code', 'researcher', 'plan', 'objective'] as const) {
      const t = themeForMode(DARK_THEME, mode)

      expect(t.color.userBar).toBe(DARK_THEME.color.userBar)
      expect(t.color.userText).toBe(DARK_THEME.color.userText)
      expect(t.color.toolName).toBe(DARK_THEME.color.toolName)
      expect(t.color.system).toBe(DARK_THEME.color.system)
      expect(t.color.thinking).toBe(DARK_THEME.color.thinking)
      expect(t.color.brandGold).toBe(DARK_THEME.color.brandGold)
      expect(t.color.brandLapis).toBe(DARK_THEME.color.brandLapis)
    }
  })

  it('keeps markdown heading hierarchy and panel labels out of the mode accent', () => {
    const code = themeForMode(DARK_THEME, 'code')

    // `primary` (h2/h3, bold) must stay distinct from `accent` (h1), and
    // `label` is a panel-header role, not the user transcript band.
    expect(code.color.primary).toBe(DARK_THEME.color.primary)
    expect(code.color.label).toBe(DARK_THEME.color.label)
    expect(code.color.primary).not.toBe(code.color.accent)
  })
})

describe('fromSkin', () => {
  it('merges skin_engine roles over the default theme', () => {
    const t = fromSkin(
      { primary: '#ff0000', accent: '#00ff00', tool_name: '#0000ff', diff_add: '#123456' },
      { agent_name: 'Ares', prompt_symbol: '›' }
    )
    expect(t.color.primary).toBe('#ff0000')
    expect(t.color.accent).toBe('#00ff00')
    expect(t.color.toolName).toBe('#0000ff')
    expect(t.color.ok).toBe('#123456') // diff_add → ok/statusGood
    expect(t.brand.name).toBe('Ares')
    expect(t.brand.prompt).toBe('›')
  })

  it('falls back to defaults for missing roles', () => {
    const t = fromSkin({})
    expect(t.color.primary).toBe(DARK_THEME.color.primary)
    expect(t.brand.name).toBe('XERXES')
  })

  it('keeps custom mark and hero skin payloads intact', () => {
    const t = fromSkin({}, {}, '[#ffffff]mark[/]', '[#ffffff]hero[/]')

    expect(t.bannerLogo).toBe('[#ffffff]mark[/]')
    expect(t.bannerHero).toBe('[#ffffff]hero[/]')
  })
})
