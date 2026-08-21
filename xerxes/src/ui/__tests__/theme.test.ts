// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { DARK_THEME, detectLightMode, fromSkin, LIGHT_THEME, themeForMode } from '../theme.js'

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

describe('Night Standard palette', () => {
  it('ships a restrained high-contrast dark terminal palette', () => {
    expect(DARK_THEME.color.primary).toBe('#e6e6e6')
    expect(DARK_THEME.color.accent).toBe('#d8ae58')
    expect(DARK_THEME.color.border).toBe('#333333')
    expect(DARK_THEME.color.statusBg).toBe('#101010')
    expect(DARK_THEME.color.completionBg).toBe('#111111')
    expect(DARK_THEME.color.completionCurrentBg).toBe('#1a1a1a')
    expect(DARK_THEME.color.warn).toBe('#d8ae58')
    expect(DARK_THEME.color.error).toBe('#dd7c88')
    expect(DARK_THEME.brand.name).toBe('XERXES')
    expect(DARK_THEME.brand.prompt).toBe('❯')
    expect(DARK_THEME.brand.welcome).toBe('Ready for your next command.')
  })

  it('light theme keeps the same color shape with readable darker foregrounds', () => {
    expect(Object.keys(LIGHT_THEME.color).sort()).toEqual(Object.keys(DARK_THEME.color).sort())
    expect(LIGHT_THEME.color.primary).toBe('#172533')
    expect(LIGHT_THEME.color.accent).toBe('#006f94')
    expect(LIGHT_THEME.color.text).toBe('#172533')
  })
})

describe('interaction mode palettes', () => {
  it('uses neutral gray for code and blue, gold, purple for the other modes', () => {
    const code = themeForMode(DARK_THEME, 'code')
    const researcher = themeForMode(DARK_THEME, 'researcher')
    const plan = themeForMode(DARK_THEME, 'plan')
    const objective = themeForMode(DARK_THEME, 'objective')

    expect(code.color.accent).toBe('#aeb4bb')
    expect(code.color.statusBg).toBe('#101010')
    expect(code.color.completionBg).toBe('#111111')
    expect(code.color.completionCurrentBg).toBe('#1a1a1a')
    expect(researcher.color.accent).toBe('#6ea8fe')
    expect(plan.color.accent).toBe('#d8ae58')
    expect(objective.color.accent).toBe('#b18be8')
  })

  it('preserves semantic colors, branding, and the amber Derafsh signal', () => {
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
