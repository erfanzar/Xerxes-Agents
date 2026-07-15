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
