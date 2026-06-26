// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { DARK_THEME, detectLightMode, fromSkin, LIGHT_THEME } from '../theme.js'

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

describe('Persepolis Lapis palette', () => {
  it('ships the lapis hero + gold/turquoise/carmine roles', () => {
    expect(DARK_THEME.color.primary).toBe('#4f86ff')
    expect(DARK_THEME.color.accent).toBe('#2fd4c4')
    expect(DARK_THEME.color.warn).toBe('#f0b429')
    expect(DARK_THEME.color.error).toBe('#e0556b')
    expect(DARK_THEME.color.system).toBe('#c77dff')
    expect(DARK_THEME.brand.name).toBe('Xerxes-Agents')
    expect(DARK_THEME.brand.welcome).toBe('The court awaits your word.')
  })

  it('light theme keeps the same shape', () => {
    expect(Object.keys(LIGHT_THEME.color).sort()).toEqual(Object.keys(DARK_THEME.color).sort())
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
    expect(t.brand.name).toBe('Xerxes-Agents')
  })
})
