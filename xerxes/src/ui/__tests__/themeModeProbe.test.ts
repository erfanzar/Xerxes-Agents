// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it } from 'vitest'

import {
  currentBaseTheme,
  currentLightMode,
  DARK_THEME,
  explicitLightMode,
  fromSkin,
  LIGHT_THEME,
  resetTerminalThemeMode,
  subscribeTerminalThemeMode,
  type TerminalThemeMode,
  type TerminalThemeModeSource,
  type Theme
} from '../theme.js'

class FakeRenderer implements TerminalThemeModeSource {
  themeMode: null | TerminalThemeMode
  private listeners: ((mode: TerminalThemeMode) => void)[] = []
  private pending: null | TerminalThemeMode

  constructor(
    { probed = null, pending = null }: { pending?: null | TerminalThemeMode; probed?: null | TerminalThemeMode } = {}
  ) {
    this.themeMode = probed
    this.pending = pending
  }

  get listenerCount(): number {
    return this.listeners.length
  }

  on(_event: 'theme_mode', listener: (mode: TerminalThemeMode) => void): void {
    this.listeners.push(listener)
  }

  off(_event: 'theme_mode', listener: (mode: TerminalThemeMode) => void): void {
    this.listeners = this.listeners.filter(entry => entry !== listener)
  }

  waitForThemeMode(): Promise<null | TerminalThemeMode> {
    return Promise.resolve(this.pending)
  }

  emit(mode: TerminalThemeMode): void {
    this.themeMode = mode

    for (const listener of [...this.listeners]) {
      listener(mode)
    }
  }
}

const darkEnv = {} as NodeJS.ProcessEnv

afterEach(() => {
  resetTerminalThemeMode(darkEnv)
})

describe('explicitLightMode', () => {
  it('reports only deliberate overrides', () => {
    expect(explicitLightMode({ XERXES_TUI_THEME: 'light' } as NodeJS.ProcessEnv)).toBe(true)
    expect(explicitLightMode({ XERXES_TUI_LIGHT: 'off' } as NodeJS.ProcessEnv)).toBe(false)
    expect(explicitLightMode({ XERXES_TUI_BACKGROUND: '#ffffff' } as NodeJS.ProcessEnv)).toBe(true)
    // COLORFGBG is a guess, so the probe is allowed to overrule it.
    expect(explicitLightMode({ COLORFGBG: '0;15' } as NodeJS.ProcessEnv)).toBeNull()
    expect(explicitLightMode(darkEnv)).toBeNull()
  })
})

describe('subscribeTerminalThemeMode', () => {
  it('adopts the already probed answer', () => {
    resetTerminalThemeMode(darkEnv)
    const renderer = new FakeRenderer({ probed: 'light' })
    const applied: Theme[] = []

    subscribeTerminalThemeMode(renderer, theme => applied.push(theme), { env: darkEnv })

    expect(applied).toHaveLength(1)
    expect(currentLightMode()).toBe(true)
    expect(currentBaseTheme().color.text).toBe(LIGHT_THEME.color.text)
  })

  it('adopts a probe that is still in flight', async () => {
    resetTerminalThemeMode(darkEnv)
    const renderer = new FakeRenderer({ pending: 'light' })
    const applied: Theme[] = []

    subscribeTerminalThemeMode(renderer, theme => applied.push(theme), { env: darkEnv })

    expect(applied).toHaveLength(0)

    await Promise.resolve()
    await Promise.resolve()

    expect(applied).toHaveLength(1)
    expect(currentLightMode()).toBe(true)
  })

  it('follows a mid-session switch and back again', () => {
    resetTerminalThemeMode(darkEnv)
    const renderer = new FakeRenderer()
    const applied: Theme[] = []

    subscribeTerminalThemeMode(renderer, theme => applied.push(theme), { env: darkEnv })

    renderer.emit('light')
    renderer.emit('light')
    renderer.emit('dark')

    // The repeated 'light' must not re-render: only real flips are applied.
    expect(applied.map(theme => theme.color.text)).toEqual([LIGHT_THEME.color.text, DARK_THEME.color.text])
  })

  it('never overrules a pinned palette', () => {
    resetTerminalThemeMode(darkEnv)
    const renderer = new FakeRenderer({ probed: 'light' })
    const applied: Theme[] = []

    const stop = subscribeTerminalThemeMode(renderer, theme => applied.push(theme), {
      env: { XERXES_TUI_THEME: 'dark' } as NodeJS.ProcessEnv
    })

    renderer.emit('light')
    stop()

    expect(applied).toHaveLength(0)
    expect(currentLightMode()).toBe(false)
    expect(renderer.listenerCount).toBe(0)
  })

  it('stops following after unsubscribe', () => {
    resetTerminalThemeMode(darkEnv)
    const renderer = new FakeRenderer()
    const applied: Theme[] = []

    subscribeTerminalThemeMode(renderer, theme => applied.push(theme), { env: darkEnv })()

    renderer.emit('light')

    expect(applied).toHaveLength(0)
    expect(renderer.listenerCount).toBe(0)
  })

  it('merges daemon skins over the live palette, not the boot seed', () => {
    resetTerminalThemeMode(darkEnv)
    const renderer = new FakeRenderer()

    subscribeTerminalThemeMode(renderer, () => void 0, { env: darkEnv })
    renderer.emit('light')

    expect(fromSkin({ accent: '#006f94' }).color.text).toBe(LIGHT_THEME.color.text)
  })
})
