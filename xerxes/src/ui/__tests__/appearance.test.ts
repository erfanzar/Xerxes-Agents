// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { $appearance, loadAppearance, saveAppearance } from '../app/appearance.js'
import { $uiTheme, getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { DARK_THEME, LIGHT_THEME, themeForAppearance, themeForMode } from '../theme.js'
import { tuiSlashCompletions } from '../hooks/useCompletion.js'

describe('local TUI appearance', () => {
  afterEach(() => { $appearance.set('chrome'); resetUiState() })
  it('uses the terminal canvas without losing readable foregrounds, popups or selections', () => {
    for (const base of [DARK_THEME, LIGHT_THEME]) for (const mode of ['code', 'plan', 'objective', 'researcher']) {
      const chrome = themeForMode(base, mode)
      const transparent = themeForAppearance(chrome, 'transparent')
      expect(transparent.color.statusBg).toBe('transparent')
      expect(transparent.ds.screen).toBe('transparent')
      expect(transparent.color.text).toBe(chrome.color.text)
      expect(transparent.color.completionBg).toBe(chrome.color.completionBg)
      expect(transparent.color.selectionBg).toBe(chrome.color.selectionBg)
      expect(transparent.color.diffAddedBg).toBe(chrome.color.diffAddedBg)
      expect(themeForAppearance(chrome, 'chrome')).toBe(chrome)
    }
  })
  it('switches live without changing session, working status or base skin; restores Chrome exactly', () => {
    patchUiState({ sid: 'running-session', busy: true, theme: LIGHT_THEME })
    const state = getUiState()
    const chrome = $uiTheme.get()
    $appearance.set('transparent')
    expect($uiTheme.get().color.statusBg).toBe('transparent')
    expect(getUiState()).toBe(state)
    patchUiState({ theme: DARK_THEME })
    expect($uiTheme.get().color.statusBg).toBe('transparent')
    $appearance.set('chrome')
    patchUiState({ theme: LIGHT_THEME })
    expect($uiTheme.get()).toEqual(chrome)
  })
  it('persists both choices and leaves the current appearance unchanged on write failure', async () => {
    const home = await mkdtemp(join(tmpdir(), 'xr-appearance-'))
    try {
      const path = join(home, 'nested', 'appearance.json')
      expect(await loadAppearance(path)).toBe('chrome')
      await saveAppearance('transparent', path)
      expect(await loadAppearance(path)).toBe('transparent')
      await expect(saveAppearance('chrome', join(path, 'impossible'))).rejects.toThrow()
      expect($appearance.get()).toBe('transparent')
      expect(await loadAppearance(path)).toBe('transparent')
      await saveAppearance('chrome', path)
      expect(await loadAppearance(path)).toBe('chrome')
      await Bun.write(path, '{broken')
      expect(await loadAppearance(path)).toBe('chrome')
    } finally { await rm(home, { recursive: true, force: true }) }
  })
  it('is discoverable even when the remote daemon catalog is unavailable', () => {
    expect(JSON.stringify(tuiSlashCompletions('/appe', null))).toContain('/appearance')
  })
})
