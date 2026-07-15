// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it } from 'vitest'

import { $uiState, $uiTheme, patchUiState, resetUiState } from '../app/uiStore.js'

describe('UI store update batching', () => {
  afterEach(resetUiState)

  it('does not publish no-op state patches or rebuild the mode theme for status-only updates', () => {
    let stateUpdates = 0
    let themeUpdates = 0
    const unlistenState = $uiState.listen(() => stateUpdates++)
    const unlistenTheme = $uiTheme.listen(() => themeUpdates++)
    const initialStateUpdates = stateUpdates
    const initialThemeUpdates = themeUpdates

    patchUiState({ status: 'ready' })
    expect(stateUpdates).toBe(initialStateUpdates)
    expect(themeUpdates).toBe(initialThemeUpdates)

    patchUiState({ status: 'running…' })
    expect(stateUpdates).toBe(initialStateUpdates + 1)
    expect(themeUpdates).toBe(initialThemeUpdates)

    patchUiState(state => ({
      ...state,
      info: { model: 'kimi-for-coding', mode: 'plan', skills: {}, tools: {} }
    }))
    expect(themeUpdates).toBe(initialThemeUpdates + 1)

    unlistenTheme()
    unlistenState()
  })
})
