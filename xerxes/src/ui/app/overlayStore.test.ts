// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it } from 'vitest'

import {
  clearApprovalOverlay,
  clearClarifyOverlay,
  getOverlayState,
  patchOverlayState,
  resetOverlayState
} from './overlayStore.js'
import { getUiState, patchUiState, resetUiState } from './uiStore.js'

describe('approval overlay lifecycle', () => {
  afterEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('does not let an old response erase a newer approval', () => {
    patchOverlayState({
      approval: {
        command: 'WriteFile',
        description: 'Write the first file',
        requestId: 'permission-1'
      }
    })
    patchOverlayState({
      approval: {
        command: 'ExecCommand',
        description: 'Run the tests',
        requestId: 'permission-2'
      }
    })

    expect(clearApprovalOverlay('permission-1')).toBe(false)
    expect(getOverlayState().approval?.requestId).toBe('permission-2')
    expect(clearApprovalOverlay('permission-2')).toBe(true)
    expect(getOverlayState().approval).toBeNull()
  })
})

describe('provider clarify overlay lifecycle', () => {
  afterEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('keeps a next provider question visible when the prior answer settles', () => {
    patchUiState({ status: 'ready' })
    patchOverlayState({
      clarify: {
        choices: ['kimi-code'],
        question: 'Provider profile:',
        requestId: 'provider-step-1',
        source: 'provider'
      }
    })

    // The native daemon emits the next question before resolving the answer
    // request. The old callback must not clear this replacement overlay.
    patchOverlayState({
      clarify: { choices: ['gpt-5'], question: 'Model:', requestId: 'provider-step-2', source: 'provider' }
    })

    expect(clearClarifyOverlay('provider-step-1')).toBe(false)
    expect(getOverlayState().clarify).toEqual({
      choices: ['gpt-5'],
      question: 'Model:',
      requestId: 'provider-step-2',
      source: 'provider'
    })
    expect(getUiState().status).toBe('ready')
  })

  it('removes only the current provider prompt on cancellation or an error', () => {
    patchUiState({ status: 'ready' })
    patchOverlayState({
      clarify: {
        choices: ['Cancel'],
        question: 'Provider profile:',
        requestId: 'provider-step-1',
        source: 'provider'
      }
    })

    expect(clearClarifyOverlay('provider-step-1')).toBe(true)
    expect(getOverlayState().clarify).toBeNull()
    expect(getUiState().status).toBe('ready')
  })
})
