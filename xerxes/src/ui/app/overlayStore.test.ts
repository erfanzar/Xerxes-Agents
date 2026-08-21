// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it } from 'vitest'

import {
  $backgroundHotkeysBlocked,
  clearApprovalOverlay,
  clearClarifyOverlay,
  getOverlayState,
  OVERLAY_BLOCKS_BACKGROUND_HOTKEYS,
  overlayBlocksBackgroundHotkeys,
  patchOverlayState,
  resetFlowOverlays,
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

describe('overlay reset boundaries', () => {
  afterEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('preserves the reasoning picker at a normal turn boundary', () => {
    patchOverlayState({
      confirm: {
        detail: 'flow-scoped',
        onConfirm: () => undefined,
        title: 'Confirm'
      },
      reasoningPicker: true
    })

    resetFlowOverlays()

    expect(getOverlayState().confirm).toBeNull()
    expect(getOverlayState().reasoningPicker).toBe(true)
  })
})

describe('background hotkey blocking policy', () => {
  afterEach(() => {
    resetOverlayState()
  })

  it('does not block while every overlay is closed', () => {
    expect(overlayBlocksBackgroundHotkeys(getOverlayState())).toBe(false)
    expect($backgroundHotkeysBlocked.get()).toBe(false)
  })

  it('blocks for every overlay flagged in the policy table', () => {
    const cases: [string, Parameters<typeof patchOverlayState>[0]][] = [
      ['approval', { approval: { command: 'ExecCommand', description: 'run it', requestId: 'r1' } }],
      ['clarify', { clarify: { choices: ['a'], question: 'q', requestId: 'r1', source: 'provider' } }],
      ['confirm', { confirm: { detail: 'd', onConfirm: () => undefined, title: 't' } }],
      ['copyPicker', { copyPicker: { items: [] } }],
      ['diff', { diff: true }],
      ['modelPicker', { modelPicker: true }],
      ['pager', { pager: { lines: [], offset: 0 } }],
      ['pluginsHub', { pluginsHub: true }],
      ['reasoningPicker', { reasoningPicker: true }],
      ['secret', { secret: { envVar: 'API_KEY', prompt: 'Key?', requestId: 'r1' } }],
      ['sessions', { sessions: true }],
      ['skillsHub', { skillsHub: true }],
      ['sudo', { sudo: { requestId: 'r1' } }],
      ['terminals', { terminals: true }]
    ]

    for (const [name, patch] of cases) {
      resetOverlayState()
      patchOverlayState(patch)
      expect(OVERLAY_BLOCKS_BACKGROUND_HOTKEYS[name as keyof typeof OVERLAY_BLOCKS_BACKGROUND_HOTKEYS]).toBe(true)
      expect(overlayBlocksBackgroundHotkeys(getOverlayState())).toBe(true)
      expect($backgroundHotkeysBlocked.get()).toBe(true)
    }
  })

  it('keeps hotkeys live while only the agents overlay is open, so F6 can close it', () => {
    patchOverlayState({ agents: true, agentsInitialHistoryIndex: 3, agentsInspectId: 'agent-1' })

    expect(overlayBlocksBackgroundHotkeys(getOverlayState())).toBe(false)
    expect($backgroundHotkeysBlocked.get()).toBe(false)
  })

  it('blocks when the agents overlay is stacked on top of a blocking overlay', () => {
    patchOverlayState({ agents: true, modelPicker: true })

    expect(overlayBlocksBackgroundHotkeys(getOverlayState())).toBe(true)
  })
})
