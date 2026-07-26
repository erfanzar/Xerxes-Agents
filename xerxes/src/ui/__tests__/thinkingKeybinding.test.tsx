// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// Ctrl+T is the global thinking expand/collapse chord. Shift+O was rejected:
// a bare shifted letter would swallow capital-O typing in the focused
// composer because the global key handler runs before the textarea.
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { InputHandlerContext } from '../app/interfaces.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { getThinkingVisibility, resetThinkingVisibility } from '../app/thinkingVisibilityStore.js'
import { useInputHandlers } from '../app/useInputHandlers.js'
import { resetUiState } from '../app/uiStore.js'
import { HOTKEYS } from '../content/hotkeys.js'

function Harness({ ctx }: { ctx: InputHandlerContext }) {
  useInputHandlers(ctx)

  return <text>ready</text>
}

function makeCtx() {
  const noop = vi.fn()
  const ctx = {
    actions: {
      answerClarify: noop,
      appendMessage: noop,
      die: noop,
      dispatchQueuedSubmission: noop,
      dispatchSubmission: noop,
      getHistoryItems: () => [],
      guardBusySessionSwitch: () => false,
      newSession: noop,
      sys: noop
    },
    composer: {
      actions: {
        clearIn: noop,
        dequeue: vi.fn(),
        dismissCompletions: noop,
        enqueue: noop,
        handleTextPaste: noop,
        openEditor: vi.fn(async () => {}),
        pushHistory: noop,
        removeQueue: noop,
        replaceQueue: noop,
        setCompIdx: noop,
        setHistoryIdx: noop,
        setInput: noop,
        setInputBuf: noop,
        setPasteSnips: noop,
        setQueueEdit: noop,
        syncQueue: noop
      },
      refs: {
        historyDraftRef: { current: '' },
        historyRef: { current: [] },
        queueEditRef: { current: null },
        queueRef: { current: [] },
        submitRef: { current: noop }
      },
      state: {
        compIdx: 0,
        compReplace: 0,
        completions: [],
        historyIdx: null,
        input: '',
        inputBuf: [],
        pasteSnips: [],
        queueEditIdx: null,
        queuedDisplay: []
      }
    },
    gateway: { gw: {}, rpc: vi.fn() },
    terminal: {
      hasSelection: false,
      scrollRef: { current: null },
      scrollWithSelection: noop,
      selection: {
        captureScrolledRows: noop,
        clearSelection: noop,
        copySelection: vi.fn(async () => ''),
        copySelectionNoClear: vi.fn(async () => ''),
        getState: vi.fn(),
        shiftAnchor: noop,
        shiftSelection: noop,
        version: vi.fn(() => 0)
      },
      stdout: undefined
    },
    voice: {
      enabled: false,
      recordKey: { ch: 'b', mod: 'ctrl', raw: 'ctrl+b' },
      recording: false,
      setProcessing: noop,
      setRecording: noop,
      setVoiceEnabled: noop,
      setVoiceTts: noop
    },
    wheelStep: 1
  } as unknown as InputHandlerContext

  return ctx
}

describe('Ctrl+T thinking toggle keybinding', () => {
  afterEach(() => {
    resetOverlayState()
    resetThinkingVisibility()
    resetUiState()
  })

  it('Ctrl+T flips the global thinking expansion, repeatedly', async () => {
    const setup = await testRender(<Harness ctx={makeCtx()} />, { height: 12, width: 60 })

    await act(async () => {
      await Bun.sleep(0)
    })
    await setup.flush()

    try {
      expect(getThinkingVisibility().allExpanded).toBe(false)

      for (const expected of [true, false, true]) {
        act(() => setup.mockInput.pressKey('t', { ctrl: true }))
        await act(async () => {
          await Bun.sleep(0)
        })
        await setup.flush()
        expect(getThinkingVisibility().allExpanded).toBe(expected)
      }
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('documents the chord in the hotkey help', () => {
    const entry = HOTKEYS.find(([chord]) => chord === 'Ctrl+T')

    expect(entry).toBeDefined()
    expect(entry![1]).toContain('thinking')
  })
})
