// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { InputHandlerContext } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { useInputHandlers } from '../app/useInputHandlers.js'
import { resetUiState } from '../app/uiStore.js'
import { copyableMessages, copyLatestAssistantMessage } from '../lib/copyText.js'
import type { Msg } from '../types.js'

vi.mock('../lib/copyText.js', async importOriginal => {
  const actual = await importOriginal<typeof import('../lib/copyText.js')>()

  return { ...actual, copyLatestAssistantMessage: vi.fn(async () => 'copied 9 characters') }
})

const mockedCopyLatest = vi.mocked(copyLatestAssistantMessage)

function Harness({ ctx }: { ctx: InputHandlerContext }) {
  useInputHandlers(ctx)

  return <text>ready</text>
}

const history: Msg[] = [
  { role: 'user', text: 'question' },
  { role: 'assistant', text: 'use /copy' }
]

function makeCtx(overrides: Partial<InputHandlerContext['actions']> = {}) {
  const sys = vi.fn()
  const noop = vi.fn()
  const ctx = {
    actions: {
      answerClarify: noop,
      appendMessage: noop,
      die: noop,
      dispatchQueuedSubmission: noop,
      dispatchSubmission: noop,
      getHistoryItems: () => history,
      guardBusySessionSwitch: () => false,
      newSession: noop,
      sys,
      ...overrides
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

  return { ctx, sys }
}

const renderHarness = async (ctx: InputHandlerContext) => {
  const setup = await testRender(<Harness ctx={ctx} />, { height: 12, width: 60 })

  await act(async () => {
    await Bun.sleep(0)
  })
  await setup.flush()

  return setup
}

describe('Ctrl+O copy keybinding', () => {
  afterEach(() => {
    resetOverlayState()
    resetUiState()
    mockedCopyLatest.mockClear()
  })

  it('Ctrl+O copies the last assistant message and confirms in the transcript', async () => {
    const { ctx, sys } = makeCtx()
    const setup = await renderHarness(ctx)

    try {
      act(() => setup.mockInput.pressKey('o', { ctrl: true }))
      await act(async () => {
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(mockedCopyLatest).toHaveBeenCalledWith(history)
      expect(sys).toHaveBeenCalledWith('copied 9 characters')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('Ctrl+O reports when there is no assistant message yet', async () => {
    mockedCopyLatest.mockResolvedValueOnce('nothing to copy — no assistant message yet')
    const { ctx, sys } = makeCtx({ getHistoryItems: () => [{ role: 'user', text: 'hi' }] })
    const setup = await renderHarness(ctx)

    try {
      act(() => setup.mockInput.pressKey('o', { ctrl: true }))
      await act(async () => {
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(sys).toHaveBeenCalledWith('nothing to copy — no assistant message yet')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('Esc closes an open copy picker overlay and keeps the transcript intact', async () => {
    const { ctx, sys } = makeCtx()
    const setup = await renderHarness(ctx)

    await act(async () => {
      patchOverlayState({ copyPicker: { items: copyableMessages(history) } })
    })
    await setup.flush()
    expect(getOverlayState().copyPicker).not.toBeNull()

    try {
      act(() => setup.mockInput.pressEscape())
      // The renderer holds a bare ESC briefly to disambiguate escape sequences.
      await act(async () => {
        await Bun.sleep(50)
      })
      await setup.flush()

      expect(getOverlayState().copyPicker).toBeNull()
      expect(sys).not.toHaveBeenCalled()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('Ctrl+C closes an open copy picker overlay', async () => {
    const { ctx } = makeCtx()
    const setup = await renderHarness(ctx)

    await act(async () => {
      patchOverlayState({ copyPicker: { items: copyableMessages(history) } })
    })
    await setup.flush()

    try {
      await act(async () => {
        setup.mockInput.pressCtrlC()
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(getOverlayState().copyPicker).toBeNull()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('a single idle Ctrl+C arms exit instead of killing the session', async () => {
    const die = vi.fn()
    const { ctx, sys } = makeCtx({ die })
    const setup = await renderHarness(ctx)

    try {
      await act(async () => {
        setup.mockInput.pressCtrlC()
        await Bun.sleep(0)
      })
      await setup.flush()

      // One stray Ctrl+C — usually a failed copy — must not destroy the session.
      expect(die).not.toHaveBeenCalled()
      expect(sys).toHaveBeenCalledWith('Press Ctrl+C again to exit')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('a second idle Ctrl+C inside the window exits', async () => {
    const die = vi.fn()
    const { ctx } = makeCtx({ die })
    const setup = await renderHarness(ctx)

    try {
      await act(async () => {
        setup.mockInput.pressCtrlC()
        await Bun.sleep(0)
        setup.mockInput.pressCtrlC()
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(die).toHaveBeenCalledTimes(1)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('an idle Ctrl+C after the window lapses arms again instead of exiting', async () => {
    const die = vi.fn()
    const { ctx, sys } = makeCtx({ die })
    const setup = await renderHarness(ctx)

    try {
      await act(async () => {
        setup.mockInput.pressCtrlC()
        await Bun.sleep(0)
      })
      // Let the 2s arming window lapse, then press once more: that is a fresh
      // first press, not a confirmation.
      await act(async () => {
        await Bun.sleep(2_100)
        setup.mockInput.pressCtrlC()
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(die).not.toHaveBeenCalled()
      expect(sys).toHaveBeenCalledTimes(2)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
