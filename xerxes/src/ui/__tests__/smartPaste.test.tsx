// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// Ctrl+V is the Composer's smart-paste chord. Terminals only deliver Cmd+V /
// bracketed paste when the clipboard carries TEXT, so an image-only clipboard
// never reaches the app; the chord must invoke handleTextPaste with
// hotkey: true, which pastes clipboard text or attaches a clipboard image.
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { resetOverlayState } from '../app/overlayStore.js'
import { resetUiState } from '../app/uiStore.js'
import { HOTKEYS } from '../content/hotkeys.js'
import { Composer } from '../opentui/appLayout.js'

function makeComposer(overrides: Record<string, unknown> = {}) {
  return {
    cols: 100,
    compIdx: 0,
    compReplace: 0,
    completions: [],
    empty: true,
    handleTextPaste: vi.fn(),
    input: '',
    queueEditIdx: null,
    queuedDisplay: [],
    submit: vi.fn(),
    updateInput: vi.fn(),
    ...overrides
  } as never
}

describe('Ctrl+V smart paste', () => {
  afterEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('routes Ctrl+V to handleTextPaste with hotkey: true and applies the draft', async () => {
    const handleTextPaste = vi.fn(async () => ({ cursor: 5, value: 'pasted' }))
    const updateInput = vi.fn()
    const composer = makeComposer({ handleTextPaste, updateInput })
    const setup = await testRender(<Composer composer={composer} />, { height: 12, width: 80 })

    await act(async () => {
      await Bun.sleep(0)
    })
    await setup.flush()

    try {
      act(() => setup.mockInput.pressKey('v', { ctrl: true }))
      await act(async () => {
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(handleTextPaste).toHaveBeenCalledTimes(1)
      const arg = handleTextPaste.mock.calls[0]![0] as Record<string, unknown>
      expect(arg.hotkey).toBe(true)
      expect(arg.bracketed).toBe(false)
      expect(typeof arg.cursor).toBe('number')
      expect(typeof arg.value).toBe('string')
      expect(updateInput).toHaveBeenCalledWith('pasted')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('does not route a null paste result to the composer draft', async () => {
    // Image-only clipboard: the hotkey path fires onClipboardPaste and
    // resolves null — the draft must stay untouched.
    const handleTextPaste = vi.fn(async () => null)
    const updateInput = vi.fn()
    const composer = makeComposer({ handleTextPaste, updateInput })
    const setup = await testRender(<Composer composer={composer} />, { height: 12, width: 80 })

    await act(async () => {
      await Bun.sleep(0)
    })
    await setup.flush()

    try {
      act(() => setup.mockInput.pressKey('v', { ctrl: true }))
      await act(async () => {
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(handleTextPaste).toHaveBeenCalledTimes(1)
      expect(updateInput).not.toHaveBeenCalled()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('does not overwrite keystrokes typed while the clipboard read is pending', async () => {
    let resolvePaste!: (value: { cursor: number; value: string }) => void
    const handleTextPaste = vi.fn(
      () => new Promise<{ cursor: number; value: string }>(resolve => (resolvePaste = resolve))
    )
    const updateInput = vi.fn()
    const composer = makeComposer({ handleTextPaste, updateInput })
    const setup = await testRender(<Composer composer={composer} />, { height: 12, width: 80 })

    await act(async () => {
      await Bun.sleep(0)
    })
    await setup.flush()

    try {
      act(() => setup.mockInput.pressKey('v', { ctrl: true }))
      await act(async () => {
        await setup.mockInput.typeText('later')
      })
      await setup.flush()

      await act(async () => {
        resolvePaste({ cursor: 6, value: 'pasted' })
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(updateInput).toHaveBeenLastCalledWith('pastedlater')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('ignores plain v and meta+v so typing and terminal chords stay intact', async () => {
    const handleTextPaste = vi.fn()
    const composer = makeComposer({ handleTextPaste })
    const setup = await testRender(<Composer composer={composer} />, { height: 12, width: 80 })

    await act(async () => {
      await Bun.sleep(0)
    })
    await setup.flush()

    try {
      act(() => setup.mockInput.pressKey('v'))
      act(() => setup.mockInput.pressKey('v', { meta: true }))
      await act(async () => {
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(handleTextPaste).not.toHaveBeenCalled()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('submits the composer value when OpenTUI reports Enter', async () => {
    const submit = vi.fn()
    const updateInput = vi.fn()
    const composer = makeComposer({ submit, updateInput })
    const setup = await testRender(<Composer composer={composer} />, { height: 12, width: 80 })

    await act(async () => {
      await setup.mockInput.typeText('steer this turn')
    })
    await setup.flush()

    try {
      act(() => setup.mockInput.pressEnter())
      await setup.flush()

      expect(submit).toHaveBeenCalledWith('steer this turn')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('documents the chord in the hotkey help', () => {
    const entry = HOTKEYS.find(([chord]) => chord.startsWith('Ctrl+V'))

    expect(entry).toBeDefined()
    expect(entry![1]).toContain('image')
  })
})
