// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import type { CopyOutcome } from '../lib/copyText.js'
import { copyableMessages } from '../lib/copyText.js'
import { CopyPicker } from '../opentui/copyPicker.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg } from '../types.js'

const history: Msg[] = [
  { role: 'user', text: 'first user message about copying' },
  { role: 'assistant', text: 'first assistant reply' },
  { role: 'user', text: 'second user message' },
  { role: 'assistant', text: 'latest assistant reply' }
]

const items = copyableMessages(history)

const openPicker = async ({
  copyFn = vi.fn(async (text: string): Promise<CopyOutcome> => ({ backend: 'native', characters: text.length })),
  height = 18,
  width = 96
}: {
  copyFn?: (text: string) => Promise<CopyOutcome>
  height?: number
  width?: number
} = {}) => {
  const onCopied = vi.fn()
  const onCancel = vi.fn()

  patchOverlayState({ copyPicker: { items } })

  const setup = await testRender(
    <box flexDirection="column" height="100%" width="100%">
      <text>transcript stays underneath</text>
      <CopyPicker copyFn={copyFn} onCancel={onCancel} onCopied={onCopied} t={DEFAULT_THEME} />
    </box>,
    { height, width }
  )

  await act(async () => {
    await Bun.sleep(0)
  })
  await setup.flush()

  return { copyFn, onCancel, onCopied, setup }
}

describe('OpenTUI copy picker', () => {
  afterEach(() => resetOverlayState())

  it('lists recent messages of all roles, newest last, selected by default', async () => {
    const { setup } = await openPicker()

    try {
      const frame = setup.captureCharFrame()

      expect(frame).toContain('Copy message')
      expect(frame).toContain('you #1')
      expect(frame).toContain('xerxes #1')
      expect(frame).toContain('you #2')
      expect(frame).toContain('xerxes #2')
      expect(frame).toContain('latest assistant reply')
      expect(frame).toContain('Enter copy')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('Enter copies the selected (newest) message and closes with a confirmation', async () => {
    const copyFn = vi.fn(async (text: string): Promise<CopyOutcome> => ({ backend: 'native', characters: text.length }))
    const { onCopied, setup } = await openPicker({ copyFn })

    try {
      act(() => setup.mockInput.pressEnter())
      await act(async () => {
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(copyFn).toHaveBeenCalledWith('latest assistant reply')
      expect(onCopied).toHaveBeenCalledWith(`copied ${'latest assistant reply'.length} characters`)
      expect(getOverlayState().copyPicker).toBeNull()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('arrow keys move the selection to older messages before copying', async () => {
    const copyFn = vi.fn(async (text: string): Promise<CopyOutcome> => ({ backend: 'native', characters: text.length }))
    const { onCopied, setup } = await openPicker({ copyFn })

    try {
      act(() => setup.mockInput.pressArrow('up'))
      act(() => setup.mockInput.pressArrow('up'))
      act(() => setup.mockInput.pressArrow('up'))
      await setup.flush()
      act(() => setup.mockInput.pressEnter())
      await act(async () => {
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(copyFn).toHaveBeenCalledWith('first user message about copying')
      expect(onCopied).toHaveBeenCalledWith(`copied ${'first user message about copying'.length} characters`)
      expect(getOverlayState().copyPicker).toBeNull()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('Esc cancels without copying and restores the prior screen', async () => {
    const copyFn = vi.fn()
    const { onCancel, setup } = await openPicker({ copyFn: copyFn as never })

    try {
      expect(setup.captureCharFrame()).toContain('Copy message')

      act(() => setup.mockInput.pressEscape())
      // The renderer holds a bare ESC briefly to disambiguate escape sequences.
      await act(async () => {
        await Bun.sleep(50)
      })
      await setup.flush()

      expect(copyFn).not.toHaveBeenCalled()
      expect(onCancel).toHaveBeenCalled()
      expect(getOverlayState().copyPicker).toBeNull()

      const frame = setup.captureCharFrame()
      expect(frame).not.toContain('Copy message')
      expect(frame).toContain('transcript stays underneath')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('a failed copy still closes the picker and reports the failure', async () => {
    const copyFn = vi.fn(async (): Promise<CopyOutcome> => ({ backend: null, characters: 5 }))
    const { onCopied, setup } = await openPicker({ copyFn })

    try {
      act(() => setup.mockInput.pressEnter())
      await act(async () => {
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(onCopied).toHaveBeenCalledWith(expect.stringContaining('copy failed'))
      expect(getOverlayState().copyPicker).toBeNull()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('renders nothing when the overlay is closed', async () => {
    const setup = await testRender(<CopyPicker t={DEFAULT_THEME} />, { height: 12, width: 60 })

    await act(async () => {
      await Bun.sleep(0)
    })
    await setup.flush()

    try {
      expect(setup.captureCharFrame()).not.toContain('Copy message')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
