// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// F7 diff viewer: hotkey toggles the overlay, the overlay renders parsed diff
// rows, closes on F7/Esc/q, and Shift+Ctrl+←/→ resizes the shared panel width.
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getPanelWidthDelta, resetPanelWidth } from '../app/panelSizeStore.js'
import { DiffPanelHotkey, DiffPanelOverlay } from '../opentui/diffPanel.js'
import { DEFAULT_THEME } from '../theme.js'

const DIFF_RESULT = {
  kind: 'ok' as const,
  diff: {
    deletions: 1,
    files: 1,
    insertions: 1,
    lines: [
      { kind: 'file' as const, text: 'src/a.ts' },
      { kind: 'hunk' as const, text: '@@ -1,1 +1,1 @@' },
      { kind: 'del' as const, oldLine: 1, text: '-old' },
      { kind: 'add' as const, newLine: 1, text: '+new' }
    ],
    truncated: false,
    untracked: ['draft.ts'],
    untrackedTruncated: false
  }
}

vi.mock('../lib/gitDiff.js', async importOriginal => {
  const original = await importOriginal<typeof import('../lib/gitDiff.js')>()
  return {
    ...original,
    collectGitDiff: vi.fn(async () => DIFF_RESULT)
  }
})

describe('DiffPanelHotkey', () => {
  afterEach(() => {
    resetPanelWidth()
  })

  it('toggles with F7', async () => {
    const transitions: boolean[] = []
    const setup = await testRender(
      <box>
        <DiffPanelHotkey disabled={false} onToggle={open => transitions.push(open)} open={false} />
        <text>ready</text>
      </box>,
      { height: 4, width: 30 }
    )

    try {
      setup.mockInput.pressKey('F7')
      await setup.flush()
      expect(transitions).toEqual([true])
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('ignores F7 when disabled', async () => {
    const transitions: boolean[] = []
    const setup = await testRender(
      <box>
        <DiffPanelHotkey disabled onToggle={open => transitions.push(open)} open={false} />
        <text>ready</text>
      </box>,
      { height: 4, width: 30 }
    )

    try {
      setup.mockInput.pressKey('F7')
      await setup.flush()
      expect(transitions).toEqual([])
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})

describe('DiffPanelOverlay', () => {
  afterEach(() => {
    resetPanelWidth()
  })

  it('renders the diff rows and closes with F7', async () => {
    let closed = 0
    const setup = await testRender(<DiffPanelOverlay onClose={() => closed++} t={DEFAULT_THEME} />, {
      height: 24,
      width: 80
    })

    await act(async () => {
      await Bun.sleep(10)
    })
    await setup.flush()

    try {
      const frame = setup.captureCharFrame()
      expect(frame).toContain('SOURCE CONTROL')
      expect(frame).toContain('CHANGES')
      expect(frame).toContain('src/a.ts')
      expect(frame).toContain('OLD')
      expect(frame).toContain('NEW')
      expect(frame).toContain('UNTRACKED FILES')

      setup.mockInput.pressKey('F7')
      await setup.flush()
      expect(closed).toBe(1)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('resizes the shared panel width with Shift+Ctrl+Left/Right', async () => {
    const setup = await testRender(<DiffPanelOverlay onClose={() => {}} t={DEFAULT_THEME} />, {
      height: 24,
      width: 80
    })

    await act(async () => {
      await Bun.sleep(10)
    })
    await setup.flush()

    try {
      expect(getPanelWidthDelta()).toBe(0)
      setup.mockInput.pressArrow('right', { ctrl: true, shift: true })
      await setup.flush()
      expect(getPanelWidthDelta()).toBe(4)
      setup.mockInput.pressArrow('left', { ctrl: true, shift: true })
      setup.mockInput.pressArrow('left', { ctrl: true, shift: true })
      await setup.flush()
      expect(getPanelWidthDelta()).toBe(-4)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
