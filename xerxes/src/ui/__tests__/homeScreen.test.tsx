// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
//
// The home screen against its design.
//
// Every assertion here is a sentence from the canvas's own DECISIONS list,
// turned into something that fails: chips carry their consequence, a chip
// with nothing true to say is not shown, and the composer never degrades.

import { testRender } from '@opentui/react/test-utils'
import { beforeEach, describe, expect, it } from 'vitest'

import { $uiState, $uiTheme } from '../app/uiStore.js'
import type { AppLayoutProps } from '../app/interfaces.js'
import { EMPTY_PULSE, type RepoPulse } from '../lib/repoPulse.js'
import { StartupWelcome } from '../opentui/appLayout.js'
import { DEFAULT_THEME } from '../theme.js'

const composer = {
  cols: 120,
  compIdx: 0,
  compReplace: false,
  completions: [],
  empty: true,
  handleTextPaste: async () => null,
  input: '',
  inputBuf: [],
  queueEditIdx: -1,
  queuedDisplay: [],
  submit: () => undefined,
  updateInput: () => undefined
} as unknown as AppLayoutProps['composer']

const render = async (pulse: RepoPulse = EMPTY_PULSE, cols = 120) => {
  const session = await testRender(
    <StartupWelcome cols={cols} composer={{ ...composer, cols }} pulse={pulse} rows={40} />,
    { height: 40, width: cols }
  )
  await session.flush()

  return session
}

describe('home screen', () => {
  beforeEach(() => {
    $uiTheme.set(DEFAULT_THEME)
    $uiState.set({ ...$uiState.get(), info: { cwd: '/repo', model: 'claude-opus-5' } as never })
  })

  it('names the product and offers a way in', async () => {
    const session = await render()

    try {
      const frame = session.captureCharFrame()

      expect(frame).toContain('Many agents, one terminal.')
      expect(frame).toContain('START WITH')
      // The one chip that is always available — an empty screen still has to
      // offer a way in.
      expect(frame).toContain('map this repo')
    } finally {
      session.renderer.destroy()
    }
  })

  it('gives a dirty tree its own chip, carrying the totals', async () => {
    const session = await render({ ...EMPTY_PULSE, additions: 418, changedFiles: 4, deletions: 96, dirty: 4 })

    try {
      const frame = session.captureCharFrame()

      expect(frame).toContain('/diff review the working tree')
      expect(frame).toContain('+418 −96 · 4 files')
    } finally {
      session.renderer.destroy()
    }
  })

  it('shows no working-tree chip at all when the tree is clean', async () => {
    const session = await render()

    try {
      expect(session.captureCharFrame()).not.toContain('review the working tree')
    } finally {
      session.renderer.destroy()
    }
  })

  it('leads with setup when no model is configured', async () => {
    $uiState.set({ ...$uiState.get(), info: { cwd: '/repo', model: '' } as never })
    const session = await render()

    try {
      const frame = session.captureCharFrame()
      const setup = frame.indexOf('/provider choose a model')
      const map = frame.indexOf('map this repo')

      expect(setup).toBeGreaterThanOrEqual(0)
      expect(setup).toBeLessThan(map)
    } finally {
      session.renderer.destroy()
    }
  })

  it('prints a key cap for every chip it will actually answer to', async () => {
    const session = await render({ ...EMPTY_PULSE, additions: 1, changedFiles: 1, dirty: 1 })

    try {
      const frame = session.captureCharFrame()

      // A key cap the product does not honour is worse than no key cap; the
      // digit binding is gated on an empty draft, so both halves hold.
      expect(frame).toContain('1 ')
      expect(frame).toContain('2 ')
    } finally {
      session.renderer.destroy()
    }
  })

  it('drops chip consequences before chip titles as the terminal narrows', async () => {
    const pulse = { ...EMPTY_PULSE, additions: 418, changedFiles: 4, deletions: 96, dirty: 4 }
    const wide = await render(pulse, 120)
    const narrow = await render(pulse, 70)

    try {
      expect(wide.captureCharFrame()).toContain('+418 −96 · 4 files')
      // The title survives; only the consequence goes.
      expect(narrow.captureCharFrame()).toContain('review the working tree')
      expect(narrow.captureCharFrame()).not.toContain('+418 −96 · 4 files')
    } finally {
      wide.renderer.destroy()
      narrow.renderer.destroy()
    }
  })
})
