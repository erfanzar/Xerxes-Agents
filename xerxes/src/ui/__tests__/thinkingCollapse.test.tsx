// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// Bug B: reasoning/thinking blocks are collapsed by default (one-line
// `▸ thinking` indicator) and expand only on an explicit toggle — per
// message via the header affordance, or globally via Ctrl+T. Covers settled
// transcript rows, the live-streaming reasoning segment, and thinking
// replayed from a resumed session (all the same MessageLine path).
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import {
  getThinkingVisibility,
  resetThinkingVisibility,
  thinkingRowExpanded,
  toggleAllThinking,
  toggleThinkingRow
} from '../app/thinkingVisibilityStore.js'
import { resetUiState } from '../app/uiStore.js'
import { estimatedMsgHeight } from '../lib/virtualHeights.js'
import { MessageLine } from '../opentui/messageLine.js'
import { DEFAULT_THEME, themeForMode } from '../theme.js'
import type { Msg } from '../types.js'

const theme = themeForMode(DEFAULT_THEME, 'code')

const thinkingMsg: Msg = {
  kind: 'trail',
  role: 'system',
  text: '',
  thinking: 'SECRET_TRACE line one\nSECRET_TRACE line two',
  thinkingTokens: 420
}

const flush = async (setup: Awaited<ReturnType<typeof testRender>>) => {
  await act(async () => {
    await Bun.sleep(0)
  })
  await setup.flush()
}

// Native markdown parses on a worker; poll until the text lands in the frame.
const waitForText = async (setup: Awaited<ReturnType<typeof testRender>>, text: string) => {
  for (let pass = 0; pass < 30; pass++) {
    await act(async () => {
      await Bun.sleep(10)
    })
    await setup.flush()

    const frame = setup.captureCharFrame()

    if (frame.includes(text)) {
      return frame
    }
  }

  throw new Error(`timed out waiting for ${text}`)
}

describe('thinking collapse store', () => {
  afterEach(resetThinkingVisibility)

  it('is collapsed by default and flips per row, repeatedly', () => {
    expect(thinkingRowExpanded(getThinkingVisibility(), 'row:1')).toBe(false)

    toggleThinkingRow('row:1')
    expect(thinkingRowExpanded(getThinkingVisibility(), 'row:1')).toBe(true)
    expect(thinkingRowExpanded(getThinkingVisibility(), 'row:2')).toBe(false)

    toggleThinkingRow('row:1')
    expect(thinkingRowExpanded(getThinkingVisibility(), 'row:1')).toBe(false)

    toggleThinkingRow('row:1')
    expect(thinkingRowExpanded(getThinkingVisibility(), 'row:1')).toBe(true)
  })

  it('flips globally while keeping explicit per-row overrides', () => {
    toggleAllThinking()
    expect(thinkingRowExpanded(getThinkingVisibility(), 'row:1')).toBe(true)

    // Explicit collapse of one row survives later global flips.
    toggleThinkingRow('row:1')
    toggleAllThinking()
    expect(thinkingRowExpanded(getThinkingVisibility(), 'row:1')).toBe(false)
    expect(thinkingRowExpanded(getThinkingVisibility(), 'row:2')).toBe(false)

    toggleAllThinking()
    expect(thinkingRowExpanded(getThinkingVisibility(), 'row:1')).toBe(false)
    expect(thinkingRowExpanded(getThinkingVisibility(), 'row:2')).toBe(true)
  })
})

describe('collapsed thinking rendering', () => {
  afterEach(() => {
    resetUiState()
    resetThinkingVisibility()
  })

  it('renders a settled thinking block as a one-line collapsed indicator by default', async () => {
    const setup = await testRender(
      <box flexDirection="column">
        <MessageLine msg={thinkingMsg} msgKey="settled:1" t={theme} />
      </box>,
      { height: 12, width: 80 }
    )

    try {
      await flush(setup)
      const frame = setup.captureCharFrame()

      expect(frame).toContain('▸ thinking')
      expect(frame).toContain('~420 tok')
      expect(frame).not.toContain('SECRET_TRACE')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('toggles a settled thinking block expand → collapse → expand', async () => {
    const setup = await testRender(
      <box flexDirection="column">
        <MessageLine msg={thinkingMsg} msgKey="settled:2" t={theme} />
      </box>,
      { height: 12, width: 80 }
    )

    try {
      await flush(setup)
      expect(setup.captureCharFrame()).not.toContain('SECRET_TRACE')

      act(() => toggleThinkingRow('settled:2'))
      await flush(setup)
      const expanded = setup.captureCharFrame()

      expect(expanded).toContain('▾ thinking')
      expect(expanded).toContain('SECRET_TRACE line one')
      expect(expanded).toContain('SECRET_TRACE line two')

      act(() => toggleThinkingRow('settled:2'))
      await flush(setup)
      expect(setup.captureCharFrame()).not.toContain('SECRET_TRACE')

      act(() => toggleThinkingRow('settled:2'))
      await flush(setup)
      expect(setup.captureCharFrame()).toContain('SECRET_TRACE line one')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('collapses thinking on a settled assistant message (live flush and resume replay share this path)', async () => {
    const replayed: Msg = {
      role: 'assistant',
      text: 'The visible answer.',
      thinking: 'REPLAYED_TRACE from the persisted turn'
    }
    const setup = await testRender(
      <box flexDirection="column">
        <MessageLine msg={replayed} msgKey="replay:1" t={theme} />
      </box>,
      { height: 12, width: 80 }
    )

    try {
      const collapsed = await waitForText(setup, 'The visible answer.')

      expect(collapsed).toContain('▸ thinking')
      expect(collapsed).not.toContain('REPLAYED_TRACE')

      act(() => toggleThinkingRow('replay:1'))
      const expanded = await waitForText(setup, 'REPLAYED_TRACE')

      expect(expanded).toContain('The visible answer.')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('collapses the live-streaming reasoning segment and expands it via Ctrl+T global state', async () => {
    const liveSegment: Msg = {
      kind: 'trail',
      role: 'system',
      text: '',
      thinking: 'LIVE_TRACE streaming reasoning'
    }
    const setup = await testRender(
      <box flexDirection="column">
        <MessageLine msg={liveSegment} msgKey="live-segment:0" t={theme} />
      </box>,
      { height: 12, width: 80 }
    )

    try {
      await flush(setup)
      const collapsed = setup.captureCharFrame()

      expect(collapsed).toContain('▸ thinking')
      expect(collapsed).not.toContain('LIVE_TRACE')

      act(() => toggleAllThinking())
      await flush(setup)
      expect(setup.captureCharFrame()).toContain('LIVE_TRACE')

      act(() => toggleAllThinking())
      await flush(setup)
      expect(setup.captureCharFrame()).not.toContain('LIVE_TRACE')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('keeps two thinking blocks independent', async () => {
    const other: Msg = { kind: 'trail', role: 'system', text: '', thinking: 'OTHER_TRACE block' }
    const setup = await testRender(
      <box flexDirection="column">
        <MessageLine msg={thinkingMsg} msgKey="pair:a" t={theme} />
        <MessageLine msg={other} msgKey="pair:b" t={theme} />
      </box>,
      { height: 12, width: 80 }
    )

    try {
      await flush(setup)
      act(() => toggleThinkingRow('pair:a'))
      await flush(setup)
      const frame = setup.captureCharFrame()

      expect(frame).toContain('SECRET_TRACE line one')
      expect(frame).not.toContain('OTHER_TRACE')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})

describe('thinking height estimate parity', () => {
  it('reserves one row for a collapsed thinking block and the full trace when expanded', () => {
    const collapsed = estimatedMsgHeight(thinkingMsg, 80, { compact: false, details: true, thinkingExpanded: false })
    const expanded = estimatedMsgHeight(thinkingMsg, 80, { compact: false, details: true, thinkingExpanded: true })

    expect(collapsed).toBe(2) // blank trail text row + collapsed header row
    expect(expanded).toBeGreaterThan(collapsed)
  })
})
