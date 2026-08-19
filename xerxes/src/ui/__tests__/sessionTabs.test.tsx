// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'

import type { SessionTab } from '../app/interfaces.js'
import { SessionTabStrip } from '../opentui/appChrome.js'
import { SessionTabsHotkey } from '../opentui/appLayout.js'
import { DEFAULT_THEME } from '../theme.js'

const tabs: SessionTab[] = [
  { id: 'a', status: 'working', title: 'Fix auth' },
  { id: 'b', status: 'idle', title: 'Docs pass' },
  { id: 'c', status: 'waiting', title: 'Review PR' }
]

const render = async (props: {
  activeId?: null | string
  width?: number
  value?: SessionTab[]
} = {}) => {
  const { activeId = 'a', width = 120, value = tabs } = props
  const session = await testRender(
    <SessionTabStrip activeId={activeId} tabs={value} t={DEFAULT_THEME} width={width} />,
    { height: 10, width: Math.max(width, 40) }
  )
  await session.flush()
  return session
}

describe('SessionTabStrip', () => {
  it('renders nothing for a single session', async () => {
    const session = await render({ value: [tabs[0]!] })
    expect(session.captureCharFrame()).not.toContain('Fix auth')
  })

  it('renders a tab per session with its status glyph', async () => {
    const session = await render()
    const out = session.captureCharFrame()
    expect(out).toContain('◆ Fix auth')
    expect(out).toContain('✓ Docs pass')
    expect(out).toContain('? Review PR')
  })

  it('collapses to a position indicator when the strip cannot fit', async () => {
    const session = await render({ width: 20 })
    const out = session.captureCharFrame()
    expect(out).toContain('‹ 1/3 ›')
    expect(out).not.toContain('Fix auth')
  })

  it('shows the active position in the collapsed indicator', async () => {
    const session = await render({ activeId: 'c', width: 20 })
    expect(session.captureCharFrame()).toContain('‹ 3/3 ›')
  })
})

describe('SessionTabsHotkey', () => {
  const hotkey = async (props: {
    activeId?: null | string
    composerEmpty?: boolean
    disabled?: boolean
    value?: SessionTab[]
  } = {}) => {
    const {
      activeId = 'a',
      composerEmpty = true,
      disabled = false,
      value = tabs
    } = props
    const actions = { activateLiveSession: vi.fn() }
    const setup = await testRender(
      <SessionTabsHotkey
        actions={actions}
        activeId={activeId}
        composerEmpty={composerEmpty}
        disabled={disabled}
        tabs={value}
      />,
      { height: 10, width: 80 }
    )
    await setup.flush()
    return { actions, setup }
  }

  it('cycles to the next tab on right and wraps around', async () => {
    const { actions, setup } = await hotkey({ activeId: 'a' })
    try {
      act(() => setup.mockInput.pressArrow('right'))
      expect(actions.activateLiveSession).toHaveBeenCalledWith('b')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('cycles to the previous tab on left and wraps around', async () => {
    const { actions, setup } = await hotkey({ activeId: 'a' })
    try {
      act(() => setup.mockInput.pressArrow('left'))
      expect(actions.activateLiveSession).toHaveBeenCalledWith('c')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('does not steal arrows while the composer holds text', async () => {
    const { actions, setup } = await hotkey({ composerEmpty: false })
    try {
      act(() => setup.mockInput.pressArrow('right'))
      expect(actions.activateLiveSession).not.toHaveBeenCalled()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('does not steal arrows while an overlay is open', async () => {
    const { actions, setup } = await hotkey({ disabled: true })
    try {
      act(() => setup.mockInput.pressArrow('right'))
      expect(actions.activateLiveSession).not.toHaveBeenCalled()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('does nothing with fewer than two tabs', async () => {
    const { actions, setup } = await hotkey({ value: [tabs[0]!] })
    try {
      act(() => setup.mockInput.pressArrow('right'))
      expect(actions.activateLiveSession).not.toHaveBeenCalled()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
