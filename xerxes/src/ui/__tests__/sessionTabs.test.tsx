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
  onSelect?: (id: string) => void
  width?: number
  value?: SessionTab[]
} = {}) => {
  const { activeId = 'a', onSelect, width = 120, value = tabs } = props
  const session = await testRender(
    <SessionTabStrip activeId={activeId} onSelect={onSelect} tabs={value} t={DEFAULT_THEME} width={width} />,
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
    // v2 chrome: the active tab is a gold dot, not its status glyph — state
    // for the chat you are looking at is already on the composer row.
    expect(out).toContain('● Fix auth')
    expect(out).toContain('○ Docs pass')
    expect(out).toContain('? Review PR')
  })

  it('underlines the active tab on the band row (mockup 02)', async () => {
    const session = await render()
    const out = session.captureCharFrame()
    expect(out).toContain('━')
  })

  it('shows the + affordance and key hints when the width allows', async () => {
    const session = await render({ width: 120 })
    const out = session.captureCharFrame()
    expect(out).toContain('+')
    expect(out).toContain('← switch · ←← agent view')
  })

  it('drops the hints first on a medium strip while keeping every tab', async () => {
    const session = await render({ width: 70 })
    const out = session.captureCharFrame()
    expect(out).toContain('Fix auth')
    expect(out).not.toContain('switch')
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

describe('SessionTabStrip mouse', () => {
  it('activates a tab when its cell is clicked', async () => {
    const onSelect = vi.fn()
    const session = await render({ onSelect })
    try {
      // Strip has paddingX=2; the active "● Fix auth" cell is 12 wide, so the
      // "○ Docs pass" cell spans columns 14..26 — click its label.
      await session.mockMouse.click(16, 0)
      await session.flush()
      expect(onSelect).toHaveBeenCalledWith('b')
    } finally {
      act(() => session.renderer.destroy())
    }
  })

  it("activates the active tab too (a no-op switch is the session's call)", async () => {
    const onSelect = vi.fn()
    const session = await render({ onSelect })
    try {
      await session.mockMouse.click(4, 0)
      await session.flush()
      expect(onSelect).toHaveBeenCalledWith('a')
    } finally {
      act(() => session.renderer.destroy())
    }
  })
})

describe('SessionTabsHotkey', () => {
  const hotkey = async (props: {
    activeId?: null | string
    busy?: boolean
    composerEmpty?: boolean
    disabled?: boolean
    value?: SessionTab[]
  } = {}) => {
    const {
      activeId = 'a',
      busy = false,
      composerEmpty = true,
      disabled = false,
      value = tabs
    } = props
    const actions = { activateLiveSession: vi.fn(), newLiveSession: vi.fn() }
    const openAgentView = vi.fn()
    const setup = await testRender(
      <SessionTabsHotkey
        actions={actions}
        activeId={activeId}
        busy={busy}
        composerEmpty={composerEmpty}
        disabled={disabled}
        onOpenAgentView={openAgentView}
        tabs={value}
      />,
      { height: 10, width: 80 }
    )
    await setup.flush()
    return { actions, openAgentView, setup }
  }

  it('leaves right to the agent view instead of switching or creating sessions', async () => {
    const { actions, openAgentView, setup } = await hotkey({ activeId: 'a' })
    try {
      act(() => setup.mockInput.pressArrow('right'))
      expect(openAgentView).not.toHaveBeenCalled()
      expect(actions.activateLiveSession).not.toHaveBeenCalled()
      expect(actions.newLiveSession).not.toHaveBeenCalled()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('switches one tab left when a tab exists to the left', async () => {
    const { actions, openAgentView, setup } = await hotkey({ activeId: 'b' })
    try {
      act(() => setup.mockInput.pressArrow('left'))
      expect(actions.activateLiveSession).toHaveBeenCalledWith('a')
      expect(openAgentView).not.toHaveBeenCalled()
      expect(actions.newLiveSession).not.toHaveBeenCalled()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('opens the agent view on left from the leftmost tab without switching or creating sessions', async () => {
    const { actions, openAgentView, setup } = await hotkey({ activeId: 'a' })
    try {
      act(() => setup.mockInput.pressArrow('left'))
      expect(openAgentView).toHaveBeenCalledOnce()
      expect(actions.activateLiveSession).not.toHaveBeenCalled()
      expect(actions.newLiveSession).not.toHaveBeenCalled()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('keeps a busy current session running when left opens the agent view', async () => {
    const { actions, openAgentView, setup } = await hotkey({ busy: true, value: [tabs[0]!] })
    try {
      act(() => setup.mockInput.pressArrow('left'))
      expect(openAgentView).toHaveBeenCalledOnce()
      expect(actions.newLiveSession).not.toHaveBeenCalled()
      expect(actions.activateLiveSession).not.toHaveBeenCalled()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('does not steal arrows while the composer holds text', async () => {
    const { actions, openAgentView, setup } = await hotkey({ activeId: 'b', composerEmpty: false })
    try {
      // Left keeps its caret job in a non-empty composer: no tab switch, no
      // agent view.
      act(() => setup.mockInput.pressArrow('left'))
      act(() => setup.mockInput.pressArrow('right'))
      expect(actions.activateLiveSession).not.toHaveBeenCalled()
      expect(openAgentView).not.toHaveBeenCalled()
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

  it('opens the agent view on left when only one idle tab exists', async () => {
    const { actions, openAgentView, setup } = await hotkey({ value: [tabs[0]!] })
    try {
      act(() => setup.mockInput.pressArrow('left'))
      expect(openAgentView).toHaveBeenCalledOnce()
      expect(actions.newLiveSession).not.toHaveBeenCalled()
      expect(actions.activateLiveSession).not.toHaveBeenCalled()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('does nothing on right with fewer than two tabs', async () => {
    const { actions, setup } = await hotkey({ value: [tabs[0]!] })
    try {
      act(() => setup.mockInput.pressArrow('right'))
      expect(actions.activateLiveSession).not.toHaveBeenCalled()
      expect(actions.newLiveSession).not.toHaveBeenCalled()
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
