// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { AppLayoutProps, GatewayServices } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import {
  getActiveSideQuestionSlice,
  resetSideQuestionStore,
  setSideQuestionModel
} from '../app/sideQuestionStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { WorkspaceFooter } from '../opentui/appChrome.js'
import { AppLayout, PromptZone } from '../opentui/appLayout.js'
import { SessionPicker } from '../opentui/sessionPicker.js'
import { SideQuestionOverlay } from '../opentui/sideQuestionOverlay.js'
import { DARK_THEME, LIGHT_THEME } from '../theme.js'

afterEach(() => {
  resetOverlayState()
  resetUiState()
  resetSideQuestionStore()
})

describe('OpenTUI terminal UI/UX rendering & interaction fixes', () => {
  it('consumes confirm prompt keys and closes confirm cleanly', async () => {
    const confirmed = vi.fn()
    patchOverlayState({ confirm: { title: 'Delete branch?', onConfirm: confirmed } })
    const screen = await testRender(<PromptZone actions={{} as never} />, { width: 80, height: 18 })
    try {
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Delete branch?')
      act(() => screen.mockInput.pressKey('y'))
      await screen.flush()
      expect(confirmed).toHaveBeenCalledTimes(1)
      expect(getOverlayState().confirm).toBeNull()
    } finally {
      act(() => screen.renderer.destroy())
    }
  })

  it('renders WorkspaceFooter in 1 row with smart path shortening and legible model at 80 cols', async () => {
    const screen = await testRender(
      <WorkspaceFooter
        cwdLabel="~/Documents/Projects/Xerxes-Agents"
        providerModel="anthropic/claude-opus-4-6"
        pulse={{
          branch: 'feature/sidebar-polish',
          dirty: 3,
          ahead: 0,
          behind: 0,
          additions: 10,
          deletions: 2,
          changedFiles: 3
        } as never}
        rightLabel="F6 · F7 · F8 · F9 · F10"
        t={DARK_THEME}
      />,
      { width: 80, height: 4 }
    )
    try {
      await screen.flush()
      const lines = screen.captureCharFrame().split('\n')
      expect(lines[0]).toContain('XERXES')
      expect(lines[0]).toContain('Xerxes-Agents')
      expect(lines[0]).toContain('claude-opus-4-6')
      expect(lines[1]?.trim()).toBe('')
    } finally {
      act(() => screen.renderer.destroy())
    }
  })

  it('collapses top chrome cleanly on 80x24 terminals', async () => {
    patchUiState({
      busy: false,
      sid: 'sess-1234',
      sessionTitle: 'Compact terminal check',
      info: { cwd: '/repo', model: 'claude-sonnet-4.6', mode: 'code', permission_mode: 'default' } as never
    })
    const noop = () => undefined
    const layoutProps = {
      actions: new Proxy({}, { get: () => noop }),
      composer: {
        cols: 80,
        compIdx: 0,
        compReplace: 0,
        completions: [],
        empty: false,
        handleTextPaste: async () => null,
        input: 'hello',
        inputBuf: [],
        queueEditIdx: null,
        queuedDisplay: [],
        submit: noop,
        updateInput: noop
      },
      progress: { showProgressArea: false },
      status: { cwdLabel: '~/Documents/Projects/Xerxes-Agents', statusColor: DARK_THEME.color.muted },
      transcript: {
        historyItems: [],
        scrollRef: { current: null },
        virtualRows: [],
        virtualHistory: {
          start: 0,
          end: 0,
          topSpacer: 0,
          bottomSpacer: 0,
          totalHeight: 0,
          measureRef: () => noop,
          setScrollHandle: noop
        }
      }
    } as unknown as AppLayoutProps

    const screen = await testRender(<AppLayout {...layoutProps} />, { width: 80, height: 24 })
    try {
      await screen.flush()
      const lines = screen.captureCharFrame().split('\n')
      expect(lines[0]).toContain('XERXES')
      expect(lines[0]).toContain('claude-sonnet-4.6')
      expect(lines.join('\n')).not.toContain('session 1234')
    } finally {
      act(() => screen.renderer.destroy())
    }
  })

  it('renders floating SessionPicker with a crisp rounded border and elevated light-theme chrome', async () => {
    const request = vi.fn(async (method: string) =>
      method === 'session.active_list'
        ? { sessions: [{ id: 'live-1', title: 'Border check', status: 'working', current: true, message_count: 2 }] }
        : { sessions: [] }
    )
    const services = { gw: { request }, rpc: vi.fn() } as unknown as GatewayServices
    const screen = await testRender(
      <GatewayProvider value={services}>
        <SessionPicker
          actions={{ activateLiveSession: vi.fn(), resumeById: vi.fn() }}
          currentSessionId="live-1"
          t={LIGHT_THEME}
        />
      </GatewayProvider>,
      { width: 140, height: 32 }
    )
    try {
      await act(async () => {
        await Bun.sleep(0)
      })
      await screen.flush()
      const frame = screen.captureCharFrame()
      expect(frame).toContain('╭')
      expect(frame).toContain('╰')
      expect(frame).toContain('Agent View')
    } finally {
      act(() => screen.renderer.destroy())
    }
  })

  it('renders Markdown answers and supports searchable inline model picker in SideQuestionOverlay', async () => {
    patchUiState({ sid: 'sq-md', info: { model: 'claude-opus-4-6' } as never })
    const rpc = vi.fn(async (method: string) => {
      if (method === 'model.options') {
        return {
          model: 'claude-opus-4-6',
          providers: [
            { slug: 'anthropic', name: 'Anthropic', models: ['claude-opus-4-6', 'claude-haiku-4-5'] },
            { slug: 'google', name: 'Google', models: ['gemini-2.5-flash'] }
          ]
        }
      }
      if (method === 'session.side_question') {
        return {
          ok: true,
          request_id: 'r-md',
          session_id: 'sq-md',
          question: 'Show markdown',
          answer: 'Use **bold** and `parseColor()`:\n\n```ts\nconst ok = true\n```',
          model: 'claude-opus-4-6'
        }
      }
      return { ok: true }
    })

    const screen = await testRender(
      <GatewayProvider value={{ rpc, gw: {} } as unknown as GatewayServices}>
        <SideQuestionOverlay sessionId="sq-md" t={DARK_THEME} />
      </GatewayProvider>,
      { width: 100, height: 28 }
    )
    try {
      await screen.flush()
      await act(async () => {
        await screen.mockInput.typeText('Show markdown')
      })
      await screen.flush()
      await act(async () => {
        screen.mockInput.pressKey('RETURN')
        await Bun.sleep(30)
      })
      await screen.flush()

      let frame = ''
      for (let i = 0; i < 60; i += 1) {
        await Bun.sleep(20)
        await screen.flush()
        frame = screen.captureCharFrame()
        if (frame.includes('bold') && frame.includes('const ok = true')) {
          await screen.waitForVisualIdle()
          frame = screen.captureCharFrame()
          break
        }
      }
      expect(frame).toContain('bold')
      expect(frame).toContain('parseColor()')
      expect(frame).toContain('const ok = true')
      expect(frame).not.toContain('**bold**')
      expect(frame).not.toContain('```ts')

      // Press 'm' in browse mode to open searchable inline model picker
      act(() => screen.mockInput.pressKey('m'))
      await act(async () => {
        await Bun.sleep(20)
      })
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Type to filter models…')
      expect(screen.captureCharFrame()).toContain('gemini-2.5-flash')

      // Type 'flash' to filter and press Enter to select
      await act(async () => {
        await screen.mockInput.typeText('flash')
      })
      await screen.flush()
      const filteredFrame = screen.captureCharFrame()
      expect(filteredFrame).toContain('gemini-2.5-flash')
      expect(filteredFrame).not.toContain('claude-haiku-4-5')

      act(() => screen.mockInput.pressKey('RETURN'))
      await screen.flush()
      expect(getActiveSideQuestionSlice().selectedModel).toBe('gemini-2.5-flash')
      expect(screen.captureCharFrame()).toContain('gemini-2.5-flash')

      // Reopen picker and press Esc: closes picker only, keeping SideQuestionOverlay open
      setSideQuestionModel('gemini-2.5-flash', 'sq-md')
      act(() => screen.mockInput.pressKey('m'))
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Type to filter models…')
      await act(async () => {
        screen.mockInput.pressKey('ESCAPE')
        await Bun.sleep(30)
      })
      await screen.flush()
      expect(screen.captureCharFrame()).not.toContain('Type to filter models…')
      expect(screen.captureCharFrame()).toContain('Side Question')
    } finally {
      act(() => screen.renderer.destroy())
    }
  })
})
