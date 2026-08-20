// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { SessionActiveListResponse, SessionListResponse, SessionPeekResponse } from '../gatewayTypes.js'
import { SessionPicker } from '../opentui/sessionPicker.js'
import { DEFAULT_THEME } from '../theme.js'

const active: SessionActiveListResponse = {
  sessions: [
    {
      current: true,
      id: 'live-main',
      message_count: 4,
      model: 'provider/main-model',
      status: 'working',
      title: 'Current implementation'
    },
    {
      activity: 'waiting for your answer',
      id: 'needs-input',
      message_count: 2,
      model: 'provider/main-model',
      status: 'waiting',
      title: 'Choose release target'
    }
  ]
}

const saved: SessionListResponse = {
  sessions: [
    {
      id: 'saved-main',
      kind: 'main',
      last_message_at: Date.now() / 1000 - 120,
      message_count: 12,
      preview: 'first authored prompt must not become a title',
      started_at: Date.now() / 1000 - 86_400,
      title: 'Authentication audit'
    },
    {
      agent_id: 'researcher',
      id: 'child-agent',
      kind: 'subagent',
      message_count: 7,
      parent_session_id: 'saved-main',
      preview: 'Policy review',
      started_at: Date.now() / 1000 - 60,
      status: 'completed',
      subagent_id: 'subagent-policy',
      title: 'Policy review'
    }
  ]
}

const peek: SessionPeekResponse = {
  inflight: { assistant: 'Still checking the final file', streaming: true, user: 'Review the patch' },
  messages: [
    { role: 'user', text: 'Review the patch' },
    { role: 'assistant', text: 'I found one issue.' }
  ],
  session_id: 'live-main',
  status: 'working'
}

const picker = async ({
  activeResponse = active,
  currentSessionId = 'live-main',
  height = 16,
  peekResponse = peek,
  savedResponse = saved,
  width = 100
}: {
  activeResponse?: SessionActiveListResponse
  currentSessionId?: string
  height?: number
  peekResponse?: SessionPeekResponse
  savedResponse?: SessionListResponse
  width?: number
} = {}) => {
  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'session.active_list') return activeResponse
    if (method === 'session.list' && params?.kind === 'all') return savedResponse
    if (method === 'session.peek') return { ...peekResponse, session_id: params?.session_id }
    if (method === 'prompt.background') return { task_id: 'bg-new-task' }
    if (method === 'prompt.submit') return { ok: true }
    if (method === 'session.steer') return { ok: true, status: 'queued' }
    throw new Error(`unexpected request: ${method}`)
  })
  const actions = {
    activateLiveSession: vi.fn(),
    resumeById: vi.fn()
  }
  const services = {
    gw: { request } as unknown as GatewayClient,
    rpc: vi.fn()
  } as unknown as GatewayServices
  const setup = await testRender(
    <GatewayProvider value={services}>
      <SessionPicker actions={actions} currentSessionId={currentSessionId} t={DEFAULT_THEME} />
    </GatewayProvider>,
    { height, width }
  )

  await act(async () => {
    await Bun.sleep(0)
  })
  await setup.flush()

  return { actions, request, setup }
}

describe('OpenTUI Agent View', () => {
  it('renders a full-screen grouped manager for independent chats and keeps subagents in their parent', async () => {
    const { request, setup } = await picker()

    try {
      const frame = setup.captureCharFrame()

      expect(frame).toContain('Agent view')
      expect(frame).toContain('NEEDS INPUT')
      expect(frame).toContain('WORKING')
      expect(frame).toContain('READY')
      expect(frame).toContain('Choose release target')
      expect(frame).toContain('Current implementation')
      expect(frame).toContain('Authentication audit')
      expect(frame).not.toContain('Policy review')
      expect(frame).toContain('Subagents stay inside their parent chat')
      expect(request).toHaveBeenCalledWith('session.list', { kind: 'all', limit: 0 })
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('attaches the highlighted live chat with Right without cancelling it', async () => {
    const { actions, setup } = await picker()

    try {
      // Initial selection follows the currently attached session even after
      // status grouping moves it away from the first row.
      act(() => setup.mockInput.pressArrow('right'))
      await setup.flush()

      expect(actions.activateLiveSession).toHaveBeenCalledWith('live-main')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('dispatches typed text as a new independent live chat while staying in Agent View', async () => {
    const { actions, request, setup } = await picker()

    try {
      await act(async () => setup.mockInput.typeText('audit the parser'))
      act(() => setup.mockInput.pressEnter())
      await act(async () => Bun.sleep(0))
      await setup.flush()

      expect(request).toHaveBeenCalledWith('prompt.background', {
        session_id: 'live-main',
        text: 'audit the parser'
      })
      expect(actions.activateLiveSession).not.toHaveBeenCalled()
      expect(setup.captureCharFrame()).toContain('dispatched bg-new-ta…')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('peeks without attaching and steers a working chat from the preview', async () => {
    const oneWorking: SessionActiveListResponse = { sessions: [active.sessions![0]!] }
    const { actions, request, setup } = await picker({ activeResponse: oneWorking })

    try {
      await act(async () => setup.mockInput.typeText(' '))
      await act(async () => Bun.sleep(0))
      await setup.flush()

      expect(request).toHaveBeenCalledWith('session.peek', { session_id: 'live-main' })
      expect(setup.captureCharFrame()).toContain('assistant: I found one issue.')
      expect(setup.captureCharFrame()).toContain('Still checking the final file')
      expect(actions.activateLiveSession).not.toHaveBeenCalled()

      await act(async () => setup.mockInput.typeText('also check cancellation'))
      act(() => setup.mockInput.pressEnter())
      await act(async () => Bun.sleep(0))

      expect(request).toHaveBeenCalledWith('session.steer', {
        session_id: 'live-main',
        text: 'also check cancellation'
      })
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('replies to an idle chat from peek using a targeted prompt submission', async () => {
    const idle: SessionActiveListResponse = {
      sessions: [{ ...active.sessions![0]!, status: 'idle' }]
    }
    const { request, setup } = await picker({
      activeResponse: idle,
      peekResponse: { ...peek, inflight: null, status: 'idle' }
    })

    try {
      await act(async () => setup.mockInput.typeText(' '))
      await act(async () => Bun.sleep(0))
      await act(async () => setup.mockInput.typeText('continue'))
      act(() => setup.mockInput.pressEnter())
      await act(async () => Bun.sleep(0))

      expect(request).toHaveBeenCalledWith('prompt.submit', {
        session_id: 'live-main',
        text: 'continue'
      })
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('never promotes prompt preview or a session id into an untitled chat title', async () => {
    const { setup } = await picker({
      activeResponse: {
        sessions: [{ ...active.sessions![0]!, activity: 'secret first prompt', title: '' }]
      },
      savedResponse: {
        sessions: [{ ...saved.sessions![0]!, preview: 'secret saved first prompt', title: '' }]
      }
    })

    try {
      const frame = setup.captureCharFrame()
      expect(frame.match(/Untitled chat/g)?.length).toBe(2)
      expect(frame).not.toContain('secret saved first prompt')
      // Activity can describe live work, but remains separate from the title.
      expect(frame).toContain('secret first prompt')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('keeps the manager and dispatch input usable in a narrow terminal', async () => {
    const { setup } = await picker({ height: 8, width: 34 })

    try {
      const frame = setup.captureCharFrame()
      expect(frame).toContain('Agent view')
      expect(frame).toContain('Dispatch a ne')
      expect(frame).not.toContain('Sessions')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
