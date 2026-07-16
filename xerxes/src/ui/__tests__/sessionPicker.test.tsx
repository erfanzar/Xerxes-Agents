// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { SessionActiveListResponse, SessionListResponse } from '../gatewayTypes.js'
import { SessionPicker } from '../opentui/sessionPicker.js'
import { DEFAULT_THEME } from '../theme.js'

const active: SessionActiveListResponse = {
  sessions: [
    {
      current: true,
      id: 'live-main',
      message_count: 4,
      model: 'provider/main-model',
      status: 'idle',
      title: 'Current implementation'
    }
  ]
}

const chats: SessionListResponse = {
  sessions: [
    {
      id: 'parent123',
      kind: 'main',
      message_count: 12,
      preview: 'Authentication audit',
      started_at: Date.now() / 1000 - 120,
      title: 'Authentication audit'
    }
  ]
}

const agents: SessionListResponse = {
  sessions: [
    {
      agent_id: 'researcher',
      id: 'agent123',
      kind: 'subagent',
      message_count: 7,
      model: 'provider/research-model',
      parent_session_id: 'parent123',
      preview: 'Policy review',
      root_session_id: 'parent123',
      started_at: Date.now() / 1000 - 60,
      status: 'completed',
      subagent_id: 'subagent_policy',
      title: 'Policy review'
    }
  ]
}

const picker = async ({
  activeResponse = active,
  agentResponse = agents,
  chatResponse = chats,
  currentSessionId = 'live-main',
  height = 18,
  width = 96
}: {
  activeResponse?: SessionActiveListResponse
  agentResponse?: SessionListResponse
  chatResponse?: SessionListResponse
  currentSessionId?: string
  height?: number
  width?: number
} = {}) => {
  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'session.active_list') return activeResponse
    if (method === 'session.list' && params?.kind === 'all') {
      return { sessions: [...(chatResponse.sessions ?? []), ...(agentResponse.sessions ?? [])] }
    }
    throw new Error(`unexpected request: ${method}`)
  })
  const actions = {
    activateLiveSession: vi.fn(),
    newLiveSession: vi.fn(),
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

describe('OpenTUI session picker histories', () => {
  it('keeps chats clean by default and opens a linked child history from the Agents view', async () => {
    const { actions, request, setup } = await picker()

    try {
      const chatsFrame = setup.captureCharFrame()

      expect(chatsFrame).toContain('[Chats 2]')
      expect(chatsFrame).toContain('Agents 1')
      expect(chatsFrame).toContain('Current implementation')
      expect(chatsFrame).toContain('Authentication audit')
      expect(chatsFrame).not.toContain('Policy review')
      expect(request).toHaveBeenCalledWith('session.list', { kind: 'all', limit: 0 })

      act(() => setup.mockInput.pressArrow('right'))
      await setup.flush()

      const agentsFrame = setup.captureCharFrame()
      expect(agentsFrame).toContain('[Agents 1]')
      expect(agentsFrame).toContain('Policy review')
      expect(agentsFrame).toContain('researcher')
      expect(agentsFrame).toContain('completed')
      expect(agentsFrame).toContain('7 msgs')
      expect(agentsFrame).toContain('← Authentication audit')
      expect(agentsFrame).not.toContain('+  new session')

      act(() => setup.mockInput.pressEnter())
      await setup.flush()

      expect(actions.resumeById).toHaveBeenCalledWith('agent123')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('windows a large agent history list and keeps the last row keyboard-accessible', async () => {
    const manyAgents: SessionListResponse = {
      sessions: Array.from({ length: 80 }, (_, index) => ({
        agent_id: index % 2 ? 'coder' : 'researcher',
        id: `agent-${index}`,
        kind: 'subagent',
        message_count: index + 1,
        parent_session_id: 'parent123',
        preview: `Agent history ${index}`,
        started_at: Date.now() / 1000 - index,
        status: 'completed',
        subagent_id: `subagent-${index}`,
        title: `Agent history ${index}`
      }))
    }
    const { actions, setup } = await picker({ agentResponse: manyAgents, height: 12, width: 72 })

    try {
      act(() => setup.mockInput.pressTab())
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('[Agents 80]')
      expect(setup.captureCharFrame()).toContain('↓ ')

      act(() => setup.mockInput.pressKey('END'))
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('Agent history 79')

      act(() => setup.mockInput.pressEnter())
      await setup.flush()
      expect(actions.resumeById).toHaveBeenCalledWith('agent-79')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('opens on Agents for a current child and keeps Tab navigation and activation usable', async () => {
    const activeChild: SessionActiveListResponse = {
      sessions: [
        {
          agent_id: 'coder',
          current: true,
          id: 'agent-live',
          kind: 'subagent',
          message_count: 5,
          model: 'provider/code-model',
          parent_session_id: 'parent123',
          root_session_id: 'parent123',
          status: 'working',
          subagent_id: 'subagent_live',
          title: 'Live patch review'
        }
      ]
    }
    const { actions, setup } = await picker({ activeResponse: activeChild, currentSessionId: 'agent-live' })

    try {
      expect(setup.captureCharFrame()).toContain('[Agents 2]')
      expect(setup.captureCharFrame()).toContain('Live patch review')
      expect(setup.captureCharFrame()).not.toContain('+  new session')

      act(() => setup.mockInput.pressArrow('left'))
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('[Chats 1]')

      act(() => setup.mockInput.pressTab())
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('[Agents 2]')

      act(() => setup.mockInput.pressEnter())
      await setup.flush()
      expect(actions.activateLiveSession).toHaveBeenCalledWith('agent-live')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('retains view controls and a usable empty state in a narrow terminal', async () => {
    const { setup } = await picker({ agentResponse: { sessions: [] }, height: 9, width: 34 })

    try {
      act(() => setup.mockInput.pressArrow('right'))
      await setup.flush()

      const frame = setup.captureCharFrame()
      expect(frame).toContain('Sessions')
      expect(frame).toContain('[Agents 0]')
      expect(frame).toContain('No agent histories')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('does not resume a child transcript while its native agent still owns it', async () => {
    const running: SessionListResponse = {
      sessions: [{
        ...agents.sessions[0]!,
        resumable: false,
        status: 'running'
      }]
    }
    const { actions, setup } = await picker({ agentResponse: running })

    try {
      act(() => setup.mockInput.pressArrow('right'))
      await setup.flush()
      act(() => setup.mockInput.pressEnter())
      await setup.flush()

      expect(actions.resumeById).not.toHaveBeenCalled()
      expect(setup.captureCharFrame()).toContain('still running')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
