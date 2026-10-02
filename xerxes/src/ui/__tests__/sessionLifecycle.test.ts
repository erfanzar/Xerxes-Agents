// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { testRender } from '@opentui/react/test-utils'
import type { ScrollBoxHandle } from '../lib/terminalTypes.js'
import { act, createElement } from 'react'
import { describe, expect, it, vi } from 'vitest'

import type { ComposerActions, GatewayRpc } from '../app/interfaces.js'
import { turnController } from '../app/turnController.js'
import { getTurnState, patchTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { hydrateLiveSessionInflight, liveSessionInflightMessages, trimLastExchange, useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { clearSpawnHistory, getSpawnHistory } from '../app/spawnHistoryStore.js'
import { subagentProgressFromSnapshot } from '../domain/subagentProgress.js'
import type { GatewayClient } from '../gatewayClient.js'
import type {
  SessionActivateResponse,
  SessionCreateResponse,
  SessionResumeResponse,
  SetupStatusResponse
} from '../gatewayTypes.js'
import type { Msg } from '../types.js'

it('retains live buffers and manual transcript position before applying a reclaimed journal', async () => {
  resetUiState(); turnController.fullReset()
  patchUiState({ disconnected: true, sid: null, info: { session_id: 'kept' } })
  patchTurnState({ streaming: 'before ', todos: [{ id: 'one', content: 'Keep task context', status: 'in_progress' }],
    activity: [{ id: 1, tone: 'warn', text: 'gateway connection lost · recovering session…' }] })
  const history = vi.fn(), sys = vi.fn()
  const scroll = { getScrollTop: () => 17, isSticky: () => false, scrollTo: vi.fn(), scrollToBottom: vi.fn() }
  const finishSessionRecovery = vi.fn(() => {
    expect(getUiState()).toMatchObject({ sid: 'kept', disconnected: false })
    expect(getTurnState().streaming).toBe('before ')
    patchTurnState({ streaming: 'before offline ' })
  })
  const gw = { request: vi.fn(async () => ({ session_id: 'kept', messages: [], reconnected: true,
    recovery_pending: true, running: true, info: { session_id: 'kept' } })), finishSessionRecovery } as unknown as GatewayClient
  const rpc = vi.fn(async () => ({ provider_configured: true })) as GatewayRpc
  let lifecycle!: ReturnType<typeof useSessionLifecycle>
  const Probe = () => {
    lifecycle = useSessionLifecycle({ colsRef: { current: 120 }, composerActions: { activateSessionQueue: vi.fn(), releaseSwitchHold: vi.fn(), setPasteSnips: vi.fn() } as unknown as ComposerActions,
      gw, panel: vi.fn(), rpc, scrollRef: { current: scroll as unknown as ScrollBoxHandle }, setHistoryItems: history,
      setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(), setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn(), sys })
    return null
  }
  const screen = await testRender(createElement(Probe), { width: 80, height: 24 })
  try {
    act(() => lifecycle.resumeById('kept', { preserveView: true, keepCurrent: true }))
    await vi.waitFor(() => expect(finishSessionRecovery).toHaveBeenCalledWith('kept'))
    expect(history).not.toHaveBeenCalled()
    expect(getTurnState().streaming).toBe('before offline ')
    expect(getTurnState().todos).toHaveLength(1)
    expect(getTurnState().activity).toEqual([])
    await vi.waitFor(() => expect(scroll.scrollTo).toHaveBeenCalledWith(17))
    expect(scroll.scrollToBottom).not.toHaveBeenCalled()
    expect(sys).not.toHaveBeenCalled()
  } finally { act(() => screen.renderer.destroy()); turnController.fullReset(); resetUiState() }
})
const deferred = <T,>() => Promise.withResolvers<T>()

describe('useSessionLifecycle', () => {
  it('returns from Agent View to the attached chat without resetting its live turn', async () => {
    resetUiState()
    turnController.fullReset()
    patchUiState({ busy: true, sid: 'live-main' })
    const liveAgent = subagentProgressFromSnapshot(
      { id: 'agent-1', name: 'structure', status: 'running', title: 'Analyze project structure' },
      0
    )
    patchTurnState({ streaming: 'still working', subagents: [liveAgent] })
    const setHistoryItems = vi.fn()
    const setTurnStartedAt = vi.fn()
    const gw = { request: vi.fn() } as unknown as GatewayClient
    let lifecycle: ReturnType<typeof useSessionLifecycle> | undefined

    const Probe = () => {
      lifecycle = useSessionLifecycle({
        colsRef: { current: 120 },
        composerActions: { activateSessionQueue: vi.fn(), releaseSwitchHold: vi.fn(), setPasteSnips: vi.fn() } as unknown as ComposerActions,
        gw,
        panel: vi.fn(),
        rpc: vi.fn() as GatewayRpc,
        scrollRef: { current: null },
        setHistoryItems,
        setLastUserMsg: vi.fn(),
        setSessionStartedAt: vi.fn(),
        setStickyPrompt: vi.fn(),
        setTurnStartedAt,
        setVoiceProcessing: vi.fn(),
        setVoiceRecording: vi.fn(),
        sys: vi.fn()
      })
      return null
    }

    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })
    try {
      await setup.flush()
      if (!lifecycle) throw new Error('lifecycle hook did not mount')
      lifecycle.activateLiveSession('live-main')
      await setup.flush()

      expect(gw.request).not.toHaveBeenCalled()
      expect(setHistoryItems).not.toHaveBeenCalled()
      expect(setTurnStartedAt).not.toHaveBeenCalled()
      expect(getTurnState().streaming).toBe('still working')
      expect(getTurnState().subagents.map(agent => agent.id)).toEqual(['agent-1'])
      expect(getUiState()).toMatchObject({ busy: true, sid: 'live-main' })
    } finally {
      act(() => setup.renderer.destroy())
      turnController.fullReset()
      resetUiState()
    }
  })

  it('attaches saved history from Agent View while the current chat keeps working', async () => {
    resetUiState()
    turnController.fullReset()
    patchUiState({ busy: true, sid: 'live-main' })
    const sys = vi.fn()
    const gw = {
      request: vi.fn(async (method: string) => {
        if (method !== 'session.resume') throw new Error(`unexpected gateway request: ${method}`)
        return { messages: [], resumed: 'saved-main', session_id: 'saved-main', status: 'idle' }
      })
    } as unknown as GatewayClient
    const rpc = vi.fn(async (method: string) =>
      method === 'setup.status' ? { provider_configured: true } : null
    ) as GatewayRpc
    let lifecycle: ReturnType<typeof useSessionLifecycle> | undefined

    const Probe = () => {
      lifecycle = useSessionLifecycle({
        colsRef: { current: 120 },
        composerActions: { activateSessionQueue: vi.fn(), releaseSwitchHold: vi.fn(), setPasteSnips: vi.fn() } as unknown as ComposerActions,
        gw,
        panel: vi.fn(),
        rpc,
        scrollRef: { current: null },
        setHistoryItems: vi.fn(),
        setLastUserMsg: vi.fn(),
        setSessionStartedAt: vi.fn(),
        setStickyPrompt: vi.fn(),
        setTurnStartedAt: vi.fn(),
        setVoiceProcessing: vi.fn(),
        setVoiceRecording: vi.fn(),
        sys
      })
      return null
    }

    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })
    try {
      await setup.flush()
      if (!lifecycle) throw new Error('lifecycle hook did not mount')
      lifecycle.attachSavedSession('saved-main')
      await act(async () => Bun.sleep(0))
      await setup.flush()

      expect(gw.request).toHaveBeenCalledWith('session.resume', { cols: 120, session_id: 'saved-main' })
      expect(sys).not.toHaveBeenCalledWith('interrupt the current turn before trying to switch sessions')
      expect(getUiState().sid).toBe('saved-main')
    } finally {
      act(() => setup.renderer.destroy())
      turnController.fullReset()
      resetUiState()
    }
  })

  it('lets only the newest overlapping new-session request replace visible state', async () => {
    resetUiState()
    const firstSetup = deferred<null | SetupStatusResponse>()
    const secondSetup = deferred<null | SetupStatusResponse>()
    const calls: string[] = []
    const rpc = vi.fn(async (method: string) => {
      calls.push(method)
      if (method === 'setup.status') {
        return calls.filter(call => call === 'setup.status').length === 1
          ? firstSetup.promise
          : secondSetup.promise
      }
      if (method === 'session.create') {
        return { session_id: 'newest' } satisfies SessionCreateResponse
      }
      return null
    }) as GatewayRpc
    const activateSessionQueue = vi.fn()
    let lifecycle: ReturnType<typeof useSessionLifecycle> | undefined

    const Probe = () => {
      lifecycle = useSessionLifecycle({
        colsRef: { current: 80 },
        composerActions: { activateSessionQueue, releaseSwitchHold: vi.fn(), setPasteSnips: vi.fn() } as unknown as ComposerActions,
        gw: {} as GatewayClient,
        panel: vi.fn(),
        rpc,
        scrollRef: { current: null },
        setHistoryItems: vi.fn(),
        setLastUserMsg: vi.fn(),
        setSessionStartedAt: vi.fn(),
        setStickyPrompt: vi.fn(),
        setVoiceProcessing: vi.fn(),
        setVoiceRecording: vi.fn(),
        sys: vi.fn(),
      })
      return null
    }

    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })
    try {
      await setup.flush()
      if (!lifecycle) throw new Error('lifecycle hook did not mount')
      const first = lifecycle.newSession()
      const second = lifecycle.newSession()
      secondSetup.resolve({ provider_configured: true })
      await second
      firstSetup.resolve({ provider_configured: true })
      await first

      expect(calls.filter(call => call === 'session.create')).toHaveLength(1)
      expect(getUiState().sid).toBe('newest')
      expect(activateSessionQueue).toHaveBeenCalledWith('newest')
    } finally {
      act(() => setup.renderer.destroy())
      resetUiState()
    }
  })

  it('serializes create with a newer activation without deadlocking', async () => {
    resetUiState()
    const create = deferred<SessionCreateResponse>()
    const activate = deferred<SessionActivateResponse>()
    const mutations: string[] = []
    const rpc = vi.fn((method: string) => {
      if (method === 'setup.status') {
        return Promise.resolve({ provider_configured: true })
      }
      if (method === 'session.create') {
        mutations.push(method)
        return create.promise
      }
      return Promise.resolve(null)
    }) as GatewayRpc
    const gw = {
      request: vi.fn((method: string) => {
        mutations.push(method)
        if (method === 'session.activate') return activate.promise
        throw new Error(`unexpected gateway request: ${method}`)
      })
    } as unknown as GatewayClient
    const activateSessionQueue = vi.fn()
    let lifecycle: ReturnType<typeof useSessionLifecycle> | undefined

    const Probe = () => {
      lifecycle = useSessionLifecycle({
        colsRef: { current: 80 },
        composerActions: { activateSessionQueue, releaseSwitchHold: vi.fn(), setPasteSnips: vi.fn() } as unknown as ComposerActions,
        gw,
        panel: vi.fn(),
        rpc,
        scrollRef: { current: null },
        setHistoryItems: vi.fn(),
        setLastUserMsg: vi.fn(),
        setSessionStartedAt: vi.fn(),
        setStickyPrompt: vi.fn(),
        setVoiceProcessing: vi.fn(),
        setVoiceRecording: vi.fn(),
        sys: vi.fn(),
      })
      return null
    }

    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })
    try {
      await setup.flush()
      if (!lifecycle) throw new Error('lifecycle hook did not mount')
      const creating = lifecycle.newSession()
      await setup.flush()
      expect(mutations).toEqual(['session.create'])

      lifecycle.activateLiveSession('newest')
      expect(mutations).toEqual(['session.create'])

      create.resolve({ session_id: 'older' })
      await creating
      await setup.flush()
      expect(mutations).toEqual(['session.create', 'session.activate'])

      activate.resolve({ messages: [], session_id: 'newest', session_key: 'key:newest' })
      await setup.flush()
      expect(getUiState().sid).toBe('newest')
      expect(activateSessionQueue).toHaveBeenCalledTimes(1)
      expect(activateSessionQueue).toHaveBeenCalledWith('newest')
    } finally {
      act(() => setup.renderer.destroy())
      resetUiState()
    }
  })

  it('serializes overlapping resume and activate client commits and keeps the newest selection visible', async () => {
    resetUiState()
    const resume = deferred<SessionResumeResponse>()
    const activate = deferred<SessionActivateResponse>()
    const requests: string[] = []
    const committedKeys: string[] = []
    const gw = {
      request: vi.fn((method: string) => {
        requests.push(method)
        if (method === 'session.resume') {
          return resume.promise.then(result => {
            committedKeys.push(result.session_id)
            return result
          })
        }
        if (method === 'session.activate') {
          return activate.promise.then(result => {
            committedKeys.push(result.session_id)
            return result
          })
        }
        throw new Error(`unexpected gateway request: ${method}`)
      })
    } as unknown as GatewayClient
    const rpc = vi.fn(async (method: string) =>
      method === 'setup.status' ? { provider_configured: true } : null
    ) as GatewayRpc
    const activateSessionQueue = vi.fn()
    let lifecycle: ReturnType<typeof useSessionLifecycle> | undefined

    const Probe = () => {
      lifecycle = useSessionLifecycle({
        colsRef: { current: 80 },
        composerActions: { activateSessionQueue, releaseSwitchHold: vi.fn(), setPasteSnips: vi.fn() } as unknown as ComposerActions,
        gw,
        panel: vi.fn(),
        rpc,
        scrollRef: { current: null },
        setHistoryItems: vi.fn(),
        setLastUserMsg: vi.fn(),
        setSessionStartedAt: vi.fn(),
        setStickyPrompt: vi.fn(),
        setVoiceProcessing: vi.fn(),
        setVoiceRecording: vi.fn(),
        sys: vi.fn(),
      })
      return null
    }

    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })
    try {
      await setup.flush()
      if (!lifecycle) throw new Error('lifecycle hook did not mount')
      lifecycle.resumeById('older')
      await setup.flush()
      expect(requests).toEqual(['session.resume'])
      lifecycle.activateLiveSession('newest')

      // The newer gateway mutation must wait; otherwise the older resume can
      // finish last and overwrite GatewayClient's active session key.
      expect(requests).toEqual(['session.resume'])
      resume.resolve({ messages: [], resumed: 'older', session_id: 'older' })
      await setup.flush()
      expect(requests).toEqual(['session.resume', 'session.activate'])
      activate.resolve({ messages: [], session_id: 'newest', session_key: 'key:newest' })
      await setup.flush()
      expect(getUiState().sid).toBe('newest')

      expect(committedKeys).toEqual(['older', 'newest'])
      expect(activateSessionQueue).toHaveBeenCalledTimes(1)
      expect(activateSessionQueue).toHaveBeenCalledWith('newest')
    } finally {
      act(() => setup.renderer.destroy())
      resetUiState()
    }
  })
  it('filters internal prompts from the inflight user row on reattach', () => {
    expect(
      liveSessionInflightMessages({ user: "[Skill 'deepscan' activated]\n\n## Skill: deepscan\nprivate" })
    ).toEqual([])
    expect(liveSessionInflightMessages({ user: 'Please compact this conversation: now' })).toEqual([])
    expect(liveSessionInflightMessages({ user: 'visible prompt' })).toEqual([{ role: 'user', text: 'visible prompt' }])
  })

  it('hydrates the mid-turn inflight trail as live tool state', () => {
    turnController.fullReset()
    try {
      hydrateLiveSessionInflight({
        // Long enough to clear the reasoning filter's 32-char lookahead tail.
        assistant: 'partial reply that keeps streaming past the filter tail boundary',
        started_at: 1_700_000_000,
        streaming: true,
        thinking: 'trace so far',
        tools: [
          {
            arguments: '{"agents":[{"title":"Analyze structure","prompt":"truncated…',
            context: '2 agents: Analyze structure, Audit security',
            duration_ms: 184_300,
            id: 'call-spawn',
            name: 'SpawnAgents',
            ok: true
          },
          { arguments: '{"path":"a.ts"}', duration_ms: 200, id: 'call-1', name: 'ReadFile', ok: true },
          { arguments: '{"path":"b.ts"}', id: 'call-2', name: 'WriteFile' }
        ],
        user: 'go'
      })

      const turn = getTurnState()
      // The settled call renders as a completed trail line; the call still in
      // flight stays active so its late tool_result settles it live.
      expect(turn.streamPendingTools.some(line => line.includes('a.ts'))).toBe(true)
      const spawn = turn.streamPendingTools.find(line => line.includes('Spawn Agents'))
      expect(spawn).toContain('2 agents: Analyze structure, Audit security')
      expect(spawn).not.toContain('{"agents"')
      expect(turn.streamPendingTools).toHaveLength(2)
      expect(turn.tools.map(tool => tool.name)).toEqual(['WriteFile'])
      expect(turn.streaming).toContain('partial reply that keeps')
    } finally {
      turnController.fullReset()
    }
  })

  it('rehydrates persisted subagent snapshots as trail progress rows', () => {
    const row = subagentProgressFromSnapshot(
      {
        agent_id: 'explorer',
        api_calls: 3,
        created_at: '2026-07-16T10:00:00.000Z',
        id: 'sa-1',
        model: 'gpt-4o',
        name: 'scan',
        parent_id: null,
        status: 'completed',
        summary: 'scanned the repo',
        title: 'Scan repo',
        tool_count: 5
      },
      0
    )

    expect(row).toMatchObject({
      agentType: 'explorer',
      apiCalls: 3,
      goal: 'Scan repo',
      id: 'sa-1',
      index: 0,
      model: 'gpt-4o',
      name: 'scan',
      parentId: null,
      status: 'completed',
      summary: 'scanned the repo',
      toolCount: 5
    })
    expect(row.startedAt).toBe(Date.parse('2026-07-16T10:00:00.000Z'))
    expect(subagentProgressFromSnapshot({ id: 'sa-2', status: 'cancelled' }, 1).status).toBe('interrupted')
    expect(subagentProgressFromSnapshot({ id: 'sa-3', status: 'mystery' }, 2).status).toBe('interrupted')
    expect(subagentProgressFromSnapshot({ closed: true, id: 'sa-4', status: 'mystery' }, 3).status).toBe('completed')
    // A handle spawned but not yet given a turn is waiting, not dead.
    expect(subagentProgressFromSnapshot({ id: 'sa-5', status: 'idle' }, 4).status).toBe('queued')
    expect(subagentProgressFromSnapshot({ closed: true, id: 'sa-6', status: 'idle' }, 5).status).toBe('completed')
  })

  it('reattaches a mid-turn session with its unfinished subagents still live', async () => {
    resetUiState()
    turnController.fullReset()
    clearSpawnHistory()
    const historyItems: Msg[][] = []
    const gw = {
      request: vi.fn(async (method: string) => {
        if (method !== 'session.activate') throw new Error(`unexpected gateway request: ${method}`)
        return {
          messages: [],
          running: true,
          session_id: 'busy-session',
          session_key: 'key:busy',
          status: 'working',
          subagent_snapshots: [
            { id: 'sa-done', name: 'scout', status: 'completed', summary: 'found it', title: 'Scout repo' },
            { id: 'sa-live', name: 'auditor', status: 'running', title: 'Audit runtime' },
            { id: 'sa-queued', name: 'waiter', status: 'idle', title: 'Queued work' }
          ]
        } satisfies SessionActivateResponse
      })
    } as unknown as GatewayClient
    let lifecycle: ReturnType<typeof useSessionLifecycle> | undefined

    const Probe = () => {
      lifecycle = useSessionLifecycle({
        colsRef: { current: 80 },
        composerActions: { activateSessionQueue: vi.fn(), releaseSwitchHold: vi.fn(), setPasteSnips: vi.fn() } as unknown as ComposerActions,
        gw,
        panel: vi.fn(),
        rpc: vi.fn() as GatewayRpc,
        scrollRef: { current: null },
        setHistoryItems: vi.fn((next: Msg[] | ((prev: Msg[]) => Msg[])) => {
          historyItems.push(typeof next === 'function' ? next([]) : next)
        }),
        setLastUserMsg: vi.fn(),
        setSessionStartedAt: vi.fn(),
        setStickyPrompt: vi.fn(),
        setVoiceProcessing: vi.fn(),
        setVoiceRecording: vi.fn(),
        sys: vi.fn()
      })
      return null
    }

    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })
    try {
      await setup.flush()
      if (!lifecycle) throw new Error('lifecycle hook did not mount')
      lifecycle.activateLiveSession('busy-session')
      await setup.flush()

      // Unfinished children come back LIVE: this is what repopulates the F6
      // rail's WORKING count and gives every subsequent update-only
      // `subagent.*` event a row to land on.
      expect(getTurnState().subagents.map(agent => agent.id)).toEqual(['sa-live', 'sa-queued'])
      expect(getTurnState().subagents.map(agent => agent.status)).toEqual(['running', 'queued'])

      // Finished children stay history — a folded trail card plus a snapshot.
      const trail = historyItems.at(-1)?.find(item => item.kind === 'trail')
      expect(trail?.subagents?.map(agent => agent.id)).toEqual(['sa-done'])
      expect(getSpawnHistory().flatMap(snapshot => snapshot.subagents.map(agent => agent.id))).toEqual(['sa-done'])
    } finally {
      act(() => setup.renderer.destroy())
      turnController.fullReset()
      clearSpawnHistory()
      resetUiState()
    }
  })

  it('archives every persisted subagent when the reattached session is idle', async () => {
    resetUiState()
    turnController.fullReset()
    clearSpawnHistory()
    const gw = {
      request: vi.fn(async () => ({
        messages: [],
        running: false,
        session_id: 'idle-session',
        session_key: 'key:idle',
        status: 'idle',
        subagent_snapshots: [{ id: 'orphan', name: 'ghost', status: 'running', title: 'Orphaned child' }]
      } satisfies SessionActivateResponse))
    } as unknown as GatewayClient
    let lifecycle: ReturnType<typeof useSessionLifecycle> | undefined

    const Probe = () => {
      lifecycle = useSessionLifecycle({
        colsRef: { current: 80 },
        composerActions: { activateSessionQueue: vi.fn(), releaseSwitchHold: vi.fn(), setPasteSnips: vi.fn() } as unknown as ComposerActions,
        gw,
        panel: vi.fn(),
        rpc: vi.fn() as GatewayRpc,
        scrollRef: { current: null },
        setHistoryItems: vi.fn(),
        setLastUserMsg: vi.fn(),
        setSessionStartedAt: vi.fn(),
        setStickyPrompt: vi.fn(),
        setVoiceProcessing: vi.fn(),
        setVoiceRecording: vi.fn(),
        sys: vi.fn()
      })
      return null
    }

    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })
    try {
      await setup.flush()
      lifecycle?.activateLiveSession('idle-session')
      await setup.flush()

      // A manifest that still says "running" on an idle session describes
      // children orphaned by a daemon restart. Re-animating those would show
      // work that can never report again as permanently in flight.
      expect(getTurnState().subagents).toEqual([])
      expect(getSpawnHistory().flatMap(snapshot => snapshot.subagents.map(agent => agent.id))).toEqual(['orphan'])
    } finally {
      act(() => setup.renderer.destroy())
      turnController.fullReset()
      clearSpawnHistory()
      resetUiState()
    }
  })
})

it.each([false, true])('restores todos and messages after visiting another chat (running=%s)', async (running) => {
  resetUiState(); turnController.fullReset()
  const todos = Array.from({ length: 5 }, (_, i) => ({ id: String(i), content: `Task ${i}`, status: i < 3 ? 'completed' : 'pending' }))
  const history = vi.fn()
  const messages = Array.from({ length: 24 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', text: `Message ${i}` }))
  const gw = { request: vi.fn(async (_method: string, params: { session_id: string }) => ({ session_id: params.session_id, messages: params.session_id === 'original' ? messages : [], todos: params.session_id === 'original' ? todos : [], running: params.session_id === 'original' && running })) } as unknown as GatewayClient
  let lifecycle!: ReturnType<typeof useSessionLifecycle>
  const Probe = () => {
    lifecycle = useSessionLifecycle({ colsRef: { current: 120 }, composerActions: { activateSessionQueue: vi.fn(), releaseSwitchHold: vi.fn(), setPasteSnips: vi.fn() } as unknown as ComposerActions, gw, panel: vi.fn(), rpc: vi.fn() as GatewayRpc, scrollRef: { current: null }, setHistoryItems: history, setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(), setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn(), sys: vi.fn() })
    return null
  }
  const screen = await testRender(createElement(Probe), { width: 120, height: 30 })
  try {
    for (const id of ['original', 'other', 'original']) {
      act(() => lifecycle.activateLiveSession(id))
      await vi.waitFor(() => expect(getUiState().sid).toBe(id))
      expect(getTurnState().todos).toHaveLength(id === 'original' ? 5 : 0)
    }
    expect(getTurnState().todos.filter(todo => todo.status === 'completed')).toHaveLength(3)
    expect(history.mock.lastCall?.[0]).toHaveLength(24)
    turnController.startMessage()
    expect(getTurnState().todos).toHaveLength(5)
    turnController.recordTodos([])
    expect(getTurnState().todos).toHaveLength(0)
  } finally { act(() => screen.renderer.destroy()); turnController.fullReset(); resetUiState() }
})

describe('session switch delivery', () => {
  const mount = async (gw: GatewayClient, rpc: GatewayRpc = vi.fn(async () => ({ provider_configured: true })) as GatewayRpc,
    setHistoryItems: (...args: unknown[]) => void = vi.fn()) => {
    let lifecycle: ReturnType<typeof useSessionLifecycle> | undefined
    const composerActions = { activateSessionQueue: vi.fn(), releaseSwitchHold: vi.fn(), setPasteSnips: vi.fn() }
    const sys = vi.fn()
    const Probe = () => {
      lifecycle = useSessionLifecycle({ colsRef: { current: 80 }, composerActions: composerActions as unknown as ComposerActions,
        gw, panel: vi.fn(), rpc, scrollRef: { current: null }, setHistoryItems, setLastUserMsg: vi.fn(),
        setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(), setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn(), sys })
      return null
    }
    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })
    await setup.flush()
    if (!lifecycle) throw new Error('lifecycle hook did not mount')
    return { composerActions, lifecycle, setup, sys }
  }

  it('marks the composer as switching until a tab switch or /new commits or fails', async () => {
    resetUiState(); turnController.fullReset()
    patchUiState({ sid: 'old' })
    const activate = deferred<SessionActivateResponse>()
    const create = deferred<null | SessionCreateResponse>()
    const gw = { request: vi.fn(() => activate.promise) } as unknown as GatewayClient
    const rpc = vi.fn((method: string) => method === 'session.create' ? create.promise : Promise.resolve({ provider_configured: true })) as GatewayRpc
    const { lifecycle, setup } = await mount(gw, rpc)
    try {
      lifecycle.activateLiveSession('next')
      expect(getUiState()).toMatchObject({ sid: 'old', switching: true })
      activate.resolve({ messages: [], session_id: 'next' })
      await vi.waitFor(() => expect(getUiState()).toMatchObject({ sid: 'next', switching: false }))

      const creating = lifecycle.newSession()
      await setup.flush()
      expect(getUiState()).toMatchObject({ sid: 'next', switching: true })
      create.resolve(null)
      await creating
      expect(getUiState()).toMatchObject({ sid: 'next', switching: false })
    } finally { act(() => setup.renderer.destroy()); turnController.fullReset(); resetUiState() }
  })

  it('releases the stream held during activation only after resetting onto the new session', async () => {
    resetUiState(); turnController.fullReset()
    patchUiState({ sid: 'old' })
    turnController.bufRef = 'old view text'
    const finishSessionRecovery = vi.fn(() => {
      expect(getUiState().sid).toBe('next')
      expect(turnController.bufRef).toBe('one ')
      // A delta held during the switch now appends to the hydrated snapshot.
      turnController.recordMessageDelta({ text: 'two' })
    })
    const gw = { request: vi.fn(async () => ({ messages: [], session_id: 'next', recovery_pending: true, running: true, status: 'working',
      inflight: { assistant: 'one ', streaming: true } })), finishSessionRecovery } as unknown as GatewayClient
    const { lifecycle, setup } = await mount(gw)
    try {
      lifecycle.activateLiveSession('next')
      await vi.waitFor(() => expect(finishSessionRecovery).toHaveBeenCalledWith('next'))
      expect(turnController.bufRef).toBe('one two')
    } finally { act(() => setup.renderer.destroy()); turnController.fullReset(); resetUiState() }
  })

  it('discards the held stream of an activation superseded before it could be shown', async () => {
    resetUiState(); turnController.fullReset()
    patchUiState({ sid: 'old' })
    const first = deferred<SessionActivateResponse>()
    const finishSessionRecovery = vi.fn(), discardSessionRecovery = vi.fn()
    const gw = { request: vi.fn((_method: string, params: { session_id: string }) => params.session_id === 'first'
      ? first.promise
      : Promise.resolve({ messages: [], session_id: 'second', recovery_pending: true })), finishSessionRecovery, discardSessionRecovery } as unknown as GatewayClient
    const { lifecycle, setup } = await mount(gw)
    try {
      lifecycle.activateLiveSession('first')
      await vi.waitFor(() => expect(gw.request).toHaveBeenCalledTimes(1))
      lifecycle.activateLiveSession('second')
      first.resolve({ messages: [], session_id: 'first', recovery_pending: true })
      await vi.waitFor(() => expect(finishSessionRecovery).toHaveBeenCalledWith('second'))
      expect(discardSessionRecovery).toHaveBeenCalledWith('first')
      expect(finishSessionRecovery).not.toHaveBeenCalledWith('first')
      expect(getUiState()).toMatchObject({ sid: 'second', switching: false })
    } finally { act(() => setup.renderer.destroy()); turnController.fullReset(); resetUiState() }
  })

  it.each(['activate', 'resume'] as const)('disarms the transport hold when committing a %s fails', async kind => {
    resetUiState(); turnController.fullReset()
    patchUiState({ sid: 'old' })
    const finishSessionRecovery = vi.fn(), discardSessionRecovery = vi.fn()
    const gw = { request: vi.fn(async () => ({ messages: [], session_id: 'next', recovery_pending: true })),
      finishSessionRecovery, discardSessionRecovery } as unknown as GatewayClient
    const { lifecycle, setup, sys } = await mount(gw, undefined, () => { throw new Error('render failed') })
    try {
      if (kind === 'activate') lifecycle.activateLiveSession('next')
      else lifecycle.resumeById('next')
      await vi.waitFor(() => expect(sys).toHaveBeenCalledWith('error: render failed'))
      // Left armed, the hold would buffer every later event until the 1 MiB
      // cap destroyed the socket.
      expect(discardSessionRecovery).toHaveBeenCalledWith('next')
      expect(finishSessionRecovery).not.toHaveBeenCalled()
      expect(getUiState().switching).toBe(false)
    } finally { act(() => setup.renderer.destroy()); turnController.fullReset(); resetUiState() }
  })

  it('disarms the transport hold when the switch answer is not a session', async () => {
    resetUiState(); turnController.fullReset()
    patchUiState({ sid: 'old' })
    const discardSessionRecovery = vi.fn()
    const gw = { request: vi.fn(async () => 'garbage'), discardSessionRecovery } as unknown as GatewayClient
    const { lifecycle, setup, sys } = await mount(gw)
    try {
      lifecycle.activateLiveSession('next')
      await vi.waitFor(() => expect(sys).toHaveBeenCalledWith('error: invalid response: session.activate'))
      expect(discardSessionRecovery).toHaveBeenCalledWith()
      expect(getUiState()).toMatchObject({ sid: 'old', switching: false })
    } finally { act(() => setup.renderer.destroy()); turnController.fullReset(); resetUiState() }
  })

  it('releases prompts held for a switch that never commits', async () => {
    resetUiState(); turnController.fullReset()
    patchUiState({ sid: 'old' })
    const rpc = vi.fn(async (method: string) => method === 'session.create' ? null : { provider_configured: true }) as GatewayRpc
    const { composerActions, lifecycle, setup } = await mount({ request: vi.fn() } as unknown as GatewayClient, rpc)
    try {
      await lifecycle.newSession()
      expect(composerActions.activateSessionQueue).not.toHaveBeenCalled()
      expect(composerActions.releaseSwitchHold).toHaveBeenCalledTimes(1)
      expect(getUiState()).toMatchObject({ sid: 'old', switching: false })
    } finally { act(() => setup.renderer.destroy()); turnController.fullReset(); resetUiState() }
  })
})

describe('trimLastExchange', () => {
  it('removes the undone exchange even behind the /undo echo and a tool trail', () => {
    const items: Msg[] = [
      { role: 'user', text: 'earlier' },
      { role: 'assistant', text: 'earlier reply' },
      { role: 'user', text: 'run the tests' },
      { kind: 'trail', role: 'system', text: '', tools: ['bash'] },
      { role: 'assistant', text: 'all green' },
      { kind: 'outcome', role: 'assistant', text: 'done' },
      { kind: 'slash', role: 'system', text: '/undo' }
    ]

    expect(trimLastExchange(items)).toEqual([
      { role: 'user', text: 'earlier' },
      { role: 'assistant', text: 'earlier reply' },
      { kind: 'slash', role: 'system', text: '/undo' }
    ])
  })

  it('leaves a transcript with no user prompt untouched', () => {
    const items: Msg[] = [{ kind: 'intro', role: 'system', text: '' }, { kind: 'slash', role: 'system', text: '/undo' }]

    expect(trimLastExchange(items)).toEqual(items)
  })
})
