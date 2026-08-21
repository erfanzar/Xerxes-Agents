// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { testRender } from '@opentui/react/test-utils'
import { act, createElement } from 'react'
import { describe, expect, it, vi } from 'vitest'

import type { ComposerActions, GatewayRpc } from '../app/interfaces.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import type { GatewayClient } from '../gatewayClient.js'
import type {
  SessionActivateResponse,
  SessionCreateResponse,
  SessionResumeResponse,
  SetupStatusResponse
} from '../gatewayTypes.js'

const deferred = <T,>() => Promise.withResolvers<T>()

describe('useSessionLifecycle', () => {
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
        composerActions: { activateSessionQueue, setPasteSnips: vi.fn() } as unknown as ComposerActions,
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
        composerActions: { activateSessionQueue, setPasteSnips: vi.fn() } as unknown as ComposerActions,
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
        composerActions: { activateSessionQueue, setPasteSnips: vi.fn() } as unknown as ComposerActions,
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
})
