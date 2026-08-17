// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { testRender } from '@opentui/react/test-utils'
import { act, createElement } from 'react'
import { describe, expect, it, vi } from 'vitest'

import type { ComposerActions, GatewayRpc } from '../app/interfaces.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { SessionCreateResponse, SetupStatusResponse } from '../gatewayTypes.js'

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
    let lifecycle: ReturnType<typeof useSessionLifecycle> | undefined

    const Probe = () => {
      lifecycle = useSessionLifecycle({
        colsRef: { current: 80 },
        composerActions: { setPasteSnips: vi.fn() } as unknown as ComposerActions,
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
    } finally {
      act(() => setup.renderer.destroy())
      resetUiState()
    }
  })
})
