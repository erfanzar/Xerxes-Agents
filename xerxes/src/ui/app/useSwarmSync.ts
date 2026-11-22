// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type { GatewayEvent, SwarmStatusSnapshot } from '../gatewayTypes.js'

import { applySwarmStatus } from './swarmStore.js'
import { $uiSessionId, getUiState } from './uiStore.js'

/**
 * Reconcile durable swarm status on session attach/switch, then rely on
 * live `'swarm_event'` broadcasts with zero background polling.
 */
export function useSwarmSync(gw: GatewayClient) {
  const sid = useStore($uiSessionId)

  useEffect(() => {
    if (!sid) return
    let active = true
    let version = 0

    const refresh = async () => {
      const requestVersion = ++version
      try {
        const status = await gw.request<SwarmStatusSnapshot>('swarm.status', { session_id: sid })
        if (active && getUiState().sid === sid && requestVersion === version && status) {
          applySwarmStatus(status, sid)
        }
      } catch {
        // Best-effort discovery: older test harnesses or offline sockets omit swarm.status.
      }
    }

    const onEvent = (event: GatewayEvent) => {
      if (event.type === 'gateway.ready') {
        void refresh()
        return
      }
      if (event.type === 'swarm_event' && (event.payload?.session_id === sid || event.session_id === sid)) {
        // A live broadcast supersedes any in-flight initial status fetch.
        version += 1
      }
    }

    gw.on('event', onEvent)
    void refresh()

    return () => {
      active = false
      gw.off('event', onEvent)
    }
  }, [gw, sid])
}
