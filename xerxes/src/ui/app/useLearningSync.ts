// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import type { GatewayEvent, LearningCounts, LearningListResponse } from '../gatewayTypes.js'
import { $uiSessionId, getUiState, patchUiState } from './uiStore.js'

/** Primitive fields + change-only patches keep idle status chrome from flickering. */
export function applyLearningCounts(workspace: string, counts: LearningCounts) {
  patchUiState({
    learningWorkspace: workspace,
    pendingProposals: counts.pending_proposals,
    pendingLessons: counts.pending_lessons
  })
}

/** Reconcile on session transitions/reconnect, then consume broadcasts. No polling. */
export function useLearningSync(gw: GatewayClient) {
  const sid = useStore($uiSessionId)
  useEffect(() => {
    let active = true
    let version = 0
    const refresh = async () => {
      const requestVersion = ++version
      try {
        const result = await gw.request<LearningListResponse>('learning.list', sid ? { session_key: sid } : {})
        if (active && getUiState().sid === sid && requestVersion === version && result.ok) applyLearningCounts(result.workspace, result.counts)
      } catch {
        // Count discovery is best-effort while reconnecting. The review overlay
        // displays actionable RPC errors and offers an explicit retry.
      }
    }
    const onEvent = (event: GatewayEvent) => {
      if (event.type === 'gateway.ready') void refresh()
      if (event.type === 'learning.changed') {
        const state = getUiState()
        if (event.payload.workspace !== (state.learningWorkspace ?? state.info?.cwd)) return
        ++version // A list response started before this broadcast must not regress it.
        applyLearningCounts(event.payload.workspace, event.payload.counts)
      }
    }
    gw.on('event', onEvent)
    void refresh()
    return () => { active = false; gw.off('event', onEvent) }
  }, [gw, sid])
}
