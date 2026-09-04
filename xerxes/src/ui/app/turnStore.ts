// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { atom, computed } from 'nanostores'
import { useSyncExternalStore } from 'react'

import type { ActiveTool, ActivityItem, Msg, SubagentProgress, TodoItem } from '../types.js'

import { $uiState } from './uiStore.js'

const buildTurnState = (): TurnState => ({
  activity: [],
  compacting: false,
  outcome: '',
  reasoning: '',
  reasoningActive: false,
  reasoningStreaming: false,
  reasoningTokens: 0,
  streamPendingTools: [],
  streamSegments: [],
  streaming: '',
  subagents: [],
  todoCollapsed: false,
  todos: [],
  toolRecords: {},
  toolTokens: 0,
  tools: [],
  turnTrail: []
})

export const $turnState = atom<TurnState>(buildTurnState())

export const getTurnState = () => $turnState.get()

const subscribeTurn = (cb: () => void) => $turnState.listen(() => cb())

export const useTurnSelector = <T>(selector: (state: TurnState) => T): T =>
  useSyncExternalStore(
    subscribeTurn,
    () => selector($turnState.get()),
    () => selector($turnState.get())
  )

export const patchTurnState = (next: Partial<TurnState> | ((state: TurnState) => TurnState)) => {
  const previous = $turnState.get()
  const updated = typeof next === 'function' ? next(previous) : { ...previous, ...next }

  stampTurnDelta(previous, updated)
  $turnState.set(updated)
}

export const toggleTodoCollapsed = () => patchTurnState(state => ({ ...state, todoCollapsed: !state.todoCollapsed }))

export const resetTurnState = () => {
  endTurnPulse()
  $turnState.set(buildTurnState())
}

export interface TurnPulse {
  /**
   * Monotonic identifier of the current liveness window. Wall-clock
   * `startedAt` equality cannot distinguish a turn that ended and restarted
   * within one millisecond from the original turn still running; the epoch
   * can.
   */
  epoch: number
  /** Wall clock of the most recent streaming delta; 0 before the first one. */
  lastDeltaAt: number
  /** Wall clock the current busy stretch began; 0 while idle. */
  startedAt: number
}

const $turnPulse = atom<TurnPulse>({ epoch: 0, lastDeltaAt: 0, startedAt: 0 })

export const getTurnPulse = () => $turnPulse.get()

/**
 * Coarse gate for the live indicator.  A boolean recomputed from two stores
 * settles to the same value on nearly every tick, and nanostores drops
 * identical writes, so the indicator leaf mounts once per turn instead of
 * re-rendering per delta — the fine-grained animation is driven by mutating
 * its renderables from an interval, not by React.
 */
export const $turnLive = computed([$uiState, $turnState], (ui, turn) => ui.busy || turn.tools.length > 0)

/** Open a liveness window; a turn already in flight keeps its original start. */
export const beginTurnPulse = (now = Date.now()): TurnPulse => {
  const current = $turnPulse.get()

  if (current.startedAt > 0) {
    return current
  }

  const started = { epoch: current.epoch + 1, lastDeltaAt: 0, startedAt: now }
  $turnPulse.set(started)

  return started
}

export const endTurnPulse = () => {
  const current = $turnPulse.get()
  $turnPulse.set({ ...current, lastDeltaAt: 0, startedAt: 0 })
}

// The window is opened and closed by the gate itself, not by whoever happens to
// render the indicator: the live-turn view unmounts and remounts as the
// transcript changes shape, and a start stamp owned by that component would
// survive into the next turn and report an hours-old elapsed clock.
$turnLive.listen(live => (live ? beginTurnPulse() : endTurnPulse()))

/**
 * Stamped centrally rather than at each producer: the turn controller pushes
 * deltas through a dozen separate patch paths, and one missed call site would
 * make the indicator report a perfectly healthy stream as stalled.  Growth is
 * the signal — the end-of-turn patches that blank `streaming` back to '' are
 * not deltas and must not refresh the stamp.
 */
const stampTurnDelta = (previous: TurnState, updated: TurnState) => {
  if (updated.streaming.length > previous.streaming.length || updated.reasoning.length > previous.reasoning.length) {
    $turnPulse.set({ ...$turnPulse.get(), lastDeltaAt: Date.now() })
  }
}

export interface TurnState {
  activity: ActivityItem[]
  /** True while the daemon is compacting this session's transcript. */
  compacting: boolean
  outcome: string
  reasoning: string
  reasoningActive: boolean
  reasoningStreaming: boolean
  reasoningTokens: number
  /** Completed tool rows kept visible until the live turn settles. */
  streamPendingTools: string[]
  streamSegments: Msg[]
  streaming: string
  subagents: SubagentProgress[]
  todoCollapsed: boolean
  todos: TodoItem[]
  toolTokens: number
  tools: ActiveTool[]
  /** Full call/response records keyed by tool id, for the expanded detail view. */
  toolRecords: Record<string, ToolCallRecord>
  turnTrail: string[]
}

export interface ToolCallRecord {
  args?: string
  durationS?: number
  error?: string
  name: string
  reasoning?: string
  result?: string
}
