// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { atom, computed } from 'nanostores'

import type {
  SwarmBoardMessage,
  SwarmBoardResponse,
  SwarmEventPayload,
  SwarmMemberActivitySnapshot,
  SwarmStatusSnapshot
} from '../gatewayTypes.js'
import {
  normalizeSwarmBoardResponse,
  normalizeSwarmEventPayload,
  normalizeSwarmStatus
} from '../lib/swarmFormat.js'

import type { GatewayRpc } from './interfaces.js'
import { $uiSessionId, getUiState } from './uiStore.js'

const MAX_CACHED_BOARD_MESSAGES = 500
const UNSCOPED_SESSION_KEY = '__unscoped__'

export interface SwarmSessionSlice {
  readonly status: SwarmStatusSnapshot | null
  readonly messages: readonly SwarmBoardMessage[]
  readonly activities: Readonly<Record<string, SwarmMemberActivitySnapshot>>
  readonly head: number
  readonly updatedAt: number
}

export const EMPTY_SWARM_SLICE: SwarmSessionSlice = {
  status: null,
  messages: [],
  activities: {},
  head: 0,
  updatedAt: 0
}

export const $swarmStore = atom<Record<string, SwarmSessionSlice>>({})

const resolveTargetKey = (sessionId?: string | null): string => {
  const explicit = typeof sessionId === 'string' ? sessionId.trim() : ''
  if (explicit) return explicit
  const current = getUiState().sid?.trim() ?? ''
  return current || UNSCOPED_SESSION_KEY
}

const mergeMessagesBySeq = (
  existing: readonly SwarmBoardMessage[],
  incoming: readonly SwarmBoardMessage[]
): SwarmBoardMessage[] => {
  if (!incoming.length) return [...existing]
  const bySeq = new Map<number, SwarmBoardMessage>()
  for (const msg of existing) bySeq.set(msg.seq, msg)
  for (const msg of incoming) bySeq.set(msg.seq, msg)
  const merged = [...bySeq.values()].sort((a, b) => a.seq - b.seq)
  return merged.length > MAX_CACHED_BOARD_MESSAGES
    ? merged.slice(merged.length - MAX_CACHED_BOARD_MESSAGES)
    : merged
}

export function getSwarmSliceForSession(
  store: Readonly<Record<string, SwarmSessionSlice>>,
  sessionId?: string | null
): SwarmSessionSlice {
  const sid = typeof sessionId === 'string' ? sessionId.trim() : ''
  if (sid) {
    if (store[sid]) return store[sid]!
    const unscoped = store[UNSCOPED_SESSION_KEY]
    if (unscoped && (!unscoped.status?.session_id || unscoped.status.session_id === sid)) {
      return unscoped
    }
    return EMPTY_SWARM_SLICE
  }
  return store[UNSCOPED_SESSION_KEY] ?? Object.values(store).at(-1) ?? EMPTY_SWARM_SLICE
}

export const $activeSwarmSlice = computed([$swarmStore, $uiSessionId], (store, sid) =>
  getSwarmSliceForSession(store, sid)
)

export const $activeSwarmStatus = computed($activeSwarmSlice, slice => slice.status)

export function getActiveSwarmSlice(): SwarmSessionSlice {
  return $activeSwarmSlice.get()
}

export function applySwarmStatus(rawStatus: unknown, fallbackSessionId?: string | null): SwarmStatusSnapshot {
  const fallback = typeof fallbackSessionId === 'string' ? fallbackSessionId.trim() : getUiState().sid?.trim() ?? ''
  const status = normalizeSwarmStatus(rawStatus, fallback)
  const key = resolveTargetKey(status.session_id || fallback)
  const current = $swarmStore.get()
  const prev = current[key] ?? EMPTY_SWARM_SLICE
  const messages = status.head === 0 && !status.active ? [] : prev.messages
  const head = Math.max(status.head, messages.at(-1)?.seq ?? 0)
  const nextStatus = head !== status.head ? { ...status, head } : status

  $swarmStore.set({
    ...current,
    [key]: {
      status: nextStatus,
      messages,
      activities: status.head === 0 && !status.active ? {} : prev.activities,
      head,
      updatedAt: Date.now()
    }
  })

  return nextStatus
}

export function applySwarmBoard(
  sessionId: string | null | undefined,
  rawBoard: unknown,
  options: { resetIfEmptyHead?: boolean } = {}
): SwarmBoardResponse {
  const board = normalizeSwarmBoardResponse(rawBoard)
  const key = resolveTargetKey(sessionId)
  const current = $swarmStore.get()
  const prev = current[key] ?? EMPTY_SWARM_SLICE
  const messages =
    options.resetIfEmptyHead && board.head === 0 && board.messages.length === 0
      ? []
      : mergeMessagesBySeq(prev.messages, board.messages)
  const head = Math.max(prev.head, board.head, messages.at(-1)?.seq ?? 0)
  const status = prev.status ? { ...prev.status, head: Math.max(prev.status.head, head) } : null

  $swarmStore.set({
    ...current,
    [key]: {
      status,
      messages,
      activities: prev.activities,
      head,
      updatedAt: Date.now()
    }
  })

  return { head, messages }
}

export function applySwarmEvent(rawEvent: unknown, fallbackSessionId?: string | null): SwarmEventPayload {
  const event = normalizeSwarmEventPayload(rawEvent)
  const key = resolveTargetKey(event.session_id || event.status?.session_id || fallbackSessionId)
  const current = $swarmStore.get()
  const prev = current[key] ?? EMPTY_SWARM_SLICE

  const messages = event.message ? mergeMessagesBySeq(prev.messages, [event.message]) : prev.messages
  const head = Math.max(
    prev.head,
    event.head,
    event.status?.head ?? 0,
    messages.at(-1)?.seq ?? 0
  )

  const targetMemberName = event.member_name ?? event.member?.name
  const nextActivities =
    targetMemberName && event.activity
      ? { ...prev.activities, [targetMemberName]: event.activity }
      : prev.activities

  let nextStatus: SwarmStatusSnapshot | null = event.status ?? prev.status
  if (nextStatus) {
    let nextMembers = nextStatus.members
    if (!event.status && targetMemberName && (event.member || event.activity)) {
      nextMembers = nextMembers.map(existing => {
        if (existing.name !== targetMemberName) return existing
        const merged = event.member ? { ...existing, ...event.member } : existing
        if (!event.activity) return merged
        const activityLabel =
          merged.state === 'working'
            ? event.activity.summary ??
              (event.activity.phase === 'thinking'
                ? 'Thinking…'
                : event.activity.phase === 'responding'
                  ? 'Responding…'
                  : undefined)
            : undefined
        const { current_activity: _prevAct, ...withoutAct } = merged
        return activityLabel ? { ...withoutAct, current_activity: activityLabel } : withoutAct
      })
    }
    nextStatus = {
      ...nextStatus,
      members: nextMembers,
      head: Math.max(nextStatus.head, head),
      active: event.status ? event.status.active : nextStatus.active || Boolean(event.message)
    }
  } else if (event.message || event.kind === 'member_added' || event.kind === 'titled') {
    nextStatus = normalizeSwarmStatus(
      {
        active: true,
        session_id: event.session_id || (typeof fallbackSessionId === 'string' ? fallbackSessionId : ''),
        head
      },
      typeof fallbackSessionId === 'string' ? fallbackSessionId : ''
    )
  }

  $swarmStore.set({
    ...current,
    [key]: {
      status: nextStatus,
      messages,
      activities: nextActivities,
      head,
      updatedAt: Date.now()
    }
  })

  return event
}

export async function refreshSwarmSession(
  rpc: GatewayRpc,
  sessionId?: string | null,
  options: { reportError?: boolean; channel?: string; limit?: number } = {}
): Promise<{ status: SwarmStatusSnapshot | null; board: SwarmBoardResponse | null }> {
  const sid = typeof sessionId === 'string' && sessionId.trim() ? sessionId.trim() : getUiState().sid?.trim()
  const sessionParam = sid ? { session_id: sid } : {}
  const boardParams: Record<string, unknown> = {
    ...sessionParam,
    limit: options.limit ?? 100,
    ...(options.channel && options.channel !== '*' ? { channel: options.channel } : {})
  }

  const rpcOpts = options.reportError === undefined ? undefined : { reportError: options.reportError }

  const [rawStatus, rawBoard] = await Promise.all([
    rpc<Record<string, unknown>>('swarm.status', sessionParam, rpcOpts),
    rpc<Record<string, unknown>>('swarm.board', boardParams, rpcOpts)
  ])

  const status =
    rawStatus && rawStatus.ok !== false ? applySwarmStatus(rawStatus, sid) : null
  const board =
    rawBoard && rawBoard.ok !== false
      ? applySwarmBoard(status?.session_id ?? sid, rawBoard, { resetIfEmptyHead: !options.channel || options.channel === '*' })
      : null

  return { status, board }
}

export function resetSwarmStore(): void {
  $swarmStore.set({})
}
