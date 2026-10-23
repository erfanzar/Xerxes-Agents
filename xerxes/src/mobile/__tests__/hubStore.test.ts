// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { inboxCounts, inboxTasks, orderTasks, reduceHub, connectionBanner, createHubStore, type HubState } from '../../client/hubStore.js'
import { parsePresence, presenceChips } from '../../client/hubParsers.js'
import { mergeUsageReports, usageFromResult } from '../../client/usageMerge.js'
import { TAIL_BLOCK_LIMIT, EMPTY_TAIL, foldAll, foldSessionEvent } from '../../client/sessionFold.js'
import { hubEvent, liveState, minutesAgo, summary, workspace } from './helpers.js'

function withSelected(events: ReturnType<typeof hubEvent>[] = []): HubState {
  let state = liveState([summary({ id: 's1', title: 'Fix bug' })])
  state = reduceHub(state, { type: 'select', workspaceId: 'w1', sessionId: 's1' })
  state = reduceHub(state, { type: 'events', events })
  return reduceHub(state, { type: 'sessionReady', workspaceId: 'w1', sessionId: 's1' })
}

describe('Inbox ordering and counts', () => {
  it('puts needs_you first, then working, then failed, then idle by recency', () => {
    const tasks = [
      summary({ id: 'idle-old', status: 'idle', updatedAt: minutesAgo(300) }),
      summary({ id: 'working', status: 'working', updatedAt: minutesAgo(50) }),
      summary({ id: 'needs', status: 'needs_you', pendingCount: 1, updatedAt: minutesAgo(200) }),
      summary({ id: 'idle-new', status: 'idle', updatedAt: minutesAgo(2) }),
      summary({ id: 'failed', status: 'failed', updatedAt: minutesAgo(30) }),
    ]
    const state = liveState(tasks)
    expect(inboxTasks(state).map(t => t.id)).toEqual(['needs', 'working', 'failed', 'idle-new', 'idle-old'])
  })

  it('sinks tasks of unavailable workspaces and keeps them out of the needs-you count', () => {
    const state = liveState(
      [summary({ id: 'a', workspaceId: 'w1', status: 'idle' }), summary({ id: 'b', workspaceId: 'w2', status: 'needs_you', pendingCount: 1 })],
      [workspace('w1', 'Mac'), workspace('w2', 'Server', 'offline')],
    )
    expect(orderTasks(Object.values(state.tasks), state.workspaces).map(t => t.id)).toEqual(['a', 'b'])
    const counts = inboxCounts(state)
    expect(counts.needsYou).toBe(0)
    expect(counts.unavailable).toBe(1)
  })

  it('filters by workspace and status', () => {
    const state = liveState(
      [summary({ id: 'a', workspaceId: 'w1', status: 'working' }), summary({ id: 'b', workspaceId: 'w2', status: 'working' }), summary({ id: 'c', workspaceId: 'w1', status: 'idle' })],
      [workspace('w1', 'Mac'), workspace('w2', 'Server')],
    )
    expect(inboxTasks(state, { workspaceId: 'w1', status: 'working' }).map(t => t.id)).toEqual(['a'])
  })

  it('counts a pending interaction once however often its request is replayed', () => {
    let state = liveState([summary({ id: 's1', status: 'working' })])
    const request = hubEvent('approval_request', { id: 'r1', tool_name: 'exec_command' })
    state = reduceHub(state, { type: 'events', events: [request], tail: false })
    state = reduceHub(state, { type: 'events', events: [{ ...request, event_id: 'again', sequence: 99 }], tail: false })
    const task = Object.values(state.tasks)[0]
    expect(task?.status).toBe('needs_you')
    expect(task?.pendingCount).toBe(1)
    expect(inboxCounts(state).needsYou).toBe(1)
    state = reduceHub(state, { type: 'events', events: [hubEvent('approval_response', { request_id: 'r1', response: 'approve' })], tail: false })
    expect(Object.values(state.tasks)[0]?.pendingCount).toBe(0)
  })

  it('does not fold summary-only events into the open task transcript', () => {
    const state = withSelected()
    const next = reduceHub(state, { type: 'events', events: [hubEvent('text_part', { text: 'hello' })], tail: false })
    expect(next.selected?.tail.blocks).toHaveLength(0)
  })
})

describe('presence', () => {
  it('collapses several devices of one person into one chip and drops yourself', () => {
    const chips = presenceChips([
      { userId: 'bob', name: 'Bob', deviceId: 'd1', state: 'watching' },
      { userId: 'bob', name: 'Bob', deviceId: 'd2', state: 'typing' },
      { userId: 'me', name: 'Ann', deviceId: 'd3', state: 'watching' },
    ], 'me')
    expect(chips).toEqual([{ userId: 'bob', name: 'Bob', state: 'typing', devices: 2 }])
  })

  it('reads the Hub participants payload', () => {
    const parsed = parsePresence({ workspace_id: 'w1', session_id: 's1', participants: [{ principal_id: 'u2', display_name: 'Bob', state: 'typing' }, { display_name: 'no id' }] })
    expect(parsed?.entries).toEqual([{ userId: 'u2', name: 'Bob', deviceId: '', state: 'typing' }])
  })
})

describe('selected session', () => {
  it('keeps a bounded tail and says older blocks exist', () => {
    const inputs = Array.from({ length: 120 }, (_, i) => ({ id: `e${i}`, type: 'tool_call', payload: { id: `t${i}`, name: 'exec_command', arguments: '{"command":"ls"}' } }))
    const tail = foldAll(EMPTY_TAIL, inputs)
    expect(tail.blocks).toHaveLength(TAIL_BLOCK_LIMIT)
    expect(tail.truncated).toBe(true)
    expect(tail.blocks.at(-1)).toMatchObject({ callId: 't119' })
  })

  it('streams text into one block and settles it at turn end', () => {
    let tail = foldSessionEvent(EMPTY_TAIL, { id: 'a', type: 'text_part', payload: { text: 'Hel' } })
    tail = foldSessionEvent(tail, { id: 'b', type: 'text_part', payload: { text: 'lo' } })
    expect(tail.blocks).toHaveLength(1)
    expect(tail.blocks[0]).toMatchObject({ kind: 'assistant', text: 'Hello', streaming: true })
    tail = foldSessionEvent(tail, { id: 'c', type: 'turn_end', payload: {} })
    expect(tail.blocks[0]).toMatchObject({ streaming: false })
    expect(tail.status).toBe('idle')
  })

  it('does not mutate the previous tail', () => {
    const before = foldSessionEvent(EMPTY_TAIL, { id: 'a', type: 'text_part', payload: { text: 'x' } })
    const snapshot = JSON.stringify(before)
    foldSessionEvent(before, { id: 'b', type: 'text_part', payload: { text: 'y' } })
    expect(JSON.stringify(before)).toBe(snapshot)
  })

  it('holds a requested model apart from the effective one until the task reports it', () => {
    let state = withSelected([hubEvent('status_update', { model: 'model-a' })])
    expect(state.selected?.model).toMatchObject({ effective: 'model-a', phase: 'idle' })

    state = reduceHub(state, { type: 'config', field: 'model', value: { requested: 'model-b', phase: 'requested' } })
    expect(state.selected?.model).toMatchObject({ effective: 'model-a', requested: 'model-b', phase: 'requested' })

    state = reduceHub(state, { type: 'config', field: 'model', value: { phase: 'accepted' } })
    // Accepted by the Hub is still not applied.
    expect(state.selected?.model).toMatchObject({ effective: 'model-a', requested: 'model-b', phase: 'accepted' })

    state = reduceHub(state, { type: 'events', events: [hubEvent('status_update', { model: 'model-b' })] })
    expect(state.selected?.model.effective).toBe('model-b')
    expect(state.selected?.model.phase).toBe('idle')
    expect(state.selected?.model.requested).toBeUndefined()
  })

  it('keeps a rejected request visible with its message', () => {
    let state = withSelected([hubEvent('status_update', { model: 'model-a' })])
    state = reduceHub(state, { type: 'config', field: 'model', value: { requested: 'model-b', phase: 'requested' } })
    state = reduceHub(state, { type: 'config', field: 'model', value: { phase: 'rejected', message: 'not permitted' } })
    expect(state.selected?.model).toMatchObject({ effective: 'model-a', phase: 'rejected', message: 'not permitted' })
  })

  it('tracks a pending approval with the real command and clears it when resolved by someone else', () => {
    let state = withSelected([hubEvent('approval_request', { id: 'r1', tool_name: 'exec_command', arguments: { command: 'rm -rf build', cwd: '/proj' }, cwd: '/proj' })])
    const pending = state.selected?.tail.pending[0]
    expect(pending).toMatchObject({ id: 'r1', toolName: 'exec_command', command: 'rm -rf build', workspace: '/proj' })
    state = reduceHub(state, { type: 'decision', interactionId: 'r1', state: { phase: 'sending', decision: 'once' } })
    state = reduceHub(state, {
      type: 'events',
      events: [hubEvent('approval_response', { request_id: 'r1', response: 'reject' }, { actor: { principal_kind: 'user', principal_id: 'u2', display_name: 'Bob' } })],
    })
    expect(state.selected?.tail.pending).toHaveLength(0)
    expect(state.selected?.tail.resolved.r1).toMatchObject({ by: 'Bob', decision: 'reject' })
  })

  it('does not replay a resolved interaction into the pending list', () => {
    const request = hubEvent('approval_request', { id: 'r1', tool_name: 'exec_command' })
    let state = withSelected([request, hubEvent('approval_response', { request_id: 'r1', response: 'approve' })])
    state = reduceHub(state, { type: 'events', events: [{ ...request, event_id: 'dup', sequence: 500 }] })
    expect(state.selected?.tail.pending).toHaveLength(0)
  })

  it('resets the folded tail on a stream reset', () => {
    let state = withSelected([hubEvent('text_part', { text: 'hello' })])
    expect(state.selected?.tail.blocks).toHaveLength(1)
    state = reduceHub(state, { type: 'sessionReset', workspaceId: 'w1', sessionId: 's1' })
    expect(state.selected).toMatchObject({ phase: 'loading', hasOlder: false })
    expect(state.selected?.tail.blocks).toHaveLength(0)
  })

  it('bounds the older pages it keeps', () => {
    let state = withSelected()
    const page = Array.from({ length: 300 }, (_, i) => ({ kind: 'notice' as const, id: `n${i}`, text: 't', error: false }))
    for (let i = 0; i < 3; i += 1) state = reduceHub(state, { type: 'olderLoaded', blocks: page, hasMore: true, cursor: i })
    expect(state.selected?.older.length).toBeLessThanOrEqual(400)
  })
})

describe('store', () => {
  it('notifies subscribers only when state changes and gives a stable snapshot', () => {
    const store = createHubStore(liveState())
    let calls = 0
    const stop = store.subscribe(() => { calls += 1 })
    const before = store.getSnapshot()
    store.dispatch({ type: 'inboxLoaded' })
    expect(store.getSnapshot()).toBe(before)
    expect(calls).toBe(0)
    store.dispatch({ type: 'connection', status: { state: 'reconnecting', attempt: 1 } })
    expect(calls).toBe(1)
    stop()
    store.dispatch({ type: 'connection', status: { state: 'live' } })
    expect(calls).toBe(1)
  })

  it('words the connection banner honestly', () => {
    expect(connectionBanner({ state: 'live' })).toBeUndefined()
    expect(connectionBanner({ state: 'reconnecting', attempt: 2 })?.text).toBe('Reconnecting — status may be outdated')
    expect(connectionBanner({ state: 'revoked', reason: 'Signed out.' })?.tone).toBe('problem')
  })
})

describe('usage merge', () => {
  const profile = (fetchedAt: number) => ({
    profile: 'claude-max', label: 'Claude Max', provider: 'anthropic', model: 'x', active: true, source: 'claude', status: 'ok' as const, plan: 'Max',
    windows: [{ label: '5-hour', usedPercent: 40 }], fetchedAt,
  })

  it('counts a plan shared by several workspaces once, newest wins', () => {
    const merged = mergeUsageReports([
      { fetchedAt: 1, profiles: [profile(100)] },
      { fetchedAt: 2, profiles: [profile(200)] },
    ])
    expect(merged.profiles).toHaveLength(1)
    expect(merged.profiles[0]?.fetchedAt).toBe(200)
  })

  it('counts a runtime once when its report arrives twice', () => {
    const report = { models: [{ model: 'm', calls: 2, input_tokens: 10, output_tokens: 5 }] }
    const merged = usageFromResult({ reports: [{ runtime_id: 'r1', report }, { runtime_id: 'r1', report }] })
    expect(merged.models?.[0]?.calls).toBe(2)
  })

  it('reads the Hub by_model rows and keeps plan-paid cost labelled', () => {
    const report = usageFromResult({ by_model: [{ model: 'm', calls: 1, input_tokens: 5, output_tokens: 1, cost_usd: 2, cost_basis: 'subscription' }] })
    expect(report.models?.[0]).toMatchObject({ costBasis: 'subscription', costUsd: 2 })
  })
})
