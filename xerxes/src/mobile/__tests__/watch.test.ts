// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it, vi } from 'vitest'

import type { HubConnection } from '../../client/hubConnection.js'
import { createHubController } from '../../client/hubController.js'
import { parseAutopilotView, parseWatchRow, parseWatchView } from '../../client/hubParsers.js'
import { activeWatchRows, autopilotThreadAvailable, createHubStore, initialHubState, isHubOwner, reduceHub, watchKey, watchRowFor } from '../../client/hubStore.js'
import {
  WATCH_EXPLANATION, activityAgo, anomalyLabel, reviewMinutesOf, reviewText, reviewValueOf, watchCardRows, watchChipText, watchContextPercent, watchFacts,
  watchReviewOptions, watchStateText,
} from '../../client/watch.js'
import { goLive, harness, settle } from './fakes.js'

export const STATS = {
  since: '2026-10-24T09:00:00.000Z',
  last_activity_at: '2026-10-24T10:00:00.000Z',
  status: 'working',
  turns: 3,
  tool_calls: 40,
  tool_failures: 2,
  errors: 1,
  edits: 4,
  files_touched: 3,
  tests: { runs: 3, passed: 10, failed: 2, last: 'fail', last_failing: ['test_autoscale'] },
  repeated_call: { name: 'exec_command', count: 4 },
  context_tokens: 90_000,
  max_context: 100_000,
  total_tokens: 200_000,
  agents: { running: 2, total: 3 },
  anomalies: [{ kind: 'loop', since: '2026-10-24T09:55:00.000Z', detail: 'exec_command ×5 in the last 20 calls' }],
  reviews: { wakes_this_hour: 1, last_wake_at: '2026-10-24T09:30:00.000Z' },
  summary_line: '4 edits · tests failing (test_autoscale) · 2 agents · active 40 s ago',
}

export const ROW = {
  workspace_id: 'w1',
  task_id: 't1',
  title: 'Fix tests',
  workspace_label: 'Mac · app',
  enabled: true,
  active: true,
  review_interval_minutes: 15,
  expires_at: '2026-10-25T00:00:00.000Z',
  enabled_by: { id: 'u1', name: 'Ann' },
  updated_at: '2026-10-24T10:00:00.000Z',
  stats: STATS,
}
const BARE_ROW = { ...ROW, task_id: 't2', title: 'Other', enabled: false, active: false, review_interval_minutes: null, expires_at: null, enabled_by: null, updated_at: null, stats: null }
const VIEW = { can_edit: true, watches: [ROW, BARE_ROW] }
const AUTOPILOT_OWNER = { can_edit: true, workspaces: [{ workspace_id: 'w1', label: 'Mac', enabled: false, instruction: '', expires_at: null, enabled_by: null, updated_at: null, active: false }] }

describe('watch parsers', () => {
  it('reads a full row and a row without stats', () => {
    const row = parseWatchRow(ROW)
    expect(row).toMatchObject({
      workspaceId: 'w1', taskId: 't1', title: 'Fix tests', workspaceLabel: 'Mac · app', enabled: true, active: true, reviewIntervalMinutes: 15,
      expiresAt: '2026-10-25T00:00:00.000Z', enabledBy: { id: 'u1', name: 'Ann' },
    })
    expect(row?.stats).toMatchObject({
      status: 'working', edits: 4, toolCalls: 40, toolFailures: 2, filesTouched: 3, contextTokens: 90_000, maxContext: 100_000,
      tests: { runs: 3, passed: 10, failed: 2, last: 'fail', lastFailing: ['test_autoscale'] }, repeatedCall: { name: 'exec_command', count: 4 },
      agents: { running: 2, total: 3 }, reviews: { wakesThisHour: 1, lastWakeAt: '2026-10-24T09:30:00.000Z' },
    })
    expect(row?.stats?.anomalies).toEqual([{ kind: 'loop', since: '2026-10-24T09:55:00.000Z', detail: 'exec_command ×5 in the last 20 calls' }])
    expect(parseWatchRow(BARE_ROW)).toMatchObject({ enabled: false, reviewIntervalMinutes: null, expiresAt: null, enabledBy: null, updatedAt: null, stats: null })
  })

  it('accepts null for the nullable stats and a null repeated call', () => {
    const row = parseWatchRow({ ...ROW, stats: { ...STATS, last_activity_at: null, repeated_call: null, context_tokens: null, max_context: null, total_tokens: null, tests: { runs: 0, passed: 0, failed: 0, last: null, last_failing: [] } } })
    expect(row?.stats).toMatchObject({ lastActivityAt: null, repeatedCall: null, contextTokens: null, maxContext: null, totalTokens: null })
  })

  it('rejects a row with a missing or wrongly typed field instead of guessing', () => {
    const without = (key: string): unknown => { const copy: Record<string, unknown> = { ...ROW }; delete copy[key]; return copy }
    for (const key of ['workspace_id', 'task_id', 'title', 'workspace_label', 'enabled', 'active', 'review_interval_minutes', 'expires_at', 'enabled_by', 'updated_at', 'stats']) {
      expect(parseWatchRow(without(key)), `missing ${key}`).toBeUndefined()
    }
    expect(parseWatchRow({ ...ROW, enabled: 'yes' })).toBeUndefined()
    expect(parseWatchRow({ ...ROW, active: 1 })).toBeUndefined()
    expect(parseWatchRow({ ...ROW, review_interval_minutes: '15' })).toBeUndefined()
    expect(parseWatchRow({ ...ROW, review_interval_minutes: 2.5 })).toBeUndefined()
    expect(parseWatchRow({ ...ROW, expires_at: 'tomorrow' })).toBeUndefined()
    expect(parseWatchRow({ ...ROW, updated_at: 12 })).toBeUndefined()
    expect(parseWatchRow({ ...ROW, enabled_by: { id: 'u1' } })).toBeUndefined()
    expect(parseWatchRow({ ...ROW, stats: 'ok' })).toBeUndefined()
    expect(parseWatchRow(null)).toBeUndefined()
    expect(parseWatchRow([ROW])).toBeUndefined()
    expect(parseWatchRow('junk')).toBeUndefined()
  })

  it('rejects the whole row when anything inside stats is malformed', () => {
    const bad = (patch: Record<string, unknown>): unknown => ({ ...ROW, stats: { ...STATS, ...patch } })
    expect(parseWatchRow(bad({ since: 'never' }))).toBeUndefined()
    expect(parseWatchRow(bad({ last_activity_at: 'soon' }))).toBeUndefined()
    expect(parseWatchRow(bad({ status: 3 }))).toBeUndefined()
    expect(parseWatchRow(bad({ summary_line: null }))).toBeUndefined()
    expect(parseWatchRow(bad({ edits: '4' }))).toBeUndefined()
    expect(parseWatchRow(bad({ tool_calls: -1 }))).toBeUndefined()
    expect(parseWatchRow(bad({ context_tokens: 'many' }))).toBeUndefined()
    expect(parseWatchRow(bad({ tests: { ...STATS.tests, last: 'maybe' } }))).toBeUndefined()
    expect(parseWatchRow(bad({ tests: { ...STATS.tests, last_failing: [1] } }))).toBeUndefined()
    expect(parseWatchRow(bad({ repeated_call: { name: 'x' } }))).toBeUndefined()
    expect(parseWatchRow(bad({ agents: { running: 1 } }))).toBeUndefined()
    expect(parseWatchRow(bad({ reviews: { wakes_this_hour: 1 } }))).toBeUndefined()
    expect(parseWatchRow(bad({ anomalies: 'loop' }))).toBeUndefined()
  })

  it('accepts only the fixed anomaly kinds and well-formed anomalies', () => {
    for (const kind of ['loop', 'stall', 'test_fail_repeat', 'error_burst', 'context_high', 'goal_blocked']) {
      const row = parseWatchRow({ ...ROW, stats: { ...STATS, anomalies: [{ kind, since: STATS.since, detail: 'd' }] } })
      expect(row?.stats?.anomalies[0]?.kind, kind).toBe(kind)
    }
    const withAnomaly = (anomaly: unknown): unknown => ({ ...ROW, stats: { ...STATS, anomalies: [anomaly] } })
    expect(parseWatchRow(withAnomaly({ kind: 'on_fire', since: STATS.since, detail: 'd' }))).toBeUndefined()
    expect(parseWatchRow(withAnomaly({ kind: 'loop', since: 'x', detail: 'd' }))).toBeUndefined()
    expect(parseWatchRow(withAnomaly({ kind: 'loop', since: STATS.since }))).toBeUndefined()
    expect(parseWatchRow(withAnomaly('loop'))).toBeUndefined()
  })

  it('keeps untrusted text exactly as text', () => {
    const row = parseWatchRow({ ...ROW, title: '<img src=x onerror=alert(1)>', stats: { ...STATS, summary_line: '<b>boom</b>' } })
    expect(row?.title).toBe('<img src=x onerror=alert(1)>')
    expect(row?.stats?.summaryLine).toBe('<b>boom</b>')
  })

  it('reads the list answer and rejects it whole when anything is malformed or repeated', () => {
    const view = parseWatchView(VIEW)
    expect(view?.canEdit).toBe(true)
    expect(view?.watches.map(row => row.taskId)).toEqual(['t1', 't2'])
    expect(parseWatchView({ can_edit: false, watches: [] })).toEqual({ canEdit: false, watches: [] })
    expect(parseWatchView({ watches: [] })).toBeUndefined()
    expect(parseWatchView({ can_edit: 'yes', watches: [] })).toBeUndefined()
    expect(parseWatchView({ can_edit: true })).toBeUndefined()
    expect(parseWatchView({ can_edit: true, watches: [ROW, { ...ROW, task_id: 't9', active: 'x' }] })).toBeUndefined()
    expect(parseWatchView({ can_edit: true, watches: [ROW, ROW] })).toBeUndefined()
    // The same task id in another workspace is a different watch.
    expect(parseWatchView({ can_edit: true, watches: [ROW, { ...ROW, workspace_id: 'w2' }] })?.watches).toHaveLength(2)
    expect(parseWatchView(undefined)).toBeUndefined()
  })
})

describe('watch helpers', () => {
  const row = parseWatchRow(ROW)!

  it('words the state, chip and review text', () => {
    expect(watchStateText(row)).toBe('Watching')
    expect(watchStateText({ active: false, enabled: true })).toBe('Off (expired or the owner changed)')
    expect(watchStateText({ active: false, enabled: false })).toBe('Off')
    expect(watchChipText([row, { ...row, taskId: 't3' }])).toBe('Watching · 2 tasks')
    expect(watchChipText([row])).toBe('Watching · 1 task')
    expect(watchChipText([{ ...row, active: false }])).toBeUndefined()
    expect(reviewText(null)).toBe('No reviews')
    expect(reviewText(15)).toBe('Review every 15 min')
    expect(anomalyLabel('stall')).toBe('Stalled')
    expect(WATCH_EXPLANATION).toContain('no model cost')
  })

  it('offers Off, 5, 15, 30 and 60 minutes and keeps an off-preset saved interval', () => {
    expect(watchReviewOptions(null).map(o => o.value)).toEqual(['off', '5', '15', '30', '60'])
    expect(watchReviewOptions(45).map(o => o.value)).toEqual(['off', '5', '15', '30', '45', '60'])
    expect(reviewValueOf(null)).toBe('off')
    expect(reviewValueOf(30)).toBe('30')
    expect(reviewMinutesOf('off')).toBeNull()
    expect(reviewMinutesOf('30')).toBe(30)
  })

  it('shows a relative time that moves in seconds', () => {
    const now = Date.parse('2026-10-24T10:01:00.000Z')
    expect(activityAgo(null, now)).toBe('no activity yet')
    expect(activityAgo('2026-10-24T10:00:58.000Z', now)).toBe('just now')
    expect(activityAgo('2026-10-24T10:00:20.000Z', now)).toBe('40 s ago')
    expect(activityAgo('2026-10-24T10:01:00.000Z'.replace('10:01', '09:57'), now)).toBe('4 min ago')
    expect(activityAgo('2026-10-24T07:01:00.000Z', now)).toBe('3 h ago')
    expect(activityAgo('2026-10-21T10:01:00.000Z', now)).toBe('3 d ago')
    expect(activityAgo('nope', now)).toBe('unknown')
  })

  it('computes context percent only when the window is known', () => {
    expect(watchContextPercent(row.stats!)).toBe(90)
    expect(watchContextPercent({ contextTokens: 5, maxContext: null })).toBeNull()
    expect(watchContextPercent({ contextTokens: null, maxContext: 10 })).toBeNull()
    expect(watchContextPercent({ contextTokens: 500, maxContext: 100 })).toBe(100)
  })

  it('lists the counters, marking failures and repeats', () => {
    const facts = watchFacts(row.stats!)
    expect(facts.map(f => f.key)).toEqual(['edits', 'tools', 'failures', 'errors', 'tests', 'agents', 'context', 'repeated'])
    expect(facts.find(f => f.key === 'tests')).toMatchObject({ text: 'tests: 3 runs, 10 passed, 2 failed', tone: 'bad' })
    expect(facts.find(f => f.key === 'agents')?.text).toBe('2/3 agents running')
    expect(facts.find(f => f.key === 'context')).toMatchObject({ text: 'context 90%', tone: 'bad' })
    expect(facts.find(f => f.key === 'repeated')?.text).toBe('repeated exec_command ×4')
    const quiet = watchFacts({ ...row.stats!, toolFailures: 0, errors: 0, tests: { runs: 0, passed: 0, failed: 0, last: null, lastFailing: [] }, agents: { running: 0, total: 0 }, contextTokens: null, repeatedCall: null })
    expect(quiet.map(f => f.key)).toEqual(['edits', 'tools'])
  })

  it('orders cards: open problems first, then watched, and drops disabled rows', () => {
    const calm = { ...row, taskId: 'a', title: 'Calm', stats: { ...row.stats!, anomalies: [] } }
    const loud = { ...row, taskId: 'b', title: 'Zebra loud' }
    const lapsed = { ...row, taskId: 'c', title: 'Lapsed', active: false, stats: { ...row.stats!, anomalies: [] } }
    const off = { ...row, taskId: 'd', title: 'Off', enabled: false, active: false }
    expect(watchCardRows([lapsed, calm, off, loud]).map(r => r.taskId)).toEqual(['b', 'a', 'c'])
  })
})

describe('watch reducer', () => {
  it('starts empty, read-only and supported', () => {
    expect(initialHubState().watch).toEqual({ phase: 'idle', canEdit: false, rows: [], unsupported: false })
  })

  it('a view replaces every row and clears errors; a row replaces or appends one', () => {
    let state = reduceHub(initialHubState(), { type: 'watch', patch: { phase: 'error', error: 'boom' } })
    state = reduceHub(state, { type: 'watch', patch: { saveError: 'nope' } })
    state = reduceHub(state, { type: 'watchView', view: parseWatchView(VIEW)! })
    expect(state.watch).toMatchObject({ phase: 'ready', canEdit: true })
    expect(state.watch.error).toBeUndefined()
    expect(state.watch.saveError).toBeUndefined()
    expect(state.watch.rows).toHaveLength(2)

    state = reduceHub(state, { type: 'watchRow', row: parseWatchRow({ ...ROW, active: false, enabled: false })! })
    expect(state.watch.rows.map(r => [r.taskId, r.active])).toEqual([['t1', false], ['t2', false]])
    expect(activeWatchRows(state)).toEqual([])

    state = reduceHub(state, { type: 'watchRow', row: parseWatchRow({ ...ROW, task_id: 't9' })! })
    expect(state.watch.rows.map(r => r.taskId)).toEqual(['t1', 't2', 't9'])
    expect(activeWatchRows(state).map(r => r.taskId)).toEqual(['t9'])
    expect(watchRowFor(state, 'w1', 't9')?.title).toBe('Fix tests')
    expect(watchRowFor(state, 'w2', 't9')).toBeUndefined()
  })

  it('tracks which task is saving and ends the save explicitly', () => {
    let state = reduceHub(initialHubState(), { type: 'watch', patch: { saveError: 'old' } })
    state = reduceHub(state, { type: 'watch', patch: { savingKey: watchKey('w1', 't1') } })
    expect(state.watch.savingKey).toBe(watchKey('w1', 't1'))
    expect(state.watch.saveError).toBeUndefined()
    state = reduceHub(state, { type: 'watch', patch: { savingKey: undefined } })
    expect('savingKey' in state.watch).toBe(false)
  })
})

describe('the Autopilot thread view and watches', () => {
  function owner(rows: readonly unknown[], autopilotActive = false) {
    let state = reduceHub(initialHubState(), { type: 'self', userId: 'u1' })
    state = reduceHub(state, { type: 'autopilotView', view: parseAutopilotView({ can_edit: true, workspaces: [{ ...AUTOPILOT_OWNER.workspaces[0], enabled: autopilotActive, active: autopilotActive }] })! })
    return reduceHub(state, { type: 'watchView', view: parseWatchView({ can_edit: true, watches: rows })! })
  }

  it('is offered to the owner when only a watch is active', () => {
    expect(autopilotThreadAvailable(owner([ROW]))).toBe(true)
  })

  it('is not offered for a lapsed or stopped watch, or without a known user', () => {
    expect(autopilotThreadAvailable(owner([{ ...ROW, active: false }]))).toBe(false)
    expect(autopilotThreadAvailable(owner([]))).toBe(false)
    const noSelf = { ...owner([ROW]) }
    const { selfUserId: _self, ...withoutSelf } = noSelf
    expect(autopilotThreadAvailable(withoutSelf)).toBe(false)
  })

  it('is never offered to a person who cannot edit Autopilot, or after the Hub hid it', () => {
    const viewer = reduceHub(owner([ROW]), { type: 'autopilotView', view: parseAutopilotView({ can_edit: false, workspaces: [] })! })
    expect(autopilotThreadAvailable(viewer)).toBe(false)
    expect(autopilotThreadAvailable(reduceHub(owner([ROW]), { type: 'autopilotThread', patch: { phase: 'hidden' } }))).toBe(false)
  })

  it('still follows Autopilot when no watch is on', () => {
    expect(autopilotThreadAvailable(owner([], true))).toBe(true)
  })
})

function fakeConnection(): { connection: HubConnection; notice: (type: string, payload: Record<string, unknown>) => void; requests: string[] } {
  const listeners = new Set<(type: string, payload: Readonly<Record<string, unknown>>) => void>()
  const requests: string[] = []
  const connection = {
    status: () => ({ state: 'closed' as const }),
    helloInfo: () => ({}),
    onStatus: () => () => undefined,
    onNotice: (listener: (type: string, payload: Readonly<Record<string, unknown>>) => void) => { listeners.add(listener); return () => { listeners.delete(listener) } },
    connect: () => undefined,
    close: () => undefined,
    request: async (method: string) => { requests.push(method); return {} },
  } as unknown as HubConnection
  return { connection, notice: (type, payload) => { for (const l of listeners) l(type, payload) }, requests }
}

function liveHarness() {
  const h = harness()
  h.hub.handlers.set('hub.workspaces.list', () => ({ workspaces: [] }))
  h.hub.handlers.set('hub.machines.list', () => ({ machines: [] }))
  return h
}

describe('watch controller', () => {
  it('loads the rows from supervisor.watch.list', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.watch.list', () => VIEW)
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    await controller.loadWatches()
    expect(h.hub.all('supervisor.watch.list')[0]?.params).toEqual({})
    expect(store.getSnapshot().watch).toMatchObject({ phase: 'ready', canEdit: true, unsupported: false })
    expect(store.getSnapshot().watch.rows.map(row => row.taskId)).toEqual(['t1', 't2'])
  })

  it('opening the supervisor thread also loads Watch', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.thread.open', () => ({ thread_id: 't', messages: [], confirmations: [] }))
    h.hub.handlers.set('supervisor.config.get', () => ({ status: 'unconfigured', can_edit: true }))
    h.hub.handlers.set('supervisor.autopilot.get', () => ({ can_edit: false, workspaces: [] }))
    h.hub.handlers.set('supervisor.watch.list', () => ({ can_edit: false, watches: [ROW] }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    await controller.supervisorOpen()
    await settle()
    expect(h.hub.all('supervisor.watch.list')).toHaveLength(1)
    expect(store.getSnapshot().watch).toMatchObject({ phase: 'ready', canEdit: false })
    expect(store.getSnapshot().watch.rows).toHaveLength(1)
  })

  it('reloads on reconnect once it has loaded, and not before', async () => {
    const fresh = liveHarness()
    fresh.hub.handlers.set('supervisor.watch.list', () => VIEW)
    const freshController = createHubController({ connection: fresh.conn, store: createHubStore() })
    freshController.start()
    await goLive(fresh)
    await settle()
    expect(fresh.hub.all('supervisor.watch.list')).toHaveLength(0)

    const h = liveHarness()
    h.hub.handlers.set('supervisor.watch.list', () => VIEW)
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    controller.start()
    const socket = await goLive(h)
    await controller.loadWatches()
    expect(h.hub.all('supervisor.watch.list')).toHaveLength(1)
    socket.drop(1006)
    await settle()
    h.timers.advance(10_000)
    await settle()
    const again = h.hub.sockets.at(-1)
    again?.open()
    await settle()
    expect(h.hub.all('supervisor.watch.list').length).toBeGreaterThanOrEqual(2)
  })

  it('saves with the exact wire arguments and applies the returned row', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.watch.list', () => VIEW)
    h.hub.handlers.set('supervisor.watch.set', () => ({ watch: { ...BARE_ROW, enabled: true, active: true, review_interval_minutes: 30, expires_at: '2026-10-24T13:00:00.000Z', enabled_by: { id: 'u1', name: 'Ann' }, updated_at: '2026-10-24T12:00:00.000Z' } }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    await controller.loadWatches()
    const saved = await controller.setWatch({ workspaceId: 'w1', taskId: 't2', enabled: true, reviewIntervalMinutes: 30, expiresAt: '2026-10-24T13:00:00.000Z' })
    expect(saved).toBe(true)
    expect(h.hub.all('supervisor.watch.set')[0]?.params).toEqual({ workspace_id: 'w1', task_id: 't2', enabled: true, review_interval_minutes: 30, expires_at: '2026-10-24T13:00:00.000Z' })
    expect(store.getSnapshot().watch.rows.find(r => r.taskId === 't2')).toMatchObject({ enabled: true, active: true, reviewIntervalMinutes: 30 })
    expect('savingKey' in store.getSnapshot().watch).toBe(false)
  })

  it('omits unchanged fields and sends null to clear', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.watch.set', () => ({ watch: ROW }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    await controller.setWatch({ workspaceId: 'w1', taskId: 't1', enabled: false })
    await controller.setWatch({ workspaceId: 'w1', taskId: 't1', enabled: true, reviewIntervalMinutes: null, expiresAt: null })
    const frames = h.hub.all('supervisor.watch.set')
    expect(frames[0]?.params).toEqual({ workspace_id: 'w1', task_id: 't1', enabled: false })
    expect(frames[1]?.params).toEqual({ workspace_id: 'w1', task_id: 't1', enabled: true, review_interval_minutes: null, expires_at: null })
  })

  it('refuses an interval outside 5..120 locally and sends nothing', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.watch.set', () => ({ watch: ROW }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    for (const bad of [4, 121, 7.5, Number.NaN]) {
      expect(await controller.setWatch({ workspaceId: 'w1', taskId: 't1', enabled: true, reviewIntervalMinutes: bad })).toBe(false)
    }
    expect(h.hub.all('supervisor.watch.set')).toHaveLength(0)
    expect(store.getSnapshot().watch.saveError).toContain('5')
    expect(await controller.setWatch({ workspaceId: 'w1', taskId: 't1', enabled: true, reviewIntervalMinutes: 5 })).toBe(true)
    expect(await controller.setWatch({ workspaceId: 'w1', taskId: 't1', enabled: true, reviewIntervalMinutes: 120 })).toBe(true)
  })

  it('keeps a refused save visible and never reports success', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.watch.set', () => ({ __error: { code: 'limit', message: 'At most 50 tasks can be watched.' } }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    expect(await controller.setWatch({ workspaceId: 'w1', taskId: 't1', enabled: true })).toBe(false)
    expect(store.getSnapshot().watch.saveError).toBe('At most 50 tasks can be watched.')
    expect(store.getSnapshot().watch.unsupported).toBe(false)
    expect('savingKey' in store.getSnapshot().watch).toBe(false)
  })

  it('does not trust an unreadable save answer: it reads back what the Hub holds', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.watch.set', () => ({ watch: { workspace_id: 'w1', task_id: 't1' } }))
    h.hub.handlers.set('supervisor.watch.list', () => VIEW)
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    expect(await controller.setWatch({ workspaceId: 'w1', taskId: 't1', enabled: true })).toBe(false)
    await settle()
    expect(h.hub.all('supervisor.watch.list')).toHaveLength(1)
    expect(store.getSnapshot().watch.rows).toHaveLength(2)
  })

  it('never retries an uncertain write: it asks what the Hub holds instead', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.watch.set', () => undefined)
    h.hub.handlers.set('supervisor.watch.list', () => VIEW)
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    const pending = controller.setWatch({ workspaceId: 'w1', taskId: 't1', enabled: true })
    await settle()
    h.timers.advance(16_000)
    expect(await pending).toBe(false)
    await settle()
    expect(h.hub.all('supervisor.watch.set')).toHaveLength(1)
    expect(h.hub.all('supervisor.watch.list')).toHaveLength(1)
    expect(store.getSnapshot().watch.saveError).toContain('Could not confirm')
    expect('savingKey' in store.getSnapshot().watch).toBe(false)
    expect(store.getSnapshot().watch.rows).toHaveLength(2)
  })

  it('hides Watch on a Hub that does not know it, and shows a real error otherwise', async () => {
    for (const code of ['unsupported', 'invalid_request']) {
      const old = liveHarness()
      old.hub.handlers.set('supervisor.watch.list', () => ({ __error: { code, message: 'unknown hub method' } }))
      const oldStore = createHubStore()
      const oldController = createHubController({ connection: old.conn, store: oldStore })
      await goLive(old)
      oldController.start()
      await oldController.loadWatches()
      expect(oldStore.getSnapshot().watch, code).toMatchObject({ phase: 'ready', unsupported: true, rows: [] })
    }

    const broken = liveHarness()
    broken.hub.handlers.set('supervisor.watch.list', () => ({ can_edit: true, watches: [{ workspace_id: 'w1' }] }))
    const brokenStore = createHubStore()
    const brokenController = createHubController({ connection: broken.conn, store: brokenStore })
    await goLive(broken)
    brokenController.start()
    await brokenController.loadWatches()
    expect(brokenStore.getSnapshot().watch.phase).toBe('error')
    expect(brokenStore.getSnapshot().watch.rows).toEqual([])
    expect(brokenStore.getSnapshot().watch.unsupported).toBe(false)
  })

  it('replaces one row on each supervisor.watch_update, latest wins, without asking again', () => {
    const fake = fakeConnection()
    const store = createHubStore()
    createHubController({ connection: fake.connection, store }).start()
    store.dispatch({ type: 'watchView', view: parseWatchView(VIEW)! })
    for (let edits = 5; edits <= 9; edits += 1) {
      fake.notice('supervisor.watch_update', { watch: { ...ROW, stats: { ...STATS, edits, summary_line: `${edits} edits` } } })
    }
    const rows = store.getSnapshot().watch.rows
    expect(rows.map(r => r.taskId)).toEqual(['t1', 't2'])
    expect(rows[0]?.stats?.edits).toBe(9)
    expect(rows[0]?.stats?.summaryLine).toBe('9 edits')
    // A cleared anomaly list replaces the old one whole.
    fake.notice('supervisor.watch_update', { watch: { ...ROW, stats: { ...STATS, anomalies: [] } } })
    expect(store.getSnapshot().watch.rows[0]?.stats?.anomalies).toEqual([])
    expect(store.getSnapshot().watch.canEdit).toBe(true)
    expect(fake.requests).toEqual([])
  })

  it('adds a row for a task it has not listed yet', () => {
    const fake = fakeConnection()
    const store = createHubStore()
    createHubController({ connection: fake.connection, store }).start()
    fake.notice('supervisor.watch_update', { watch: ROW })
    expect(store.getSnapshot().watch.rows.map(r => r.taskId)).toEqual(['t1'])
  })

  it('asks the Hub again when a supervisor.watch_update payload is unreadable', () => {
    const fake = fakeConnection()
    const store = createHubStore()
    createHubController({ connection: fake.connection, store }).start()
    fake.notice('supervisor.watch_update', { watch: { workspace_id: 'w1', task_id: 't1', active: 'maybe' } })
    expect(fake.requests).toEqual(['supervisor.watch.list'])
    expect(store.getSnapshot().watch.rows).toEqual([])
  })

  it('an unknown anomaly kind in an update counts as unreadable and keeps the old row', () => {
    const fake = fakeConnection()
    const store = createHubStore()
    createHubController({ connection: fake.connection, store }).start()
    store.dispatch({ type: 'watchView', view: parseWatchView(VIEW)! })
    fake.notice('supervisor.watch_update', { watch: { ...ROW, stats: { ...STATS, edits: 99, anomalies: [{ kind: 'on_fire', since: STATS.since, detail: 'x' }] } } })
    expect(store.getSnapshot().watch.rows[0]?.stats?.edits).toBe(4)
    expect(fake.requests).toEqual(['supervisor.watch.list'])
  })

  it('the owner opens the Autopilot thread by itself when only a watch is active', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.get', () => AUTOPILOT_OWNER)
    h.hub.handlers.set('supervisor.watch.list', () => ({ can_edit: true, watches: [ROW] }))
    h.hub.handlers.set('supervisor.thread.open', params => ({ thread_id: params.thread_id, messages: [], confirmations: [] }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    store.dispatch({ type: 'self', userId: 'u1' })
    // Autopilot is off everywhere: nothing opens yet.
    await controller.loadAutopilot()
    await settle()
    expect(h.hub.all('supervisor.thread.open')).toHaveLength(0)
    await controller.loadWatches()
    await settle()
    expect(h.hub.all('supervisor.thread.open').map(f => f.params)).toEqual([{ thread_id: 'autopilot-u1' }])
    expect(store.getSnapshot().autopilotThread.phase).toBe('ready')
  })

  it('a non-owner with an active watch never requests the thread', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.get', () => ({ can_edit: false, workspaces: [] }))
    h.hub.handlers.set('supervisor.watch.list', () => ({ can_edit: false, watches: [ROW] }))
    h.hub.handlers.set('supervisor.thread.open', () => ({ thread_id: 'x', messages: [], confirmations: [] }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    store.dispatch({ type: 'self', userId: 'u2' })
    await controller.loadAutopilot()
    await controller.loadWatches()
    await settle()
    expect(h.hub.all('supervisor.thread.open')).toHaveLength(0)
  })
})

describe('re-listing after unreadable watch updates is debounced', () => {
  const unreadable = { watch: { workspace_id: 'w1', task_id: 't1', active: 'maybe' } }

  it('sends the first re-list at once, collapses a burst into one trailing re-list, and never loses the last one', () => {
    vi.useFakeTimers()
    try {
      const fake = fakeConnection()
      const controller = createHubController({ connection: fake.connection, store: createHubStore() })
      controller.start()
      for (let i = 0; i < 6; i += 1) fake.notice('supervisor.watch_update', unreadable)
      expect(fake.requests).toEqual(['supervisor.watch.list'])
      vi.advanceTimersByTime(1_999)
      expect(fake.requests).toHaveLength(1)
      vi.advanceTimersByTime(1)
      expect(fake.requests).toHaveLength(2)
      // Nothing else is pending: the burst produced exactly one trailing list.
      vi.advanceTimersByTime(1_000)
      expect(fake.requests).toHaveLength(2)

      // A bad update inside the window of the trailing list waits for the window to end and is still sent.
      fake.notice('supervisor.watch_update', unreadable)
      expect(fake.requests).toHaveLength(2)
      vi.advanceTimersByTime(999)
      expect(fake.requests).toHaveLength(2)
      vi.advanceTimersByTime(1)
      expect(fake.requests).toHaveLength(3)

      // After a quiet spell the next one goes out immediately.
      vi.advanceTimersByTime(10_000)
      fake.notice('supervisor.watch_update', unreadable)
      expect(fake.requests).toHaveLength(4)
      vi.advanceTimersByTime(60_000)
      expect(fake.requests).toHaveLength(4)
      controller.dispose()
    } finally {
      vi.useRealTimers()
    }
  })

  it('readable updates never re-list, and dispose cancels a pending trailing re-list', () => {
    vi.useFakeTimers()
    try {
      const fake = fakeConnection()
      const store = createHubStore()
      const controller = createHubController({ connection: fake.connection, store })
      controller.start()
      for (let i = 0; i < 5; i += 1) fake.notice('supervisor.watch_update', { watch: ROW })
      expect(fake.requests).toEqual([])
      fake.notice('supervisor.watch_update', unreadable)
      fake.notice('supervisor.watch_update', unreadable)
      expect(fake.requests).toHaveLength(1)
      controller.dispose()
      vi.advanceTimersByTime(10_000)
      expect(fake.requests).toHaveLength(1)
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('the Autopilot thread when the Autopilot list is unknown', () => {
  it('opens for the owner when the Autopilot list failed but an active watch says they own the Hub', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.get', () => ({ __error: { code: 'internal', message: 'boom' } }))
    h.hub.handlers.set('supervisor.watch.list', () => ({ can_edit: true, watches: [ROW] }))
    h.hub.handlers.set('supervisor.thread.open', params => ({ thread_id: params.thread_id, messages: [], confirmations: [] }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    store.dispatch({ type: 'self', userId: 'u1' })
    await controller.loadAutopilot()
    expect(store.getSnapshot().autopilot.phase).toBe('error')
    await controller.loadWatches()
    await settle()
    expect(h.hub.all('supervisor.thread.open').map(f => f.params)).toEqual([{ thread_id: 'autopilot-u1' }])
    expect(store.getSnapshot().autopilotThread.phase).toBe('ready')
    expect(autopilotThreadAvailable(store.getSnapshot())).toBe(true)
  })

  it('does the same on a Hub without Autopilot, and when Autopilot was never loaded', () => {
    let state = reduceHub(initialHubState(), { type: 'self', userId: 'u1' })
    state = reduceHub(state, { type: 'watchView', view: parseWatchView({ can_edit: true, watches: [ROW] })! })
    expect(isHubOwner(state)).toBe(true)
    expect(autopilotThreadAvailable(state)).toBe(true)
    const old = reduceHub(state, { type: 'autopilot', patch: { phase: 'ready', unsupported: true } })
    expect(autopilotThreadAvailable(old)).toBe(true)
    const failed = reduceHub(state, { type: 'autopilot', patch: { phase: 'error', error: 'boom' } })
    expect(autopilotThreadAvailable(failed)).toBe(true)
  })

  it('stays closed for non-owners, for a lapsed watch, and when a loaded Autopilot answer says not owner', () => {
    let state = reduceHub(initialHubState(), { type: 'self', userId: 'u2' })
    const viewerWatches = reduceHub(state, { type: 'watchView', view: parseWatchView({ can_edit: false, watches: [ROW] })! })
    expect(isHubOwner(viewerWatches)).toBe(false)
    expect(autopilotThreadAvailable(viewerWatches)).toBe(false)
    state = reduceHub(state, { type: 'watchView', view: parseWatchView({ can_edit: true, watches: [{ ...ROW, active: false }] })! })
    expect(autopilotThreadAvailable(state)).toBe(false)
    // A loaded Autopilot answer wins over the watch signal.
    const loadedViewer = reduceHub(
      reduceHub(initialHubState(), { type: 'self', userId: 'u2' }),
      { type: 'autopilotView', view: parseAutopilotView({ can_edit: false, workspaces: [] })! },
    )
    const withWatches = reduceHub(loadedViewer, { type: 'watchView', view: parseWatchView({ can_edit: true, watches: [ROW] })! })
    expect(isHubOwner(withWatches)).toBe(false)
    expect(autopilotThreadAvailable(withWatches)).toBe(false)
  })

  it('a non-owner whose Autopilot list failed and whose watch list is read-only never requests the thread', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.get', () => ({ __error: { code: 'internal', message: 'boom' } }))
    h.hub.handlers.set('supervisor.watch.list', () => ({ can_edit: false, watches: [ROW] }))
    h.hub.handlers.set('supervisor.thread.open', () => ({ thread_id: 'x', messages: [], confirmations: [] }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    store.dispatch({ type: 'self', userId: 'u2' })
    await controller.loadAutopilot()
    await controller.loadWatches()
    await settle()
    expect(h.hub.all('supervisor.thread.open')).toHaveLength(0)
  })
})

describe('supervisor.watch_update through the real frame parser', () => {
  it('is accepted as a Hub event frame and replaces the row', async () => {
    const h = liveHarness()
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    const socket = await goLive(h)
    controller.start()
    store.dispatch({ type: 'watchView', view: parseWatchView(VIEW)! })
    socket.event('supervisor.watch_update', { watch: { ...ROW, stats: { ...STATS, edits: 12, summary_line: '12 edits' } } })
    expect(h.errors).toEqual([])
    expect(store.getSnapshot().watch.rows[0]?.stats?.edits).toBe(12)
    expect(store.getSnapshot().watch.rows[0]?.stats?.summaryLine).toBe('12 edits')
    expect(store.getSnapshot().watch.rows.map(r => r.taskId)).toEqual(['t1', 't2'])
  })

  it('an unreadable update that arrives as a real frame triggers one re-list', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.watch.list', () => VIEW)
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    const socket = await goLive(h)
    controller.start()
    socket.event('supervisor.watch_update', { watch: { workspace_id: 'w1', task_id: 't1', active: 'maybe' } })
    await settle()
    expect(h.errors).toEqual([])
    expect(h.hub.all('supervisor.watch.list')).toHaveLength(1)
    expect(store.getSnapshot().watch.rows.map(r => r.taskId)).toEqual(['t1', 't2'])
  })

  it('an event type the protocol does not list is still rejected by the parser', async () => {
    const h = liveHarness()
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    const socket = await goLive(h)
    controller.start()
    socket.event('supervisor.watch_updated', { watch: ROW })
    expect(h.errors).toHaveLength(1)
    expect(store.getSnapshot().watch.rows).toEqual([])
  })
})
