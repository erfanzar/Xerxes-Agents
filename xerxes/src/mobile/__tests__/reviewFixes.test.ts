// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { AGENTS_REFRESH_MS, AGENT_DETAIL_REFRESH_MS, createHubController } from '../../client/hubController.js'
import { parseHistoryPage } from '../../client/hubParsers.js'
import { createHubStore, reduceHub, type HubState } from '../../client/hubStore.js'
import { EMPTY_TAIL, foldAll, foldSessionEvent } from '../../client/sessionFold.js'
import { createResumeGate } from '../pwa/resumeGate.js'
import { PUSH_CHANGED_MESSAGE, installServiceWorker, safeTaskPath, type SwClient, type SwScope } from '../pwa/swCore.js'
import { parseRoute } from '../routes.js'
import { batch, ev, goLive, harness, settle } from './fakes.js'
import { hubEvent, liveState, summary } from './helpers.js'

const ORIGIN = 'https://hub.example'
const page = (actions: readonly unknown[]) => ({ history: { actions, before: null, has_more: false } })
const message = (id: string, role: string, content: string) => ({ id, messages: [{ role, content }], executions: [], thinking: [] })

function selectedState(): HubState {
  return reduceHub(liveState([summary({ id: 's1', status: 'working' })]), { type: 'select', workspaceId: 'w1', sessionId: 's1' })
}

describe('P2-1: heartbeat probes do not resync; foreground probes do', () => {
  it('a quiet visible phone re-reads nothing on the heartbeat, and a resync keeps a failed send', async () => {
    const h = harness({ heartbeatMs: 25_000 })
    let lists = 0
    h.hub.handlers.set('hub.workspaces.list', () => { lists += 1; return { workspaces: [] } })
    const socket = await goLive(h)
    const store = createHubStore(selectedState())
    store.dispatch({ type: 'send', state: { phase: 'failed', message: 'Offline. Nothing was sent.' }, workspaceId: 'w1', sessionId: 's1' })
    const controller = createHubController({ connection: h.conn, store, timers: h.timers, now: () => h.timers.now })
    controller.start()
    for (let beat = 0; beat < 4; beat += 1) {
      h.timers.advance(25_000)
      await settle()
    }
    expect(socket.framesFor('hub.hello').length).toBeGreaterThanOrEqual(4)
    expect(lists).toBe(0)

    await controller.resync()
    await settle()
    expect(lists).toBe(1)
    // The same link answered: the failed send is still news until a retry.
    expect(store.getSnapshot().selected?.send.phase).toBe('failed')

    // A real reconnect clears it (UI-15).
    socket.drop(1006)
    h.timers.advance(10_000)
    await settle()
    h.hub.socket(h.hub.sockets.length - 1).open()
    await settle()
    await settle()
    expect(lists).toBe(2)
    expect(store.getSnapshot().selected?.send.phase).toBe('idle')
    controller.dispose()
  })
})

describe('P3-9: one foreground resumes once', () => {
  it('drops repeated foreground signals within the interval; the network coming back always runs', () => {
    let clock = 0
    let runs = 0
    const gate = createResumeGate(() => clock, () => { runs += 1 }, 4_000)
    expect(gate.trigger()).toBe(true)
    // visibilitychange and pageshow of the same iOS foreground.
    expect(gate.trigger()).toBe(false)
    clock = 3_999
    expect(gate.trigger()).toBe(false)
    clock = 4_000
    expect(gate.trigger()).toBe(true)
    gate.force()
    expect(runs).toBe(3)
  })
})

describe('P2-2: streamed text that joined mid-reply', () => {
  it('a live suffix is covered by the stored message once the turn ended: no duplicate', () => {
    let state = selectedState()
    state = reduceHub(state, { type: 'events', events: [hubEvent('text_part', { text: 'the answer.' }), hubEvent('turn_end', {})] })
    const live = state.selected!.tail.blocks
    expect(live).toHaveLength(1)
    const parsed = parseHistoryPage(page([message('0:message', 'user', 'q'), message('1:message', 'assistant', 'Here is the answer.')]))
    state = reduceHub(state, { type: 'historyLoaded', workspaceId: 'w1', sessionId: 's1', page: parsed, throughIds: live.map(block => block.id) })
    expect(state.selected!.tail.blocks).toEqual([])
    expect(state.selected!.older.filter(block => block.kind === 'assistant')).toHaveLength(1)
  })

  it('history is authoritative after the turn ended, and a block never keeps streaming after a missed turn_end', () => {
    let state = reduceHub(selectedState(), { type: 'events', events: [hubEvent('think_part', { think: 'pondering' }), hubEvent('text_part', { text: 'unrelated words' })] })
    const known = state.selected!.tail.blocks.map(block => block.id)
    // A block that arrived while the read was in flight, then the session reported idle (turn_end lost).
    state = reduceHub(state, { type: 'events', events: [hubEvent('turn_begin', { user_input: 'next' }), hubEvent('text_part', { text: 'late' })] })
    const selected = state.selected!
    state = { ...state, selected: { ...selected, tail: { ...selected.tail, status: 'idle' } } }
    const parsed = parseHistoryPage(page([message('0:message', 'assistant', 'Stored reply')]))
    state = reduceHub(state, { type: 'historyLoaded', workspaceId: 'w1', sessionId: 's1', page: parsed, throughIds: known })
    const blocks = state.selected!.tail.blocks
    // Text that existed when the read started is dropped (history has the final copy).
    expect(blocks.some(block => block.kind === 'thinking')).toBe(false)
    expect(blocks.some(block => block.kind === 'assistant' && block.text === 'unrelated words')).toBe(false)
    const late = blocks.find(block => block.kind === 'assistant')
    expect(late).toMatchObject({ text: 'late', streaming: false })
  })

  it('mid-turn, a short fragment is not matched to an older answer by containment', () => {
    let state = reduceHub(selectedState(), { type: 'events', events: [hubEvent('text_part', { text: 'yes' })] })
    const live = state.selected!.tail.blocks.map(block => block.id)
    const parsed = parseHistoryPage(page([message('0:message', 'assistant', 'Earlier answer: yes, it works.')]))
    state = reduceHub(state, { type: 'historyLoaded', workspaceId: 'w1', sessionId: 's1', page: parsed, throughIds: live })
    expect(state.selected!.tail.blocks).toMatchObject([{ kind: 'assistant', text: 'yes', streaming: true }])
  })
})

describe('P2-5: a complete pending list from the Hub replaces the stale one', () => {
  const request = (id: string) => ({ id, type: 'approval_request', payload: { request_id: id, tool_name: 'exec_command', arguments: { command: `echo ${id}` } } })

  it('drops a card resolved while away, keeps a card that arrived during the read, and merges a partial list', () => {
    let state = reduceHub(selectedState(), { type: 'events', events: [hubEvent('approval_request', request('p1').payload), hubEvent('approval_request', request('p2').payload)] })
    const served = foldAll(EMPTY_TAIL, [request('p2')]).pending
    // Partial (pending_complete: false): merge only.
    const partial = reduceHub(state, { type: 'pendingRestored', workspaceId: 'w1', sessionId: 's1', pending: served })
    expect(partial.selected!.tail.pending.map(item => item.id)).toEqual(['p1', 'p2'])
    // p3 arrives live while the history read is in flight.
    state = reduceHub(state, { type: 'events', events: [hubEvent('approval_request', request('p3').payload)] })
    state = reduceHub(state, { type: 'pendingRestored', workspaceId: 'w1', sessionId: 's1', pending: served, complete: true, throughIds: ['p1', 'p2'] })
    expect(state.selected!.tail.pending.map(item => item.id)).toEqual(['p2', 'p3'])
    expect(state.selected!.pendingChecked).toBe(true)
  })

  it('the controller passes pending_complete through from hub.history', async () => {
    const h = harness()
    h.hub.handlers.set('hub.events.read', () => ({ watermark: 'live-edge' }))
    h.hub.handlers.set('hub.presence.update', () => ({}))
    h.hub.handlers.set('agents.list', () => ({ agents: [] }))
    h.hub.handlers.set('hub.history', () => ({ ...page([message('0:message', 'user', 'q')]), pending_interactions: [], pending_complete: true }))
    await goLive(h)
    const store = createHubStore(liveState([summary({ id: 's1', status: 'needs_you', pendingCount: 1 })]))
    const controller = createHubController({ connection: h.conn, store, timers: h.timers, now: () => h.timers.now })
    controller.start()
    controller.openSession('w1', 's1')
    await settle()
    await settle()
    // A card this phone holds whose resolution was lost to a gap.
    store.dispatch({ type: 'events', events: [hubEvent('approval_request', request('gone').payload)] })
    expect(store.getSnapshot().selected!.tail.pending).toHaveLength(1)
    // A foreground resync re-reads history; the Hub's complete (empty) list wins.
    await controller.resync()
    await settle()
    await settle()
    expect(store.getSnapshot().selected!.tail.pending).toEqual([])
    controller.dispose()
  })
})

describe('P3-2 / P3-8: permission flags and streaming kinds', () => {
  it('a real permission with {} inputs and no description keeps Allow; only the Hub flags hide details', () => {
    const plain = foldSessionEvent(EMPTY_TAIL, { id: 'a', type: 'approval_request', payload: { request_id: 'r1', tool_name: 'mcp__x__ping', arguments: {} } })
    expect(plain.pending[0]).toMatchObject({ id: 'r1', inputs: [], detailsHidden: false })
    const redacted = foldSessionEvent(EMPTY_TAIL, { id: 'b', type: 'approval_request', payload: { request_id: 'r2', redacted: true } })
    expect(redacted.pending[0]?.detailsHidden).toBe(true)
    const hidden = foldSessionEvent(EMPTY_TAIL, { id: 'c', type: 'approval_request', payload: { request_id: 'r3', details_hidden: true, description: 'x' } })
    expect(hidden.pending[0]?.detailsHidden).toBe(true)
  })

  it('text after thinking ends the thinking block', () => {
    const tail = foldAll(EMPTY_TAIL, [
      { id: 't', type: 'think_part', payload: { think: 'hmm' } },
      { id: 'x', type: 'text_part', payload: { text: 'Answer' } },
    ])
    expect(tail.blocks).toMatchObject([{ kind: 'thinking', streaming: false }, { kind: 'assistant', streaming: true }])
  })
})

describe('P3-4: the open agent refreshes at most every 5 s, only while open', () => {
  it('throttles agents.inspect and stops once the sheet closes', async () => {
    const h = harness()
    h.hub.handlers.set('hub.events.read', () => ({ watermark: 'live-edge' }))
    h.hub.handlers.set('hub.presence.update', () => ({}))
    h.hub.handlers.set('hub.history', () => page([message('0:message', 'user', 'q')]))
    h.hub.handlers.set('agents.list', () => ({ agents: [{ id: 'a1', name: 'r', title: 'R', status: 'running' }] }))
    h.hub.handlers.set('agents.inspect', params => ({ agent: { id: params.agent_id, name: 'r', title: 'R', status: 'running' }, steps: [], truncated: false }))
    const socket = await goLive(h)
    const store = createHubStore(liveState([summary({ id: 's1', status: 'working' })]))
    const controller = createHubController({ connection: h.conn, store, timers: h.timers, now: () => h.timers.now })
    controller.openSession('w1', 's1')
    await settle()
    await controller.inspectAgent('a1')
    const inspects = () => h.hub.all('agents.inspect').length
    expect(inspects()).toBe(1)
    let seq = 1
    const burst = (): void => {
      const events = [ev(seq++, { type: 'subagent_event', payload: { agent_id: 'a1', event: { type: 'tool_call', payload: {} } } })]
      batch(socket, 'sub1', events, `c${seq}`)
    }
    burst()
    await settle()
    h.timers.advance(AGENTS_REFRESH_MS)
    await settle()
    // The list refreshed; the open agent waits for its own, longer interval.
    expect(inspects()).toBe(1)
    burst()
    await settle()
    h.timers.advance(AGENT_DETAIL_REFRESH_MS - AGENTS_REFRESH_MS)
    await settle()
    expect(inspects()).toBe(2)
    controller.closeAgent()
    for (let round = 0; round < 4; round += 1) {
      burst()
      await settle()
      h.timers.advance(AGENT_DETAIL_REFRESH_MS)
      await settle()
    }
    expect(inspects()).toBe(2)
    controller.dispose()
  })
})

interface SwRig {
  readonly scope: SwScope
  readonly listeners: Map<string, (event: never) => void>
  readonly subscribed: BufferSource[]
  readonly messages: unknown[]
  readonly navigated: string[]
  readonly opened: string[]
}

function swRig(existing: unknown = null): SwRig {
  const listeners = new Map<string, (event: never) => void>()
  const subscribed: BufferSource[] = []
  const messages: unknown[] = []
  const navigated: string[] = []
  const opened: string[] = []
  const client: SwClient = {
    url: `${ORIGIN}/`,
    focus: async () => {},
    navigate: async url => { navigated.push(url) },
    postMessage: message => { messages.push(message) },
  }
  const scope: SwScope = {
    location: { origin: ORIGIN },
    registration: {
      showNotification: async () => {},
      pushManager: {
        getSubscription: async () => existing,
        subscribe: async options => { subscribed.push(options.applicationServerKey); return {} },
      },
    },
    clients: { claim: async () => {}, matchAll: async () => [client], openWindow: async url => { opened.push(url) } },
    caches: { open: async () => ({ addAll: async () => {}, match: async () => undefined, put: async () => {} }), keys: async () => [], delete: async () => true },
    skipWaiting: async () => {},
    addEventListener: (type, listener) => { listeners.set(type, listener as (event: never) => void) },
  }
  installServiceWorker(scope, { version: 'v1', shell: ['/index.html'] })
  return { scope, listeners, subscribed, messages, navigated, opened }
}

async function dispatch(rig: SwRig, type: string, event: Record<string, unknown>): Promise<void> {
  const waited: Promise<unknown>[] = []
  ;(rig.listeners.get(type) as (e: unknown) => void)({ ...event, waitUntil: (p: Promise<unknown>) => { waited.push(p) } })
  await Promise.all(waited)
}

describe('C3: pushsubscriptionchange', () => {
  it('re-subscribes with the old VAPID key and asks open pages to re-register with the Hub', async () => {
    const rig = swRig()
    const key = new Uint8Array(new ArrayBuffer(3))
    key.set([1, 2, 3])
    await dispatch(rig, 'pushsubscriptionchange', { oldSubscription: { options: { applicationServerKey: key } }, newSubscription: null })
    expect(rig.subscribed).toEqual([key])
    expect(rig.messages).toEqual([{ type: PUSH_CHANGED_MESSAGE }])
  })

  it('does not subscribe twice when the browser already gave a replacement', async () => {
    const rig = swRig({ endpoint: 'https://push.example/new' })
    await dispatch(rig, 'pushsubscriptionchange', { oldSubscription: { options: { applicationServerKey: new Uint8Array(new ArrayBuffer(1)) } }, newSubscription: { endpoint: 'x' } })
    expect(rig.subscribed).toEqual([])
    expect(rig.messages).toEqual([{ type: PUSH_CHANGED_MESSAGE }])
  })
})

describe('XT-2: notification targets route in the app', () => {
  it('a task push and a XERXES push open URLs the router understands', async () => {
    const taskUrl = `/w/${encodeURIComponent('m-1 ws')}/s/${encodeURIComponent('sess/1')}`
    const rig = swRig()
    await dispatch(rig, 'notificationclick', { notification: { close() {}, data: { url: taskUrl } } })
    await dispatch(rig, 'notificationclick', { notification: { close() {}, data: { url: '/xerxes' } } })
    expect(rig.navigated).toEqual([taskUrl, '/xerxes'])
    expect(parseRoute(rig.navigated[0]!)).toEqual({ name: 'task', workspaceId: 'm-1 ws', sessionId: 'sess/1' })
    expect(parseRoute(rig.navigated[1]!)).toEqual({ name: 'xerxes' })
    expect(parseRoute(safeTaskPath('/w/a/s/b', ORIGIN))).toEqual({ name: 'task', workspaceId: 'a', sessionId: 'b' })
  })
})
