// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { autopilotThreadId } from '../../client/autopilot.js'
import type { HubConnection } from '../../client/hubConnection.js'
import { createHubController } from '../../client/hubController.js'
import { parseAutopilotView } from '../../client/hubParsers.js'
import { createHubStore } from '../../client/hubStore.js'
import { goLive, harness, settle } from './fakes.js'

const ROW = {
  workspace_id: 'w1', label: 'Mac · app', enabled: true, instruction: '', expires_at: null,
  enabled_by: { id: 'u1', name: 'Ann' }, updated_at: '2026-10-24T10:00:00.000Z', active: true,
}
const OWNER_VIEW = { can_edit: true, workspaces: [ROW] }
const NOW_ISO = '2026-10-24T10:00:00.000Z'

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

describe('autopilot thread controller', () => {
  it('uses the id autopilot-<selfUserId>', () => {
    expect(autopilotThreadId('u1')).toBe('autopilot-u1')
  })

  it('the owner opens autopilot-<self> and gets its messages and cards', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.thread.open', params => ({
      thread_id: params.thread_id, busy: false,
      messages: [{ id: 'm1', role: 'system', text: 'Autopilot: woke for a question in “Fix”.', at: NOW_ISO }],
      confirmations: [{ id: 'c1', tool: 'task.start', summary: 'Start', state: 'pending' }, { id: 'c2', tool: 'task.steer', summary: 'Steer', state: 'confirmed', auto: true, confirmed_by: 'Ann' }],
    }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    store.dispatch({ type: 'self', userId: 'u1' })
    store.dispatch({ type: 'autopilotView', view: parseAutopilotView(OWNER_VIEW)! })
    await controller.autopilotThreadOpen()
    expect(h.hub.all('supervisor.thread.open')[0]?.params).toEqual({ thread_id: 'autopilot-u1' })
    const thread = store.getSnapshot().autopilotThread
    expect(thread).toMatchObject({ phase: 'ready', threadId: 'autopilot-u1' })
    expect(thread.messages.map(m => m.id)).toEqual(['m1'])
    expect(thread.confirmations.map(c => [c.id, c.state, c.auto === true])).toEqual([['c1', 'pending', false], ['c2', 'confirmed', true]])
    // Nothing leaked into the shared thread.
    expect(store.getSnapshot().supervisor.messages).toEqual([])
    expect(store.getSnapshot().supervisor.confirmations).toEqual([])
  })

  it('loading Autopilot as the owner with an active workspace opens the thread by itself', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.get', () => OWNER_VIEW)
    h.hub.handlers.set('supervisor.thread.open', params => ({ thread_id: params.thread_id, messages: [], confirmations: [] }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    store.dispatch({ type: 'self', userId: 'u1' })
    await controller.loadAutopilot()
    await settle()
    expect(h.hub.all('supervisor.thread.open').map(f => f.params)).toEqual([{ thread_id: 'autopilot-u1' }])
    expect(store.getSnapshot().autopilotThread.phase).toBe('ready')
  })

  it('a non-owner never requests the thread', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.get', () => ({ can_edit: false, workspaces: [ROW] }))
    h.hub.handlers.set('supervisor.thread.open', () => ({ thread_id: 'x', messages: [], confirmations: [] }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    store.dispatch({ type: 'self', userId: 'u2' })
    await controller.loadAutopilot()
    await controller.autopilotThreadOpen()
    await settle()
    expect(h.hub.all('supervisor.thread.open')).toHaveLength(0)
    expect(store.getSnapshot().autopilotThread.phase).toBe('idle')
  })

  it('no active workspace means no request; not_found hides the view and is never asked again', async () => {
    const quiet = liveHarness()
    quiet.hub.handlers.set('supervisor.autopilot.get', () => ({ can_edit: true, workspaces: [{ ...ROW, enabled: false, active: false }] }))
    const quietStore = createHubStore()
    const quietController = createHubController({ connection: quiet.conn, store: quietStore })
    await goLive(quiet)
    quietController.start()
    quietStore.dispatch({ type: 'self', userId: 'u1' })
    await quietController.loadAutopilot()
    await settle()
    expect(quiet.hub.all('supervisor.thread.open')).toHaveLength(0)

    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.get', () => OWNER_VIEW)
    h.hub.handlers.set('supervisor.thread.open', () => ({ __error: { code: 'not_found', message: 'no such thread' } }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    store.dispatch({ type: 'self', userId: 'u1' })
    await controller.loadAutopilot()
    await settle()
    expect(store.getSnapshot().autopilotThread.phase).toBe('hidden')
    // Another load, another change event and a direct open: still one request.
    await controller.loadAutopilot()
    await controller.autopilotThreadOpen()
    await settle()
    expect(h.hub.all('supervisor.thread.open')).toHaveLength(1)
  })

  it('routes thread_event and confirmation events by thread_id, never mixing the two threads', () => {
    const fake = fakeConnection()
    const store = createHubStore()
    createHubController({ connection: fake.connection, store }).start()
    store.dispatch({ type: 'self', userId: 'u1' })

    fake.notice('supervisor.thread_event', { thread_id: 'autopilot-u1', message: { id: 'a1', role: 'system', text: 'Autopilot: woke for …', at: NOW_ISO } })
    fake.notice('supervisor.confirmation', { thread_id: 'autopilot-u1', confirmation: { id: 'c1', tool: 'task.start', summary: 'Start', state: 'pending' } })
    fake.notice('supervisor.thread_event', { thread_id: 'autopilot-u1', confirmation: { id: 'c2', tool: 'task.steer', summary: 'Steer', state: 'confirmed', auto: true, confirmed_by: 'Ann' } })
    fake.notice('supervisor.thread_event', { thread_id: 'shared', message: { id: 's1', role: 'user', text: 'hello', at: NOW_ISO } })
    fake.notice('supervisor.confirmation', { thread_id: 'shared', confirmation: { id: 'c9', tool: 'task.stop', summary: 'Stop', state: 'pending' } })
    // Another person's autopilot thread is not ours, and never lands in the shared thread either.
    fake.notice('supervisor.thread_event', { thread_id: 'autopilot-u2', message: { id: 'o1', role: 'system', text: 'not mine', at: NOW_ISO } })
    fake.notice('supervisor.confirmation', { thread_id: 'autopilot-u2', confirmation: { id: 'o2', tool: 'task.stop', summary: 'Stop', state: 'pending' } })

    const { autopilotThread, supervisor } = store.getSnapshot()
    expect(autopilotThread.messages.map(m => m.id)).toEqual(['a1'])
    expect(autopilotThread.confirmations.map(c => c.id)).toEqual(['c1', 'c2'])
    expect(supervisor.messages.map(m => m.id)).toEqual(['s1'])
    expect(supervisor.confirmations.map(c => c.id)).toEqual(['c9'])
  })

  it('tracks the working flag of the Autopilot thread without touching the shared one', () => {
    const fake = fakeConnection()
    const store = createHubStore()
    createHubController({ connection: fake.connection, store }).start()
    store.dispatch({ type: 'self', userId: 'u1' })
    fake.notice('supervisor.thread_event', { thread_id: 'autopilot-u1', activity: 'working' })
    expect(store.getSnapshot().autopilotThread.working).toBe(true)
    expect(store.getSnapshot().supervisor.working).toBeFalsy()
    fake.notice('supervisor.thread_event', { thread_id: 'autopilot-u1', activity: 'idle' })
    expect(store.getSnapshot().autopilotThread.working).toBe(false)
  })

  it('confirming a card in the Autopilot thread sends that thread_id and updates only that thread', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.confirmation.respond', () => ({ thread_id: 'autopilot-u1', confirmation: { id: 'c1', tool: 'task.start', summary: 'Start', state: 'confirmed' } }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    store.dispatch({ type: 'self', userId: 'u1' })
    store.dispatch({ type: 'autopilotThreadConfirmations', confirmations: [{ id: 'c1', tool: 'task.start', summary: 'Start', targets: [], state: 'pending' }] })
    store.dispatch({ type: 'supervisorConfirmations', confirmations: [{ id: 'c1', tool: 'other', summary: 'Shared card with same id', targets: [], state: 'pending' }] })
    await controller.autopilotThreadRespond('c1', 'confirm')
    const frame = h.hub.all('supervisor.confirmation.respond')[0]
    expect(frame?.params).toMatchObject({ confirmation_id: 'c1', decision: 'confirm', thread_id: 'autopilot-u1' })
    expect(store.getSnapshot().autopilotThread.confirmations[0]?.state).toBe('confirmed')
    expect(store.getSnapshot().supervisor.confirmations[0]?.state).toBe('pending')
  })

  it('shows the Hub\'s "choose a new expiry" refusal and keeps the save visible', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.set', () => ({ __error: { code: 'invalid_request', message: 'choose a new expiry' } }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    expect(await controller.setAutopilot({ workspaceId: 'w1', enabled: true, instruction: '' })).toBe(false)
    // No expires_at was sent: the stored one is kept by the Hub.
    expect(h.hub.all('supervisor.autopilot.set')[0]?.params).toEqual({ workspace_id: 'w1', enabled: true, instruction: '' })
    expect(store.getSnapshot().autopilot.saveError).toBe('choose a new expiry')
    expect(store.getSnapshot().autopilot.unsupported).toBe(false)
    expect('savingId' in store.getSnapshot().autopilot).toBe(false)
  })
})
