// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import {
  AUTOPILOT_COMPOSER_HINT, autoCardNote, autopilotChipText, autopilotExpiresAt, autopilotExpiryOptions, autopilotThreadId, defaultAutopilotExpiry,
  effectiveExpiryChoice, isExpiryLapsed,
} from '../../client/autopilot.js'
import type { HubConnection } from '../../client/hubConnection.js'
import { createHubController } from '../../client/hubController.js'
import { parseAutopilotRow, parseAutopilotView, parseConfirmation } from '../../client/hubParsers.js'
import { activeAutopilotRows, createHubStore, initialHubState, reduceHub } from '../../client/hubStore.js'
import { goLive, harness, settle } from './fakes.js'

const ROW = {
  workspace_id: 'w1',
  label: 'Mac · app',
  enabled: true,
  instruction: 'Keep going',
  expires_at: '2026-10-25T00:00:00.000Z',
  enabled_by: { id: 'u1', name: 'Ann' },
  updated_at: '2026-10-24T10:00:00.000Z',
  active: true,
}
const IDLE_ROW = { workspace_id: 'w2', label: 'Server · api', enabled: false, instruction: '', expires_at: null, enabled_by: null, updated_at: null, active: false }
const VIEW = { can_edit: true, workspaces: [ROW, IDLE_ROW] }

describe('autopilot parsers', () => {
  it('reads a full row and a never-configured row', () => {
    expect(parseAutopilotRow(ROW)).toEqual({
      workspaceId: 'w1', label: 'Mac · app', enabled: true, instruction: 'Keep going', expiresAt: '2026-10-25T00:00:00.000Z',
      enabledBy: { id: 'u1', name: 'Ann' }, updatedAt: '2026-10-24T10:00:00.000Z', active: true,
    })
    expect(parseAutopilotRow(IDLE_ROW)).toEqual({
      workspaceId: 'w2', label: 'Server · api', enabled: false, instruction: '', expiresAt: null, enabledBy: null, updatedAt: null, active: false,
    })
  })

  it('rejects a row with a missing or wrongly typed field instead of guessing', () => {
    const without = (key: string): unknown => { const copy: Record<string, unknown> = { ...ROW }; delete copy[key]; return copy }
    for (const key of ['workspace_id', 'label', 'enabled', 'instruction', 'expires_at', 'enabled_by', 'updated_at', 'active']) {
      expect(parseAutopilotRow(without(key)), `missing ${key}`).toBeUndefined()
    }
    expect(parseAutopilotRow({ ...ROW, enabled: 'yes' })).toBeUndefined()
    expect(parseAutopilotRow({ ...ROW, active: 1 })).toBeUndefined()
    expect(parseAutopilotRow({ ...ROW, instruction: null })).toBeUndefined()
    expect(parseAutopilotRow({ ...ROW, expires_at: 'tomorrow' })).toBeUndefined()
    expect(parseAutopilotRow({ ...ROW, expires_at: 12 })).toBeUndefined()
    expect(parseAutopilotRow({ ...ROW, updated_at: 'nope' })).toBeUndefined()
    expect(parseAutopilotRow({ ...ROW, enabled_by: { id: 'u1' } })).toBeUndefined()
    expect(parseAutopilotRow({ ...ROW, enabled_by: 'Ann' })).toBeUndefined()
    expect(parseAutopilotRow(null)).toBeUndefined()
    expect(parseAutopilotRow([ROW])).toBeUndefined()
    expect(parseAutopilotRow('junk')).toBeUndefined()
  })

  it('reads the get answer and rejects it whole when anything in it is malformed', () => {
    const view = parseAutopilotView(VIEW)
    expect(view?.canEdit).toBe(true)
    expect(view?.workspaces.map(row => row.workspaceId)).toEqual(['w1', 'w2'])
    expect(parseAutopilotView({ can_edit: false, workspaces: [] })).toEqual({ canEdit: false, workspaces: [] })
    expect(parseAutopilotView({ workspaces: [] })).toBeUndefined()
    expect(parseAutopilotView({ can_edit: 'yes', workspaces: [] })).toBeUndefined()
    expect(parseAutopilotView({ can_edit: true })).toBeUndefined()
    expect(parseAutopilotView({ can_edit: true, workspaces: [ROW, { ...ROW, active: 'x' }] })).toBeUndefined()
    expect(parseAutopilotView({ can_edit: true, workspaces: [ROW, ROW] })).toBeUndefined()
    expect(parseAutopilotView(undefined)).toBeUndefined()
  })

  it('reads auto and confirmed_by on a confirmation card, and leaves ordinary cards unmarked', () => {
    const auto = parseConfirmation({ id: 'a1', tool: 'task.steer', summary: 'Steer', state: 'confirmed', auto: true, confirmed_by: 'Ann' })
    expect(auto).toMatchObject({ id: 'a1', auto: true, confirmedBy: 'Ann', state: 'confirmed' })
    const plain = parseConfirmation({ id: 'a2', tool: 'task.stop', summary: 'Stop', state: 'pending' })
    expect(plain?.auto).toBeUndefined()
    expect(plain?.confirmedBy).toBeUndefined()
    // Only a literal true marks a card as automatic.
    expect(parseConfirmation({ id: 'a3', auto: 'true' })?.auto).toBeUndefined()
  })
})

describe('autopilot helpers', () => {
  it('turns expiry choices into the wire value: keep omits, none clears, durations are ISO in the future', () => {
    const now = Date.parse('2026-10-24T12:00:00.000Z')
    expect(autopilotExpiresAt('keep', now)).toBeUndefined()
    expect(autopilotExpiresAt('none', now)).toBeNull()
    expect(autopilotExpiresAt('1h', now)).toBe('2026-10-24T13:00:00.000Z')
    expect(autopilotExpiresAt('8h', now)).toBe('2026-10-24T20:00:00.000Z')
    expect(autopilotExpiresAt('24h', now)).toBe('2026-10-25T12:00:00.000Z')
    expect(autopilotExpiresAt('7d', now)).toBe('2026-10-31T12:00:00.000Z')
  })

  it('offers "keep current expiry" only while a row has one', () => {
    const now = Date.parse('2026-10-24T12:00:00.000Z')
    expect(autopilotExpiryOptions({ expiresAt: null }, now).map(o => o.label)).toEqual(['No expiry', '1 hour', '8 hours', '24 hours', '7 days'])
    expect(autopilotExpiryOptions({ expiresAt: '2026-10-25T00:00:00.000Z' }, now)[0]).toEqual({ value: 'keep', label: 'Keep current expiry' })
    expect(defaultAutopilotExpiry({ expiresAt: null }, now)).toBe('none')
    expect(defaultAutopilotExpiry({ expiresAt: '2026-10-25T00:00:00.000Z' }, now)).toBe('keep')
  })

  it('does not offer "keep" once the expiry has passed, and starts on a new short expiry instead', () => {
    const now = Date.parse('2026-10-24T12:00:00.000Z')
    const lapsed = { expiresAt: '2026-10-24T11:00:00.000Z' }
    expect(isExpiryLapsed(lapsed.expiresAt, now)).toBe(true)
    expect(isExpiryLapsed(null, now)).toBe(false)
    expect(autopilotExpiryOptions(lapsed, now).map(o => o.value)).toEqual(['none', '1h', '8h', '24h', '7d'])
    expect(defaultAutopilotExpiry(lapsed, now)).toBe('24h')
    expect(effectiveExpiryChoice('keep', lapsed, now)).toBe('24h')
    expect(effectiveExpiryChoice('keep', { expiresAt: '2026-10-25T00:00:00.000Z' }, now)).toBe('keep')
    expect(effectiveExpiryChoice('8h', lapsed, now)).toBe('8h')
  })

  it('words the chip, the composer hint and the auto-card note', () => {
    const rows = [parseAutopilotRow(ROW)!, parseAutopilotRow(IDLE_ROW)!]
    expect(autopilotChipText(rows)).toBe('Autopilot on · 1 workspace')
    expect(autopilotChipText([rows[0]!, { ...rows[0]!, workspaceId: 'w3' }])).toBe('Autopilot on · 2 workspaces')
    // Enabled but not active (expired, or the enabler is no longer owner) is not "on".
    expect(autopilotChipText([{ ...rows[0]!, active: false }])).toBeUndefined()
    expect(autopilotChipText([])).toBeUndefined()
    expect(AUTOPILOT_COMPOSER_HINT).toContain('acts for you at once')
    expect(AUTOPILOT_COMPOSER_HINT).toContain('wait for Confirm')
    expect(autoCardNote('Ann')).toBe('Done automatically on behalf of Ann.')
    expect(autoCardNote(undefined)).toBe('Done automatically by Autopilot.')
  })
})

describe('autopilot reducer', () => {
  it('starts empty, read-only and supported', () => {
    expect(initialHubState().autopilot).toEqual({ phase: 'idle', canEdit: false, rows: [], unsupported: false })
  })

  it('a view replaces every row and clears errors; a row updates or appends one', () => {
    let state = reduceHub(initialHubState(), { type: 'autopilot', patch: { phase: 'error', error: 'boom' } })
    state = reduceHub(state, { type: 'autopilot', patch: { saveError: 'nope' } })
    state = reduceHub(state, { type: 'autopilotView', view: parseAutopilotView(VIEW)! })
    expect(state.autopilot).toMatchObject({ phase: 'ready', canEdit: true })
    expect(state.autopilot.error).toBeUndefined()
    expect(state.autopilot.saveError).toBeUndefined()
    expect(state.autopilot.rows).toHaveLength(2)

    state = reduceHub(state, { type: 'autopilotRow', row: parseAutopilotRow({ ...ROW, active: false, enabled: false })! })
    expect(state.autopilot.rows.map(r => [r.workspaceId, r.active])).toEqual([['w1', false], ['w2', false]])
    expect(activeAutopilotRows(state)).toEqual([])

    state = reduceHub(state, { type: 'autopilotRow', row: parseAutopilotRow({ ...ROW, workspace_id: 'w9' })! })
    expect(state.autopilot.rows.map(r => r.workspaceId)).toEqual(['w1', 'w2', 'w9'])
    expect(activeAutopilotRows(state).map(r => r.workspaceId)).toEqual(['w9'])
  })

  it('tracks which workspace is saving and ends the save explicitly', () => {
    let state = reduceHub(initialHubState(), { type: 'autopilot', patch: { saveError: 'old' } })
    state = reduceHub(state, { type: 'autopilot', patch: { savingId: 'w1' } })
    expect(state.autopilot.savingId).toBe('w1')
    expect(state.autopilot.saveError).toBeUndefined()
    state = reduceHub(state, { type: 'autopilot', patch: { savingId: undefined } })
    expect('savingId' in state.autopilot).toBe(false)
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

describe('autopilot controller', () => {
  it('loads the rows from supervisor.autopilot.get', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.get', () => VIEW)
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    await controller.loadAutopilot()
    expect(h.hub.all('supervisor.autopilot.get')[0]?.params).toEqual({})
    expect(store.getSnapshot().autopilot).toMatchObject({ phase: 'ready', canEdit: true, unsupported: false })
    expect(store.getSnapshot().autopilot.rows.map(row => row.workspaceId)).toEqual(['w1', 'w2'])
  })

  it('opening the supervisor thread also loads Autopilot', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.thread.open', () => ({ thread_id: 't1', messages: [], confirmations: [] }))
    h.hub.handlers.set('supervisor.config.get', () => ({ status: 'unconfigured', can_edit: true }))
    h.hub.handlers.set('supervisor.autopilot.get', () => ({ can_edit: false, workspaces: [ROW] }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    await controller.supervisorOpen()
    await settle()
    expect(h.hub.all('supervisor.autopilot.get')).toHaveLength(1)
    expect(store.getSnapshot().autopilot).toMatchObject({ phase: 'ready', canEdit: false })
  })

  it('saves with the exact wire arguments and applies the returned row', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.get', () => VIEW)
    h.hub.handlers.set('supervisor.autopilot.set', () => ({ workspace: { ...IDLE_ROW, enabled: true, active: true, instruction: 'Be bold', expires_at: '2026-10-24T13:00:00.000Z', enabled_by: { id: 'u1', name: 'Ann' }, updated_at: '2026-10-24T12:00:00.000Z' } }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    await controller.loadAutopilot()
    const saved = await controller.setAutopilot({ workspaceId: 'w2', enabled: true, instruction: 'Be bold', expiresAt: '2026-10-24T13:00:00.000Z' })
    expect(saved).toBe(true)
    expect(h.hub.all('supervisor.autopilot.set')[0]?.params).toEqual({ workspace_id: 'w2', enabled: true, instruction: 'Be bold', expires_at: '2026-10-24T13:00:00.000Z' })
    const row = store.getSnapshot().autopilot.rows.find(r => r.workspaceId === 'w2')
    expect(row).toMatchObject({ enabled: true, active: true, instruction: 'Be bold' })
    expect('savingId' in store.getSnapshot().autopilot).toBe(false)
  })

  it('omits instruction and expires_at when they are not given, and sends null to clear the expiry', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.set', () => ({ workspace: ROW }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    await controller.setAutopilot({ workspaceId: 'w1', enabled: false })
    await controller.setAutopilot({ workspaceId: 'w1', enabled: true, expiresAt: null })
    const frames = h.hub.all('supervisor.autopilot.set')
    expect(frames[0]?.params).toEqual({ workspace_id: 'w1', enabled: false })
    expect(frames[1]?.params).toEqual({ workspace_id: 'w1', enabled: true, expires_at: null })
  })

  it('keeps a refused save visible and never reports success', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.set', () => ({ __error: { code: 'forbidden', message: 'Only the Hub owner can do this.' } }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    expect(await controller.setAutopilot({ workspaceId: 'w1', enabled: true })).toBe(false)
    expect(store.getSnapshot().autopilot.saveError).toBe('Only the Hub owner can do this.')
    expect('savingId' in store.getSnapshot().autopilot).toBe(false)
  })

  it('refuses an over-long instruction locally and sends nothing', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.set', () => ({ workspace: ROW }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    expect(await controller.setAutopilot({ workspaceId: 'w1', enabled: true, instruction: 'x'.repeat(2001) })).toBe(false)
    expect(h.hub.all('supervisor.autopilot.set')).toHaveLength(0)
    expect(store.getSnapshot().autopilot.saveError).toContain('2000')
  })

  it('does not trust an unreadable save answer: it reads back what the Hub holds', async () => {
    const h = liveHarness()
    h.hub.handlers.set('supervisor.autopilot.set', () => ({ workspace: { workspace_id: 'w1' } }))
    h.hub.handlers.set('supervisor.autopilot.get', () => VIEW)
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    expect(await controller.setAutopilot({ workspaceId: 'w1', enabled: true })).toBe(false)
    await settle()
    expect(h.hub.all('supervisor.autopilot.get')).toHaveLength(1)
    expect(store.getSnapshot().autopilot.rows).toHaveLength(2)
  })

  it('hides the control on a Hub that does not know Autopilot, and shows a real error otherwise', async () => {
    const old = liveHarness()
    old.hub.handlers.set('supervisor.autopilot.get', () => ({ __error: { code: 'invalid_request', message: 'unknown hub method' } }))
    const oldStore = createHubStore()
    const oldController = createHubController({ connection: old.conn, store: oldStore })
    await goLive(old)
    oldController.start()
    await oldController.loadAutopilot()
    expect(oldStore.getSnapshot().autopilot).toMatchObject({ phase: 'ready', unsupported: true, rows: [] })

    const broken = liveHarness()
    broken.hub.handlers.set('supervisor.autopilot.get', () => ({ can_edit: true, workspaces: [{ workspace_id: 'w1' }] }))
    const brokenStore = createHubStore()
    const brokenController = createHubController({ connection: broken.conn, store: brokenStore })
    await goLive(broken)
    brokenController.start()
    await brokenController.loadAutopilot()
    expect(brokenStore.getSnapshot().autopilot.phase).toBe('error')
    expect(brokenStore.getSnapshot().autopilot.rows).toEqual([])
  })

  it('updates one row on supervisor.autopilot_changed without asking again', () => {
    const fake = fakeConnection()
    const store = createHubStore()
    createHubController({ connection: fake.connection, store }).start()
    store.dispatch({ type: 'autopilotView', view: parseAutopilotView(VIEW)! })
    fake.notice('supervisor.autopilot_changed', { workspace: { ...ROW, active: false, enabled: false } })
    expect(store.getSnapshot().autopilot.rows.map(r => [r.workspaceId, r.active])).toEqual([['w1', false], ['w2', false]])
    expect(store.getSnapshot().autopilot.canEdit).toBe(true)
    expect(fake.requests).toEqual([])
  })

  it('asks the Hub again when a supervisor.autopilot_changed payload is unreadable', () => {
    const fake = fakeConnection()
    const store = createHubStore()
    createHubController({ connection: fake.connection, store }).start()
    fake.notice('supervisor.autopilot_changed', { workspace: { workspace_id: 'w1', active: 'maybe' } })
    expect(fake.requests).toEqual(['supervisor.autopilot.get'])
  })

  it('shows an Autopilot card as automatic from a thread event, without a pending state', () => {
    const fake = fakeConnection()
    const store = createHubStore()
    createHubController({ connection: fake.connection, store }).start()
    fake.notice('supervisor.thread_event', { confirmation: { id: 'c1', tool: 'task.steer', summary: 'Steer', state: 'confirmed', auto: true, confirmed_by: 'Ann' } })
    expect(store.getSnapshot().supervisor.confirmations[0]).toMatchObject({ id: 'c1', auto: true, confirmedBy: 'Ann', state: 'confirmed' })
  })
})
