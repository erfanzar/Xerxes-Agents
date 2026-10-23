// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import type { HubConnection } from '../../client/hubConnection.js'
import { UNCONFIGURED_PROMPT, createHubController } from '../../client/hubController.js'
import { parseMachines, parseSupervisorConfig, parseSupervisorOptions } from '../../client/hubParsers.js'
import { createHubStore, initialHubState, reduceHub } from '../../client/hubStore.js'
import { goLive, harness, settle } from './fakes.js'

const CONFIGURED = {
  status: 'configured',
  can_edit: true,
  config: { machine_id: 'm1', machine_label: 'Mac', profile: 'work', model: 'sonnet', effort: 'high' },
}

describe('machine and supervisor config parsers', () => {
  it('reads machines, drops unreadable rows and repeated ids, and defaults safely', () => {
    const machines = parseMachines({
      machines: [
        { id: 'm1', label: 'Mac', health: 'online', read_only: false, models_supported: true, workspace_ids: ['w1', 7, '', 'w2'] },
        { id: 'm1', label: 'duplicate' },
        { label: 'no id' },
        'junk',
        { id: 'm2', health: 'bogus' },
      ],
    })
    expect(machines.map(m => m.id)).toEqual(['m1', 'm2'])
    expect(machines[0]).toMatchObject({ label: 'Mac', health: 'online', readOnly: false, modelsSupported: true, workspaceIds: ['w1', 'w2'] })
    // Unknown health is stale and a missing flag is the restrictive value.
    expect(machines[1]).toMatchObject({ label: 'm2', health: 'stale', readOnly: false, modelsSupported: false, workspaceIds: [] })
    expect(parseMachines({ machines: 'nope' })).toEqual([])
  })

  it('reads a configured answer and keeps the effort null when none is set', () => {
    expect(parseSupervisorConfig(CONFIGURED)).toEqual({
      status: 'configured',
      canEdit: true,
      config: { machineId: 'm1', machineLabel: 'Mac', profile: 'work', model: 'sonnet', effort: 'high' },
    })
    const noEffort = parseSupervisorConfig({ ...CONFIGURED, config: { ...CONFIGURED.config, effort: null } })
    expect(noEffort.config?.effort).toBeNull()
  })

  it('treats an unknown status as invalid, never grants editing by default, and ignores a half-written config', () => {
    expect(parseSupervisorConfig({ status: 'weird', can_edit: 'yes' })).toEqual({ status: 'invalid', canEdit: false })
    expect(parseSupervisorConfig(undefined)).toEqual({ status: 'invalid', canEdit: false })
    const partial = parseSupervisorConfig({ status: 'configured', config: { machine_id: 'm1', profile: 'work' }, message: 'm' })
    expect(partial.config).toBeUndefined()
    expect(partial.message).toBe('m')
  })

  it('reads options, dedupes, accepts plain-string efforts and rejects an answer without a machine', () => {
    const options = parseSupervisorOptions({
      machine_id: 'm1',
      profiles: [{ name: 'work', label: 'Work', provider: 'anthropic', default_model: 'sonnet' }, { name: 'work' }, { label: 'no name' }],
      profile: 'work',
      models: [{ id: 'sonnet', label: 'Sonnet' }, { id: 'sonnet' }, { id: 'opus' }, 3],
      model: 'sonnet',
      efforts: [{ effort: 'low', description: 'Fast' }, 'high', { effort: 'low' }, {}],
      default_effort: 'low',
    })
    expect(options?.profiles).toEqual([{ name: 'work', label: 'Work', provider: 'anthropic', defaultModel: 'sonnet' }])
    expect(options?.models).toEqual([{ id: 'sonnet', label: 'Sonnet' }, { id: 'opus', label: 'opus' }])
    expect(options?.efforts).toEqual([{ effort: 'low', description: 'Fast' }, { effort: 'high' }])
    expect(options).toMatchObject({ machineId: 'm1', profile: 'work', model: 'sonnet', defaultEffort: 'low' })
    expect(parseSupervisorOptions({ profiles: [] })).toBeUndefined()
    expect(parseSupervisorOptions({ machine_id: 'm1' })).toMatchObject({ profiles: [], models: [], efforts: [], defaultEffort: null })
  })
})

describe('supervisor config reducer', () => {
  it('applies a config_changed view and clears stale errors', () => {
    let state = reduceHub(initialHubState(), { type: 'supervisorConfig', patch: { phase: 'error', error: 'boom' } })
    state = reduceHub(state, { type: 'supervisorConfig', patch: { saveError: 'nope' } })
    state = reduceHub(state, { type: 'supervisorConfigView', view: parseSupervisorConfig(CONFIGURED) })
    expect(state.supervisorConfig).toMatchObject({ phase: 'ready', status: 'configured', canEdit: true, saving: false })
    expect(state.supervisorConfig.config?.model).toBe('sonnet')
    expect(state.supervisorConfig.error).toBeUndefined()
    expect(state.supervisorConfig.saveError).toBeUndefined()
    // A later "unconfigured" answer removes the old choice.
    state = reduceHub(state, { type: 'supervisorConfigView', view: parseSupervisorConfig({ status: 'unconfigured', can_edit: false }) })
    expect(state.supervisorConfig.status).toBe('unconfigured')
    expect(state.supervisorConfig.config).toBeUndefined()
    expect(state.supervisorConfig.canEdit).toBe(false)
  })

  it('keeps option lists when only the loading state changes', () => {
    const options = parseSupervisorOptions({ machine_id: 'm1', profiles: [{ name: 'work' }] })!
    let state = reduceHub(initialHubState(), { type: 'supervisorConfigOptions', state: { phase: 'ready', options } })
    state = reduceHub(state, { type: 'supervisorConfig', patch: { saving: true } })
    expect(state.supervisorConfig.options.options).toBe(options)
    expect(state.supervisorConfig.saving).toBe(true)
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

describe('supervisor config controller', () => {
  it('handles the supervisor.config_changed event without asking again', () => {
    const fake = fakeConnection()
    const store = createHubStore()
    createHubController({ connection: fake.connection, store }).start()
    fake.notice('supervisor.config_changed', CONFIGURED)
    expect(store.getSnapshot().supervisorConfig).toMatchObject({ status: 'configured', canEdit: true })
    expect(store.getSnapshot().supervisorConfig.config?.machineLabel).toBe('Mac')
    expect(fake.requests).toEqual([])
  })

  it('XERXES stays "working" from the Hub\'s activity event until it says idle or the reply lands', () => {
    const fake = fakeConnection()
    const store = createHubStore()
    createHubController({ connection: fake.connection, store }).start()
    expect(store.getSnapshot().supervisor.working).toBeFalsy()
    fake.notice('supervisor.thread_event', { activity: 'working' })
    expect(store.getSnapshot().supervisor.working).toBe(true)
    // A person's own message echoed back does not end the turn.
    fake.notice('supervisor.thread_event', { message: { id: 'u1', role: 'user', text: 'hi', at: new Date().toISOString() } })
    expect(store.getSnapshot().supervisor.working).toBe(true)
    fake.notice('supervisor.thread_event', { message: { id: 'x1', role: 'xerxes', text: 'hello', at: new Date().toISOString() } })
    expect(store.getSnapshot().supervisor.working).toBe(false)
    fake.notice('supervisor.thread_event', { activity: 'working' })
    fake.notice('supervisor.thread_event', { activity: 'idle' })
    expect(store.getSnapshot().supervisor.working).toBe(false)
    // A failed turn from a Hub that sends no `idle` still ends the spinner.
    fake.notice('supervisor.thread_event', { activity: 'working' })
    fake.notice('supervisor.thread_event', { message: { id: 's1', role: 'system', text: 'XERXES stopped: provider failed', at: new Date().toISOString() } })
    expect(store.getSnapshot().supervisor.working).toBe(false)
  })

  it('asks the Hub again when a config_changed payload is unreadable', () => {
    const fake = fakeConnection()
    const store = createHubStore()
    createHubController({ connection: fake.connection, store }).start()
    fake.notice('supervisor.config_changed', { status: 'weird' })
    expect(fake.requests).toEqual(['supervisor.config.get'])
  })

  it('loads machines, config and options and saves a choice with the exact wire arguments', async () => {
    const h = harness()
    h.hub.handlers.set('hub.workspaces.list', () => ({ workspaces: [] }))
    h.hub.handlers.set('hub.machines.list', () => ({ machines: [{ id: 'm1', label: 'Mac', health: 'online', models_supported: true, workspace_ids: ['w1'] }] }))
    h.hub.handlers.set('supervisor.config.get', () => ({ status: 'unconfigured', can_edit: true }))
    h.hub.handlers.set('supervisor.config.options', () => ({ machine_id: 'm1', profiles: [{ name: 'work', default_model: 'sonnet' }], models: [{ id: 'sonnet' }], efforts: [{ effort: 'high' }], default_effort: null }))
    h.hub.handlers.set('supervisor.config.set', () => CONFIGURED)
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    controller.start()
    await settle()
    h.hub.socket(0).open()
    await settle()

    expect(store.getSnapshot().machines).toMatchObject({ phase: 'ready', list: [{ id: 'm1', label: 'Mac' }] })
    await controller.supervisorConfigGet()
    expect(store.getSnapshot().supervisorConfig).toMatchObject({ status: 'unconfigured', canEdit: true, phase: 'ready' })

    await controller.supervisorConfigOptions({ machineId: 'm1', profile: 'work' })
    expect(h.hub.all('supervisor.config.options')[0]?.params).toEqual({ machine_id: 'm1', profile: 'work' })
    expect(store.getSnapshot().supervisorConfig.options.options?.profiles[0]?.name).toBe('work')

    expect(await controller.supervisorConfigSet({ machineId: 'm1', profile: 'work', model: 'sonnet', effort: 'high' })).toBe(true)
    expect(h.hub.all('supervisor.config.set')[0]?.params).toEqual({ machine_id: 'm1', profile: 'work', model: 'sonnet', effort: 'high' })
    expect(store.getSnapshot().supervisorConfig).toMatchObject({ status: 'configured', saving: false })
  })

  it('keeps a refused save visible and never reports success', async () => {
    const h = harness()
    h.hub.handlers.set('hub.workspaces.list', () => ({ workspaces: [] }))
    h.hub.handlers.set('hub.machines.list', () => ({ machines: [] }))
    h.hub.handlers.set('supervisor.config.set', () => ({ __error: { code: 'forbidden', message: 'Only the owner can do this.' } }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    expect(await controller.supervisorConfigSet({ machineId: 'm1', profile: 'work', model: 'sonnet', effort: null })).toBe(false)
    expect(store.getSnapshot().supervisorConfig).toMatchObject({ saving: false, saveError: 'Only the owner can do this.' })
  })

  it('treats an older Hub without these methods as "ungrouped, nothing to edit"', async () => {
    const h = harness()
    h.hub.handlers.set('hub.workspaces.list', () => ({ workspaces: [] }))
    h.hub.handlers.set('hub.machines.list', () => ({ __error: { code: 'invalid_request', message: 'unknown hub method' } }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    await controller.machinesList()
    expect(store.getSnapshot().machines).toEqual({ phase: 'ready', list: [] })
  })

  it("shows an 'unconfigured' submit error as a prompt to choose a model", async () => {
    const h = harness()
    h.hub.handlers.set('hub.workspaces.list', () => ({ workspaces: [] }))
    h.hub.handlers.set('hub.machines.list', () => ({ machines: [] }))
    h.hub.handlers.set('supervisor.thread.open', () => ({ thread_id: 't1', messages: [], confirmations: [], config_status: 'unconfigured' }))
    h.hub.handlers.set('supervisor.config.get', () => ({ status: 'unconfigured', can_edit: true }))
    h.hub.handlers.set('supervisor.message.submit', () => ({ __error: { code: 'unconfigured', message: "Choose XERXES's model in XERXES settings" } }))
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store })
    await goLive(h)
    controller.start()
    await controller.supervisorOpen()
    await settle()
    expect(store.getSnapshot().supervisorConfig.status).toBe('unconfigured')
    expect(await controller.supervisorSend('hello')).toBe(false)
    const { supervisor, supervisorConfig } = store.getSnapshot()
    expect(supervisor.error).toBe(UNCONFIGURED_PROMPT)
    expect(supervisor.busy).toBe(false)
    expect(supervisorConfig.status).toBe('unconfigured')
  })
})
