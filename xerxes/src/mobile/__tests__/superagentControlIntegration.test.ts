// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, it } from 'vitest'
import { createHubController } from '../../client/hubController.js'
import { createHubStore } from '../../client/hubStore.js'
import { goLive, harness, settle } from './fakes.js'

it('loads owner authority and controls even when opening the conversational thread fails', async () => {
  const h = harness()
  h.hub.handlers.set('hub.workspaces.list', () => ({ workspaces: [] }))
  h.hub.handlers.set('hub.machines.list', () => ({ machines: [] }))
  h.hub.handlers.set('supervisor.thread.open', () => ({ __error: { code: 'unavailable', message: 'thread unavailable' } }))
  h.hub.handlers.set('supervisor.autopilot.get', () => ({ can_edit: true, workspaces: [] }))
  h.hub.handlers.set('supervisor.watch.list', () => ({ can_edit: true, watches: [] }))
  const reply = { control: { revision: 1, paused: false, killed: false, limits: {} }, usage: { actions: 0, wakes: 0, modelCalls: 0, tokens: null, usd: null } }
  h.hub.handlers.set('supervisor.control.get', () => reply)
  h.hub.handlers.set('supervisor.control.set', () => ({ ...reply, control: { ...reply.control, revision: 2, paused: true } }))
  const store = createHubStore()
  const controller = createHubController({ connection: h.conn, store })
  try {
    await goLive(h); controller.start(); await controller.supervisorOpen(); await settle()
    expect(store.getSnapshot().supervisor.phase).toBe('error')
    expect(store.getSnapshot().autopilot.canEdit).toBe(true)
    await controller.superagentControl.load()
    expect(await controller.superagentControl.save({ paused: true })).toBe(true)
    expect(h.hub.all('supervisor.control.set')[0]?.params).toEqual({ revision: 1, paused: true })
    controller.dispose(); await controller.superagentControl.load()
    expect(h.hub.all('supervisor.control.get')).toHaveLength(1)
  } finally { controller.dispose() }
})
