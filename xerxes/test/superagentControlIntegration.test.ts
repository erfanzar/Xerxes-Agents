// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { join } from 'node:path'
import { createHubSupervisor } from '../src/hub/supervisor/index.js'
import { disposeAll, registerCleanup, tempDir } from './hubHarness.js'
import { makeWorld, scriptedModel } from './hubSupervisorHarness.js'

afterEach(disposeAll)

test('production composition exposes human-only durable controls and pauses all model/task admission', async () => {
  const world = makeWorld()
  const dir = await tempDir('superagent-control-')
  const fake = scriptedModel([{ text: 'ok', toolCalls: [], usage: { inputTokens: 2, outputTokens: 1 } }])
  const options = { identity: world.identity, overlay: world.overlay, workspaces: world.workspaces, backend: world.backend, clock: world.clock, databasePath: join(dir, 'hub.sqlite'), model: fake.model, persona: 'Offline test',
    execute: async () => ({ ok: true as const, command_id: 'test', state: 'accepted' as const, result: {} }),
    isHubOwner: (id: string) => id === 'owner', autopilot: { checkInMs: 0 } }
  const supervisor = createHubSupervisor(options)
  registerCleanup(() => supervisor.close())
  const owner = { context: world.identity.context('owner'), grants: world.identity.grantsFor('owner') }
  const other = { context: world.identity.context('alice'), grants: world.identity.grantsFor('alice') }
  const update = supervisor.handlers['supervisor.control.set']
  const get = supervisor.handlers['supervisor.control.get']
  expect(typeof update).toBe('function'); expect(typeof get).toBe('function')
  if (!update || !get) throw new Error('Control handlers absent')
  await expect(Promise.resolve().then(() => update(other, { revision: 0, paused: true }))).rejects.toThrow()
  await update(owner, { revision: 0, paused: true })
  const thread = supervisor.threads.privateFor(world.user('owner'))
  await expect(Promise.resolve().then(() => supervisor.handlers['supervisor.message.submit']!(owner, { thread_id: thread.id, message_id: 'paused-message', text: 'act' }))).rejects.toThrow()
  const paused = await supervisor.runner.enqueue({ threadId: thread.id, requester: world.user('owner') })
  expect(paused.stop).not.toBe('completed'); expect(fake.requests).toHaveLength(0)
  await update(owner, { revision: 1, paused: false, limits: { modelCalls: 1 } })
  await supervisor.runner.enqueue({ threadId: thread.id, requester: world.user('owner') })
  expect(fake.requests).toHaveLength(1)
  await supervisor.runner.enqueue({ threadId: thread.id, requester: world.user('owner') })
  expect(fake.requests).toHaveLength(1)
  await update(owner, { revision: 2, killed: true })
  expect(await get(owner, {})).toMatchObject({ control: { paused: true, killed: true }, usage: { modelCalls: 1 } })
  supervisor.close()
  const reopened = createHubSupervisor(options)
  registerCleanup(() => reopened.close())
  expect(await reopened.handlers['supervisor.control.get']!(owner, {})).toMatchObject({ control: { killed: true } })
})

test('global pause prevents pending action dispatch rather than merely stopping new model calls', async () => {
  const world = makeWorld(); const dir = await tempDir('superagent-dispatch-')
  let sent = 0
  const supervisor = createHubSupervisor({ identity: world.identity, overlay: world.overlay, workspaces: world.workspaces, backend: world.backend, clock: world.clock, databasePath: join(dir, 'hub.sqlite'), persona: 'Offline test',
    model: scriptedModel([{ text: '', toolCalls: [] }]).model,
    execute: async () => { sent += 1; return { ok: true, command_id: 'c', state: 'accepted', result: {} } },
    isHubOwner: id => id === 'owner', autopilot: { checkInMs: 0 } })
  registerCleanup(() => supervisor.close())
  const thread = supervisor.threads.ensureShared()
  supervisor.threads.access(thread.id, world.user('owner'))
  const action = await supervisor.actions.createIntent({ threadId: thread.id, requester: world.user('alice'), reader: world.reader('alice'),
    tool: 'task.steer', args: { workspace_id: 'w1', task_id: 's1', text: 'test' } })
  const ctx = { context: world.identity.context('owner'), grants: world.identity.grantsFor('owner') }
  const update = supervisor.handlers['supervisor.control.set']
  expect(typeof update).toBe('function')
  if (!update) throw new Error('Control handler absent')
  await update(ctx, { revision: 0, paused: true })
  await supervisor.actions.respond(ctx.context, { action_id: action.id, decision: 'confirm' })
  expect(sent).toBe(0)
})
