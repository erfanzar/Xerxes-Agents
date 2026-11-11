// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { join } from 'node:path'
import { createHubSupervisor } from '../src/hub/supervisor/index.js'
import { SuperagentControl } from '../src/hub/supervisor/abilities/control.js'
import { createModelDigestSummarizer } from '../src/hub/supervisor/runner.js'
import type { HubCommandResult } from '../src/hub/protocol.js'
import type { JsonObject } from '../src/types/toolCalls.js'
import { makeWorld, scriptedModel, toolCall } from './hubSupervisorHarness.js'
import { disposeAll, registerCleanup, tempDir } from './hubHarness.js'

afterEach(disposeAll)
function deferred() {
  let resolve = (): void => {}
  const promise = new Promise<void>(done => { resolve = done })
  return { promise, resolve }
}
async function fixture(tool: string, args: JsonObject) {
  const world = makeWorld(); const dir = await tempDir('superagent-races-')
  const ownerState = { active: true }
  const sent: Readonly<Record<string, unknown>>[] = []
  const behavior: { run: (params: Readonly<Record<string, unknown>>) => Promise<HubCommandResult> } = {
    run: async () => ({ ok: true, command_id: 'c', state: 'accepted', result: { session_id: 'fresh' } }) }
  const fake = scriptedModel([{ text: '', toolCalls: [toolCall(tool, args)] }, { text: 'finished', toolCalls: [] }])
  const supervisor = createHubSupervisor({ identity: world.identity, overlay: world.overlay, workspaces: world.workspaces, backend: world.backend,
    clock: world.clock, databasePath: join(dir, 'hub.sqlite'), model: fake.model, persona: 'Offline fixture',
    isHubOwner: id => id === 'owner' && ownerState.active, autopilot: { checkInMs: 0 },
    events: { watermark: () => 1, read: after => ({ throughSequence: 1, hasMore: false, events: after < 1 ? [{
      event_id: 'event-1', cursor: 'fixture-cursor', workspace_id: 'w1', session_id: 's1', type: 'question_request', occurred_at: new Date(world.time.now).toISOString(),
      payload: { id: 'q1', questions: [{ id: 'answer', question: 'Build?', options: ['nightly'] }] }, sequence: 1,
    }] : [] }) },
    execute: async (_context, params) => { sent.push(params); return behavior.run(params) } })
  registerCleanup(() => supervisor.close())
  const owner = world.user('owner'); const thread = supervisor.threads.privateFor(owner)
  const context = { context: world.identity.context('owner'), grants: world.identity.grantsFor('owner') }
  const update = (params: Record<string, unknown>) => supervisor.handlers['supervisor.control.set']!(context, params)
  const run = () => supervisor.runner.enqueue({ threadId: thread.id, requester: owner })
  return { ...world, supervisor, ownerState, sent, behavior, update, run }
}
const question: JsonObject = { workspace_id: 'w1', task_id: 's1', request_id: 'q1', answers: { answer: 'nightly' } }

test('question answers obey the durable daily action budget', async () => {
  const world = await fixture('answer_question', question)
  await world.update({ revision: 0, limits: { actions: 0 } })
  await world.run()
  expect(world.sent).toHaveLength(0)
})

test('pause, kill and shutdown during question audit prevent the answer dispatch', async () => {
  for (const change of ['pause', 'kill', 'close']) {
    const world = await fixture('answer_question', question)
    const ready = deferred(); const finish = deferred()
    world.identity.audit = async record => { if (record.action === 'supervisor.dispatch:question.answer') { ready.resolve(); await finish.promise } }
    const running = world.run(); await ready.promise
    if (change === 'close') world.supervisor.close()
    else await world.update({ revision: 0, ...(change === 'kill' ? { killed: true } : { paused: true }) })
    finish.resolve(); await running
    expect(world.sent).toHaveLength(0)
  }
})

test('task.start cannot submit a new turn after kill or shutdown during session.open', async () => {
  for (const close of [false, true]) {
    const world = await fixture('task_start', { workspace_id: 'w1', text: 'work' })
    const ready = deferred(); const finish = deferred()
    world.behavior.run = async params => {
      if (params.action === 'session.open') { ready.resolve(); await finish.promise }
      return { ok: true, command_id: 'c', state: 'accepted', result: { session_id: 'fresh' } }
    }
    const running = world.run(); await ready.promise
    if (close) world.supervisor.close()
    else await world.update({ revision: 0, killed: true })
    finish.resolve(); await running
    expect(world.sent.map(row => row.action)).toEqual(['session.open'])
  }
})

test('a captured owner flag cannot authorize an auto action after live owner revocation', async () => {
  const world = await fixture('task_steer', { workspace_id: 'w1', task_id: 's1', text: 'work' })
  const ready = deferred(); const finish = deferred()
  world.identity.audit = async record => { if (record.action === 'supervisor.dispatch:task.steer') { ready.resolve(); await finish.promise } }
  const running = world.run(); await ready.promise
  world.ownerState.active = false
  finish.resolve(); await running
  expect(world.sent).toHaveLength(0)
})

test('re-enabling after kill cannot resurrect an old task.start continuation', async () => {
  const world = await fixture('task_start', { workspace_id: 'w1', text: 'work' })
  const ready = deferred(); const finish = deferred()
  world.behavior.run = async params => {
    if (params.action === 'session.open') { ready.resolve(); await finish.promise }
    return { ok: true, command_id: 'c', state: 'accepted', result: { session_id: 'fresh' } }
  }
  const running = world.run(); await ready.promise
  await world.update({ revision: 0, killed: true })
  await world.update({ revision: 1, killed: false, paused: false })
  finish.resolve(); await running
  expect(world.sent.map(row => row.action)).toEqual(['session.open'])
})

test('new task submission rechecks explicit denials on the actual created session', async () => {
  const world = await fixture('task_start', { workspace_id: 'w1', text: 'work' })
  const ready = deferred(); const finish = deferred()
  world.behavior.run = async params => {
    if (params.action === 'session.open') { ready.resolve(); await finish.promise }
    return { ok: true, command_id: 'c', state: 'accepted', result: { session_id: 'fresh' } }
  }
  const running = world.run(); await ready.promise
  world.identity.grantsFor = id => id !== 'owner' ? [] : [{ principalId: id,
    resource: { type: 'session', workspaceId: 'w1', sessionId: 'fresh' }, effect: 'deny', scopes: ['session.submit'] }]
  finish.resolve(); await running
  expect(world.sent.map(row => row.action)).toEqual(['session.open'])
})

test('re-enabling after kill cannot revive a question waiting in its dispatch audit', async () => {
  const world = await fixture('answer_question', question)
  const ready = deferred(); const finish = deferred()
  world.identity.audit = async record => { if (record.action === 'supervisor.dispatch:question.answer') { ready.resolve(); await finish.promise } }
  const running = world.run(); await ready.promise
  await world.update({ revision: 0, killed: true })
  await world.update({ revision: 1, killed: false, paused: false })
  finish.resolve(); await running
  expect(world.sent).toHaveLength(0)
})

test('controlled model metadata and digest versions follow a live model hot-swap', () => {
  const world = makeWorld(); let id = 'first'; let priced = false
  const control = new SuperagentControl({ db: world.db, clock: world.clock, isOwner: () => true, audit: async () => {} })
  const port = control.model({ get id() { return id }, get reportsCost() { return priced }, async complete() { return { text: '', toolCalls: [] } } })
  const digest = createModelDigestSummarizer(port)
  const before = digest.version
  id = 'second'; priced = true
  expect(port.id).toBe('second'); expect(port.reportsCost).toBe(true)
  expect(digest.version).not.toBe(before)
})
