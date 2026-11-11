// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { ActionService, actionToWire, type ActionRecord } from '../src/hub/supervisor/actions.js'
import { disposeAll } from './hubHarness.js'
import { makeWorld, toolCall } from './hubSupervisorHarness.js'
import { SuperagentAbilities } from '../src/hub/supervisor/abilities/service.js'
import { createSupervisorToolRegistry, TURN_METADATA_KEY } from '../src/hub/supervisor/tools.js'

afterEach(disposeAll)

function fixture() {
  const world = makeWorld()
  const owner = world.user('owner')
  const thread = world.threads.privateFor(owner)
  const state = { owner: true, revision: 'remote-ref-1', fail: false }
  const executed: ActionRecord[] = []
  const service = new ActionService({ ...world, execute: async () => { throw new Error('Risk action must not use generic task routing') },
    access: (id, principal) => { world.threads.access(id, principal) },
    autopilot: () => ({ owner, wake: false }),
    risk: { isOwner: id => id === owner.id && state.owner,
      revision: () => state.revision,
      execute: async action => { executed.push(action); if (state.fail) throw new Error('ack lost'); return { applied: true } } } })
  const args = { workspace_id: 'w1', operation: 'git.push', target: 'origin/main', expected_revision: 'remote-ref-1', details: 'Push only reviewed commit abc123; no force' }
  const propose = () => service.createIntent({ threadId: thread.id, tool: 'risk.execute', args, requester: owner, reader: world.reader('owner', thread.id),
    turn: { trigger: 'autopilot', othersInContext: false } })
  const confirm = (id: string, who = 'owner') => service.respond(world.identity.context(who), { action_id: id, thread_id: thread.id, decision: 'confirm' })
  return { ...world, service, state, executed, args, propose, confirm, ownerThread: thread }
}

test('risky actions always wait for an exact owner confirmation even with Autopilot active', async () => {
  const world = fixture()
  const action = await world.propose()
  expect(action.state).toBe('pending'); expect(action.auto).toBe(false); expect(world.executed).toHaveLength(0)
  const wire = actionToWire(action)
  expect(String(wire.summary)).toContain(world.args.details)
  expect(String(wire.summary)).toContain(world.args.expected_revision)
  await expect(world.confirm(action.id, 'alice')).rejects.toThrow()
  expect(world.executed).toHaveLength(0)
  const result = await world.confirm(action.id)
  expect(result.state).toBe('accepted'); expect(result.auto).toBe(false)
  expect(world.executed).toHaveLength(1)
  expect(world.executed[0]?.args).toEqual(world.args)
  await world.confirm(action.id)
  expect(world.executed).toHaveLength(1)
})

test('risky confirmations reject target drift, ownership revocation and corrupted exact arguments', async () => {
  for (const change of ['target', 'owner', 'args']) {
    const world = fixture()
    const action = await world.propose()
    if (change === 'target') world.state.revision = 'changed'
    if (change === 'owner') world.state.owner = false
    if (change === 'args') world.db.query('UPDATE hub_supervisor_actions SET args=? WHERE id=?').run(JSON.stringify({ ...world.args, target: 'different remote' }), action.id)
    try { await world.confirm(action.id) } catch { /* Owner revocation is an explicit admission refusal. */ }
    expect(world.executed).toHaveLength(0)
    const stored = world.db.query<{ state: string }, [string]>('SELECT state FROM hub_supervisor_actions WHERE id=?').get(action.id)
    expect(stored?.state).not.toBe('accepted')
  }
})

test('the provider-facing risky tool only proposes an owner card and never uses observation dispatch', async () => {
  const world = fixture()
  const abilities = new SuperagentAbilities({ ...world, authorized: () => true, deliver: async () => {}, audit: async () => {} })
  try {
    const registry = createSupervisorToolRegistry({ ...world, actions: world.service, abilities })
    const turn = { threadId: world.ownerThread.id, requester: world.user('owner'), reader: world.reader('owner', world.ownerThread.id), denial: undefined }
    const raw = await registry.execute(toolCall('propose_risky_action', world.args), { metadata: { [TURN_METADATA_KEY]: turn } })
    const result: unknown = JSON.parse(raw)
    expect(result).toMatchObject({ state: 'pending_confirmation' })
    expect(world.executed).toHaveLength(0); expect(world.backendCalls).toHaveLength(0)
  } finally { abilities.dispose() }
})

test('direct risky intents reject unknown operations and extra authority fields', async () => {
  const world = fixture()
  for (const extra of [{ operation: 'shell.run' }, { confirmed: true }, { owner_approved: true }]) {
    await expect(world.service.createIntent({ threadId: world.ownerThread.id, tool: 'risk.execute', args: { ...world.args, ...extra },
      requester: world.user('owner'), reader: world.reader('owner', world.ownerThread.id) })).rejects.toThrow()
  }
  expect(world.executed).toHaveLength(0)
})

test('an unknown risky action outcome is never replayed', async () => {
  const world = fixture(); world.state.fail = true
  const action = await world.propose()
  expect((await world.confirm(action.id)).state).toBe('unknown')
  expect((await world.confirm(action.id)).state).toBe('unknown')
  expect(world.executed).toHaveLength(1)
})
