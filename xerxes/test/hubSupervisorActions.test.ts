// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'

import { HubProtocolError } from '../src/hub/protocol.js'
import { ActionService, UNKNOWN_OUTCOME_MESSAGE, actionToWire } from '../src/hub/supervisor/actions.js'
import { TURN_METADATA_KEY } from '../src/hub/supervisor/tools.js'
import { disposeAll } from './hubHarness.js'
import { makeWorld, steerArgs, toolCall } from './hubSupervisorHarness.js'

afterEach(disposeAll)

async function code(promise: Promise<unknown>): Promise<string | undefined> {
  try {
    await promise
    return undefined
  } catch (error) {
    return error instanceof HubProtocolError ? error.code : error instanceof Error ? error.message : String(error)
  }
}

describe('supervisor confirmed actions', () => {
  test('a mutation tool only proposes: pending_confirmation, a bound card, nothing executed', async () => {
    const world = makeWorld()
    const turn = { threadId: 'shared', requester: world.user('alice'), reader: world.reader('alice'), denial: undefined }
    const out = JSON.parse(await world.registry.execute(toolCall('task_steer', steerArgs()), { metadata: { [TURN_METADATA_KEY]: turn } }))
    expect(out.state).toBe('pending_confirmation')
    expect(world.calls).toHaveLength(0)
    const card = world.actions.get(out.action_id)!
    expect(card.state).toBe('pending')
    expect(card.expiresAt - card.createdAt).toBe(10 * 60 * 1000)
    expect(card.argsHash).toMatch(/^[0-9a-f]{64}$/)
    expect(card.instanceId).toBe('i1')
    expect(card.targetRevision).toBeTruthy()
    expect(card.idempotencyKey.length).toBeGreaterThan(20)
  })

  test('a human with the needed scope confirms; the router sees an agent actor carrying requested_by and confirmed_by', async () => {
    const world = makeWorld()
    const card = await world.propose('task.steer', steerArgs())
    const done = await world.confirm('owner', card.id)
    expect(done.state).toBe('accepted')
    expect(world.calls).toHaveLength(1)
    const call = world.calls[0]!
    expect(call.params).toMatchObject({ workspace_id: 'w1', session_id: 's1', action: 'turn.steer', args: { text: 'please focus on the tests' } })
    expect(call.params.command_id).toBe(`${card.idempotencyKey}:0`)
    expect(call.context.principal).toMatchObject({ kind: 'agent', id: 'xerxes-supervisor', onBehalfOf: 'alice' })
    expect(call.context.actor).toMatchObject({ principal_kind: 'agent', requested_by: 'alice', confirmed_by: 'owner' })
    const scopes = call.grants.flatMap(grant => grant.scopes ?? [])
    expect(scopes).toContain('session.steer')
    expect(scopes).not.toContain('permission.respond')
    expect(scopes).not.toContain('grants.manage')
    // The delegated authority exists only inside the dispatch.
    expect(world.overlay.current()).toEqual([])
    expect(actionToWire(done)).toMatchObject({ state: 'confirmed', confirmed_by: 'OWNER', requested_by: 'ALICE' })
  })

  test('an agent principal cannot confirm, structurally', async () => {
    const world = makeWorld()
    const card = await world.propose('task.steer', steerArgs())
    expect(await code(world.confirm('robot', card.id))).toBe('forbidden')
    expect(await code(world.actions.status(world.identity.context('robot'), { confirmation_id: card.id }))).toBe('forbidden')
    expect(world.actions.get(card.id)!.state).toBe('pending')
    expect(world.calls).toHaveLength(0)
  })

  test('a viewer cannot confirm a mutation that needs session.steer', async () => {
    const world = makeWorld()
    const card = await world.propose('task.steer', steerArgs())
    expect(await code(world.confirm('viewer', card.id))).toBe('forbidden')
    expect(world.actions.get(card.id)!.state).toBe('pending')
    expect(world.calls).toHaveLength(0)
    // The refusal is audited as a denial.
    expect(world.identity.audits.some(entry => entry.decision === 'denied' && entry.action.startsWith('supervisor.denied'))).toBe(true)
  })

  test('a thread with a viewer in it cannot even propose a steer: the proposal is denied and nothing is stored', async () => {
    const world = makeWorld()
    world.reader('viewer')
    expect(await code(world.propose('task.steer', steerArgs()))).toContain('session.steer')
    expect(world.actions.pendingCount('shared')).toBe(0)
  })

  test('an expired card cannot be confirmed and nothing runs', async () => {
    const world = makeWorld()
    const card = await world.propose('task.steer', steerArgs())
    world.time.now += 11 * 60 * 1000
    expect(await code(world.confirm('alice', card.id))).toBe('conflict')
    expect(world.actions.get(card.id)!.state).toBe('expired')
    expect(world.calls).toHaveLength(0)
  })

  test('a replayed confirmation reports the recorded state and executes once', async () => {
    const world = makeWorld()
    const card = await world.propose('task.steer', steerArgs())
    expect((await world.confirm('alice', card.id)).state).toBe('accepted')
    expect((await world.confirm('alice', card.id)).state).toBe('accepted')
    expect((await world.confirm('alice', card.id, 'cancel')).state).toBe('accepted')
    expect(world.calls).toHaveLength(1)
  })

  test('concurrent confirmations execute exactly once', async () => {
    const world = makeWorld()
    const card = await world.propose('task.steer', steerArgs())
    const [a, b] = await Promise.all([world.confirm('alice', card.id), world.confirm('owner', card.id)])
    expect(['confirming', 'accepted']).toContain(a.state)
    expect(['confirming', 'accepted']).toContain(b.state)
    expect(world.calls).toHaveLength(1)
    expect(world.actions.get(card.id)!.state).toBe('accepted')
  })

  test('a stale task revision is refused at execution and nothing is sent', async () => {
    const world = makeWorld()
    const card = await world.propose('task.steer', steerArgs())
    world.setSessions([{ id: 's1', title: 'First task', status: 'working' }, { id: 's2', title: 'Second task', status: 'working' }])
    const done = await world.confirm('alice', card.id)
    expect(done.state).toBe('rejected')
    expect(done.errorMessage).toContain('changed')
    expect(world.calls).toHaveLength(0)
  })

  test('a runtime restart or an offline workspace is refused before anything is sent', async () => {
    const world = makeWorld()
    const first = await world.propose('task.steer', steerArgs())
    world.state.instance = 'i2'
    expect((await world.confirm('alice', first.id)).errorMessage).toContain('restarted')
    world.state.instance = 'i1'
    world.time.now += 6000
    const second = await world.propose('task.steer', steerArgs('another nudge'))
    world.state.online = false
    expect((await world.confirm('alice', second.id)).errorMessage).toContain('not online')
    expect(world.calls).toHaveLength(0)
  })

  test('an unknown outcome is recorded, shown honestly and never retried', async () => {
    const world = makeWorld()
    world.behavior.run = params => ({ ok: false, command_id: String(params.command_id), state: 'unknown', error: { code: 'unavailable', message: 'ssh dropped' } })
    const card = await world.propose('task.steer', steerArgs())
    const done = await world.confirm('alice', card.id)
    expect(done.state).toBe('unknown')
    expect(actionToWire(done)).toMatchObject({ state: 'failed', message: UNKNOWN_OUTCOME_MESSAGE })
    expect((await world.confirm('alice', card.id)).state).toBe('unknown')
    expect(world.calls).toHaveLength(1)
  })

  test('a router refusal is a rejected outcome, not a crash', async () => {
    const world = makeWorld()
    world.behavior.run = () => { throw new HubProtocolError('forbidden', 'router refused') }
    const card = await world.propose('task.steer', steerArgs())
    const done = await world.confirm('alice', card.id)
    expect(done.state).toBe('rejected')
    expect(done.errorMessage).toBe('router refused')
  })

  test('task.stop pauses goal continuation, then cancels, and reports requested versus stopped', async () => {
    const world = makeWorld()
    const card = await world.propose('task.stop', { workspace_id: 'w1', task_id: 's1', cascade: true })
    const done = await world.confirm('alice', card.id)
    expect(world.calls.map(call => call.params.action)).toEqual(['goal.set', 'turn.cancel'])
    expect(world.calls[0]!.params.args).toEqual({ input: 'pause' })
    expect(done.outcome).toMatchObject({ requested: true, stopped: true, goal_paused: true, cascade_requested: true, cascade_applied: false })
  })

  test('task.start opens a task, then submits to the id it was given', async () => {
    const world = makeWorld()
    const card = await world.propose('task.start', { workspace_id: 'w1', text: 'run the tests', preset: 'coder' })
    const done = await world.confirm('alice', card.id)
    expect(done.state).toBe('accepted')
    expect(world.calls.map(call => call.params.action)).toEqual(['session.open', 'turn.submit'])
    expect(world.calls[0]!.params.args).toEqual({ agent_id: 'coder' })
    expect('session_id' in world.calls[0]!.params).toBe(false)
    expect(world.calls[1]!.params).toMatchObject({ session_id: 'new-1', args: { text: 'run the tests' } })
    expect(new Set(world.calls.map(call => call.params.command_id)).size).toBe(2)
  })

  test('cancel by a person records the decision and nothing runs', async () => {
    const world = makeWorld()
    const card = await world.propose('task.steer', steerArgs())
    const done = await world.confirm('alice', card.id, 'cancel')
    expect(done.state).toBe('cancelled')
    expect(world.calls).toHaveLength(0)
    expect((await world.confirm('owner', card.id)).state).toBe('cancelled')
    expect(world.calls).toHaveLength(0)
  })

  test('goal.set refuses a stale expected_revision at proposal and at execution', async () => {
    const world = makeWorld()
    world.setSessions([{ id: 's1', title: 'First task', status: 'idle', goal: 'ship it' }, { id: 's2', title: 'Second task', status: 'idle' }])
    expect(await code(world.propose('goal.set', { workspace_id: 'w1', task_id: 's1', expected_revision: 0, objective: 'new goal' }))).toContain('stale')
    const card = await world.propose('goal.set', { workspace_id: 'w1', task_id: 's1', expected_revision: 1, objective: 'new goal', criteria: [{ id: 'c1', description: 'tests pass' }] })
    const done = await world.confirm('alice', card.id)
    expect(done.state).toBe('accepted')
    expect(world.calls[0]!.params).toMatchObject({ action: 'goal.set' })
    expect(String((world.calls[0]!.params.args as { input: string }).input)).toContain('tests pass')
  })

  test('model.set is offered only when the runtime guarantees a session-only change', async () => {
    const without = makeWorld()
    expect(await code(without.propose('model.set', { workspace_id: 'w1', task_id: 's1', provider_profile: 'p', model: 'm' }))).toContain('session-only')
    const withIt = makeWorld({ modelSetSessionOnly: true })
    const card = await withIt.propose('model.set', { workspace_id: 'w1', task_id: 's1', provider_profile: 'p', model: 'm' })
    expect((await withIt.confirm('alice', card.id)).state).toBe('accepted')
    expect(withIt.calls[0]!.params).toMatchObject({ action: 'model.set', args: { model: 'm', provider_profile: 'p' } })
  })

  test('no self-targeting and no recursive supervisors', async () => {
    const world = makeWorld()
    expect(await code(world.propose('task.start', { workspace_id: 'w1', text: 'spawn', preset: 'xerxes-supervisor' }))).toContain('supervisor')
    expect(await code(world.propose('task.steer', { workspace_id: 'w1', task_id: 'shared', text: 'x' }))).toContain('supervisor thread')
    expect(await code(world.propose('agent.message', { workspace_id: 'w1', task_id: 's1', agent_id: 'XERXES', text: 'x' }))).toContain('supervisor')
  })

  test('a repeated identical proposal returns the same card and a rapid second steer is refused', async () => {
    const world = makeWorld()
    const a = await world.propose('task.steer', steerArgs())
    const b = await world.propose('task.steer', steerArgs())
    expect(b.id).toBe(a.id)
    expect(await code(world.propose('task.steer', steerArgs('a different nudge')))).toContain('just steered')
    world.time.now += 6000
    expect((await world.propose('task.steer', steerArgs('a different nudge'))).id).not.toBe(a.id)
  })

  test('every stage is audited with a request hash only', async () => {
    const world = makeWorld()
    const card = await world.propose('task.steer', steerArgs('secret instruction text'))
    await world.confirm('alice', card.id)
    const stages = world.identity.audits.map(entry => entry.action)
    for (const stage of ['intent', 'confirm', 'dispatch', 'outcome']) expect(stages).toContain(`supervisor.${stage}:task.steer`)
    for (const entry of world.identity.audits) expect(entry.requestHash).toMatch(/^[0-9a-f]{64}$/)
    expect(JSON.stringify(world.identity.audits)).not.toContain('secret instruction text')
    expect(world.identity.audits.at(-1)).toMatchObject({ action: 'supervisor.outcome:task.steer', outcome: 'accepted' })
    expect(world.identity.audits.find(entry => entry.action === 'supervisor.dispatch:task.steer')!.actor).toMatchObject({ principal_kind: 'agent', requested_by: 'alice', confirmed_by: 'alice' })
  })

  test('the audit log failing fails closed: no card at proposal, nothing sent at confirmation', async () => {
    const world = makeWorld()
    world.identity.failAudit = true
    expect(await code(world.propose('task.steer', steerArgs()))).toContain('audit log is unavailable')
    expect(world.actions.pendingCount('shared')).toBe(0)
    world.identity.failAudit = false
    const card = await world.propose('task.steer', steerArgs())
    world.identity.failAudit = true
    const done = await world.confirm('alice', card.id)
    expect(done.state).toBe('rejected')
    expect(world.calls).toHaveLength(0)
  })

  test('a dispatch that was in flight when the Hub stopped becomes unknown and is never replayed', async () => {
    const world = makeWorld()
    const card = await world.propose('task.steer', steerArgs())
    world.db.query("UPDATE hub_supervisor_actions SET state = 'confirming' WHERE id = ?").run(card.id)
    const reopened = new ActionService({
      db: world.db, threads: world.threads, identity: world.identity, targets: world.targets,
      execute: async () => { throw new Error('must not be called') }, overlay: world.overlay, access: () => {}, clock: world.clock,
    })
    expect(reopened.get(card.id)!.state).toBe('unknown')
    expect((await reopened.respond(world.identity.context('alice'), { confirmation_id: card.id, decision: 'confirm' })).state).toBe('unknown')
  })

  test('unknown parameters and bad decisions are rejected', async () => {
    const world = makeWorld()
    const card = await world.propose('task.steer', steerArgs())
    expect(await code(world.actions.respond(world.identity.context('alice'), { confirmation_id: card.id, decision: 'maybe' }))).toBe('invalid_request')
    expect(await code(world.actions.respond(world.identity.context('alice'), { confirmation_id: card.id, decision: 'confirm', extra: 1 }))).toBe('invalid_request')
    expect(await code(world.actions.respond(world.identity.context('alice'), { confirmation_id: 'nope', decision: 'confirm' }))).toBe('not_found')
  })
})
