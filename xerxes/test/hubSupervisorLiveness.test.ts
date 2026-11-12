// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { join } from 'node:path'

import { emptyProjection, replaceSessionSummaries } from '../src/hub/projections.js'
import { UNKNOWN_OUTCOME_MESSAGE, actionToWire, type ActionRecord } from '../src/hub/supervisor/actions.js'
import { formatSupervisorStatus, type StatusSnapshot, type StatusTaskFacts } from '../src/hub/supervisor/digests.js'
import { SupervisorGrantOverlay, createHubSupervisor } from '../src/hub/supervisor/index.js'
import type { SupervisorStatusPort } from '../src/hub/supervisor/runner.js'
import { TURN_METADATA_KEY, createSupervisorToolRegistry } from '../src/hub/supervisor/tools.js'
import { FakeIdentity, disposeAll, registerCleanup, standardUsers, tempDir, until } from './hubHarness.js'
import { T0, makeRunner, makeWorld, scriptedModel, toolCall, type World } from './hubSupervisorHarness.js'

/**
 * XERXES liveness: a wake or a person's question never becomes an empty or canned
 * reply. An empty model answer is replaced by a Hub-authored digest of the state
 * the Hub can actually observe, or by an explicit "I could not check" that names
 * the failure. With nothing observable at all, a wake stays silent.
 */

afterEach(disposeAll)

const PLACEHOLDER = 'I have nothing to add.'

const WATCHED = {
  tool_calls: 42,
  edits: 6,
  tests: { runs: 1, passed: 12, failed: 0, last: 'pass' as const },
  last_activity_at: T0 - 120_000,
  anomalies: [] as readonly string[],
}

function workingTask(): StatusTaskFacts {
  return {
    workspace_id: 'w1',
    session_id: 's2',
    workspace_label: 'One',
    title: 'Second task',
    status: 'working',
    agents_total: 3,
    agents_active: 2,
    pending_human_count: 1,
    goal_objective: 'ship the release',
    updated_at: new Date(T0 - 30_000).toISOString(),
    stale: false,
    watched: WATCHED,
  }
}

function snapshot(tasks: readonly StatusTaskFacts[] = [workingTask()]): StatusSnapshot {
  return { tasks, attention: [] }
}

function port(read: () => StatusSnapshot): SupervisorStatusPort {
  return { read }
}

function turn(world: World, who = 'alice') {
  return { threadId: 'shared', requester: world.user(who) }
}

const wake = (world: World, target = { workspaceId: 'w1', sessionId: 's2' }) => ({
  ...turn(world),
  trigger: { kind: 'autopilot' as const, text: 'Autopilot: the Hub woke you for a periodic check-in.', target },
})

function say(world: World, text = 'did u run the inference engine?', who = 'alice'): void {
  world.threads.access('shared', world.user(who))
  world.threads.append({ threadId: 'shared', role: 'user', text, author: { id: who, name: who.toUpperCase() } })
}

const replies = (world: World) => world.threads.messagesFor('shared', 'alice').filter(message => message.role === 'xerxes')
const messageEvents = (world: World) =>
  world.events.filter(event => event.type === 'supervisor.thread_event' && typeof event.payload.message === 'object')

describe('supervisor status digest', () => {
  test('a wake digest reports the task, its state, progress counts and what waits', () => {
    const text = formatSupervisorStatus({ snapshot: snapshot(), now: T0, mode: 'wake', focus: { workspaceId: 'w1', sessionId: 's2' } })
    expect(text).toBeDefined()
    expect(text!).toContain('Second task')
    expect(text!).toContain('working')
    expect(text!).toContain('2 of 3 agents')
    expect(text!).toContain('42 tool calls')
    expect(text!).toContain('1 item is waiting')
    expect(text!).toContain('last activity')
    expect(text!).not.toContain(PLACEHOLDER)
  })

  test('an unwatched working task still reports the facts the Hub has', () => {
    const bare: StatusTaskFacts = {
      workspace_id: 'w1', session_id: 's2', workspace_label: 'One', title: 'Second task', status: 'working',
      agents_total: 0, agents_active: 0, pending_human_count: 0, goal_objective: '',
      updated_at: new Date(T0 - 30_000).toISOString(), stale: false,
    }
    const text = formatSupervisorStatus({ snapshot: snapshot([bare]), now: T0, mode: 'wake' })
    expect(text!).toContain('Second task')
    expect(text!).toContain('working')
    expect(text!).toContain('last activity')
  })

  test('nothing observable: a wake has no digest, a direct turn always gets an honest answer', () => {
    const empty: StatusSnapshot = { tasks: [], attention: [] }
    expect(formatSupervisorStatus({ snapshot: empty, now: T0, mode: 'wake' })).toBeUndefined()
    expect(formatSupervisorStatus({ snapshot: empty, now: T0, mode: 'wake', failure: 'the Hub database is closed' })).toBeUndefined()
    const direct = formatSupervisorStatus({ snapshot: empty, now: T0, mode: 'direct' })
    expect(direct!).toContain('I could not check')
    expect(direct!).not.toContain(PLACEHOLDER)
    const failed = formatSupervisorStatus({ snapshot: empty, now: T0, mode: 'direct', failure: 'the Hub database is closed' })
    expect(failed!).toContain('I could not check')
    expect(failed!).toContain('the Hub database is closed')
  })

  test('a stale task says so instead of reporting a live state as fact', () => {
    const text = formatSupervisorStatus({ snapshot: snapshot([{ ...workingTask(), stale: true }]), now: T0, mode: 'direct' })
    expect(text!).toContain('lost contact')
  })
})

describe('supervisor liveness', () => {
  test('a check-in wake with an empty model reply posts the observed progress, never a placeholder', async () => {
    const world = makeWorld()
    world.setSessions([{ id: 's1', title: 'First task', status: 'idle' }, { id: 's2', title: 'Second task', status: 'working' }])
    const scripted = scriptedModel([{ text: '   ', toolCalls: [] }])
    const result = await makeRunner(world, scripted.model, { status: port(() => snapshot()) }).enqueue(wake(world))
    expect(result.stop).toBe('completed')
    expect(result.reply?.text).toBeDefined()
    expect(result.reply!.text).not.toBe(PLACEHOLDER)
    expect(result.reply!.text).toContain('Second task')
    expect(result.reply!.text).toContain('working')
    expect(result.reply!.text).toContain('42 tool calls')
    expect(result.reply!.origin).toBe('wake')
    expect(replies(world).map(message => message.text)).toEqual([result.reply!.text])
  })

  test('a wake whose state is not observable posts no message at all', async () => {
    const world = makeWorld()
    const scripted = scriptedModel([{ text: '', toolCalls: [] }])
    const result = await makeRunner(world, scripted.model, { status: port(() => ({ tasks: [], attention: [] })) }).enqueue(wake(world))
    expect(result.stop).toBe('completed')
    expect(result.reply).toBeUndefined()
    expect(world.threads.messagesFor('shared', 'alice')).toEqual([])
    expect(messageEvents(world)).toEqual([])
  })

  test("a person's question with an empty model reply is answered from the observed state", async () => {
    const world = makeWorld()
    say(world)
    const asked = world.threads.messagesFor('shared', 'alice').at(-1)!
    const result = await makeRunner(world, scriptedModel([{ text: '', toolCalls: [] }]).model, { status: port(() => snapshot()) })
      .enqueue(turn(world))
    expect(result.stop).toBe('completed')
    expect(result.reply!.text).not.toBe(PLACEHOLDER)
    expect(result.reply!.text).toContain('Second task')
    expect(result.reply!.text).toContain('working')
    // It answers the person who asked, not a generic notice.
    expect(result.reply!.causalMessageId).toBe(asked.id)
    expect(replies(world)).toHaveLength(1)
  })

  test('a person is told which check failed, by name, rather than given filler', async () => {
    const world = makeWorld()
    say(world)
    const result = await makeRunner(world, scriptedModel([{ text: '', toolCalls: [] }]).model, {
      status: port(() => { throw new Error('the Hub database is closed') }),
    }).enqueue(turn(world))
    expect(result.stop).toBe('completed')
    expect(result.reply!.text).toContain('I could not check')
    expect(result.reply!.text).toContain('the Hub database is closed')
    expect(result.reply!.text).not.toBe(PLACEHOLDER)
  })

  test('without a status reader a direct turn still says what it could not read', async () => {
    const world = makeWorld()
    say(world)
    const result = await makeRunner(world, scriptedModel([{ text: '', toolCalls: [] }]).model).enqueue(turn(world))
    expect(result.stop).toBe('completed')
    expect(result.reply!.text).toContain('I could not check')
    expect(result.reply!.text).not.toBe(PLACEHOLDER)
  })

  test('a real model reply is never replaced by a digest', async () => {
    const world = makeWorld()
    say(world)
    const result = await makeRunner(world, scriptedModel([{ text: 'Yes, it is running.', toolCalls: [] }]).model, {
      status: port(() => snapshot()),
    }).enqueue(turn(world))
    expect(result.reply?.text).toBe('Yes, it is running.')
    expect(result.reply?.causalMessageId).toBeDefined()
  })

  test('a scheduled check-in carries the Hub facts even when the model did write something', async () => {
    const world = makeWorld()
    const scripted = scriptedModel([{ text: 'Nothing needs doing.', toolCalls: [] }])
    const result = await makeRunner(world, scripted.model, { status: port(() => snapshot()) }).enqueue({
      ...wake(world),
      trigger: { kind: 'autopilot' as const, text: 'Autopilot: the Hub woke you for a periodic check-in.', target: { workspaceId: 'w1', sessionId: 's2' }, scheduled: true },
    })
    expect(result.reply!.text).toContain('Nothing needs doing.')
    expect(result.reply!.text).toContain('Second task')
    expect(result.reply!.text).toContain('42 tool calls')
    expect(result.reply!.text).toContain('1 item is waiting')
    expect(result.reply!.origin).toBe('wake')
  })

  test('an anomaly wake keeps a real reply as it is, without tacking facts on', async () => {
    const world = makeWorld()
    const result = await makeRunner(world, scriptedModel([{ text: 'Steered it back to the tests.', toolCalls: [] }]).model, {
      status: port(() => snapshot()),
    }).enqueue(wake(world))
    expect(result.reply?.text).toBe('Steered it back to the tests.')
  })

  test('the composed Hub wires real task state into a check-in that produced no reply', async () => {
    const dir = await tempDir('supervisor-liveness-')
    const identity = standardUsers(new FakeIdentity())
    const time = { now: T0 }
    let projection = replaceSessionSummaries(
      emptyProjection('w1'),
      [{ id: 's1', title: 'Working task', status: 'working' }, { id: 's2', title: 'Idle task', status: 'idle' }],
      new Date(T0).toISOString(),
    )
    const supervisor = createHubSupervisor({
      databasePath: join(dir, 'hub.sqlite'),
      identity,
      overlay: new SupervisorGrantOverlay(),
      execute: async (_context, params) => ({ ok: true, command_id: String(params.command_id), state: 'accepted', result: {} }),
      workspaces: {
        list: () => [{ id: 'w1', label: 'One' }],
        projection: id => (id === 'w1' ? projection : undefined),
        health: id => (id === 'w1' ? { state: 'online', degraded: false, instanceId: 'i1' } : undefined),
      },
      backend: { call: async () => ({ state: 'accepted', result: { ok: true } }) },
      model: scriptedModel([{ text: '', toolCalls: [] }]).model,
      persona: 'You are XERXES (test persona).',
      isHubOwner: userId => userId === 'owner',
      autopilot: { checkInMs: 0 },
      clock: () => time.now,
    })
    registerCleanup(() => supervisor.close())
    await supervisor.handlers['supervisor.autopilot.set']!(
      { context: identity.context('owner'), grants: identity.grantsFor('owner') },
      { workspace_id: 'w1', enabled: true },
    )
    expect(supervisor.waker.checkIn()).toBe(1)
    await until(() => supervisor.threads.messagesFor('autopilot-owner', 'owner').some(message => message.role === 'xerxes'))
    const reply = supervisor.threads.messagesFor('autopilot-owner', 'owner').find(message => message.role === 'xerxes')!
    expect(reply.text).toContain('Working task')
    expect(reply.text).toContain('working')
    expect(reply.text).toContain('last activity')
    expect(reply.text).not.toContain(PLACEHOLDER)
    // A check-in is a scheduled wake: the facts ride with it.
    expect(reply.origin).toBe('wake')
  })
})

describe('supervisor unconfirmed mutation', () => {
  test('an unknown outcome is checked against current state before it is reported', async () => {
    const world = makeWorld()
    const card = await world.propose('task.start', { workspace_id: 'w1', task_id: 's1', text: 'run the failing suite again' })
    // The command may have applied; the task is working when the supervisor checks.
    world.behavior.run = params => {
      world.setSessions([{ id: 's1', title: 'First task', status: 'working' }, { id: 's2', title: 'Second task', status: 'working' }])
      return { ok: false, command_id: String(params.command_id), state: 'unknown', error: { code: 'unavailable', message: 'ssh dropped' } }
    }
    const done = await world.confirm('alice', card.id)
    expect(done.state).toBe('unknown')
    expect(done.errorMessage).toContain('Could not confirm whether this was applied')
    expect(done.errorMessage).toContain('First task')
    expect(done.errorMessage).toContain('working')
    expect(done.errorMessage).toContain('XERXES will not retry it')
    expect(done.outcome).toMatchObject({ verification: { conclusion: 'looks_applied' } })
    // Nothing is retried, and the card says what it checked.
    expect(world.calls).toHaveLength(1)
    const wire = actionToWire(done)
    expect(wire.state).toBe('failed')
    expect(String(wire.message)).toContain('XERXES checked')
    expect(String(wire.message)).toContain('working')
  })

  test('an unknown outcome that left the task unchanged says so, and still never retries', async () => {
    const world = makeWorld()
    const card = await world.propose('task.start', { workspace_id: 'w1', task_id: 's1', text: 'run the failing suite again' })
    world.behavior.run = params => ({ ok: false, command_id: String(params.command_id), state: 'unknown', error: { code: 'unavailable', message: 'ssh dropped' } })
    const done = await world.confirm('alice', card.id)
    expect(done.state).toBe('unknown')
    expect(done.errorMessage).toContain('does not show it was applied')
    expect(done.outcome).toMatchObject({ verification: { conclusion: 'looks_not_applied' } })
    expect(world.calls).toHaveLength(1)
    expect((await world.confirm('alice', card.id)).state).toBe('unknown')
    expect(world.calls).toHaveLength(1)
  })

  test('a new task whose id never arrived cannot be matched, and that is said plainly', async () => {
    const world = makeWorld()
    const card = await world.propose('task.start', { workspace_id: 'w1', text: 'a brand new task' })
    world.behavior.run = params => ({ ok: false, command_id: String(params.command_id), state: 'unknown', error: { code: 'unavailable', message: 'ssh dropped' } })
    const done = await world.confirm('alice', card.id)
    expect(done.state).toBe('unknown')
    expect(done.outcome).toMatchObject({ verification: { conclusion: 'undetermined' } })
    expect(done.errorMessage).toContain('no task id')
    expect(world.calls).toHaveLength(1)
  })

  test('a steer leaves no trace the projections can prove, and the message admits it', async () => {
    const world = makeWorld()
    const card = await world.propose('task.steer', { workspace_id: 'w1', task_id: 's1', text: 'focus on the failing test' })
    world.behavior.run = params => ({ ok: false, command_id: String(params.command_id), state: 'unknown', error: { code: 'unavailable', message: 'ssh dropped' } })
    const done = await world.confirm('alice', card.id)
    expect(done.state).toBe('unknown')
    expect(done.outcome).toMatchObject({ verification: { conclusion: 'undetermined' } })
    expect(done.errorMessage).toContain('does not prove')
    expect(done.errorMessage).toContain('will not retry')
    expect(UNKNOWN_OUTCOME_MESSAGE).toContain('will not retry')
    expect(world.calls).toHaveLength(1)
  })

  test('the model is told what was checked, so its reply can be substantive', async () => {
    const world = makeWorld()
    const card = await world.propose('task.stop', { workspace_id: 'w1', task_id: 's1' })
    const done = await world.confirm('alice', card.id)
    expect(done.state).toBe('accepted')
    const unknown: ActionRecord = {
      ...done,
      state: 'unknown',
      auto: true,
      confirmedByName: 'OWNER',
      outcome: { verification: { conclusion: 'looks_applied', checked: '“First task” in One', observed: 'it now shows working' } },
      errorMessage: 'Could not confirm whether this was applied. XERXES checked “First task” in One: it now shows working. That matches what was asked, so it was most likely applied. Check the task before trying again. XERXES will not retry it.',
    }
    expect(card.id).toBe(done.id)
    const registry = createSupervisorToolRegistry({
      workspaces: world.workspaces,
      backend: world.backend,
      digests: world.digests,
      actions: { createIntent: async () => unknown },
      threads: world.threads,
    })
    const turnState = { threadId: 'shared', requester: world.user('alice'), reader: world.reader('alice'), denial: undefined }
    const raw = await registry.execute(toolCall('task_stop', { workspace_id: 'w1', task_id: 's1' }), { metadata: { [TURN_METADATA_KEY]: turnState } })
    const result = JSON.parse(String(raw)) as Record<string, unknown>
    expect(result.state).toBe('unknown')
    expect(result.message).toContain('XERXES checked')
    expect(result.outcome).toMatchObject({ verification: { conclusion: 'looks_applied' } })
    expect(String(result.note)).toContain('checked')
  })
})