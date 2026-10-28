// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { join } from 'node:path'

import { HubProtocolError, type HubActor } from '../src/hub/protocol.js'
import { SupervisorGrantOverlay, createHubSupervisor, createUnsupportedSupervisorHandlers, SUPERVISOR_SETUP_HINT } from '../src/hub/supervisor/index.js'
import { emptyProjection, replaceSessionSummaries } from '../src/hub/projections.js'
import { disposeAll, startFixture, tempDir, FakeIdentity, standardUsers, until } from './hubHarness.js'
import { makeWorld, scriptedModel, T0 } from './hubSupervisorHarness.js'

afterEach(disposeAll)

describe('supervisor threads', () => {
  test('a private thread is isolated: separate transcript, hidden from others, never in the shared thread', () => {
    const world = makeWorld()
    const alice = world.user('alice')
    const owner = world.user('owner')
    const priv = world.threads.privateFor(alice)
    world.threads.access(priv.id, alice)
    world.threads.append({ threadId: priv.id, role: 'user', text: 'my secret question', author: { id: 'alice', name: 'ALICE' } })
    world.threads.append({ threadId: 'shared', role: 'user', text: 'public question', author: { id: 'alice', name: 'ALICE' } })
    expect(() => world.threads.access(priv.id, owner)).toThrow(HubProtocolError)
    expect(world.threads.messagesFor('shared', 'owner').map(m => m.text)).toEqual(['public question'])
    expect(world.threads.messagesFor(priv.id, 'alice').map(m => m.text)).toEqual(['my secret question'])
    expect(world.threads.privateFor(alice).id).toBe(priv.id)
    expect(world.threads.privateFor(alice, true).id).not.toBe(priv.id)
    expect(world.threads.recipients(priv)).toEqual(['alice'])
  })

  test('events go only to authorized participants of the thread', () => {
    const world = makeWorld()
    world.threads.access('shared', world.user('alice'))
    const priv = world.threads.privateFor(world.user('alice'))
    world.threads.emit(priv.id, 'supervisor.thread_event', { x: 1 })
    expect(world.events.at(-1)).toMatchObject({ to: ['alice'], type: 'supervisor.thread_event' })
    world.threads.emit('shared', 'supervisor.confirmation', { y: 1 })
    expect(world.events.at(-1)!.to).toEqual(['alice'])
  })

  test('shared read scope is the intersection of the people in the thread', () => {
    const world = makeWorld()
    const alone = world.reader('alice')
    expect(alone.scopes({ workspaceId: 'w1' }).has('session.steer')).toBe(true)
    const withViewer = world.reader('viewer')
    expect(withViewer.scopes({ workspaceId: 'w1' }).has('session.steer')).toBe(false)
    expect(withViewer.canRead({ workspaceId: 'w1' })).toBe(true)
    // the agent ceiling excludes human-only scopes even for the owner
    expect(world.reader('owner').scopes({ workspaceId: 'w1' }).has('permission.respond')).toBe(false)
  })

  test('every message is attributed and a repeated client id is deduplicated', () => {
    const world = makeWorld()
    const first = world.threads.append({ threadId: 'shared', role: 'user', text: 'hi', author: { id: 'alice', name: 'ALICE', deviceId: 'dev-alice' }, source: 'voice', clientMessageId: 'm1' })
    const again = world.threads.append({ threadId: 'shared', role: 'user', text: 'hi', author: { id: 'alice', name: 'ALICE' }, clientMessageId: 'm1' })
    expect(again.duplicate).toBe(true)
    expect(again.message.id).toBe(first.message.id)
    expect(first.message).toMatchObject({ id: 'm1', authorId: 'alice', authorName: 'ALICE', deviceId: 'dev-alice', source: 'voice' })
  })
})

describe('supervisor service', () => {
  test('an unconfigured supervisor answers unsupported with the setup hint on every method', async () => {
    const fixture = await startFixture({ server: { methodHandlers: createUnsupportedSupervisorHandlers(SUPERVISOR_SETUP_HINT) } })
    const alice = await fixture.client('alice')
    for (const method of ['supervisor.thread.open', 'supervisor.message.submit', 'supervisor.confirmation.respond', 'supervisor.action.status'] as const) {
      const response = await alice.request(method, {})
      expect(response.ok).toBe(false)
      if (!response.ok) {
        expect(response.error.code).toBe('unsupported')
        expect(response.error.message).toContain('XERXES_HUB_SUPERVISOR_PROFILE')
      }
    }
  })

  async function configured() {
    const dir = await tempDir('xh-sup-')
    const identity = standardUsers(new FakeIdentity())
    const overlay = new SupervisorGrantOverlay()
    let projection = replaceSessionSummaries(emptyProjection('w1'), [{ id: 's1', title: 'First task', status: 'idle' }], new Date(T0).toISOString())
    const scripted = scriptedModel([{ text: 'All quiet.', toolCalls: [] }])
    const sent: { to: readonly string[]; type: string }[] = []
    const supervisor = createHubSupervisor({
      databasePath: join(dir, 'hub.sqlite'),
      identity,
      overlay,
      execute: async (_c, params) => ({ ok: true, command_id: String(params.command_id), state: 'accepted', result: {} }),
      workspaces: {
        list: () => [{ id: 'w1', label: 'One' }],
        projection: () => projection,
        health: () => ({ state: 'online', degraded: false, instanceId: 'i1' }),
      },
      backend: { call: async () => ({ state: 'accepted', result: { ok: true } }) },
      model: scripted.model,
      persona: 'You are XERXES.',
    })
    supervisor.attachSink({ send: (to, type) => { sent.push({ to, type }) } })
    void projection
    projection = projection
    return { supervisor, identity, scripted, sent }
  }

  test('open returns the shared thread; submit stores an attributed message, runs one turn and delivers the reply', async () => {
    const t = await configured()
    const ctx = { context: t.identity.context('alice'), grants: t.identity.grantsFor('alice') }
    const opened = await t.supervisor.handlers['supervisor.thread.open']!(ctx, {})
    expect(opened).toMatchObject({ thread_id: 'shared', kind: 'shared', messages: [], confirmations: [] })
    const replies: string[] = []
    t.supervisor.voice.onReply('shared', reply => { replies.push(reply.text) })
    const submitted = await t.supervisor.handlers['supervisor.message.submit']!(ctx, { message_id: 'm1', text: 'status?', source: 'text' })
    expect(submitted).toMatchObject({ thread_id: 'shared', duplicate: false })
    await t.supervisor.runner.enqueue({ threadId: 'shared', requester: t.supervisor.threads.principalOrStored('alice')! })
    expect(replies.length).toBeGreaterThan(0)
    const dup = await t.supervisor.handlers['supervisor.message.submit']!(ctx, { message_id: 'm1', text: 'status?' })
    expect(dup.duplicate).toBe(true)
    expect(t.sent.some(item => item.type === 'supervisor.thread_event')).toBe(true)
  })

  test('submitted shared-thread voice and text messages retain their own reply correlation through enqueue and wire events', async () => {
    const t = await configured()
    try {
      const events: Record<string, unknown>[] = []
      t.supervisor.attachSink({ send: (_to, type, payload) => {
        if (type === 'supervisor.thread_event' && typeof payload.message === 'object') events.push(payload.message as Record<string, unknown>)
      } })
      const ctx = (name: string) => ({ context: t.identity.context(name), grants: t.identity.grantsFor(name) })
      await Promise.all([
        t.supervisor.handlers['supervisor.message.submit']!(ctx('alice'), { message_id: 'voice-a', text: 'My question', source: 'voice' }),
        t.supervisor.handlers['supervisor.message.submit']!(ctx('owner'), { message_id: 'text-b', text: 'Another question', source: 'text' }),
      ])
      await until(() => !t.supervisor.runner.busy('shared'))
      expect(t.supervisor.threads.messagesFor('shared', 'alice').filter(message => message.role === 'xerxes').map(message => message.causalMessageId)).toEqual(['voice-a', 'text-b'])
      expect(events.filter(message => message.role === 'xerxes').map(message => message.causal_message_id)).toEqual(['voice-a', 'text-b'])
    } finally { t.supervisor.close() }
  })

  test('an agent principal cannot use the supervisor methods, and bad input is rejected', async () => {
    const t = await configured()
    const robot = { context: t.identity.context('robot'), grants: t.identity.grantsFor('robot') }
    for (const method of ['supervisor.thread.open', 'supervisor.message.submit'] as const) {
      await expect(Promise.resolve().then(() => t.supervisor.handlers[method]!(robot, { message_id: 'm', text: 'x' }))).rejects.toMatchObject({ code: 'forbidden' })
    }
    const alice = { context: t.identity.context('alice'), grants: [] }
    await expect(Promise.resolve().then(() => t.supervisor.handlers['supervisor.message.submit']!(alice, { message_id: 'm', text: '   ' }))).rejects.toMatchObject({ code: 'invalid_request' })
    await expect(Promise.resolve().then(() => t.supervisor.handlers['supervisor.thread.open']!(alice, { extra: 1 }))).rejects.toMatchObject({ code: 'invalid_request' })
    await expect(Promise.resolve().then(() => t.supervisor.handlers['supervisor.thread.open']!(alice, { thread_id: 'private-nope' }))).rejects.toMatchObject({ code: 'not_found' })
  })

  test('a person with no visible workspace cannot join the shared thread', async () => {
    const t = await configured()
    t.identity.add({ kind: 'user', id: 'nobody', displayName: 'NOBODY', deviceId: 'd', hubOwner: false }, [])
    const ctx = { context: t.identity.context('nobody'), grants: [] }
    await expect(Promise.resolve().then(() => t.supervisor.handlers['supervisor.thread.open']!(ctx, {}))).rejects.toMatchObject({ code: 'forbidden' })
  })

  test('the voice bridge submits once per key as a voice message, reports status, cancels, and refuses agents', async () => {
    const t = await configured()
    const actor: HubActor = { principal_kind: 'user', principal_id: 'alice', display_name: 'ALICE', device_id: 'dev-alice' }
    // alice must be known to the thread before a voice actor can act
    t.supervisor.threads.access('shared', t.identity.context('alice').principal as never)
    expect(await t.supervisor.voice.authorize('shared', actor)).toBe(true)
    const first = await t.supervisor.voice.submit('shared', 'what is blocked?', actor, 'k1')
    expect(first).toMatchObject({ accepted: true, duplicate: false })
    const second = await t.supervisor.voice.submit('shared', 'what is blocked?', actor, 'k1')
    expect(second).toMatchObject({ accepted: true, duplicate: true, messageId: first.messageId })
    const stored = t.supervisor.threads.messagesFor('shared', 'alice')
    expect(stored.filter(message => message.role === 'user')).toHaveLength(1)
    expect(stored[0]).toMatchObject({ source: 'voice' })
    expect((await t.supervisor.voice.status('shared')).summary).toContain('waiting for confirmation')
    expect(await t.supervisor.voice.cancel('shared', actor)).toHaveProperty('cancelled')
    const agent: HubActor = { principal_kind: 'agent', principal_id: 'robot', display_name: 'robot' }
    expect(await t.supervisor.voice.authorize('shared', agent)).toBe(false)
    expect(await t.supervisor.voice.submit('shared', 'hi', agent, 'k2')).toMatchObject({ accepted: false })
    expect(await t.supervisor.voice.submit('shared', '   ', actor, 'k3')).toMatchObject({ accepted: false })
  })
})
