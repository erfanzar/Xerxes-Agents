// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'

import { utf8ByteLength } from '../src/hub/protocol.js'
import { TAIL_MAX_BYTES, TURN_METADATA_KEY, renderTail } from '../src/hub/supervisor/tools.js'
import type { JsonObject } from '../src/types/toolCalls.js'
import { disposeAll } from './hubHarness.js'
import { makeWorld, toolCall, type World } from './hubSupervisorHarness.js'

afterEach(disposeAll)

async function run(world: World, name: string, args: JsonObject, who = 'alice'): Promise<Record<string, any>> {
  const state = { threadId: 'shared', requester: world.user(who), reader: world.reader(who), denial: undefined }
  return JSON.parse(await world.registry.execute(toolCall(name, args), { metadata: { [TURN_METADATA_KEY]: state } })) as Record<string, any>
}

describe('supervisor read tools', () => {
  test('tools refuse to run outside a supervisor turn', async () => {
    const world = makeWorld()
    await expect(world.registry.execute(toolCall('workspace_list'), { metadata: {} })).rejects.toThrow('supervisor turn')
  })

  test('reads are filtered by the thread ACL: a session-only guest sees one task and no workspace', async () => {
    const world = makeWorld()
    const listed = await run(world, 'tasks_list', {}, 'guest')
    expect(listed.tasks.map((task: { task_id: string }) => task.task_id)).toEqual(['s1'])
    expect((await run(world, 'workspace_list', {}, 'guest')).workspaces).toEqual([])
    await expect(run(world, 'task_digest', { workspace_id: 'w1', task_id: 's2' }, 'guest')).rejects.toThrow('cannot read')
    await expect(run(world, 'task_tail', { workspace_id: 'w1', task_id: 's2' }, 'guest')).rejects.toThrow('cannot read')
    await expect(run(world, 'task_search', { workspace_id: 'w1', task_id: 's2', query: 'x' }, 'guest')).rejects.toThrow('cannot read')
    expect(world.backendCalls).toHaveLength(0)
  })

  test('a person with a workspace grant sees the workspace and all of its tasks', async () => {
    const world = makeWorld()
    expect((await run(world, 'workspace_list', {})).workspaces).toMatchObject([{ id: 'w1', label: 'One', health: 'online' }])
    const listed = await run(world, 'tasks_list', { workspace_id: 'w1' })
    expect(listed.tasks).toHaveLength(2)
    expect(listed.tasks[0]).toHaveProperty('pending_human_count')
    await expect(run(world, 'tasks_list', { workspace_id: 'nope' })).rejects.toThrow('cannot read')
  })

  test('tasks_list pages with an opaque cursor', async () => {
    const world = makeWorld()
    const first = await run(world, 'tasks_list', { limit: 1 })
    expect(first.tasks).toHaveLength(1)
    expect(typeof first.next_cursor).toBe('string')
    const second = await run(world, 'tasks_list', { limit: 1, cursor: first.next_cursor })
    expect(second.tasks).toHaveLength(1)
    expect(second.tasks[0].task_id).not.toBe(first.tasks[0].task_id)
    expect(second.next_cursor).toBeUndefined()
  })

  test('task_tail is bounded to 8 KiB and keeps the newest steps', async () => {
    const world = makeWorld()
    const messages = Array.from({ length: 60 }, (_, index) => ({ role: 'assistant', content: `${index}:${'x'.repeat(600)}` }))
    world.backendBehavior.run = method => (method === 'session.active_list'
      ? { state: 'accepted', result: { ok: true, sessions: [{ id: 's1', key: 'k1' }] } }
      : { state: 'accepted', result: { ok: true, history: { actions: [{ messages }] } } })
    const tail = await run(world, 'task_tail', { workspace_id: 'w1', task_id: 's1' })
    expect(tail.truncated).toBe(true)
    expect(utf8ByteLength(JSON.stringify(tail.steps))).toBeLessThanOrEqual(TAIL_MAX_BYTES)
    expect(tail.steps.at(-1).startsWith('assistant: 59:')).toBe(true)
    const history = world.backendCalls.find(call => call.method === 'session.history')!
    expect(history.params).toMatchObject({ session_key: 'k1', history_limit: 10 })
    await expect(run(world, 'task_tail', { workspace_id: 'w1', task_id: 's1', after: 3 })).rejects.toThrow('not supported')
  })

  test('renderTail reports tool steps without arguments or output', () => {
    const tail = renderTail({ actions: [{ messages: [], executions: [{ name: 'ReadFile', arguments: { path: '/secret' }, result: 'TOP SECRET' }, { toolName: 'Grep', is_error: true }] }] }, 1024)
    expect(tail.lines).toEqual(['tool ReadFile', 'tool Grep (failed)'])
  })

  test('task_search is scoped to one task and drops hits from other sessions', async () => {
    const world = makeWorld()
    world.backendBehavior.run = () => ({
      state: 'accepted',
      result: { ok: true, hits: [{ session_id: 's1', text: 'mine' }, { session_id: 'other', text: 'leak' }, { text: 'no session field' }] },
    })
    const found = await run(world, 'task_search', { workspace_id: 'w1', task_id: 's1', query: 'mine' })
    expect(found.hits).toEqual([{ session_id: 's1', text: 'mine' }, { text: 'no session field' }])
    expect(world.backendCalls[0]!.params).toMatchObject({ session_id: 's1', query: 'mine', limit: 5 })
  })

  test('task_search strips argument-like fields from hits at any depth before they reach the model', async () => {
    const world = makeWorld()
    world.backendBehavior.run = () => ({
      state: 'accepted',
      result: { ok: true, hits: [{
        session_id: 's1', text: 'ran a tool', arguments: '{"cmd":"rm -rf secret"}', input: { path: 'secret' }, inputs: { a: 1 },
        tool_input: { x: 1 }, params: { y: 2 }, nested: [{ tool: 'Bash', arguments: { cmd: 'leak' }, note: 'kept' }],
      }] },
    })
    const found = await run(world, 'task_search', { workspace_id: 'w1', task_id: 's1', query: 'tool' })
    expect(found.hits).toEqual([{ session_id: 's1', text: 'ran a tool', nested: [{ tool: 'Bash', note: 'kept' }] }])
    expect(JSON.stringify(found)).not.toContain('secret')
    expect(JSON.stringify(found)).not.toContain('leak')
  })

  test('an unavailable workspace surfaces as a tool error, not an invented result', async () => {
    const world = makeWorld()
    world.backendBehavior.run = () => ({ state: 'rejected', error: { code: 'unavailable', message: 'Workspace is not online' } })
    await expect(run(world, 'task_search', { workspace_id: 'w1', task_id: 's1', query: 'x' })).rejects.toThrow('not online')
  })

  test('arguments are validated with the schema bounds', async () => {
    const world = makeWorld()
    await expect(run(world, 'task_search', { workspace_id: 'w1', task_id: 's1', query: 'x'.repeat(300) })).rejects.toThrow()
    await expect(run(world, 'tasks_list', { limit: 500 })).rejects.toThrow()
    await expect(run(world, 'workspace_list', { extra: 1 })).rejects.toThrow()
  })
})

describe('notify.user', () => {
  test('reaches thread participants only', async () => {
    const world = makeWorld()
    world.reader('owner')
    const out = await run(world, 'notify_user', { recipients: ['alice', 'mallory'], text: 'Heads up', severity: 'info' })
    expect(out.delivered_to).toEqual(['alice'])
    expect(out.skipped_not_in_thread).toEqual(['mallory'])
    const last = world.events.at(-1)!
    expect(last.type).toBe('supervisor.thread_event')
    expect(last.to).toEqual(['alice'])
    // The notice is not visible to anyone else reading the thread.
    expect(world.threads.messagesFor('shared', 'owner').some(message => message.text.includes('Heads up'))).toBe(false)
    expect(world.threads.messagesFor('shared', 'alice').some(message => message.text.includes('Heads up'))).toBe(true)
  })

  test('refuses when nobody named takes part, and resolves people by display name', async () => {
    const world = makeWorld()
    await expect(run(world, 'notify_user', { recipients: ['mallory'], text: 'x', severity: 'info' })).rejects.toThrow('participants')
    const byName = await run(world, 'notify_user', { recipients: ['ALICE'], text: 'hello', severity: 'attention' })
    expect(byName.delivered_to).toEqual(['alice'])
  })

  test('in a private thread the only audience is its owner', async () => {
    const world = makeWorld()
    world.reader('owner')
    const thread = world.threads.privateFor(world.user('alice'))
    const state = { threadId: thread.id, requester: world.user('alice'), reader: world.reader('alice', thread.id), denial: undefined }
    const out = JSON.parse(await world.registry.execute(toolCall('notify_user', { recipients: ['alice', 'owner'], text: 'psst', severity: 'info' }), { metadata: { [TURN_METADATA_KEY]: state } }))
    expect(out.delivered_to).toEqual(['alice'])
    expect(world.events.at(-1)!.to).toEqual(['alice'])
  })
})
