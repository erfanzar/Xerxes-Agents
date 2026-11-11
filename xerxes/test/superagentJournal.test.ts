// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { Database } from 'bun:sqlite'
import { join } from 'node:path'
import { HubEventStore } from '../src/hub/eventStore.js'
import { createSupervisorToolRegistry, TURN_METADATA_KEY } from '../src/hub/supervisor/tools.js'
import { createHubSupervisor } from '../src/hub/supervisor/index.js'
import { disposeAll, registerCleanup, tempDir } from './hubHarness.js'
import { makeWorld, scriptedModel, toolCall } from './hubSupervisorHarness.js'

const scope = { workspaceIds: ['w1'], sessionIds: ['s1'] }
afterEach(disposeAll)
function journal(options: ConstructorParameters<typeof HubEventStore>[1] = {}) {
  const value = new HubEventStore(':memory:', options)
  registerCleanup(() => value.close())
  return value
}
function append(value: HubEventStore, text: string, sessionId = 's1') {
  return value.append({ workspace_id: 'w1', session_id: sessionId, type: 'tool_result', payload: { name: 'exec_command', result: text } })
}

test('retained journal reverse pages are stable across new appends and scoped to one task', () => {
  const value = journal()
  append(value, 'first'); append(value, 'foreign', 's2'); append(value, 'second'); append(value, 'third')
  const latest = value.readLatest(undefined, scope, 2)
  expect(latest.events.map(row => row.payload.result)).toEqual(['second', 'third'])
  expect(latest.hasMore).toBe(true); expect(latest.before).toBeString()
  append(value, 'newest')
  const older = value.readLatest(latest.before, scope, 2)
  expect(older.events.map(row => row.payload.result)).toEqual(['first'])
  expect(older.hasMore).toBe(false)
  expect(() => value.readLatest(latest.before, { workspaceIds: ['w1'], sessionIds: ['s2'] })).toThrow('scope')
  const other = journal()
  expect(() => other.readLatest(latest.before, scope)).toThrow('stream')
})

test('reverse page byte caps include multibyte metadata and oversized payloads never wedge paging', () => {
  const value = journal()
  append(value, '界'.repeat(200_000)); append(value, 'small')
  const page = value.readLatest(undefined, scope, 20, 2048)
  expect(Buffer.byteLength(JSON.stringify(page.events))).toBeLessThanOrEqual(2048)
  expect(page.events[0]?.payload.truncated).toBe(true)
  expect(page.events[1]?.payload.result).toBe('small')
  expect(() => value.readLatest(undefined, scope, 0)).toThrow()
  expect(() => value.readLatest(undefined, scope, 20, 1)).toThrow()
})

test('a terminal event cursor in an unpruned stream ends normally, not as a retention error', () => {
  const value = journal(); append(value, 'only')
  const cursor = value.readLatest(undefined, scope).events[0]!.cursor
  expect(value.readLatest(cursor, scope)).toMatchObject({ events: [], hasMore: false, retentionGap: false })
})

test('task reverse-page metadata uses a covering index without payload or actor table reads', async () => {
  const path = join(await tempDir('superagent-journal-plan-'), 'events.sqlite')
  const value = new HubEventStore(path); registerCleanup(() => value.close())
  append(value, 'large'.repeat(100_000))
  const probe = new Database(path, { readonly: true }); registerCleanup(() => probe.close())
  const plan = probe.query<{ detail: string }, []>(`EXPLAIN QUERY PLAN SELECT sequence, event_id, workspace_id, session_id, type, occurred_at, bytes
    FROM events WHERE sequence < 100 AND workspace_id = 'w1' AND session_id = 's1' ORDER BY sequence DESC LIMIT 2`).all()
  expect(plan.some(row => row.detail.includes('COVERING INDEX events_workspace_session_metadata'))).toBe(true)
})

test('retention is visible and a cursor into expired history refuses instead of silently ending', () => {
  let now = 1000
  const value = journal({ clock: () => now, retention: { maxBytes: 1_000_000, maxAgeMs: 10 } })
  append(value, 'first'); append(value, 'second')
  const before = value.readLatest(undefined, scope, 1).before
  now += 1000; append(value, 'current')
  expect(value.readLatest(undefined, scope).retentionGap).toBe(true)
  expect(() => value.readLatest(before, scope)).toThrow('retention')
})

test('retained records survive reopening without a daemon or a live session key', async () => {
  const path = join(await tempDir('superagent-journal-'), 'events.sqlite')
  const first = new HubEventStore(path); append(first, 'FAIL archived test'); first.close()
  const restored = new HubEventStore(path); registerCleanup(() => restored.close())
  expect(restored.readLatest(undefined, scope).events[0]?.payload.result).toBe('FAIL archived test')
})

function tools(value: HubEventStore) {
  const world = makeWorld()
  world.backendBehavior.run = () => { throw new Error('offline: daemon calls are forbidden') }
  const registry = createSupervisorToolRegistry({ ...world, journal: value })
  const execute = (args: Record<string, string | number>, who = 'alice', signal?: AbortSignal) => registry.execute(toolCall('task_records', {
    workspace_id: 'w1', task_id: 's1', ...args,
  }), { metadata: { [TURN_METADATA_KEY]: { threadId: 'shared', requester: world.user(who), reader: world.reader(who), denial: undefined } } }, signal)
  return { world, execute }
}

test('offline result and literal search reads retain exact failures, explicit coverage and bounded replies', async () => {
  const value = journal()
  append(value, 'FAIL regression build output'); append(value, 'PASS unrelated')
  const { world, execute } = tools(value)
  const raw = await execute({ query: 'regression', kind: 'results' })
  expect(raw).toContain('FAIL regression build output')
  expect(raw).not.toContain('PASS unrelated')
  expect(JSON.parse(raw)).toMatchObject({ source: 'retained_hub_journal', coverage: 'retained_only', has_more: false })
  expect(Buffer.byteLength(raw)).toBeLessThanOrEqual(64 * 1024)
  expect(world.backendCalls).toHaveLength(0)
})

test('a query with oversized records exposes incomplete coverage instead of claiming no matches', async () => {
  const value = journal(); append(value, 'failure'.repeat(20_000))
  const { execute } = tools(value)
  expect(JSON.parse(await execute({ query: 'failure' }))).toMatchObject({ truncated_records: 1, records: [] })
})

test('filtered empty pages still advance and the full response is bounded for multibyte output', async () => {
  const value = journal(); append(value, 'wanted')
  for (let i = 0; i < 8; i++) append(value, '界'.repeat(4000))
  const { execute } = tools(value)
  const raw = await execute({ limit: 2, query: 'wanted' }); const first = JSON.parse(raw)
  expect(first).toMatchObject({ records: [], has_more: true })
  expect(first.before).toBeString()
  expect(Buffer.byteLength(await execute({}))).toBeLessThanOrEqual(64 * 1024)
})

test('production supervisor composition gives the model retained records without daemon reads', async () => {
  const value = journal(); append(value, 'FAIL production-wired result')
  const world = makeWorld(); const dir = await tempDir('superagent-journal-composition-')
  const fake = scriptedModel([{ text: '', toolCalls: [toolCall('task_records', { workspace_id: 'w1', task_id: 's1' })] }, { text: 'finished', toolCalls: [] }])
  const supervisor = createHubSupervisor({ databasePath: join(dir, 'hub.sqlite'), journal: value, model: fake.model,
    identity: world.identity, overlay: world.overlay, workspaces: world.workspaces, backend: world.backend, execute: async () => { throw new Error('read must not mutate') },
    clock: world.clock, isHubOwner: id => id === 'owner', autopilot: { checkInMs: 0 }, persona: 'Offline fixture' })
  registerCleanup(() => supervisor.close())
  const owner = world.user('owner'); const thread = supervisor.threads.privateFor(owner)
  expect((await supervisor.runner.enqueue({ threadId: thread.id, requester: owner })).stop).toBe('completed')
  expect(JSON.stringify(fake.requests)).toContain('FAIL production-wired result')
  expect(world.backendCalls).toHaveLength(0)
})

test('record reads filter permission payloads, recheck ACLs and stop before reading on cancellation', async () => {
  const value = journal()
  append(value, 'visible')
  value.append({ workspace_id: 'w1', session_id: 's1', type: 'approval_request', payload: { inputs: 'private approval material' } })
  const { execute } = tools(value)
  expect(await execute({})).not.toContain('private approval material')
  await expect(execute({}, 'guest')).resolves.toContain('visible')
  await expect(execute({ task_id: 's2' }, 'guest')).rejects.toThrow()
  const cancelled = new AbortController(); cancelled.abort()
  await expect(execute({}, 'alice', cancelled.signal)).rejects.toThrow()
})
