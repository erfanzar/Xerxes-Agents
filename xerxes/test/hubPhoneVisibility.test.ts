// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'

import { emptyProjection, reduceProjectionEvent, replaceSessionSummaries } from '../src/hub/projections.js'
import type { UpstreamEvent } from '../src/hub/ports.js'
import { agentStepsFromEvents } from '../src/hub/server.js'
import { disposeAll, startFixture, tempDir, type Fixture } from './hubHarness.js'

afterEach(disposeAll)

const rows = (value: unknown): Array<Record<string, unknown>> => value as Array<Record<string, unknown>>
const APPROVAL = { id: 'p1', request_id: 'p1', tool_call_id: 't1', session_id: 's1', tool_name: 'exec_command', description: 'run tests', inputs: { cmd: 'bun test' } }
const QUESTION = { id: 'q1', tool_call_id: 't2', session_id: 's1', questions: [{ id: 'answer', question: 'Which build?', options: ['nightly', 'release'] }] }

function seedInteractions(f: Fixture): void {
  f.store.append({ workspace_id: 'w1', session_id: 's1', type: 'turn_begin', payload: {} })
  f.store.append({ workspace_id: 'w1', session_id: 's1', type: 'approval_request', payload: APPROVAL })
  f.store.append({ workspace_id: 'w1', session_id: 's1', type: 'question_request', payload: QUESTION })
}
function interactions(events: unknown): Record<string, Record<string, unknown>> {
  return Object.fromEntries(rows(events).filter(event => event.type === 'approval_request' || event.type === 'question_request')
    .map(event => [String(event.type), event.payload as Record<string, unknown>]))
}

describe('Hub: interaction payloads are redacted per recipient', () => {
  test('events.read: owner and operator see the command and question; a viewer sees only that something waits', async () => {
    const f = await startFixture()
    seedInteractions(f)
    for (const who of ['owner', 'alice']) {
      const seen = interactions((await (await f.client(who)).ok('hub.events.read', { workspace_ids: ['w1'] })).events)
      expect(seen.approval_request).toMatchObject({ tool_name: 'exec_command', inputs: { cmd: 'bun test' } })
      expect(seen.question_request).toMatchObject({ questions: [{ question: 'Which build?', options: ['nightly', 'release'] }] })
    }
    for (const who of ['viewer', 'guest', 'robot']) {
      const seen = interactions((await (await f.client(who)).ok('hub.events.read', { workspace_ids: ['w1'] })).events)
      expect(seen.approval_request).toMatchObject({ id: 'p1', request_id: 'p1', redacted: true })
      expect(JSON.stringify(seen)).not.toContain('bun test')
      expect(JSON.stringify(seen)).not.toContain('Which build?')
      expect(seen.question_request).toMatchObject({ id: 'q1', questions: [], redacted: true })
    }
  })

  test('live subscription batches are redacted per socket', async () => {
    const f = await startFixture()
    const alice = await f.client('alice')
    const viewer = await f.client('viewer')
    await alice.ok('hub.events.subscribe', { workspace_ids: ['w1'] })
    await viewer.ok('hub.events.subscribe', { workspace_ids: ['w1'] })
    seedInteractions(f)
    const batch = (client: Awaited<ReturnType<Fixture['client']>>) =>
      client.event('hub.event_batch', payload => rows(payload.events).some(event => event.type === 'question_request'))
    const full = interactions((await batch(alice)).payload.events)
    const redacted = interactions((await batch(viewer)).payload.events)
    expect(full.approval_request).toMatchObject({ inputs: { cmd: 'bun test' } })
    expect(redacted.approval_request).toMatchObject({ redacted: true })
    expect(JSON.stringify(redacted)).not.toContain('bun test')
  })

  test('history carries the pending cards (resolved and ended ones dropped), redacted for viewers', async () => {
    const f = await startFixture()
    seedInteractions(f)
    f.store.append({ workspace_id: 'w1', session_id: 's1', type: 'approval_request', payload: { ...APPROVAL, id: 'p0', request_id: 'p0' } })
    f.store.append({ workspace_id: 'w1', session_id: 's1', type: 'approval_response', payload: { request_id: 'p0' } })
    const page = await (await f.client('alice')).ok('hub.history', { workspace_id: 'w1', session_id: 's1' })
    expect(page.pending_complete).toBe(true)
    expect(rows(page.pending_interactions).map(frame => frame.type)).toEqual(['approval_request', 'question_request'])
    expect(rows(page.pending_interactions)[0]!.payload).toMatchObject({ request_id: 'p1', inputs: { cmd: 'bun test' } })
    const viewerPage = await (await f.client('viewer')).ok('hub.history', { workspace_id: 'w1', session_id: 's1' })
    expect(rows(viewerPage.pending_interactions).map(frame => frame.payload)).toEqual([
      expect.objectContaining({ request_id: 'p1', redacted: true }), expect.objectContaining({ id: 'q1', questions: [], redacted: true }),
    ])
    // An older page never repeats the cards; a turn end clears them.
    expect(await (await f.client('alice')).ok('hub.history', { workspace_id: 'w1', session_id: 's1', before: 'x' })).not.toHaveProperty('pending_interactions')
    f.store.append({ workspace_id: 'w1', session_id: 's1', type: 'turn_end', payload: {} })
    expect((await (await f.client('alice')).ok('hub.history', { workspace_id: 'w1', session_id: 's1' })).pending_interactions).toEqual([])
  })

  test('snapshot attaches pending cards to tasks that need a person', async () => {
    const f = await startFixture()
    seedInteractions(f)
    f.projections.set('w1', replaceSessionSummaries(emptyProjection('w1'), [
      { id: 's1', title: 'First task', status: 'waiting', pending_interaction_count: 2 }, { id: 's2', title: 'Second task', status: 'working' },
    ], new Date().toISOString()))
    const session = (snapshot: Record<string, unknown>, id: string) => rows(rows(snapshot.workspaces)[0]!.sessions).find(row => row.id === id)!
    const owner = await f.client('owner')
    const own = await owner.ok('hub.snapshot', { workspace_ids: ['w1'] })
    expect(rows(session(own, 's1').pending_interactions).map(frame => frame.payload)).toEqual([
      expect.objectContaining({ inputs: { cmd: 'bun test' } }), expect.objectContaining({ questions: [expect.objectContaining({ question: 'Which build?' })] }),
    ])
    expect(session(own, 's2')).not.toHaveProperty('pending_interactions')
    const guest = await (await f.client('guest')).ok('hub.snapshot', { workspace_ids: ['w1'] })
    expect(JSON.stringify(session(guest, 's1'))).not.toContain('bun test')
    expect(rows(session(guest, 's1').pending_interactions)[0]!.payload).toMatchObject({ redacted: true })
  })
})

describe('Hub: agents.list and agents.inspect', () => {
  const snapshots = [
    { id: 'a1', status: 'completed', name: 'researcher', title: 'Read the docs', model: 'm1', tool_count: 2, summary: 'done', created_at: '2026-10-29T10:00:00Z', updated_at: '2026-10-29T10:05:00Z', prompt_profile: 'secret-profile' },
    { id: 'a2', status: 'running', name: 'coder' },
    { status: 'running' },
  ]
  function withAgents(f: Fixture): void {
    f.backend.handler = call => {
      if (call.method === 'session.active_list') return { state: 'accepted', result: { ok: true, sessions: [{ id: 's1', key: 'k1', subagent_snapshots: snapshots }, { id: 's2', key: 'k2' }] } }
      if (call.method === 'subagent.inspect') {
        return call.params.task === 'a1'
          ? { state: 'accepted', result: { ok: true, agent: { id: 'a1', prompt: 'Summarise README', output: 'Final: it is a Bun runtime.' } } }
          : { state: 'accepted', result: { ok: false, error: 'not available' } }
      }
      return undefined
    }
    const sub = (event: Record<string, unknown>, agent = 'a1') => f.store.append({ workspace_id: 'w1', session_id: 's1', type: 'subagent_event', payload: { agent_id: agent, event } })
    sub({ type: 'turn_begin', payload: { status: 'running' } })
    sub({ type: 'think_part', payload: { think: 'plan' } })
    sub({ type: 'tool_call', payload: { id: 'c1', name: 'ReadFile', arguments: '{"file_path":"README.md"}' } })
    sub({ type: 'tool_result', payload: { tool_call_id: 'c1', name: 'ReadFile', return_value: '# Xerxes', permitted: true } })
    sub({ type: 'tool_call', payload: { id: 'c2', name: 'ExecuteShell', arguments: '{"cmd":"false"}' } })
    sub({ type: 'tool_result', payload: { tool_call_id: 'c2', name: 'ExecuteShell', return_value: 'Tool execution failed: exit 1', permitted: true } })
    sub({ type: 'text_part', payload: { text: 'Final: it is ' } })
    sub({ type: 'text_part', payload: { text: 'a Bun runtime.' } })
    sub({ type: 'turn_end', payload: { status: 'completed', summary: 'done' } })
    sub({ type: 'text_part', payload: { text: 'other agent text' } }, 'a2')
  }

  test('agents.list returns the contract rows from the daemon snapshots, with access checks', async () => {
    const f = await startFixture()
    withAgents(f)
    const list = await (await f.client('guest')).ok('agents.list', { workspace_id: 'w1', session_id: 's1' })
    expect(list.agents).toEqual([
      { id: 'a1', name: 'researcher', title: 'Read the docs', status: 'completed', model: 'm1', started_at: '2026-10-29T10:00:00Z', updated_at: '2026-10-29T10:05:00Z', tool_calls: 2, summary: 'done' },
      { id: 'a2', name: 'coder', title: 'coder', status: 'running' },
    ])
    expect(await (await f.client('guest')).fails('agents.list', { workspace_id: 'w1', session_id: 's2' })).toBe('forbidden')
    expect(await (await f.client('alice')).fails('agents.list', { workspace_id: 'w1', session_id: 'ghost' })).toBe('not_found')
    expect(await (await f.client('alice')).fails('agents.list', { workspace_id: 'w1', session_id: 's1', extra: 1 })).toBe('invalid_request')
    expect((await (await f.client('alice')).ok('agents.list', { workspace_id: 'w1', session_id: 's2' })).agents).toEqual([])
  })

  test('agents.inspect returns the agent steps with exact args and outputs, bounded by limit', async () => {
    const f = await startFixture()
    withAgents(f)
    const alice = await f.client('alice')
    const detail = await alice.ok('agents.inspect', { workspace_id: 'w1', session_id: 's1', agent_id: 'a1' })
    expect(detail.agent).toMatchObject({ id: 'a1', status: 'completed' })
    expect(detail.truncated).toBe(false)
    expect(detail.steps).toEqual([
      { kind: 'notice', text: 'Task: Summarise README' },
      { kind: 'notice', text: 'started', state: 'working' },
      { kind: 'thinking', text: 'plan' },
      { kind: 'tool', name: 'ReadFile', args: '{"file_path":"README.md"}', output: '# Xerxes', state: 'done' },
      { kind: 'tool', name: 'ExecuteShell', args: '{"cmd":"false"}', output: 'Tool execution failed: exit 1', state: 'error' },
      { kind: 'text', text: 'Final: it is a Bun runtime.' },
      { kind: 'notice', text: 'completed: done', state: 'done' },
    ])
    expect(f.backend.callsTo('subagent.inspect')[0]!.params).toEqual({ session_key: 'k1', task: 'a1' })
    const limited = await alice.ok('agents.inspect', { workspace_id: 'w1', session_id: 's1', agent_id: 'a1', limit: 2 })
    expect(rows(limited.steps).map(step => step.kind)).toEqual(['text', 'notice'])
    expect(limited.truncated).toBe(true)
    // A running agent whose daemon detail is unavailable still shows its journal steps.
    expect(rows((await alice.ok('agents.inspect', { workspace_id: 'w1', session_id: 's1', agent_id: 'a2' })).steps)).toEqual([{ kind: 'text', text: 'other agent text' }])
    expect(await alice.fails('agents.inspect', { workspace_id: 'w1', session_id: 's1', agent_id: 'zz' })).toBe('not_found')
    expect(await alice.fails('agents.inspect', { workspace_id: 'w1', session_id: 's1', agent_id: 'a1', limit: 201 })).toBe('invalid_request')
    expect(await (await f.client('guest')).fails('agents.inspect', { workspace_id: 'w1', session_id: 's2', agent_id: 'a1' })).toBe('forbidden')
  })

  test('a huge agent detail shrinks to fit one frame instead of failing', async () => {
    const f = await startFixture()
    withAgents(f)
    for (let index = 0; index < 60; index += 1) {
      f.store.append({ workspace_id: 'w1', session_id: 's1', type: 'subagent_event', payload: { agent_id: 'a2', event: { type: 'tool_call', payload: { id: `x${index}`, name: 'Write', arguments: 'y'.repeat(20_000) } } } })
    }
    const detail = await (await f.client('alice')).ok('agents.inspect', { workspace_id: 'w1', session_id: 's1', agent_id: 'a2', limit: 200 })
    expect(JSON.stringify(detail).length).toBeLessThanOrEqual(256 * 1024)
    expect(detail.truncated).toBe(true)
    expect(rows(detail.steps).length).toBeGreaterThan(0)
  })

  test('agent steps are built only from the named agent', () => {
    expect(agentStepsFromEvents([{ type: 'subagent_event', payload: { agent_id: 'b', event: { type: 'text_part', payload: { text: 'x' } } } }], 'a')).toEqual([])
  })
})

describe('Hub: projections and caching', () => {
  test('S7: the daemon\'s approval_request / approval_response events drive needs_you', () => {
    const event = (type: string, payload: Record<string, unknown>, sequence: number): UpstreamEvent =>
      ({ event_id: `e${sequence}`, sequence, cursor: `c${sequence}`, runtime_id: 'r', instance_id: 'i', project_dir: '/a', session_id: 's', type, payload, occurred_at: new Date(1_800_000_000_000 + sequence).toISOString() })
    let state = replaceSessionSummaries(emptyProjection('w'), [{ id: 's', status: 'working', updated_at: new Date(1_800_000_000_000).toISOString() }], new Date(1_800_000_000_000).toISOString())
    state = reduceProjectionEvent(state, event('approval_request', { id: 'p', request_id: 'p' }, 1))
    expect(state.sessions.s).toMatchObject({ status: 'needs_you', pending_interaction_count: 1 })
    state = reduceProjectionEvent(state, event('approval_response', { request_id: 'p' }, 2))
    expect(state.sessions.s).toMatchObject({ status: 'working', pending_interaction_count: 0 })
  })

  test('V14: only content-hashed /assets/ files are immutable; ONNX runtime files revalidate', async () => {
    const dir = await tempDir('xh-cache-')
    const root = join(dir, 'public')
    await mkdir(join(root, 'assets'), { recursive: true })
    await mkdir(join(root, 'voice-ort'), { recursive: true })
    await Bun.write(join(root, 'index.html'), '<!doctype html>')
    await Bun.write(join(root, 'assets', 'main-abcdef12.js'), '1')
    await Bun.write(join(root, 'voice-ort', 'ort-wasm-simd-threaded.mjs'), '2')
    await Bun.write(join(root, 'voice-ort', 'ort-wasm-simd-threaded.jsep.mjs'), '3')
    await Bun.write(join(root, 'voice-call-worker.js'), '4')
    const f = await startFixture({ staticRoot: root })
    expect((await fetch(f.url('/assets/main-abcdef12.js'))).headers.get('cache-control')).toBe('public, max-age=31536000, immutable')
    for (const path of ['/voice-ort/ort-wasm-simd-threaded.mjs', '/voice-ort/ort-wasm-simd-threaded.jsep.mjs', '/voice-call-worker.js']) {
      expect((await fetch(f.url(path))).headers.get('cache-control')).toBe('no-cache')
    }
  })
})
