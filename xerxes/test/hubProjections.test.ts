// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { aggregateUsage, emptyProjection, markProjectionInterrupted, markProjectionStale, reduceProjectionEvent, replaceSessionSummaries } from '../src/hub/projections.js'
import type { UpstreamEvent } from '../src/hub/ports.js'

const time = '2026-10-21T10:00:00.000Z'
function event(type: string, payload: Record<string, unknown> = {}, sequence = 1): UpstreamEvent {
  return { event_id: `e${sequence}`, sequence, cursor: `c${sequence}`, runtime_id: 'r', instance_id: 'i', project_dir: '/a', session_id: 's', type, payload, occurred_at: time }
}

test('summary projection preserves source timestamps, goals, agents and interaction state without transcripts', () => {
  const state = replaceSessionSummaries(emptyProjection('w'), [{ id: 's', title: 'Task', status: 'running', model: 'm', updated_at: time, goal: { objective: 'Ship' }, pending_interactions: [{ type: 'permission' }], subagents: [{ status: 'running' }, { status: 'done' }], history: ['must not retain'] }], time)
  expect(state.sessions.s).toMatchObject({ id: 's', title: 'Task', status: 'needs_you', model: 'm', goal_summary: 'Ship', pending_interaction_count: 1, agent_counts: { total: 2, active: 1 }, source_timestamp: time, stale: false })
  expect(state.sessions.s).not.toHaveProperty('history')
})

test('event reducer handles working, needs-you, resolution, failure and duplicate delivery immutably', () => {
  const original = replaceSessionSummaries(emptyProjection('w'), [{ id: 's', title: 'Task', status: 'idle' }], time)
  const working = reduceProjectionEvent(original, event('turn_begin'))
  expect(working.sessions.s?.status).toBe('working')
  expect(original.sessions.s?.status).toBe('idle')
  const pending = reduceProjectionEvent(working, event('permission_request', { request_id: 'p' }, 2))
  expect(pending.sessions.s?.status).toBe('needs_you')
  expect(reduceProjectionEvent(pending, event('permission_request', { request_id: 'p' }, 2))).toBe(pending)
  const resolved = reduceProjectionEvent(pending, event('permission_resolved', { request_id: 'p' }, 3))
  expect(resolved.sessions.s?.pending_interaction_count).toBe(0)
  expect(resolved.sessions.s?.status).toBe('working')
  expect(reduceProjectionEvent(resolved, event('turn_end', { error: 'failed' }, 4)).sessions.s?.status).toBe('failed')
})

test('stale, restart interruption, old summaries and project scoping do not invent a healthy state', () => {
  const state = replaceSessionSummaries(emptyProjection('w'), [{ id: 's', status: 'working', updated_at: time }], time)
  expect(markProjectionStale(state).sessions.s?.stale).toBe(true)
  const interrupted = markProjectionInterrupted(state)
  expect(interrupted.sessions.s).toMatchObject({ status: 'unknown', stale: true, interrupted: true })
  expect(replaceSessionSummaries(state, [{ id: 's', status: 'idle', updated_at: '2020-01-01T00:00:00Z' }], time).sessions.s?.status).toBe('working')
  expect(reduceProjectionEvent(state, { ...event('turn_end'), project_dir: '/other' }, '/a')).toBe(state)
})

test('usage counts own parent/child counters once and separates plan price estimates from bills', () => {
  const total = aggregateUsage([
    { id: 'parent', input: 100, output: 10, cost_usd: 2, includes_children: true },
    { id: 'child', parent_session_id: 'parent', input: 20, output: 2, cost_usd: 1 },
    { id: 'plan', input: 50, output: 5, cost_usd: 9, cost_basis: 'subscription' },
    { id: 'plan', input: 50, output: 5, cost_usd: 9, cost_basis: 'subscription' },
  ])
  expect(total).toMatchObject({ input: 150, output: 15, billable_cost_usd: 2, subscription_price_estimate_usd: 9, subscription_label: 'Included in plan' })
  expect(aggregateUsage([{ id: 'p', input: 100 }, { id: 'c', parent_session_id: 'p', input: 20 }]).input).toBe(120)
  expect(aggregateUsage([{ id: 'unknown', input: 1 }]).billable_cost_usd).toBeUndefined()
})

test('usage prefers explicit own counters when a parent also reports rolled-up counters', () => {
  expect(aggregateUsage([{ id: 'p', input: 120, own_usage: { input: 100, cost_usd: 2 } }, { id: 'c', parent_session_id: 'p', input: 20, cost_usd: 1 }])).toMatchObject({ input: 120, billable_cost_usd: 3 })
})

test('native response, goal and subagent snapshot events update the summary and preserve pending interactions on polling', () => {
  let state = replaceSessionSummaries(emptyProjection('w'), [{ id: 's', status: 'working' }], time)
  state = reduceProjectionEvent(state, event('question_request', { id: 'q' }, 1))
  state = replaceSessionSummaries(state, [{ id: 's', status: 'running' }], time)
  expect(state.sessions.s?.pending_interaction_count).toBe(1)
  expect(state.sessions.s?.status).toBe('needs_you')
  state = reduceProjectionEvent(state, event('question_response', { id: 'q' }, 2))
  expect(state.sessions.s?.pending_interaction_count).toBe(0)
  state = reduceProjectionEvent(state, event('session_summary', { title: 'New', goal: { objective: 'Verify' }, subagent_snapshots: [{ status: 'running' }, { status: 'completed' }] }, 3))
  expect(state.sessions.s).toMatchObject({ title: 'New', goal_summary: 'Verify', agent_counts: { total: 2, active: 1 } })
  state = reduceProjectionEvent(state, event('session_deleted', {}, 4))
  expect(state.sessions.s).toBeUndefined()
})

test('duplicate interaction requests and unrelated resolution events do not decrement the wrong pending card', () => {
  let state = replaceSessionSummaries(emptyProjection('w'), [{ id: 's', status: 'working' }], time)
  state = reduceProjectionEvent(state, event('permission_request', { request_id: 'p' }, 1))
  state = reduceProjectionEvent(state, event('permission_request', { request_id: 'p' }, 2))
  state = reduceProjectionEvent(state, event('permission_response', { request_id: 'other' }, 3))
  expect(state.sessions.s?.pending_interaction_count).toBe(1)
  expect(state.sessions.s?.status).toBe('needs_you')
})

test('an empty status is idle unless a turn runs, and an empty title falls back to the user message', () => {
  const state = replaceSessionSummaries(emptyProjection('w'), [
    { id: 'a', title: '', preview: 'fix the build', status: '' },
    { id: 'b', status: 'ready', first_user_message: 'hello there' },
    { id: 'c', last_user_message: 'last words', active_turn_id: 't1' },
    { id: 'd', status: 'error' },
    { id: 'e' },
  ], time)
  expect(state.sessions.a).toMatchObject({ title: 'fix the build', status: 'idle' })
  expect(state.sessions.b).toMatchObject({ title: 'hello there', status: 'idle' })
  expect(state.sessions.c).toMatchObject({ title: 'last words', status: 'working' })
  expect(state.sessions.d?.status).toBe('failed')
  expect(state.sessions.e).toMatchObject({ title: 'Untitled task', status: 'idle' })
})
