// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'

import { HubEventStore } from '../src/hub/eventStore.js'
import { PendingInteractionIndex, interactionPayloadFor, isInteractionRequestEvent, redactEventsFor } from '../src/hub/interactions.js'

const stores: HubEventStore[] = []
afterEach(() => { for (const store of stores.splice(0)) store.close() })

function store(): HubEventStore {
  const value = new HubEventStore(':memory:', { clock: () => Date.parse('2026-10-30T12:00:00Z') })
  stores.push(value)
  return value
}
const add = (s: HubEventStore, type: string, payload: Record<string, unknown> = {}, session = 's1') =>
  s.append({ workspace_id: 'w1', session_id: session, type, payload })
const approval = (id: string, extra: Record<string, unknown> = {}) => ({ id, request_id: id, tool_name: 'exec_command', inputs: { cmd: 'bun test' }, ...extra })

describe('PendingInteractionIndex completeness (P2-4)', () => {
  test('a long-lived journal is complete once the scan saw a turn boundary of the task', () => {
    const s = store()
    for (let index = 0; index < 30; index += 1) add(s, 'notification', { message: `noise ${index}` }, 'other')
    add(s, 'turn_begin')
    add(s, 'approval_request', approval('p1'))
    const index = new PendingInteractionIndex(s, { windowEvents: 5 })
    const found = index.frames('w1', 's1')
    expect(found.frames.map(frame => frame.payload.request_id)).toEqual(['p1'])
    expect(found.complete).toBe(true)
  })

  test('without a turn boundary in a window that misses the journal start, the view is incomplete', () => {
    const s = store()
    add(s, 'approval_request', approval('old'))
    for (let index = 0; index < 30; index += 1) add(s, 'notification', { message: `noise ${index}` }, 'other')
    add(s, 'approval_request', approval('p1'))
    const found = new PendingInteractionIndex(s, { windowEvents: 5 }).frames('w1', 's1')
    expect(found.frames.map(frame => frame.payload.request_id)).toEqual(['p1'])
    expect(found.complete).toBe(false)
  })

  test('turn_begin also clears older cards; a later boundary makes an incremental view complete', () => {
    const s = store()
    add(s, 'approval_request', approval('stale'))
    for (let index = 0; index < 30; index += 1) add(s, 'notification', {}, 'other')
    const index = new PendingInteractionIndex(s, { windowEvents: 5 })
    expect(index.frames('w1', 's1').complete).toBe(false)
    add(s, 'turn_begin')
    add(s, 'question_request', { id: 'q1', questions: [{ question: 'Which?' }] })
    const found = index.frames('w1', 's1')
    expect(found.frames.map(frame => frame.payload.id)).toEqual(['q1'])
    expect(found.complete).toBe(true)
  })

  test('out of page budget: no stale cards are returned; later looks catch up and show the current set', () => {
    const s = store()
    add(s, 'turn_begin')
    add(s, 'approval_request', approval('resolved-later'))
    // More than one page (500 events) of the task's own events before the resolution.
    for (let index = 0; index < 1200; index += 1) add(s, 'tool_call', { id: `c${index}`, name: 'Read' })
    add(s, 'approval_response', { request_id: 'resolved-later' })
    add(s, 'approval_request', approval('p2'))
    const index = new PendingInteractionIndex(s, { maxPages: 1 })
    const first = index.frames('w1', 's1')
    // Behind the watermark the resolved card must not be shown as waiting.
    expect(first).toEqual({ frames: [], complete: false })
    expect(index.frames('w1', 's1')).toEqual({ frames: [], complete: false })
    const caught = index.frames('w1', 's1')
    expect(caught.frames.map(frame => frame.payload.request_id)).toEqual(['p2'])
    expect(caught.complete).toBe(true)
  })
})

describe('PendingInteractionIndex memory bound (P3-5)', () => {
  test('total card bytes stay under the cap; the least recently looked-at task is evicted first', () => {
    const s = store()
    const big = 'x'.repeat(4000)
    for (const session of ['a', 'b', 'c', 'd']) {
      add(s, 'turn_begin', {}, session)
      add(s, 'approval_request', approval(`p-${session}`, { description: big }), session)
    }
    const index = new PendingInteractionIndex(s, { maxBytes: 10_000 })
    for (const session of ['a', 'b', 'c']) expect(index.frames('w1', session).frames).toHaveLength(1)
    expect(index.heldBytes).toBeLessThanOrEqual(10_000)
    expect(index.size).toBe(2)
    // `a` was evicted (least recently used); looking again rescans and still finds its card.
    index.frames('w1', 'b')
    index.frames('w1', 'd')
    expect(index.heldBytes).toBeLessThanOrEqual(10_000)
    const again = index.frames('w1', 'a')
    expect(again.frames.map(frame => frame.payload.request_id)).toEqual(['p-a'])
    expect(index.heldBytes).toBeLessThanOrEqual(10_000)
  })

  test('one task over its share drops its oldest cards and reports itself incomplete', () => {
    const s = store()
    add(s, 'turn_begin')
    for (let index = 0; index < 6; index += 1) add(s, 'approval_request', approval(`p${index}`, { description: 'y'.repeat(1000) }))
    const index = new PendingInteractionIndex(s, { maxBytes: 16_000 })
    const found = index.frames('w1', 's1')
    expect(found.complete).toBe(false)
    expect(found.frames.at(-1)?.payload.request_id).toBe('p5')
    expect(index.heldBytes).toBeLessThanOrEqual(4000)
  })
})

describe('cron_event-wrapped interactions (P3-6)', () => {
  const wrapped = { job_id: 'job-1', event_type: 'approval_request', payload: approval('cron-p1') }
  const question = { job_id: 'job-1', event_type: 'question_request', payload: { id: 'cron-q1', questions: [{ question: 'Deploy now?' }] } }

  test('wrapped requests are redacted for viewers and kept for operators', () => {
    expect(isInteractionRequestEvent('cron_event', wrapped)).toBe(true)
    expect(isInteractionRequestEvent('cron_event', { job_id: 'j', event_type: 'text_part', payload: { text: 'x' } })).toBe(false)
    expect(interactionPayloadFor('cron_event', wrapped, 'full')).toBe(wrapped)
    const hidden = interactionPayloadFor('cron_event', wrapped, 'none')
    expect(hidden).toMatchObject({ job_id: 'job-1', event_type: 'approval_request', payload: { id: 'cron-p1', request_id: 'cron-p1', redacted: true } })
    expect(JSON.stringify(hidden)).not.toContain('bun test')
    expect(interactionPayloadFor('cron_event', question, 'questions')).toBe(question)
    expect(JSON.stringify(interactionPayloadFor('cron_event', question, 'none'))).not.toContain('Deploy now?')
    const nested = { job_id: 'outer', event_type: 'cron_event', payload: wrapped }
    expect(JSON.stringify(interactionPayloadFor('cron_event', nested, 'none'))).not.toContain('bun test')
    const events = redactEventsFor([{ type: 'cron_event', workspace_id: 'w1', session_id: 's1', payload: wrapped }], () => 'none')
    expect(JSON.stringify(events)).not.toContain('bun test')
  })

  test('the pending index tracks wrapped requests and their resolution', () => {
    const s = store()
    add(s, 'turn_begin')
    add(s, 'cron_event', wrapped)
    const index = new PendingInteractionIndex(s)
    expect(index.frames('w1', 's1').frames).toEqual([{ type: 'approval_request', payload: wrapped.payload }])
    add(s, 'cron_event', { job_id: 'job-1', event_type: 'approval_response', payload: { request_id: 'cron-p1' } })
    expect(index.frames('w1', 's1')).toEqual({ frames: [], complete: true })
  })
})
