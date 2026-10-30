// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { afterEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { cursorScopeKey, decodeCursor } from '../src/hub/cursor.js'
import { HubEventStore, HubEventStoreError, type UpstreamEventBatch } from '../src/hub/eventStore.js'
import { HUB_LIMITS, utf8ByteLength } from '../src/hub/protocol.js'

const stores: HubEventStore[] = []
const directories: string[] = []
const now = Date.parse('2026-10-21T12:00:00Z')
const scope = { workspaceIds: ['w1'] }

function store(path = ':memory:', options: ConstructorParameters<typeof HubEventStore>[1] = {}): HubEventStore {
  const value = new HubEventStore(path, { clock: () => now, ...options })
  stores.push(value)
  return value
}

function path(): string {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-hub-events-'))
  directories.push(directory)
  return join(directory, 'events.sqlite')
}

function batch(sequences: readonly number[], stream = 'upstream-1', payload: Readonly<Record<string, unknown>> = {}): UpstreamEventBatch {
  return {
    stream_id: stream,
    events: sequences.map(sequence => ({
      event_id: `${stream}-e${sequence}`, sequence, cursor: `up-${sequence}`, type: 'text_delta',
      payload, session_id: 's1', occurred_at: new Date(now).toISOString(),
    })),
    through_cursor: `up-${sequences.at(-1) ?? 0}`,
  }
}

afterEach(() => {
  for (const value of stores.splice(0)) value.close()
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true })
})

describe('HubEventStore', () => {
  test('dedupes reconnect delivery and atomically advances the upstream cursor', () => {
    const value = store()
    expect(value.ingest('r1', 'w1', batch([1, 2])).events.map(event => event.sequence)).toEqual([1, 2])
    const resumed = value.ingest('r1', 'w1', batch([1, 2, 3]))
    expect(resumed.duplicates).toBe(2)
    expect(resumed.events.map(event => event.sequence)).toEqual([3])
    expect(value.read(0, scope).events.map(event => event.event_id)).toEqual(['upstream-1-e1', 'upstream-1-e2', 'upstream-1-e3'])
    expect(value.upstreamCursor('r1')).toEqual({ runtime_id: 'r1', stream_id: 'upstream-1', last_sequence: 3, last_cursor: 'up-3' })
    value.ingest('r1', 'w1', batch([1]))
    expect(value.upstreamCursor('r1')?.last_sequence).toBe(3)
    expect(value.upstreamCursor('r1')?.last_cursor).toBe('up-3')
  })

  test('counts repeated upstream coordinates inside a batch as duplicates', () => {
    const value = store()
    const result = value.ingest('r1', 'w1', batch([1, 1, 2]))
    expect(result.duplicates).toBe(1)
    expect(result.events.map(event => event.sequence)).toEqual([1, 2])
    expect(value.upstreamCursor('r1')?.last_sequence).toBe(2)
  })

  test('accepts interleaved workspace batches below the resume checkpoint without misclassifying them as duplicates', () => {
    const value = store()
    value.ingest('r1', 'w1', batch([1, 3]))
    const secondWorkspace = value.ingest('r1', 'w2', batch([2]))
    expect(secondWorkspace.duplicates).toBe(0)
    expect(secondWorkspace.events.map(event => event.event_id)).toEqual(['upstream-1-e2'])
    expect(value.upstreamCursor('r1')?.last_sequence).toBe(3)
  })

  test('empty upstream batches can persist an explicit filtered-through checkpoint', () => {
    const value = store()
    value.ingest('r1', 'w1', { stream_id: 'upstream-1', events: [], through_cursor: 'up-10', through_sequence: 10 })
    expect(value.watermark()).toBe(0)
    expect(value.upstreamCursor('r1')?.last_cursor).toBe('up-10')
    expect(value.upstreamCursor('r1')?.last_sequence).toBe(10)
  })

  test('SQLite failure mid-ingest rolls back events, sequence and upstream cursor', () => {
    const filename = path()
    const value = store(filename)
    value.ingest('r1', 'w1', batch([1]))
    const external = new Database(filename)
    external.exec("CREATE TRIGGER fail_second BEFORE INSERT ON events WHEN NEW.upstream_sequence = 3 BEGIN SELECT RAISE(ABORT, 'injected write failure'); END")
    external.close()
    expect(() => value.ingest('r1', 'w1', batch([2, 3]))).toThrow(HubEventStoreError)
    expect(value.watermark()).toBe(1)
    expect(value.read(0, scope).events).toHaveLength(1)
    expect(value.upstreamCursor('r1')?.last_cursor).toBe('up-1')
  })

  test('cursor-write failure rolls back all inserted events and emits no committed notification', () => {
    const filename = path()
    const value = store(filename)
    value.ingest('r1', 'w1', batch([1]))
    const observed: unknown[] = []
    value.onEvents(events => { observed.push(events) })
    const external = new Database(filename)
    external.exec("CREATE TRIGGER fail_cursor BEFORE INSERT ON upstream_cursors WHEN NEW.last_sequence = 3 BEGIN SELECT RAISE(ABORT, 'injected cursor failure'); END")
    external.close()
    expect(() => value.ingest('r1', 'w1', batch([2, 3]))).toThrow(HubEventStoreError)
    expect(value.watermark()).toBe(1)
    expect(value.upstreamCursor('r1')?.last_cursor).toBe('up-1')
    expect(value.read(0, scope).events).toHaveLength(1)
    expect(observed).toEqual([])
  })

  test('persists the stream id, uses WAL, and a different DB has a new stream id', () => {
    const filename = path()
    const value = store(filename)
    const stream = value.streamId
    value.ingest('r1', 'w1', batch([1]))
    const inspection = new Database(filename)
    expect(inspection.query<{ journal_mode: string }, []>('PRAGMA journal_mode').get()?.journal_mode).toBe('wal')
    inspection.close()
    value.close()
    const reopened = store(filename)
    expect(reopened.streamId).toBe(stream)
    expect(reopened.watermark()).toBe(1)
    expect(store(path()).streamId).not.toBe(stream)
  })

  test('encodes each event cursor for the reader scope, not the ingestion scope', () => {
    const value = store()
    value.ingest('r1', 'w1', batch([1]))
    const a = value.read(0, scope).events[0]!
    const b = value.read(0, { workspaceIds: ['w1', 'w2'] }).events[0]!
    expect(a.cursor).not.toBe(b.cursor)
    expect(decodeCursor(a.cursor)).toEqual({ streamId: value.streamId, sequence: 1, scopeKey: cursorScopeKey(scope) })
    expect(a.event_id).toBe(b.event_id)
  })

  test('filters workspaces and sessions and supports a fixed read watermark', () => {
    const value = store()
    value.ingest('r1', 'w1', batch([1]))
    value.ingest('r2', 'w2', batch([1], 'upstream-2'))
    value.ingest('r1', 'w1', batch([2]))
    expect(value.read(0, scope, 500, HUB_LIMITS.normalFrameBytes, 1).events.map(event => event.sequence)).toEqual([1])
    expect(value.read(0, { workspaceIds: ['w1'], sessionIds: ['other'] }).events).toEqual([])
    expect(value.read(0, scope).events.map(event => event.sequence)).toEqual([1, 3])
  })

  test('returns byte-bounded pages without losing the next event', () => {
    const value = store()
    value.ingest('r1', 'w1', batch([1, 2, 3], 'upstream-1', { text: '😀'.repeat(60) }))
    const first = value.read(0, scope, 500, 1000)
    expect(first.events.length).toBeGreaterThan(0)
    expect(first.events.length).toBeLessThan(3)
    expect(first.bytes).toBe(utf8ByteLength(JSON.stringify(first.events)))
    expect(first.bytes).toBeLessThanOrEqual(1000)
    const rest = value.read(first.throughSequence, scope, 500, 1000)
    expect(rest.events[0]?.sequence).toBe(first.events.at(-1)!.sequence + 1)
    expect(first.hasMore).toBe(true)
  })

  test('oversized events are stored intact but read as bounded stubs', () => {
    const value = store()
    value.ingest('r1', 'w1', batch([1], 'upstream-1', { text: 'x'.repeat(HUB_LIMITS.maxFrameBytes + 1) }))
    const page = value.read(0, scope)
    expect(page.events[0]?.payload.truncated).toBe(true)
    expect(page.events[0]?.payload.original_bytes).toBeGreaterThan(HUB_LIMITS.maxFrameBytes)
    expect(page.bytes).toBeLessThanOrEqual(HUB_LIMITS.normalFrameBytes)
    expect(page.throughSequence).toBe(1)
  })

  test('an oversized interaction stub keeps its ids and hides its details', () => {
    const value = store()
    const big = 'x'.repeat(HUB_LIMITS.maxFrameBytes + 1)
    const event = (sequence: number, type: string, payload: Readonly<Record<string, unknown>>) => ({
      event_id: `big-e${sequence}`, sequence, cursor: `up-${sequence}`, type, payload, session_id: 's1', occurred_at: new Date(now).toISOString(),
    })
    value.ingest('r1', 'w1', {
      stream_id: 'upstream-1',
      events: [
        event(1, 'approval_request', { id: 'perm-1', request_id: 'perm-1', tool_call_id: 'call-1', session_id: 's1', inputs: { command: big } }),
        event(2, 'question_request', { id: 'q-1', tool_call_id: 'call-2', questions: [{ question: big }] }),
        event(3, 'approval_response', { request_id: 'perm-1', note: big }),
        event(4, 'cron_event', { job_id: 'job-1', event_type: 'approval_request', payload: { id: 'perm-2', request_id: 'perm-2', inputs: { command: big } } }),
      ],
      through_cursor: 'up-4',
    })
    const page = value.read(0, scope, 10, HUB_LIMITS.normalFrameBytes)
    const [approval, question, resolution, cron] = page.events.map(row => row.payload)
    expect(approval).toEqual({ id: 'perm-1', request_id: 'perm-1', tool_call_id: 'call-1', session_id: 's1', redacted: true, details_hidden: true, truncated: true, original_bytes: expect.any(Number) })
    expect(question).toMatchObject({ id: 'q-1', tool_call_id: 'call-2', redacted: true, details_hidden: true, questions: [], truncated: true })
    expect(resolution).toMatchObject({ request_id: 'perm-1', truncated: true })
    expect(resolution?.redacted).toBeUndefined()
    expect(cron).toMatchObject({ job_id: 'job-1', event_type: 'approval_request', truncated: true, payload: { id: 'perm-2', request_id: 'perm-2', redacted: true, details_hidden: true, truncated: true } })
    expect(page.bytes).toBeLessThanOrEqual(HUB_LIMITS.normalFrameBytes)
  })

  test('byte retention raises the floor, returns a gap, and never reuses sequences', () => {
    const value = store(':memory:', { retention: { maxBytes: 1, maxAgeMs: 86_400_000 } })
    value.ingest('r1', 'w1', batch([1, 2]))
    expect(value.watermark()).toBe(2)
    expect(value.oldestSequence()).toBe(3)
    expect(value.read(0, scope).gap).toBe('retention')
    expect(value.read(2, scope).gap).toBeUndefined()
    expect(value.ingest('r1', 'w1', batch([3])).events[0]?.sequence).toBe(3)
    expect(value.ingest('r1', 'w1', batch([1])).duplicates).toBe(1)
  })

  test('age retention is driven by the injected clock and persisted floor', () => {
    let time = now
    const filename = path()
    const value = store(filename, { clock: () => time, retention: { maxBytes: 1_000_000, maxAgeMs: 1000 } })
    value.ingest('r1', 'w1', batch([1]))
    time += 1001
    value.prune()
    expect(value.oldestSequence()).toBe(2)
    value.close()
    const reopened = store(filename)
    expect(reopened.oldestSequence()).toBe(2)
    expect(reopened.watermark()).toBe(1)
  })

  test('appends durable Hub events, rejects ephemeral presence, and notifies only after commit', () => {
    const value = store()
    const observed: number[] = []
    value.onEvents(events => { observed.push(...events.map(event => event.sequence)); expect(value.watermark()).toBe(events.at(-1)!.sequence) })
    const event = value.append({ workspace_id: 'w1', type: 'hub.command_result', payload: { command_id: 'c1' } })
    expect(event.sequence).toBe(1)
    expect(observed).toEqual([1])
    expect(value.read(0, scope).events[0]?.event_id).toBe(event.event_id)
    expect(() => value.append({ workspace_id: 'w1', type: 'hub.presence', payload: {} })).toThrow(/ephemeral/)
  })

  test('upstream restarts have independent dedupe keys and cursor state', () => {
    const value = store()
    value.ingest('r1', 'w1', batch([1]))
    expect(value.ingest('r1', 'w1', batch([1], 'replacement')).events[0]?.sequence).toBe(2)
    expect(value.upstreamCursor('r1')?.stream_id).toBe('replacement')
  })

  test('closed stores fail with a typed storage error', () => {
    const value = store()
    value.close()
    expect(() => value.ingest('r1', 'w1', batch([1]))).toThrow(HubEventStoreError)
    expect(() => value.watermark()).toThrow(HubEventStoreError)
  })
})
