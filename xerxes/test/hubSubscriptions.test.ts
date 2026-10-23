// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'

import { cursorScopeKey, decodeCursor, encodeCursor } from '../src/hub/cursor.js'
import { HubEventStore } from '../src/hub/eventStore.js'
import { HubProtocolError, encodeHubFrame, utf8ByteLength, type HubEvent, type HubEventFrame } from '../src/hub/protocol.js'
import { SubscriptionManager, type SubscriptionSink } from '../src/hub/subscriptions.js'
import type { HubGrant, HubPrincipal } from '../src/security/principal.js'

const resources: { close(): void }[] = []
const principal: HubPrincipal = { kind: 'user', id: 'alice', displayName: 'Alice', deviceId: 'phone', hubOwner: false }
const grants: readonly HubGrant[] = [{ principalId: 'alice', resource: { type: 'workspace', workspaceId: 'w1' }, effect: 'allow', role: 'viewer' }]
const now = Date.parse('2026-10-21T12:00:00Z')

function setup(options: ConstructorParameters<typeof SubscriptionManager>[1] = {}) {
  const store = new HubEventStore(':memory:', { clock: () => now })
  const manager = new SubscriptionManager(store, { clock: () => now, ...options })
  resources.push(store, manager)
  return { store, manager }
}

function event(store: HubEventStore, id: string, workspace = 'w1', session = 's1', text = '') {
  return store.append({ event_id: id, workspace_id: workspace, session_id: session, type: 'text_delta', payload: { text } })
}

function sink(): SubscriptionSink & { frames: HubEventFrame[]; codes: number[] } {
  const frames: HubEventFrame[] = []
  const codes: number[] = []
  return { frames, codes, send: frame => { frames.push(frame) }, close: code => { codes.push(code) } }
}

function delivered(target: { frames: HubEventFrame[] }): HubEvent[] {
  return target.frames.filter(frame => frame.type === 'hub.event_batch').flatMap(frame => frame.payload.events as HubEvent[])
}

async function tick(): Promise<void> {
  await new Promise<void>(resolve => setTimeout(resolve, 0))
}

function protocolError(run: () => unknown): string {
  try { run() } catch (error) { if (error instanceof HubProtocolError) return error.code; throw error }
  throw new Error('expected HubProtocolError')
}

afterEach(() => {
  for (const resource of resources.splice(0).reverse()) resource.close()
})

describe('SubscriptionManager', () => {
  test('backfills through the captured watermark then delivers concurrent and live appends exactly once', async () => {
    const { store, manager } = setup({ pageEventLimit: 1 })
    event(store, 'one')
    event(store, 'two')
    const target = sink()
    const original = target.send
    target.send = frame => {
      original(frame)
      if (frame.type === 'hub.event_batch' && delivered(target).length === 1) event(store, 'during-backfill')
    }
    const result = manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, target)
    expect(decodeCursor(result.watermark).sequence).toBe(2)
    event(store, 'after-subscribe')
    await tick()
    expect(delivered(target).map(value => value.event_id)).toEqual(['one', 'two', 'after-subscribe', 'during-backfill'])
    expect(new Set(delivered(target).map(value => value.sequence)).size).toBe(4)
  })

  test('a fast sink stays below its queue budget when delivery triggers concurrent appends', async () => {
    const { store, manager } = setup({ pageEventLimit: 1, maxQueueBytes: 1800 })
    event(store, 'one')
    const target = sink()
    const original = target.send
    target.send = frame => {
      original(frame)
      if (frame.type === 'hub.event_batch' && delivered(target).length === 1) {
        for (let i = 0; i < 20; i += 1) event(store, `concurrent-${i}`)
      }
    }
    manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, target)
    await tick()
    expect(delivered(target)).toHaveLength(21)
    expect(target.codes).toEqual([])
  })

  test('refuses forbidden workspaces instead of silently dropping them', () => {
    const { manager } = setup()
    expect(protocolError(() => manager.subscribe(principal, grants, { workspaceIds: ['w1', 'w2'] }, sink()))).toBe('forbidden')
  })

  test('session-only grants expose only granted sessions, not workspace-level events', async () => {
    const { store, manager } = setup()
    const narrow: HubGrant[] = [{ principalId: 'alice', resource: { type: 'session', workspaceId: 'w1', sessionId: 's1' }, effect: 'allow', role: 'viewer' }]
    event(store, 'visible')
    event(store, 'secret', 'w1', 's2')
    store.append({ event_id: 'inventory', workspace_id: 'w1', type: 'hub.workspace_status', payload: {} })
    const target = sink()
    manager.subscribe(principal, narrow, { workspaceIds: ['w1'] }, target)
    await tick()
    expect(delivered(target).map(value => value.event_id)).toEqual(['visible'])
    expect(protocolError(() => manager.subscribe(principal, narrow, { workspaceIds: ['w1'], sessionIds: ['s2'] }, sink()))).toBe('forbidden')
  })

  test('per-event checks honor session denies and explicit requested sessions', async () => {
    const { store, manager } = setup()
    event(store, 'a', 'w1', 's1')
    event(store, 'b', 'w1', 's2')
    const denied: HubGrant[] = [...grants, { principalId: 'alice', resource: { type: 'session', workspaceId: 'w1', sessionId: 's2' }, effect: 'deny', scopes: ['session.read'] }]
    const target = sink()
    manager.subscribe(principal, denied, { workspaceIds: ['w1'], sessionIds: ['s1', 's2'] }, target)
    await tick()
    expect(delivered(target).map(value => value.event_id)).toEqual(['a'])
  })

  test('binds resume and ack cursors to the subscription scope', async () => {
    const { store, manager } = setup()
    event(store, 'one')
    const target = sink()
    const result = manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, target)
    await tick()
    const cursor = delivered(target)[0]!.cursor
    manager.ack(result.subscriptionId, cursor)
    expect(manager.lastAckedCursor(result.subscriptionId)).toBe(cursor)
    const foreign = encodeCursor({ streamId: store.streamId, sequence: 1, scopeKey: cursorScopeKey({ workspaceIds: ['w2'] }) })
    expect(protocolError(() => manager.ack(result.subscriptionId, foreign))).toBe('invalid_request')
    expect(protocolError(() => manager.subscribe(principal, grants, { workspaceIds: ['w1'], afterCursor: foreign }, sink()))).toBe('invalid_request')
    const future = encodeCursor({ streamId: store.streamId, sequence: 2, scopeKey: cursorScopeKey({ workspaceIds: ['w1'] }) })
    expect(protocolError(() => manager.ack(result.subscriptionId, future))).toBe('invalid_request')
  })

  test('new store stream requires reset and emits a stream_replaced gap', async () => {
    const first = setup()
    event(first.store, 'one')
    const old = first.manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, sink()).watermark
    const { manager } = setup()
    const target = sink()
    const result = manager.subscribe(principal, grants, { workspaceIds: ['w1'], afterCursor: old }, target)
    expect(result.resetRequired).toBe(true)
    await tick()
    expect(target.frames[0]).toMatchObject({ type: 'hub.stream_gap', payload: { reason: 'stream_replaced', requested_cursor: old } })
  })

  test('expired replay returns a retention gap and the current oldest and watermark', async () => {
    const store = new HubEventStore(':memory:', { clock: () => now, retention: { maxBytes: 1, maxAgeMs: 1000 } })
    const manager = new SubscriptionManager(store)
    resources.push(store, manager)
    const cursor = encodeCursor({ streamId: store.streamId, sequence: 0, scopeKey: cursorScopeKey({ workspaceIds: ['w1'] }) })
    event(store, 'one')
    const target = sink()
    const result = manager.subscribe(principal, grants, { workspaceIds: ['w1'], afterCursor: cursor }, target)
    expect(result.resetRequired).toBe(true)
    expect(decodeCursor(result.oldest).sequence).toBe(1)
    await tick()
    expect(target.frames[0]).toMatchObject({ type: 'hub.stream_gap', payload: { reason: 'retention', oldest_cursor: result.oldest, watermark: result.watermark } })
  })

  test('the returned oldest cursor is a safe replay anchor that includes the first retained event', async () => {
    const { store, manager } = setup()
    event(store, 'first')
    const initial = manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, sink())
    const target = sink()
    manager.subscribe(principal, grants, { workspaceIds: ['w1'], afterCursor: initial.oldest }, target)
    await tick()
    expect(delivered(target).map(value => value.event_id)).toEqual(['first'])
  })

  test('rechecks grants before each queued delivery and emits scope_changed after revocation', async () => {
    let current = grants
    const { store, manager } = setup({ pageEventLimit: 1, grantsForPrincipal: () => current })
    event(store, 'one')
    event(store, 'secret')
    let release!: () => void
    const blocked = new Promise<void>(resolve => { release = resolve })
    const target = sink()
    const original = target.send
    target.send = frame => { original(frame); return frame.type === 'hub.event_batch' ? blocked : undefined }
    manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, target)
    await tick()
    current = []
    release()
    await tick()
    expect(delivered(target).map(value => value.event_id)).toEqual(['one'])
    expect(target.frames.at(-1)).toMatchObject({ type: 'hub.stream_gap', payload: { reason: 'scope_changed' } })
    expect(target.codes).toEqual([1008])
  })

  test('expired grants are evaluated using the injected clock', async () => {
    let time = now
    const { store, manager } = setup({ clock: () => time })
    const expiring: HubGrant[] = [{ ...grants[0]!, expiresAt: new Date(now + 100).toISOString() }]
    const target = sink()
    manager.subscribe(principal, expiring, { workspaceIds: ['w1'] }, target)
    await tick()
    time += 101
    event(store, 'secret')
    await tick()
    expect(delivered(target)).toEqual([])
    expect(target.frames.at(-1)?.payload.reason).toBe('scope_changed')
  })

  test('slow client overflows with 1013 and resumes after its last ack without event loss', async () => {
    const { store, manager } = setup({ maxQueueBytes: 1800, pageEventLimit: 1 })
    event(store, 'one')
    const target = sink()
    const result = manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, target)
    await tick()
    const cursor = delivered(target)[0]!.cursor
    manager.ack(result.subscriptionId, cursor)
    let release!: () => void
    const blocked = new Promise<void>(resolve => { release = resolve })
    target.send = frame => { target.frames.push(frame); return blocked }
    for (let i = 2; i <= 8; i += 1) event(store, `event-${i}`, 'w1', 's1', 'x'.repeat(100))
    await tick()
    expect(target.codes).toEqual([1013])
    const resumed = sink()
    manager.subscribe(principal, grants, { workspaceIds: ['w1'], afterCursor: cursor }, resumed)
    release()
    await tick()
    expect(delivered(resumed).map(value => value.event_id)).toEqual(Array.from({ length: 7 }, (_, i) => `event-${i + 2}`))
  })

  test('revokePrincipal immediately sends hub.revoked then closes all of that principal\'s subscriptions', async () => {
    const { store, manager } = setup()
    const first = sink()
    const second = sink()
    const other = sink()
    manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, first)
    manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, second)
    manager.subscribe({ ...principal, id: 'bob', hubOwner: true }, [], { workspaceIds: ['w1'] }, other)
    manager.revokePrincipal('alice')
    expect(first.frames.at(-1)?.type).toBe('hub.revoked')
    expect(first.codes).toEqual([1008])
    expect(second.codes).toEqual([1008])
    event(store, 'one')
    await tick()
    expect(delivered(first)).toEqual([])
    expect(delivered(other)).toHaveLength(1)
  })

  test('unsubscribe cancels queued delivery and all frames respect byte bounds', async () => {
    const { store, manager } = setup({ maxBatchBytes: 2048 })
    event(store, 'one', 'w1', 's1', 'x'.repeat(5000))
    const stopped = sink()
    const result = manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, stopped)
    manager.unsubscribe(result.subscriptionId)
    const target = sink()
    manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, target)
    await tick()
    expect(stopped.frames).toEqual([])
    expect(delivered(target)[0]?.payload.truncated).toBe(true)
    for (const frame of target.frames) expect(utf8ByteLength(encodeHubFrame(frame))).toBeLessThanOrEqual(2048)
  })

  test('unsubscribe discards queued frames even while the sink has an in-flight send', async () => {
    const { store, manager } = setup({ pageEventLimit: 1 })
    event(store, 'one')
    event(store, 'two')
    let release!: () => void
    const blocked = new Promise<void>(resolve => { release = resolve })
    const target = sink()
    target.send = frame => { target.frames.push(frame); return blocked }
    const result = manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, target)
    await tick()
    manager.unsubscribe(result.subscriptionId)
    release()
    event(store, 'three')
    await tick()
    expect(delivered(target).map(value => value.event_id)).toEqual(['one'])
    expect(target.codes).toEqual([])
  })

  test('a rejected send closes the failed transport without unhandled rejection or further delivery', async () => {
    const { store, manager } = setup({ pageEventLimit: 1 })
    event(store, 'one')
    event(store, 'two')
    const target = sink()
    target.send = frame => { target.frames.push(frame); return Promise.reject(new Error('transport disconnected')) }
    manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, target)
    await tick()
    expect(delivered(target).map(value => value.event_id)).toEqual(['one'])
    expect(target.codes).toEqual([1011])
  })

  test('updateGrants revokes an idle subscription before its next data delivery', async () => {
    const { store, manager } = setup()
    const target = sink()
    manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, target)
    await tick()
    manager.updateGrants('alice', [])
    await tick()
    event(store, 'secret')
    await tick()
    expect(delivered(target)).toEqual([])
    expect(target.frames.at(-1)?.payload.reason).toBe('scope_changed')
    expect(target.codes).toEqual([1008])
  })

  test('storage read failure emits journal_unavailable rather than claiming replay', async () => {
    const { store, manager } = setup()
    const target = sink()
    manager.subscribe(principal, grants, { workspaceIds: ['w1'] }, target)
    store.close()
    await tick()
    expect(target.frames.at(-1)?.payload.reason).toBe('journal_unavailable')
    expect(target.codes).toEqual([1011])
  })
})
