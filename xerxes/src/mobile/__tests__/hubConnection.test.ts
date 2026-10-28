// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { HubRequestError, backoffDelay } from '../../client/hubConnection.js'
import { batch, ev, goLive, harness, settle } from './fakes.js'

describe('HubConnection lifecycle', () => {
  it('does the hello exchange, goes live and keeps tokens out of the URL', async () => {
    const h = harness()
    await goLive(h)
    expect(h.conn.status().state).toBe('live')
    expect(h.conn.helloInfo().userId).toBe('u1')
    expect(h.hub.all('hub.hello')).toHaveLength(1)
  })

  it('treats an unauthorized hello as revoked and does not reconnect', async () => {
    const h = harness()
    h.hub.handlers.set('hub.hello', () => ({ __error: { code: 'unauthorized', message: 'no' } }))
    await goLive(h)
    expect(h.conn.status().state).toBe('revoked')
    h.timers.advance(120_000)
    await settle()
    expect(h.hub.sockets).toHaveLength(1)
  })

  it('refreshes once and reconnects after an unauthorized hello instead of signing out', async () => {
    let checks = 0
    let helloCalls = 0
    const h = harness({ ensureSession: async () => { checks += 1; return 'ok' } })
    h.hub.handlers.set('hub.hello', () => (++helloCalls === 1
      ? { __error: { code: 'unauthorized', message: 'session lapsed' } }
      : { principal: { id: 'u1', display_name: 'Ann' } }))
    await goLive(h)
    expect(h.conn.status().state).toBe('reconnecting')
    h.timers.advance(60_000)
    await settle()
    h.hub.socket(h.hub.sockets.length - 1).open()
    await settle()
    expect(h.conn.status().state).toBe('live')
    expect(checks).toBeGreaterThanOrEqual(2)
  })

  it('signs out after an unauthorized hello only when the refresh is also denied', async () => {
    let checks = 0
    const h = harness({ ensureSession: async () => (++checks === 1 ? 'ok' : 'denied') })
    h.hub.handlers.set('hub.hello', () => ({ __error: { code: 'unauthorized', message: 'no' } }))
    await goLive(h)
    expect(h.conn.status().state).toBe('revoked')
  })

  it('stops reconnecting after hub.revoked', async () => {
    const h = harness()
    const socket = await goLive(h)
    socket.event('hub.revoked', { reason: 'revoked' })
    expect(h.conn.status().state).toBe('revoked')
    socket.drop(1008)
    h.timers.advance(120_000)
    await settle()
    expect(h.hub.sockets).toHaveLength(1)
    const result = await h.conn.command({ workspaceId: 'w1', sessionId: 's1', action: 'turn.cancel', args: {} })
    expect(result.state).toBe('rejected')
    expect(h.hub.all('hub.command')).toHaveLength(0)
  })

  it('ends as revoked when the session refresh is denied on reconnect', async () => {
    let calls = 0
    const h = harness({ ensureSession: async () => (++calls === 1 ? 'ok' : 'denied') })
    const socket = await goLive(h)
    socket.drop()
    h.timers.advance(60_000)
    await settle()
    expect(h.conn.status().state).toBe('revoked')
    expect(h.hub.sockets).toHaveLength(1)
  })

  it('reconnects with capped, jittered backoff', () => {
    const options = { baseMs: 500, maxMs: 30_000 }
    expect(backoffDelay(1, options, () => 0)).toBe(250)
    expect(backoffDelay(1, options, () => 1)).toBe(500)
    expect(backoffDelay(3, options, () => 1)).toBe(2000)
    for (const attempt of [10, 20, 100]) expect(backoffDelay(attempt, options, () => 1)).toBe(30_000)
    expect(backoffDelay(50, options, () => 0)).toBe(15_000)
  })

  it('fails requests fast while offline', async () => {
    const h = harness()
    await expect(h.conn.request('hub.usage')).rejects.toMatchObject({ code: 'offline' })
  })
})

describe('HubConnection frames', () => {
  it('refuses inbound frames over the limit', async () => {
    const h = harness()
    const socket = await goLive(h)
    socket.handlers.message('x'.repeat(1024 * 1024 + 10))
    expect(h.errors.at(-1)).toMatchObject({ code: 'frame_too_large' })
  })

  it('refuses to send a frame over the limit', async () => {
    const h = harness()
    await goLive(h)
    await expect(h.conn.request('hub.usage', { blob: 'x'.repeat(1_100_000) })).rejects.toMatchObject({ code: 'frame_too_large' })
  })

  it('reports malformed frames without crashing', async () => {
    const h = harness()
    const socket = await goLive(h)
    socket.handlers.message('not json')
    socket.handlers.binary()
    expect(h.errors).toHaveLength(2)
    expect(h.conn.status().state).toBe('live')
  })

  it('times a request out as uncertain', async () => {
    const h = harness()
    await goLive(h)
    h.hub.handlers.set('hub.usage', () => undefined)
    const pending = h.conn.request('hub.usage')
    const caught = pending.catch((error: unknown) => error)
    h.timers.advance(16_000)
    const error = await caught
    expect(error).toBeInstanceOf(HubRequestError)
    expect(error).toMatchObject({ code: 'timeout', uncertain: true })
  })
})

describe('HubConnection subscriptions', () => {
  it('applies events once, in order, and acks only after applying', async () => {
    const h = harness()
    const socket = await goLive(h)
    const order: string[] = []
    h.hub.handlers.set('hub.events.ack', params => { order.push(`ack:${String(params.cursor)}`); return {} })
    const applied: string[] = []
    h.conn.subscribe({ workspaceIds: ['w1'] }, { apply: event => { applied.push(event.event_id); order.push(`apply:${event.sequence}`) }, reset: () => undefined }, { cursor: 'base' })
    await settle()
    expect(socket.framesFor('hub.events.subscribe')[0]?.params.after_cursor).toBe('base')
    const e1 = ev(1)
    const e2 = ev(2)
    batch(socket, 'sub1', [e1, e2], 'T2')
    batch(socket, 'sub1', [e1, ev(3)], 'T3')
    await settle()
    expect(applied).toHaveLength(3)
    expect(order).toEqual(['apply:1', 'apply:2', 'ack:T2', 'apply:3', 'ack:T3'])
  })

  it('resumes after a reconnect from the last applied cursor', async () => {
    const h = harness()
    const socket = await goLive(h)
    const handle = h.conn.subscribe({ workspaceIds: ['w1'], sessionIds: ['s1'] }, { apply: () => undefined, reset: () => undefined }, { cursor: 'base' })
    await settle()
    batch(socket, 'sub1', [ev(1), ev(2)], 'T2')
    await settle()
    expect(handle.cursor()).toBe('T2')
    socket.drop(1006)
    expect(h.conn.status().state).toBe('reconnecting')
    h.timers.advance(1000)
    await settle()
    const second = h.hub.socket(1)
    second.open()
    await settle()
    expect(h.conn.status().state).toBe('live')
    expect(second.framesFor('hub.events.subscribe')[0]?.params).toMatchObject({ workspace_ids: ['w1'], session_ids: ['s1'], after_cursor: 'T2' })
  })

  it('resets from a snapshot on a stream gap and can replay from the start', async () => {
    const h = harness()
    const socket = await goLive(h)
    const reasons: string[] = []
    h.conn.subscribe({ workspaceIds: ['w1'] }, {
      apply: () => undefined,
      reset: ({ reason }) => { reasons.push(reason); return { fromStart: true } },
    })
    await settle()
    socket.event('hub.stream_gap', { subscription_id: 'sub1', reason: 'stream_replaced', watermark: 'W9' })
    await settle()
    expect(reasons).toEqual(['stream_replaced'])
    const subscribes = socket.framesFor('hub.events.subscribe')
    expect(subscribes).toHaveLength(2)
    expect(subscribes[1]?.params.after_cursor).toBeUndefined()
    expect(socket.framesFor('hub.events.unsubscribe')[0]?.params.subscription_id).toBe('sub1')
  })

  it('treats an expired cursor on subscribe as a reset', async () => {
    const h = harness()
    const socket = await goLive(h)
    h.hub.handlers.set('hub.events.subscribe', () => ({ subscription_id: 'subX', watermark: 'W5', oldest_cursor: 'O', reset_required: true }))
    const reasons: string[] = []
    h.conn.subscribe({ workspaceIds: ['w1'] }, { apply: () => undefined, reset: ({ reason }) => { reasons.push(reason); return { cursor: 'S1' } } }, { cursor: 'old' })
    await settle()
    expect(reasons).toEqual(['retention'])
    expect(socket.framesFor('hub.events.subscribe').at(-1)?.params.after_cursor).toBe('S1')
  })

  it('resets when applying an event throws, without acking that batch', async () => {
    const h = harness()
    const socket = await goLive(h)
    const reasons: string[] = []
    h.conn.subscribe({ workspaceIds: ['w1'] }, {
      apply: event => { if (event.sequence === 2) throw new Error('boom') },
      reset: ({ reason }) => { reasons.push(reason); return { cursor: 'S' } },
    }, { cursor: 'base' })
    await settle()
    batch(socket, 'sub1', [ev(1), ev(2)], 'T2')
    await settle()
    expect(reasons).toEqual(['apply_failed'])
    expect(socket.framesFor('hub.events.ack')).toHaveLength(0)
    expect(socket.framesFor('hub.events.subscribe').at(-1)?.params.after_cursor).toBe('S')
  })

  it('tolerates holes in scoped sequences but flags them when contiguity is required', async () => {
    for (const contiguous of [false, true]) {
      const h = harness({ contiguousSequences: contiguous })
      const socket = await goLive(h)
      const applied: number[] = []
      const reasons: string[] = []
      h.conn.subscribe({ workspaceIds: ['w1'] }, { apply: event => { applied.push(event.sequence) }, reset: ({ reason }) => { reasons.push(reason); return { cursor: 'S' } } }, { cursor: 'base' })
      await settle()
      batch(socket, 'sub1', [ev(1), ev(5)], 'T5')
      await settle()
      if (contiguous) {
        expect(applied).toEqual([1])
        expect(reasons).toEqual(['sequence_gap'])
      } else {
        expect(applied).toEqual([1, 5])
        expect(reasons).toEqual([])
      }
    }
  })

  it('ignores events for an unknown subscription', async () => {
    const h = harness()
    const socket = await goLive(h)
    const applied: number[] = []
    h.conn.subscribe({ workspaceIds: ['w1'] }, { apply: event => { applied.push(event.sequence) }, reset: () => undefined }, { cursor: 'base' })
    await settle()
    batch(socket, 'someone-else', [ev(1)], 'T1')
    await settle()
    expect(applied).toEqual([])
  })
})

describe('HubConnection commands', () => {
  const command = { workspaceId: 'w1', sessionId: 's1', action: 'permission.respond', args: { request_id: 'r1', response: 'approve' } } as const

  it('sends once and asks for the outcome when the acknowledgement is lost', async () => {
    const h = harness()
    const socket = await goLive(h)
    h.hub.handlers.set('hub.command', () => undefined)
    h.hub.handlers.set('hub.command.status', params => ({ ok: true, command_id: params.command_id, state: 'accepted', result: {} }))
    const pending = h.conn.command({ ...command, commandId: 'cmd-1' })
    await settle()
    expect(socket.framesFor('hub.command')).toHaveLength(1)
    socket.drop(1006)
    h.timers.advance(1000)
    await settle()
    h.hub.socket(1).open()
    await settle()
    const result = await pending
    expect(result.state).toBe('accepted')
    expect(h.hub.all('hub.command')).toHaveLength(1)
    const status = h.hub.all('hub.command.status')
    expect(status).toHaveLength(1)
    expect(status[0]?.params).toEqual({ workspace_id: 'w1', session_id: 's1', command_id: 'cmd-1' })
  })

  it('reports unknown, never replays, when the link does not come back', async () => {
    const h = harness()
    const socket = await goLive(h)
    h.hub.handlers.set('hub.command', () => undefined)
    const pending = h.conn.command({ ...command, commandId: 'cmd-2' })
    await settle()
    socket.drop(1006)
    // Let the rejected send reach commandOutcome so its wait timer exists before time moves.
    await settle()
    h.timers.advance(21_000)
    await settle()
    const result = await pending
    expect(result.state).toBe('unknown')
    expect(result.ok).toBe(false)
    expect(h.hub.all('hub.command')).toHaveLength(1)
    expect(h.hub.all('hub.command.status')).toHaveLength(0)
  })

  it('does not send anything while offline', async () => {
    const h = harness()
    const result = await h.conn.command({ ...command })
    expect(result.state).toBe('rejected')
    expect(h.hub.sockets).toHaveLength(0)
  })

  it('maps a server rejection to a rejected result, not a retry', async () => {
    const h = harness()
    await goLive(h)
    h.hub.handlers.set('hub.command', () => ({ __error: { code: 'forbidden', message: 'not permitted' } }))
    const result = await h.conn.command({ ...command })
    expect(result).toMatchObject({ ok: false, state: 'rejected', error: { code: 'forbidden' } })
    expect(h.hub.all('hub.command')).toHaveLength(1)
  })

  it('keeps the interaction-already-resolved answer from the Hub', async () => {
    const h = harness()
    await goLive(h)
    h.hub.handlers.set('hub.command', params => ({ ok: false, command_id: params.command_id, state: 'rejected', error: { code: 'interaction_already_resolved', message: 'Resolved by Bob' } }))
    const result = await h.conn.command({ ...command })
    expect(result.error).toEqual({ code: 'interaction_already_resolved', message: 'Resolved by Bob' })
  })
})

describe('scoped bridges', () => {
  it('routes events only to the view whose scope matches and adds scope to scoped methods', async () => {
    const h = harness()
    const socket = await goLive(h)
    h.conn.subscribe({ workspaceIds: ['w1', 'w2'] }, { apply: () => undefined, reset: () => undefined }, { cursor: 'base' })
    await settle()
    h.hub.handlers.set('hub.history', params => ({ history: params }))
    const session = h.conn.bridge({ workspaceId: 'w1', sessionId: 's1' })
    const other = h.conn.bridge({ workspaceId: 'w2' })
    const seen: string[] = []
    session.onEvent(event => seen.push(`session:${event.event_id}`))
    other.onEvent(event => seen.push(`other:${event.event_id}`))
    batch(socket, 'sub1', [ev(1, { event_id: 'a' }), ev(2, { event_id: 'b', session_id: 's2' }), ev(3, { event_id: 'c', workspace_id: 'w2', session_id: 's9' })], 'T3')
    await settle()
    expect(seen).toEqual(['session:a', 'other:c'])
    const answer = await session.call('hub.history', { limit: 5 })
    expect(answer.history).toMatchObject({ workspace_id: 'w1', session_id: 's1', limit: 5 })
    await expect(session.call('session.history')).rejects.toMatchObject({ code: 'invalid_request' })
    session.dispose()
  })
})
