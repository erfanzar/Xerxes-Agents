// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { chmod, mkdir, symlink } from 'node:fs/promises'
import { join } from 'node:path'

import { emptyProjection, replaceSessionSummaries } from '../src/hub/projections.js'
import { verifyHubAuth } from '../src/security/hubAuthEnvelope.js'
import { disposeAll, registerCleanup, startFixture, tempDir, until, type Fixture } from './hubHarness.js'

afterEach(disposeAll)

const submit = (id = 'c1', text = 'hello') => ({ workspace_id: 'w1', session_id: 's1', command_id: id, action: 'turn.submit', args: { text } })
const rows = (value: unknown): Array<Record<string, unknown>> => value as Array<Record<string, unknown>>

async function staticFixture(): Promise<{ fixture: Fixture; root: string; dir: string }> {
  const dir = await tempDir('xh-static-')
  const root = join(dir, 'public')
  await mkdir(join(root, 'assets'), { recursive: true })
  await Bun.write(join(root, 'index.html'), '<!doctype html><html><body>phone shell</body></html>')
  await Bun.write(join(root, 'assets', 'app-abcdef12.js'), 'console.log(1)')
  await Bun.write(join(root, 'sw.js'), 'self.skip = 1')
  await Bun.write(join(dir, 'secret.txt'), 'TOP SECRET')
  await symlink(join(dir, 'secret.txt'), join(root, 'leak.txt'))
  return { fixture: await startFixture({ staticRoot: root }), root, dir }
}

describe('Hub server: connection admission', () => {
  test('refuses an unauthenticated WebSocket and never trusts forwarded identity headers', async () => {
    const f = await startFixture()
    const origin = `http://127.0.0.1:${f.port}`
    const bare = await fetch(f.url('/api/hub/v1/ws'), { headers: { upgrade: 'websocket', connection: 'Upgrade', origin, 'sec-websocket-key': 'dGhlIHNhbXBsZSBub25jZQ==', 'sec-websocket-version': '13', 'x-forwarded-user': 'owner', 'x-forwarded-for': '10.0.0.1' } })
    expect(bare.status).toBe(401)
    expect(await (await import('./hubHarness.js')).TestClient.connect(f.port, undefined)).toBeUndefined()
    expect(await (await import('./hubHarness.js')).TestClient.connect(f.port, 'nobody')).toBeUndefined()
  })

  test('refuses a wrong or missing Origin before authenticating', async () => {
    const f = await startFixture()
    const { TestClient } = await import('./hubHarness.js')
    const before = f.identity.authenticateCalls
    expect(await TestClient.connect(f.port, 'alice', { origin: 'https://evil.example.com' })).toBeUndefined()
    expect(await TestClient.connect(f.port, 'alice', { origin: `http://127.0.0.1:${f.port + 1}` })).toBeUndefined()
    expect(await TestClient.connect(f.port, 'alice', { origin: null })).toBeUndefined()
    expect(f.identity.authenticateCalls).toBe(before)
    expect(await TestClient.connect(f.port, 'alice')).toBeDefined()
  })

  test('hub.hello reports the protocol, principal and a scope summary', async () => {
    const f = await startFixture()
    const owner = await f.client('owner')
    const hello = await owner.ok('hub.hello')
    expect(hello).toMatchObject({ protocol_version: 1, hub_id: f.signer.hubId, stream_id: f.store.streamId, principal: { kind: 'user', id: 'owner', display_name: 'OWNER', hub_owner: true } })
    expect(hello.hub_scopes).toContain('grants.manage')
    const alice = await f.client('alice')
    expect((await alice.ok('hub.hello')).hub_scopes).toEqual([])
    const robot = await f.client('robot')
    const agent = await robot.ok('hub.hello')
    expect(agent.principal).toMatchObject({ kind: 'agent', id: 'robot' })
  })

  test('methods owned by other subsystems answer unsupported, never fake success', async () => {
    const f = await startFixture()
    const alice = await f.client('alice')
    for (const method of ['hub.pairing.create', 'hub.devices.list', 'hub.grants.update', 'supervisor.thread.open', 'hub.notifications.subscribe'] as const) {
      expect(await alice.fails(method)).toBe('unsupported')
    }
  })

  test('an injected method handler receives the authenticated context', async () => {
    const seen: string[] = []
    const f = await startFixture({ server: { methodHandlers: { 'hub.devices.list': ({ context }) => { seen.push(context.principal.id); return { devices: [] } } } } })
    const alice = await f.client('alice')
    expect(await alice.ok('hub.devices.list')).toEqual({ devices: [] })
    expect(seen).toEqual(['alice'])
  })

  test('refuses oversized frames, unknown methods and binary frames', async () => {
    const f = await startFixture()
    const alice = await f.client('alice')
    alice.sendRaw('x'.repeat(1_100_000))
    await alice.waitFor(frame => frame.kind === 'response' && frame.ok === false && frame.error.code === 'frame_too_large')
    alice.sendRaw(JSON.stringify({ v: 1, kind: 'request', id: 'z', method: 'turn.submit', params: {} }))
    await alice.waitFor(frame => frame.kind === 'response' && frame.id === 'z' && frame.ok === false && frame.error.code === 'invalid_request')
    expect(await alice.ok('hub.hello')).toBeDefined()
    alice.ws.send(new Uint8Array([1, 2, 3]))
    expect(await alice.closed()).toBe(1003)
  })

  test('rate limits per principal with an injectable clock, and per address before authentication', async () => {
    const f = await startFixture({ server: { rateLimit: { principal: { limit: 3, windowMs: 1000 }, api: { limit: 2, windowMs: 60_000 } } } })
    const alice = await f.client('alice')
    for (let index = 0; index < 3; index += 1) await alice.ok('hub.hello')
    expect(await alice.fails('hub.hello')).toBe('rate_limited')
    f.time.now += 2000
    expect(await alice.ok('hub.hello')).toBeDefined()
    // The WebSocket upgrade consumed one API slot; one ping is left, then the address is limited.
    const statuses: number[] = []
    for (let index = 0; index < 3; index += 1) statuses.push((await fetch(f.url('/api/identity/ping'))).status)
    expect(statuses).toContain(429)
  })
})

describe('Hub server: reads are filtered server-side', () => {
  test('workspaces.list hides what a session-only grant must not reveal, and exposes no endpoints or paths', async () => {
    const f = await startFixture()
    const guest = await f.client('guest')
    const guestList = await guest.ok('hub.workspaces.list')
    expect(guestList.workspaces).toEqual([{ id: 'w1', label: 'One', health: 'online', degraded: false, read_only: false, access: 'sessions', session_ids: ['s1'] }])
    const alice = await f.client('alice')
    const aliceList = rows((await alice.ok('hub.workspaces.list')).workspaces)
    expect(aliceList.map(row => row.id)).toEqual(['w1'])
    expect(aliceList[0]!.access).toBe('workspace')
    expect(aliceList[0]!.scopes).toContain('session.submit')
    const owner = await f.client('owner')
    const ownerText = JSON.stringify(await owner.ok('hub.workspaces.list'))
    expect(ownerText).toContain('"w2"')
    for (const secret of ['/proj', 'socket', 'endpoint', 'projectDir', 'ssh']) expect(ownerText).not.toContain(secret)
    f.health.set('w2', { state: 'stale', degraded: true })
    expect(rows((await owner.ok('hub.workspaces.list')).workspaces).find(row => row.id === 'w2')).toMatchObject({ health: 'stale', degraded: true })
  })

  test('snapshot returns only readable sessions', async () => {
    const f = await startFixture()
    const guest = await f.client('guest')
    const snapshot = await guest.ok('hub.snapshot')
    const workspaces = rows(snapshot.workspaces)
    expect(workspaces).toHaveLength(1)
    expect(rows(workspaces[0]!.sessions).map(session => session.id)).toEqual(['s1'])
    expect(await guest.fails('hub.snapshot', { workspace_ids: ['w2'] })).toBe('forbidden')
    const owner = await f.client('owner')
    expect(rows(rows((await owner.ok('hub.snapshot')).workspaces)[0]!.sessions).map(session => session.id).sort()).toEqual(['s1', 's2'])
  })

  test('events.read filters per session and refuses foreign scopes', async () => {
    const f = await startFixture()
    f.store.append({ workspace_id: 'w1', session_id: 's1', type: 'turn_begin', payload: { n: 1 } })
    f.store.append({ workspace_id: 'w1', session_id: 's2', type: 'turn_begin', payload: { n: 2 } })
    f.store.append({ workspace_id: 'w2', type: 'hub.workspace_status', payload: { health: 'online' } })
    const guest = await f.client('guest')
    const page = await guest.ok('hub.events.read', { workspace_ids: ['w1'] })
    expect(rows(page.events).map(event => event.session_id)).toEqual(['s1'])
    expect(await guest.fails('hub.events.read', { workspace_ids: ['w2'] })).toBe('forbidden')
    const alice = await f.client('alice')
    expect(rows((await alice.ok('hub.events.read', { workspace_ids: ['w1'] })).events)).toHaveLength(2)
    expect(await alice.fails('hub.events.read', { workspace_ids: ['w1'], limit: 100_000 })).toBe('invalid_request')
  })

  test('history is read through the pool with small pages and per-session access', async () => {
    const f = await startFixture()
    f.backend.handler = call => call.method === 'session.history'
      ? { state: 'accepted', result: { ok: true, history: { blob: Number(call.params.history_limit) > 2 ? 'x'.repeat(300_000) : 'small' } } }
      : undefined
    const alice = await f.client('alice')
    const page = await alice.ok('hub.history', { workspace_id: 'w1', session_id: 's1', limit: 8 })
    expect(page).toMatchObject({ session_id: 's1', history: { blob: 'small' } })
    expect(f.backend.callsTo('session.history').map(call => call.params.history_limit)).toEqual([8, 4, 2])
    expect(f.backend.callsTo('session.history')[0]!.params.session_key).toBe('k1')
    const guest = await f.client('guest')
    expect(await guest.ok('hub.history', { workspace_id: 'w1', session_id: 's1' })).toMatchObject({ session_id: 's1' })
    expect(await guest.fails('hub.history', { workspace_id: 'w1', session_id: 's2' })).toBe('forbidden')
    expect(await alice.fails('hub.history', { workspace_id: 'w1', session_id: 's1', limit: 26 })).toBe('invalid_request')
    expect(await alice.fails('hub.history', { workspace_id: 'w1', session_id: 'ghost' })).toBe('not_found')
  })

  test('usage asks each runtime once, never double counts, and keeps plan-paid cost separate', async () => {
    const f = await startFixture()
    f.backend.handler = call => call.method === 'usage.report'
      ? { state: 'accepted', result: { ok: true, models: [
        { model: 'm', profile: 'p', calls: 2, input_tokens: 100, output_tokens: 50, cost_usd: 1.5 },
        { model: 'm2', calls: 1, input_tokens: 10, output_tokens: 5, cost_usd: 9, cost_basis: 'subscription' },
      ] } }
      : undefined
    const owner = await f.client('owner')
    const usage = await owner.ok('hub.usage')
    expect(f.backend.callsTo('usage.report')).toHaveLength(1)
    expect(usage.runtimes).toBe(1)
    expect(usage.totals).toMatchObject({ input: 110, output: 55, billable_cost_usd: 1.5, subscription_price_estimate_usd: 9, subscription_label: 'Included in plan' })
    expect(rows(usage.by_model)).toHaveLength(2)
    const guest = await f.client('guest')
    expect((await guest.ok('hub.usage')).totals).toMatchObject({ input: 0, output: 0 })
    expect(await guest.fails('hub.usage', { workspace_ids: ['w1'] })).toBe('forbidden')
  })
})

describe('Hub server: events', () => {
  test('subscribe pushes batches, honours acks and refuses foreign subscriptions', async () => {
    const f = await startFixture()
    const alice = await f.client('alice')
    const viewer = await f.client('viewer')
    const subscribed = await alice.ok('hub.events.subscribe', { workspace_ids: ['w1'] })
    expect(subscribed.reset_required).toBe(false)
    f.store.append({ workspace_id: 'w1', session_id: 's1', type: 'turn_begin', payload: { n: 1 } })
    const batch = await alice.event('hub.event_batch', payload => rows(payload.events).length > 0)
    expect(rows(batch.payload.events)[0]).toMatchObject({ type: 'turn_begin', session_id: 's1' })
    await alice.ok('hub.events.ack', { subscription_id: String(subscribed.subscription_id), cursor: String(batch.payload.through_cursor) })
    expect(await viewer.fails('hub.events.unsubscribe', { subscription_id: String(subscribed.subscription_id) })).toBe('not_found')
    expect(await viewer.fails('hub.events.ack', { subscription_id: String(subscribed.subscription_id), cursor: 'x' })).toBe('not_found')
    await alice.ok('hub.events.unsubscribe', { subscription_id: String(subscribed.subscription_id) })
    expect(await alice.fails('hub.events.unsubscribe', { subscription_id: String(subscribed.subscription_id) })).toBe('not_found')
  })

  test('a session-only subscriber never receives other sessions and cannot subscribe to other workspaces', async () => {
    const f = await startFixture()
    const guest = await f.client('guest')
    await guest.ok('hub.events.subscribe', { workspace_ids: ['w1'] })
    expect(await guest.fails('hub.events.subscribe', { workspace_ids: ['w2'] })).toBe('forbidden')
    f.store.append({ workspace_id: 'w1', session_id: 's2', type: 'turn_begin', payload: { secret: 'other session' } })
    f.store.append({ workspace_id: 'w1', session_id: 's1', type: 'turn_end', payload: {} })
    await guest.event('hub.event_batch', payload => rows(payload.events).some(event => event.type === 'turn_end'))
    expect(JSON.stringify(guest.frames)).not.toContain('other session')
  })

  test('a stale or foreign cursor is a gap, not a silent empty replay', async () => {
    const f = await startFixture()
    f.store.append({ workspace_id: 'w1', session_id: 's1', type: 'turn_begin', payload: {} })
    const alice = await f.client('alice')
    expect(await alice.fails('hub.events.subscribe', { workspace_ids: ['w1'], after_cursor: 'garbage' })).toBe('invalid_request')
    const first = await alice.ok('hub.events.subscribe', { workspace_ids: ['w1'] })
    const other = await alice.ok('hub.events.read', { workspace_ids: ['w1'], after_cursor: String(first.watermark) })
    expect(other.events).toEqual([])
    expect(await alice.fails('hub.events.read', { workspace_ids: ['w2'], after_cursor: String(first.watermark) })).toBe('forbidden')
  })
})

describe('Hub server: commands over the socket', () => {
  test('a command runs once, replays its recorded outcome, and reports it to the caller', async () => {
    const f = await startFixture()
    const alice = await f.client('alice')
    const first = await alice.ok('hub.command', submit('c1'))
    expect(first).toMatchObject({ ok: true, command_id: 'c1', state: 'accepted' })
    const event = await alice.event('hub.command_result', payload => payload.command_id === 'c1')
    expect(event.payload).toMatchObject({ workspace_id: 'w1', session_id: 's1', state: 'accepted' })
    expect(await alice.ok('hub.command', submit('c1'))).toEqual(first)
    expect(f.backend.callsTo('turn.submit')).toHaveLength(1)
    expect(await alice.fails('hub.command', submit('c1', 'different text'))).toBe('submission_conflict')
    expect(await alice.ok('hub.command.status', { workspace_id: 'w1', session_id: 's1', command_id: 'c1' })).toEqual(first)
  })

  test('the command result is only pushed to the caller\'s own sockets', async () => {
    const f = await startFixture()
    const alice = await f.client('alice')
    const phone = await f.client('alice')
    const owner = await f.client('owner')
    await alice.ok('hub.command', submit('c1'))
    await phone.event('hub.command_result', payload => payload.command_id === 'c1')
    await Bun.sleep(30)
    expect(owner.events('hub.command_result')).toEqual([])
  })

  test('scope denial and agent interaction refusal surface as forbidden', async () => {
    const f = await startFixture()
    const viewer = await f.client('viewer')
    expect(await viewer.fails('hub.command', submit('c1'))).toBe('forbidden')
    const robot = await f.client('robot')
    expect(await robot.fails('hub.command', { workspace_id: 'w1', session_id: 's1', command_id: 'p1', action: 'permission.respond', args: { request_id: 'r1', response: 'approve' } })).toBe('forbidden')
    expect(await robot.fails('hub.command', { workspace_id: 'w1', session_id: 's1', command_id: 'q1', action: 'question.respond', args: { request_id: 'r1', answers: { a: 'b' } } })).toBe('forbidden')
    expect(f.backend.calls.filter(call => call.method !== 'session.active_list')).toEqual([])
    expect((await robot.ok('hub.command', submit('ok'))).state).toBe('accepted')
  })

  test('the forwarded params carry a hub_auth envelope that verifies', async () => {
    const f = await startFixture()
    const alice = await f.client('alice')
    await alice.ok('hub.command', submit('c1'))
    const call = f.backend.callsTo('turn.submit')[0]!
    const { hub_auth: envelope, ...params } = call.params
    const verified = verifyHubAuth(envelope, { publicKey: f.signer.publicKeyPem, method: 'turn.submit', params, now: f.time.now })
    expect(verified.ok).toBe(true)
    if (verified.ok) expect(verified.claims).toMatchObject({ principal_id: 'alice', workspace_id: 'w1', session_id: 's1' })
  })

  test('an audit failure refuses the mutation', async () => {
    const f = await startFixture()
    const alice = await f.client('alice')
    f.identity.failAudit = true
    expect(await alice.fails('hub.command', submit('c1'))).toBe('unavailable')
    expect(f.backend.callsTo('turn.submit')).toEqual([])
  })

  test('an unknown outcome is reported and never replayed', async () => {
    const f = await startFixture()
    f.backend.handler = call => call.method === 'turn.submit' ? { state: 'unknown', error: { code: 'outcome_unknown', message: 'no answer' } } : undefined
    const alice = await f.client('alice')
    const first = await alice.ok('hub.command', submit('c1'))
    expect(first).toMatchObject({ ok: false, state: 'unknown' })
    f.backend.handler = undefined
    expect(await alice.ok('hub.command', submit('c1'))).toEqual(first)
    expect(await alice.ok('hub.command.status', { workspace_id: 'w1', session_id: 's1', command_id: 'c1' })).toMatchObject({ state: 'unknown' })
    expect(f.backend.callsTo('turn.submit')).toHaveLength(1)
  })

  test('without a router commands are unsupported, not silently accepted', async () => {
    const f = await startFixture({ router: false })
    const alice = await f.client('alice')
    expect(await alice.fails('hub.command', submit('c1'))).toBe('unsupported')
    expect(await alice.fails('hub.command.status', { workspace_id: 'w1', command_id: 'c1' })).toBe('unsupported')
  })
})

describe('Hub server: presence', () => {
  const target = { workspace_id: 'w1', session_id: 's1' }

  test('is broadcast to authorized subscribers, collapsed per user, and expires', async () => {
    const f = await startFixture()
    const viewer = await f.client('viewer')
    await viewer.ok('hub.events.subscribe', { workspace_ids: ['w1'], session_ids: ['s1'] })
    const laptop = await f.client('alice')
    const phone = await f.client('alice')
    expect(await laptop.ok('hub.presence.update', { ...target, state: 'viewing' })).toEqual({ ttl_ms: 30_000 })
    await phone.ok('hub.presence.update', { ...target, state: 'typing' })
    await until(() => viewer.events('hub.presence').length >= 2)
    const last = viewer.events('hub.presence').at(-1)!.payload
    expect(rows(last.participants)).toHaveLength(1)
    expect(rows(last.participants)[0]).toMatchObject({ principal_id: 'alice', display_name: 'ALICE', state: 'typing' })
    expect(Object.keys(rows(last.participants)[0]!).sort()).toEqual(['display_name', 'principal_id', 'principal_kind', 'state', 'updated_at'])

    f.time.now += 31_000
    f.server.sweepPresence()
    await until(() => rows(viewer.events('hub.presence').at(-1)!.payload.participants).length === 0 || undefined)
  })

  test('refuses drafts and unknown fields instead of dropping them', async () => {
    const f = await startFixture()
    const alice = await f.client('alice')
    expect(await alice.fails('hub.presence.update', { ...target, state: 'typing', draft: 'half-written secret' })).toBe('invalid_request')
    expect(await alice.fails('hub.presence.update', { ...target, state: 'typing', text: 'abc' })).toBe('invalid_request')
    expect(await alice.fails('hub.presence.update', { ...target, state: 'dancing' })).toBe('invalid_request')
    expect(JSON.stringify(alice.frames)).not.toContain('half-written secret')
  })

  test('is only shown to people who may read that session, and a disconnect clears the entry', async () => {
    const f = await startFixture()
    const guest = await f.client('guest')
    await guest.ok('hub.events.subscribe', { workspace_ids: ['w1'] })
    const viewer = await f.client('viewer')
    await viewer.ok('hub.events.subscribe', { workspace_ids: ['w1'], session_ids: ['s2'] })
    const alice = await f.client('alice')
    await alice.ok('hub.presence.update', target)
    await guest.event('hub.presence', payload => payload.session_id === 's1')
    await Bun.sleep(30)
    expect(viewer.events('hub.presence')).toEqual([])
    expect(await guest.fails('hub.presence.update', { workspace_id: 'w1', session_id: 's2' })).toBe('forbidden')
    alice.close()
    await until(() => rows(guest.events('hub.presence').at(-1)!.payload.participants).length === 0 || undefined)
  })
})

describe('Hub server: backpressure and revocation', () => {
  test('a client that cannot keep up is closed with 1013', async () => {
    const f = await startFixture({ server: { maxBufferedBytes: 1024 } })
    f.projections.set('w1', replaceSessionSummaries(emptyProjection('w1'), Array.from({ length: 60 }, (_, index) => ({ id: `s${index}`, title: `Task number ${index}`, status: 'idle' })), new Date().toISOString()))
    const owner = await f.client('owner')
    owner.sendRaw(JSON.stringify({ v: 1, kind: 'request', id: 'big', method: 'hub.snapshot', params: {} }))
    expect(await owner.closed()).toBe(1013)
  })

  test('revocation sends hub.revoked, then closes only the matching sockets', async () => {
    const f = await startFixture()
    const alice = await f.client('alice')
    const viewer = await f.client('viewer')
    await alice.ok('hub.events.subscribe', { workspace_ids: ['w1'] })
    f.identity.revoke({ principalId: 'alice', deviceId: 'some-other-device', reason: 'x' })
    f.identity.revoke({ principalId: 'alice', browserSessionId: 'other-session', reason: 'x' })
    await Bun.sleep(30)
    expect(alice.closeCode).toBeUndefined()
    f.identity.revoke({ principalId: 'alice', deviceId: 'dev-alice', reason: 'device revoked' })
    expect(await alice.closed()).toBe(1008)
    expect(alice.events('hub.revoked')).toHaveLength(1)
    expect(JSON.stringify(alice.events('hub.revoked')[0])).not.toContain('device revoked')
    expect(viewer.closeCode).toBeUndefined()
    expect(await viewer.ok('hub.hello')).toBeDefined()
  })

  test('revocation by browser session closes that session\'s sockets', async () => {
    const f = await startFixture()
    const viewer = await f.client('viewer')
    f.identity.revoke({ principalId: 'viewer', browserSessionId: 'bs-viewer', reason: 'logout' })
    expect(await viewer.closed()).toBe(1008)
    expect(viewer.events('hub.revoked')).toHaveLength(1)
  })

  test('stopping the server closes sockets with 1001 (read from the raw close frame)', async () => {
    const f = await startFixture()
    // The Bun WebSocket client rewrites a server 1001 to 1000, so read the wire bytes directly.
    let received = Buffer.alloc(0)
    const socket = await Bun.connect({
      hostname: '127.0.0.1', port: f.port,
      socket: {
        data(_socket, chunk) { received = Buffer.concat([received, chunk]) },
        open(open) {
          open.write([
            'GET /api/hub/v1/ws HTTP/1.1', `Host: 127.0.0.1:${f.port}`, 'Upgrade: websocket', 'Connection: Upgrade',
            `Origin: http://127.0.0.1:${f.port}`, 'Cookie: who=alice', 'Sec-WebSocket-Version: 13',
            'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==', '', '',
          ].join('\r\n'))
        },
      },
    })
    registerCleanup(() => { socket.end() })
    await until(() => received.includes('\r\n\r\n') || undefined)
    expect(received.toString('latin1').startsWith('HTTP/1.1 101')).toBe(true)
    await until(() => f.server.connectionCount === 1 || undefined)
    await f.server.stop()
    const closeCode = (): number | undefined => {
      let offset = received.indexOf('\r\n\r\n') + 4
      while (offset + 2 <= received.length) {
        const opcode = received[offset]! & 0x0f
        const length = received[offset + 1]! & 0x7f
        if (length >= 126) return undefined
        if (offset + 2 + length > received.length) return undefined
        if (opcode === 0x8) return length >= 2 ? received.readUInt16BE(offset + 2) : 1005
        offset += 2 + length
      }
      return undefined
    }
    expect(await until(closeCode)).toBe(1001)
  })
})

describe('Hub server: HTTP', () => {
  test('healthz needs no auth, reports readiness and degraded workspaces, and leaks nothing', async () => {
    const f = await startFixture()
    const ok = await fetch(f.url('/healthz'))
    expect(ok.status).toBe(200)
    expect(await ok.json()).toEqual({ status: 'ok', live: true, ready: true, identity: 'configured', workspaces: { total: 2, online: 2, degraded: 0 } })
    f.health.set('w2', { state: 'offline', degraded: false })
    const degraded = await (await fetch(f.url('/healthz'))).json() as Record<string, unknown>
    expect(degraded).toMatchObject({ status: 'degraded', workspaces: { total: 2, online: 1, degraded: 1 } })
    expect(JSON.stringify(degraded)).not.toContain('w1')
    expect((await fetch(f.url('/healthz'), { method: 'POST' })).status).toBe(405)
    expect((await fetch(f.url('/healthz'), { method: 'HEAD' })).status).toBe(200)
  })

  test('without an identity port the Hub fails closed and says so', async () => {
    const f = await startFixture({ identity: false })
    expect(f.server.identityConfigured).toBe(false)
    const health = await fetch(f.url('/healthz'))
    expect(await health.json()).toMatchObject({ status: 'identity_unavailable', ready: false, identity: 'missing' })
    for (const path of ['/', '/index.html', '/api/identity/ping', '/api/hub/v1/ws', '/anything']) {
      const response = await fetch(f.url(path))
      expect(response.status).toBe(503)
      expect(await response.text()).toContain('identity_unavailable')
    }
    const { TestClient } = await import('./hubHarness.js')
    expect(await TestClient.connect(f.port, 'alice')).toBeUndefined()
  })

  test('identity routes are delegated first and still get the security headers', async () => {
    const f = await startFixture()
    const response = await fetch(f.url('/api/identity/ping'))
    expect(await response.text()).toBe('pong')
    expect(response.headers.get('x-content-type-options')).toBe('nosniff')
    expect((await fetch(f.url('/api/unknown'))).status).toBe(404)
  })

  test('serves the phone bundle with strict headers and correct caching', async () => {
    const { fixture: f } = await staticFixture()
    const index = await fetch(f.url('/'))
    expect(index.status).toBe(200)
    expect(await index.text()).toContain('phone shell')
    expect(index.headers.get('cache-control')).toBe('no-store')
    expect(index.headers.get('x-content-type-options')).toBe('nosniff')
    expect(index.headers.get('referrer-policy')).toBe('no-referrer')
    const csp = index.headers.get('content-security-policy') ?? ''
    expect(csp).toContain("frame-ancestors 'none'")
    expect(csp).toContain("default-src 'self'")
    expect(csp).toContain("object-src 'none'")
    expect(csp).toContain(`ws://127.0.0.1:${f.port}`)
    expect(csp).not.toContain("'unsafe-inline'")
    expect(csp).not.toContain("'unsafe-eval'")

    const hashed = await fetch(f.url('/assets/app-abcdef12.js'))
    expect(hashed.status).toBe(200)
    expect(hashed.headers.get('cache-control')).toBe('public, max-age=31536000, immutable')
    expect(hashed.headers.get('content-type')).toContain('javascript')
    expect((await fetch(f.url('/sw.js'))).headers.get('cache-control')).toBe('no-cache')

    const route = await fetch(f.url('/inbox/task-1'), { headers: { accept: 'text/html' } })
    expect(await route.text()).toContain('phone shell')
    expect(route.headers.get('cache-control')).toBe('no-store')
    expect((await fetch(f.url('/assets/missing.js'))).status).toBe(404)
    expect((await fetch(f.url('/'), { method: 'POST' })).status).toBe(405)
  })

  test('static serving cannot escape its root by traversal or symlink', async () => {
    const { fixture: f } = await staticFixture()
    const traversal = await fetch(f.url('/..%2f..%2fsecret.txt'))
    expect(traversal.status).toBe(400)
    expect(await traversal.text()).not.toContain('TOP SECRET')
    const encoded = await fetch(f.url('/%2e%2e/secret.txt'))
    expect(await encoded.text()).not.toContain('TOP SECRET')
    const leak = await fetch(f.url('/leak.txt'))
    expect(leak.status).toBe(404)
    expect(await leak.text()).not.toContain('TOP SECRET')
    expect((await fetch(f.url('/%00'))).status).toBe(400)
  })

  test('without a phone bundle the shell is a clear 404', async () => {
    const f = await startFixture()
    const response = await fetch(f.url('/'))
    expect(response.status).toBe(404)
    expect(await response.text()).toContain('phone bundle')
  })

  test('a bundle directory that is unreadable does not take the Hub down', async () => {
    const { fixture: f, root } = await staticFixture()
    await chmod(root, 0o000)
    try {
      const response = await fetch(f.url('/'))
      expect([404, 500]).toContain(response.status)
      expect((await fetch(f.url('/healthz'))).status).toBe(200)
    } finally {
      await chmod(root, 0o755)
    }
  })
})
