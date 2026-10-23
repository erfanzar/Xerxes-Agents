// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ServerWebSocket } from 'bun'
import { afterEach, describe, expect, test } from 'bun:test'

import type { AuthenticatedContext } from '../src/hub/identityPort.js'
import { VOICE_LIMITS, VOICE_PATH } from '../src/hub/voice/protocol.js'
import { createHubVoiceRoute, voiceSocketData, type HubVoiceRoute, type VoiceSocketData } from '../src/hub/voice/route.js'
import { FakeIdentity, standardUsers } from './hubHarness.js'
import { Rig, control, micFrame, settle } from './hubVoiceHarness.js'

interface FakeWs {
  data: VoiceSocketData
  readyState: number
  sent: (string | Uint8Array)[]
  closed: { code: number; reason: string } | undefined
  send(message: string | Uint8Array): number
  getBufferedAmount(): number
  close(code: number, reason: string): void
}

function fakeWs(context: AuthenticatedContext): FakeWs {
  const ws: FakeWs = {
    data: voiceSocketData(context), readyState: 1, sent: [], closed: undefined,
    send(message) { ws.sent.push(message); return 1 },
    getBufferedAmount: () => 0,
    close(code, reason) { ws.closed = { code, reason }; ws.readyState = 3 },
  }
  return ws
}

const asBun = (ws: FakeWs): ServerWebSocket<VoiceSocketData> => ws as unknown as ServerWebSocket<VoiceSocketData>
const texts = (ws: FakeWs): Record<string, unknown>[] => ws.sent.filter((m): m is string => typeof m === 'string').map(m => JSON.parse(m) as Record<string, unknown>)

const routes: HubVoiceRoute[] = []
afterEach(() => { for (const route of routes.splice(0)) route.close() })

function setup(options: { maxConnectionsPerPrincipal?: number } = {}) {
  const rig = new Rig()
  const identity = standardUsers(new FakeIdentity())
  const route = createHubVoiceRoute({ identity, supervisor: rig.bridge, config: rig.config, clock: rig.clock, ...options })
  routes.push(route)
  return { rig, identity, route }
}

const request = (): Request => new Request(`http://hub.test${VOICE_PATH}`)
const info = (who?: string) => ({ method: 'GET', url: `http://hub.test${VOICE_PATH}`, ...(who === undefined ? {} : { cookie: `who=${who}` }) })

describe('createHubVoiceRoute', () => {
  test('exposes the documented path and frame limits', () => {
    const { route } = setup()
    expect(route.path).toBe('/api/hub/v1/voice')
    expect(route.limits.maxPayloadLength).toBeGreaterThan(VOICE_LIMITS.headerBytes + VOICE_LIMITS.maxFramePayloadBytes)
    expect(route.limits.backpressureLimit).toBe(VOICE_LIMITS.maxBufferedBytes)
  })

  test('upgrade is refused for anonymous callers, unknown people, agents and other paths', async () => {
    const { route } = setup()
    expect(await route.authenticateUpgrade(request(), info())).toBeNull()
    expect(await route.authenticateUpgrade(request(), info('nobody'))).toBeNull()
    expect(await route.authenticateUpgrade(request(), info('robot'))).toBeNull()
    expect(await route.authenticateUpgrade(new Request('http://hub.test/api/hub/v1/ws'), info('alice'))).toBeNull()
    const ok = await route.authenticateUpgrade(request(), info('alice'))
    expect(ok?.principal.id).toBe('alice')
  })

  test('a full push-to-talk exchange runs through the Bun handlers', async () => {
    const { rig, route, identity } = setup()
    const ws = fakeWs(identity.context('alice'))
    route.websocket.open(asBun(ws))
    expect(route.sessionCount).toBe(1)

    route.websocket.message(asBun(ws), control({ type: 'capabilities' }))
    await settle()
    expect(texts(ws).at(-1)).toMatchObject({ type: 'capabilities', capabilities: { enabled: true, modes: ['push_to_talk'] } })

    route.websocket.message(asBun(ws), control({ type: 'start', mode: 'push_to_talk', conversation_id: 'thread-1', utterance_id: 'u1', codec: 'webm-opus' }))
    await settle()
    const generation = Number(texts(ws).filter(m => m.type === 'state').at(-1)?.generation)
    route.websocket.message(asBun(ws), Buffer.from(micFrame(generation, 0, 64)))
    route.websocket.message(asBun(ws), control({ type: 'commit', utterance_id: 'u1' }))
    await settle(20)
    expect(rig.stt.calls).toHaveLength(1)
    expect(texts(ws).find(m => m.type === 'transcript')).toMatchObject({ text: 'hello world', final: true })
    expect(rig.errors).toEqual([])
  })

  test('closing a socket disposes its session and releases its lease and listener', async () => {
    const { rig, route, identity } = setup()
    const ws = fakeWs(identity.context('alice'))
    route.websocket.open(asBun(ws))
    route.websocket.message(asBun(ws), control({ type: 'start', mode: 'push_to_talk', conversation_id: 'c1', utterance_id: 'u1', codec: 'webm-opus' }))
    await settle()
    expect(rig.bridge.listenerCount('c1')).toBe(1)
    route.websocket.close(asBun(ws), 1006, '')
    expect(route.sessionCount).toBe(0)
    expect(rig.bridge.listenerCount('c1')).toBe(0)
    expect(rig.leases.size).toBe(0)
  })

  test('two people on one conversation: only one can hold the speaker lease across sockets', async () => {
    const { route, identity } = setup()
    const alice = fakeWs(identity.context('alice'))
    const owner = fakeWs(identity.context('owner'))
    route.websocket.open(asBun(alice))
    route.websocket.open(asBun(owner))
    const start = (id: string) => control({ type: 'start', mode: 'push_to_talk', conversation_id: 'shared', utterance_id: id, codec: 'webm-opus' })
    route.websocket.message(asBun(alice), start('ua'))
    route.websocket.message(asBun(owner), start('ub'))
    await settle()
    const busy = [alice, owner].filter(ws => texts(ws).some(m => m.type === 'error' && m.code === 'VOICE_BUSY'))
    expect(busy).toHaveLength(1)
  })

  test('connection caps are enforced per person', () => {
    const { route, identity } = setup({ maxConnectionsPerPrincipal: 1 })
    const first = fakeWs(identity.context('alice'))
    const second = fakeWs(identity.context('alice'))
    route.websocket.open(asBun(first))
    route.websocket.open(asBun(second))
    expect(first.closed).toBeUndefined()
    expect(second.closed?.code).toBe(1013)
    expect(route.sessionCount).toBe(1)
  })

  test('revoking a device or browser session closes its voice socket', async () => {
    const { route, identity } = setup()
    const alice = fakeWs(identity.context('alice'))
    const owner = fakeWs(identity.context('owner'))
    route.websocket.open(asBun(alice))
    route.websocket.open(asBun(owner))
    identity.revoke({ principalId: 'alice', reason: 'device revoked' })
    expect(alice.closed).toEqual({ code: 1008, reason: 'access revoked' })
    expect(owner.closed).toBeUndefined()
    identity.revoke({ principalId: 'owner', browserSessionId: 'bs-owner', reason: 'logout' })
    expect(owner.closed?.code).toBe(1008)
  })

  test('close() ends every session and refuses new upgrades', async () => {
    const { route, identity } = setup()
    const ws = fakeWs(identity.context('alice'))
    route.websocket.open(asBun(ws))
    route.close()
    expect(ws.closed?.code).toBe(1001)
    expect(route.sessionCount).toBe(0)
    expect(await route.authenticateUpgrade(request(), info('alice'))).toBeNull()
    const late = fakeWs(identity.context('alice'))
    route.websocket.open(asBun(late))
    expect(late.closed?.code).toBe(1001)
  })

  test('a message for a socket that never opened a session is ignored', () => {
    const { route, identity } = setup()
    expect(() => route.websocket.message(asBun(fakeWs(identity.context('alice'))), 'x')).not.toThrow()
  })
})
