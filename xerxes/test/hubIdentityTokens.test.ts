// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'

import { createHubIdentity } from '../src/hub/identity/index.js'
import type { RevocationListener } from '../src/hub/identityPort.js'
import {
  DAY,
  FINGERPRINT,
  MINUTE,
  ORIGIN,
  SECOND,
  authenticate,
  cleanupAll,
  errorCode,
  errorOf,
  http,
  inviteUser,
  makeHarness,
  must,
  pairDevice,
  refresh,
  secondInstance,
  sessionCookie,
  sessionOf,
  tempPaths,
} from './hubIdentityHarness.js'

afterEach(cleanupAll)

type RevocationEvent = Parameters<RevocationListener>[0]

describe('refresh rotation', () => {
  test('each refresh issues a new refresh token and a working session', async () => {
    const h = await makeHarness()
    const s0 = await pairDevice(h)
    const r1 = await refresh(h, s0.refresh)
    expect(r1.status).toBe(200)
    const s1 = sessionOf(r1)
    expect(s1.refresh).not.toBe(s0.refresh)
    expect(s1.session).not.toBe(s0.session)
    expect(await authenticate(h, s1)).not.toBeNull()
    const r2 = await refresh(h, s1.refresh)
    expect(r2.status).toBe(200)
    expect(sessionOf(r2).refresh).not.toBe(s1.refresh)
  })

  test('a retired refresh token from a second cookie jar starts its own branch; the device stays signed in', async () => {
    const h = await makeHarness()
    const events: RevocationEvent[] = []
    h.identity.onRevoked(event => events.push(event))
    const s0 = await pairDevice(h)
    const s1 = sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(11 * SECOND)
    const fork = await refresh(h, s0.refresh)
    expect(fork.status).toBe(200)
    const branch = sessionOf(fork)
    expect(branch.refresh).not.toBe(s1.refresh)
    expect(await authenticate(h, branch)).not.toBeNull()
    expect(await authenticate(h, s1)).not.toBeNull()
    expect((await refresh(h, s1.refresh)).status).toBe(200)
    expect((await refresh(h, branch.refresh)).status).toBe(200)
    expect(events).toHaveLength(0)
    expect(h.admin.listDevices().find(device => device.id === s0.deviceId)?.status).toBe('active')
    expect(h.admin.listAudit().some(event => event.action === `identity.family.fork:${s0.deviceId}`)).toBe(true)
  })

  test('a token that already started a branch is refused once that branch is in use, without revoking the device', async () => {
    const h = await makeHarness()
    const events: RevocationEvent[] = []
    h.identity.onRevoked(event => events.push(event))
    const s0 = await pairDevice(h)
    const s1 = sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(11 * SECOND)
    const branch = sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(3 * SECOND)
    // A racing retry inside the grace window gets the same branch.
    const retry = sessionOf(await refresh(h, s0.refresh))
    expect(retry.refresh).toBe(branch.refresh)
    h.clock.advance(11 * SECOND)
    // The holder uses its branch: the spent token can no longer collect anything.
    const used = await refresh(h, branch.refresh)
    expect(used.status).toBe(200)
    const late = await refresh(h, s0.refresh)
    expect(late.status).toBe(401)
    expect(errorOf(late)).toMatchObject({ code: 'unauthorized' })
    // A spent-token refusal leaves the cookie jar alone: a concurrent winner may have just set newer cookies.
    expect(late.setCookies).toHaveLength(0)
    expect(events).toHaveLength(0)
    expect(await authenticate(h, s1)).not.toBeNull()
    expect((await refresh(h, sessionOf(used).refresh)).status).toBe(200)
  })

  test('a fork reply lost while suspended can be collected again while its branch is unused, within three minutes', async () => {
    // A short browser session so the replay has to mint a fresh one inside the replay window.
    const h = await makeHarness({ limits: { sessionTtlMs: MINUTE } })
    const events: RevocationEvent[] = []
    h.identity.onRevoked(event => events.push(event))
    const s0 = await pairDevice(h)
    sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(11 * SECOND)
    // The fork reply never reaches the phone.
    const lost = sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(2 * MINUTE)
    const again = await refresh(h, s0.refresh)
    expect(again.status).toBe(200)
    const recovered = sessionOf(again)
    expect(recovered.refresh).toBe(lost.refresh)
    // The remembered browser session lapsed meanwhile, so a fresh one is minted on the same branch.
    expect(recovered.session).not.toBe(lost.session)
    expect(await authenticate(h, recovered)).not.toBeNull()
    expect(h.admin.listAudit().some(event => event.action === `identity.family.fork_replay:${s0.deviceId}`)).toBe(true)
    expect((await refresh(h, recovered.refresh)).status).toBe(200)
    expect(events).toHaveLength(0)
  })

  test('a lost fork reply is not collectable after three minutes', async () => {
    const h = await makeHarness()
    const s0 = await pairDevice(h)
    sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(11 * SECOND)
    sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(3 * MINUTE + SECOND)
    const late = await refresh(h, s0.refresh)
    expect(late.status).toBe(401)
    expect(errorOf(late)).toMatchObject({ code: 'unauthorized' })
    expect(h.admin.listDevices().find(device => device.id === s0.deviceId)?.status).toBe('active')
  })

  test('a lost fork reply can be collected only once; a second replay is refused', async () => {
    const h = await makeHarness()
    const s0 = await pairDevice(h)
    sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(11 * SECOND)
    const lost = sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(MINUTE)
    const first = await refresh(h, s0.refresh)
    expect(first.status).toBe(200)
    expect(sessionOf(first).refresh).toBe(lost.refresh)
    h.clock.advance(SECOND)
    const second = await refresh(h, s0.refresh)
    expect(second.status).toBe(401)
    expect(errorOf(second)).toMatchObject({ code: 'unauthorized' })
    expect(h.admin.listAudit().filter(event => event.action === `identity.family.fork_replay:${s0.deviceId}`)).toHaveLength(1)
    // The branch itself still works for whoever collected it.
    expect((await refresh(h, lost.refresh)).status).toBe(200)
    expect(h.admin.listDevices().find(device => device.id === s0.deviceId)?.status).toBe('active')
  })

  test('a lost fork reply is not collectable once its branch was retired by the family limit', async () => {
    const h = await makeHarness()
    const s0 = await pairDevice(h)
    // Rotate so several older tokens exist that can each start a branch.
    const retired: string[] = [s0.refresh]
    let current = s0.refresh
    for (let index = 0; index < 5; index += 1) {
      h.clock.advance(SECOND)
      current = sessionOf(await refresh(h, current)).refresh
      retired.push(current)
    }
    retired.pop()
    h.clock.advance(11 * SECOND)
    // The first fork is the one whose reply is "lost".
    sessionOf(await refresh(h, retired[0]!))
    // The main family stays in use, so the unused lost branch becomes the least recently used one.
    h.clock.advance(SECOND)
    current = sessionOf(await refresh(h, current)).refresh
    // Two more branches reach the four-family limit; a third retires the lost branch.
    for (const token of retired.slice(1, 4)) {
      h.clock.advance(SECOND)
      expect((await refresh(h, token)).status).toBe(200)
    }
    expect((await refresh(h, current)).status).toBe(200)
    h.clock.advance(15 * SECOND)
    const replay = await refresh(h, retired[0]!)
    expect(replay.status).toBe(401)
    expect(h.admin.listAudit().some(event => event.action === `identity.family.fork_replay:${s0.deviceId}`)).toBe(false)
  })

  test('a lost fork reply is not collectable after logout', async () => {
    const h = await makeHarness()
    const s0 = await pairDevice(h)
    const s1 = sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(11 * SECOND)
    sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(MINUTE)
    const out = await http(h, 'POST', 'session/logout', undefined, { cookie: s1.cookie })
    expect(out.status).toBe(200)
    const replay = await refresh(h, s0.refresh)
    expect(replay.status).toBe(401)
    expect(h.admin.listAudit().some(event => event.action === `identity.family.fork_replay:${s0.deviceId}`)).toBe(false)
  })

  test('a lost fork reply is not collectable once the device is revoked', async () => {
    const h = await makeHarness()
    const s0 = await pairDevice(h)
    sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(11 * SECOND)
    sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(MINUTE)
    h.admin.revokeDevice(s0.deviceId, 'lost phone')
    const late = await refresh(h, s0.refresh)
    expect(late.status).toBe(401)
    // Revocation refusals still clear the cookies.
    expect(late.setCookies.every(line => line.includes('Max-Age=0'))).toBe(true)
  })

  test('a device keeps at most four branches; the least recently used one is retired alone', async () => {
    const h = await makeHarness()
    const events: RevocationEvent[] = []
    h.identity.onRevoked(event => events.push(event))
    const s0 = await pairDevice(h)
    // Rotate the original family several times so its old tokens can each start a branch.
    const retired: string[] = [s0.refresh]
    let current = s0.refresh
    for (let index = 0; index < 5; index += 1) {
      h.clock.advance(SECOND)
      current = sessionOf(await refresh(h, current)).refresh
      retired.push(current)
    }
    retired.pop()
    h.clock.advance(11 * SECOND)
    const branches = []
    for (const token of retired.slice(0, 4)) {
      h.clock.advance(SECOND)
      const result = await refresh(h, token)
      expect(result.status).toBe(200)
      branches.push(sessionOf(result))
    }
    // The original family was the least recently used and has been retired; the device is still active.
    expect((await refresh(h, current)).status).toBe(401)
    expect(events.every(event => event.browserSessionId !== undefined && event.reason === 'family_limit')).toBe(true)
    expect(h.admin.listDevices().find(device => device.id === s0.deviceId)?.status).toBe('active')
    for (const branch of branches) expect(await authenticate(h, branch)).not.toBeNull()
  })

  test('an idle branch expires alone while another branch of the device is in use', async () => {
    const h = await makeHarness()
    const s0 = await pairDevice(h)
    const s1 = sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(11 * SECOND)
    const idle = sessionOf(await refresh(h, s0.refresh))
    let active = s1.refresh
    for (let day = 1; day <= 8; day += 1) {
      h.clock.advance(DAY)
      const result = await refresh(h, active)
      expect(result.status).toBe(200)
      active = sessionOf(result).refresh
    }
    expect((await refresh(h, idle.refresh)).status).toBe(401)
    expect(h.admin.listDevices().find(device => device.id === s0.deviceId)?.status).toBe('active')
    expect((await refresh(h, active)).status).toBe(200)
  })

  test('grace window: a retried request within it gets the same successor, nothing is revoked', async () => {
    const h = await makeHarness()
    const events: RevocationEvent[] = []
    h.identity.onRevoked(event => events.push(event))
    const s0 = await pairDevice(h)
    const first = sessionOf(await refresh(h, s0.refresh))
    h.clock.advance(3 * SECOND)
    const retry = await refresh(h, s0.refresh)
    expect(retry.status).toBe(200)
    const second = sessionOf(retry)
    expect(second.refresh).toBe(first.refresh)
    expect(second.session).toBe(first.session)
    expect(events).toHaveLength(0)
    expect((await refresh(h, first.refresh)).status).toBe(200)
  })

  test('grace window: concurrent refreshes with one token agree on a single successor', async () => {
    const h = await makeHarness()
    const s0 = await pairDevice(h)
    const results = await Promise.all([refresh(h, s0.refresh), refresh(h, s0.refresh), refresh(h, s0.refresh)])
    expect(results.every(result => result.status === 200)).toBe(true)
    expect(new Set(results.map(result => sessionOf(result).refresh)).size).toBe(1)
  })

  test('grace window covers only the immediately previous token; an older one forks', async () => {
    const h = await makeHarness()
    const s0 = await pairDevice(h)
    const s1 = sessionOf(await refresh(h, s0.refresh))
    const s2 = sessionOf(await refresh(h, s1.refresh))
    const older = await refresh(h, s0.refresh)
    expect(older.status).toBe(200)
    expect(sessionOf(older).refresh).not.toBe(s2.refresh)
    expect(h.admin.listDevices().find(device => device.id === s0.deviceId)?.status).toBe('active')
  })

  test('refresh needs an allowed Origin and a cookie', async () => {
    const h = await makeHarness()
    const s0 = await pairDevice(h)
    expect((await refresh(h, s0.refresh, { origin: 'https://evil.test' })).status).toBe(403)
    expect((await refresh(h, s0.refresh, { origin: null })).status).toBe(403)
    expect((await http(h, 'POST', 'session/refresh')).status).toBe(401)
    expect((await refresh(h, 'xr1.not-a-real-token')).status).toBe(401)
  })

  test('a pending device cannot refresh into a session', async () => {
    const h = await makeHarness()
    const s0 = await pairDevice(h, { confirm: true })
    expect((await refresh(h, s0.refresh)).status).toBe(202)
  })
})

describe('expiry', () => {
  test('a browser session lasts five minutes; refresh mints a new one', async () => {
    const h = await makeHarness()
    const s0 = await pairDevice(h)
    h.clock.advance(5 * MINUTE + SECOND)
    expect(await authenticate(h, s0)).toBeNull()
    const renewed = sessionOf(await refresh(h, s0.refresh))
    expect(await authenticate(h, renewed)).not.toBeNull()
  })

  test('7 days of inactivity expire the device', async () => {
    const h = await makeHarness()
    const events: RevocationEvent[] = []
    h.identity.onRevoked(event => events.push(event))
    const s0 = await pairDevice(h)
    h.clock.advance(7 * DAY + SECOND)
    expect((await refresh(h, s0.refresh)).status).toBe(401)
    expect(events.some(event => event.reason === 'expired')).toBe(true)
  })

  test('a device that keeps refreshing still ends at 30 days', async () => {
    const h = await makeHarness()
    let current = (await pairDevice(h)).refresh
    for (let day = 1; day <= 29; day += 1) {
      h.clock.advance(DAY)
      const result = await refresh(h, current)
      expect(result.status).toBe(200)
      current = sessionOf(result).refresh
      const maxAge = Number(/Max-Age=(\d+)/.exec(result.setCookies.find(line => line.startsWith('__Host-xerxes_refresh=')) ?? '')?.[1])
      expect(maxAge).toBeLessThanOrEqual(7 * 86_400)
      expect(maxAge).toBeLessThanOrEqual((30 - day) * 86_400)
    }
    h.clock.advance(DAY)
    expect((await refresh(h, current)).status).toBe(401)
  })

  test('limits can be tightened but not raised', async () => {
    const h = await makeHarness({ limits: { inactivityMs: 1000 } })
    const s0 = await pairDevice(h)
    h.clock.advance(2 * SECOND)
    expect((await refresh(h, s0.refresh)).status).toBe(401)
    const { paths } = tempPaths()
    await expect(createHubIdentity({ paths, signerFingerprint: FINGERPRINT, limits: { absoluteMs: 365 * DAY } })).rejects.toThrow(RangeError)
    await expect(createHubIdentity({ paths, signerFingerprint: 'not-hex' })).rejects.toThrow(RangeError)
  })
})

describe('authenticate', () => {
  test('checks the exact Origin', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    expect(await authenticate(h, s, 'https://evil.test')).toBeNull()
    expect(await authenticate(h, s, 'null')).toBeNull()
    expect(await authenticate(h, s, `${ORIGIN}/`)).toBeNull()
    expect(await authenticate(h, s, 'https://hub.test:444')).toBeNull()
    expect(await authenticate(h, s, 'http://hub.test')).toBeNull()
    expect(await authenticate(h, s, ORIGIN)).not.toBeNull()
    expect(await authenticate(h, s, 'http://localhost:5173')).not.toBeNull()
    expect(await authenticate(h, s, 'http://127.0.0.1:7777')).not.toBeNull()
    expect(await authenticate(h, s, null)).not.toBeNull()
  })

  test('rejects missing, forged and refresh-token-as-session cookies', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    expect(await h.identity.authenticate({ method: 'GET', url: ORIGIN })).toBeNull()
    expect(await h.identity.authenticate({ method: 'GET', url: ORIGIN, cookie: sessionCookie('xs1.forged') })).toBeNull()
    expect(await h.identity.authenticate({ method: 'GET', url: ORIGIN, cookie: sessionCookie(s.refresh) })).toBeNull()
    expect(await h.identity.authenticate({ method: 'GET', url: ORIGIN, cookie: `__Host-xerxes_session=${'x'.repeat(500)}` })).toBeNull()
  })

  test('HTTP authentication needs the CSRF header on mutating routes only', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    const noCsrf = await http(h, 'POST', 'push/unsubscribe', {}, { cookie: s.cookie })
    expect(noCsrf.status).toBe(403)
    expect(errorOf(noCsrf).message).toContain('csrf')
    expect((await http(h, 'POST', 'push/unsubscribe', {}, { cookie: s.cookie, csrf: 'wrong' })).status).toBe(403)
    expect((await http(h, 'POST', 'push/unsubscribe', {}, { cookie: s.cookie, csrf: s.csrf })).status).toBe(200)
    const read = await http(h, 'GET', 'session', undefined, { cookie: s.cookie })
    expect(read.status).toBe(200)
    expect(read.json.csrf_token).toBe(s.csrf)
    expect((read.json.user as { owner: boolean }).owner).toBe(true)
  })

  test('a CSRF token from another session does not work', async () => {
    const h = await makeHarness()
    const a = await pairDevice(h)
    const b = await pairDevice(h)
    expect((await http(h, 'POST', 'push/unsubscribe', {}, { cookie: a.cookie, csrf: b.csrf })).status).toBe(403)
  })

  test('unauthenticated protected routes are 401; foreign Origin is 403', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    expect((await http(h, 'GET', 'session')).status).toBe(401)
    expect((await http(h, 'GET', 'devices')).status).toBe(401)
    expect((await http(h, 'GET', 'session', undefined, { cookie: s.cookie, origin: 'https://evil.test' })).status).toBe(403)
  })
})

describe('logout', () => {
  test('revokes the device, clears cookies and notifies listeners', async () => {
    const h = await makeHarness()
    const events: RevocationEvent[] = []
    h.identity.onRevoked(event => events.push(event))
    const s = await pairDevice(h)
    const out = await http(h, 'POST', 'session/logout', undefined, { cookie: s.cookie })
    expect(out.status).toBe(200)
    expect(out.setCookies.every(line => line.includes('Max-Age=0'))).toBe(true)
    expect(await authenticate(h, s)).toBeNull()
    expect((await refresh(h, s.refresh)).status).toBe(401)
    expect(events.some(event => event.reason === 'logout' && event.deviceId === s.deviceId)).toBe(true)
  })

  test('requires an Origin so another site cannot log a user out', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    expect((await http(h, 'POST', 'session/logout', undefined, { cookie: s.cookie, origin: null })).status).toBe(403)
    expect(await authenticate(h, s)).not.toBeNull()
  })
})

describe('revocation', () => {
  test('revokeDevice fires listeners synchronously and authenticate returns null', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    const ctx = must(await authenticate(h, s))
    const events: RevocationEvent[] = []
    const stop = h.identity.onRevoked(event => events.push(event))
    h.admin.revokeDevice(s.deviceId, 'lost phone')
    expect(events.length).toBeGreaterThanOrEqual(2)
    expect(events.some(event => event.deviceId === s.deviceId && event.browserSessionId === undefined && event.reason === 'lost phone')).toBe(true)
    expect(events.some(event => event.browserSessionId === ctx.browserSessionId)).toBe(true)
    expect(events.every(event => event.principalId === h.ownerId)).toBe(true)
    expect(await authenticate(h, s)).toBeNull()
    expect((await refresh(h, s.refresh)).status).toBe(401)
    const count = events.length
    stop()
    const other = await pairDevice(h)
    h.admin.revokeDevice(other.deviceId)
    expect(events).toHaveLength(count)
  })

  test('revoking an unknown device is a not_found; revoking twice is quiet', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    expect(errorCode(() => h.admin.revokeDevice('dev_missing'))).toBe('not_found')
    h.admin.revokeDevice(s.deviceId)
    const events: RevocationEvent[] = []
    h.identity.onRevoked(event => events.push(event))
    h.admin.revokeDevice(s.deviceId)
    expect(events).toHaveLength(0)
  })

  test('revokeUser kills that user everywhere and spares others; the owner cannot be revoked', async () => {
    const h = await makeHarness()
    const owner = await pairDevice(h)
    const guest = await inviteUser(h, { workspaceId: 'w1', role: 'viewer' })
    const events: RevocationEvent[] = []
    h.identity.onRevoked(event => events.push(event))
    h.admin.revokeUser(guest.userId)
    expect(events.some(event => event.principalId === guest.userId && event.deviceId === guest.deviceId)).toBe(true)
    expect(await authenticate(h, guest)).toBeNull()
    expect((await refresh(h, guest.refresh)).status).toBe(401)
    expect(await authenticate(h, owner)).not.toBeNull()
    expect(errorCode(() => h.admin.revokeUser(h.ownerId))).toBe('conflict')
    expect(h.admin.listUsers().find(user => user.id === guest.userId)?.revoked).toBe(true)
  })

  test('revokeAll closes every device', async () => {
    const h = await makeHarness()
    const a = await pairDevice(h)
    const b = await pairDevice(h)
    const guest = await inviteUser(h, { workspaceId: 'w1', role: 'viewer' })
    const events: RevocationEvent[] = []
    h.identity.onRevoked(event => events.push(event))
    h.admin.revokeAll()
    for (const session of [a, b, guest]) {
      expect(await authenticate(h, session)).toBeNull()
      expect(events.some(event => event.deviceId === session.deviceId)).toBe(true)
    }
    expect(h.admin.listDevices()).toHaveLength(0)
  })

  test('a throwing listener does not stop the others and is reported', async () => {
    const errors: unknown[] = []
    const h = await makeHarness({ onListenerError: error => errors.push(error) })
    const s = await pairDevice(h)
    const seen: string[] = []
    h.identity.onRevoked(() => {
      throw new Error('socket already closed')
    })
    h.identity.onRevoked(event => seen.push(event.reason))
    h.admin.revokeDevice(s.deviceId, 'boom')
    expect(seen).toContain('boom')
    expect(errors.length).toBeGreaterThan(0)
    expect(await authenticate(h, s)).toBeNull()
  })

  test('revocations written by another process (the CLI) are announced on sync', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    const cli = await secondInstance(h)
    const events: RevocationEvent[] = []
    h.identity.onRevoked(event => events.push(event))
    h.admin.syncRevocations()
    expect(events).toHaveLength(0)
    cli.admin.revokeDevice(s.deviceId, 'revoked from cli')
    expect(await authenticate(h, s)).toBeNull()
    h.admin.syncRevocations()
    expect(events.some(event => event.deviceId === s.deviceId && event.reason === 'revoked from cli')).toBe(true)
    const count = events.length
    h.admin.syncRevocations()
    expect(events).toHaveLength(count)
  })

  test('devices can be listed and revoked over HTTP by their owner', async () => {
    const h = await makeHarness()
    const a = await pairDevice(h, { label: 'Phone A' })
    const b = await pairDevice(h, { label: 'Phone B' })
    const list = await http(h, 'GET', 'devices', undefined, { cookie: a.cookie })
    const devices = list.json.devices as { id: string; current: boolean; label: string }[]
    expect(devices.map(device => device.label).sort()).toEqual(['Phone A', 'Phone B'])
    expect(devices.find(device => device.id === a.deviceId)?.current).toBe(true)
    const revoked = await http(h, 'POST', 'devices/revoke', { device_id: b.deviceId }, { cookie: a.cookie, csrf: a.csrf })
    expect(revoked.status).toBe(200)
    expect(await authenticate(h, b)).toBeNull()
    expect(await authenticate(h, a)).not.toBeNull()
  })

  test('a guest cannot revoke or list the owner devices', async () => {
    const h = await makeHarness()
    const owner = await pairDevice(h)
    const guest = await inviteUser(h, { workspaceId: 'w1', role: 'operator' })
    const attempt = await http(h, 'POST', 'devices/revoke', { device_id: owner.deviceId }, { cookie: guest.cookie, csrf: guest.csrf })
    expect(attempt.status).toBe(403)
    expect(await authenticate(h, owner)).not.toBeNull()
    const list = await http(h, 'GET', 'devices?all=1', undefined, { cookie: guest.cookie })
    expect((list.json.devices as { user_id: string }[]).every(device => device.user_id === guest.userId)).toBe(true)
  })
})
