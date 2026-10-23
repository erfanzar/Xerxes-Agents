// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { statSync } from 'node:fs'
import { join } from 'node:path'

import { comparisonCodeFor } from '../src/hub/identity/pairing.js'
import { effectiveScopes } from '../src/security/principal.js'
import {
  FINGERPRINT,
  MINUTE,
  ORIGIN,
  authenticate,
  cleanupAll,
  errorCode,
  errorOf,
  http,
  makeHarness,
  must,
  pairDevice,
  raw,
  ticketOf,
} from './hubIdentityHarness.js'

afterEach(cleanupAll)

describe('owner bootstrap', () => {
  test('creates exactly one owner and refuses a second', async () => {
    const h = await makeHarness({ owner: false })
    const { userId } = h.admin.bootstrapOwner({ name: 'Erfan' })
    expect(h.admin.ownerUserId()).toBe(userId)
    expect(errorCode(() => h.admin.bootstrapOwner({ name: 'Mallory' }))).toBe('conflict')
    expect(h.admin.listUsers()).toHaveLength(1)
  })

  test('requires a name', async () => {
    const h = await makeHarness({ owner: false })
    expect(errorCode(() => h.admin.bootstrapOwner({ name: '   ' }))).toBe('invalid_request')
  })

  test('there is no remote claim route', async () => {
    const h = await makeHarness({ owner: false })
    for (const path of ['owner', 'owner/init', 'claim', 'bootstrap']) {
      const url = `${ORIGIN}/api/hub/v1/${path}`
      expect(await h.identity.handleHttp(new Request(url, { method: 'POST' }), { method: 'POST', url })).toBeUndefined()
    }
  })

  test('non-identity paths are not handled', async () => {
    const h = await makeHarness()
    const url = `${ORIGIN}/api/other`
    expect(await h.identity.handleHttp(new Request(url), { method: 'GET', url })).toBeUndefined()
  })
})

describe('pairing creation', () => {
  test('returns a fragment-only URL, a comparison code and a QR', async () => {
    const h = await makeHarness()
    const created = h.admin.createPairing({ userId: h.ownerId, label: 'Pixel' })
    const url = new URL(created.url)
    expect(url.origin).toBe(ORIGIN)
    expect(url.pathname).toBe('/pair')
    expect(url.search).toBe('')
    const ticket = ticketOf(created.url)
    expect(Object.keys(ticket).sort()).toEqual(['code', 'hub_id', 'pairing_id'])
    expect(Buffer.from(ticket.hub_id ?? '', 'base64url').toString('hex')).toBe(FINGERPRINT)
    expect(ticket.pairing_id).toBe(created.pairingId)
    expect(created.expiresAt - h.clock.now()).toBe(2 * MINUTE)
    expect(created.comparisonCode).toMatch(/^\d{6}$/)
    expect(created.comparisonCode).toBe(comparisonCodeFor(ticket.code ?? '', FINGERPRINT))
    expect(created.confirmRequired).toBe(true)
    expect(created.qrSvg?.startsWith('<svg')).toBe(true)
    expect(created.qrTerminal).not.toBeNull()
  })

  test('rejects unknown users and over-long lifetimes', async () => {
    const h = await makeHarness()
    expect(errorCode(() => h.admin.createPairing({ userId: 'usr_missing' }))).toBe('not_found')
    expect(errorCode(() => h.admin.createPairing({ userId: h.ownerId, ttlMs: 3 * MINUTE }))).toBe('invalid_request')
  })
})

describe('pairing consumption', () => {
  test('--no-confirm style pairing activates the device immediately', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h, { confirm: false })
    expect(s.status).toBe('active')
    const ctx = must(await authenticate(h, s))
    expect(ctx.principal.kind).toBe('user')
    expect(ctx.principal.id).toBe(h.ownerId)
    expect(ctx.principal.kind === 'user' && ctx.principal.hubOwner).toBe(true)
    expect(ctx.principal.kind === 'user' && ctx.principal.deviceId).toBe(s.deviceId)
    expect(ctx.actor.device_id).toBe(s.deviceId)
  })

  test('a confirmed-by-default pairing stays pending until the owner confirms', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h, { confirm: true })
    expect(s.status).toBe('pending_confirmation')
    expect(await authenticate(h, s)).toBeNull()
    const wrong = String((Number(s.comparisonCode) + 1) % 1_000_000).padStart(6, '0')
    expect(errorCode(() => h.admin.confirmPairing(s.pairingId, wrong))).toBe('forbidden')
    expect(await authenticate(h, s)).toBeNull()
    h.admin.confirmPairing(s.pairingId, s.comparisonCode)
    expect(await authenticate(h, s)).not.toBeNull()
    expect(errorCode(() => h.admin.confirmPairing(s.pairingId, s.comparisonCode))).toBe('conflict')
  })

  test('five wrong comparison codes revoke the pending device', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h, { confirm: true })
    const wrong = String((Number(s.comparisonCode) + 1) % 1_000_000).padStart(6, '0')
    for (let i = 0; i < 5; i += 1) expect(errorCode(() => h.admin.confirmPairing(s.pairingId, wrong))).toBe('forbidden')
    expect(h.admin.listDevices({ includeRevoked: true }).find(device => device.id === s.deviceId)?.status).toBe('revoked')
  })

  test('an unconfirmed device is revoked after the confirmation window', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h, { confirm: true })
    h.clock.advance(6 * MINUTE)
    expect(errorCode(() => h.admin.confirmPairing(s.pairingId, s.comparisonCode))).toBe('conflict')
    expect(h.admin.listDevices({ includeRevoked: true }).find(device => device.id === s.deviceId)?.status).toBe('revoked')
  })

  test('an expired code is refused', async () => {
    const h = await makeHarness()
    const created = h.admin.createPairing({ userId: h.ownerId, confirm: false })
    const ticket = ticketOf(created.url)
    h.clock.advance(2 * MINUTE + 1000)
    const result = await http(h, 'POST', 'pair/consume', { pairing_id: ticket.pairing_id, code: ticket.code })
    expect(result.status).toBe(401)
    expect(h.admin.listDevices({ includeRevoked: true })).toHaveLength(0)
  })

  test('a code is single use', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    const again = await http(h, 'POST', 'pair/consume', { pairing_id: s.pairingId, code: s.code })
    expect(again.status).toBe(401)
    expect(h.admin.listDevices()).toHaveLength(1)
  })

  test('competing consumers: exactly one wins', async () => {
    const h = await makeHarness()
    const created = h.admin.createPairing({ userId: h.ownerId, confirm: false })
    const ticket = ticketOf(created.url)
    const results = await Promise.all(
      Array.from({ length: 6 }, (_, index) => http(h, 'POST', 'pair/consume', { pairing_id: ticket.pairing_id, code: ticket.code }, { ip: `10.0.0.${index + 1}` })),
    )
    expect(results.filter(result => result.status === 200)).toHaveLength(1)
    expect(results.filter(result => result.status === 401)).toHaveLength(5)
    expect(h.admin.listDevices({ includeRevoked: true })).toHaveLength(1)
  })

  test('five wrong codes lock the pairing even for the right code', async () => {
    const h = await makeHarness()
    const created = h.admin.createPairing({ userId: h.ownerId, confirm: false })
    const ticket = ticketOf(created.url)
    for (let i = 0; i < 5; i += 1) {
      expect((await http(h, 'POST', 'pair/consume', { pairing_id: ticket.pairing_id, code: `wrong-code-${i}` })).status).toBe(401)
    }
    expect((await http(h, 'POST', 'pair/consume', { pairing_id: ticket.pairing_id, code: ticket.code })).status).toBe(401)
    expect(h.admin.listAudit().some(event => event.action.startsWith('identity.ticket.locked:'))).toBe(true)
  })

  test('attempts from one address are rate limited', async () => {
    const h = await makeHarness()
    for (let i = 0; i < 10; i += 1) {
      expect((await http(h, 'POST', 'pair/consume', { pairing_id: `pr_unknown${i}`, code: 'nope' })).status).toBe(401)
    }
    const limited = await http(h, 'POST', 'pair/consume', { pairing_id: 'pr_unknown', code: 'nope' })
    expect(limited.status).toBe(429)
    expect(errorOf(limited).code).toBe('rate_limited')
    h.clock.advance(61_000)
    expect((await http(h, 'POST', 'pair/consume', { pairing_id: 'pr_unknown', code: 'nope' })).status).toBe(401)
  })

  test('a foreign Origin is refused; a missing Origin (non-browser) is not', async () => {
    const h = await makeHarness()
    const created = h.admin.createPairing({ userId: h.ownerId, confirm: false })
    const ticket = ticketOf(created.url)
    const body = { pairing_id: ticket.pairing_id, code: ticket.code }
    expect((await http(h, 'POST', 'pair/consume', body, { origin: 'https://evil.test' })).status).toBe(403)
    expect((await http(h, 'POST', 'pair/consume', body, { origin: null })).status).toBe(200)
  })

  test('malformed bodies are 400, wrong methods are 405', async () => {
    const h = await makeHarness()
    expect((await http(h, 'POST', 'pair/consume', { pairing_id: 5, code: 'x' })).status).toBe(400)
    expect((await http(h, 'GET', 'pair/consume')).status).toBe(405)
    const url = `${ORIGIN}/api/hub/v1/pair/consume`
    const plain = await h.identity.handleHttp(new Request(url, { method: 'POST', headers: { 'content-type': 'text/plain' }, body: '{}' }), { method: 'POST', url })
    expect(plain?.status).toBe(400)
  })

  test('the pairing role caps the device at that role', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h, { role: 'viewer' })
    const ctx = must(await authenticate(h, s))
    const scopes = effectiveScopes(ctx.principal, ctx.grants, { workspaceId: 'w1' })
    expect(scopes.has('session.read')).toBe(true)
    expect(scopes.has('session.submit')).toBe(false)
    expect(scopes.has('devices.manage')).toBe(false)
  })
})

describe('cookies and storage', () => {
  test('cookie flags: __Host- prefix, Path=/, Secure, HttpOnly, SameSite=Strict, no Domain', async () => {
    const h = await makeHarness()
    const created = h.admin.createPairing({ userId: h.ownerId, confirm: false })
    const ticket = ticketOf(created.url)
    const result = await http(h, 'POST', 'pair/consume', { pairing_id: ticket.pairing_id, code: ticket.code })
    expect(result.setCookies).toHaveLength(2)
    const refresh = must(result.setCookies.find(line => line.startsWith('__Host-xerxes_refresh=')))
    const session = must(result.setCookies.find(line => line.startsWith('__Host-xerxes_session=')))
    for (const cookie of [refresh, session]) {
      expect(cookie).toContain('Path=/')
      expect(cookie).toContain('Secure')
      expect(cookie).toContain('HttpOnly')
      expect(cookie).toContain('SameSite=Strict')
      expect(cookie).not.toContain('Domain')
    }
    expect(session).toContain('Max-Age=300')
    expect(result.res.headers.get('cache-control')).toBe('no-store')
    expect(typeof result.json.csrf_token).toBe('string')
    expect(result.json.comparison_code).toMatch(/^\d{6}$/)
  })

  test('only hashes reach the database; files are private', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    const db = raw(h)
    const tables = db.query("SELECT name FROM sqlite_master WHERE type = 'table'").all() as { name: string }[]
    const dump = tables.map(table => JSON.stringify(db.query(`SELECT * FROM ${table.name}`).all())).join('\n')
    for (const secret of [s.code, s.refresh, s.session, s.csrf]) {
      expect(secret.length).toBeGreaterThan(20)
      expect(dump).not.toContain(secret)
    }
    expect((db.query('PRAGMA journal_mode').get() as { journal_mode: string }).journal_mode).toBe('wal')
    for (const name of ['identity.sqlite', 'identity-pepper', 'vapid-key.pem']) {
      expect(statSync(join(h.paths.stateDir, name)).mode & 0o777).toBe(0o600)
    }
    expect(JSON.stringify(h.admin.listAudit())).not.toContain(s.code)
  })
})
