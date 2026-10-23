// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { createDecipheriv, createECDH, createPublicKey, hkdfSync, randomBytes, verify, type ECDH } from 'node:crypto'

import type { PushFetch } from '../src/hub/identity/core.js'
import { MAX_PUSH_PLAINTEXT_BYTES, createVapidJwt, encryptWebPush, generateVapidPem, vapidPublicKeyOf } from '../src/hub/identity/webPush.js'
import { b64u, cleanupAll, errorOf, http, makeHarness, pairDevice, raw, type BrowserSession, type Harness } from './hubIdentityHarness.js'

afterEach(cleanupAll)

interface Captured {
  readonly url: string
  readonly headers: Record<string, string>
  readonly body: Uint8Array
}

function pushPort(): { fetch: PushFetch; calls: Captured[]; setStatus(status: number): void; failNext(): void } {
  const calls: Captured[] = []
  let status = 201
  let fail = false
  return {
    calls,
    setStatus: next => {
      status = next
    },
    failNext: () => {
      fail = true
    },
    fetch: async (url, init) => {
      if (fail) {
        fail = false
        throw new Error('connection reset')
      }
      calls.push({ url, headers: init.headers, body: init.body })
      return { status }
    },
  }
}

interface Receiver {
  readonly ecdh: ECDH
  readonly auth: Buffer
}

function receiver(): Receiver {
  const ecdh = createECDH('prime256v1')
  ecdh.generateKeys()
  return { ecdh, auth: randomBytes(16) }
}

/** The browser side of RFC 8291: decrypt a single aes128gcm record. */
function decrypt(body: Uint8Array, who: Receiver): { readonly recordSize: number; readonly keyId: Buffer; readonly plaintext: Buffer; readonly delimiter: number } {
  const buffer = Buffer.from(body)
  const salt = buffer.subarray(0, 16)
  const recordSize = buffer.readUInt32BE(16)
  const idLength = buffer.readUInt8(20)
  const senderPublic = buffer.subarray(21, 21 + idLength)
  const encrypted = buffer.subarray(21 + idLength)
  const shared = who.ecdh.computeSecret(senderPublic)
  const info = Buffer.concat([Buffer.from('WebPush: info\0'), who.ecdh.getPublicKey(), senderPublic])
  const ikm = Buffer.from(hkdfSync('sha256', shared, who.auth, info, 32))
  const key = Buffer.from(hkdfSync('sha256', ikm, salt, Buffer.from('Content-Encoding: aes128gcm\0'), 16))
  const nonce = Buffer.from(hkdfSync('sha256', ikm, salt, Buffer.from('Content-Encoding: nonce\0'), 12))
  const decipher = createDecipheriv('aes-128-gcm', key, nonce)
  decipher.setAuthTag(encrypted.subarray(encrypted.length - 16))
  const padded = Buffer.concat([decipher.update(encrypted.subarray(0, encrypted.length - 16)), decipher.final()])
  return { recordSize, keyId: senderPublic, plaintext: padded.subarray(0, padded.length - 1), delimiter: padded[padded.length - 1] ?? -1 }
}

function verifyJwt(jwt: string, publicKey: string): { readonly valid: boolean; readonly header: Record<string, unknown>; readonly claims: Record<string, unknown> } {
  const [head = '', payload = '', signature = ''] = jwt.split('.')
  const point = Buffer.from(publicKey, 'base64url')
  const key = createPublicKey({ key: { kty: 'EC', crv: 'P-256', x: point.subarray(1, 33).toString('base64url'), y: point.subarray(33).toString('base64url') }, format: 'jwk' })
  const valid = verify('sha256', Buffer.from(`${head}.${payload}`), { key, dsaEncoding: 'ieee-p1363' }, Buffer.from(signature, 'base64url'))
  return {
    valid,
    header: JSON.parse(Buffer.from(head, 'base64url').toString('utf8')) as Record<string, unknown>,
    claims: JSON.parse(Buffer.from(payload, 'base64url').toString('utf8')) as Record<string, unknown>,
  }
}

const ENDPOINT = 'https://fcm.googleapis.com/fcm/send/abc123'

async function subscribed(options: { vapidSubject?: string | undefined } = { vapidSubject: 'mailto:ops@hub.test' }) {
  const port = pushPort()
  const h = await makeHarness({ fetch: port.fetch, ...(options.vapidSubject === undefined ? {} : { vapidSubject: options.vapidSubject }) })
  const s = await pairDevice(h)
  const who = receiver()
  const res = await subscribe(h, s, who)
  expect(res.status).toBe(201)
  return { h, s, who, port }
}

function subscribe(h: Harness, s: BrowserSession, who: Receiver, endpoint: string = ENDPOINT) {
  return http(h, 'POST', 'push/subscribe', { endpoint, keys: { p256dh: b64u(who.ecdh.getPublicKey()), auth: b64u(who.auth) } }, { cookie: s.cookie, csrf: s.csrf })
}

const message = { title: 'Approval needed', body: 'A task is waiting for you', url: '/task/abc' }

describe('RFC 8291 encryption', () => {
  test('round-trips through an independent decryption', () => {
    const who = receiver()
    const payload = Buffer.from(JSON.stringify({ title: 'hi', body: 'there', url: '/x' }))
    const body = encryptWebPush({ payload, p256dh: who.ecdh.getPublicKey(), auth: who.auth })
    const opened = decrypt(body, who)
    expect(opened.plaintext.equals(payload)).toBe(true)
    expect(opened.delimiter).toBe(2)
    expect(opened.recordSize).toBe(4096)
    expect(opened.keyId).toHaveLength(65)
    expect(body.length).toBeLessThanOrEqual(4096)
  })

  test('a pinned salt is placed in the header; a different receiver cannot decrypt', () => {
    const who = receiver()
    const salt = Buffer.alloc(16, 7)
    const body = encryptWebPush({ payload: Buffer.from('x'), p256dh: who.ecdh.getPublicKey(), auth: who.auth, salt })
    expect(Buffer.from(body).subarray(0, 16).equals(salt)).toBe(true)
    expect(() => decrypt(body, receiver())).toThrow()
  })

  test('input is validated', () => {
    const who = receiver()
    const base = { p256dh: who.ecdh.getPublicKey(), auth: who.auth }
    expect(() => encryptWebPush({ ...base, payload: new Uint8Array(MAX_PUSH_PLAINTEXT_BYTES + 1) })).toThrow(RangeError)
    expect(encryptWebPush({ ...base, payload: new Uint8Array(MAX_PUSH_PLAINTEXT_BYTES) }).length).toBeLessThanOrEqual(4096)
    expect(() => encryptWebPush({ ...base, payload: Buffer.from('x'), auth: Buffer.alloc(8) })).toThrow(RangeError)
    expect(() => encryptWebPush({ ...base, payload: Buffer.from('x'), p256dh: Buffer.alloc(65) })).toThrow(RangeError)
  })
})

describe('RFC 8292 VAPID', () => {
  test('the JWT is ES256 and verifies against the advertised public key', () => {
    const pem = generateVapidPem()
    const publicKey = vapidPublicKeyOf(pem)
    expect(Buffer.from(publicKey, 'base64url')).toHaveLength(65)
    const jwt = createVapidJwt({ privateKeyPem: pem, audience: 'https://fcm.googleapis.com', subject: 'mailto:ops@hub.test', expiresAtSeconds: 1_800_000_000 })
    const parsed = verifyJwt(jwt, publicKey)
    expect(parsed.valid).toBe(true)
    expect(parsed.header).toEqual({ typ: 'JWT', alg: 'ES256' })
    expect(parsed.claims).toEqual({ aud: 'https://fcm.googleapis.com', exp: 1_800_000_000, sub: 'mailto:ops@hub.test' })
    expect(Buffer.from(jwt.split('.')[2] ?? '', 'base64url')).toHaveLength(64)
    const other = verifyJwt(jwt, vapidPublicKeyOf(generateVapidPem()))
    expect(other.valid).toBe(false)
  })
})

describe('subscriptions', () => {
  test('subscribe needs a session and CSRF; the public key is served to signed-in users only', async () => {
    const port = pushPort()
    const h = await makeHarness({ fetch: port.fetch })
    const s = await pairDevice(h)
    const who = receiver()
    const body = { endpoint: ENDPOINT, keys: { p256dh: b64u(who.ecdh.getPublicKey()), auth: b64u(who.auth) } }
    expect((await http(h, 'POST', 'push/subscribe', body)).status).toBe(401)
    expect((await http(h, 'POST', 'push/subscribe', body, { cookie: s.cookie })).status).toBe(403)
    expect((await http(h, 'GET', 'push/vapid-key')).status).toBe(401)
    const key = await http(h, 'GET', 'push/vapid-key', undefined, { cookie: s.cookie })
    expect(key.json.public_key).toBe(h.admin.vapidPublicKey)
  })

  test('endpoints must be https push services; bad keys are refused', async () => {
    const port = pushPort()
    const h = await makeHarness({ fetch: port.fetch })
    const s = await pairDevice(h)
    const who = receiver()
    const keys = { p256dh: b64u(who.ecdh.getPublicKey()), auth: b64u(who.auth) }
    for (const endpoint of ['http://fcm.googleapis.com/x', 'https://127.0.0.1/x', 'https://evil.test/x', 'https://fcm.googleapis.com.evil.test/x', 'https://user:pw@fcm.googleapis.com/x', 'https://fcm.googleapis.com:8443/x', 'not a url']) {
      expect([endpoint, (await http(h, 'POST', 'push/subscribe', { endpoint, keys }, { cookie: s.cookie, csrf: s.csrf })).status >= 400]).toEqual([endpoint, true])
    }
    expect((await http(h, 'POST', 'push/subscribe', { endpoint: ENDPOINT, keys: { ...keys, auth: b64u(randomBytes(8)) } }, { cookie: s.cookie, csrf: s.csrf })).status).toBe(400)
    expect((await http(h, 'POST', 'push/subscribe', { endpoint: ENDPOINT, keys: { ...keys, p256dh: b64u(randomBytes(65)) } }, { cookie: s.cookie, csrf: s.csrf })).status).toBe(400)
    expect(raw(h).query('SELECT COUNT(*) AS n FROM hub_push_subscriptions').get()).toEqual({ n: 0 })
  })

  test('unsubscribe removes the subscription', async () => {
    const { h, s } = await subscribed()
    const removed = await http(h, 'POST', 'push/unsubscribe', { endpoint: ENDPOINT }, { cookie: s.cookie, csrf: s.csrf })
    expect(removed.json.removed).toBe(1)
    const result = await h.admin.sendPush(h.ownerId, message)
    expect(result).toMatchObject({ delivered: false, failure: 'no_subscriptions' })
  })

  test('revoking the device purges its subscriptions', async () => {
    const { h, s, port } = await subscribed()
    expect(raw(h).query('SELECT COUNT(*) AS n FROM hub_push_subscriptions').get()).toEqual({ n: 1 })
    h.admin.revokeDevice(s.deviceId)
    expect(raw(h).query('SELECT COUNT(*) AS n FROM hub_push_subscriptions').get()).toEqual({ n: 0 })
    expect(await h.admin.sendPush(h.ownerId, message)).toMatchObject({ delivered: false, failure: 'no_subscriptions' })
    expect(port.calls).toHaveLength(0)
  })

  test('revoking the user purges every device subscription', async () => {
    const port = pushPort()
    const h = await makeHarness({ fetch: port.fetch })
    const guestCreated = h.admin.createInvite({ createdBy: h.ownerId, workspaceId: 'w1', role: 'viewer', confirm: false })
    const ticket = JSON.parse(Buffer.from(new URL(guestCreated.url).hash.slice(1), 'base64url').toString('utf8')) as Record<string, string>
    const accepted = await http(h, 'POST', 'invites/accept', { invite_id: ticket.invite_id, code: ticket.code })
    const guest: BrowserSession = {
      status: 'active',
      refresh: accepted.cookies['__Host-xerxes_refresh'] ?? '',
      session: accepted.cookies['__Host-xerxes_session'] ?? '',
      csrf: String(accepted.json.csrf_token),
      deviceId: String(accepted.json.device_id),
      userId: (accepted.json.user as { id: string }).id,
      cookie: `__Host-xerxes_refresh=${accepted.cookies['__Host-xerxes_refresh'] ?? ''}; __Host-xerxes_session=${accepted.cookies['__Host-xerxes_session'] ?? ''}`,
    }
    expect((await subscribe(h, guest, receiver())).status).toBe(201)
    h.admin.revokeUser(guest.userId)
    expect(raw(h).query('SELECT COUNT(*) AS n FROM hub_push_subscriptions').get()).toEqual({ n: 0 })
  })
})

describe('sendPush', () => {
  test('delivers an encrypted, VAPID-signed message and reports delivery only on 201', async () => {
    const { h, who, port } = await subscribed()
    const result = await h.admin.sendPush(h.ownerId, message)
    expect(result).toMatchObject({ delivered: true, attempted: 1, accepted: 1 })
    expect(result.deliveries[0]).toMatchObject({ ok: true, reason: 'accepted', status: 201 })
    const call = port.calls[0]
    if (!call) throw new Error('no push request was made')
    expect(call.url).toBe(ENDPOINT)
    expect(call.headers['content-encoding']).toBe('aes128gcm')
    expect(call.headers['content-type']).toBe('application/octet-stream')
    expect(call.headers.ttl).toBe('3600')
    const opened = decrypt(call.body, who)
    expect(JSON.parse(opened.plaintext.toString('utf8'))).toEqual(message)
    const match = /^vapid t=([^,]+), k=(.+)$/.exec(call.headers.authorization ?? '')
    expect(match).not.toBeNull()
    expect(match?.[2]).toBe(h.admin.vapidPublicKey)
    const jwt = verifyJwt(match?.[1] ?? '', h.admin.vapidPublicKey)
    expect(jwt.valid).toBe(true)
    expect(jwt.claims.aud).toBe('https://fcm.googleapis.com')
    expect(jwt.claims.sub).toBe('mailto:ops@hub.test')
    const exp = Number(jwt.claims.exp)
    const now = Math.floor(h.clock.now() / 1000)
    expect(exp).toBeGreaterThan(now)
    expect(exp - now).toBeLessThanOrEqual(24 * 60 * 60)
  })

  test('the lock-screen payload carries only title, body and url', async () => {
    const { h, who, port } = await subscribed()
    await h.admin.sendPush(h.ownerId, message)
    const payload = JSON.parse(decrypt(port.calls[0]?.body ?? new Uint8Array(), who).plaintext.toString('utf8')) as Record<string, unknown>
    expect(Object.keys(payload).sort()).toEqual(['body', 'title', 'url'])
  })

  test('anything but 201 or 202 is not delivery', async () => {
    const { h, port } = await subscribed()
    port.setStatus(200)
    expect(await h.admin.sendPush(h.ownerId, message)).toMatchObject({ delivered: false, failure: 'all_failed', accepted: 0 })
    port.setStatus(500)
    const failed = await h.admin.sendPush(h.ownerId, message)
    expect(failed).toMatchObject({ delivered: false, failure: 'all_failed' })
    expect(failed.deliveries[0]).toMatchObject({ ok: false, reason: 'rejected', status: 500 })
    port.setStatus(202)
    expect(await h.admin.sendPush(h.ownerId, message)).toMatchObject({ delivered: true, accepted: 1 })
  })

  test('a network error is typed, not thrown, and stays retryable', async () => {
    const { h, port } = await subscribed()
    port.failNext()
    const failed = await h.admin.sendPush(h.ownerId, message)
    expect(failed.deliveries[0]).toMatchObject({ ok: false, reason: 'network_error' })
    expect(failed.delivered).toBe(false)
    expect((await h.admin.sendPush(h.ownerId, message)).delivered).toBe(true)
  })

  test('404 and 410 delete the dead subscription', async () => {
    for (const status of [404, 410]) {
      const { h, port } = await subscribed()
      port.setStatus(status)
      const result = await h.admin.sendPush(h.ownerId, message)
      expect(result.deliveries[0]).toMatchObject({ ok: false, reason: 'gone', status })
      expect(raw(h).query('SELECT COUNT(*) AS n FROM hub_push_subscriptions').get()).toEqual({ n: 0 })
    }
  })

  test('identical messages to one user are deduplicated inside the window', async () => {
    const { h, port } = await subscribed()
    expect((await h.admin.sendPush(h.ownerId, message)).delivered).toBe(true)
    expect(await h.admin.sendPush(h.ownerId, message)).toMatchObject({ delivered: false, failure: 'deduped', attempted: 0 })
    expect((await h.admin.sendPush(h.ownerId, { ...message, dedupeKey: 'another-event' })).delivered).toBe(true)
    expect(port.calls).toHaveLength(2)
    h.clock.advance(61_000)
    expect((await h.admin.sendPush(h.ownerId, message)).delivered).toBe(true)
  })

  test('typed failures: no subscriptions, unknown user, missing VAPID subject', async () => {
    const port = pushPort()
    const h = await makeHarness({ fetch: port.fetch, vapidSubject: 'mailto:ops@hub.test' })
    expect(await h.admin.sendPush(h.ownerId, message)).toMatchObject({ delivered: false, failure: 'no_subscriptions' })
    expect(await h.admin.sendPush('usr_missing', message)).toMatchObject({ delivered: false, failure: 'user_inactive' })
    const { h: noSubject } = await subscribed({ vapidSubject: undefined })
    expect(await noSubject.admin.sendPush(noSubject.ownerId, message)).toMatchObject({ delivered: false, failure: 'vapid_subject_missing' })
  })

  test('messages that are oversized, off-origin or look like secrets are refused', async () => {
    const { h, port } = await subscribed()
    await expect(h.admin.sendPush(h.ownerId, { ...message, url: 'https://evil.test/x' })).rejects.toThrow('same-origin')
    await expect(h.admin.sendPush(h.ownerId, { ...message, url: '//evil.test/x' })).rejects.toThrow('same-origin')
    await expect(h.admin.sendPush(h.ownerId, { ...message, body: 'x'.repeat(161) })).rejects.toThrow('body')
    await expect(h.admin.sendPush(h.ownerId, { ...message, title: '' })).rejects.toThrow('title')
    await expect(h.admin.sendPush(h.ownerId, { ...message, body: 'token=abcdef123456' })).rejects.toThrow('secret')
    expect(port.calls).toHaveLength(0)
  })

  test('a pending (unconfirmed) device receives nothing', async () => {
    const port = pushPort()
    const h = await makeHarness({ fetch: port.fetch, vapidSubject: 'mailto:ops@hub.test' })
    const pending = await pairDevice(h, { confirm: true })
    const result = await subscribe(h, pending, receiver())
    expect(result.status).toBeGreaterThanOrEqual(400)
    expect(errorOf(result).code).toBeTruthy()
  })
})
