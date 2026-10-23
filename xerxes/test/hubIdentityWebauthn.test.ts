// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { createHash, generateKeyPairSync, randomBytes, sign, type KeyObject } from 'node:crypto'

import { decodeCbor, parseAuthData } from '../src/hub/identity/webauthn.js'
import {
  MINUTE,
  ORIGIN,
  RP_ID,
  SECOND,
  authenticate,
  b64u,
  cleanupAll,
  errorCode,
  errorOf,
  http,
  makeHarness,
  must,
  pairDevice,
  type BrowserSession,
  type Harness,
  type HttpResult,
} from './hubIdentityHarness.js'

afterEach(cleanupAll)

// ---------------------------------------------------------------- tiny CBOR encoder + fake authenticator

type Cbor = number | string | Uint8Array | Cbor[] | Map<Cbor, Cbor>

function head(major: number, n: number): Buffer {
  if (n < 24) return Buffer.from([(major << 5) | n])
  if (n < 256) return Buffer.from([(major << 5) | 24, n])
  const bytes = Buffer.alloc(3)
  bytes[0] = (major << 5) | 25
  bytes.writeUInt16BE(n, 1)
  return bytes
}

function cbor(value: Cbor): Buffer {
  if (typeof value === 'number') return value >= 0 ? head(0, value) : head(1, -1 - value)
  if (typeof value === 'string') {
    const text = Buffer.from(value, 'utf8')
    return Buffer.concat([head(3, text.length), text])
  }
  if (value instanceof Uint8Array) return Buffer.concat([head(2, value.length), value])
  if (Array.isArray(value)) return Buffer.concat([head(4, value.length), ...value.map(cbor)])
  return Buffer.concat([head(5, value.size), ...[...value].flatMap(([key, item]) => [cbor(key), cbor(item)])])
}

const UP = 0x01
const UV = 0x04
const AT = 0x40

interface Authenticator {
  readonly alg: -7 | -8
  readonly privateKey: KeyObject
  readonly credentialId: Buffer
  signCount: number
  cose: Map<Cbor, Cbor>
}

function authenticator(alg: -7 | -8 = -7): Authenticator {
  const credentialId = randomBytes(32)
  if (alg === -7) {
    const { privateKey, publicKey } = generateKeyPairSync('ec', { namedCurve: 'prime256v1' })
    const jwk = publicKey.export({ format: 'jwk' })
    const cose = new Map<Cbor, Cbor>([
      [1, 2],
      [3, -7],
      [-1, 1],
      [-2, Buffer.from(jwk.x ?? '', 'base64url')],
      [-3, Buffer.from(jwk.y ?? '', 'base64url')],
    ])
    return { alg, privateKey, credentialId, signCount: 0, cose }
  }
  const { privateKey, publicKey } = generateKeyPairSync('ed25519')
  const jwk = publicKey.export({ format: 'jwk' })
  const cose = new Map<Cbor, Cbor>([
    [1, 1],
    [3, -8],
    [-1, 6],
    [-2, Buffer.from(jwk.x ?? '', 'base64url')],
  ])
  return { alg, privateKey, credentialId, signCount: 0, cose }
}

interface AuthDataOptions {
  readonly rpId?: string | undefined
  readonly flags: number
  readonly signCount: number
  readonly attested?: Authenticator | undefined
}

function authData(options: AuthDataOptions): Buffer {
  const rpIdHash = createHash('sha256').update(options.rpId ?? RP_ID).digest()
  const counter = Buffer.alloc(4)
  counter.writeUInt32BE(options.signCount)
  const parts: Buffer[] = [rpIdHash, Buffer.from([options.flags]), counter]
  if (options.attested) {
    const idLength = Buffer.alloc(2)
    idLength.writeUInt16BE(options.attested.credentialId.length)
    parts.push(Buffer.alloc(16), idLength, options.attested.credentialId, cbor(options.attested.cose))
  }
  return Buffer.concat(parts)
}

interface Ceremony {
  readonly challenge?: string | undefined
  readonly origin?: string | undefined
  readonly rpId?: string | undefined
  readonly flags?: number | undefined
  readonly type?: string | undefined
  readonly fmt?: string | undefined
  readonly signCount?: number | undefined
  readonly corruptSignature?: boolean | undefined
}

async function challengeFor(h: Harness, s: BrowserSession, path: string): Promise<string> {
  const result = await http(h, 'POST', path, {}, { cookie: s.cookie, csrf: s.csrf })
  expect(result.status).toBe(200)
  return String(result.json.challenge)
}

function register(h: Harness, s: BrowserSession, a: Authenticator, o: Ceremony = {}): Promise<HttpResult> {
  return (async () => {
    const challenge = o.challenge ?? (await challengeFor(h, s, 'passkeys/register/options'))
    const clientData = Buffer.from(JSON.stringify({ type: o.type ?? 'webauthn.create', challenge, origin: o.origin ?? ORIGIN }))
    const data = authData({ rpId: o.rpId, flags: o.flags ?? UP | UV | AT, signCount: 0, attested: a })
    const attestation = cbor(new Map<Cbor, Cbor>([['fmt', o.fmt ?? 'none'], ['attStmt', new Map<Cbor, Cbor>()], ['authData', data]]))
    return http(h, 'POST', 'passkeys/register/verify', { clientDataJSON: b64u(clientData), attestationObject: b64u(attestation), label: 'Test key' }, { cookie: s.cookie, csrf: s.csrf })
  })()
}

function assertWith(h: Harness, s: BrowserSession, a: Authenticator, o: Ceremony = {}): Promise<HttpResult> {
  return (async () => {
    const challenge = o.challenge ?? (await challengeFor(h, s, 'passkeys/assert/options'))
    a.signCount = o.signCount ?? a.signCount + 1
    const clientData = Buffer.from(JSON.stringify({ type: o.type ?? 'webauthn.get', challenge, origin: o.origin ?? ORIGIN }))
    const data = authData({ rpId: o.rpId, flags: o.flags ?? UP | UV, signCount: a.signCount })
    const signed = Buffer.concat([data, createHash('sha256').update(clientData).digest()])
    let signature = sign(a.alg === -8 ? null : 'sha256', signed, a.privateKey)
    if (o.corruptSignature) {
      signature = Buffer.from(signature)
      signature[signature.length - 1] = (signature[signature.length - 1] ?? 0) ^ 1
    }
    return http(
      h,
      'POST',
      'passkeys/assert/verify',
      { id: b64u(a.credentialId), clientDataJSON: b64u(clientData), authenticatorData: b64u(data), signature: b64u(signature) },
      { cookie: s.cookie, csrf: s.csrf },
    )
  })()
}

function createInvite(h: Harness, s: BrowserSession): Promise<HttpResult> {
  return http(h, 'POST', 'invites', { workspace_id: 'w1', role: 'viewer' }, { cookie: s.cookie, csrf: s.csrf })
}

async function enrolled(alg: -7 | -8 = -7): Promise<{ h: Harness; s: BrowserSession; a: Authenticator }> {
  const h = await makeHarness()
  const s = await pairDevice(h)
  const a = authenticator(alg)
  const result = await register(h, s, a)
  expect(result.status).toBe(201)
  return { h, s, a }
}

// ---------------------------------------------------------------- tests

describe('passkey registration', () => {
  test('ES256 and EdDSA credentials register and are listed', async () => {
    for (const alg of [-7, -8] as const) {
      const { h, s, a } = await enrolled(alg)
      const list = await http(h, 'GET', 'passkeys', undefined, { cookie: s.cookie })
      const keys = list.json.passkeys as { id: string; alg: number }[]
      expect(keys).toHaveLength(1)
      expect(keys[0]?.id).toBe(b64u(a.credentialId))
      expect(keys[0]?.alg).toBe(alg)
    }
  })

  test('options bind the RP id to the public origin host and require user verification', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    const result = await http(h, 'POST', 'passkeys/register/options', {}, { cookie: s.cookie, csrf: s.csrf })
    expect(result.status).toBe(200)
    expect((result.json.rp as { id: string }).id).toBe(RP_ID)
    expect((result.json.authenticatorSelection as { userVerification: string }).userVerification).toBe('required')
    expect(result.json.attestation).toBe('none')
    expect((result.json.pubKeyCredParams as { alg: number }[]).map(entry => entry.alg)).toEqual([-7, -8])
  })

  test('rejects a wrong RP id, a wrong origin, a wrong type and missing UV or UP', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    const wrongRp = await register(h, s, authenticator(), { rpId: 'evil.test' })
    expect(wrongRp.status).toBe(403)
    expect(errorOf(wrongRp).message).toContain('relying party')
    const wrongOrigin = await register(h, s, authenticator(), { origin: 'https://evil.test' })
    expect(wrongOrigin.status).toBe(403)
    expect(errorOf(wrongOrigin).message).toContain('origin')
    expect((await register(h, s, authenticator(), { origin: 'http://hub.test' })).status).toBe(403)
    expect((await register(h, s, authenticator(), { type: 'webauthn.get' })).status).toBe(400)
    const noUv = await register(h, s, authenticator(), { flags: UP | AT })
    expect(noUv.status).toBe(403)
    expect(errorOf(noUv).message).toContain('verification')
    expect((await register(h, s, authenticator(), { flags: UV | AT })).status).toBe(403)
    expect((await register(h, s, authenticator(), { fmt: 'packed' })).status).toBe(400)
    const rsa = authenticator()
    rsa.cose = new Map<Cbor, Cbor>([[1, 3], [3, -257]])
    expect((await register(h, s, rsa)).status).toBe(400)
    expect((await http(h, 'GET', 'passkeys', undefined, { cookie: s.cookie })).json.passkeys).toEqual([])
  })

  test('a challenge is single use, even when the first attempt failed', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    const a = authenticator()
    const challenge = await challengeFor(h, s, 'passkeys/register/options')
    expect((await register(h, s, a, { challenge, flags: UP | AT })).status).toBe(403)
    const retry = await register(h, s, a, { challenge })
    expect(retry.status).toBe(401)
    expect(errorOf(retry).message).toContain('challenge')
    const fresh = await challengeFor(h, s, 'passkeys/register/options')
    expect((await register(h, s, a, { challenge: fresh })).status).toBe(201)
    // With a passkey enrolled the route now demands step-up before it even looks at the challenge.
    const after = await register(h, s, a, { challenge: fresh })
    expect(after.status).toBe(403)
    expect(errorOf(after).code).toBe('step_up_required')
  })

  test('a challenge cannot be used from another browser session', async () => {
    const h = await makeHarness()
    const first = await pairDevice(h)
    const second = await pairDevice(h)
    const challenge = await challengeFor(h, first, 'passkeys/register/options')
    expect((await register(h, second, authenticator(), { challenge })).status).toBe(401)
    expect((await register(h, first, authenticator(), { challenge })).status).toBe(401)
  })

  test('a challenge expires', async () => {
    const h = await makeHarness({ limits: { challengeTtlMs: SECOND } })
    const s = await pairDevice(h)
    const challenge = await challengeFor(h, s, 'passkeys/register/options')
    h.clock.advance(2 * SECOND)
    const late = await register(h, s, authenticator(), { challenge })
    expect(late.status).toBe(401)
    expect(errorOf(late).message).toContain('challenge')
  })

  test('registering the same credential twice is a conflict', async () => {
    const { h, s, a } = await enrolled()
    expect((await assertWith(h, s, a)).status).toBe(200)
    expect((await register(h, s, a)).status).toBe(409)
  })

  test('the first passkey needs no step-up; a second one does', async () => {
    const { h, s, a } = await enrolled()
    const blocked = await http(h, 'POST', 'passkeys/register/options', {}, { cookie: s.cookie, csrf: s.csrf })
    expect(blocked.status).toBe(403)
    expect(errorOf(blocked).code).toBe('step_up_required')
    expect((await assertWith(h, s, a)).status).toBe(200)
    expect((await register(h, s, authenticator())).status).toBe(201)
  })

  test('registration requires the CSRF header', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    expect((await http(h, 'POST', 'passkeys/register/options', {}, { cookie: s.cookie })).status).toBe(403)
  })
})

describe('passkey assertion and step-up', () => {
  test('without an enrolled passkey, HTTP admin actions are refused and the local API is the way', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    const refused = await createInvite(h, s)
    expect(refused.status).toBe(403)
    expect(errorOf(refused).code).toBe('step_up_required')
    expect(errorOf(refused).message).toContain('local admin')
    const noPasskeySession = must(await authenticate(h, s)).browserSessionId
    expect(errorCode(() => h.admin.requireRecentStepUp(noPasskeySession))).toBe('step_up_required')
    expect(h.admin.createInvite({ createdBy: h.ownerId, workspaceId: 'w1', role: 'viewer' }).inviteId).toMatch(/^inv_/)
    expect((await http(h, 'POST', 'passkeys/assert/options', {}, { cookie: s.cookie, csrf: s.csrf })).status).toBe(409)
  })

  test('a verified assertion unlocks sensitive HTTP actions for a limited time', async () => {
    const { h, s, a } = await enrolled()
    const before = await createInvite(h, s)
    expect(before.status).toBe(403)
    expect(errorOf(before).message).toContain('recent passkey')
    const verified = await assertWith(h, s, a)
    expect(verified.status).toBe(200)
    expect(typeof verified.json.step_up_at).toBe('string')
    const created = await createInvite(h, s)
    expect(created.status).toBe(201)
    expect(String(created.json.url)).toContain('/invite#')
    const sessionId = must(await authenticate(h, s)).browserSessionId
    h.admin.requireRecentStepUp(sessionId, MINUTE)
    h.clock.advance(2 * MINUTE)
    expect(errorCode(() => h.admin.requireRecentStepUp(sessionId, MINUTE))).toBe('step_up_required')
    h.admin.requireRecentStepUp(sessionId, 10 * MINUTE)
  })

  test('step-up belongs to one browser session', async () => {
    const { h, s, a } = await enrolled()
    const other = await pairDevice(h)
    expect((await assertWith(h, s, a)).status).toBe(200)
    expect((await createInvite(h, s)).status).toBe(201)
    expect(errorOf(await createInvite(h, other)).code).toBe('step_up_required')
  })

  test('a failed assertion grants nothing', async () => {
    const { h, s, a } = await enrolled()
    const sessionId = must(await authenticate(h, s)).browserSessionId
    const results = [
      await assertWith(h, s, a, { rpId: 'evil.test' }),
      await assertWith(h, s, a, { origin: 'https://evil.test' }),
      await assertWith(h, s, a, { flags: UP }),
      await assertWith(h, s, a, { flags: UV }),
      await assertWith(h, s, a, { corruptSignature: true }),
      await assertWith(h, s, a, { type: 'webauthn.create' }),
      await assertWith(h, s, authenticator()),
    ]
    expect(results.map(result => result.status)).toEqual([403, 403, 403, 403, 401, 400, 401])
    expect(errorCode(() => h.admin.requireRecentStepUp(sessionId))).toBe('step_up_required')
    expect(errorOf(await createInvite(h, s)).code).toBe('step_up_required')
  })

  test('a replayed assertion challenge is refused', async () => {
    const { h, s, a } = await enrolled()
    const challenge = await challengeFor(h, s, 'passkeys/assert/options')
    expect((await assertWith(h, s, a, { challenge })).status).toBe(200)
    const replay = await assertWith(h, s, a, { challenge })
    expect(replay.status).toBe(401)
    expect(errorOf(replay).message).toContain('challenge')
  })

  test('the signature counter must increase once it is in use', async () => {
    const { h, s, a } = await enrolled()
    expect((await assertWith(h, s, a, { signCount: 5 })).status).toBe(200)
    const cloned = await assertWith(h, s, a, { signCount: 5 })
    expect(cloned.status).toBe(403)
    expect(errorOf(cloned).message).toContain('counter')
    expect((await assertWith(h, s, a, { signCount: 4 })).status).toBe(403)
    expect((await assertWith(h, s, a, { signCount: 6 })).status).toBe(200)
  })

  test('authenticators without a counter (always 0) are accepted', async () => {
    const { h, s, a } = await enrolled()
    expect((await assertWith(h, s, a, { signCount: 0 })).status).toBe(200)
    expect((await assertWith(h, s, a, { signCount: 0 })).status).toBe(200)
  })

  test('EdDSA assertions verify', async () => {
    const { h, s, a } = await enrolled(-8)
    expect((await assertWith(h, s, a)).status).toBe(200)
    expect((await createInvite(h, s)).status).toBe(201)
  })

  test('removing a passkey needs step-up and returns the user to the local-only path', async () => {
    const { h, s, a } = await enrolled()
    const id = b64u(a.credentialId)
    expect((await http(h, 'POST', 'passkeys/remove', { id }, { cookie: s.cookie, csrf: s.csrf })).status).toBe(403)
    expect((await assertWith(h, s, a)).status).toBe(200)
    expect((await http(h, 'POST', 'passkeys/remove', { id }, { cookie: s.cookie, csrf: s.csrf })).status).toBe(200)
    expect((await http(h, 'GET', 'passkeys', undefined, { cookie: s.cookie })).json.passkeys).toEqual([])
    expect(errorOf(await createInvite(h, s)).code).toBe('step_up_required')
  })

  test('enrolment and step-up are audited without credential material', async () => {
    const { h, s, a } = await enrolled()
    expect((await assertWith(h, s, a)).status).toBe(200)
    const actions = h.admin.listAudit().map(event => event.action)
    expect(actions.some(action => action.startsWith('identity.passkey.register:'))).toBe(true)
    expect(actions).toContain('identity.passkey.step_up')
    expect(JSON.stringify(h.admin.listAudit())).not.toContain(b64u(a.credentialId))
    expect(h.admin.verifyAuditChain().ok).toBe(true)
  })
})

describe('CBOR and authenticator data parsing', () => {
  test('decodes integers, text, bytes, arrays and maps', () => {
    const map = decodeCbor(Buffer.from('a1626964182a', 'hex'))
    expect(map instanceof Map && map.get('id')).toBe(42)
    expect(decodeCbor(Buffer.from('3863', 'hex'))).toBe(-100)
    expect(decodeCbor(cbor([1, 'a', Buffer.from([1, 2])]))).toEqual([1, 'a', new Uint8Array([1, 2])])
  })

  test('refuses malformed or unsupported input', () => {
    const bad: [string, string][] = [
      ['trailing bytes', '0000'],
      ['indefinite length', '9f01ff'],
      ['tag', 'c101'],
      ['float', 'f90000'],
      ['duplicate key', 'a2616101616102'],
      ['truncated byte string', '4401'],
      ['truncated input', ''],
      ['too deep', '81'.repeat(12) + '00'],
    ]
    for (const [name, hex] of bad) expect([name, errorCode(() => decodeCbor(Buffer.from(hex, 'hex')))]).toEqual([name, 'invalid_request'])
  })

  test('authenticator data parsing checks lengths', () => {
    expect(errorCode(() => parseAuthData(new Uint8Array(10)))).toBe('invalid_request')
    const plain = parseAuthData(authData({ flags: UP | UV, signCount: 7 }))
    expect(plain.signCount).toBe(7)
    expect(plain.credential).toBeUndefined()
    const truncated = authData({ flags: UP | UV | AT, signCount: 0 })
    expect(errorCode(() => parseAuthData(truncated))).toBe('invalid_request')
    const a = authenticator()
    const parsed = parseAuthData(authData({ flags: UP | UV | AT, signCount: 0, attested: a }))
    expect(Buffer.from(parsed.credential?.id ?? []).equals(a.credentialId)).toBe(true)
  })
})
