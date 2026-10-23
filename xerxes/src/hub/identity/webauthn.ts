// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, createPublicKey, verify } from 'node:crypto'

import { HubIdentityError, base64url, cleanText, constantTimeEqual, newId, randomToken, type IdentityContext } from './core.js'
import type { PasskeyRow, SessionRow, UserRow } from './store.js'

/**
 * Server-verified WebAuthn (passkeys) used as a *step-up* for sensitive HTTP
 * administration (membership, grants, invites). Verified here, natively:
 *  - single-use challenges bound to the browser session, 5 minute expiry;
 *  - RP ID = host of the Hub public origin; exact `clientDataJSON.origin`;
 *  - `rpIdHash`, user presence AND user verification flags;
 *  - ES256 (COSE -7) and EdDSA (COSE -8) public keys;
 *  - signCount monotonicity (a non-increasing non-zero counter is a clone signal);
 *  - attestation format `none` only (no attestation trust decisions are made).
 *
 * Residual risk (documented): a stolen live session can enrol the user's first
 * passkey and then step up. Enrolment is audited, and the local admin CLI never
 * needs a passkey.
 */

export const STEP_UP_MAX_AGE_MS = 5 * 60 * 1000
const MAX_CBOR_DEPTH = 8
const MAX_CBOR_ITEMS = 64
const FLAG_UP = 0x01
const FLAG_UV = 0x04
const FLAG_AT = 0x40
const FLAG_ED = 0x80
const MAX_PASSKEYS_PER_USER = 10

// ---------------------------------------------------------------- CBOR (minimal)

export type CborValue = number | string | boolean | null | Uint8Array | CborValue[] | Map<CborValue, CborValue>

function cborError(message: string): never {
  throw new HubIdentityError('invalid_request', `malformed CBOR: ${message}`)
}

/** Decode one definite-length CBOR item at `offset`. Floats, tags and indefinite lengths are refused. */
export function decodeCborAt(bytes: Uint8Array, offset: number, depth = 0): { readonly value: CborValue; readonly next: number } {
  if (depth > MAX_CBOR_DEPTH) cborError('nesting too deep')
  const initial = bytes[offset]
  if (initial === undefined) cborError('unexpected end')
  const major = initial >> 5
  const info = initial & 31
  let cursor = offset + 1
  let argument: number
  if (info < 24) argument = info
  else if (info >= 24 && info <= 27) {
    const width = 1 << (info - 24)
    if (cursor + width > bytes.length) cborError('unexpected end')
    let value = 0n
    for (let i = 0; i < width; i += 1) value = (value << 8n) | BigInt(bytes[cursor + i] ?? 0)
    cursor += width
    if (major === 7) cborError('floats and extended simple values are not supported')
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) cborError('integer too large')
    argument = Number(value)
  } else cborError('indefinite lengths and reserved values are not supported')

  switch (major) {
    case 0:
      return { value: argument, next: cursor }
    case 1:
      return { value: -1 - argument, next: cursor }
    case 2: {
      if (cursor + argument > bytes.length) cborError('byte string overruns input')
      return { value: bytes.slice(cursor, cursor + argument), next: cursor + argument }
    }
    case 3: {
      if (cursor + argument > bytes.length) cborError('text string overruns input')
      let text: string
      try {
        text = new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(cursor, cursor + argument))
      } catch {
        return cborError('text string is not UTF-8')
      }
      return { value: text, next: cursor + argument }
    }
    case 4: {
      if (argument > MAX_CBOR_ITEMS) cborError('array too long')
      const items: CborValue[] = []
      for (let i = 0; i < argument; i += 1) {
        const item = decodeCborAt(bytes, cursor, depth + 1)
        items.push(item.value)
        cursor = item.next
      }
      return { value: items, next: cursor }
    }
    case 5: {
      if (argument > MAX_CBOR_ITEMS) cborError('map too large')
      const map = new Map<CborValue, CborValue>()
      for (let i = 0; i < argument; i += 1) {
        const key = decodeCborAt(bytes, cursor, depth + 1)
        const value = decodeCborAt(bytes, key.next, depth + 1)
        if (typeof key.value !== 'string' && typeof key.value !== 'number') cborError('map keys must be strings or integers')
        if (map.has(key.value)) cborError('duplicate map key')
        map.set(key.value, value.value)
        cursor = value.next
      }
      return { value: map, next: cursor }
    }
    case 7:
      if (info === 20) return { value: false, next: cursor }
      if (info === 21) return { value: true, next: cursor }
      if (info === 22) return { value: null, next: cursor }
      return cborError('unsupported simple value')
    default:
      return cborError('tags are not supported')
  }
}

export function decodeCbor(bytes: Uint8Array): CborValue {
  const { value, next } = decodeCborAt(bytes, 0)
  if (next !== bytes.length) cborError('trailing bytes')
  return value
}

// ---------------------------------------------------------------- authenticator data

export interface ParsedAuthData {
  readonly rpIdHash: Uint8Array
  readonly flags: number
  readonly signCount: number
  readonly credential?: { readonly aaguid: Uint8Array; readonly id: Uint8Array; readonly publicKey: Map<CborValue, CborValue> }
}

export function parseAuthData(bytes: Uint8Array): ParsedAuthData {
  if (bytes.length < 37) throw new HubIdentityError('invalid_request', 'authenticator data is too short')
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const flags = bytes[32] ?? 0
  const base = { rpIdHash: bytes.slice(0, 32), flags, signCount: view.getUint32(33, false) }
  if ((flags & FLAG_AT) === 0) {
    if (bytes.length > 37 && (flags & FLAG_ED) === 0) throw new HubIdentityError('invalid_request', 'unexpected trailing authenticator data')
    return base
  }
  if (bytes.length < 55) throw new HubIdentityError('invalid_request', 'attested credential data is truncated')
  const idLength = view.getUint16(53, false)
  if (idLength === 0 || idLength > 1023 || 55 + idLength > bytes.length) throw new HubIdentityError('invalid_request', 'credential id length is invalid')
  const key = decodeCborAt(bytes, 55 + idLength)
  if (!(key.value instanceof Map)) throw new HubIdentityError('invalid_request', 'credential public key must be a COSE map')
  if (key.next !== bytes.length && (flags & FLAG_ED) === 0) throw new HubIdentityError('invalid_request', 'unexpected trailing authenticator data')
  return { ...base, credential: { aaguid: bytes.slice(37, 53), id: bytes.slice(55, 55 + idLength), publicKey: key.value } }
}

export interface CoseKey {
  readonly alg: -7 | -8
  /** SPKI DER. */
  readonly spki: Buffer
}

function coseBytes(cose: Map<CborValue, CborValue>, label: number): Uint8Array {
  const value = cose.get(label)
  if (!(value instanceof Uint8Array) || value.length !== 32) throw new HubIdentityError('invalid_request', 'COSE key coordinate must be 32 bytes')
  return value
}

/** Convert a COSE_Key (ES256 P-256 or EdDSA Ed25519) to SPKI DER. */
export function coseToSpki(cose: Map<CborValue, CborValue>): CoseKey {
  const kty = cose.get(1)
  const alg = cose.get(3)
  const crv = cose.get(-1)
  try {
    if (kty === 2 && alg === -7 && crv === 1) {
      const key = createPublicKey({ key: { kty: 'EC', crv: 'P-256', x: base64url(coseBytes(cose, -2)), y: base64url(coseBytes(cose, -3)) }, format: 'jwk' })
      return { alg: -7, spki: Buffer.from(key.export({ type: 'spki', format: 'der' })) }
    }
    if (kty === 1 && alg === -8 && crv === 6) {
      const key = createPublicKey({ key: { kty: 'OKP', crv: 'Ed25519', x: base64url(coseBytes(cose, -2)) }, format: 'jwk' })
      return { alg: -8, spki: Buffer.from(key.export({ type: 'spki', format: 'der' })) }
    }
  } catch (error) {
    if (error instanceof HubIdentityError) throw error
    throw new HubIdentityError('invalid_request', 'credential public key is not a valid point', { cause: error })
  }
  throw new HubIdentityError('invalid_request', 'only ES256 (-7) and EdDSA (-8) credential keys are supported')
}

// ---------------------------------------------------------------- challenge + origin handling

export function rpIdOf(ctx: Pick<IdentityContext, 'publicOrigin'>): string {
  if (ctx.publicOrigin === undefined) {
    throw new HubIdentityError('unavailable', 'passkeys need the Hub public origin to be configured')
  }
  return new URL(ctx.publicOrigin).hostname
}

function expectedOrigins(ctx: Pick<IdentityContext, 'publicOrigin' | 'extraOrigins'>): string[] {
  const rpId = rpIdOf(ctx)
  const origins = new Set<string>()
  if (ctx.publicOrigin !== undefined) origins.add(ctx.publicOrigin)
  for (const origin of ctx.extraOrigins) if (new URL(origin).hostname === rpId) origins.add(origin)
  return [...origins]
}

function issueChallenge(ctx: IdentityContext, purpose: 'register' | 'assert', session: SessionRow): string {
  const now = ctx.clock.now()
  for (const [key, entry] of ctx.challenges) if (entry.expiresAt <= now) ctx.challenges.delete(key)
  if (ctx.challenges.size > 1000) throw new HubIdentityError('rate_limited', 'too many pending passkey challenges')
  const challenge = randomToken(ctx, 32)
  ctx.challenges.set(challenge, { purpose, sessionId: session.id, userId: session.user_id, expiresAt: now + ctx.limits.challengeTtlMs })
  return challenge
}

/** Single use: the challenge is removed before any check, so a failed attempt cannot be retried. */
function consumeChallenge(ctx: IdentityContext, challenge: string, purpose: 'register' | 'assert', session: SessionRow): void {
  const entry = ctx.challenges.get(challenge)
  ctx.challenges.delete(challenge)
  if (!entry || entry.purpose !== purpose || entry.sessionId !== session.id || entry.userId !== session.user_id || entry.expiresAt <= ctx.clock.now()) {
    throw new HubIdentityError('unauthorized', 'the passkey challenge is invalid, expired or was already used')
  }
}

function decodeField(value: unknown, field: string, max = 16 * 1024): Buffer {
  if (typeof value !== 'string' || value.length === 0 || value.length > max * 2 || !/^[A-Za-z0-9_-]+$/.test(value)) {
    throw new HubIdentityError('invalid_request', `${field} must be base64url`)
  }
  const bytes = Buffer.from(value, 'base64url')
  if (bytes.length > max) throw new HubIdentityError('invalid_request', `${field} is too large`)
  return bytes
}

interface ClientData {
  readonly type: string
  readonly challenge: string
  readonly origin: string
  readonly crossOrigin?: boolean
}

function parseClientData(raw: Buffer, expectedType: 'webauthn.create' | 'webauthn.get', ctx: IdentityContext): ClientData {
  let value: unknown
  try {
    value = JSON.parse(raw.toString('utf8')) as unknown
  } catch {
    throw new HubIdentityError('invalid_request', 'clientDataJSON is not JSON')
  }
  if (typeof value !== 'object' || value === null) throw new HubIdentityError('invalid_request', 'clientDataJSON must be an object')
  const record = value as Record<string, unknown>
  if (record.type !== expectedType) throw new HubIdentityError('invalid_request', `clientDataJSON.type must be ${expectedType}`)
  if (typeof record.challenge !== 'string' || typeof record.origin !== 'string') throw new HubIdentityError('invalid_request', 'clientDataJSON is missing challenge or origin')
  if (record.crossOrigin === true) throw new HubIdentityError('forbidden', 'cross-origin passkey ceremonies are refused')
  if (!expectedOrigins(ctx).includes(record.origin)) throw new HubIdentityError('forbidden', 'passkey origin does not match this Hub')
  return { type: expectedType, challenge: record.challenge, origin: record.origin }
}

function checkAuthData(ctx: IdentityContext, auth: ParsedAuthData): void {
  const expected = createHash('sha256').update(rpIdOf(ctx)).digest('hex')
  if (!constantTimeEqual(Buffer.from(auth.rpIdHash).toString('hex'), expected)) throw new HubIdentityError('forbidden', 'passkey relying party does not match this Hub')
  if ((auth.flags & FLAG_UP) === 0) throw new HubIdentityError('forbidden', 'user presence is required')
  if ((auth.flags & FLAG_UV) === 0) throw new HubIdentityError('forbidden', 'user verification is required')
}

// ---------------------------------------------------------------- ceremonies

export function passkeysFor(ctx: Pick<IdentityContext, 'store'>, userId: string): PasskeyRow[] {
  return ctx.store.all<PasskeyRow>('SELECT * FROM hub_passkeys WHERE user_id = ? ORDER BY created_at ASC, id ASC', userId)
}

export function registrationOptions(ctx: IdentityContext, user: UserRow, session: SessionRow): Record<string, unknown> {
  const rpId = rpIdOf(ctx)
  const existing = passkeysFor(ctx, user.id)
  if (existing.length >= MAX_PASSKEYS_PER_USER) throw new HubIdentityError('conflict', `at most ${MAX_PASSKEYS_PER_USER} passkeys per user`)
  return {
    challenge: issueChallenge(ctx, 'register', session),
    rp: { id: rpId, name: 'Xerxes Hub' },
    user: { id: base64url(new TextEncoder().encode(user.id)), name: user.display_name, displayName: user.display_name },
    pubKeyCredParams: [{ type: 'public-key', alg: -7 }, { type: 'public-key', alg: -8 }],
    authenticatorSelection: { userVerification: 'required', residentKey: 'preferred' },
    attestation: 'none',
    timeout: ctx.limits.challengeTtlMs,
    excludeCredentials: existing.map(key => ({ type: 'public-key', id: key.id })),
  }
}

export function verifyRegistration(
  ctx: IdentityContext,
  user: UserRow,
  session: SessionRow,
  body: { readonly clientDataJSON: unknown; readonly attestationObject: unknown; readonly label?: unknown },
): { readonly id: string; readonly alg: number; readonly label: string } {
  const clientRaw = decodeField(body.clientDataJSON, 'clientDataJSON')
  const attestationRaw = decodeField(body.attestationObject, 'attestationObject')
  const clientData = parseClientData(clientRaw, 'webauthn.create', ctx)
  consumeChallenge(ctx, clientData.challenge, 'register', session)
  const attestation = decodeCbor(attestationRaw)
  if (!(attestation instanceof Map)) throw new HubIdentityError('invalid_request', 'attestationObject must be a CBOR map')
  if (attestation.get('fmt') !== 'none') throw new HubIdentityError('invalid_request', 'only attestation format "none" is accepted')
  const authBytes = attestation.get('authData')
  if (!(authBytes instanceof Uint8Array)) throw new HubIdentityError('invalid_request', 'attestationObject has no authData')
  const auth = parseAuthData(authBytes)
  checkAuthData(ctx, auth)
  if (!auth.credential) throw new HubIdentityError('invalid_request', 'registration carried no credential')
  const { alg, spki } = coseToSpki(auth.credential.publicKey)
  const id = base64url(auth.credential.id)
  const label = cleanText(body.label, 'Passkey', 60)
  ctx.store.transaction(() => {
    if (passkeysFor(ctx, user.id).length >= MAX_PASSKEYS_PER_USER) throw new HubIdentityError('conflict', `at most ${MAX_PASSKEYS_PER_USER} passkeys per user`)
    if (ctx.store.one('SELECT id FROM hub_passkeys WHERE id = ?', id)) throw new HubIdentityError('conflict', 'this credential is already registered')
    ctx.store.run(
      'INSERT INTO hub_passkeys (id, user_id, public_key, alg, sign_count, label, created_at, last_used_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)',
      id,
      user.id,
      spki.toString('base64url'),
      alg,
      auth.signCount,
      label,
      ctx.clock.now(),
    )
  })
  return { id, alg, label }
}

export function assertionOptions(ctx: IdentityContext, user: UserRow, session: SessionRow): Record<string, unknown> {
  const rpId = rpIdOf(ctx)
  const keys = passkeysFor(ctx, user.id)
  if (keys.length === 0) throw new HubIdentityError('conflict', 'no passkey is enrolled for this user')
  return {
    challenge: issueChallenge(ctx, 'assert', session),
    rpId,
    allowCredentials: keys.map(key => ({ type: 'public-key', id: key.id })),
    userVerification: 'required',
    timeout: ctx.limits.challengeTtlMs,
  }
}

export function verifyAssertion(
  ctx: IdentityContext,
  user: UserRow,
  session: SessionRow,
  body: { readonly id: unknown; readonly clientDataJSON: unknown; readonly authenticatorData: unknown; readonly signature: unknown },
): { readonly stepUpAt: number } {
  const clientRaw = decodeField(body.clientDataJSON, 'clientDataJSON')
  const authRaw = decodeField(body.authenticatorData, 'authenticatorData')
  const signature = decodeField(body.signature, 'signature', 1024)
  if (typeof body.id !== 'string' || body.id.length > 1400) throw new HubIdentityError('invalid_request', 'id is required')
  const clientData = parseClientData(clientRaw, 'webauthn.get', ctx)
  consumeChallenge(ctx, clientData.challenge, 'assert', session)
  const key = ctx.store.one<PasskeyRow>('SELECT * FROM hub_passkeys WHERE id = ? AND user_id = ?', body.id, user.id)
  if (!key) throw new HubIdentityError('unauthorized', 'unknown passkey')
  const auth = parseAuthData(authRaw)
  checkAuthData(ctx, auth)
  const signed = Buffer.concat([authRaw, createHash('sha256').update(clientRaw).digest()])
  const publicKey = createPublicKey({ key: Buffer.from(key.public_key, 'base64url'), format: 'der', type: 'spki' })
  let valid = false
  try {
    valid = verify(key.alg === -8 ? null : 'sha256', signed, publicKey, signature)
  } catch {
    valid = false
  }
  if (!valid) throw new HubIdentityError('unauthorized', 'passkey signature is invalid')
  const now = ctx.clock.now()
  ctx.store.transaction(() => {
    const current = ctx.store.one<PasskeyRow>('SELECT * FROM hub_passkeys WHERE id = ?', key.id)
    if (!current) throw new HubIdentityError('unauthorized', 'unknown passkey')
    if ((auth.signCount !== 0 || current.sign_count !== 0) && auth.signCount <= current.sign_count) {
      throw new HubIdentityError('forbidden', 'passkey counter did not increase; the credential may be cloned')
    }
    ctx.store.run('UPDATE hub_passkeys SET sign_count = ?, last_used_at = ? WHERE id = ?', auth.signCount, now, key.id)
    ctx.store.run('UPDATE hub_sessions SET stepup_at = ? WHERE id = ?', now, session.id)
  })
  return { stepUpAt: now }
}

/**
 * Guard for sensitive admin actions over HTTP. Throws `step_up_required` when
 * the user has no passkey (the action must then be done from the local admin
 * API/CLI) or when the browser session has no passkey assertion within `maxAgeMs`.
 */
export function requireRecentStepUp(ctx: IdentityContext, sessionId: string, maxAgeMs: number = STEP_UP_MAX_AGE_MS): void {
  const session = ctx.store.one<SessionRow>('SELECT * FROM hub_sessions WHERE id = ?', sessionId)
  if (!session || session.revoked_at !== null) throw new HubIdentityError('unauthorized', 'session is not active')
  if (passkeysFor(ctx, session.user_id).length === 0) {
    throw new HubIdentityError('step_up_required', 'no passkey is enrolled: enrol one, or run this action from the local admin CLI')
  }
  if (session.stepup_at === null || ctx.clock.now() - session.stepup_at > maxAgeMs) {
    throw new HubIdentityError('step_up_required', 'a recent passkey verification is required')
  }
}

export function removePasskey(ctx: IdentityContext, userId: string, id: string): void {
  if (ctx.store.run('DELETE FROM hub_passkeys WHERE id = ? AND user_id = ?', id, userId) !== 1) throw new HubIdentityError('not_found', 'passkey does not exist')
}

export function newPasskeyLabelId(ctx: Pick<IdentityContext, 'random'>): string {
  return newId(ctx, 'pk')
}
