// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { generateKeyPairSync } from 'node:crypto'

import { expect, test } from 'bun:test'

import { canonicalJson, paramsHash, publicKeyFingerprint, signHubAuth, verifyHubAuth, type HubAuthClaims } from '../src/security/hubAuthEnvelope.js'

const { privateKey, publicKey } = generateKeyPairSync('ed25519')
const other = generateKeyPairSync('ed25519')

function claims(params: Record<string, unknown>, overrides: Partial<HubAuthClaims> = {}): HubAuthClaims {
  return {
    hub_id: 'hub1', principal_id: 'alice', principal_kind: 'user', display_name: 'Alice', workspace_id: 'w1',
    method: 'turn.steer', params_hash: paramsHash(params), request_id: 'r1', nonce: 'n1',
    issued_at: 1_000_000, expires_at: 1_060_000, scopes: ['session.steer'], ...overrides,
  }
}

test('canonical JSON sorts keys and drops undefined', () => {
  expect(canonicalJson({ b: 1, a: [2, { d: undefined, c: 'x' }] })).toBe('{"a":[2,{"c":"x"}],"b":1}')
  expect(() => canonicalJson({ a: Number.NaN })).toThrow()
})

test('params hash ignores hub_auth and key order', () => {
  expect(paramsHash({ text: 'go', session_id: 's' })).toBe(paramsHash({ session_id: 's', text: 'go', hub_auth: { any: 1 } }))
  expect(paramsHash({ text: 'go' })).not.toBe(paramsHash({ text: 'stop' }))
})

test('valid envelopes verify; tampering, wrong key, method, params and time window are refused', () => {
  const params = { session_id: 's', text: 'go' }
  const envelope = signHubAuth(claims(params), privateKey)
  const base = { publicKey, method: 'turn.steer', params, now: 1_010_000 }
  expect(verifyHubAuth(envelope, base)).toMatchObject({ ok: true })
  expect(verifyHubAuth(envelope, { ...base, publicKey: other.publicKey })).toEqual({ ok: false, reason: 'bad_signature' })
  expect(verifyHubAuth({ ...envelope, claims: { ...envelope.claims, principal_id: 'mallory' } }, base)).toEqual({ ok: false, reason: 'bad_signature' })
  expect(verifyHubAuth(envelope, { ...base, method: 'turn.submit' })).toEqual({ ok: false, reason: 'wrong_method' })
  expect(verifyHubAuth(envelope, { ...base, params: { session_id: 's', text: 'rm -rf' } })).toEqual({ ok: false, reason: 'params_mismatch' })
  expect(verifyHubAuth(envelope, { ...base, now: 2_000_000 })).toEqual({ ok: false, reason: 'expired' })
  expect(verifyHubAuth(envelope, { ...base, now: 0 })).toEqual({ ok: false, reason: 'not_yet_valid' })
  expect(verifyHubAuth({ v: 1, claims: {}, signature: 'x' }, base)).toEqual({ ok: false, reason: 'malformed' })
})

test('fingerprints are stable sha256 hex and refuse non-Ed25519 keys for signing', () => {
  expect(publicKeyFingerprint(publicKey)).toMatch(/^[0-9a-f]{64}$/)
  expect(publicKeyFingerprint(publicKey)).toBe(publicKeyFingerprint(publicKey.export({ type: 'spki', format: 'pem' }).toString()))
  const rsa = generateKeyPairSync('rsa', { modulusLength: 1024 })
  expect(() => signHubAuth(claims({}), rsa.privateKey)).toThrow(/Ed25519/)
})
