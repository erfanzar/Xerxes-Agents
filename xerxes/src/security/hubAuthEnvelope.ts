// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, createPrivateKey, createPublicKey, sign, verify, type KeyObject } from 'node:crypto'

/**
 * Signed identity envelope the Hub attaches to forwarded daemon RPCs
 * (`params.hub_auth`). The Hub signs with its Ed25519 application key; a
 * daemon verifies against a key it pinned through explicit enrollment.
 *
 * Signature input: UTF-8 of `canonicalJson(claims)`. `params_hash` is the
 * sha256 hex of `canonicalJson(params without hub_auth)`, so a signature can't
 * be moved onto different arguments.
 */

export const HUB_AUTH_VERSION = 1 as const

export interface HubAuthClaims {
  readonly hub_id: string
  readonly principal_id: string
  readonly principal_kind: 'user' | 'agent'
  readonly user_id?: string
  readonly device_id?: string
  readonly display_name: string
  /** For agent principals: the human who asked and the human who confirmed (attribution only). */
  readonly requested_by?: string
  readonly confirmed_by?: string
  readonly workspace_id: string
  readonly session_id?: string
  readonly method: string
  readonly params_hash: string
  readonly request_id: string
  readonly nonce: string
  /** Unix milliseconds. */
  readonly issued_at: number
  readonly expires_at: number
  readonly scopes: readonly string[]
  /** Present only for a human interaction answer: binds the decision to one action. */
  readonly human_action_id?: string
}

export interface HubAuthEnvelope {
  readonly v: typeof HUB_AUTH_VERSION
  readonly claims: HubAuthClaims
  /** base64url Ed25519 signature. */
  readonly signature: string
}

/** Deterministic JSON: object keys sorted, no whitespace, undefined members dropped. */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    if (typeof value === 'number' && !Number.isFinite(value)) throw new TypeError('canonicalJson: non-finite number')
    if (value === undefined) throw new TypeError('canonicalJson: undefined is not JSON')
    return JSON.stringify(value)
  }
  if (Array.isArray(value)) return '[' + value.map(item => (item === undefined ? 'null' : canonicalJson(item))).join(',') + ']'
  const record = value as Record<string, unknown>
  const keys = Object.keys(record).filter(key => record[key] !== undefined).sort()
  return '{' + keys.map(key => JSON.stringify(key) + ':' + canonicalJson(record[key])).join(',') + '}'
}

export function paramsHash(params: Readonly<Record<string, unknown>>): string {
  const { hub_auth: _ignored, ...rest } = params as Record<string, unknown>
  return createHash('sha256').update(canonicalJson(rest)).digest('hex')
}

/** sha256 hex of the SPKI DER encoding; shown in the pairing QR and on enrollment. */
export function publicKeyFingerprint(publicKey: KeyObject | string): string {
  const key = typeof publicKey === 'string' ? createPublicKey(publicKey) : publicKey
  return createHash('sha256').update(key.export({ type: 'spki', format: 'der' })).digest('hex')
}

export function signHubAuth(claims: HubAuthClaims, privateKey: KeyObject | string): HubAuthEnvelope {
  const key = typeof privateKey === 'string' ? createPrivateKey(privateKey) : privateKey
  if (key.asymmetricKeyType !== 'ed25519') throw new Error('hub_auth requires an Ed25519 signing key')
  const signature = sign(null, Buffer.from(canonicalJson(claims), 'utf8'), key).toString('base64url')
  return { v: HUB_AUTH_VERSION, claims, signature }
}

export type HubAuthVerification =
  | { readonly ok: true; readonly claims: HubAuthClaims }
  | { readonly ok: false; readonly reason: 'malformed' | 'bad_signature' | 'expired' | 'not_yet_valid' | 'wrong_method' | 'params_mismatch' }

/**
 * Check structure, signature, time window, method and params binding.
 * Audience (workspace/hub id), nonce replay and scope checks belong to the
 * caller, which knows its enrollment and keeps the nonce cache.
 */
export function verifyHubAuth(
  envelope: unknown,
  expected: { readonly publicKey: KeyObject | string; readonly method: string; readonly params: Readonly<Record<string, unknown>>; readonly now?: number; readonly clockSkewMs?: number },
): HubAuthVerification {
  const parsed = parseEnvelope(envelope)
  if (!parsed) return { ok: false, reason: 'malformed' }
  const key = typeof expected.publicKey === 'string' ? createPublicKey(expected.publicKey) : expected.publicKey
  let valid = false
  try {
    valid = verify(null, Buffer.from(canonicalJson(parsed.claims), 'utf8'), key, Buffer.from(parsed.signature, 'base64url'))
  } catch {
    valid = false
  }
  if (!valid) return { ok: false, reason: 'bad_signature' }
  const now = expected.now ?? Date.now()
  const skew = expected.clockSkewMs ?? 30_000
  if (parsed.claims.issued_at > now + skew) return { ok: false, reason: 'not_yet_valid' }
  if (parsed.claims.expires_at < now - skew) return { ok: false, reason: 'expired' }
  if (parsed.claims.method !== expected.method) return { ok: false, reason: 'wrong_method' }
  if (parsed.claims.params_hash !== paramsHash(expected.params)) return { ok: false, reason: 'params_mismatch' }
  return { ok: true, claims: parsed.claims }
}

function parseEnvelope(value: unknown): HubAuthEnvelope | undefined {
  if (typeof value !== 'object' || value === null) return undefined
  const record = value as Record<string, unknown>
  if (record.v !== HUB_AUTH_VERSION || typeof record.signature !== 'string') return undefined
  const claims = record.claims
  if (typeof claims !== 'object' || claims === null) return undefined
  const c = claims as Record<string, unknown>
  const strings = ['hub_id', 'principal_id', 'display_name', 'workspace_id', 'method', 'params_hash', 'request_id', 'nonce'] as const
  for (const field of strings) if (typeof c[field] !== 'string' || !(c[field] as string)) return undefined
  if (c.principal_kind !== 'user' && c.principal_kind !== 'agent') return undefined
  if (!Number.isSafeInteger(c.issued_at) || !Number.isSafeInteger(c.expires_at)) return undefined
  if (!Array.isArray(c.scopes) || !c.scopes.every(scope => typeof scope === 'string')) return undefined
  for (const field of ['user_id', 'device_id', 'requested_by', 'confirmed_by', 'session_id', 'human_action_id'] as const) {
    if (c[field] !== undefined && typeof c[field] !== 'string') return undefined
  }
  return { v: HUB_AUTH_VERSION, claims: c as unknown as HubAuthClaims, signature: record.signature }
}
