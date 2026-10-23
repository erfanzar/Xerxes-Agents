// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { generateKeyPairSync, randomUUID, type KeyObject } from 'node:crypto'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { paramsHash, publicKeyFingerprint, signHubAuth, type HubAuthClaims } from '../src/security/hubAuthEnvelope.js'
import {
  HubIdentityVerifier,
  enrollHub,
  hubAuthorizationFailure,
  hubMethodPolicy,
  listHubEnrollments,
  revokeHub,
  type VerifiedHubContext,
} from '../src/security/hubIdentityVerifier.js'

const NOW = 1_700_000_000_000
const directories: string[] = []
afterEach(async () => { for (const directory of directories.splice(0)) await rm(directory, { recursive: true, force: true }) })

async function workspace(): Promise<{ root: string; file: string }> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-hub-verifier-'))
  directories.push(root)
  return { root, file: join(root, 'state', 'hub-enrollments.json') }
}

function keys(): { privateKey: KeyObject; pem: string } {
  const pair = generateKeyPairSync('ed25519')
  return { privateKey: pair.privateKey, pem: pair.publicKey.export({ type: 'spki', format: 'pem' }).toString() }
}

function envelope(key: KeyObject, method: string, params: Record<string, unknown>, over: Partial<HubAuthClaims> = {}) {
  const claims: HubAuthClaims = {
    hub_id: 'hub-1', principal_id: 'alice', principal_kind: 'user', user_id: 'user-alice', device_id: 'phone-1', display_name: 'Alice',
    workspace_id: 'ws-1', session_id: 'session-1', method, params_hash: paramsHash(params), request_id: randomUUID(), nonce: randomUUID(),
    issued_at: NOW, expires_at: NOW + 60_000, scopes: ['session.submit', 'session.steer', 'session.read'], ...over,
  }
  return signHubAuth(claims, key)
}

function verifier(file: string, extra: { maxNonces?: number; now?: () => number } = {}): HubIdentityVerifier {
  return new HubIdentityVerifier({ path: file, now: extra.now ?? (() => NOW), ...(extra.maxNonces ? { maxNonces: extra.maxNonces } : {}) })
}

const request = (projectDir: string, over: { sessionId?: string } = {}) => ({
  method: 'turn.submit', params: { text: 'hi' }, projectDir, sessionId: over.sessionId ?? 'session-1',
})

test('a pinned hub verifies and yields a context built only from signed claims; the file is 0600', async () => {
  const { root, file } = await workspace()
  const { privateKey, pem } = keys()
  const record = enrollHub(file, { hub_id: 'hub-1', public_key_pem: pem, project_dirs: [root] })
  expect((await stat(file)).mode & 0o777).toBe(0o600)
  expect(record.fingerprint).toBe(publicKeyFingerprint(pem))
  const result = verifier(file).verify(envelope(privateKey, 'turn.submit', { text: 'hi' }), request(root))
  expect(result).toMatchObject({ ok: true, context: {
    hub_id: 'hub-1', principal_kind: 'user', principal_id: 'alice', user_id: 'user-alice', device_id: 'phone-1',
    display_name: 'Alice', scopes: ['session.submit', 'session.steer', 'session.read'], session_id: 'session-1',
  } })
  expect(listHubEnrollments(file)).toHaveLength(1)
})

test('forged, unknown, mismatched, expired and over-long envelopes are rejected with typed reasons', async () => {
  const { root, file } = await workspace()
  const { privateKey, pem } = keys()
  const attacker = keys()
  enrollHub(file, { hub_id: 'hub-1', public_key_pem: pem, project_dirs: [root] })
  const v = verifier(file)
  const call = (envelopeValue: unknown, over: { sessionId?: string; projectDir?: string } = {}) =>
    v.verify(envelopeValue, { ...request(root), ...(over.projectDir ? { projectDir: over.projectDir } : {}), ...(over.sessionId ? { sessionId: over.sessionId } : {}) })
  expect(call(envelope(attacker.privateKey, 'turn.submit', { text: 'hi' }))).toMatchObject({ ok: false, reason: 'bad_signature' })
  expect(call(envelope(privateKey, 'turn.submit', { text: 'hi' }, { hub_id: 'hub-2' }))).toMatchObject({ ok: false, reason: 'unknown_hub' })
  expect(call({ claims: {} })).toMatchObject({ ok: false, reason: 'malformed' })
  expect(call('nope')).toMatchObject({ ok: false, reason: 'malformed' })
  expect(call(envelope(privateKey, 'turn.steer', { text: 'hi' }))).toMatchObject({ ok: false, reason: 'wrong_method' })
  expect(call(envelope(privateKey, 'turn.submit', { text: 'other' }))).toMatchObject({ ok: false, reason: 'params_mismatch' })
  expect(call(envelope(privateKey, 'turn.submit', { text: 'hi' }, { issued_at: NOW - 200_000, expires_at: NOW - 100_000 }))).toMatchObject({ ok: false, reason: 'expired' })
  expect(call(envelope(privateKey, 'turn.submit', { text: 'hi' }, { issued_at: NOW + 500_000, expires_at: NOW + 560_000 }))).toMatchObject({ ok: false, reason: 'not_yet_valid' })
  expect(call(envelope(privateKey, 'turn.submit', { text: 'hi' }, { expires_at: NOW + 3_600_000 }))).toMatchObject({ ok: false, reason: 'lifetime_too_long' })
  // Wrong audience: another project, another workspace id, another session.
  expect(call(envelope(privateKey, 'turn.submit', { text: 'hi' }), { projectDir: tmpdir() })).toMatchObject({ ok: false, reason: 'hub_not_enrolled_for_project' })
  expect(call(envelope(privateKey, 'turn.submit', { text: 'hi' }), { sessionId: 'session-2' })).toMatchObject({ ok: false, reason: 'session_mismatch' })
  enrollHub(file, { hub_id: 'hub-1', public_key_pem: pem, workspace_ids: ['ws-other'], replace: true })
  expect(call(envelope(privateKey, 'turn.submit', { text: 'hi' }))).toMatchObject({ ok: false, reason: 'hub_not_enrolled_for_workspace' })
})

test('a nonce is single-use, a refused request never burns it, and the cache is bounded', async () => {
  const { root, file } = await workspace()
  const { privateKey, pem } = keys()
  enrollHub(file, { hub_id: 'hub-1', public_key_pem: pem, project_dirs: [root] })
  const v = verifier(file, { maxNonces: 3 })
  const signed = envelope(privateKey, 'turn.submit', { text: 'hi' })
  expect(v.verify(signed, request(root, { sessionId: 'wrong' }))).toMatchObject({ ok: false, reason: 'session_mismatch' })
  expect(v.verify(signed, request(root))).toMatchObject({ ok: true })
  expect(v.verify(signed, request(root))).toMatchObject({ ok: false, reason: 'replayed_nonce' })
  for (let index = 0; index < 8; index++) v.verify(envelope(privateKey, 'turn.submit', { text: 'hi' }), request(root))
  expect(v.nonceCount).toBeLessThanOrEqual(3)
})

test('enrollment refuses a changed key, a non-Ed25519 key, relative paths and never rewrites a corrupt file', async () => {
  const { root, file } = await workspace()
  const first = keys()
  const second = keys()
  enrollHub(file, { hub_id: 'hub-1', public_key_pem: first.pem, project_dirs: [root] })
  expect(() => enrollHub(file, { hub_id: 'hub-1', public_key_pem: second.pem, project_dirs: [root] })).toThrow('different key')
  expect(enrollHub(file, { hub_id: 'hub-1', public_key_pem: second.pem, project_dirs: [root], replace: true }).fingerprint).toBe(publicKeyFingerprint(second.pem))
  const rsa = generateKeyPairSync('rsa', { modulusLength: 2048 }).publicKey.export({ type: 'spki', format: 'pem' }).toString()
  expect(() => enrollHub(file, { hub_id: 'rsa', public_key_pem: rsa, project_dirs: [root] })).toThrow('Ed25519')
  expect(() => enrollHub(file, { hub_id: 'rel', public_key_pem: first.pem, project_dirs: ['relative/path'] })).toThrow('absolute')
  expect(() => enrollHub(file, { hub_id: 'none', public_key_pem: first.pem })).toThrow('at least one')
  await writeFile(file, '{ not json')
  expect(() => enrollHub(file, { hub_id: 'hub-2', public_key_pem: first.pem, project_dirs: [root] })).toThrow('unreadable')
  expect(await readFile(file, 'utf8')).toBe('{ not json')
  // A corrupt enrollment file fails closed: no hub request is accepted.
  expect(verifier(file).verify(envelope(second.privateKey, 'turn.submit', { text: 'hi' }), request(root))).toMatchObject({ ok: false, reason: 'enrollment_unavailable' })
})

test('the verifier sees enrollment and revocation made through the CLI-usable functions', async () => {
  const { root, file } = await workspace()
  const { privateKey, pem } = keys()
  const v = verifier(file)
  expect(v.verify(envelope(privateKey, 'turn.submit', { text: 'hi' }), request(root))).toMatchObject({ ok: false, reason: 'unknown_hub' })
  enrollHub(file, { hub_id: 'hub-1', public_key_pem: pem, project_dirs: [root] })
  expect(v.verify(envelope(privateKey, 'turn.submit', { text: 'hi' }), request(root))).toMatchObject({ ok: true })
  expect(revokeHub(file, 'hub-1')).toBe(true)
  expect(v.verify(envelope(privateKey, 'turn.submit', { text: 'hi' }), request(root))).toMatchObject({ ok: false, reason: 'unknown_hub' })
})

test('enrolling the same key for another project adds to its scope; replace sets it exactly', async () => {
  const { root, file } = await workspace()
  const other = await mkdtemp(join(tmpdir(), 'xh-other-'))
  const { privateKey, pem } = keys()
  const v = verifier(file)
  enrollHub(file, { hub_id: 'hub-1', public_key_pem: pem, project_dirs: [root] })
  const merged = enrollHub(file, { hub_id: 'hub-1', public_key_pem: pem, project_dirs: [other] })
  expect(merged.project_dirs).toHaveLength(2)
  expect(v.verify(envelope(privateKey, 'turn.submit', { text: 'hi' }), request(root))).toMatchObject({ ok: true })
  expect(v.verify(envelope(privateKey, 'turn.submit', { text: 'hi' }), request(other))).toMatchObject({ ok: true })
  // Enrolling an already-pinned project again does not duplicate it.
  expect(enrollHub(file, { hub_id: 'hub-1', public_key_pem: pem, project_dirs: [root] }).project_dirs).toHaveLength(2)
  const replaced = enrollHub(file, { hub_id: 'hub-1', public_key_pem: pem, project_dirs: [other], replace: true })
  expect(replaced.project_dirs).toHaveLength(1)
  expect(v.verify(envelope(privateKey, 'turn.submit', { text: 'hi' }), request(root))).toMatchObject({ ok: false, reason: 'hub_not_enrolled_for_project' })
  await rm(other, { recursive: true, force: true })
})

function context(over: Partial<VerifiedHubContext>): VerifiedHubContext {
  return { hub_id: 'hub-1', workspace_id: 'ws-1', principal_kind: 'user', principal_id: 'alice', display_name: 'Alice', scopes: [], request_id: 'r', ...over }
}

test('method scopes are enforced and an agent principal can never use a human-only or unlisted route', async () => {
  const user = context({ scopes: ['session.submit', 'session.steer', 'session.cancel', 'session.configure', 'goal.write', 'permission.respond', 'question.respond', 'session.read'] })
  expect(hubAuthorizationFailure(user, 'turn.submit', {})).toBeUndefined()
  expect(hubAuthorizationFailure(user, 'turn.steer', {})).toBeUndefined()
  expect(hubAuthorizationFailure(user, 'subagent.steer', {})).toBeUndefined()
  expect(hubAuthorizationFailure(user, 'turn.cancel', {})).toBeUndefined()
  expect(hubAuthorizationFailure(user, 'permission_response', {})).toBeUndefined()
  expect(hubAuthorizationFailure(user, 'question_response', {})).toBeUndefined()
  expect(hubAuthorizationFailure(user, 'set_model', {})).toBeUndefined()
  expect(hubAuthorizationFailure(user, 'session.goal', { input: 'ship it' })).toBeUndefined()
  expect(hubMethodPolicy('session.goal', { input: 'ship it' })).toEqual({ kind: 'scope', scope: 'goal.write' })
  expect(hubMethodPolicy('session.goal', { input: '' })).toEqual({ kind: 'scope', scope: 'session.read' })
  const viewer = context({ scopes: ['session.read'] })
  expect(hubAuthorizationFailure(viewer, 'turn.submit', {})).toContain('session.submit')
  expect(hubAuthorizationFailure(viewer, 'permission_response', {})).toContain('permission.respond')
  // Routes outside the allowlist are refused for everyone: slash/shell, permission mode, enrollment, goal decisions.
  for (const method of ['slash', 'set_mode', 'set_plan_mode', 'goal.decision', 'hub.enroll', 'plugins.secret.set', 'provider_save', 'shutdown']) {
    expect(hubAuthorizationFailure(user, method, {})).toContain('not accepted')
  }
  // Agent: even a forged claim of human-only scopes is refused structurally.
  const agent = context({ principal_kind: 'agent', scopes: ['session.submit', 'permission.respond', 'question.respond', 'approvals.manage'] })
  for (const method of ['permission_response', 'question_response', 'learning.review']) {
    expect(hubAuthorizationFailure(agent, method, {})).toContain('human-only')
  }
  expect(hubAuthorizationFailure(agent, 'slash', { command: '!rm -rf /' })).toContain('not accepted')
  expect(hubAuthorizationFailure(agent, 'turn.submit', {})).toBeUndefined()
})

test('the hub model methods need supervisor.model, which an agent may hold and is clamped like any other scope', async () => {
  for (const method of ['hub.models.catalog', 'hub.models.options', 'hub.model.complete']) {
    expect(hubMethodPolicy(method, {})).toEqual({ kind: 'scope', scope: 'supervisor.model' })
    expect(hubAuthorizationFailure(context({ principal_kind: 'agent', scopes: ['supervisor.model'] }), method, {})).toBeUndefined()
    expect(hubAuthorizationFailure(context({ principal_kind: 'agent', scopes: ['session.read', 'session.submit'] }), method, {})).toContain('supervisor.model')
  }
  const { root, file } = await workspace()
  const { privateKey, pem } = keys()
  enrollHub(file, { hub_id: 'hub-1', public_key_pem: pem, project_dirs: ['/'] })
  const result = verifier(file).verify(envelope(privateKey, 'hub.models.catalog', {}, {
    principal_kind: 'agent', principal_id: 'xerxes', display_name: 'XERXES', scopes: ['supervisor.model', 'grants.manage'],
  }), { method: 'hub.models.catalog', params: {}, projectDir: root })
  expect(result).toMatchObject({ ok: true, context: { scopes: ['supervisor.model'] } })
})

test('a whole-machine pin (project_dirs ["/"]) verifies for unrelated project directories', async () => {
  const { root, file } = await workspace()
  const other = await mkdtemp(join(tmpdir(), 'xh-machine-'))
  directories.push(other)
  const { privateKey, pem } = keys()
  expect(enrollHub(file, { hub_id: 'hub-1', public_key_pem: pem, project_dirs: ['/'] }).project_dirs).toEqual(['/'])
  const v = verifier(file)
  expect(v.verify(envelope(privateKey, 'turn.submit', { text: 'hi' }), request(root))).toMatchObject({ ok: true })
  expect(v.verify(envelope(privateKey, 'turn.submit', { text: 'hi' }), request(other))).toMatchObject({ ok: true })
})

test('an agent envelope has its claimed scopes clamped to the agent ceiling', async () => {
  const { root, file } = await workspace()
  const { privateKey, pem } = keys()
  enrollHub(file, { hub_id: 'hub-1', public_key_pem: pem, project_dirs: [root] })
  const result = verifier(file).verify(envelope(privateKey, 'turn.submit', { text: 'hi' }, {
    principal_kind: 'agent', principal_id: 'xerxes', display_name: 'XERXES', requested_by: 'alice', confirmed_by: 'bob',
    scopes: ['session.submit', 'permission.respond', 'grants.manage'],
  }), request(root))
  expect(result).toMatchObject({ ok: true, context: { principal_kind: 'agent', requested_by: 'alice', confirmed_by: 'bob', scopes: ['session.submit'] } })
})
