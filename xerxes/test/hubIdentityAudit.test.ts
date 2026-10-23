// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'

import { AUDIT_GENESIS_HASH, looksLikeSecret } from '../src/hub/identity/audit.js'
import type { HubAuditInput } from '../src/hub/identityPort.js'
import { cleanupAll, makeHarness, pairDevice, raw } from './hubIdentityHarness.js'

afterEach(cleanupAll)

const HASH = 'ab'.repeat(32)
const actor = { principal_kind: 'user', principal_id: 'u1', display_name: 'Una' } as const

function record(overrides: Partial<HubAuditInput> = {}): HubAuditInput {
  return { actor, action: 'command.turn.submit', workspaceId: 'w1', sessionId: 's1', requestHash: HASH, decision: 'allowed', outcome: 'accepted', ...overrides }
}

describe('audit chain', () => {
  test('rows carry the documented fields and chain to each other', async () => {
    const h = await makeHarness()
    await h.identity.audit(record())
    await h.identity.audit(record({ action: 'command.turn.cancel', decision: 'denied', outcome: 'rejected', interactionId: 'i1', submissionId: 'sub1' }))
    const events = h.admin.listAudit()
    const tail = events.slice(-2)
    const [first, second] = tail
    expect(first?.schema_version).toBe(1)
    expect(second?.sequence).toBe((first?.sequence ?? 0) + 1)
    expect(second?.previous_hash).toBe(first?.event_hash)
    expect(second?.interaction_id).toBe('i1')
    expect(second?.submission_id).toBe('sub1')
    expect(first?.workspace_id).toBe('w1')
    expect(first?.event_hash).toMatch(/^[0-9a-f]{64}$/)
    expect(Date.parse(first?.occurred_at ?? '')).toBe(h.clock.now())
    expect(h.admin.listAudit()[0]?.previous_hash).toBe(AUDIT_GENESIS_HASH)
    const verified = h.admin.verifyAuditChain()
    expect(verified.ok).toBe(true)
    expect(verified.ok && verified.count).toBe(events.length)
  })

  test('an empty chain verifies', async () => {
    const h = await makeHarness({ owner: false })
    expect(h.admin.verifyAuditChain()).toEqual({ ok: true, count: 0, headHash: AUDIT_GENESIS_HASH })
  })

  test('identity actions are audited and the chain stays valid', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    h.admin.revokeDevice(s.deviceId)
    const actions = h.admin.listAudit().map(event => event.action)
    expect(actions.some(action => action.startsWith('identity.owner.bootstrap:'))).toBe(true)
    expect(actions.some(action => action.startsWith('identity.pairing.create:'))).toBe(true)
    expect(actions.some(action => action.startsWith('identity.pairing.consume:'))).toBe(true)
    expect(actions).toContain(`identity.device.revoke:${s.deviceId}`)
    expect(h.admin.verifyAuditChain().ok).toBe(true)
  })

  test('the table refuses UPDATE and DELETE', async () => {
    const h = await makeHarness()
    await h.identity.audit(record())
    const db = raw(h)
    expect(() => db.run("UPDATE hub_audit_events SET action = 'x' WHERE sequence = 1")).toThrow()
    expect(() => db.run('DELETE FROM hub_audit_events WHERE sequence = 1')).toThrow()
    expect(h.admin.verifyAuditChain().ok).toBe(true)
  })

  test('editing a row is detected', async () => {
    const h = await makeHarness()
    await h.identity.audit(record())
    await h.identity.audit(record({ action: 'command.turn.cancel' }))
    await h.identity.audit(record({ action: 'command.goal.set' }))
    const db = raw(h)
    db.run('DROP TRIGGER hub_audit_no_update')
    const target = h.admin.listAudit().find(event => event.action === 'command.turn.cancel')
    db.run("UPDATE hub_audit_events SET decision = 'denied' WHERE sequence = ?", [target?.sequence ?? 0])
    const result = h.admin.verifyAuditChain()
    expect(result).toEqual({ ok: false, sequence: target?.sequence ?? -1, reason: 'event_hash_mismatch' })
  })

  test('re-hashing an edited row still breaks the next link', async () => {
    const h = await makeHarness()
    await h.identity.audit(record())
    await h.identity.audit(record({ action: 'command.turn.cancel' }))
    const db = raw(h)
    db.run('DROP TRIGGER hub_audit_no_update')
    const events = h.admin.listAudit()
    const victim = events[events.length - 2]
    db.run("UPDATE hub_audit_events SET event_hash = ? WHERE sequence = ?", ['f'.repeat(64), victim?.sequence ?? 0])
    const result = h.admin.verifyAuditChain()
    expect(result.ok).toBe(false)
  })

  test('deleting or reordering a middle row is detected', async () => {
    const h = await makeHarness()
    for (let i = 0; i < 3; i += 1) await h.identity.audit(record({ action: `command.step.${i}` }))
    const db = raw(h)
    db.run('DROP TRIGGER hub_audit_no_delete')
    const middle = h.admin.listAudit().find(event => event.action === 'command.step.1')
    db.run('DELETE FROM hub_audit_events WHERE sequence = ?', [middle?.sequence ?? 0])
    const result = h.admin.verifyAuditChain()
    expect(result.ok).toBe(false)
    expect(!result.ok && result.reason).toBe('sequence_gap')
  })

  test('a write failure rejects instead of being swallowed', async () => {
    const h = await makeHarness()
    h.identity.close()
    await expect(h.identity.audit(record())).rejects.toThrow()
  })

  test('invalid records are rejected', async () => {
    const h = await makeHarness()
    await expect(h.identity.audit(record({ requestHash: 'not-a-hash' }))).rejects.toThrow('sha256')
    await expect(h.identity.audit(record({ action: '' }))).rejects.toThrow()
    await expect(h.identity.audit(record({ decision: 'maybe' as 'allowed' }))).rejects.toThrow()
    await expect(h.identity.audit(record({ outcome: 'weird' as 'accepted' }))).rejects.toThrow()
    expect(h.admin.verifyAuditChain().ok).toBe(true)
  })
})

describe('audit redaction', () => {
  const secrets = [
    'xr1.AbCdEfGhIjKlMnOpQrStUvWxYz0123456789',
    'xs1.abcdefghijklmnop',
    'Bearer abcdefghijklmnop',
    'token=abcdef123456',
    'code: 123456789',
    'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkw',
    '-----BEGIN PRIVATE KEY-----',
    'https://hub.test/pair#eyJodWJfaWQiOiJhYmMiLCJwYWlyaW5nX2lkIjoicHIxIn0',
    'aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bD1fH3',
  ]

  test.each(secrets)('rejects %s in the action', async secret => {
    const h = await makeHarness()
    await expect(h.identity.audit(record({ action: `identity.thing:${secret}` }))).rejects.toThrow('secret')
  })

  test('rejects secrets hidden in the actor and in ids', async () => {
    const h = await makeHarness()
    await expect(h.identity.audit(record({ actor: { ...actor, display_name: 'xr1.AbCdEfGhIjKlMnOp' } }))).rejects.toThrow('secret')
    await expect(h.identity.audit(record({ sessionId: 'Bearer abcdefghijklmnop' }))).rejects.toThrow('secret')
    expect(h.admin.verifyAuditChain().ok).toBe(true)
  })

  test('ordinary ids, hashes, uuids and slugs are not mistaken for secrets', () => {
    for (const fine of ['identity.pairing.create:pr_Ab3dEf9h', 'dev_Qw3Rty7uI', 'a'.repeat(64), 'f47ac10b-58cc-4372-a567-0e02b2c3d479', 'workspace-with-a-long-hyphenated-name-for-the-hub', 'Erfan']) {
      expect(looksLikeSecret(fine)).toBe(false)
    }
  })

  test('the stored request hash is a digest, never raw arguments', async () => {
    const h = await makeHarness()
    const s = await pairDevice(h)
    const dump = JSON.stringify(h.admin.listAudit())
    expect(dump).not.toContain(s.code)
    expect(dump).not.toContain(s.refresh)
    expect(dump).not.toContain(s.session)
    expect(h.admin.listAudit().every(event => /^[0-9a-f]{64}$/.test(event.request_hash))).toBe(true)
  })
})
