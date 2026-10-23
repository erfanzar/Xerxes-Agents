// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'

import { canSeeWorkspace, effectiveScopes, grantedSessionIds, hasScope, type HubGrant } from '../src/security/principal.js'
import {
  DAY,
  authenticate,
  cleanupAll,
  errorCode,
  errorOf,
  http,
  inviteUser,
  makeHarness,
  must,
  pairDevice,
  raw,
  ticketOf,
} from './hubIdentityHarness.js'

afterEach(cleanupAll)

const workspaceGrant = (principalId: string, workspaceId: string, role: 'viewer' | 'steer' | 'operator' | 'owner' = 'viewer'): HubGrant => ({
  principalId,
  effect: 'allow',
  resource: { type: 'workspace', workspaceId },
  role,
})

describe('invites', () => {
  test('a session-only invite reveals that session and never the workspace inventory', async () => {
    const h = await makeHarness()
    const guest = await inviteUser(h, { workspaceId: 'w1', sessionId: 's1', role: 'viewer' })
    const ctx = must(await authenticate(h, guest))
    expect(ctx.principal.kind === 'user' && ctx.principal.hubOwner).toBe(false)
    expect(canSeeWorkspace(ctx.principal, ctx.grants, 'w1')).toBe(false)
    expect(grantedSessionIds(ctx.principal, ctx.grants, 'w1')).toEqual(['s1'])
    expect(hasScope(ctx.principal, ctx.grants, { workspaceId: 'w1', sessionId: 's1' }, 'session.read')).toBe(true)
    expect(hasScope(ctx.principal, ctx.grants, { workspaceId: 'w1', sessionId: 's2' }, 'session.read')).toBe(false)
    expect(hasScope(ctx.principal, ctx.grants, { workspaceId: 'w2', sessionId: 's1' }, 'session.read')).toBe(false)
    expect(hasScope(ctx.principal, ctx.grants, { workspaceId: 'w1', sessionId: 's1' }, 'session.submit')).toBe(false)
    expect(h.admin.listGrants(guest.userId)).toEqual([{ principalId: guest.userId, effect: 'allow', resource: { type: 'session', workspaceId: 'w1', sessionId: 's1' }, role: 'viewer' }])
  })

  test('a workspace invite is limited to that workspace and to its maximum role', async () => {
    const h = await makeHarness()
    const guest = await inviteUser(h, { workspaceId: 'w1', role: 'operator' })
    const ctx = must(await authenticate(h, guest))
    expect(canSeeWorkspace(ctx.principal, ctx.grants, 'w1')).toBe(true)
    expect(canSeeWorkspace(ctx.principal, ctx.grants, 'w2')).toBe(false)
    expect(hasScope(ctx.principal, ctx.grants, { workspaceId: 'w1' }, 'session.submit')).toBe(true)
    expect(hasScope(ctx.principal, ctx.grants, { workspaceId: 'w1' }, 'permission.respond')).toBe(true)
    expect(hasScope(ctx.principal, ctx.grants, { workspaceId: 'w1' }, 'grants.manage')).toBe(false)
    expect(hasScope(ctx.principal, ctx.grants, { workspaceId: 'w1' }, 'devices.manage')).toBe(false)
    expect(hasScope(ctx.principal, ctx.grants, { workspaceId: 'w2' }, 'session.read')).toBe(false)
  })

  test('accepting creates a new, non-owner user with its own device', async () => {
    const h = await makeHarness()
    const guest = await inviteUser(h, { workspaceId: 'w1', role: 'viewer', displayName: 'Mina' })
    expect(guest.userId).not.toBe(h.ownerId)
    const users = h.admin.listUsers()
    expect(users).toHaveLength(2)
    expect(users.find(user => user.id === guest.userId)).toMatchObject({ displayName: 'Mina', owner: false, revoked: false })
    expect(h.admin.listDevices({ userId: guest.userId })).toHaveLength(1)
  })

  test('the owner role can never be invited', async () => {
    const h = await makeHarness()
    expect(errorCode(() => h.admin.createInvite({ createdBy: h.ownerId, workspaceId: 'w1', role: 'owner' }))).toBe('forbidden')
  })

  test('only the hub owner creates invites', async () => {
    const h = await makeHarness()
    const guest = await inviteUser(h, { workspaceId: 'w1', role: 'operator' })
    expect(errorCode(() => h.admin.createInvite({ createdBy: guest.userId, workspaceId: 'w1', role: 'viewer' }))).toBe('forbidden')
    expect(errorCode(() => h.admin.createInvite({ createdBy: 'usr_missing', workspaceId: 'w1', role: 'viewer' }))).toBe('not_found')
    const over = await http(h, 'POST', 'invites', { workspace_id: 'w1', role: 'viewer' }, { cookie: guest.cookie, csrf: guest.csrf })
    expect(over.status).toBe(403)
    expect(errorOf(over).code).toBe('forbidden')
  })

  test('lifetime is bounded and an expired invite is refused', async () => {
    const h = await makeHarness()
    expect(errorCode(() => h.admin.createInvite({ createdBy: h.ownerId, workspaceId: 'w1', role: 'viewer', expiresInMs: 8 * DAY }))).toBe('invalid_request')
    const created = h.admin.createInvite({ createdBy: h.ownerId, workspaceId: 'w1', role: 'viewer', expiresInMs: 1000, confirm: false })
    const ticket = ticketOf(created.url)
    h.clock.advance(2000)
    expect((await http(h, 'POST', 'invites/accept', { invite_id: ticket.invite_id, code: ticket.code })).status).toBe(401)
    expect(h.admin.listUsers()).toHaveLength(1)
  })

  test('single use: a second accept fails and competing acceptors yield exactly one user', async () => {
    const h = await makeHarness()
    const created = h.admin.createInvite({ createdBy: h.ownerId, workspaceId: 'w1', role: 'viewer', confirm: false })
    const ticket = ticketOf(created.url)
    const body = { invite_id: ticket.invite_id, code: ticket.code }
    const results = await Promise.all(Array.from({ length: 5 }, (_, index) => http(h, 'POST', 'invites/accept', body, { ip: `10.1.0.${index + 1}` })))
    expect(results.filter(result => result.status === 200)).toHaveLength(1)
    expect(results.filter(result => result.status === 401)).toHaveLength(4)
    expect(h.admin.listUsers()).toHaveLength(2)
    expect((await http(h, 'POST', 'invites/accept', body)).status).toBe(401)
  })

  test('wrong codes lock the invite; a cancelled invite cannot be used', async () => {
    const h = await makeHarness()
    const locked = h.admin.createInvite({ createdBy: h.ownerId, workspaceId: 'w1', role: 'viewer', confirm: false })
    const lockedTicket = ticketOf(locked.url)
    for (let i = 0; i < 5; i += 1) {
      expect((await http(h, 'POST', 'invites/accept', { invite_id: lockedTicket.invite_id, code: `wrong-${i}` })).status).toBe(401)
    }
    expect((await http(h, 'POST', 'invites/accept', { invite_id: lockedTicket.invite_id, code: lockedTicket.code }, { ip: '10.2.0.1' })).status).toBe(401)
    const cancelled = h.admin.createInvite({ createdBy: h.ownerId, workspaceId: 'w1', role: 'viewer', confirm: false })
    const cancelledTicket = ticketOf(cancelled.url)
    h.admin.cancelInvite(cancelled.inviteId)
    expect((await http(h, 'POST', 'invites/accept', { invite_id: cancelledTicket.invite_id, code: cancelledTicket.code }, { ip: '10.2.0.2' })).status).toBe(401)
    expect(errorCode(() => h.admin.cancelInvite(cancelled.inviteId))).toBe('conflict')
    expect(h.admin.listUsers()).toHaveLength(1)
  })

  test('by default the invited device is pending until the owner confirms the comparison code', async () => {
    const h = await makeHarness()
    const created = h.admin.createInvite({ createdBy: h.ownerId, workspaceId: 'w1', role: 'viewer' })
    expect(created.confirmRequired).toBe(true)
    const ticket = ticketOf(created.url)
    const accepted = await http(h, 'POST', 'invites/accept', { invite_id: ticket.invite_id, code: ticket.code, display_name: 'Ravi' })
    expect(accepted.status).toBe(200)
    expect(accepted.json.status).toBe('pending_confirmation')
    const session = { session: accepted.cookies['__Host-xerxes_session'] ?? '' }
    expect(await authenticate(h, session)).toBeNull()
    h.admin.confirmPairing(created.inviteId, created.comparisonCode)
    expect(await authenticate(h, session)).not.toBeNull()
  })

  test('the invite URL carries the code only in the fragment and is never audited', async () => {
    const h = await makeHarness()
    const created = h.admin.createInvite({ createdBy: h.ownerId, workspaceId: 'w1', sessionId: 's1', role: 'steer' })
    const url = new URL(created.url)
    expect(url.pathname).toBe('/invite')
    expect(url.search).toBe('')
    const ticket = ticketOf(created.url)
    expect(JSON.stringify(h.admin.listAudit())).not.toContain(ticket.code ?? 'missing')
    expect(created.resources).toEqual([{ type: 'session', workspaceId: 'w1', sessionId: 's1' }])
    expect(h.admin.verifyAuditChain().ok).toBe(true)
  })
})

describe('grants', () => {
  function addAgent(h: Awaited<ReturnType<typeof makeHarness>>): string {
    raw(h).run('INSERT INTO hub_users (id, kind, display_name, is_owner, created_at) VALUES (?, ?, ?, ?, ?)', ['agt_xerxes', 'agent', 'XERXES', 0, h.clock.now()])
    return 'agt_xerxes'
  }

  test('replace and add; records are validated with parseGrant', async () => {
    const h = await makeHarness()
    const guest = await inviteUser(h, { workspaceId: 'w1', role: 'viewer' })
    h.admin.updateGrants({ principalId: guest.userId, grants: [workspaceGrant(guest.userId, 'w2', 'steer')] })
    expect(h.admin.listGrants(guest.userId).map(grant => grant.resource)).toEqual([{ type: 'workspace', workspaceId: 'w2' }])
    h.admin.updateGrants({ principalId: guest.userId, mode: 'add', grants: [workspaceGrant(guest.userId, 'w3')] })
    expect(h.admin.listGrants(guest.userId)).toHaveLength(2)
    expect(errorCode(() => h.admin.updateGrants({ principalId: guest.userId, grants: [{ principalId: guest.userId, effect: 'maybe', resource: { type: 'hub' }, role: 'viewer' }] }))).toBe('invalid_request')
    expect(errorCode(() => h.admin.updateGrants({ principalId: guest.userId, grants: [{ principalId: guest.userId, effect: 'allow', resource: { type: 'hub' }, scopes: ['root'] }] }))).toBe('invalid_request')
    expect(errorCode(() => h.admin.updateGrants({ principalId: guest.userId, grants: [workspaceGrant('someone-else', 'w1')] }))).toBe('invalid_request')
    expect(h.admin.listGrants(guest.userId)).toHaveLength(2)
    expect(errorCode(() => h.admin.updateGrants({ principalId: 'usr_missing', grants: [] }))).toBe('not_found')
    const many = Array.from({ length: 101 }, (_, index) => workspaceGrant(guest.userId, `w${index}`))
    expect(errorCode(() => h.admin.updateGrants({ principalId: guest.userId, grants: many }))).toBe('invalid_request')
  })

  test('grants change while a session is open and grantsFor reflects them', async () => {
    const h = await makeHarness()
    const guest = await inviteUser(h, { workspaceId: 'w1', role: 'viewer' })
    const ctx = must(await authenticate(h, guest))
    expect(canSeeWorkspace(ctx.principal, h.identity.grantsFor(guest.userId), 'w2')).toBe(false)
    h.admin.updateGrants({ principalId: guest.userId, mode: 'add', grants: [workspaceGrant(guest.userId, 'w2')] })
    expect(canSeeWorkspace(ctx.principal, h.identity.grantsFor(guest.userId), 'w2')).toBe(true)
    h.admin.updateGrants({ principalId: guest.userId, mode: 'add', grants: [{ principalId: guest.userId, effect: 'deny', resource: { type: 'workspace', workspaceId: 'w1' }, role: 'viewer' }] })
    expect(effectiveScopes(ctx.principal, h.identity.grantsFor(guest.userId), { workspaceId: 'w1' }).size).toBe(0)
  })

  test('agent principals can never be granted a human-only scope', async () => {
    const h = await makeHarness()
    const agent = addAgent(h)
    const attempt = (grant: Partial<HubGrant> & Pick<HubGrant, 'resource'>): string =>
      errorCode(() => h.admin.updateGrants({ principalId: agent, mode: 'add', grants: [{ principalId: agent, effect: 'allow', ...grant }] }))
    expect(attempt({ resource: { type: 'hub' }, role: 'owner' })).toBe('forbidden')
    expect(attempt({ resource: { type: 'hub' }, role: 'operator' })).toBe('forbidden')
    for (const scope of ['permission.respond', 'question.respond', 'grants.manage', 'approvals.manage', 'devices.manage'] as const) {
      expect(attempt({ resource: { type: 'workspace', workspaceId: 'w1' }, scopes: [scope] })).toBe('forbidden')
    }
    expect(h.admin.listGrants(agent)).toEqual([])
    // Within the agent ceiling and explicit denials are fine.
    h.admin.updateGrants({ principalId: agent, grants: [workspaceGrant(agent, 'w1', 'steer'), { principalId: agent, effect: 'deny', resource: { type: 'workspace', workspaceId: 'w2' }, scopes: ['permission.respond'] }] })
    expect(h.admin.listGrants(agent)).toHaveLength(2)
    // Even so, hasScope refuses human-only scopes for an agent structurally.
    const principal = { kind: 'agent', id: agent, displayName: 'XERXES' } as const
    expect(hasScope(principal, h.admin.listGrants(agent), { workspaceId: 'w1' }, 'permission.respond')).toBe(false)
    expect(hasScope(principal, h.admin.listGrants(agent), { workspaceId: 'w1' }, 'session.steer')).toBe(true)
  })

  test('agent principals have no devices and cannot be paired or authenticated', async () => {
    const h = await makeHarness()
    const agent = addAgent(h)
    expect(errorCode(() => h.admin.createPairing({ userId: agent }))).toBe('invalid_request')
  })

  test('a revoked principal cannot receive grants', async () => {
    const h = await makeHarness()
    const guest = await inviteUser(h, { workspaceId: 'w1', role: 'viewer' })
    h.admin.revokeUser(guest.userId)
    expect(errorCode(() => h.admin.updateGrants({ principalId: guest.userId, grants: [workspaceGrant(guest.userId, 'w1')] }))).toBe('conflict')
  })

  test('over HTTP, grant changes need the owner and a passkey step-up', async () => {
    const h = await makeHarness()
    const owner = await pairDevice(h)
    const guest = await inviteUser(h, { workspaceId: 'w1', role: 'operator' })
    const body = { principal_id: guest.userId, grants: [workspaceGrant(guest.userId, 'w9')] }
    const byGuest = await http(h, 'POST', 'grants', body, { cookie: guest.cookie, csrf: guest.csrf })
    expect(byGuest.status).toBe(403)
    expect(errorOf(byGuest).code).toBe('forbidden')
    const byOwner = await http(h, 'POST', 'grants', body, { cookie: owner.cookie, csrf: owner.csrf })
    expect(byOwner.status).toBe(403)
    expect(errorOf(byOwner).code).toBe('step_up_required')
    expect(h.admin.listGrants(guest.userId).some(grant => grant.resource.type === 'workspace' && grant.resource.workspaceId === 'w9')).toBe(false)
  })
})
