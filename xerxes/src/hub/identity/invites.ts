// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isHubRole, type HubRole } from '../../security/principal.js'
import type { HubActor } from '../protocol.js'
import { LOCAL_ADMIN_ACTOR, auditAction } from './audit.js'
import { HubIdentityError, actorFor, cleanText, hashSecret, newId, randomToken, type IdentityContext, type IssuedTokens } from './core.js'
import { insertGrant } from './grants.js'
import { buildTicketUrl, claimTicket, comparisonCodeFor, fingerprintId, provisionDevice } from './pairing.js'
import { tryRenderQr } from './qr.js'
import type { DeviceRow, InviteRow, UserRow } from './store.js'
import { startFamily } from './tokens.js'

/**
 * Invitations. The owner creates an expiring, single-use invite that names exact
 * resources (one workspace, or one session) and a maximum role. Accepting it
 * creates a NEW user plus a device and writes grants limited to those resources,
 * so a session-only invite never reveals the workspace inventory. The `owner`
 * role can never be invited. Confirmation works exactly like pairing.
 */

export type InviteResource =
  | { readonly type: 'workspace'; readonly workspaceId: string }
  | { readonly type: 'session'; readonly workspaceId: string; readonly sessionId: string }

export interface CreateInviteInput {
  /** Must be the hub owner. */
  readonly createdBy: string
  readonly workspaceId: string
  readonly sessionId?: string
  readonly role: HubRole
  readonly expiresInMs?: number
  readonly label?: string
  /** Default true; pass `false` explicitly to activate the invited device without owner confirmation. */
  readonly confirm?: boolean
}

export interface InviteCreated {
  readonly inviteId: string
  readonly url: string
  readonly comparisonCode: string
  readonly expiresAt: number
  readonly resources: readonly InviteResource[]
  readonly role: HubRole
  readonly confirmRequired: boolean
  readonly qrSvg: string | null
  readonly qrTerminal: string | null
}

const DEFAULT_INVITE_TTL_MS = 24 * 60 * 60 * 1000

function readId(value: string | undefined, field: string): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > 200) throw new HubIdentityError('invalid_request', `${field} must be a non-empty string of at most 200 characters`)
  return value
}

export function createInvite(ctx: IdentityContext, input: CreateInviteInput, actor: HubActor = LOCAL_ADMIN_ACTOR): InviteCreated {
  if (ctx.publicOrigin === undefined) {
    throw new HubIdentityError('invalid_request', 'the Hub public origin is not configured; set XERXES_HUB_PUBLIC_ORIGINS or pass --origin')
  }
  if (!isHubRole(input.role)) throw new HubIdentityError('invalid_request', 'unknown role')
  if (input.role === 'owner') throw new HubIdentityError('forbidden', 'the owner role cannot be granted by invite')
  const workspaceId = readId(input.workspaceId, 'workspaceId')
  const resources: InviteResource[] = [
    input.sessionId === undefined
      ? { type: 'workspace', workspaceId }
      : { type: 'session', workspaceId, sessionId: readId(input.sessionId, 'sessionId') },
  ]
  const ttl = input.expiresInMs ?? DEFAULT_INVITE_TTL_MS
  if (!Number.isFinite(ttl) || ttl < 1 || ttl > ctx.limits.inviteMaxTtlMs) {
    throw new HubIdentityError('invalid_request', `invite lifetime must be between 1 ms and ${ctx.limits.inviteMaxTtlMs} ms`)
  }
  const confirm = input.confirm !== false
  const code = randomToken(ctx, 16)
  const inviteId = newId(ctx, 'inv', 6)
  const comparisonCode = comparisonCodeFor(code, ctx.signerFingerprint)
  const now = ctx.clock.now()
  const expiresAt = now + ttl
  ctx.store.transaction(() => {
    const creator = ctx.store.one<UserRow>('SELECT * FROM hub_users WHERE id = ?', input.createdBy)
    if (!creator || creator.revoked_at !== null) throw new HubIdentityError('not_found', 'inviting user does not exist')
    if (creator.is_owner !== 1) throw new HubIdentityError('forbidden', 'only the hub owner can create invites')
    ctx.store.run(
      `INSERT INTO hub_invites (id, created_by, code_hash, resources, role, confirm_required, label, created_at, expires_at, status, failed_attempts, consumed_at, accepted_user_id, comparison_hash, confirm_attempts, device_id)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', 0, NULL, NULL, ?, 0, NULL)`,
      inviteId,
      creator.id,
      hashSecret(ctx, 'ticket', `${inviteId}:${code}`),
      JSON.stringify(resources),
      input.role,
      confirm ? 1 : 0,
      cleanText(input.label, 'Invited device', 80),
      now,
      expiresAt,
      hashSecret(ctx, 'compare', `${inviteId}:${comparisonCode}`),
    )
    auditAction(ctx, { actor, action: `identity.invite.create:${inviteId}`, params: { resources, role: input.role, confirm, expiresAt } })
  })
  const url = buildTicketUrl(ctx, '/invite', { hub_id: fingerprintId(ctx), invite_id: inviteId, code })
  return { inviteId, url, comparisonCode, expiresAt, resources, role: input.role, confirmRequired: confirm, ...tryRenderQr(url) }
}

export interface AcceptedInvite {
  readonly status: 'active' | 'pending_confirmation'
  readonly tokens: IssuedTokens
  readonly user: UserRow
  readonly device: DeviceRow
  readonly comparisonCode: string
}

export function acceptInvite(
  ctx: IdentityContext,
  input: { readonly inviteId: string; readonly code: string; readonly displayName?: string; readonly deviceLabel?: string; readonly remoteAddress?: string },
): AcceptedInvite {
  const now = ctx.clock.now()
  const address = input.remoteAddress ?? 'unknown'
  if (!ctx.rate.allow(`accept:${address}`, 10, 60_000, now) || !ctx.rate.allow(`accept-id:${input.inviteId}`, 10, 60_000, now)) {
    throw new HubIdentityError('rate_limited', 'too many invite attempts; try again shortly')
  }
  const result = claimTicket<InviteRow, AcceptedInvite>(ctx, 'hub_invites', input.inviteId, input.code, row => {
    const creator = ctx.store.one<UserRow>('SELECT * FROM hub_users WHERE id = ?', row.created_by)
    if (!creator || creator.revoked_at !== null) throw new HubIdentityError('conflict', 'the inviting owner is no longer active')
    const userId = newId(ctx, 'usr')
    const displayName = cleanText(input.displayName, 'Guest', 60)
    ctx.store.run("INSERT INTO hub_users (id, kind, display_name, is_owner, created_at) VALUES (?, 'user', ?, 0, ?)", userId, displayName, ctx.clock.now())
    const resources = JSON.parse(row.resources) as InviteResource[]
    for (const resource of resources) insertGrant(ctx, { principalId: userId, resource, effect: 'allow', role: row.role as HubRole }, row.created_by)
    const active = row.confirm_required === 0
    const device = provisionDevice(ctx, { userId, label: cleanText(input.deviceLabel, row.label, 80), active, scopes: null })
    const tokens = startFamily(ctx, device)
    ctx.store.run('UPDATE hub_invites SET device_id = ?, accepted_user_id = ? WHERE id = ?', device.id, userId, row.id)
    const user = ctx.store.one<UserRow>('SELECT * FROM hub_users WHERE id = ?', userId)
    if (!user) throw new HubIdentityError('internal', 'invited user was not created')
    auditAction(ctx, { actor: actorFor(user, device.id), action: `identity.invite.accept:${row.id}`, params: { userId, resources, role: row.role, active } })
    return { status: active ? 'active' : 'pending_confirmation', tokens, user, device, comparisonCode: comparisonCodeFor(input.code, ctx.signerFingerprint) } satisfies AcceptedInvite
  })
  if (result.kind === 'invalid') throw new HubIdentityError('unauthorized', 'invite code is invalid, expired or already used')
  return result.value
}

export function cancelInvite(ctx: IdentityContext, id: string, actor: HubActor = LOCAL_ADMIN_ACTOR): void {
  ctx.store.transaction(() => {
    if (ctx.store.run("UPDATE hub_invites SET status = 'cancelled' WHERE id = ? AND status = 'open'", id) !== 1) {
      throw new HubIdentityError('conflict', 'only an open invite can be cancelled')
    }
    auditAction(ctx, { actor, action: `identity.invite.cancel:${id}` })
  })
}
