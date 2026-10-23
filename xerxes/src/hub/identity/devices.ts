// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { RevocationListener } from '../identityPort.js'
import type { HubActor } from '../protocol.js'
import { isHubScope, type HubScope } from '../../security/principal.js'
import { LOCAL_ADMIN_ACTOR, auditAction } from './audit.js'
import { HubIdentityError, type IdentityContext } from './core.js'
import type { DeviceRow, UserRow } from './store.js'

/**
 * Device listing and revocation. Revocation writes rows inside the caller's
 * transaction and returns events; the caller fires them with
 * `fireRevocations` only after COMMIT, so listeners never observe state that
 * could still roll back. Listeners run synchronously.
 *
 * Event shape: one device-level event (`deviceId`, no `browserSessionId`) plus
 * one event per live browser session, so a server can close sockets by either.
 */

export type RevocationEvent = Parameters<RevocationListener>[0]

function makeEvent(userId: string, deviceId: string | undefined, reason: string, browserSessionId?: string): RevocationEvent {
  return {
    principalId: userId,
    ...(deviceId === undefined ? {} : { deviceId }),
    ...(browserSessionId === undefined ? {} : { browserSessionId }),
    reason,
  }
}

export function revokeDeviceRows(ctx: IdentityContext, deviceId: string, reason: string, actor: HubActor = LOCAL_ADMIN_ACTOR): RevocationEvent[] {
  const device = ctx.store.one<DeviceRow>('SELECT * FROM hub_devices WHERE id = ?', deviceId)
  if (!device || device.status === 'revoked') return []
  const now = ctx.clock.now()
  const sessions = ctx.store.all<{ id: string }>('SELECT id FROM hub_sessions WHERE device_id = ? AND revoked_at IS NULL AND expires_at > ?', deviceId, now)
  ctx.store.run("UPDATE hub_devices SET status = 'revoked', revoked_at = ?, revoked_reason = ? WHERE id = ?", now, reason, deviceId)
  ctx.store.run('UPDATE hub_token_families SET revoked_at = ? WHERE device_id = ? AND revoked_at IS NULL', now, deviceId)
  ctx.store.run('UPDATE hub_sessions SET revoked_at = ? WHERE device_id = ? AND revoked_at IS NULL', now, deviceId)
  ctx.store.run('DELETE FROM hub_push_subscriptions WHERE device_id = ?', deviceId)
  auditAction(ctx, { actor, action: `identity.device.revoke:${deviceId}`, params: { reason } })
  return [makeEvent(device.user_id, deviceId, reason), ...sessions.map(session => makeEvent(device.user_id, deviceId, reason, session.id))]
}

export function revokeUserRows(ctx: IdentityContext, userId: string, reason: string, actor: HubActor = LOCAL_ADMIN_ACTOR): RevocationEvent[] {
  const user = ctx.store.one<UserRow>('SELECT * FROM hub_users WHERE id = ?', userId)
  if (!user) throw new HubIdentityError('not_found', 'user does not exist')
  if (user.is_owner === 1) throw new HubIdentityError('conflict', 'the hub owner cannot be revoked; revoke their devices instead')
  const now = ctx.clock.now()
  if (user.revoked_at === null) {
    ctx.store.run('UPDATE hub_users SET revoked_at = ?, revoked_reason = ? WHERE id = ?', now, reason, userId)
  }
  const events: RevocationEvent[] = []
  for (const device of ctx.store.all<{ id: string }>("SELECT id FROM hub_devices WHERE user_id = ? AND status != 'revoked'", userId)) {
    events.push(...revokeDeviceRows(ctx, device.id, reason, actor))
  }
  ctx.store.run('DELETE FROM hub_push_subscriptions WHERE user_id = ?', userId)
  if (events.length === 0) events.push(makeEvent(userId, undefined, reason))
  auditAction(ctx, { actor, action: `identity.user.revoke:${userId}`, params: { reason } })
  return events
}

export function revokeAllRows(ctx: IdentityContext, reason: string, actor: HubActor = LOCAL_ADMIN_ACTOR): RevocationEvent[] {
  const events: RevocationEvent[] = []
  for (const device of ctx.store.all<{ id: string }>("SELECT id FROM hub_devices WHERE status != 'revoked'")) {
    events.push(...revokeDeviceRows(ctx, device.id, reason, actor))
  }
  auditAction(ctx, { actor, action: 'identity.devices.revoke_all', params: { reason } })
  return events
}

/** Run after the transaction committed. A throwing listener never blocks the others. */
export function fireRevocations(ctx: IdentityContext, events: readonly RevocationEvent[]): void {
  for (const event of events) {
    if (event.deviceId !== undefined) ctx.revocationState.fired.add(event.deviceId)
    for (const listener of [...ctx.listeners]) {
      try {
        listener(event)
      } catch (error) {
        ctx.onListenerError(error)
      }
    }
  }
}

/**
 * Announce revocations committed by another process (the local CLI writes the
 * same database). SQLite's `data_version` changes only for other connections'
 * commits, so this is a cheap no-op when nothing external happened.
 */
export function syncRevocations(ctx: IdentityContext): void {
  const version = ctx.store.one<{ data_version: number }>('PRAGMA data_version')?.data_version
  if (version === ctx.revocationState.dataVersion) return
  ctx.revocationState.dataVersion = version
  const events: RevocationEvent[] = []
  const rows = ctx.store.all<{ id: string; user_id: string; revoked_reason: string | null }>(
    "SELECT id, user_id, revoked_reason FROM hub_devices WHERE status = 'revoked' AND revoked_at >= ?",
    ctx.revocationState.floor,
  )
  for (const row of rows) {
    if (ctx.revocationState.fired.has(row.id)) continue
    const reason = row.revoked_reason ?? 'revoked'
    events.push(makeEvent(row.user_id, row.id, reason))
    for (const session of ctx.store.all<{ id: string }>('SELECT id FROM hub_sessions WHERE device_id = ?', row.id)) {
      events.push(makeEvent(row.user_id, row.id, reason, session.id))
    }
  }
  fireRevocations(ctx, events)
}

export interface DeviceSummary {
  readonly id: string
  readonly userId: string
  readonly displayName: string
  readonly label: string
  readonly status: DeviceRow['status']
  readonly scopeCeiling: readonly string[] | null
  readonly createdAt: number
  readonly activatedAt: number | null
  readonly lastSeenAt: number | null
  readonly expiresAt: number
  readonly revokedAt: number | null
  readonly revokedReason: string | null
}

export function listDevices(ctx: Pick<IdentityContext, 'store'>, filter: { readonly userId?: string; readonly includeRevoked?: boolean } = {}): DeviceSummary[] {
  const rows = ctx.store.all<DeviceRow & { display_name: string }>(
    `SELECT d.*, u.display_name AS display_name FROM hub_devices d JOIN hub_users u ON u.id = d.user_id
     WHERE (?1 IS NULL OR d.user_id = ?1) AND (?2 = 1 OR d.status != 'revoked') ORDER BY d.created_at ASC, d.id ASC`,
    filter.userId ?? null,
    filter.includeRevoked === true ? 1 : 0,
  )
  return rows.map(row => ({
    id: row.id,
    userId: row.user_id,
    displayName: row.display_name,
    label: row.label,
    status: row.status,
    scopeCeiling: row.scope_ceiling === null ? null : parseCeilingList(row.scope_ceiling),
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    lastSeenAt: row.last_seen_at,
    expiresAt: row.absolute_expires_at,
    revokedAt: row.revoked_at,
    revokedReason: row.revoked_reason,
  }))
}

/** Revoke one device in a transaction, then notify listeners after COMMIT. */
export function revokeDevice(ctx: IdentityContext, deviceId: string, reason: string, actor: HubActor = LOCAL_ADMIN_ACTOR): void {
  const events = ctx.store.transaction(() => {
    if (!ctx.store.one('SELECT id FROM hub_devices WHERE id = ?', deviceId)) throw new HubIdentityError('not_found', 'device does not exist')
    return revokeDeviceRows(ctx, deviceId, reason, actor)
  })
  fireRevocations(ctx, events)
}

export function revokeUser(ctx: IdentityContext, userId: string, reason: string, actor: HubActor = LOCAL_ADMIN_ACTOR): void {
  fireRevocations(ctx, ctx.store.transaction(() => revokeUserRows(ctx, userId, reason, actor)))
}

export function revokeAll(ctx: IdentityContext, reason: string, actor: HubActor = LOCAL_ADMIN_ACTOR): void {
  fireRevocations(ctx, ctx.store.transaction(() => revokeAllRows(ctx, reason, actor)))
}

export function parseCeilingList(json: string): HubScope[] {
  const value: unknown = JSON.parse(json)
  return Array.isArray(value) ? value.filter(isHubScope) : []
}
