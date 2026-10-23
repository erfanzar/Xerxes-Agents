// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'

import { ROLE_SCOPES, isHubRole, type HubRole } from '../../security/principal.js'
import type { HubActor } from '../protocol.js'
import { LOCAL_ADMIN_ACTOR, auditAction } from './audit.js'
import {
  HubIdentityError,
  actorFor,
  base64url,
  cleanText,
  constantTimeEqual,
  hashSecret,
  newId,
  randomToken,
  type IdentityContext,
  type IssuedTokens,
} from './core.js'
import { fireRevocations, revokeDeviceRows, type RevocationEvent } from './devices.js'
import { tryRenderQr } from './qr.js'
import type { DeviceRow, PairingRow, UserRow } from './store.js'
import { startFamily } from './tokens.js'

/**
 * Pairing and (shared with invites) single-use "tickets".
 *
 * A pairing is a 2-minute, single-use ticket created locally by the owner. The
 * QR/URL carries the secret code in the URL *fragment* (never sent to a server
 * until the PWA POSTs it in a JSON body):
 *
 *   `<publicOrigin>/pair#<base64url JSON {hub_id, pairing_id, code}>`
 *
 * `hub_id` is the base64url of the 32-byte SHA-256 fingerprint of the Hub
 * signing key (hex would not fit a version-10 QR).
 *
 * Consuming a ticket is one transaction: the first caller flips the row from
 * `open` to `consumed`, every competing caller finds it consumed. Five wrong
 * codes lock the ticket.
 *
 * Confirmation: by default a consumed pairing creates a *pending* device that
 * cannot authenticate until the owner runs `confirmPairing(id, comparisonCode)`.
 * The comparison code (6 digits from code + signer fingerprint) is shown on the
 * phone and on the desktop; it proves the phone holds this exact ticket. It does
 * NOT stop someone who photographed the QR before the owner did, so the pending
 * device's label and time are shown to the owner too. Only a pairing created
 * with `confirm: false` (CLI `--no-confirm`) activates the device immediately.
 * A pending device not confirmed within `confirmTtlMs` is revoked.
 */

export const MAX_TICKET_FAILURES = 5

export function comparisonCodeFor(code: string, fingerprint: string): string {
  const digest = createHash('sha256').update('xerxes-hub-pairing-v1\0').update(code).update('\0').update(fingerprint).digest()
  return String(digest.readUInt32BE(0) % 1_000_000).padStart(6, '0')
}

export interface PairingCreated {
  readonly pairingId: string
  readonly url: string
  readonly comparisonCode: string
  readonly expiresAt: number
  readonly confirmRequired: boolean
  readonly fingerprint: string
  /** Null only if the URL is too long for a version-10 QR; the URL still works. */
  readonly qrSvg: string | null
  readonly qrTerminal: string | null
}

export interface CreatePairingInput {
  readonly userId: string
  readonly label?: string
  /** Caps the new device at this role's scopes (a viewer phone for an owner, for example). */
  readonly role?: HubRole
  /** Default true. Pass `false` explicitly to skip owner confirmation. */
  readonly confirm?: boolean
  readonly ttlMs?: number
}

export function buildTicketUrl(ctx: Pick<IdentityContext, 'publicOrigin' | 'signerFingerprint'>, path: '/pair' | '/invite', payload: Readonly<Record<string, string>>): string {
  if (ctx.publicOrigin === undefined) {
    throw new HubIdentityError('invalid_request', 'the Hub public origin is not configured; set XERXES_HUB_PUBLIC_ORIGINS or pass --origin')
  }
  return `${ctx.publicOrigin}${path}#${base64url(new TextEncoder().encode(JSON.stringify(payload)))}`
}

export function fingerprintId(ctx: Pick<IdentityContext, 'signerFingerprint'>): string {
  return base64url(Buffer.from(ctx.signerFingerprint, 'hex'))
}

export function createPairing(ctx: IdentityContext, input: CreatePairingInput, actor: HubActor = LOCAL_ADMIN_ACTOR): PairingCreated {
  if (ctx.publicOrigin === undefined) {
    throw new HubIdentityError('invalid_request', 'the Hub public origin is not configured; set XERXES_HUB_PUBLIC_ORIGINS or pass --origin')
  }
  const ttl = input.ttlMs ?? ctx.limits.pairingTtlMs
  if (!Number.isFinite(ttl) || ttl < 1 || ttl > ctx.limits.pairingTtlMs) {
    throw new HubIdentityError('invalid_request', `pairing lifetime must be between 1 ms and ${ctx.limits.pairingTtlMs} ms`)
  }
  if (input.role !== undefined && !isHubRole(input.role)) throw new HubIdentityError('invalid_request', 'unknown role')
  const label = cleanText(input.label, 'New device', 80)
  const confirm = input.confirm !== false
  const code = randomToken(ctx, 16)
  const pairingId = newId(ctx, 'pr', 6)
  const comparisonCode = comparisonCodeFor(code, ctx.signerFingerprint)
  const now = ctx.clock.now()
  const expiresAt = now + ttl
  ctx.store.transaction(() => {
    const user = ctx.store.one<UserRow>('SELECT * FROM hub_users WHERE id = ?', input.userId)
    if (!user) throw new HubIdentityError('not_found', 'user does not exist')
    if (user.kind !== 'user') throw new HubIdentityError('invalid_request', 'agent principals have no devices')
    if (user.revoked_at !== null) throw new HubIdentityError('conflict', 'user is revoked')
    ctx.store.run(
      `INSERT INTO hub_pairings (id, kind, user_id, label, role, code_hash, comparison_hash, confirm_required, status, failed_attempts, confirm_attempts, created_at, expires_at, consumed_at, device_id)
       VALUES (?, 'pair', ?, ?, ?, ?, ?, ?, 'open', 0, 0, ?, ?, NULL, NULL)`,
      pairingId,
      user.id,
      label,
      input.role ?? null,
      hashSecret(ctx, 'ticket', `${pairingId}:${code}`),
      hashSecret(ctx, 'compare', `${pairingId}:${comparisonCode}`),
      confirm ? 1 : 0,
      now,
      expiresAt,
    )
    auditAction(ctx, { actor, action: `identity.pairing.create:${pairingId}`, params: { userId: user.id, role: input.role ?? null, confirm } })
  })
  const url = buildTicketUrl(ctx, '/pair', { hub_id: fingerprintId(ctx), pairing_id: pairingId, code })
  const qr = tryRenderQr(url)
  return { pairingId, url, comparisonCode, expiresAt, confirmRequired: confirm, fingerprint: ctx.signerFingerprint, ...qr }
}

/** Columns shared by `hub_pairings` and `hub_invites` that the ticket machinery touches. */
export interface TicketRow {
  id: string
  code_hash: string
  comparison_hash: string
  confirm_required: number
  status: 'open' | 'consumed' | 'cancelled' | 'locked'
  failed_attempts: number
  confirm_attempts: number
  expires_at: number
  consumed_at: number | null
  device_id: string | null
}

export type TicketTable = 'hub_pairings' | 'hub_invites'

/**
 * Atomically consume a ticket. `claim` runs inside the same transaction, so a
 * failure there un-consumes the ticket. Wrong codes and expired/used tickets are
 * indistinguishable to the caller.
 */
export function claimTicket<R extends TicketRow, T>(
  ctx: IdentityContext,
  table: TicketTable,
  id: string,
  code: string,
  claim: (row: R) => T,
): { readonly kind: 'ok'; readonly value: T } | { readonly kind: 'invalid' } {
  return ctx.store.transaction(() => {
    const row = ctx.store.one<R>(`SELECT * FROM ${table} WHERE id = ?`, id)
    if (!row || row.status !== 'open') return { kind: 'invalid' as const }
    const now = ctx.clock.now()
    if (row.expires_at <= now) {
      ctx.store.run(`UPDATE ${table} SET status = 'cancelled' WHERE id = ? AND status = 'open'`, id)
      return { kind: 'invalid' as const }
    }
    if (!constantTimeEqual(hashSecret(ctx, 'ticket', `${id}:${code}`), row.code_hash)) {
      const failures = row.failed_attempts + 1
      ctx.store.run(`UPDATE ${table} SET failed_attempts = ?, status = ? WHERE id = ? AND status = 'open'`, failures, failures >= MAX_TICKET_FAILURES ? 'locked' : 'open', id)
      if (failures >= MAX_TICKET_FAILURES) {
        auditAction(ctx, { actor: { principal_kind: 'user', principal_id: 'anonymous', display_name: 'Unauthenticated caller' }, action: `identity.ticket.locked:${id}`, decision: 'denied', outcome: 'rejected' })
      }
      return { kind: 'invalid' as const }
    }
    if (ctx.store.run(`UPDATE ${table} SET status = 'consumed', consumed_at = ? WHERE id = ? AND status = 'open'`, now, id) !== 1) {
      return { kind: 'invalid' as const }
    }
    return { kind: 'ok' as const, value: claim(row) }
  })
}

export function provisionDevice(
  ctx: IdentityContext,
  args: { readonly userId: string; readonly label: string; readonly active: boolean; readonly scopes: readonly string[] | null },
): DeviceRow {
  const now = ctx.clock.now()
  const id = newId(ctx, 'dev')
  ctx.store.run(
    `INSERT INTO hub_devices (id, user_id, label, status, scope_ceiling, created_at, activated_at, last_seen_at, absolute_expires_at, revoked_at, revoked_reason)
     VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL)`,
    id,
    args.userId,
    args.label,
    args.active ? 'active' : 'pending',
    args.scopes === null ? null : JSON.stringify(args.scopes),
    now,
    args.active ? now : null,
    now + ctx.limits.absoluteMs,
  )
  const device = ctx.store.one<DeviceRow>('SELECT * FROM hub_devices WHERE id = ?', id)
  if (!device) throw new HubIdentityError('internal', 'device row was not created')
  return device
}

export interface ConsumedPairing {
  readonly status: 'active' | 'pending_confirmation'
  readonly tokens: IssuedTokens
  readonly user: UserRow
  readonly device: DeviceRow
  readonly comparisonCode: string
}

export function consumePairing(
  ctx: IdentityContext,
  input: { readonly pairingId: string; readonly code: string; readonly deviceLabel?: string; readonly remoteAddress?: string },
): ConsumedPairing {
  const address = input.remoteAddress ?? 'unknown'
  const now = ctx.clock.now()
  if (!ctx.rate.allow(`consume:${address}`, 10, 60_000, now) || !ctx.rate.allow(`consume-id:${input.pairingId}`, 10, 60_000, now)) {
    throw new HubIdentityError('rate_limited', 'too many pairing attempts; try again shortly')
  }
  const result = claimTicket<PairingRow, ConsumedPairing>(ctx, 'hub_pairings', input.pairingId, input.code, row => {
    const user = row.user_id === null ? undefined : ctx.store.one<UserRow>('SELECT * FROM hub_users WHERE id = ?', row.user_id)
    if (!user || user.revoked_at !== null || user.kind !== 'user') throw new HubIdentityError('conflict', 'the pairing user is no longer active')
    const active = row.confirm_required === 0
    const role = row.role
    const scopes = role !== null && isHubRole(role) ? [...ROLE_SCOPES[role]] : null
    const device = provisionDevice(ctx, { userId: user.id, label: cleanText(input.deviceLabel, row.label, 80), active, scopes })
    const tokens = startFamily(ctx, device)
    ctx.store.run('UPDATE hub_pairings SET device_id = ? WHERE id = ?', device.id, row.id)
    auditAction(ctx, { actor: actorFor(user, device.id), action: `identity.pairing.consume:${row.id}`, params: { deviceId: device.id, active } })
    return { status: active ? 'active' : 'pending_confirmation', tokens, user, device, comparisonCode: comparisonCodeFor(input.code, ctx.signerFingerprint) } satisfies ConsumedPairing
  })
  if (result.kind === 'invalid') throw new HubIdentityError('unauthorized', 'pairing code is invalid, expired or already used')
  return result.value
}

export interface ConfirmResult {
  readonly deviceId: string
  readonly userId: string
}

/** Owner-side confirmation of a consumed pairing or invite (`pr_…` / `inv_…`). */
export function confirmPairing(ctx: IdentityContext, id: string, comparisonCode: string, actor: HubActor = LOCAL_ADMIN_ACTOR): ConfirmResult {
  const table: TicketTable | undefined = id.startsWith('inv_') ? 'hub_invites' : id.startsWith('pr_') ? 'hub_pairings' : undefined
  if (table === undefined) throw new HubIdentityError('not_found', 'unknown pairing id')
  const code = comparisonCode.replace(/\s+/g, '')
  if (!/^\d{6}$/.test(code)) throw new HubIdentityError('invalid_request', 'the comparison code is 6 digits')
  type Outcome =
    | { readonly kind: 'ok'; readonly device: DeviceRow }
    | { readonly kind: 'wrong'; readonly locked: boolean; readonly events: readonly RevocationEvent[] }
    | { readonly kind: 'expired'; readonly events: readonly RevocationEvent[] }
  const outcome = ctx.store.transaction<Outcome>(() => {
    const row = ctx.store.one<TicketRow>(`SELECT * FROM ${table} WHERE id = ?`, id)
    if (!row) throw new HubIdentityError('not_found', 'unknown pairing id')
    if (row.status === 'open') throw new HubIdentityError('conflict', 'nobody has used this pairing yet')
    if (row.status !== 'consumed' || row.device_id === null) throw new HubIdentityError('conflict', 'this pairing is not awaiting confirmation')
    const device = ctx.store.one<DeviceRow>('SELECT * FROM hub_devices WHERE id = ?', row.device_id)
    if (!device) throw new HubIdentityError('not_found', 'the paired device does not exist')
    if (device.status === 'active') throw new HubIdentityError('conflict', 'this device is already confirmed')
    if (device.status === 'revoked') throw new HubIdentityError('conflict', 'this device was revoked')
    const now = ctx.clock.now()
    if (device.created_at + ctx.limits.confirmTtlMs <= now) {
      return { kind: 'expired', events: revokeDeviceRows(ctx, device.id, 'unconfirmed', actor) }
    }
    if (!constantTimeEqual(hashSecret(ctx, 'compare', `${id}:${code}`), row.comparison_hash)) {
      const attempts = row.confirm_attempts + 1
      ctx.store.run(`UPDATE ${table} SET confirm_attempts = ? WHERE id = ?`, attempts, id)
      const locked = attempts >= MAX_TICKET_FAILURES
      return { kind: 'wrong', locked, events: locked ? revokeDeviceRows(ctx, device.id, 'confirmation_failed', actor) : [] }
    }
    if (ctx.store.run("UPDATE hub_devices SET status = 'active', activated_at = ? WHERE id = ? AND status = 'pending'", now, device.id) !== 1) {
      throw new HubIdentityError('conflict', 'this device can no longer be confirmed')
    }
    auditAction(ctx, { actor, action: `identity.pairing.confirm:${id}`, params: { deviceId: device.id } })
    return { kind: 'ok', device }
  })
  if (outcome.kind === 'ok') return { deviceId: outcome.device.id, userId: outcome.device.user_id }
  fireRevocations(ctx, outcome.events)
  if (outcome.kind === 'expired') throw new HubIdentityError('conflict', 'the confirmation window passed; the device was revoked')
  throw new HubIdentityError('forbidden', outcome.locked ? 'too many wrong comparison codes; the device was revoked' : 'the comparison code does not match')
}

export function cancelPairing(ctx: IdentityContext, id: string, actor: HubActor = LOCAL_ADMIN_ACTOR): void {
  ctx.store.transaction(() => {
    const changed = ctx.store.run("UPDATE hub_pairings SET status = 'cancelled' WHERE id = ? AND status = 'open'", id)
    if (changed !== 1) throw new HubIdentityError('conflict', 'only an open pairing can be cancelled')
    auditAction(ctx, { actor, action: `identity.pairing.cancel:${id}` })
  })
}
