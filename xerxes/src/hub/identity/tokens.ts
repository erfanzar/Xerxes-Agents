// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubPrincipal, HubScope } from '../../security/principal.js'
import type { AuthenticatedContext, HubHttpRequestInfo } from '../identityPort.js'
import { parseCookieHeader, REFRESH_COOKIE, SESSION_COOKIE, serializeHostCookie } from './cookies.js'
import {
  actorFor,
  constantTimeEqual,
  hashSecret,
  isLoopbackOrigin,
  newId,
  normalizeOrigin,
  randomToken,
  type IdentityContext,
  type IssuedTokens,
} from './core.js'
import { auditAction } from './audit.js'
import { fireRevocations, parseCeilingList, revokeDeviceRows, revokeFamilyRows, syncRevocations, type RevocationEvent } from './devices.js'
import { listGrants } from './grants.js'
import type { DeviceRow, FamilyRow, SessionRow, UserRow } from './store.js'

/**
 * Device refresh tokens and short browser sessions.
 *
 * - One token family per device; a refresh token is `xr1.<random>`, a session
 *   token `xs1.<random>`. Only keyed hashes are stored.
 * - Refresh rotates atomically (one SQLite write transaction).
 * - A retired refresh token seen again within `refreshGraceMs` of its rotation
 *   returns the *same* successor that was already issued (two tabs or a
 *   retried request racing). The successor is held in memory only.
 * - A retired token seen later is a second cookie jar of the same device (iOS
 *   Safari vs. the home-screen app) or a refresh reply the phone lost while
 *   suspended. It starts a new token family once (`forkFamily`): the token is
 *   then spent, the fork is audited, and a device keeps at most
 *   MAX_FAMILIES_PER_DEVICE live families. It no longer revokes the device;
 *   that locked owners out of their phones several times a day.
 */

export const REFRESH_PREFIX = 'xr1.'
export const SESSION_PREFIX = 'xs1.'
const MAX_TOKEN_LENGTH = 200
const SAFE_METHODS: ReadonlySet<string> = new Set(['GET', 'HEAD', 'OPTIONS'])

export function originAllowed(ctx: Pick<IdentityContext, 'publicOrigin' | 'extraOrigins'>, origin: string): boolean {
  const normalized = normalizeOrigin(origin)
  if (normalized === undefined) return false
  if (isLoopbackOrigin(normalized)) return true
  return ctx.publicOrigin === normalized || ctx.extraOrigins.includes(normalized)
}

/** Stateless CSRF token bound to one browser session token. */
export function csrfFor(ctx: Pick<IdentityContext, 'pepper'>, sessionToken: string): string {
  return hashSecret(ctx, 'csrf', sessionToken)
}

export function mintSession(
  ctx: IdentityContext,
  familyId: string,
  device: Pick<DeviceRow, 'id' | 'user_id' | 'absolute_expires_at'>,
  refreshToken: string,
): IssuedTokens {
  const now = ctx.clock.now()
  const sessionToken = SESSION_PREFIX + randomToken(ctx, 32)
  const sessionId = newId(ctx, 'ses')
  const expiresAt = now + ctx.limits.sessionTtlMs
  ctx.store.run(
    'INSERT INTO hub_sessions (id, family_id, device_id, user_id, hash, created_at, expires_at, revoked_at, stepup_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)',
    sessionId,
    familyId,
    device.id,
    device.user_id,
    hashSecret(ctx, 'session', sessionToken),
    now,
    expiresAt,
  )
  const refreshMs = Math.max(0, Math.min(ctx.limits.inactivityMs, device.absolute_expires_at - now))
  return {
    refreshToken,
    sessionToken,
    sessionId,
    csrfToken: csrfFor(ctx, sessionToken),
    sessionExpiresAt: expiresAt,
    refreshMaxAgeSeconds: Math.floor(refreshMs / 1000),
  }
}

/** Create the device's token family (generation 1) and its first browser session. */
export function startFamily(ctx: IdentityContext, device: Pick<DeviceRow, 'id' | 'user_id' | 'absolute_expires_at'>): IssuedTokens {
  const now = ctx.clock.now()
  const familyId = newId(ctx, 'fam')
  ctx.store.run(
    'INSERT INTO hub_token_families (id, device_id, generation, rotated_at, last_used_at, created_at, revoked_at) VALUES (?, ?, 1, NULL, ?, ?, NULL)',
    familyId,
    device.id,
    now,
    now,
  )
  const refreshToken = REFRESH_PREFIX + randomToken(ctx, 32)
  ctx.store.run('INSERT INTO hub_refresh_tokens (hash, family_id, generation, issued_at) VALUES (?, ?, 1, ?)', hashSecret(ctx, 'refresh', refreshToken), familyId, now)
  return mintSession(ctx, familyId, device, refreshToken)
}

export function cookiesFor(ctx: IdentityContext, tokens: IssuedTokens): string[] {
  const sessionSeconds = Math.max(0, Math.floor((tokens.sessionExpiresAt - ctx.clock.now()) / 1000))
  return [
    serializeHostCookie(REFRESH_COOKIE, tokens.refreshToken, tokens.refreshMaxAgeSeconds),
    serializeHostCookie(SESSION_COOKIE, tokens.sessionToken, sessionSeconds),
  ]
}

export type RefreshOutcome =
  | { readonly kind: 'ok'; readonly tokens: IssuedTokens; readonly user: UserRow; readonly device: DeviceRow }
  | { readonly kind: 'invalid' }
  | { readonly kind: 'pending' }
  | { readonly kind: 'expired' }
  | { readonly kind: 'reuse' }

type InternalOutcome = { readonly outcome: RefreshOutcome; readonly events: readonly RevocationEvent[]; readonly remember?: { readonly hash: string; readonly tokens: IssuedTokens } }

export function refreshTokens(ctx: IdentityContext, refreshToken: string): RefreshOutcome {
  if (!refreshToken.startsWith(REFRESH_PREFIX) || refreshToken.length > MAX_TOKEN_LENGTH) return { kind: 'invalid' }
  const hash = hashSecret(ctx, 'refresh', refreshToken)
  const result = ctx.store.transaction<InternalOutcome>(() => {
    const token = ctx.store.one<{ hash: string; family_id: string; generation: number }>('SELECT hash, family_id, generation FROM hub_refresh_tokens WHERE hash = ?', hash)
    if (!token) return { outcome: { kind: 'invalid' }, events: [] }
    const family = ctx.store.one<FamilyRow>('SELECT * FROM hub_token_families WHERE id = ?', token.family_id)
    if (!family || family.revoked_at !== null) return { outcome: { kind: 'invalid' }, events: [] }
    const device = ctx.store.one<DeviceRow>('SELECT * FROM hub_devices WHERE id = ?', family.device_id)
    const user = device ? ctx.store.one<UserRow>('SELECT * FROM hub_users WHERE id = ?', device.user_id) : undefined
    if (!device || !user || device.status === 'revoked' || user.revoked_at !== null) return { outcome: { kind: 'invalid' }, events: [] }
    const now = ctx.clock.now()
    if (device.status === 'pending') {
      if (device.created_at + ctx.limits.confirmTtlMs <= now) {
        return { outcome: { kind: 'expired' }, events: revokeDeviceRows(ctx, device.id, 'unconfirmed') }
      }
      return { outcome: { kind: 'pending' }, events: [] }
    }
    if (device.absolute_expires_at <= now) {
      return { outcome: { kind: 'expired' }, events: revokeDeviceRows(ctx, device.id, 'expired') }
    }
    if (family.last_used_at + ctx.limits.inactivityMs <= now) {
      // An idle cookie jar ends alone while another jar of the same device is still in use.
      const otherActive = ctx.store.one<{ id: string }>(
        'SELECT id FROM hub_token_families WHERE device_id = ? AND id != ? AND revoked_at IS NULL AND last_used_at > ? LIMIT 1',
        device.id,
        family.id,
        now - ctx.limits.inactivityMs,
      )
      const events = otherActive ? revokeFamilyRows(ctx, family, 'expired') : revokeDeviceRows(ctx, device.id, 'expired')
      return { outcome: { kind: 'expired' }, events }
    }
    if (token.generation === SPENT_GENERATION) {
      // Already used once to start a branch: a racing retry inside the grace window gets that same branch.
      const grace = ctx.graceCache.get(hash)
      if (grace && grace.expiresAt > now) return { outcome: { kind: 'ok', tokens: grace.tokens, user, device }, events: [] }
      return { outcome: { kind: 'invalid' }, events: [] }
    }
    if (token.generation !== family.generation) {
      const grace = ctx.graceCache.get(hash)
      if (grace && grace.expiresAt > now && token.generation === family.generation - 1) {
        return { outcome: { kind: 'ok', tokens: grace.tokens, user, device }, events: [] }
      }
      return forkFamily(ctx, hash, token.generation, family, device, user, now)
    }
    const next = family.generation + 1
    const successor = REFRESH_PREFIX + randomToken(ctx, 32)
    ctx.store.run('INSERT INTO hub_refresh_tokens (hash, family_id, generation, issued_at) VALUES (?, ?, ?, ?)', hashSecret(ctx, 'refresh', successor), family.id, next, now)
    const changed = ctx.store.run('UPDATE hub_token_families SET generation = ?, rotated_at = ?, last_used_at = ? WHERE id = ? AND generation = ? AND revoked_at IS NULL', next, now, now, family.id, family.generation)
    if (changed !== 1) return { outcome: { kind: 'invalid' }, events: [] }
    ctx.store.run('UPDATE hub_devices SET last_seen_at = ? WHERE id = ?', now, device.id)
    ctx.store.run('DELETE FROM hub_sessions WHERE family_id = ? AND expires_at < ?', family.id, now - 60 * 60 * 1000)
    const tokens = mintSession(ctx, family.id, device, successor)
    return { outcome: { kind: 'ok', tokens, user, device }, events: [], remember: { hash, tokens } }
  })
  if (result.remember) {
    ctx.graceCache.set(result.remember.hash, { tokens: result.remember.tokens, expiresAt: ctx.clock.now() + ctx.limits.refreshGraceMs })
  }
  pruneGrace(ctx)
  fireRevocations(ctx, result.events)
  return result.outcome
}

/** A retired refresh token may start one new branch; afterwards its row carries this generation. */
const SPENT_GENERATION = -1
/** Live token families (browser cookie jars) per device; beyond this the least recently used one is retired. */
export const MAX_FAMILIES_PER_DEVICE = 4

/**
 * A retired refresh token outside the grace window. One phone can hold several
 * cookie jars for the same device (Safari and the home-screen app, which iOS
 * copies at install), and a suspended phone can lose a refresh reply; both
 * present an older token later. Revoking the whole device for that locked the
 * owner out of their phone, so the token instead starts a new family once:
 * the token is marked spent (a second presentation is refused without a
 * revocation), the fork is audited, the device keeps at most
 * MAX_FAMILIES_PER_DEVICE live families, and each family still expires after
 * the inactivity window and with the device's absolute lifetime.
 */
function forkFamily(
  ctx: IdentityContext,
  hash: string,
  generation: number,
  family: FamilyRow,
  device: DeviceRow,
  user: UserRow,
  now: number,
): InternalOutcome {
  const spent = ctx.store.run('UPDATE hub_refresh_tokens SET generation = ? WHERE hash = ? AND generation = ?', SPENT_GENERATION, hash, generation)
  if (spent !== 1) return { outcome: { kind: 'invalid' }, events: [] }
  const actor = actorFor(user, device.id)
  const events: RevocationEvent[] = []
  const live = ctx.store.all<FamilyRow>(
    'SELECT * FROM hub_token_families WHERE device_id = ? AND revoked_at IS NULL ORDER BY last_used_at ASC, created_at ASC',
    device.id,
  )
  for (const old of live.slice(0, Math.max(0, live.length - (MAX_FAMILIES_PER_DEVICE - 1)))) {
    events.push(...revokeFamilyRows(ctx, old, 'family_limit', actor))
  }
  const tokens = startFamily(ctx, device)
  ctx.store.run('UPDATE hub_devices SET last_seen_at = ? WHERE id = ?', now, device.id)
  auditAction(ctx, { actor, action: `identity.family.fork:${device.id}`, params: { from: family.id, generation } })
  return { outcome: { kind: 'ok', tokens, user, device }, events, remember: { hash, tokens } }
}

function pruneGrace(ctx: IdentityContext): void {
  const now = ctx.clock.now()
  for (const [hash, entry] of ctx.graceCache) if (entry.expiresAt <= now) ctx.graceCache.delete(hash)
}

/** Identify the device behind a refresh token without rotating it (used by logout). */
export function deviceForRefreshToken(ctx: IdentityContext, refreshToken: string): { device: DeviceRow; user: UserRow } | undefined {
  if (!refreshToken.startsWith(REFRESH_PREFIX) || refreshToken.length > MAX_TOKEN_LENGTH) return undefined
  const row = ctx.store.one<{ family_id: string }>('SELECT family_id FROM hub_refresh_tokens WHERE hash = ?', hashSecret(ctx, 'refresh', refreshToken))
  if (!row) return undefined
  const family = ctx.store.one<FamilyRow>('SELECT * FROM hub_token_families WHERE id = ?', row.family_id)
  if (!family || family.revoked_at !== null) return undefined
  const device = ctx.store.one<DeviceRow>('SELECT * FROM hub_devices WHERE id = ?', family.device_id)
  if (!device || device.status === 'revoked') return undefined
  const user = ctx.store.one<UserRow>('SELECT * FROM hub_users WHERE id = ?', device.user_id)
  return user ? { device, user } : undefined
}

export interface Authed {
  readonly auth: AuthenticatedContext
  readonly user: UserRow
  readonly device: DeviceRow
  readonly session: SessionRow
  readonly sessionToken: string
}

export type AuthResult =
  | { readonly ok: true; readonly value: Authed }
  | { readonly ok: false; readonly reason: 'origin' | 'session' | 'csrf' }

/**
 * Resolve the `__Host-xerxes_session` cookie. An `Origin`, when present, must be
 * on the allowlist (publicOrigin, configured extras, loopback). Mutating methods
 * additionally need the `x-xerxes-csrf` token; `requireCsrf` overrides that.
 */
export function authenticateRequest(ctx: IdentityContext, info: HubHttpRequestInfo, options: { readonly requireCsrf?: boolean } = {}): AuthResult {
  syncRevocations(ctx)
  if (info.origin !== undefined && !originAllowed(ctx, info.origin)) return { ok: false, reason: 'origin' }
  const sessionToken = parseCookieHeader(info.cookie).get(SESSION_COOKIE)
  if (sessionToken === undefined || !sessionToken.startsWith(SESSION_PREFIX) || sessionToken.length > MAX_TOKEN_LENGTH) return { ok: false, reason: 'session' }
  const now = ctx.clock.now()
  const session = ctx.store.one<SessionRow>('SELECT * FROM hub_sessions WHERE hash = ?', hashSecret(ctx, 'session', sessionToken))
  if (!session || session.revoked_at !== null || session.expires_at <= now) return { ok: false, reason: 'session' }
  const device = ctx.store.one<DeviceRow>('SELECT * FROM hub_devices WHERE id = ?', session.device_id)
  if (!device || device.status !== 'active' || device.absolute_expires_at <= now) return { ok: false, reason: 'session' }
  const family = ctx.store.one<FamilyRow>('SELECT * FROM hub_token_families WHERE id = ?', session.family_id)
  if (!family || family.revoked_at !== null) return { ok: false, reason: 'session' }
  const user = ctx.store.one<UserRow>('SELECT * FROM hub_users WHERE id = ?', session.user_id)
  if (!user || user.kind !== 'user' || user.revoked_at !== null) return { ok: false, reason: 'session' }
  const needsCsrf = options.requireCsrf ?? !SAFE_METHODS.has(info.method.toUpperCase())
  if (needsCsrf && !(info.csrfToken !== undefined && constantTimeEqual(info.csrfToken, csrfFor(ctx, sessionToken)))) return { ok: false, reason: 'csrf' }
  const ceiling = device.scope_ceiling === null ? undefined : new Set<HubScope>(parseCeilingList(device.scope_ceiling))
  const principal: HubPrincipal = {
    kind: 'user',
    id: user.id,
    displayName: user.display_name,
    deviceId: device.id,
    ...(ceiling === undefined ? {} : { deviceScopeCeiling: ceiling }),
    hubOwner: user.is_owner === 1,
  }
  const auth: AuthenticatedContext = {
    principal,
    grants: listGrants(ctx.store, user.id),
    browserSessionId: session.id,
    actor: actorFor(user, device.id),
  }
  return { ok: true, value: { auth, user, device, session, sessionToken } }
}
