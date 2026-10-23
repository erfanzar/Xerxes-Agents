// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomBytes } from 'node:crypto'
import { join } from 'node:path'

import type { HubGrant } from '../../security/principal.js'
import type { AuthenticatedContext, HubAuditInput, HubHttpRequestInfo, HubIdentityPort, RevocationListener } from '../identityPort.js'
import { ensureHubDirectories, type HubPaths } from '../paths.js'
import { appendAudit, listAudit, verifyAuditChain, type AuditVerification, type StoredAuditEvent } from './audit.js'
import {
  DEFAULT_LIMITS,
  HubIdentityError,
  RateLimiter,
  cleanText,
  loadOrCreateSecretFile,
  newId,
  normalizeOrigin,
  resolveLimits,
  type Clock,
  type HubIdentityLimits,
  type IdentityContext,
  type PushFetch,
  type RandomSource,
} from './core.js'
import { listDevices, revokeAll, revokeDevice, revokeUser, syncRevocations, type DeviceSummary } from './devices.js'
import { listGrants, updateGrants, type UpdateGrantsInput } from './grants.js'
import { handleIdentityHttp } from './http.js'
import { acceptInvite, cancelInvite, createInvite, type AcceptedInvite, type CreateInviteInput, type InviteCreated } from './invites.js'
import { DEFAULT_PUSH_HOSTS, sendPush, type PushMessage, type PushResult } from './notifications.js'
import { cancelPairing, confirmPairing, createPairing, type ConfirmResult, type CreatePairingInput, type PairingCreated } from './pairing.js'
import { HubIdentityStore } from './store.js'
import { authenticateRequest } from './tokens.js'
import { requireRecentStepUp as requireStepUp, STEP_UP_MAX_AGE_MS } from './webauthn.js'
import { generateVapidPem, vapidPublicKeyOf } from './webPush.js'
import { LOCAL_ADMIN_ACTOR, SYSTEM_ACTOR } from './audit.js'

export const IDENTITY_DATABASE_FILENAME = 'identity.sqlite'
export const PEPPER_FILENAME = 'identity-pepper'
export const VAPID_FILENAME = 'vapid-key.pem'

export interface CreateHubIdentityOptions {
  readonly paths: HubPaths
  /** Lowercase hex sha256 of the Hub signing key's SPKI (`HubSigner.fingerprint`). */
  readonly signerFingerprint: string
  /** Exact `https://host[:port]` origin the phone uses. Required to create pairings and passkeys. */
  readonly publicOrigin?: string
  readonly clock?: Clock
  readonly random?: RandomSource
  /** Limits can only be tightened relative to the defaults. */
  readonly limits?: Partial<HubIdentityLimits>
  /** Outbound port for web push; defaults to the global `fetch`. */
  readonly fetch?: PushFetch
  /** Push service host suffixes accepted for subscriptions. */
  readonly pushHosts?: readonly string[]
  /** `mailto:` or `https:` contact URI for VAPID. Push is refused without it. */
  readonly vapidSubject?: string
  /** Extra exact browser origins accepted besides `publicOrigin` and loopback. */
  readonly extraOrigins?: readonly string[]
  readonly onListenerError?: (error: unknown) => void
  /** Poll for revocations written by another process (the CLI). 0 disables. Default 2000 ms. */
  readonly revocationPollMs?: number
}

export interface UserSummary {
  readonly id: string
  readonly displayName: string
  readonly owner: boolean
  readonly revoked: boolean
  readonly createdAt: number
}

/** Local-only administration. Nothing here is reachable over HTTP without a session and a passkey step-up. */
export interface HubIdentityAdmin {
  /** Creates the first (and only) owner. Refuses when an owner exists. */
  bootstrapOwner(input: { readonly name: string }): { readonly userId: string }
  ownerUserId(): string | undefined
  listUsers(): UserSummary[]
  createPairing(input: CreatePairingInput): PairingCreated
  /** Confirms a consumed pairing (`pr_…`) or invite (`inv_…`) by its 6 digit comparison code. */
  confirmPairing(id: string, comparisonCode: string): ConfirmResult
  cancelPairing(pairingId: string): void
  listDevices(filter?: { readonly userId?: string; readonly includeRevoked?: boolean }): DeviceSummary[]
  revokeDevice(deviceId: string, reason?: string): void
  revokeUser(userId: string, reason?: string): void
  revokeAll(reason?: string): void
  createInvite(input: CreateInviteInput): InviteCreated
  cancelInvite(inviteId: string): void
  /** Test and embedding helper; the HTTP route is `POST /api/hub/v1/invites/accept`. */
  acceptInvite(input: { readonly inviteId: string; readonly code: string; readonly displayName?: string; readonly deviceLabel?: string }): AcceptedInvite
  updateGrants(input: UpdateGrantsInput): HubGrant[]
  listGrants(principalId: string): HubGrant[]
  listAudit(options?: { readonly afterSequence?: number; readonly limit?: number }): StoredAuditEvent[]
  verifyAuditChain(): AuditVerification
  /** Throws `step_up_required` unless the browser session passed a passkey check within `maxAgeMs`. */
  requireRecentStepUp(sessionId: string, maxAgeMs?: number): void
  sendPush(userId: string, message: PushMessage): Promise<PushResult>
  /** Announce revocations committed by another process now (otherwise polled). */
  syncRevocations(): void
  readonly vapidPublicKey: string
}

export type HubIdentity = HubIdentityPort & { close(): void; admin: HubIdentityAdmin }

function defaultFetch(): PushFetch {
  return async (url, init) => {
    const response = await fetch(url, { method: init.method, headers: init.headers, body: new Blob([Uint8Array.from(init.body)]) })
    return { status: response.status }
  }
}

export async function createHubIdentity(options: CreateHubIdentityOptions): Promise<HubIdentity> {
  if (!/^[0-9a-f]{64}$/.test(options.signerFingerprint)) throw new RangeError('signerFingerprint must be a sha256 hex digest')
  let publicOrigin: string | undefined
  if (options.publicOrigin !== undefined) {
    publicOrigin = normalizeOrigin(options.publicOrigin)
    if (publicOrigin === undefined) throw new RangeError(`publicOrigin ${JSON.stringify(options.publicOrigin)} must be an exact origin such as https://hub.example.ts.net`)
  }
  const extraOrigins = (options.extraOrigins ?? []).map(origin => {
    const normalized = normalizeOrigin(origin)
    if (normalized === undefined) throw new RangeError(`extra origin ${JSON.stringify(origin)} must be an exact origin`)
    return normalized
  })
  const clock: Clock = options.clock ?? { now: () => Date.now() }
  const random: RandomSource = options.random ?? (bytes => randomBytes(bytes))
  const limits = resolveLimits(options.limits ?? {})

  await ensureHubDirectories(options.paths)
  const pepperText = await loadOrCreateSecretFile(join(options.paths.stateDir, PEPPER_FILENAME), () => randomBytes(32).toString('base64url'))
  const pepper = Buffer.from(pepperText.trim(), 'base64url')
  if (pepper.length < 32) throw new Error('identity pepper file is corrupt')
  const vapidPem = await loadOrCreateSecretFile(join(options.paths.stateDir, VAPID_FILENAME), generateVapidPem)
  const vapid = { publicKey: vapidPublicKeyOf(vapidPem), privateKeyPem: vapidPem }
  const store = await HubIdentityStore.open(join(options.paths.stateDir, IDENTITY_DATABASE_FILENAME))

  const listeners = new Set<RevocationListener>()
  const ctx: IdentityContext = {
    store,
    clock,
    random,
    pepper,
    signerFingerprint: options.signerFingerprint,
    publicOrigin,
    limits,
    rate: new RateLimiter(),
    listeners,
    graceCache: new Map(),
    revocationState: { floor: clock.now(), fired: new Set(), dataVersion: undefined },
    challenges: new Map(),
    pushDedupe: new Map(),
    fetch: options.fetch ?? defaultFetch(),
    pushHosts: options.pushHosts ?? DEFAULT_PUSH_HOSTS,
    vapidSubject: options.vapidSubject,
    vapid,
    extraOrigins,
    onListenerError: options.onListenerError ?? (error => console.error('[hub identity] revocation listener failed:', error instanceof Error ? error.message : String(error))),
  }

  const admin: HubIdentityAdmin = {
    bootstrapOwner(input) {
      const name = cleanText(input.name, '', 60)
      if (name === '') throw new HubIdentityError('invalid_request', 'owner name is required')
      const userId = newId(ctx, 'usr')
      store.transaction(() => {
        if (store.one('SELECT id FROM hub_users WHERE is_owner = 1')) throw new HubIdentityError('conflict', 'an owner already exists; there is no remote claim')
        store.run("INSERT INTO hub_users (id, kind, display_name, is_owner, created_at) VALUES (?, 'user', ?, 1, ?)", userId, name, clock.now())
        appendAudit(ctx, { actor: LOCAL_ADMIN_ACTOR, action: `identity.owner.bootstrap:${userId}`, requestHash: '0'.repeat(64), decision: 'allowed', outcome: 'accepted' })
      })
      return { userId }
    },
    ownerUserId: () => store.one<{ id: string }>('SELECT id FROM hub_users WHERE is_owner = 1')?.id,
    listUsers: () =>
      store.all<{ id: string; display_name: string; is_owner: number; revoked_at: number | null; created_at: number }>('SELECT * FROM hub_users ORDER BY created_at ASC, id ASC').map(row => ({
        id: row.id,
        displayName: row.display_name,
        owner: row.is_owner === 1,
        revoked: row.revoked_at !== null,
        createdAt: row.created_at,
      })),
    createPairing: input => createPairing(ctx, input),
    confirmPairing: (id, code) => confirmPairing(ctx, id, code),
    cancelPairing: id => cancelPairing(ctx, id),
    listDevices: filter => listDevices(ctx, filter),
    revokeDevice: (id, reason = 'revoked_by_admin') => revokeDevice(ctx, id, reason),
    revokeUser: (id, reason = 'revoked_by_admin') => revokeUser(ctx, id, reason),
    revokeAll: (reason = 'revoked_by_admin') => revokeAll(ctx, reason),
    createInvite: input => createInvite(ctx, input),
    cancelInvite: id => cancelInvite(ctx, id),
    acceptInvite: input => acceptInvite(ctx, input),
    updateGrants: input => updateGrants(ctx, input),
    listGrants: id => listGrants(store, id),
    listAudit: opts => listAudit(store, opts),
    verifyAuditChain: () => verifyAuditChain(store),
    requireRecentStepUp: (sessionId, maxAgeMs = STEP_UP_MAX_AGE_MS) => requireStepUp(ctx, sessionId, maxAgeMs),
    sendPush: (userId, message) => sendPush(ctx, userId, message),
    syncRevocations: () => syncRevocations(ctx),
    vapidPublicKey: vapid.publicKey,
  }

  const pollMs = options.revocationPollMs ?? 2000
  const timer = pollMs > 0 ? setInterval(() => {
    try {
      syncRevocations(ctx)
    } catch (error) {
      ctx.onListenerError(error)
    }
  }, pollMs) : undefined
  timer?.unref()
  let closed = false

  return {
    admin,
    async authenticate(request: HubHttpRequestInfo): Promise<AuthenticatedContext | null> {
      const result = authenticateRequest(ctx, request)
      return result.ok ? result.value.auth : null
    },
    grantsFor(principalId: string): readonly HubGrant[] {
      syncRevocations(ctx)
      return listGrants(store, principalId)
    },
    onRevoked(listener: RevocationListener): () => void {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
    async audit(record: HubAuditInput): Promise<void> {
      appendAudit(ctx, record)
    },
    handleHttp: (request, info) => handleIdentityHttp(ctx, request, info),
    close(): void {
      if (closed) return
      closed = true
      if (timer) clearInterval(timer)
      store.close()
    },
  }
}

export { DEFAULT_LIMITS, SYSTEM_ACTOR }
export type { HubIdentityLimits }
