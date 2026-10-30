// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, createHmac, timingSafeEqual } from 'node:crypto'
import { chmod, link, lstat, readFile, rm, writeFile } from 'node:fs/promises'

import type { RevocationListener } from '../identityPort.js'
import type { HubActor } from '../protocol.js'
import type { HubIdentityStore, UserRow } from './store.js'

/** Shared plumbing for the Hub identity subsystem: errors, limits, hashing, rate limiting. */

export type HubIdentityErrorCode =
  | 'invalid_request'
  | 'unauthorized'
  | 'forbidden'
  | 'not_found'
  | 'conflict'
  | 'rate_limited'
  | 'unavailable'
  | 'step_up_required'
  | 'pending_confirmation'
  | 'internal'

const HTTP_STATUS: Readonly<Record<HubIdentityErrorCode, number>> = {
  pending_confirmation: 403,
  invalid_request: 400,
  unauthorized: 401,
  forbidden: 403,
  not_found: 404,
  conflict: 409,
  rate_limited: 429,
  unavailable: 503,
  step_up_required: 403,
  internal: 500,
}

export class HubIdentityError extends Error {
  readonly code: HubIdentityErrorCode
  constructor(code: HubIdentityErrorCode, message: string, options?: { readonly cause?: unknown }) {
    super(message, options)
    this.name = 'HubIdentityError'
    this.code = code
  }
  get status(): number {
    return HTTP_STATUS[this.code]
  }
}

export interface Clock {
  now(): number
}
export type RandomSource = (bytes: number) => Uint8Array

export interface HubIdentityLimits {
  /** Browser session lifetime. */
  readonly sessionTtlMs: number
  /** A device that does not refresh within this window is expired. */
  readonly inactivityMs: number
  /** Hard device lifetime from creation. */
  readonly absoluteMs: number
  readonly pairingTtlMs: number
  /** Time the owner has to confirm a consumed pairing. */
  readonly confirmTtlMs: number
  /** Window in which a re-presented previous refresh token returns the same successor. */
  readonly refreshGraceMs: number
  readonly inviteMaxTtlMs: number
  readonly challengeTtlMs: number
  readonly pushDedupeMs: number
}

const DAY = 24 * 60 * 60 * 1000
export const DEFAULT_LIMITS: HubIdentityLimits = Object.freeze({
  sessionTtlMs: 5 * 60 * 1000,
  inactivityMs: 7 * DAY,
  absoluteMs: 30 * DAY,
  pairingTtlMs: 2 * 60 * 1000,
  confirmTtlMs: 5 * 60 * 1000,
  refreshGraceMs: 10 * 1000,
  inviteMaxTtlMs: 7 * DAY,
  challengeTtlMs: 5 * 60 * 1000,
  pushDedupeMs: 60 * 1000,
})

/** Limits are configurable downward only; a larger value than the default is refused. */
export function resolveLimits(partial: Partial<HubIdentityLimits> = {}): HubIdentityLimits {
  const resolved: Record<string, number> = { ...DEFAULT_LIMITS }
  for (const [key, value] of Object.entries(partial)) {
    if (value === undefined) continue
    const ceiling = (DEFAULT_LIMITS as unknown as Record<string, number>)[key]
    if (ceiling === undefined) throw new RangeError(`unknown identity limit ${key}`)
    const minimum = key === 'refreshGraceMs' ? 0 : 1
    if (!Number.isFinite(value) || value < minimum || value > ceiling) {
      throw new RangeError(`identity limit ${key} must be between ${minimum} and ${ceiling}`)
    }
    resolved[key] = value
  }
  return resolved as unknown as HubIdentityLimits
}

export class RateLimiter {
  private readonly hits = new Map<string, number[]>()

  /** Record one attempt; false when `limit` attempts already happened inside `windowMs`. */
  allow(key: string, limit: number, windowMs: number, now: number): boolean {
    const fresh = (this.hits.get(key) ?? []).filter(time => now - time < windowMs)
    if (fresh.length >= limit) {
      this.hits.set(key, fresh)
      return false
    }
    fresh.push(now)
    this.hits.set(key, fresh)
    if (this.hits.size > 10_000) this.prune(now, windowMs)
    return true
  }

  private prune(now: number, windowMs: number): void {
    for (const [key, times] of this.hits) {
      const newest = times.at(-1)
      if (newest === undefined || now - newest >= windowMs) this.hits.delete(key)
    }
  }
}

export interface IssuedTokens {
  readonly refreshToken: string
  readonly sessionToken: string
  readonly sessionId: string
  readonly csrfToken: string
  readonly sessionExpiresAt: number
  readonly refreshMaxAgeSeconds: number
}

export interface GraceEntry {
  /** The successor that was already issued for this retired refresh token. */
  readonly tokens: IssuedTokens
  readonly expiresAt: number
  /** Until this time any racing retry gets the same tokens unconditionally (the original grace window). */
  readonly raceUntil?: number
  /**
   * Set when the retired token started a new branch (a fork). After `raceUntil` and until
   * `expiresAt` the same token may collect that branch again, but only while the branch was
   * never used: an unused branch means its reply was lost (a suspended phone), a used one
   * means the holder already has it and a second presentation is refused.
   */
  readonly forkFamilyId?: string
  /** Set once the fork was collected again; a fork can be replayed at most once. */
  replayed?: boolean
}

export interface IdentityContext {
  readonly store: HubIdentityStore
  readonly clock: Clock
  readonly random: RandomSource
  /** Per-install secret; keys every token/code hash. */
  readonly pepper: Buffer
  /** Lowercase hex sha256 of the Hub signing key's SPKI. */
  readonly signerFingerprint: string
  /** Normalised `https://host[:port]` (or loopback http), when configured. */
  readonly publicOrigin: string | undefined
  readonly limits: HubIdentityLimits
  readonly rate: RateLimiter
  readonly listeners: Set<RevocationListener>
  /** In-memory only: previous refresh-token hash -> already issued successor token. */
  readonly graceCache: Map<string, GraceEntry>
  readonly revocationState: RevocationState
  /** In-memory WebAuthn challenges, keyed by base64url challenge. Single use. */
  readonly challenges: Map<string, ChallengeEntry>
  /** In-memory per-user push dedupe: `${userId}:${key}` -> last send time. */
  readonly pushDedupe: Map<string, number>
  readonly fetch: PushFetch
  /** Hostname suffixes a push endpoint may use (SSRF guard). */
  readonly pushHosts: readonly string[]
  readonly vapidSubject: string | undefined
  readonly vapid: VapidKeys
  /** Extra exact origins accepted for browser requests, besides publicOrigin and loopback. */
  readonly extraOrigins: readonly string[]
  /** A revocation listener threw; the revocation itself is already committed. */
  readonly onListenerError: (error: unknown) => void
}

/**
 * Load a 0600 secret file, or create it atomically (private temp file, then
 * `link`), so a concurrent creator can never read a partial secret.
 */
export async function loadOrCreateSecretFile(path: string, generate: () => string): Promise<string> {
  const read = async (): Promise<string | undefined> => {
    try {
      const info = await lstat(path)
      if (!info.isFile()) throw new Error(`${path} must be a regular file`)
      if ((info.mode & 0o077) !== 0) await chmod(path, 0o600)
      return await readFile(path, 'utf8')
    } catch (error) {
      if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') return undefined
      throw error
    }
  }
  const existing = await read()
  if (existing !== undefined) return existing
  const temporary = `${path}.${crypto.randomUUID()}.tmp`
  try {
    await writeFile(temporary, generate(), { mode: 0o600, flag: 'wx' })
    try {
      await link(temporary, path)
    } catch (error) {
      if (!(error && typeof error === 'object' && 'code' in error && error.code === 'EEXIST')) throw error
    }
  } finally {
    await rm(temporary, { force: true })
  }
  const created = await read()
  if (created === undefined) throw new Error(`could not create ${path}`)
  return created
}

export interface RevocationState {
  /** Revocations older than this were never relevant to sockets of this process. */
  readonly floor: number
  /** Device ids whose revocation this process already announced. */
  readonly fired: Set<string>
  dataVersion: number | undefined
}

export interface ChallengeEntry {
  readonly purpose: 'register' | 'assert'
  readonly sessionId: string
  readonly userId: string
  readonly expiresAt: number
}

export interface VapidKeys {
  /** base64url of the uncompressed P-256 point (65 bytes). */
  readonly publicKey: string
  readonly privateKeyPem: string
}

/** Minimal outbound HTTP port for web push; `Response` satisfies the result. */
export type PushFetch = (
  url: string,
  init: { readonly method: 'POST'; readonly headers: Record<string, string>; readonly body: Uint8Array },
) => Promise<{ readonly status: number }>

/**
 * Return `url.origin` when `value` is exactly a normalised http(s) origin, else undefined.
 * `"null"`, paths, default-port spellings and other near-misses are rejected.
 */
export function normalizeOrigin(value: string): string | undefined {
  let url: URL
  try {
    url = new URL(value)
  } catch {
    return undefined
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') return undefined
  return url.origin === value ? url.origin : undefined
}

const LOOPBACK_ORIGIN = /^http:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?$/

export function isLoopbackOrigin(origin: string): boolean {
  return LOOPBACK_ORIGIN.test(origin)
}

export function base64url(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString('base64url')
}

export function randomToken(ctx: Pick<IdentityContext, 'random'>, bytes: number): string {
  return base64url(ctx.random(bytes))
}

export function newId(ctx: Pick<IdentityContext, 'random'>, prefix: string, bytes = 9): string {
  return `${prefix}_${randomToken(ctx, bytes)}`
}

/** HMAC-SHA256 keyed with the install pepper and domain separated; the only form in which secrets are stored. */
export function hashSecret(ctx: Pick<IdentityContext, 'pepper'>, domain: string, value: string): string {
  return createHmac('sha256', ctx.pepper).update(domain).update('\0').update(value).digest('hex')
}

export function constantTimeEqual(left: string, right: string): boolean {
  const a = createHash('sha256').update(left).digest()
  const b = createHash('sha256').update(right).digest()
  return timingSafeEqual(a, b)
}

export function sha256Hex(value: string | Uint8Array): string {
  return createHash('sha256').update(value).digest('hex')
}

/** Trim, drop control characters, bound the length. */
export function cleanText(value: unknown, fallback: string, max: number): string {
  if (typeof value !== 'string') return fallback
  // eslint-disable-next-line no-control-regex
  const cleaned = value.replace(/[\u0000-\u001f\u007f]/g, '').trim().slice(0, max)
  return cleaned.length > 0 ? cleaned : fallback
}

export function jsonResponse(status: number, body: unknown, cookies: readonly string[] = []): Response {
  const headers = new Headers({ 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' })
  for (const cookie of cookies) headers.append('set-cookie', cookie)
  return new Response(JSON.stringify(body), { status, headers })
}

export function actorFor(user: Pick<UserRow, 'id' | 'display_name' | 'kind'>, deviceId?: string): HubActor {
  return {
    principal_kind: user.kind,
    principal_id: user.id,
    display_name: user.display_name,
    user_id: user.id,
    ...(deviceId === undefined ? {} : { device_id: deviceId }),
  }
}

export function errorResponse(error: HubIdentityError): Response {
  return jsonResponse(error.status, { error: { code: error.code, message: error.message } })
}
