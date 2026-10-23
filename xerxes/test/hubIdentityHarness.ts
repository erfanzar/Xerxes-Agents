// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { REFRESH_COOKIE, SESSION_COOKIE } from '../src/hub/identity/cookies.js'
import { HubIdentityError } from '../src/hub/identity/core.js'
import { createHubIdentity, type CreateHubIdentityOptions, type HubIdentity } from '../src/hub/identity/index.js'
import type { AuthenticatedContext, HubHttpRequestInfo } from '../src/hub/identityPort.js'
import type { HubPaths } from '../src/hub/paths.js'
import type { HubRole } from '../src/security/principal.js'

/** Shared fixtures for the hubIdentity*.test.ts files. */

export const ORIGIN = 'https://hub.test'
export const RP_ID = 'hub.test'
export const FINGERPRINT = 'ab'.repeat(32)
export const START = Date.parse('2026-10-21T12:00:00Z')
export const SECOND = 1000
export const MINUTE = 60 * SECOND
export const DAY = 24 * 60 * MINUTE

export interface TestClock {
  now(): number
  advance(ms: number): void
}

export function makeClock(start: number = START): TestClock {
  let current = start
  return {
    now: () => current,
    advance(ms: number) {
      current += ms
    },
  }
}

const cleanups: Array<() => void> = []

/** Call from `afterEach`. Closes identities and databases, then removes temp directories. */
export function cleanupAll(): void {
  while (cleanups.length > 0) {
    const cleanup = cleanups.pop()
    try {
      cleanup?.()
    } catch {
      // Closing an already-closed handle during teardown is not a test failure.
    }
  }
}

export function tempPaths(): { readonly dir: string; readonly paths: HubPaths } {
  const dir = mkdtempSync(join(tmpdir(), 'xerxes-hub-identity-'))
  cleanups.push(() => rmSync(dir, { recursive: true, force: true }))
  const stateDir = join(dir, 'hub')
  const paths: HubPaths = {
    stateDir,
    controlSocketPath: join(stateDir, 'control.sock'),
    pidPath: join(stateDir, 'hub.pid.json'),
    logsDir: join(stateDir, 'logs'),
    workspacesPath: join(stateDir, 'workspaces.json'),
    databasePath: join(stateDir, 'hub.sqlite'),
    ownershipLockPath: join(stateDir, 'ownership.lock'),
    startLockPath: join(stateDir, 'start.lock'),
    machinesPath: join(dir, 'machines.json'),
  }
  return { dir, paths }
}

export type HarnessOptions = Omit<CreateHubIdentityOptions, 'paths' | 'signerFingerprint' | 'clock'> & {
  /** Bootstrap an owner (default true). */
  readonly owner?: boolean
  readonly clock?: TestClock
}

export interface Harness {
  readonly identity: HubIdentity
  readonly admin: HubIdentity['admin']
  readonly clock: TestClock
  readonly paths: HubPaths
  readonly dir: string
  readonly ownerId: string
  readonly dbPath: string
}

export async function makeHarness(options: HarnessOptions = {}): Promise<Harness> {
  const { owner = true, clock = makeClock(), ...rest } = options
  const { dir, paths } = tempPaths()
  const identity = await createHubIdentity({ paths, signerFingerprint: FINGERPRINT, publicOrigin: ORIGIN, revocationPollMs: 0, ...rest, clock })
  cleanups.push(() => identity.close())
  const ownerId = owner ? identity.admin.bootstrapOwner({ name: 'Erfan' }).userId : ''
  return { identity, admin: identity.admin, clock, paths, dir, ownerId, dbPath: join(paths.stateDir, 'identity.sqlite') }
}

/** A second Hub instance over the same state directory, as the local CLI would be. */
export async function secondInstance(h: Harness): Promise<HubIdentity> {
  const identity = await createHubIdentity({ paths: h.paths, signerFingerprint: FINGERPRINT, publicOrigin: ORIGIN, revocationPollMs: 0, clock: h.clock })
  cleanups.push(() => identity.close())
  return identity
}

/** Direct SQLite handle on the identity database, for inspection and tamper tests. */
export function raw(h: Pick<Harness, 'dbPath'>): Database {
  const db = new Database(h.dbPath)
  cleanups.push(() => db.close())
  return db
}

export function errorCode(fn: () => unknown): string {
  try {
    fn()
  } catch (error) {
    if (error instanceof HubIdentityError) return error.code
    throw error
  }
  throw new Error('expected a HubIdentityError but nothing was thrown')
}

export function must<T>(value: T | null | undefined, what = 'value'): T {
  if (value === null || value === undefined) throw new Error(`${what} was ${String(value)}`)
  return value
}

export const b64u = (bytes: Uint8Array): string => Buffer.from(bytes).toString('base64url')

export function ticketOf(url: string): Record<string, string> {
  const fragment = new URL(url).hash.slice(1)
  return JSON.parse(Buffer.from(fragment, 'base64url').toString('utf8')) as Record<string, string>
}

export const refreshCookie = (token: string): string => `${REFRESH_COOKIE}=${token}`
export const sessionCookie = (token: string): string => `${SESSION_COOKIE}=${token}`

export interface CallOptions {
  readonly cookie?: string
  readonly csrf?: string
  /** Default ORIGIN; pass null to send no Origin header. */
  readonly origin?: string | null
  readonly ip?: string
}

export interface HttpResult {
  readonly res: Response
  readonly status: number
  readonly json: Record<string, unknown>
  readonly setCookies: readonly string[]
  readonly cookies: Readonly<Record<string, string>>
}

export function errorOf(result: HttpResult): { readonly code: string; readonly message: string } {
  const error = result.json.error as { code: string; message: string } | undefined
  return error ?? { code: '', message: '' }
}

export async function http(h: Pick<Harness, 'identity'>, method: 'GET' | 'POST', path: string, body?: unknown, call: CallOptions = {}): Promise<HttpResult> {
  const url = `${ORIGIN}/api/hub/v1/${path}`
  const request = new Request(url, {
    method,
    ...(body === undefined ? {} : { headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }),
  })
  const origin = call.origin === undefined ? ORIGIN : call.origin
  const info: HubHttpRequestInfo = {
    method,
    url,
    ...(call.cookie === undefined ? {} : { cookie: call.cookie }),
    ...(origin === null ? {} : { origin }),
    ...(call.csrf === undefined ? {} : { csrfToken: call.csrf }),
    remoteAddress: call.ip ?? '127.0.0.1',
  }
  const res = await h.identity.handleHttp(request, info)
  if (!res) throw new Error(`${path} was not handled as an identity route`)
  const text = await res.text()
  const json = text.length > 0 ? (JSON.parse(text) as Record<string, unknown>) : {}
  const setCookies = res.headers.getSetCookie()
  const cookies: Record<string, string> = {}
  for (const line of setCookies) {
    const pair = line.split(';')[0] ?? ''
    const index = pair.indexOf('=')
    if (index > 0) cookies[pair.slice(0, index)] = pair.slice(index + 1)
  }
  return { res, status: res.status, json, setCookies, cookies }
}

export interface BrowserSession {
  readonly status: string
  readonly refresh: string
  readonly session: string
  readonly csrf: string
  readonly deviceId: string
  readonly userId: string
  /** Both cookies, ready for a Cookie header. */
  readonly cookie: string
}

export function sessionOf(result: HttpResult): BrowserSession {
  const user = result.json.user as { id: string } | undefined
  const refresh = result.cookies[REFRESH_COOKIE] ?? ''
  const session = result.cookies[SESSION_COOKIE] ?? ''
  return {
    status: String(result.json.status),
    refresh,
    session,
    csrf: String(result.json.csrf_token),
    deviceId: String(result.json.device_id),
    userId: user?.id ?? '',
    cookie: `${REFRESH_COOKIE}=${refresh}; ${SESSION_COOKIE}=${session}`,
  }
}

export interface PairedDevice extends BrowserSession {
  readonly pairingId: string
  readonly code: string
  readonly comparisonCode: string
}

/** Create a pairing for the owner and consume it over HTTP. Active immediately unless `confirm: true`. */
export async function pairDevice(
  h: Harness,
  options: { readonly userId?: string; readonly confirm?: boolean; readonly role?: HubRole; readonly label?: string; readonly ip?: string } = {},
): Promise<PairedDevice> {
  const label = options.label ?? 'Phone'
  const created = h.admin.createPairing({
    userId: options.userId ?? h.ownerId,
    label,
    confirm: options.confirm ?? false,
    ...(options.role === undefined ? {} : { role: options.role }),
  })
  const ticket = ticketOf(created.url)
  const result = await http(h, 'POST', 'pair/consume', { pairing_id: ticket.pairing_id, code: ticket.code, device_label: label }, options.ip === undefined ? {} : { ip: options.ip })
  if (result.status !== 200) throw new Error(`pairing failed: ${result.status} ${JSON.stringify(result.json)}`)
  return { ...sessionOf(result), pairingId: created.pairingId, code: ticket.code ?? '', comparisonCode: String(result.json.comparison_code) }
}

export interface InvitedUser extends BrowserSession {
  readonly inviteId: string
  readonly code: string
}

/** Invite a new user (no owner confirmation) and accept over HTTP. */
export async function inviteUser(
  h: Harness,
  input: { readonly workspaceId: string; readonly sessionId?: string; readonly role: HubRole; readonly displayName?: string },
): Promise<InvitedUser> {
  const created = h.admin.createInvite({
    createdBy: h.ownerId,
    workspaceId: input.workspaceId,
    ...(input.sessionId === undefined ? {} : { sessionId: input.sessionId }),
    role: input.role,
    confirm: false,
  })
  const ticket = ticketOf(created.url)
  const result = await http(h, 'POST', 'invites/accept', { invite_id: ticket.invite_id, code: ticket.code, display_name: input.displayName ?? 'Guest' })
  if (result.status !== 200) throw new Error(`invite failed: ${result.status} ${JSON.stringify(result.json)}`)
  return { ...sessionOf(result), inviteId: created.inviteId, code: ticket.code ?? '' }
}

export function refresh(h: Pick<Harness, 'identity'>, refreshToken: string, call: CallOptions = {}): Promise<HttpResult> {
  return http(h, 'POST', 'session/refresh', undefined, { cookie: refreshCookie(refreshToken), ...call })
}

export function authenticate(
  h: Pick<Harness, 'identity'>,
  session: Pick<BrowserSession, 'session'>,
  origin: string | null = ORIGIN,
): Promise<AuthenticatedContext | null> {
  return h.identity.authenticate({
    method: 'GET',
    url: `${ORIGIN}/ws`,
    cookie: sessionCookie(session.session),
    ...(origin === null ? {} : { origin }),
  })
}
