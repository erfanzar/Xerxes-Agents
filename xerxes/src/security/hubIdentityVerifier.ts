// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createPublicKey, randomUUID } from 'node:crypto'
import { chmodSync, mkdirSync, readFileSync, realpathSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path'

import { publicKeyFingerprint, verifyHubAuth, type HubAuthClaims } from './hubAuthEnvelope.js'
import { AGENT_SCOPE_CEILING, HUMAN_ONLY_SCOPES, isHubScope, type HubScope } from './principal.js'

/**
 * Daemon-side trust in a Xerxes Hub (step B of the shared-session design).
 *
 * A Hub is trusted only after explicit enrollment: its Ed25519 public key is
 * pinned in `<daemon state>/hub-enrollments.json` together with the projects
 * or workspaces it may act for. Enrollment happens only through a local
 * non-hub RPC or the CLI-usable functions below, never through a request that
 * itself carries a `hub_auth` envelope.
 */

export interface HubEnrollmentRecord {
  readonly hub_id: string
  readonly public_key_pem: string
  /** sha256 of the SPKI DER; what the operator compares against the Hub's pairing QR. */
  readonly fingerprint: string
  readonly workspace_ids: readonly string[]
  readonly project_dirs: readonly string[]
  readonly enrolled_at: string
}

export interface HubEnrollmentInput {
  readonly hub_id: string
  readonly public_key_pem: string
  readonly project_dirs?: readonly string[]
  readonly workspace_ids?: readonly string[]
  /** Replace an existing pin with a different key. Without it a changed key is refused. */
  readonly replace?: boolean
  readonly now?: () => Date
}

export type HubRejectionReason =
  | 'malformed'
  | 'bad_signature'
  | 'expired'
  | 'not_yet_valid'
  | 'wrong_method'
  | 'params_mismatch'
  | 'unknown_hub'
  | 'hub_not_enrolled_for_project'
  | 'hub_not_enrolled_for_workspace'
  | 'replayed_nonce'
  | 'session_mismatch'
  | 'lifetime_too_long'
  | 'enrollment_unavailable'

/** The only identity the daemon ever attributes work to. Built from signed claims, never from params. */
export interface VerifiedHubContext {
  readonly hub_id: string
  readonly workspace_id: string
  readonly principal_kind: 'user' | 'agent'
  readonly principal_id: string
  readonly user_id?: string
  readonly device_id?: string
  readonly display_name: string
  readonly requested_by?: string
  readonly confirmed_by?: string
  /** Claimed scopes; for agent principals already clamped to the agent ceiling. */
  readonly scopes: readonly string[]
  readonly human_action_id?: string
  readonly request_id: string
  readonly session_id?: string
}

export type HubVerification =
  | { readonly ok: true; readonly context: VerifiedHubContext }
  | { readonly ok: false; readonly reason: HubRejectionReason; readonly message: string }

export interface HubVerifyRequest {
  readonly method: string
  readonly params: Readonly<Record<string, unknown>>
  readonly projectDir: string
  /** Public session id and/or connection session key the request targets. */
  readonly sessionId?: string
  readonly sessionKey?: string
}

export interface HubIdentityVerifierOptions {
  readonly path: string
  readonly now?: () => number
  readonly maxNonces?: number
  /** Longest accepted `expires_at - issued_at`. */
  readonly maxLifetimeMs?: number
  readonly clockSkewMs?: number
}

const DEFAULT_MAX_NONCES = 50_000
const DEFAULT_MAX_LIFETIME_MS = 5 * 60_000
const REASON_TEXT: Readonly<Record<HubRejectionReason, string>> = {
  malformed: 'the hub_auth envelope is malformed',
  bad_signature: 'the signature does not match the enrolled hub key',
  expired: 'the envelope has expired',
  not_yet_valid: 'the envelope is not valid yet (clock skew?)',
  wrong_method: 'the envelope was signed for a different method',
  params_mismatch: 'the request parameters differ from the signed parameters',
  unknown_hub: 'this hub is not enrolled on this daemon',
  hub_not_enrolled_for_project: 'this hub is not enrolled for this project',
  hub_not_enrolled_for_workspace: 'this hub is not enrolled for this workspace id',
  replayed_nonce: 'this envelope was already used',
  session_mismatch: 'the envelope was signed for a different session',
  lifetime_too_long: 'the envelope lifetime exceeds the accepted maximum',
  enrollment_unavailable: 'the hub enrollment file could not be read; refusing hub requests',
}

export class HubAuthError extends Error {
  constructor(readonly reason: HubRejectionReason | 'forbidden', message: string) {
    super(message)
    this.name = 'HubAuthError'
  }
}

export class HubIdentityVerifier {
  private readonly now: () => number
  private readonly maxNonces: number
  private readonly maxLifetimeMs: number
  private readonly clockSkewMs: number
  private hubs = new Map<string, HubEnrollmentRecord>()
  private loadError: string | undefined
  private loadedSignature: string | undefined
  /** Insertion-ordered: the oldest nonce is evicted first once the bound is reached. */
  private readonly nonces = new Map<string, number>()

  constructor(private readonly options: HubIdentityVerifierOptions) {
    this.now = options.now ?? (() => Date.now())
    this.maxNonces = options.maxNonces ?? DEFAULT_MAX_NONCES
    this.maxLifetimeMs = options.maxLifetimeMs ?? DEFAULT_MAX_LIFETIME_MS
    this.clockSkewMs = options.clockSkewMs ?? 30_000
    this.refresh()
  }

  get path(): string { return this.options.path }
  get nonceCount(): number { return this.nonces.size }

  list(): readonly HubEnrollmentRecord[] {
    this.refresh()
    return [...this.hubs.values()]
  }

  enroll(input: Omit<HubEnrollmentInput, 'now'>): HubEnrollmentRecord {
    const record = enrollHub(this.options.path, { ...input, now: () => new Date(this.now()) })
    this.refresh(true)
    return record
  }

  revoke(hubId: string): boolean {
    const removed = revokeHub(this.options.path, hubId)
    this.refresh(true)
    return removed
  }

  verify(envelope: unknown, request: HubVerifyRequest): HubVerification {
    this.refresh()
    if (this.loadError) return reject('enrollment_unavailable')
    const hubId = peekHubId(envelope)
    if (!hubId) return reject('malformed')
    const enrollment = this.hubs.get(hubId)
    if (!enrollment) return reject('unknown_hub')
    const now = this.now()
    const verified = verifyHubAuth(envelope, {
      publicKey: enrollment.public_key_pem,
      method: request.method,
      params: request.params,
      now,
      clockSkewMs: this.clockSkewMs,
    })
    if (!verified.ok) return reject(verified.reason)
    const claims = verified.claims
    if (claims.expires_at - claims.issued_at > this.maxLifetimeMs) return reject('lifetime_too_long')
    if (enrollment.project_dirs.length && !projectEnrolled(enrollment.project_dirs, request.projectDir)) {
      return reject('hub_not_enrolled_for_project')
    }
    if (enrollment.workspace_ids.length && !enrollment.workspace_ids.includes(claims.workspace_id)) {
      return reject('hub_not_enrolled_for_workspace')
    }
    if (request.sessionId !== undefined || request.sessionKey !== undefined) {
      const accepted = [request.sessionId, request.sessionKey].filter((value): value is string => value !== undefined)
      if (claims.session_id === undefined || !accepted.includes(claims.session_id)) return reject('session_mismatch')
    }
    // Replay is checked last so a request refused for any other reason never burns its nonce.
    const nonceKey = `${hubId}\0${claims.nonce}`
    this.pruneNonces(now)
    if (this.nonces.has(nonceKey)) return reject('replayed_nonce')
    this.nonces.set(nonceKey, claims.expires_at + this.clockSkewMs)
    return { ok: true, context: contextFrom(claims) }
  }

  private pruneNonces(now: number): void {
    if (this.nonces.size < this.maxNonces) return
    for (const [key, expiry] of this.nonces) if (expiry < now) this.nonces.delete(key)
    while (this.nonces.size >= this.maxNonces) {
      const oldest = this.nonces.keys().next().value
      if (oldest === undefined) break
      this.nonces.delete(oldest)
    }
  }

  private refresh(force = false): void {
    // Compare content, not mtime: two writes inside one timestamp tick (common on
    // Linux) left the mtime unchanged, so a revocation could go unnoticed. The
    // file is small, and missing a revocation is not acceptable.
    let content: string | undefined
    try { content = readFileSync(this.options.path, 'utf8') } catch { content = undefined }
    const signature = content ?? '\u0000missing'
    if (!force && signature === this.loadedSignature) return
    this.loadedSignature = signature
    const loaded = readEnrollmentFile(this.options.path)
    this.loadError = loaded.error
    this.hubs = new Map(loaded.records.map(record => [record.hub_id, record]))
  }
}

function reject(reason: HubRejectionReason): HubVerification {
  return { ok: false, reason, message: REASON_TEXT[reason] }
}

function peekHubId(envelope: unknown): string | undefined {
  if (typeof envelope !== 'object' || envelope === null) return undefined
  const claims = (envelope as { claims?: unknown }).claims
  if (typeof claims !== 'object' || claims === null) return undefined
  const hubId = (claims as { hub_id?: unknown }).hub_id
  return typeof hubId === 'string' && hubId ? hubId : undefined
}

function contextFrom(claims: HubAuthClaims): VerifiedHubContext {
  const scopes = claims.principal_kind === 'agent'
    ? claims.scopes.filter(scope => isHubScope(scope) && AGENT_SCOPE_CEILING.has(scope))
    : [...claims.scopes]
  return {
    hub_id: claims.hub_id,
    workspace_id: claims.workspace_id,
    principal_kind: claims.principal_kind,
    principal_id: claims.principal_id,
    ...(claims.user_id === undefined ? {} : { user_id: claims.user_id }),
    ...(claims.device_id === undefined ? {} : { device_id: claims.device_id }),
    display_name: claims.display_name,
    ...(claims.requested_by === undefined ? {} : { requested_by: claims.requested_by }),
    ...(claims.confirmed_by === undefined ? {} : { confirmed_by: claims.confirmed_by }),
    scopes,
    ...(claims.human_action_id === undefined ? {} : { human_action_id: claims.human_action_id }),
    request_id: claims.request_id,
    ...(claims.session_id === undefined ? {} : { session_id: claims.session_id }),
  }
}

// ---------------------------------------------------------------------------
// Method → scope policy
// ---------------------------------------------------------------------------

export type HubMethodPolicy =
  | { readonly kind: 'scope'; readonly scope: HubScope }
  | { readonly kind: 'refused'; readonly reason: string }

const READ_METHODS = new Set([
  'session.open', 'session.history', 'session.active_list', 'session.list', 'goal.inspect',
  'learning.list', 'runtime.status', 'subagent.inspect', 'events.subscribe', 'events.read', 'events.unsubscribe',
])

const WRITE_METHODS: Readonly<Record<string, HubScope>> = {
  'turn.submit': 'session.submit',
  prompt: 'session.submit',
  'turn.steer': 'session.steer',
  steer: 'session.steer',
  'subagent.steer': 'session.steer',
  'turn.cancel': 'session.cancel',
  cancel: 'session.cancel',
  'subagent.interrupt': 'session.cancel',
  permission_response: 'permission.respond',
  question_response: 'question.respond',
  set_model: 'session.configure',
  // Reviewing what Xerxes learned is a human-only act.
  'learning.review': 'approvals.manage',
}

/**
 * Session-less methods that run a model on this daemon's local provider
 * profiles for the Hub's XERXES supervisor. They never return a credential.
 */
export const HUB_MODEL_METHODS: ReadonlySet<string> = new Set(['hub.models.catalog', 'hub.models.options', 'hub.model.complete'])

/** Goal text that only reads (`/goal`, `/goal milestone`) needs no write scope. */
function goalInputIsRead(input: unknown): boolean {
  const text = typeof input === 'string' ? input.trim().toLowerCase() : ''
  return text === '' || text === 'milestone'
}

/**
 * Scope a hub-authenticated request needs. Everything outside this allowlist
 * (slash and shell routes, permission-mode changes, plugin and provider
 * management, `goal.decision`, enrollment) is refused for every hub principal.
 */
export function hubMethodPolicy(method: string, params: Readonly<Record<string, unknown>>): HubMethodPolicy {
  if (method === 'session.goal') {
    const input = params.input ?? params.text
    return { kind: 'scope', scope: goalInputIsRead(input) ? 'session.read' : 'goal.write' }
  }
  if (HUB_MODEL_METHODS.has(method)) return { kind: 'scope', scope: 'supervisor.model' }
  const write = Object.hasOwn(WRITE_METHODS, method) ? WRITE_METHODS[method] : undefined
  if (write) return { kind: 'scope', scope: write }
  if (READ_METHODS.has(method)) return { kind: 'scope', scope: 'session.read' }
  return { kind: 'refused', reason: `hub_auth is not accepted for ${method}` }
}

/** Undefined when the verified context may call the method; otherwise the reason it may not. */
export function hubAuthorizationFailure(
  context: VerifiedHubContext,
  method: string,
  params: Readonly<Record<string, unknown>>,
): string | undefined {
  const policy = hubMethodPolicy(method, params)
  if (policy.kind === 'refused') return policy.reason
  if (context.principal_kind === 'agent' && HUMAN_ONLY_SCOPES.has(policy.scope)) {
    return `${method} is a human-only route; agent principals can never use it`
  }
  if (!context.scopes.includes(policy.scope)) return `${method} requires the ${policy.scope} scope`
  return undefined
}

// ---------------------------------------------------------------------------
// Enrollment file (CLI-usable)
// ---------------------------------------------------------------------------

interface EnrollmentFile {
  readonly version: 1
  readonly hubs: readonly HubEnrollmentRecord[]
}

function readEnrollmentFile(path: string): { readonly records: readonly HubEnrollmentRecord[]; readonly error?: string } {
  let text: string
  try {
    text = readFileSync(path, 'utf8')
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { records: [] }
    return { records: [], error: error instanceof Error ? error.message : String(error) }
  }
  try {
    const parsed: unknown = JSON.parse(text)
    if (typeof parsed !== 'object' || parsed === null || (parsed as { version?: unknown }).version !== 1
      || !Array.isArray((parsed as { hubs?: unknown }).hubs)) throw new Error('unsupported enrollment file')
    return { records: ((parsed as { hubs: unknown[] }).hubs).map(parseRecord) }
  } catch (error) {
    return { records: [], error: error instanceof Error ? error.message : String(error) }
  }
}

function parseRecord(value: unknown): HubEnrollmentRecord {
  if (typeof value !== 'object' || value === null) throw new Error('invalid enrollment record')
  const record = value as Record<string, unknown>
  const strings = (field: string): string[] => {
    const raw = record[field]
    if (!Array.isArray(raw) || raw.some(item => typeof item !== 'string')) throw new Error(`invalid enrollment ${field}`)
    return raw as string[]
  }
  if (typeof record.hub_id !== 'string' || !record.hub_id || typeof record.public_key_pem !== 'string'
    || typeof record.fingerprint !== 'string' || typeof record.enrolled_at !== 'string') throw new Error('invalid enrollment record')
  // The pinned fingerprint must still describe the pinned key.
  if (publicKeyFingerprint(record.public_key_pem) !== record.fingerprint) throw new Error('enrollment fingerprint does not match its key')
  return {
    hub_id: record.hub_id, public_key_pem: record.public_key_pem, fingerprint: record.fingerprint,
    workspace_ids: strings('workspace_ids'), project_dirs: strings('project_dirs'), enrolled_at: record.enrolled_at,
  }
}

function writeEnrollmentFile(path: string, hubs: readonly HubEnrollmentRecord[]): void {
  const directory = dirname(path)
  mkdirSync(directory, { recursive: true, mode: 0o700 })
  const temporary = `${path}.${randomUUID()}.tmp`
  const body: EnrollmentFile = { version: 1, hubs }
  try {
    writeFileSync(temporary, `${JSON.stringify(body, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 })
    renameSync(temporary, path)
    chmodSync(path, 0o600)
  } finally {
    rmSync(temporary, { force: true })
  }
}

function canonicalDirectory(path: string): string {
  const raw = resolve(path)
  try { return realpathSync(raw) } catch { return raw }
}

function projectEnrolled(enrolled: readonly string[], projectDir: string): boolean {
  const project = canonicalDirectory(projectDir)
  return enrolled.some(directory => {
    const rel = relative(directory, project)
    return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel) && !rel.startsWith(`..${sep}`))
  })
}

/**
 * Pin a Hub's public key. Refuses (never overwrites) a corrupt enrollment
 * file, a non-Ed25519 key, relative project paths, and a changed key for an
 * existing hub id unless `replace` is set.
 */
export function enrollHub(path: string, input: HubEnrollmentInput): HubEnrollmentRecord {
  const hubId = input.hub_id.trim()
  if (!hubId) throw new Error('hub_id is required')
  let key
  try { key = createPublicKey(input.public_key_pem) } catch { throw new Error('public_key_pem is not a valid public key') }
  if (key.asymmetricKeyType !== 'ed25519') throw new Error('hub keys must be Ed25519')
  const projectDirs = (input.project_dirs ?? []).map(directory => {
    if (typeof directory !== 'string' || !isAbsolute(directory)) throw new Error('project_dirs must be absolute paths')
    return canonicalDirectory(directory)
  })
  const workspaceIds = (input.workspace_ids ?? []).map(id => {
    if (typeof id !== 'string' || !id.trim()) throw new Error('workspace_ids must be non-empty strings')
    return id.trim()
  })
  if (!projectDirs.length && !workspaceIds.length) throw new Error('enroll for at least one project_dir or workspace_id')
  const loaded = readEnrollmentFile(path)
  if (loaded.error) throw new Error(`refusing to modify the unreadable enrollment file: ${loaded.error}`)
  const fingerprint = publicKeyFingerprint(key)
  const existing = loaded.records.find(record => record.hub_id === hubId)
  if (existing && existing.fingerprint !== fingerprint && input.replace !== true) {
    throw new Error(`hub ${hubId} is already enrolled with a different key; revoke it first or pass replace`)
  }
  // Enrolling the same key again adds projects/workspaces to its scope, so a Hub
  // can be enrolled once per project. `replace` (a deliberate re-pin) sets the
  // scope to exactly what was given.
  const merge = existing !== undefined && existing.fingerprint === fingerprint && input.replace !== true
  const record: HubEnrollmentRecord = {
    hub_id: hubId,
    public_key_pem: key.export({ type: 'spki', format: 'pem' }).toString(),
    fingerprint,
    workspace_ids: merge ? [...new Set([...existing.workspace_ids, ...workspaceIds])] : workspaceIds,
    project_dirs: merge ? [...new Set([...existing.project_dirs, ...projectDirs])] : projectDirs,
    enrolled_at: (input.now?.() ?? new Date()).toISOString(),
  }
  writeEnrollmentFile(path, [...loaded.records.filter(item => item.hub_id !== hubId), record])
  return record
}

export function revokeHub(path: string, hubId: string): boolean {
  const loaded = readEnrollmentFile(path)
  if (loaded.error) throw new Error(`refusing to modify the unreadable enrollment file: ${loaded.error}`)
  const remaining = loaded.records.filter(record => record.hub_id !== hubId)
  if (remaining.length === loaded.records.length) return false
  writeEnrollmentFile(path, remaining)
  return true
}

export function listHubEnrollments(path: string): readonly HubEnrollmentRecord[] {
  const loaded = readEnrollmentFile(path)
  if (loaded.error) throw new Error(`enrollment file is unreadable: ${loaded.error}`)
  return loaded.records
}
