// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { canonicalJson } from '../../security/hubAuthEnvelope.js'
import type { HubAuditDecision, HubAuditInput, HubAuditOutcome } from '../identityPort.js'
import type { HubActor } from '../protocol.js'
import { sha256Hex, type Clock, type IdentityContext, type RandomSource } from './core.js'
import type { AuditRow, HubIdentityStore } from './store.js'

/**
 * Append-only, hash-chained audit log. Each row commits to its predecessor, so
 * editing, reordering or deleting a middle row is detected by
 * `verifyAuditChain`. The table also carries SQLite triggers that refuse
 * UPDATE/DELETE. Tail truncation is not detectable without an external anchor
 * (record the head hash elsewhere if that matters).
 */

export const AUDIT_SCHEMA_VERSION = 1 as const
export const AUDIT_GENESIS_HASH = '0'.repeat(64)

export interface StoredAuditEvent {
  readonly schema_version: number
  readonly sequence: number
  readonly event_id: string
  readonly occurred_at: string
  readonly actor: HubActor
  readonly action: string
  readonly workspace_id?: string
  readonly session_id?: string
  readonly interaction_id?: string
  readonly submission_id?: string
  readonly request_hash: string
  readonly decision: HubAuditDecision
  readonly outcome: HubAuditOutcome
  readonly previous_hash: string
  readonly event_hash: string
}

export type AuditVerification =
  | { readonly ok: true; readonly count: number; readonly headHash: string }
  | { readonly ok: false; readonly sequence: number; readonly reason: 'sequence_gap' | 'previous_hash_mismatch' | 'event_hash_mismatch' | 'malformed' }

export const LOCAL_ADMIN_ACTOR: HubActor = Object.freeze({
  principal_kind: 'user',
  principal_id: 'local-admin',
  display_name: 'Local admin',
})

export const SYSTEM_ACTOR: HubActor = Object.freeze({
  principal_kind: 'user',
  principal_id: 'hub-identity',
  display_name: 'Hub identity service',
})

const DECISIONS: ReadonlySet<string> = new Set(['allowed', 'denied'])
const OUTCOMES: ReadonlySet<string> = new Set(['accepted', 'rejected', 'unknown', 'failed', 'n/a'])

const SECRET_PATTERNS: readonly RegExp[] = [
  /\bx[rs]1[._][A-Za-z0-9_-]{8,}/, // this module's refresh / session tokens
  /\bbearer\s+[A-Za-z0-9._~+/=-]{8,}/i,
  /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/, // JWT
  /-----BEGIN [A-Z ]*KEY-----/,
  /\b(?:token|secret|password|passcode|code|cookie|authorization)\s*[=:]\s*\S{6,}/i,
  /#[A-Za-z0-9_-]{40,}/, // pairing / invite URL fragments
]

/** A long mixed-case alphanumeric run is the shape of a random token; hex, UUIDs and slugs are not. */
function looksLikeRandomToken(run: string): boolean {
  return run.length >= 32 && /[a-z]/.test(run) && /[A-Z]/.test(run) && /[0-9]/.test(run)
}

export function looksLikeSecret(text: string): boolean {
  for (const pattern of SECRET_PATTERNS) if (pattern.test(text)) return true
  for (const run of text.match(/[A-Za-z0-9_-]{32,}/g) ?? []) if (looksLikeRandomToken(run)) return true
  return false
}

/** Throws when any string inside `value` looks like a token, code or key. */
export function assertNoSecrets(value: unknown, path = 'record'): void {
  if (typeof value === 'string') {
    if (looksLikeSecret(value)) throw new Error(`audit refuses ${path}: it looks like a secret`)
    return
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertNoSecrets(item, `${path}[${index}]`))
    return
  }
  if (typeof value === 'object' && value !== null) {
    for (const [key, item] of Object.entries(value)) assertNoSecrets(item, `${path}.${key}`)
  }
}

export function uuidFromBytes(bytes: Uint8Array): string {
  const hex = Buffer.from(bytes.slice(0, 16)).toString('hex').split('')
  hex[12] = '4'
  hex[16] = ['8', '9', 'a', 'b'][Number.parseInt(hex[16] ?? '0', 16) % 4] ?? '8'
  const text = hex.join('')
  return `${text.slice(0, 8)}-${text.slice(8, 12)}-${text.slice(12, 16)}-${text.slice(16, 20)}-${text.slice(20)}`
}

function hashBody(body: Omit<StoredAuditEvent, 'event_hash'>): string {
  return sha256Hex(canonicalJson(body))
}

function optional(row: AuditRow): Pick<StoredAuditEvent, never> & Partial<Record<'workspace_id' | 'session_id' | 'interaction_id' | 'submission_id', string>> {
  return {
    ...(row.workspace_id === null ? {} : { workspace_id: row.workspace_id }),
    ...(row.session_id === null ? {} : { session_id: row.session_id }),
    ...(row.interaction_id === null ? {} : { interaction_id: row.interaction_id }),
    ...(row.submission_id === null ? {} : { submission_id: row.submission_id }),
  }
}

export function rowToEvent(row: AuditRow): StoredAuditEvent {
  return {
    schema_version: row.schema_version,
    sequence: row.sequence,
    event_id: row.event_id,
    occurred_at: row.occurred_at,
    actor: JSON.parse(row.actor) as HubActor,
    action: row.action,
    ...optional(row),
    request_hash: row.request_hash,
    decision: row.decision as HubAuditDecision,
    outcome: row.outcome as HubAuditOutcome,
    previous_hash: row.previous_hash,
    event_hash: row.event_hash,
  }
}

/** Append one event. Throws on validation or write failure; callers fail closed. */
export function appendAudit(
  parts: { readonly store: HubIdentityStore; readonly clock: Clock; readonly random: RandomSource },
  input: HubAuditInput,
): StoredAuditEvent {
  const { store, clock, random } = parts
  if (typeof input.action !== 'string' || input.action.length === 0 || input.action.length > 200) throw new Error('audit action must be 1-200 characters')
  if (!/^[0-9a-f]{64}$/.test(input.requestHash)) throw new Error('audit requestHash must be a sha256 hex digest')
  if (!DECISIONS.has(input.decision)) throw new Error(`audit decision ${String(input.decision)} is not allowed`)
  if (!OUTCOMES.has(input.outcome)) throw new Error(`audit outcome ${String(input.outcome)} is not allowed`)
  const { requestHash: _hash, ...checked } = input
  assertNoSecrets(checked, 'audit record')

  return store.transaction(() => {
    const last = store.one<{ sequence: number; event_hash: string }>('SELECT sequence, event_hash FROM hub_audit_events ORDER BY sequence DESC LIMIT 1')
    const body: Omit<StoredAuditEvent, 'event_hash'> = {
      schema_version: AUDIT_SCHEMA_VERSION,
      sequence: (last?.sequence ?? 0) + 1,
      event_id: uuidFromBytes(random(16)),
      occurred_at: new Date(clock.now()).toISOString(),
      actor: JSON.parse(canonicalJson(input.actor)) as HubActor,
      action: input.action,
      ...(input.workspaceId === undefined ? {} : { workspace_id: input.workspaceId }),
      ...(input.sessionId === undefined ? {} : { session_id: input.sessionId }),
      ...(input.interactionId === undefined ? {} : { interaction_id: input.interactionId }),
      ...(input.submissionId === undefined ? {} : { submission_id: input.submissionId }),
      request_hash: input.requestHash,
      decision: input.decision,
      outcome: input.outcome,
      previous_hash: last?.event_hash ?? AUDIT_GENESIS_HASH,
    }
    const event: StoredAuditEvent = { ...body, event_hash: hashBody(body) }
    store.run(
      `INSERT INTO hub_audit_events (sequence, schema_version, event_id, occurred_at, actor, action, workspace_id, session_id,
         interaction_id, submission_id, request_hash, decision, outcome, previous_hash, event_hash)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      event.sequence,
      event.schema_version,
      event.event_id,
      event.occurred_at,
      canonicalJson(event.actor),
      event.action,
      event.workspace_id ?? null,
      event.session_id ?? null,
      event.interaction_id ?? null,
      event.submission_id ?? null,
      event.request_hash,
      event.decision,
      event.outcome,
      event.previous_hash,
      event.event_hash,
    )
    return event
  })
}

/** Audit one identity action. `params` is hashed, never stored. */
export function auditAction(
  ctx: Pick<IdentityContext, 'store' | 'clock' | 'random'>,
  entry: {
    readonly actor: HubActor
    readonly action: string
    readonly params?: unknown
    readonly workspaceId?: string
    readonly sessionId?: string
    readonly decision?: HubAuditDecision
    readonly outcome?: HubAuditOutcome
  },
): StoredAuditEvent {
  return appendAudit(ctx, {
    actor: entry.actor,
    action: entry.action,
    ...(entry.workspaceId === undefined ? {} : { workspaceId: entry.workspaceId }),
    ...(entry.sessionId === undefined ? {} : { sessionId: entry.sessionId }),
    requestHash: sha256Hex(canonicalJson(entry.params ?? {})),
    decision: entry.decision ?? 'allowed',
    outcome: entry.outcome ?? 'accepted',
  })
}

export function listAudit(store: HubIdentityStore, options: { readonly afterSequence?: number; readonly limit?: number } = {}): StoredAuditEvent[] {
  const limit = Math.min(Math.max(1, Math.floor(options.limit ?? 100)), 1000)
  const rows = store.all<AuditRow>('SELECT * FROM hub_audit_events WHERE sequence > ? ORDER BY sequence ASC LIMIT ?', options.afterSequence ?? 0, limit)
  return rows.map(rowToEvent)
}

export function verifyAuditChain(store: HubIdentityStore): AuditVerification {
  let expectedSequence = 1
  let previousHash = AUDIT_GENESIS_HASH
  let count = 0
  for (const row of store.all<AuditRow>('SELECT * FROM hub_audit_events ORDER BY sequence ASC')) {
    if (row.sequence !== expectedSequence) return { ok: false, sequence: row.sequence, reason: 'sequence_gap' }
    if (row.previous_hash !== previousHash) return { ok: false, sequence: row.sequence, reason: 'previous_hash_mismatch' }
    let event: StoredAuditEvent
    try {
      event = rowToEvent(row)
    } catch {
      return { ok: false, sequence: row.sequence, reason: 'malformed' }
    }
    const { event_hash: _stored, ...body } = event
    let recomputed: string
    try {
      recomputed = hashBody(body)
    } catch {
      return { ok: false, sequence: row.sequence, reason: 'malformed' }
    }
    if (recomputed !== row.event_hash) return { ok: false, sequence: row.sequence, reason: 'event_hash_mismatch' }
    previousHash = row.event_hash
    expectedSequence += 1
    count += 1
  }
  return { ok: true, count, headHash: previousHash }
}
