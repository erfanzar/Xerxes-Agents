// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { Database } from 'bun:sqlite'

import { canonicalJson, paramsHash, type HubAuthClaims } from '../security/hubAuthEnvelope.js'
import { HUMAN_ONLY_SCOPES, hasScope, type HubGrant, type HubScope } from '../security/principal.js'
import { SUPERVISOR_PRINCIPAL_ID } from './supervisor/audit.js'
import type { DaemonCallOptions } from './daemonClient.js'
import type { AuthenticatedContext, HubIdentityPort } from './identityPort.js'
import {
  HUB_COMMAND_ACTIONS,
  HubProtocolError,
  isHubCommandAction,
  isRecord,
  readId,
  type HubCommandAction,
  type HubCommandResult,
  type HubErrorCode,
} from './protocol.js'
import type { PoolCallResult } from './runtimePool.js'
import type { HubSigner } from './signer.js'

/**
 * `hub.command`: the only way a phone changes anything. Each action maps to one
 * required scope and exactly one allowlisted daemon method with typed, bounded
 * arguments. The actor is always the authenticated connection; the daemon sees
 * it through a signed `hub_auth` envelope bound to the exact forwarded params.
 *
 * Delivery is at-least-once with a durable idempotency record, never "exactly
 * once": an outcome that could not be established is stored as `unknown` and is
 * never replayed automatically.
 */

export interface CommandBackend {
  call(workspaceId: string, method: string, params: Readonly<Record<string, unknown>>, options?: DaemonCallOptions): Promise<PoolCallResult>
  /** The project directory the pool forwards for this workspace, or undefined when unknown. */
  projectDir(workspaceId: string): string | undefined
}

export interface HubCommandRouterOptions {
  readonly databasePath: string
  readonly backend: CommandBackend
  readonly identity: Pick<HubIdentityPort, 'grantsFor' | 'audit'>
  readonly signer: Pick<HubSigner, 'hubId' | 'sign'>
  readonly clock?: () => number
  /** Optimistic-concurrency token source. Without it, `expected_revision` is refused rather than ignored. */
  readonly revision?: (workspaceId: string, sessionId: string | undefined) => number | undefined
  /** Reports failures that must not change an already-established outcome (post-call audit, record update). */
  readonly onError?: (error: unknown, context: string) => void
  /** Records older than this are pruned at open. Default 14 days. */
  readonly retentionMs?: number
  /**
   * The supervisor's per-dispatch overlay grants (SupervisorGrantOverlay.current). `question.answer`
   * is accepted only when these grants carry its scope. Absent: `question.answer` is always refused.
   */
  readonly overlayGrants?: () => readonly HubGrant[]
}

export const HUB_AUTH_LIFETIME_MS = 60_000
const MAX_TEXT = 32_000
const MAX_ID = 200

interface CommandSpec {
  readonly scope: HubScope
  readonly method: string
  readonly needsSession: boolean
  readonly interaction: boolean
  readonly build: (args: Record<string, unknown>, namespacedId: string) => Record<string, unknown>
}

const invalid = (message: string): HubProtocolError => new HubProtocolError('invalid_request', message)

function shape(args: Record<string, unknown>, allowed: readonly string[], required: readonly string[]): void {
  for (const key of Object.keys(args)) {
    if (!allowed.includes(key)) throw invalid(`unknown argument ${JSON.stringify(key.slice(0, 40))}`)
  }
  for (const key of required) if (args[key] === undefined) throw invalid(`args.${key} is required`)
}

function text(value: unknown, field: string, max: number): string {
  if (typeof value !== 'string' || value.trim().length === 0 || value.length > max) {
    throw invalid(`args.${field} must be a non-empty string of at most ${max} characters`)
  }
  return value
}

function answers(value: unknown): Record<string, string> {
  if (!isRecord(value)) throw invalid('args.answers must be an object of strings')
  const entries = Object.entries(value)
  if (entries.length > 50) throw invalid('args.answers accepts at most 50 entries')
  const result: Record<string, string> = {}
  for (const [key, answer] of entries) {
    if (key.length === 0 || key.length > MAX_ID || typeof answer !== 'string' || answer.length > 8000) {
      throw invalid('args.answers entries must map short ids to strings of at most 8000 characters')
    }
    result[key] = answer
  }
  return result
}

/** Same per-turn count as the daemon (daemon/images.ts MAX_TURN_IMAGES). */
const MAX_IMAGES = 16
/** Base64 characters across all images; the Hub frame cap bounds this further in practice. */
const MAX_IMAGE_BASE64_CHARS = 28 * 1024 * 1024

/**
 * Bounded pass-through of `{ media_type, data }` attachments. The daemon is the
 * authority: it decodes, sniffs magic bytes and enforces byte limits, and it
 * rejects the whole turn on any bad entry. Client hints such as width/height
 * are dropped here so they never reach the wire (or the signed params hash).
 */
function images(value: unknown): Record<string, unknown>[] | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value)) throw invalid('args.images must be an array of { media_type, data } entries')
  if (value.length > MAX_IMAGES) throw invalid(`args.images accepts at most ${MAX_IMAGES} entries`)
  let total = 0
  return value.map((entry, index) => {
    if (!isRecord(entry)) throw invalid(`args.images[${index}] must be an object with media_type and data`)
    const { media_type: mediaType, data } = entry
    if (typeof mediaType !== 'string' || !/^image\/[a-z0-9.+-]{1,40}$/.test(mediaType)) {
      throw invalid(`args.images[${index}].media_type must be an image/* type`)
    }
    if (typeof data !== 'string' || data.length === 0) throw invalid(`args.images[${index}].data must be a non-empty base64 string`)
    total += data.length
    if (total > MAX_IMAGE_BASE64_CHARS) throw invalid('args.images exceed the combined size limit')
    return { media_type: mediaType, data }
  })
}

function textAndImages(args: Record<string, unknown>): { readonly text: string; readonly images?: Record<string, unknown>[] } {
  const attached = images(args.images)
  const hasImages = attached !== undefined && attached.length > 0
  // An image alone is a valid turn or steer, the same rule as the desktop composer.
  const body = args.text === undefined && hasImages ? '' : text(args.text, 'text', MAX_TEXT)
  return { text: body, ...(hasImages ? { images: attached } : {}) }
}

const PERMISSION_RESPONSES = ['approve', 'approve_for_session', 'reject'] as const

export const COMMAND_SPECS: Readonly<Record<HubCommandAction, CommandSpec>> = Object.freeze({
  'session.open': {
    scope: 'session.submit', method: 'session.open', needsSession: false, interaction: false,
    build: args => {
      shape(args, ['agent_id'], [])
      return args.agent_id === undefined ? {} : { agent_id: text(args.agent_id, 'agent_id', MAX_ID) }
    },
  },
  'turn.submit': {
    scope: 'session.submit', method: 'turn.submit', needsSession: true, interaction: false,
    build: (args, id) => {
      shape(args, ['text', 'images'], [])
      if (args.text === undefined && args.images === undefined) throw invalid('args.text is required')
      return { ...textAndImages(args), submission_id: id }
    },
  },
  'turn.steer': {
    scope: 'session.steer', method: 'turn.steer', needsSession: true, interaction: false,
    build: (args, id) => {
      shape(args, ['text', 'images'], [])
      if (args.text === undefined && args.images === undefined) throw invalid('args.text is required')
      return { ...textAndImages(args), client_steer_id: id.slice(0, 40) }
    },
  },
  'turn.cancel': {
    scope: 'session.cancel', method: 'turn.cancel', needsSession: true, interaction: false,
    build: args => {
      shape(args, [], [])
      return {}
    },
  },
  'permission.respond': {
    scope: 'permission.respond', method: 'permission_response', needsSession: true, interaction: true,
    build: args => {
      shape(args, ['request_id', 'response'], ['request_id', 'response'])
      const response = args.response
      if (typeof response !== 'string' || !(PERMISSION_RESPONSES as readonly string[]).includes(response)) {
        throw invalid(`args.response must be one of ${PERMISSION_RESPONSES.join(', ')}`)
      }
      return { request_id: readId(args.request_id, 'args.request_id'), response }
    },
  },
  'question.respond': {
    scope: 'question.respond', method: 'question_response', needsSession: true, interaction: true,
    build: args => {
      shape(args, ['request_id', 'answers'], ['request_id', 'answers'])
      return { request_id: readId(args.request_id, 'args.request_id'), answers: answers(args.answers) }
    },
  },
  /**
   * XERXES Autopilot: the same daemon route and arguments as `question.respond`,
   * under the non-human-only scope `question.delegated`. There is deliberately no
   * delegated counterpart of `permission.respond`.
   */
  'question.answer': {
    scope: 'question.delegated', method: 'question_response', needsSession: true, interaction: true,
    build: args => {
      shape(args, ['request_id', 'answers'], ['request_id', 'answers'])
      return { request_id: readId(args.request_id, 'args.request_id'), answers: answers(args.answers) }
    },
  },
  'goal.set': {
    scope: 'goal.write', method: 'session.goal', needsSession: true, interaction: false,
    build: args => {
      shape(args, ['input'], ['input'])
      return { input: text(args.input, 'input', 4000) }
    },
  },
  'model.set': {
    scope: 'session.configure', method: 'set_model', needsSession: true, interaction: false,
    build: args => {
      shape(args, ['model', 'provider_profile'], ['model'])
      return {
        model: text(args.model, 'model', MAX_ID),
        ...(args.provider_profile === undefined ? {} : { provider_profile: text(args.provider_profile, 'provider_profile', MAX_ID) }),
      }
    },
  },
  'subagent.message': {
    scope: 'session.steer', method: 'subagent.steer', needsSession: true, interaction: false,
    build: args => {
      shape(args, ['task', 'message'], ['task', 'message'])
      return { task: text(args.task, 'task', MAX_ID), message: text(args.message, 'message', 8000) }
    },
  },
})

/** Required scope for an action; exported for the supervisor and tests. */
export function commandScope(action: HubCommandAction): HubScope {
  return COMMAND_SPECS[action].scope
}

/** The single daemon method an action may reach. */
export function commandDaemonMethod(action: HubCommandAction): string {
  return COMMAND_SPECS[action].method
}

interface ParsedCommand {
  readonly workspaceId: string
  readonly sessionId?: string
  readonly commandId: string
  readonly action: HubCommandAction
  readonly args: Record<string, unknown>
  readonly expectedRevision?: number
}

function parseCommand(params: Readonly<Record<string, unknown>>): ParsedCommand {
  const allowed = ['workspace_id', 'session_id', 'command_id', 'action', 'args', 'expected_revision']
  for (const key of Object.keys(params)) if (!allowed.includes(key)) throw invalid(`unknown parameter ${JSON.stringify(key.slice(0, 40))}`)
  const action = params.action
  if (!isHubCommandAction(action)) throw invalid(`action must be one of ${HUB_COMMAND_ACTIONS.join(', ')}`)
  const args = params.args === undefined ? {} : params.args
  if (!isRecord(args)) throw invalid('args must be an object')
  const revision = params.expected_revision
  if (revision !== undefined && (typeof revision !== 'number' || !Number.isSafeInteger(revision) || revision < 0)) {
    throw invalid('expected_revision must be a non-negative integer')
  }
  return {
    workspaceId: readId(params.workspace_id, 'workspace_id'),
    ...(params.session_id === undefined ? {} : { sessionId: readId(params.session_id, 'session_id') }),
    commandId: readId(params.command_id, 'command_id'),
    action,
    args,
    ...(revision === undefined ? {} : { expectedRevision: revision }),
  }
}

type StoredState = 'in_flight' | 'accepted' | 'rejected' | 'unknown'
interface CommandRow {
  key: string
  command_id: string
  request_hash: string
  state: StoredState
  result: string | null
  error_code: string | null
  error_message: string | null
}

function sha256(value: string): string {
  return createHash('sha256').update(value).digest('hex')
}

function requestHash(request: ParsedCommand): string {
  return sha256(canonicalJson({
    action: request.action,
    workspace_id: request.workspaceId,
    session_id: request.sessionId ?? null,
    args: request.args,
    expected_revision: request.expectedRevision ?? null,
  }))
}

/** Idempotency key: command ids are only unique within one principal, workspace and session. */
function namespacedId(principalId: string, request: Pick<ParsedCommand, 'workspaceId' | 'sessionId' | 'commandId'>): string {
  return sha256(JSON.stringify([principalId, request.workspaceId, request.sessionId ?? '', request.commandId]))
}

function rowToResult(row: CommandRow): HubCommandResult {
  if (row.state === 'accepted') {
    return { ok: true, command_id: row.command_id, state: 'accepted', result: row.result === null ? {} : (JSON.parse(row.result) as Record<string, unknown>) }
  }
  const code = (row.error_code ?? 'internal') as HubErrorCode
  const error = { code, message: row.error_message ?? 'no detail recorded' }
  if (row.state === 'rejected') return { ok: false, command_id: row.command_id, state: 'rejected', error }
  return { ok: false, command_id: row.command_id, state: 'unknown', error }
}

/** Find the daemon session key for a session id; the key, not the id, addresses daemon RPCs. */
export async function resolveSessionKey(backend: Pick<CommandBackend, 'call'>, workspaceId: string, sessionId: string): Promise<string | undefined> {
  const listed = await backend.call(workspaceId, 'session.active_list', {})
  if (listed.state !== 'accepted') {
    throw new HubProtocolError('unavailable', 'workspace is not available; try again when it is online')
  }
  const sessions = listed.result.sessions
  if (!Array.isArray(sessions)) return undefined
  for (const row of sessions) {
    if (isRecord(row) && row.id === sessionId && typeof row.key === 'string' && row.key) return row.key
  }
  return undefined
}

function sanitizeResult(action: HubCommandAction, result: Readonly<Record<string, unknown>>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const key of ['duplicate', 'delivered', 'cancelled'] as const) {
    if (typeof result[key] === 'boolean') out[key] = result[key]
  }
  if (action === 'session.open') {
    const session = result.session
    const id = isRecord(session) && typeof session.id === 'string' ? session.id : typeof result.session_id === 'string' ? result.session_id : undefined
    if (id !== undefined) out.session_id = id
  }
  return out
}

function mapRejection(action: HubCommandAction, code: string, message: string): { code: HubErrorCode; message: string } {
  const bounded = message.length > 300 ? `${message.slice(0, 297)}...` : message
  if (code === 'forbidden' || code === 'read_only') return { code: 'forbidden', message: bounded }
  if (code === 'turn-active') return { code: 'conflict', message: bounded }
  if (COMMAND_SPECS[action].interaction) return { code: 'interaction_already_resolved', message: bounded }
  if (code.includes('not-found') || code.includes('not_found')) return { code: 'not_found', message: bounded }
  return { code: 'invalid_request', message: bounded }
}

interface Inflight {
  readonly hash: string
  readonly promise: Promise<HubCommandResult>
}

export class HubCommandRouter {
  private readonly db: Database
  private readonly clock: () => number
  private readonly inflight = new Map<string, Inflight>()
  private closed = false

  constructor(private readonly options: HubCommandRouterOptions) {
    this.clock = options.clock ?? Date.now
    this.db = new Database(options.databasePath, { create: true, strict: true })
    this.db.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA synchronous = FULL;
      PRAGMA busy_timeout = 5000;
      CREATE TABLE IF NOT EXISTS hub_commands (
        key TEXT PRIMARY KEY, principal_id TEXT NOT NULL, workspace_id TEXT NOT NULL, session_id TEXT,
        command_id TEXT NOT NULL, action TEXT NOT NULL, request_hash TEXT NOT NULL, state TEXT NOT NULL,
        result TEXT, error_code TEXT, error_message TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS hub_commands_updated ON hub_commands(updated_at);
    `)
    const now = this.clock()
    // A command that was in flight when the Hub stopped has an unknowable outcome. Never replay it.
    this.db.query(`UPDATE hub_commands SET state = 'unknown', error_code = 'unavailable',
      error_message = 'The Hub restarted before the outcome was recorded; check the session before retrying with a new command id.',
      updated_at = ? WHERE state = 'in_flight'`).run(now)
    this.db.query('DELETE FROM hub_commands WHERE updated_at < ?').run(now - (options.retentionMs ?? 14 * 24 * 60 * 60 * 1000))
  }

  close(): void {
    if (this.closed) return
    this.closed = true
    this.db.close()
  }

  /** Run one command. Throws HubProtocolError for refusals; returns a recorded outcome otherwise. */
  async execute(context: AuthenticatedContext, params: Readonly<Record<string, unknown>>): Promise<HubCommandResult> {
    const request = parseCommand(params)
    const spec = COMMAND_SPECS[request.action]
    const hash = requestHash(request)
    const target = { workspaceId: request.workspaceId, ...(request.sessionId === undefined ? {} : { sessionId: request.sessionId }) }
    const grants = this.options.identity.grantsFor(context.principal.id)
    // Structural rule first: an agent principal can never answer a human interaction,
    // whatever its grants say. hasScope enforces the same rule; this makes it explicit here.
    const humanOnly = HUMAN_ONLY_SCOPES.has(spec.scope)
    // `question.answer` is the XERXES Autopilot route only: the supervisor principal, with scopes
    // that come solely from the per-dispatch overlay (never a stored grant). People use `question.respond`.
    const autopilotRoute = request.action === 'question.answer'
    const autopilotAllowed = !autopilotRoute || (
      context.principal.kind === 'agent' && context.principal.id === SUPERVISOR_PRINCIPAL_ID
      && this.options.overlayGrants !== undefined
      && hasScope(context.principal, this.options.overlayGrants(), target, spec.scope, this.clock()))
    if ((context.principal.kind === 'agent' && humanOnly) || !autopilotAllowed
      || !hasScope(context.principal, grants, target, spec.scope, this.clock())) {
      await this.auditBestEffort(context, request, hash, 'denied', 'rejected')
      throw new HubProtocolError('forbidden', `not permitted: ${request.action} requires ${spec.scope}`)
    }
    if (spec.needsSession && request.sessionId === undefined) throw invalid(`session_id is required for ${request.action}`)
    // The daemon verifier treats session.open as a read (it returns the session),
    // so it needs session.read alongside the action's own scope. Sign it only when
    // the caller actually holds it; the envelope never claims more than the grants.
    const signedScopes: string[] = [spec.scope]
    if (spec.method === 'session.open') {
      if (!hasScope(context.principal, grants, target, 'session.read', this.clock())) {
        await this.auditBestEffort(context, request, hash, 'denied', 'rejected')
        throw new HubProtocolError('forbidden', `not permitted: ${request.action} also requires session.read`)
      }
      signedScopes.push('session.read')
    }
    const forwarded = spec.build(request.args, namespacedId(context.principal.id, request))
    if (request.expectedRevision !== undefined) {
      if (this.options.revision === undefined) {
        throw new HubProtocolError('unsupported', 'expected_revision is not supported by this Hub')
      }
      if (this.options.revision(request.workspaceId, request.sessionId) !== request.expectedRevision) {
        throw new HubProtocolError('conflict', 'the session changed since expected_revision')
      }
    }

    const key = namespacedId(context.principal.id, request)
    const running = this.inflight.get(key)
    if (running) {
      if (running.hash !== hash) throw new HubProtocolError('submission_conflict', 'command_id was already used with different content')
      return running.promise
    }
    const promise = this.run(context, request, spec, key, hash, forwarded, signedScopes)
    this.inflight.set(key, { hash, promise })
    void promise.then(() => this.inflight.delete(key), () => this.inflight.delete(key))
    return promise
  }

  /** The recorded state of one of the caller's own commands. */
  status(context: AuthenticatedContext, params: Readonly<Record<string, unknown>>): HubCommandResult {
    const allowed = ['workspace_id', 'session_id', 'command_id']
    for (const key of Object.keys(params)) if (!allowed.includes(key)) throw invalid(`unknown parameter ${JSON.stringify(key.slice(0, 40))}`)
    const request = {
      workspaceId: readId(params.workspace_id, 'workspace_id'),
      ...(params.session_id === undefined ? {} : { sessionId: readId(params.session_id, 'session_id') }),
      commandId: readId(params.command_id, 'command_id'),
    }
    const key = namespacedId(context.principal.id, request)
    if (this.inflight.has(key)) {
      return { ok: false, command_id: request.commandId, state: 'unknown', error: { code: 'unavailable', message: 'the command is still in flight' } }
    }
    const row = this.lookup(key)
    if (!row) throw new HubProtocolError('not_found', 'no such command for this caller')
    return row.state === 'in_flight'
      ? { ok: false, command_id: request.commandId, state: 'unknown', error: { code: 'unavailable', message: 'the command is still in flight' } }
      : rowToResult(row)
  }

  private async run(
    context: AuthenticatedContext, request: ParsedCommand, spec: CommandSpec, key: string, hash: string, forwarded: Record<string, unknown>,
    signedScopes: readonly string[],
  ): Promise<HubCommandResult> {
    // Fail closed: no audit record, no mutation. Nothing is journaled yet, so the caller may retry.
    try {
      await this.audit(context, request, hash, 'allowed', 'n/a')
    } catch (error) {
      this.options.onError?.(error, 'audit')
      throw new HubProtocolError('unavailable', 'the audit log is unavailable; the command was not sent')
    }
    const existing = this.lookup(key)
    if (existing) {
      if (existing.request_hash !== hash) throw new HubProtocolError('submission_conflict', 'command_id was already used with different content')
      return existing.state === 'in_flight'
        ? { ok: false, command_id: request.commandId, state: 'unknown', error: { code: 'unavailable', message: 'the command is still in flight' } }
        : rowToResult(existing)
    }
    const projectDir = this.options.backend.projectDir(request.workspaceId)
    if (projectDir === undefined) throw new HubProtocolError('not_found', 'unknown workspace')

    let sessionKey: string
    if (request.sessionId === undefined) {
      sessionKey = `hub:${key.slice(0, 32)}`
    } else {
      const resolved = await resolveSessionKey(this.options.backend, request.workspaceId, request.sessionId)
      if (resolved !== undefined) sessionKey = resolved
      else if (request.action === 'session.open') sessionKey = request.sessionId
      else throw new HubProtocolError('not_found', 'the session is not active in this workspace; open it first')
    }

    // The pool adds project_dir (and history_limit 0 for summary methods) before the wire;
    // include exactly those here so the signed params_hash matches what the daemon receives.
    const wire: Record<string, unknown> = {
      ...forwarded,
      session_key: sessionKey,
      project_dir: projectDir,
      ...(spec.method === 'session.open' ? { history_limit: 0 } : {}),
    }
    const now = this.clock()
    const principal = context.principal
    const requestedBy = context.actor.requested_by ?? (principal.kind === 'agent' ? principal.onBehalfOf : undefined)
    const claims: HubAuthClaims = {
      hub_id: this.options.signer.hubId,
      principal_id: principal.id,
      principal_kind: principal.kind,
      ...(principal.kind === 'user' ? { user_id: principal.id, device_id: principal.deviceId } : {}),
      display_name: principal.displayName || principal.id,
      ...(requestedBy === undefined ? {} : { requested_by: requestedBy }),
      ...(context.actor.confirmed_by === undefined ? {} : { confirmed_by: context.actor.confirmed_by }),
      workspace_id: request.workspaceId,
      ...(request.sessionId === undefined ? {} : { session_id: request.sessionId }),
      method: spec.method,
      params_hash: paramsHash(wire),
      request_id: key,
      nonce: crypto.randomUUID(),
      issued_at: now,
      expires_at: now + HUB_AUTH_LIFETIME_MS,
      scopes: [...signedScopes],
      ...(spec.interaction ? { human_action_id: request.commandId } : {}),
    }
    const envelope = this.options.signer.sign(claims)

    try {
      this.insertInFlight(key, context.principal.id, request, hash, now)
    } catch (error) {
      this.options.onError?.(error, 'command journal')
      throw new HubProtocolError('unavailable', 'the command journal is unavailable; the command was not sent')
    }

    const outcome = await this.options.backend.call(request.workspaceId, spec.method, { ...wire, hub_auth: envelope })
    if (outcome.state === 'rejected' && (outcome.error.code === 'unavailable' || outcome.error.code === 'cancelled')) {
      // Rejected before anything was written: safe to retry under the same id.
      this.forget(key)
      throw new HubProtocolError('unavailable', outcome.error.message)
    }
    const result = this.record(key, request, outcome)
    try {
      await this.audit(context, request, hash, 'allowed', result.state === 'accepted' ? 'accepted' : result.state === 'rejected' ? 'rejected' : 'unknown')
    } catch (error) {
      // The command already ran; the pre-call record is the authoritative audit entry.
      this.options.onError?.(error, 'audit outcome')
    }
    return result
  }

  private record(key: string, request: ParsedCommand, outcome: PoolCallResult): HubCommandResult {
    let result: HubCommandResult
    if (outcome.state === 'accepted') {
      result = { ok: true, command_id: request.commandId, state: 'accepted', result: sanitizeResult(request.action, outcome.result) }
    } else if (outcome.state === 'unknown') {
      result = { ok: false, command_id: request.commandId, state: 'unknown', error: { code: 'unavailable', message: outcome.error.message } }
    } else {
      result = { ok: false, command_id: request.commandId, state: 'rejected', error: mapRejection(request.action, outcome.error.code, outcome.error.message) }
    }
    try {
      this.db.query('UPDATE hub_commands SET state = ?, result = ?, error_code = ?, error_message = ?, updated_at = ? WHERE key = ?').run(
        result.state, result.result === undefined ? null : JSON.stringify(result.result), result.error?.code ?? null, result.error?.message ?? null, this.clock(), key,
      )
    } catch (error) {
      this.options.onError?.(error, 'command outcome')
    }
    return result
  }

  private async audit(context: AuthenticatedContext, request: ParsedCommand, hash: string, decision: 'allowed' | 'denied', outcome: 'accepted' | 'rejected' | 'unknown' | 'n/a'): Promise<void> {
    const interaction = COMMAND_SPECS[request.action].interaction && typeof request.args.request_id === 'string' ? request.args.request_id : undefined
    await this.options.identity.audit({
      actor: context.actor,
      action: `command.${request.action}`,
      workspaceId: request.workspaceId,
      ...(request.sessionId === undefined ? {} : { sessionId: request.sessionId }),
      ...(interaction === undefined ? {} : { interactionId: interaction }),
      submissionId: request.commandId,
      requestHash: hash,
      decision,
      outcome,
    })
  }

  private async auditBestEffort(context: AuthenticatedContext, request: ParsedCommand, hash: string, decision: 'denied', outcome: 'rejected'): Promise<void> {
    try {
      await this.audit(context, request, hash, decision, outcome)
    } catch (error) {
      // A denial is refused either way; surface the audit failure without changing the answer.
      this.options.onError?.(error, 'audit denial')
    }
  }

  private lookup(key: string): CommandRow | undefined {
    return this.db.query<CommandRow, [string]>('SELECT key, command_id, request_hash, state, result, error_code, error_message FROM hub_commands WHERE key = ?').get(key) ?? undefined
  }

  private insertInFlight(key: string, principalId: string, request: ParsedCommand, hash: string, now: number): void {
    this.db.query(`INSERT INTO hub_commands (key, principal_id, workspace_id, session_id, command_id, action, request_hash, state, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, 'in_flight', ?, ?)`).run(key, principalId, request.workspaceId, request.sessionId ?? null, request.commandId, request.action, hash, now, now)
  }

  private forget(key: string): void {
    try {
      this.db.query("DELETE FROM hub_commands WHERE key = ? AND state = 'in_flight'").run(key)
    } catch (error) {
      this.options.onError?.(error, 'command journal')
    }
  }
}
