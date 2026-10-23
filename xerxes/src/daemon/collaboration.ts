// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomUUID } from 'node:crypto'

import type { JsonRpcPayload } from '../protocol/jsonRpc.js'
import type { VerifiedHubContext } from '../security/hubIdentityVerifier.js'
import type { SteerAuthor, SteerInput } from '../streaming/steer.js'
import type { XerxesActor, XerxesInteractionRecord } from '../types/messages.js'
import { imageUrlContentParts, type TurnImage } from './images.js'

/**
 * Shared-session state for the Hub (step B): who is in a session, which
 * submissions were already accepted, the durable FIFO of steers queued behind
 * a busy turn, and the decisions people made on permission/question cards.
 *
 * Everything lives in `session.metadata[COLLAB_KEY]` so it is saved with the
 * transcript and survives a daemon restart. Authorship is only ever built from
 * a VerifiedHubContext, never from request parameters.
 */

export const COLLAB_KEY = 'xerxes_collaboration'
/** Maximum steers waiting behind one turn (matches the runtime's live queue). */
export const MAX_QUEUED_STEERS = 64
const MAX_SETTLED_STEERS = 128
const MAX_SUBMISSIONS = 512
const MAX_PENDING_DECISIONS = 128
const MAX_PARTICIPANTS = 64
const MAX_RESOLUTIONS = 1024

export const PENDING_INTERACTION_REVISION = 1
export const RESOLVED_INTERACTION_REVISION = 2

export type SteerStatus = 'queued' | 'applied' | 'cancelled'

export interface SteerEntry {
  readonly steer_id: string
  readonly submission_id?: string
  readonly principal_id: string
  readonly queue_sequence: number
  readonly text: string
  /** Images are not persisted; after a restart a recovered steer carries its text only. */
  readonly image_count: number
  /** Absent for a steer from a local (non-hub) caller, which has no verified identity. */
  readonly actor?: XerxesActor
  readonly display_name: string
  status: SteerStatus
  readonly queued_at: string
  settled_at?: string
  reason?: string
}

export interface SubmissionRecord {
  readonly principal_id: string
  readonly submission_id: string
  readonly digest: string
  readonly outcome: Record<string, unknown>
  readonly recorded_at: string
}

export interface PendingDecision {
  readonly tool_call_id: string
  readonly record: XerxesInteractionRecord
}

export interface CollabState {
  version: 1
  next_sequence: number
  steers: SteerEntry[]
  submissions: SubmissionRecord[]
  decisions: PendingDecision[]
  participants: string[]
  owner?: string
  shared: boolean
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** The live state object (mutations persist with the session), created on first use. */
export function collabState(metadata: Record<string, unknown>): CollabState {
  const existing = metadata[COLLAB_KEY]
  if (isRecord(existing) && existing.version === 1 && Array.isArray(existing.steers)
    && Array.isArray(existing.submissions) && Array.isArray(existing.decisions) && Array.isArray(existing.participants)
    && typeof existing.next_sequence === 'number') {
    return existing as unknown as CollabState
  }
  const created: CollabState = { version: 1, next_sequence: 1, steers: [], submissions: [], decisions: [], participants: [], shared: false }
  metadata[COLLAB_KEY] = created
  return created
}

/** Read-only view that never creates state, so legacy sessions stay byte-identical when saved. */
export function peekCollabState(metadata: Readonly<Record<string, unknown>>): CollabState | undefined {
  const existing = metadata[COLLAB_KEY]
  return isRecord(existing) && existing.version === 1 && Array.isArray(existing.steers) ? existing as unknown as CollabState : undefined
}

export function isSharedSession(metadata: Readonly<Record<string, unknown>> | undefined): boolean {
  return metadata !== undefined && peekCollabState(metadata)?.shared === true
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------

export type ActorKind = XerxesActor['action_kind']

export function actorFromContext(
  context: VerifiedHubContext,
  kind: ActorKind,
  extra: { readonly submission_id?: string; readonly interaction_id?: string; readonly now?: Date } = {},
): XerxesActor {
  return {
    principal_kind: context.principal_kind,
    principal_id: context.principal_id,
    ...(context.user_id === undefined ? {} : { user_id: context.user_id }),
    ...(context.device_id === undefined ? {} : { device_id: context.device_id }),
    display_name: context.display_name,
    // A human answer is bound to its signed human action; every other action to its signed request.
    action_id: context.human_action_id ?? context.request_id,
    ...(extra.submission_id === undefined ? {} : { submission_id: extra.submission_id }),
    ...(extra.interaction_id === undefined ? {} : { interaction_id: extra.interaction_id }),
    action_kind: kind,
    recorded_at: (extra.now ?? new Date()).toISOString(),
    ...(context.requested_by === undefined ? {} : { requested_by: context.requested_by }),
    ...(context.confirmed_by === undefined ? {} : { confirmed_by: context.confirmed_by }),
  }
}

export function authorSummary(actor: XerxesActor): JsonRpcPayload {
  return {
    display_name: actor.display_name,
    principal_kind: actor.principal_kind,
    principal_id: actor.principal_id,
    ...(actor.user_id === undefined ? {} : { user_id: actor.user_id }),
  }
}

// ---------------------------------------------------------------------------
// Participants and the shared-session rule
// ---------------------------------------------------------------------------

/**
 * Record a hub participant. The first participant is the owner, unless the
 * session already had a conversation before any hub principal appeared (then
 * the local human is the owner). Any other participant makes the session
 * shared, permanently. Returns true when this call made it shared.
 */
export function recordParticipant(metadata: Record<string, unknown>, context: VerifiedHubContext, hadHistory: boolean): boolean {
  const state = collabState(metadata)
  const id = `${context.principal_kind}:${context.user_id ?? context.principal_id}`
  if (!state.participants.includes(id)) {
    if (state.participants.length >= MAX_PARTICIPANTS) state.participants.shift()
    state.participants.push(id)
  }
  if (state.owner === undefined) state.owner = hadHistory ? 'local' : id
  const was = state.shared
  if (id !== state.owner) state.shared = true
  return state.shared && !was
}

// ---------------------------------------------------------------------------
// Submission idempotency
// ---------------------------------------------------------------------------

export function contentDigest(parts: readonly unknown[]): string {
  return createHash('sha256').update(JSON.stringify(parts)).digest('hex')
}

export type SubmissionLookup =
  | { readonly kind: 'new' }
  | { readonly kind: 'duplicate'; readonly outcome: Record<string, unknown> }
  | { readonly kind: 'conflict' }

/** Namespaced by session (the metadata owner), principal and submission id. */
export function lookupSubmission(state: CollabState, principalId: string, submissionId: string, digest: string): SubmissionLookup {
  const found = state.submissions.find(record => record.principal_id === principalId && record.submission_id === submissionId)
  if (!found) return { kind: 'new' }
  return found.digest === digest ? { kind: 'duplicate', outcome: found.outcome } : { kind: 'conflict' }
}

export function recordSubmission(state: CollabState, principalId: string, submissionId: string, digest: string, outcome: Record<string, unknown>, now = new Date()): void {
  while (state.submissions.length >= MAX_SUBMISSIONS) state.submissions.shift()
  state.submissions.push({ principal_id: principalId, submission_id: submissionId, digest, outcome, recorded_at: now.toISOString() })
}

// ---------------------------------------------------------------------------
// Durable steer queue
// ---------------------------------------------------------------------------

export interface SteerInputRecord {
  readonly text: string
  readonly image_count: number
  /** Verified author; absent for a local caller. */
  readonly actor?: XerxesActor
  readonly display_name: string
  readonly principal_id: string
  readonly submission_id?: string
  readonly now?: Date
}

/** Server-assigned order: the sequence is allocated here, never taken from a client clock. */
export function enqueueSteer(state: CollabState, input: SteerInputRecord): SteerEntry {
  const queued = state.steers.filter(entry => entry.status === 'queued').length
  if (queued >= MAX_QUEUED_STEERS) throw new SteerQueueFullError()
  const entry: SteerEntry = {
    steer_id: randomUUID().replaceAll('-', ''),
    ...(input.submission_id === undefined ? {} : { submission_id: input.submission_id }),
    principal_id: input.principal_id,
    queue_sequence: state.next_sequence++,
    text: input.text,
    image_count: input.image_count,
    ...(input.actor ? { actor: input.actor } : {}),
    display_name: input.display_name,
    status: 'queued',
    queued_at: (input.now ?? new Date()).toISOString(),
  }
  state.steers.push(entry)
  trimSteers(state)
  return entry
}

export class SteerQueueFullError extends Error {
  constructor() {
    super('steer-queue-full')
    this.name = 'SteerQueueFullError'
  }
}

function trimSteers(state: CollabState): void {
  let settled = state.steers.filter(entry => entry.status !== 'queued').length
  if (settled <= MAX_SETTLED_STEERS) return
  state.steers = state.steers.filter(entry => entry.status === 'queued' || settled-- <= MAX_SETTLED_STEERS)
}

/** Digest of everything that defines a submission, so the same id with different content is a conflict. */
export function submissionDigest(text: string, displayText: string, ifBusy: string, images: readonly TurnImage[]): string {
  return contentDigest([text, displayText, ifBusy, images.map(image => createHash('sha256').update(image.mediaType).update('\0').update(image.data).digest('hex'))])
}

/** The live steer a durable entry becomes when it is applied. Unattributed local steers stay plain strings. */
export function steerInputOf(entry: SteerEntry, images: readonly TurnImage[] = []): SteerInput {
  if (!entry.actor && !images.length) return entry.text
  const author: SteerAuthor | undefined = entry.actor
    ? {
      display_name: entry.display_name,
      actor: entry.actor,
      steer_id: entry.steer_id,
      queue_sequence: entry.queue_sequence,
      ...(entry.submission_id === undefined ? {} : { submission_id: entry.submission_id }),
    }
    : undefined
  return { text: entry.text, images: imageUrlContentParts(images), ...(author ? { author } : {}) }
}

export function hasQueuedSteers(metadata: Readonly<Record<string, unknown>>): boolean {
  return peekCollabState(metadata)?.steers.some(entry => entry.status === 'queued') === true
}

export function queuedSteers(state: CollabState): SteerEntry[] {
  return state.steers.filter(entry => entry.status === 'queued').sort((a, b) => a.queue_sequence - b.queue_sequence)
}

/** Move queued entries to a terminal status. Already-settled entries are left alone and not returned. */
export function settleSteers(state: CollabState, ids: ReadonlySet<string> | undefined, status: 'applied' | 'cancelled', reason?: string, now = new Date()): SteerEntry[] {
  const changed: SteerEntry[] = []
  for (const entry of state.steers) {
    if (entry.status !== 'queued' || (ids !== undefined && !ids.has(entry.steer_id))) continue
    entry.status = status
    entry.settled_at = now.toISOString()
    if (reason !== undefined) entry.reason = reason
    changed.push(entry)
  }
  return changed
}

export function steerStatusPayload(sessionId: string, entry: SteerEntry, turnId?: string): JsonRpcPayload {
  return {
    session_id: sessionId,
    steer_id: entry.steer_id,
    ...(entry.submission_id === undefined ? {} : { submission_id: entry.submission_id }),
    status: entry.status,
    queue_sequence: entry.queue_sequence,
    author: entry.actor ? authorSummary(entry.actor) : { display_name: entry.display_name, principal_kind: 'user', principal_id: entry.principal_id },
    ...(turnId ? { turn_id: turnId } : {}),
    ...(entry.reason === undefined ? {} : { reason: entry.reason }),
  }
}

// ---------------------------------------------------------------------------
// Interaction arbitration
// ---------------------------------------------------------------------------

export interface InteractionResolution {
  readonly interaction_id: string
  readonly kind: 'permission' | 'question'
  readonly decision: string
  readonly decided_by: string
  /** Absent when a local (non-hub) human answered. */
  readonly actor?: XerxesActor
  readonly interaction_revision: number
  readonly recorded_at: string
}

/** Bounded memory of who won each recently answered interaction. */
export class InteractionResolutions {
  private readonly resolved = new Map<string, InteractionResolution>()

  record(resolution: InteractionResolution): void {
    while (this.resolved.size >= MAX_RESOLUTIONS) {
      const oldest = this.resolved.keys().next().value
      if (oldest === undefined) break
      this.resolved.delete(oldest)
    }
    this.resolved.set(resolution.interaction_id, resolution)
  }

  get(interactionId: string): InteractionResolution | undefined {
    return this.resolved.get(interactionId)
  }
}

export function resolutionPayload(resolution: InteractionResolution): JsonRpcPayload {
  return {
    interaction_id: resolution.interaction_id,
    kind: resolution.kind,
    decided_by: resolution.decided_by,
    decision: resolution.decision,
    interaction_revision: resolution.interaction_revision,
    recorded_at: resolution.recorded_at,
    ...(resolution.actor ? { actor: authorSummary(resolution.actor) } : {}),
  }
}

/** The error a later answerer gets. `error` and `code` are both the stable machine string. */
export function alreadyResolvedFailure(resolution: InteractionResolution): JsonRpcPayload {
  return {
    ok: false,
    code: 'interaction-already-resolved',
    error: 'interaction-already-resolved',
    resolution: resolutionPayload(resolution),
  }
}

/** Fields appended to approval_response / question_response so every observer learns the winner. */
export function resolutionEventFields(resolution: InteractionResolution): JsonRpcPayload {
  return {
    decision: resolution.decision,
    ...(resolution.actor ? { resolved_by: authorSummary(resolution.actor) } : {}),
    decided_by: resolution.decided_by,
    interaction_revision: resolution.interaction_revision,
  }
}

/** Only a verified hub actor produces a transcript record; a local answer has no verified identity to record. */
export function interactionRecord(resolution: InteractionResolution & { readonly actor: XerxesActor }, extra: { readonly tool_name?: string; readonly tool_call_id?: string }): XerxesInteractionRecord {
  return {
    interaction_id: resolution.interaction_id,
    kind: resolution.kind,
    decision: resolution.decision,
    decided_by: resolution.actor,
    revision: resolution.interaction_revision,
    recorded_at: resolution.recorded_at,
    ...(extra.tool_name === undefined ? {} : { tool_name: extra.tool_name }),
    ...(extra.tool_call_id === undefined ? {} : { tool_call_id: extra.tool_call_id }),
  }
}

/** Queue a decision until the tool message it unblocked exists in the transcript. */
export function addPendingDecision(state: CollabState, decision: PendingDecision): void {
  while (state.decisions.length >= MAX_PENDING_DECISIONS) state.decisions.shift()
  state.decisions.push(decision)
}

type MessageRecord = Record<string, unknown>

/**
 * Attach queued `xerxes_interaction` records to the tool messages carrying their
 * call ids. Returns the messages unchanged (same array) when nothing applies.
 */
export function applyInteractionRecords<T extends MessageRecord>(messages: readonly T[], metadata: Readonly<Record<string, unknown>>): readonly T[] {
  const state = peekCollabState(metadata)
  if (!state?.decisions.length) return messages
  let result: T[] | undefined
  const remaining: PendingDecision[] = []
  for (const decision of state.decisions) {
    const index = (result ?? messages).findIndex(message => message.role === 'tool' && message.tool_call_id === decision.tool_call_id)
    if (index < 0) { remaining.push(decision); continue }
    result ??= [...messages]
    const target = result[index]!
    const existing = Array.isArray(target.xerxes_interaction) ? target.xerxes_interaction as XerxesInteractionRecord[] : []
    result[index] = { ...target, xerxes_interaction: [...existing, decision.record] }
  }
  state.decisions = remaining
  return result ?? messages
}

// ---------------------------------------------------------------------------
// Viewer redaction
// ---------------------------------------------------------------------------

/**
 * The pending-card variant a subscriber without `permission.respond` receives:
 * it can see that something waits, never the command, inputs or question text.
 */
export function redactInteractionPayload(type: string, payload: JsonRpcPayload): JsonRpcPayload {
  if (type === 'approval_request') {
    const id = payload.id ?? payload.request_id
    return {
      ...(id === undefined ? {} : { id }),
      ...(payload.request_id === undefined ? {} : { request_id: payload.request_id }),
      ...(payload.tool_call_id === undefined ? {} : { tool_call_id: payload.tool_call_id }),
      ...(payload.session_id === undefined ? {} : { session_id: payload.session_id }),
      redacted: true,
      interaction_revision: PENDING_INTERACTION_REVISION,
    }
  }
  if (type === 'question_request') {
    return {
      ...(payload.id === undefined ? {} : { id: payload.id }),
      ...(payload.tool_call_id === undefined ? {} : { tool_call_id: payload.tool_call_id }),
      ...(payload.session_id === undefined ? {} : { session_id: payload.session_id }),
      questions: [],
      redacted: true,
      interaction_revision: PENDING_INTERACTION_REVISION,
    }
  }
  return payload
}

/** Redact approval/question requests inside a journal page or `event_batch` payload; other events are untouched. */
export function redactEventPage(page: JsonRpcPayload): JsonRpcPayload {
  const events = page.events
  if (!Array.isArray(events)) return page
  let changed = false
  const mapped = events.map(event => {
    if (!isRecord(event) || (event.type !== 'approval_request' && event.type !== 'question_request') || !isRecord(event.payload)) return event
    changed = true
    return { ...event, payload: redactInteractionPayload(event.type, event.payload) }
  })
  return changed ? { ...page, events: mapped } : page
}

export function redactPendingFrames<T extends { readonly type: string; readonly payload: JsonRpcPayload }>(frames: readonly T[]): T[] {
  return frames.map(frame => ({ ...frame, payload: redactInteractionPayload(frame.type, frame.payload) }))
}
