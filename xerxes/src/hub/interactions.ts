// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { redactInteractionPayload } from '../daemon/collaboration.js'
import { hasScope, type AccessTarget, type HubGrant, type HubPrincipal } from '../security/principal.js'
import type { CursorScope } from './cursor.js'
import { HUB_LIMITS, isRecord, utf8ByteLength } from './protocol.js'

/**
 * Per-recipient visibility of pending questions and permission requests.
 *
 * The Hub's own daemon lane is a trusted local (unsigned) client, so the journal
 * holds every interaction in full. The Hub decides what each phone may see, with
 * the same rule the daemon applies to signed subscribers:
 * - `full`: holds `permission.respond` on the task (owners and operators);
 * - `questions`: may answer questions (`question.respond`, or an agent holding
 *   `question.delegated`), so it sees question text and options, never a
 *   permission's command or inputs;
 * - `none`: sees only that something waits.
 */
export type HubInteractionVisibility = 'full' | 'questions' | 'none'

export const INTERACTION_REQUEST_TYPES: ReadonlySet<string> = new Set(['approval_request', 'permission_request', 'question_request'])
export const INTERACTION_RESOLVED_TYPES: ReadonlySet<string> = new Set([
  'approval_response', 'permission_response', 'question_response', 'permission_resolved', 'question_resolved', 'interaction_resolved',
])

/** Deepest `cron_event` nesting unwrapped; the daemon's subscription filter uses the same bound. */
const MAX_UNWRAP_DEPTH = 4

export function hubInteractionVisibility(principal: HubPrincipal, grants: readonly HubGrant[], target: AccessTarget, now: number): HubInteractionVisibility {
  if (hasScope(principal, grants, target, 'permission.respond', now)) return 'full'
  if (hasScope(principal, grants, target, 'question.respond', now) || hasScope(principal, grants, target, 'question.delegated', now)) return 'questions'
  return 'none'
}

/**
 * The turn event an event carries: itself, or the one a `cron_event` wraps
 * (`{ job_id, event_type, payload }`), unwrapped the way daemon/eventSubscriptions.ts does.
 */
export function unwrapInteractionEvent(type: string, payload: Readonly<Record<string, unknown>>): { readonly type: string; readonly payload: Readonly<Record<string, unknown>> } {
  let current = { type, payload }
  for (let depth = 0; depth < MAX_UNWRAP_DEPTH && current.type === 'cron_event'; depth += 1) {
    const inner = current.payload.payload
    if (typeof current.payload.event_type !== 'string' || !isRecord(inner)) break
    current = { type: current.payload.event_type, payload: inner }
  }
  return current
}

/** True for a question or permission request, directly or wrapped in a `cron_event`. */
export function isInteractionRequestEvent(type: string, payload: Readonly<Record<string, unknown>>): boolean {
  return INTERACTION_REQUEST_TYPES.has(unwrapInteractionEvent(type, payload).type)
}

/** The payload one recipient may see; events other than interaction requests are returned unchanged. */
export function interactionPayloadFor(type: string, payload: Readonly<Record<string, unknown>>, visibility: HubInteractionVisibility, depth = 0): Readonly<Record<string, unknown>> {
  if (visibility === 'full') return payload
  if (type === 'cron_event') {
    const inner = payload.payload
    if (typeof payload.event_type !== 'string' || !isRecord(inner)) return payload
    if (depth >= MAX_UNWRAP_DEPTH) {
      // Deeper than any real wrapper: fail closed rather than pass an unchecked body through.
      return { ...(payload.job_id === undefined ? {} : { job_id: payload.job_id }), event_type: payload.event_type, payload: { redacted: true }, redacted: true }
    }
    const shown = interactionPayloadFor(payload.event_type, inner, visibility, depth + 1)
    return shown === inner ? payload : { ...payload, payload: shown }
  }
  if (!INTERACTION_REQUEST_TYPES.has(type)) return payload
  if (visibility === 'questions' && type === 'question_request') return payload
  // `permission_request` is the legacy name of `approval_request`; both redact to the same card.
  return redactInteractionPayload(type === 'question_request' ? 'question_request' : 'approval_request', { ...payload })
}

/** Redact interaction requests in a list of journal events, deciding per event's own task. */
export function redactEventsFor<T extends { readonly type: string; readonly workspace_id: string; readonly session_id?: string; readonly payload: Readonly<Record<string, unknown>> }>(
  events: readonly T[], visibility: (target: AccessTarget) => HubInteractionVisibility,
): T[] {
  return events.map(event => {
    if (!isInteractionRequestEvent(event.type, event.payload)) return event
    const target: AccessTarget = { workspaceId: event.workspace_id, ...(event.session_id === undefined ? {} : { sessionId: event.session_id }) }
    const payload = interactionPayloadFor(event.type, event.payload, visibility(target))
    return payload === event.payload ? event : { ...event, payload }
  })
}

export interface PendingFrame { readonly type: string; readonly payload: Readonly<Record<string, unknown>> }

export interface PendingEventSource {
  watermark(): number
  read(afterSequence: number, filter: CursorScope, limit?: number, maxBytes?: number): {
    readonly events: readonly { readonly type: string; readonly payload: Readonly<Record<string, unknown>> }[]
    readonly throughSequence: number
    readonly hasMore: boolean
    readonly gap?: string
    /** First retained position; lets a scan that hit retention restart from there. */
    readonly oldestSequence?: number
  }
}

/** Most pending cards kept for one task; a task rarely waits on more than a handful. */
export const MAX_PENDING_FRAMES = 20
/** Default bound on the pending cards held for all tasks together. */
export const PENDING_INDEX_MAX_BYTES = 8 * 1024 * 1024

interface StoredFrame { readonly frame: PendingFrame; readonly bytes: number }

interface PendingEntry {
  through: number
  /** The scan began at the start of the journal: nothing older can exist. */
  readonly fromOrigin: boolean
  /** The scan saw a turn begin or end of this task: every card older than that was settled. */
  anchored: boolean
  /** The last advance reached the watermark; otherwise the cards reflect an older moment. */
  caughtUp: boolean
  /** A card was dropped to stay within bounds since the last anchor. */
  overflowed: boolean
  bytes: number
  readonly pending: Map<string, StoredFrame>
}

/**
 * The questions and permission requests each task is waiting on, rebuilt from the
 * Hub journal in full and kept incrementally: the first look at a task scans a
 * bounded window, later looks read only newer events of that task. The caller
 * redacts per recipient.
 *
 * `complete` is true only when the view is current (the scan reached the watermark)
 * and covers everything that may still wait: the scan saw a turn boundary of the
 * task, or it started at the beginning of the journal, and no card was dropped for
 * size. A view that ran out of page budget before reaching the watermark returns no
 * cards at all (they may already be resolved); the next look continues the scan.
 *
 * Memory is bounded by entry count and by total card bytes, evicting the least
 * recently looked-at task first.
 */
export class PendingInteractionIndex {
  private readonly entries = new Map<string, PendingEntry>()
  private readonly windowEvents: number
  private readonly maxPages: number
  private readonly maxEntries: number
  private readonly maxBytes: number
  private readonly maxEntryBytes: number
  private totalBytes = 0

  constructor(private readonly events: PendingEventSource, options: {
    readonly windowEvents?: number; readonly maxPages?: number; readonly maxEntries?: number; readonly maxBytes?: number
  } = {}) {
    this.windowEvents = options.windowEvents ?? 50_000
    this.maxPages = options.maxPages ?? 40
    this.maxEntries = options.maxEntries ?? 256
    this.maxBytes = options.maxBytes ?? PENDING_INDEX_MAX_BYTES
    // One task never holds more than a quarter of the budget, so a single huge task cannot evict every other one.
    this.maxEntryBytes = Math.max(1, Math.floor(this.maxBytes / 4))
  }

  /** Bytes of pending cards currently held across all tasks. */
  get heldBytes(): number { return this.totalBytes }
  /** Tasks currently indexed. */
  get size(): number { return this.entries.size }

  frames(workspaceId: string, sessionId: string): { readonly frames: PendingFrame[]; readonly complete: boolean } {
    const key = `${workspaceId}\0${sessionId}`
    let entry = this.entries.get(key)
    if (entry !== undefined) {
      this.entries.delete(key)
      this.totalBytes -= entry.bytes
    }
    if (entry === undefined || !this.advance(entry, workspaceId, sessionId)) entry = this.scan(workspaceId, sessionId)
    this.entries.set(key, entry)
    this.totalBytes += entry.bytes
    this.evict()
    if (!entry.caughtUp) return { frames: [], complete: false }
    const frames = [...entry.pending.values()].map(stored => stored.frame)
    const complete = (entry.anchored || entry.fromOrigin) && !entry.overflowed && frames.length <= MAX_PENDING_FRAMES
    return { frames: frames.slice(-MAX_PENDING_FRAMES), complete }
  }

  /** A fresh scan from the window start; on a retention gap, from the oldest retained event. */
  private scan(workspaceId: string, sessionId: string): PendingEntry {
    const start = Math.max(0, this.events.watermark() - this.windowEvents)
    const entry = freshEntry(start, start === 0)
    if (this.advance(entry, workspaceId, sessionId)) return entry
    const oldest = this.events.read(start, { workspaceIds: [workspaceId], sessionIds: [sessionId] }, 1, HUB_LIMITS.normalFrameBytes).oldestSequence
    const retry = freshEntry(Math.max(start, (oldest ?? start + 1) - 1), false)
    if (this.advance(retry, workspaceId, sessionId)) return retry
    retry.caughtUp = false
    return retry
  }

  /** Read newer events into the entry. False on a retention gap (the caller rescans). */
  private advance(entry: PendingEntry, workspaceId: string, sessionId: string): boolean {
    for (let page = 0; page < this.maxPages; page += 1) {
      const read = this.events.read(entry.through, { workspaceIds: [workspaceId], sessionIds: [sessionId] }, HUB_LIMITS.maxBatchEvents, HUB_LIMITS.normalFrameBytes)
      if (read.gap !== undefined) return false
      for (const raw of read.events) {
        // Only the task's own turn boundaries settle its cards; a wrapped job turn is a different run.
        if (raw.type === 'turn_begin' || raw.type === 'turn_end') {
          clearEntry(entry)
          entry.anchored = true
          continue
        }
        const event = unwrapInteractionEvent(raw.type, raw.payload)
        const id = event.payload.request_id ?? event.payload.id
        if (typeof id !== 'string' || id === '' || id.length > HUB_LIMITS.maxIdLength) continue
        if (INTERACTION_REQUEST_TYPES.has(event.type)) {
          deleteFrame(entry, id)
          const frame: PendingFrame = { type: event.type, payload: event.payload }
          const bytes = utf8ByteLength(JSON.stringify(frame))
          entry.pending.set(id, { frame, bytes })
          entry.bytes += bytes
          // Bounded memory: the oldest waiting card is dropped first, and the view says so.
          while (entry.pending.size > 1 && (entry.pending.size > MAX_PENDING_FRAMES * 2 || entry.bytes > this.maxEntryBytes)) {
            deleteFrame(entry, entry.pending.keys().next().value!)
            entry.overflowed = true
          }
        } else if (INTERACTION_RESOLVED_TYPES.has(event.type)) deleteFrame(entry, id)
      }
      entry.through = read.throughSequence
      if (!read.hasMore) {
        entry.caughtUp = true
        return true
      }
    }
    // Out of page budget: the newest events were not read; the next look continues from here.
    entry.caughtUp = false
    return true
  }

  /** Drop least recently used tasks until both bounds hold; the task just looked at stays. */
  private evict(): void {
    while (this.entries.size > 1 && (this.entries.size > this.maxEntries || this.totalBytes > this.maxBytes)) {
      const oldest = this.entries.keys().next().value!
      this.totalBytes -= this.entries.get(oldest)!.bytes
      this.entries.delete(oldest)
    }
  }
}

function freshEntry(through: number, fromOrigin: boolean): PendingEntry {
  return { through, fromOrigin, anchored: false, caughtUp: false, overflowed: false, bytes: 0, pending: new Map() }
}

function deleteFrame(entry: PendingEntry, id: string): void {
  const stored = entry.pending.get(id)
  if (stored === undefined) return
  entry.pending.delete(id)
  entry.bytes -= stored.bytes
}

function clearEntry(entry: PendingEntry): void {
  entry.pending.clear()
  entry.bytes = 0
  entry.overflowed = false
}

/** Keep the newest frames that fit `maxBytes` of JSON. */
export function fitPendingFrames(frames: readonly PendingFrame[], maxBytes: number): PendingFrame[] {
  const kept: PendingFrame[] = []
  let bytes = 2
  for (let index = frames.length - 1; index >= 0; index -= 1) {
    const size = utf8ByteLength(JSON.stringify(frames[index])) + 1
    if (bytes + size > maxBytes) break
    kept.unshift(frames[index]!)
    bytes += size
  }
  return kept
}
