// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import type { Database } from 'bun:sqlite'

import {
  AGENT_SCOPE_CEILING,
  effectiveScopes,
  type AccessTarget,
  type HubGrant,
  type HubPrincipal,
  type HubScope,
} from '../../security/principal.js'
import type { HubIdentityPort } from '../identityPort.js'
import { HubProtocolError } from '../protocol.js'
import type { SupervisorReply } from '../voice/supervisorBridge.js'
import { SUPERVISOR_DISPLAY_NAME, SUPERVISOR_PRINCIPAL_ID } from './audit.js'

/**
 * Supervisor conversation threads.
 *
 * One shared thread per Hub by default, plus optional private threads. A
 * private thread has its own transcript and its own read scope: nothing said in
 * it reaches the shared thread or its context.
 *
 * Read scope is the intersection of the people who can see the thread (the
 * participants active recently, or the owner of a private thread) and the agent
 * ceiling. The supervisor therefore never shows one participant something
 * another participant of the same thread could not read.
 */

export type UserPrincipal = Extract<HubPrincipal, { readonly kind: 'user' }>
export type ThreadKind = 'shared' | 'private'
export type MessageRole = 'user' | 'xerxes' | 'system'
export type MessageSource = 'text' | 'voice'

export const SHARED_THREAD_ID = 'shared'
export const MAX_MESSAGE_CHARS = 4000
/** XERXES Autopilot's owner-only thread: wake turns run here, never in the shared thread. */
export const AUTOPILOT_THREAD_PREFIX = 'autopilot-'

export function autopilotThreadId(ownerId: string): string {
  return `${AUTOPILOT_THREAD_PREFIX}${ownerId}`
}

/** Ids that name a supervisor conversation, never a task. */
export function isSupervisorThreadId(id: string): boolean {
  return id === SHARED_THREAD_ID || id.startsWith('private-') || id.startsWith(AUTOPILOT_THREAD_PREFIX)
}

/** What the agent could do on behalf of a human: the human's effective scopes inside the agent ceiling. */
export function delegatedScopes(principal: UserPrincipal, grants: readonly HubGrant[], target: AccessTarget, now: number): ReadonlySet<HubScope> {
  const scopes = new Set<HubScope>()
  for (const scope of effectiveScopes(principal, grants, target, now)) if (AGENT_SCOPE_CEILING.has(scope)) scopes.add(scope)
  return scopes
}

export interface ThreadRecord {
  readonly id: string
  readonly kind: ThreadKind
  readonly ownerId?: string
  readonly title: string
  readonly createdAt: number
}

export interface ThreadMessage {
  readonly seq: number
  readonly id: string
  readonly threadId: string
  readonly role: MessageRole
  readonly text: string
  readonly at: string
  readonly authorId?: string
  readonly authorName?: string
  readonly deviceId?: string
  readonly source: MessageSource
  readonly clientMessageId?: string
  readonly causalMessageId?: string
  /** When present, only these principals can read the message. */
  readonly audience?: readonly string[]
}

export interface ThreadParticipant {
  readonly id: string
  readonly displayName: string
  readonly hubOwner: boolean
  readonly lastSeenAt: number
}

export type SupervisorEventType = 'supervisor.thread_event' | 'supervisor.confirmation'

export interface SupervisorEventSink {
  send(principalIds: readonly string[], type: SupervisorEventType, payload: Readonly<Record<string, unknown>>): void
}

export interface ScopeReader {
  scopes(target: AccessTarget): ReadonlySet<HubScope>
  canRead(target: AccessTarget): boolean
}

export interface SupervisorThreadsOptions {
  readonly db: Database
  readonly identity: Pick<HubIdentityPort, 'grantsFor'> & Partial<Pick<HubIdentityPort, 'onRevoked'>>
  readonly clock?: () => number
  /** Whether a person may take part in the shared thread at all (they can see something on this Hub). */
  readonly eligible?: (principal: UserPrincipal, grants: readonly HubGrant[]) => boolean
  /** Participants not seen for this long no longer narrow what the shared thread may show. Default 12 hours. */
  readonly activeWindowMs?: number
  readonly onError?: (error: unknown, context: string) => void
}

interface ThreadRow { id: string; kind: string; owner_id: string | null; title: string; created_at: number }
interface ParticipantRow { thread_id: string; principal_id: string; display_name: string; hub_owner: number; joined_at: number; last_seen_at: number }
interface MessageRow {
  seq: number; id: string; thread_id: string; role: string; text: string; author_id: string | null; author_name: string | null
  device_id: string | null; source: string; client_message_id: string | null; causal_message_id: string | null
  audience: string | null; created_at: number
}

function toThread(row: ThreadRow): ThreadRecord {
  return {
    id: row.id,
    kind: row.kind === 'private' ? 'private' : 'shared',
    ...(row.owner_id === null ? {} : { ownerId: row.owner_id }),
    title: row.title,
    createdAt: row.created_at,
  }
}

function toMessage(row: MessageRow): ThreadMessage {
  return {
    seq: row.seq,
    id: row.id,
    threadId: row.thread_id,
    role: row.role === 'user' || row.role === 'system' ? row.role : 'xerxes',
    text: row.text,
    at: new Date(row.created_at).toISOString(),
    ...(row.author_id === null ? {} : { authorId: row.author_id }),
    ...(row.author_name === null ? {} : { authorName: row.author_name }),
    ...(row.device_id === null ? {} : { deviceId: row.device_id }),
    source: row.source === 'voice' ? 'voice' : 'text',
    ...(row.client_message_id === null ? {} : { clientMessageId: row.client_message_id }),
    ...(row.causal_message_id === null ? {} : { causalMessageId: row.causal_message_id }),
    ...(row.audience === null ? {} : { audience: JSON.parse(row.audience) as string[] }),
  }
}

/** Wire form of one message, shaped for the clients' `parseSupervisorMessage`. */
export function messageToWire(message: ThreadMessage): Record<string, unknown> {
  const actor = message.role === 'xerxes'
    ? { principal_kind: 'agent', principal_id: SUPERVISOR_PRINCIPAL_ID, display_name: SUPERVISOR_DISPLAY_NAME }
    : message.authorId === undefined
      ? undefined
      : {
          principal_kind: 'user',
          principal_id: message.authorId,
          display_name: message.authorName ?? message.authorId,
          ...(message.deviceId === undefined ? {} : { device_id: message.deviceId }),
        }
  return {
    id: message.id,
    thread_id: message.threadId,
    role: message.role,
    text: message.text,
    at: message.at,
    source: message.source,
    ...(message.authorName === undefined ? {} : { author: message.authorName }),
    ...(actor === undefined ? {} : { actor }),
    ...(message.causalMessageId === undefined ? {} : { causal_message_id: message.causalMessageId }),
  }
}

export class SupervisorThreads {
  private readonly db: Database
  private readonly clock: () => number
  private sink: SupervisorEventSink | undefined
  private readonly known = new Map<string, UserPrincipal>()
  private readonly replyListeners = new Map<string, Set<(reply: SupervisorReply) => void>>()
  private readonly working = new Set<string>()
  private readonly offRevoked: (() => void) | undefined

  constructor(private readonly options: SupervisorThreadsOptions) {
    this.db = options.db
    this.clock = options.clock ?? Date.now
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS hub_supervisor_threads (
        id TEXT PRIMARY KEY, kind TEXT NOT NULL, owner_id TEXT, title TEXT NOT NULL, created_at INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS hub_supervisor_participants (
        thread_id TEXT NOT NULL, principal_id TEXT NOT NULL, display_name TEXT NOT NULL, hub_owner INTEGER NOT NULL,
        joined_at INTEGER NOT NULL, last_seen_at INTEGER NOT NULL, PRIMARY KEY (thread_id, principal_id)
      );
      CREATE TABLE IF NOT EXISTS hub_supervisor_messages (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, thread_id TEXT NOT NULL, role TEXT NOT NULL,
        text TEXT NOT NULL, author_id TEXT, author_name TEXT, device_id TEXT, source TEXT NOT NULL,
        client_message_id TEXT, causal_message_id TEXT, audience TEXT, created_at INTEGER NOT NULL
      );
      CREATE UNIQUE INDEX IF NOT EXISTS hub_supervisor_messages_client
        ON hub_supervisor_messages(thread_id, author_id, client_message_id) WHERE client_message_id IS NOT NULL;
      CREATE INDEX IF NOT EXISTS hub_supervisor_messages_thread ON hub_supervisor_messages(thread_id, seq);
    `)
    this.offRevoked = options.identity.onRevoked?.(event => {
      // A revoked person stops narrowing (and receiving) the shared thread. A single revoked device does not remove the person.
      if (event.deviceId === undefined && event.browserSessionId === undefined) this.removePrincipal(event.principalId)
    })
  }

  setSink(sink: SupervisorEventSink | undefined): void {
    this.sink = sink
  }

  close(): void {
    this.offRevoked?.()
    this.replyListeners.clear()
  }

  // ── threads ────────────────────────────────────────────────────────

  ensureShared(): ThreadRecord {
    const existing = this.get(SHARED_THREAD_ID)
    if (existing) return existing
    this.db.query('INSERT OR IGNORE INTO hub_supervisor_threads (id, kind, owner_id, title, created_at) VALUES (?, ?, NULL, ?, ?)')
      .run(SHARED_THREAD_ID, 'shared', 'Team', this.clock())
    return this.get(SHARED_THREAD_ID)!
  }

  get(id: string): ThreadRecord | undefined {
    const row = this.db.query<ThreadRow, [string]>('SELECT * FROM hub_supervisor_threads WHERE id = ?').get(id)
    return row ? toThread(row) : undefined
  }

  /**
   * The Hub owner's XERXES Autopilot thread, created on first use. It is a private
   * thread of that owner: anyone else gets `not_found`. Wake turns run here, so no
   * other participant's message can ever be in an Autopilot turn's context.
   */
  autopilotFor(ownerId: string): ThreadRecord {
    const id = autopilotThreadId(ownerId)
    this.db.query('INSERT OR IGNORE INTO hub_supervisor_threads (id, kind, owner_id, title, created_at) VALUES (?, ?, ?, ?, ?)')
      .run(id, 'private', ownerId, 'Autopilot', this.clock())
    const thread = this.get(id)!
    if (thread.kind !== 'private' || thread.ownerId !== ownerId) throw new Error('the Autopilot thread id is taken by another thread')
    return thread
  }

  /** The person's private thread, created on first use. `fresh` starts another one. */
  privateFor(principal: UserPrincipal, fresh = false): ThreadRecord {
    if (!fresh) {
      // The owner's Autopilot thread is private too, but it is never their ordinary private thread.
      const row = this.db.query<ThreadRow, [string, string]>(
        "SELECT * FROM hub_supervisor_threads WHERE kind = 'private' AND owner_id = ? AND id NOT LIKE ? ORDER BY created_at DESC LIMIT 1",
      ).get(principal.id, `${AUTOPILOT_THREAD_PREFIX}%`)
      if (row) return toThread(row)
    }
    const id = `private-${randomUUID()}`
    this.db.query('INSERT INTO hub_supervisor_threads (id, kind, owner_id, title, created_at) VALUES (?, ?, ?, ?, ?)')
      .run(id, 'private', principal.id, 'Private', this.clock())
    return this.get(id)!
  }

  /**
   * Resolve a thread for a person and record them as a participant. A private
   * thread answers `not_found` to anyone but its owner, so its existence does
   * not leak.
   */
  access(threadId: string, principal: UserPrincipal): ThreadRecord {
    const thread = this.get(threadId)
    if (!thread) throw new HubProtocolError('not_found', 'no such thread')
    this.known.set(principal.id, principal)
    if (thread.kind === 'private') {
      if (thread.ownerId !== principal.id) throw new HubProtocolError('not_found', 'no such thread')
      return thread
    }
    const grants = this.options.identity.grantsFor(principal.id)
    if (this.options.eligible && !this.options.eligible(principal, grants)) {
      throw new HubProtocolError('forbidden', 'not permitted: you can see no workspace on this Hub')
    }
    this.db.query(`INSERT INTO hub_supervisor_participants (thread_id, principal_id, display_name, hub_owner, joined_at, last_seen_at)
      VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(thread_id, principal_id) DO UPDATE SET
      display_name = excluded.display_name, hub_owner = excluded.hub_owner, last_seen_at = excluded.last_seen_at`)
      .run(thread.id, principal.id, principal.displayName, principal.hubOwner ? 1 : 0, this.clock(), this.clock())
    return thread
  }

  /** The person as last seen by an authenticated call; used by the voice bridge, which only knows an actor. */
  principalFor(principalId: string): UserPrincipal | undefined {
    return this.known.get(principalId)
  }

  /**
   * The person from the live record, or rebuilt from the stored participant row
   * after a restart. The stored form has no device and may lag a role change by
   * one visit; grants are always read fresh, so authority is never taken from it.
   */
  principalOrStored(principalId: string): UserPrincipal | undefined {
    const live = this.known.get(principalId)
    if (live) return live
    const row = this.db.query<ParticipantRow, [string]>('SELECT * FROM hub_supervisor_participants WHERE principal_id = ? LIMIT 1').get(principalId)
    return row ? { kind: 'user', id: principalId, displayName: row.display_name, deviceId: '', hubOwner: row.hub_owner === 1 } : undefined
  }

  removePrincipal(principalId: string): void {
    this.db.query('DELETE FROM hub_supervisor_participants WHERE principal_id = ?').run(principalId)
    this.known.delete(principalId)
  }

  participants(threadId: string): ThreadParticipant[] {
    return this.db.query<ParticipantRow, [string]>('SELECT * FROM hub_supervisor_participants WHERE thread_id = ? ORDER BY joined_at')
      .all(threadId)
      .map(row => ({ id: row.principal_id, displayName: row.display_name, hubOwner: row.hub_owner === 1, lastSeenAt: row.last_seen_at }))
  }

  /** Principals who receive this thread's events. */
  recipients(thread: ThreadRecord): string[] {
    return thread.kind === 'private' ? (thread.ownerId === undefined ? [] : [thread.ownerId]) : this.participants(thread.id).map(row => row.id)
  }

  // ── read scope ─────────────────────────────────────────────────────

  private readers(thread: ThreadRecord, live: UserPrincipal): UserPrincipal[] {
    if (thread.kind === 'private') return [live]
    const cutoff = this.clock() - (this.options.activeWindowMs ?? 12 * 60 * 60 * 1000)
    const readers = new Map<string, UserPrincipal>()
    for (const row of this.participants(thread.id)) {
      if (row.lastSeenAt < cutoff) continue
      readers.set(row.id, this.known.get(row.id) ?? { kind: 'user', id: row.id, displayName: row.displayName, deviceId: '', hubOwner: row.hubOwner })
    }
    readers.set(live.id, live)
    return [...readers.values()]
  }

  /** Read scope of the thread: the intersection over everyone who can see it, inside the agent ceiling. */
  scopeReader(thread: ThreadRecord, live: UserPrincipal): ScopeReader {
    const now = this.clock()
    const readers = this.readers(thread, live).map(principal => ({ principal, grants: this.options.identity.grantsFor(principal.id) }))
    const scopes = (target: AccessTarget): ReadonlySet<HubScope> => {
      let shared: Set<HubScope> | undefined
      for (const reader of readers) {
        const own = delegatedScopes(reader.principal, reader.grants, target, now)
        shared = shared === undefined ? new Set(own) : new Set([...shared].filter(scope => own.has(scope)))
      }
      return shared ?? new Set()
    }
    return { scopes, canRead: target => scopes(target).has('session.read') }
  }

  // ── messages ───────────────────────────────────────────────────────

  append(input: {
    readonly threadId: string
    readonly role: MessageRole
    readonly text: string
    readonly author?: { readonly id: string; readonly name: string; readonly deviceId?: string }
    readonly source?: MessageSource
    readonly clientMessageId?: string
    readonly causalMessageId?: string
    readonly audience?: readonly string[]
  }): { readonly message: ThreadMessage; readonly duplicate: boolean } {
    const authorId = input.author?.id
    if (input.clientMessageId !== undefined && authorId !== undefined) {
      const existing = this.db.query<MessageRow, [string, string, string]>(
        'SELECT * FROM hub_supervisor_messages WHERE thread_id = ? AND author_id = ? AND client_message_id = ?',
      ).get(input.threadId, authorId, input.clientMessageId)
      if (existing) return { message: toMessage(existing), duplicate: true }
    }
    // The client's own id keeps its optimistic copy and the stored one the same message.
    const claimed = input.clientMessageId !== undefined
      && this.db.query<{ n: number }, [string]>('SELECT COUNT(*) AS n FROM hub_supervisor_messages WHERE id = ?').get(input.clientMessageId)?.n === 0
    const id = claimed ? input.clientMessageId! : randomUUID()
    this.db.query(`INSERT INTO hub_supervisor_messages
      (id, thread_id, role, text, author_id, author_name, device_id, source, client_message_id, causal_message_id, audience, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
      id, input.threadId, input.role, input.text, authorId ?? null, input.author?.name ?? null, input.author?.deviceId ?? null,
      input.source ?? 'text', input.clientMessageId ?? null, input.causalMessageId ?? null,
      input.audience === undefined ? null : JSON.stringify(input.audience), this.clock(),
    )
    const row = this.db.query<MessageRow, [string]>('SELECT * FROM hub_supervisor_messages WHERE id = ?').get(id)!
    return { message: toMessage(row), duplicate: false }
  }

  /** The latest messages one person may read, oldest first. */
  messagesFor(threadId: string, viewerId: string, limit = 100): ThreadMessage[] {
    const rows = this.db.query<MessageRow, [string, number]>('SELECT * FROM hub_supervisor_messages WHERE thread_id = ? ORDER BY seq DESC LIMIT ?')
      .all(threadId, limit * 2)
    return rows.map(toMessage).filter(message => message.audience === undefined || message.audience.includes(viewerId)).slice(0, limit).reverse()
  }

  // ── delivery ───────────────────────────────────────────────────────

  emit(threadId: string, type: SupervisorEventType, payload: Readonly<Record<string, unknown>>, audience?: readonly string[]): void {
    const thread = this.get(threadId)
    if (!thread || !this.sink) return
    const everyone = this.recipients(thread)
    const targets = audience === undefined ? everyone : everyone.filter(id => audience.includes(id))
    if (targets.length === 0) return
    try {
      this.sink.send(targets, type, { thread_id: threadId, ...payload })
    } catch (error) {
      this.options.onError?.(error, 'supervisor event')
    }
  }

  setWorking(threadId: string, on: boolean): void {
    if (on) this.working.add(threadId)
    else this.working.delete(threadId)
    this.emit(threadId, 'supervisor.thread_event', { activity: on ? 'working' : 'idle' })
  }

  isWorking(threadId: string): boolean {
    return this.working.has(threadId)
  }

  onReply(threadId: string, listener: (reply: SupervisorReply) => void): () => void {
    const listeners = this.replyListeners.get(threadId) ?? new Set()
    listeners.add(listener)
    this.replyListeners.set(threadId, listeners)
    return () => { listeners.delete(listener) }
  }

  publishReply(threadId: string, reply: SupervisorReply): void {
    for (const listener of [...(this.replyListeners.get(threadId) ?? [])]) {
      try { listener(reply) } catch (error) { this.options.onError?.(error, 'supervisor reply listener') }
    }
  }
}
