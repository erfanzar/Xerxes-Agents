// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'

import { cursorScopeKey, encodeCursor, type CursorScope } from './cursor.js'
import { HUB_LIMITS, HubProtocolError, isRecord, readId, readIdList, utf8ByteLength, type HubActor, type HubEvent } from './protocol.js'

/** Committed journal data deliberately has no cursor: cursors belong to readers. */
export type StoredHubEvent = Omit<HubEvent, 'cursor'>

export interface UpstreamEvent {
  readonly event_id: string
  readonly sequence: number
  readonly cursor: string
  readonly type: string
  readonly payload: Readonly<Record<string, unknown>>
  readonly session_id?: string
  readonly occurred_at?: string
  readonly actor?: HubActor
}

/** Structural boundary; the daemon transport can supply additional fields. */
export interface UpstreamEventBatch {
  readonly stream_id: string
  readonly events: readonly UpstreamEvent[]
  readonly through_cursor: string
  /** Required for an empty batch that advances over filtered upstream events. */
  readonly through_sequence?: number
}

export interface UpstreamCursor {
  readonly runtime_id: string
  readonly stream_id: string
  readonly last_sequence: number
  readonly last_cursor: string
}

export interface HubEventInput {
  readonly event_id?: string
  readonly workspace_id: string
  readonly session_id?: string
  readonly type: string
  readonly payload: Readonly<Record<string, unknown>>
  readonly occurred_at?: string
  readonly actor?: HubActor
}

export interface HubEventStoreOptions {
  readonly clock?: () => number
  /** Logical retained event bytes, not SQLite/WAL file size. */
  readonly retention?: { readonly maxBytes: number; readonly maxAgeMs: number }
  readonly onObserverError?: (error: unknown) => void
}

export interface HubEventRead {
  readonly events: readonly HubEvent[]
  readonly throughSequence: number
  readonly watermark: number
  readonly oldestSequence: number
  /** UTF-8 bytes of the JSON event array, excluding its transport envelope. */
  readonly bytes: number
  readonly hasMore: boolean
  readonly gap?: 'retention'
}

export class HubEventStoreError extends Error {
  readonly code = 'HUB_EVENT_STORE_UNAVAILABLE' as const
  constructor(operation: string, cause: unknown) {
    super(`Hub event store ${operation} failed`, { cause })
    this.name = 'HubEventStoreError'
  }
}

interface EventRow {
  sequence: number
  event_id: string
  workspace_id: string
  session_id: string | null
  type: string
  payload: string
  occurred_at: string
  actor: string | null
  bytes: number
}

const DEFAULT_RETENTION = { maxBytes: 256 * 1024 * 1024, maxAgeMs: 7 * 24 * 60 * 60 * 1000 }

/** Single synchronous SQLite writer. Notifications happen only after COMMIT. */
export class HubEventStore {
  readonly streamId: string
  private readonly db: Database
  private readonly clock: () => number
  private readonly retention: { readonly maxBytes: number; readonly maxAgeMs: number }
  private readonly observers = new Set<(events: readonly StoredHubEvent[]) => void>()
  private closed = false

  constructor(path: string, private readonly options: HubEventStoreOptions = {}) {
    this.clock = options.clock ?? Date.now
    this.retention = options.retention ?? DEFAULT_RETENTION
    for (const value of [this.retention.maxBytes, this.retention.maxAgeMs]) {
      if (!Number.isFinite(value) || value < 0) throw new RangeError('retention quotas must be finite non-negative numbers')
    }
    let database: Database | undefined
    try {
      database = new Database(path, { create: true, strict: true })
      this.db = database
      this.db.exec(`
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = FULL;
        PRAGMA busy_timeout = 5000;
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS upstream_cursors (
          runtime_id TEXT PRIMARY KEY, stream_id TEXT NOT NULL,
          last_sequence INTEGER NOT NULL, last_cursor TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
          sequence INTEGER PRIMARY KEY, event_id TEXT NOT NULL UNIQUE,
          runtime_id TEXT, upstream_stream_id TEXT, upstream_sequence INTEGER,
          workspace_id TEXT NOT NULL, session_id TEXT, type TEXT NOT NULL,
          payload TEXT NOT NULL, occurred_at TEXT NOT NULL, actor TEXT, bytes INTEGER NOT NULL,
          UNIQUE(runtime_id, upstream_stream_id, upstream_sequence)
        );
        CREATE INDEX IF NOT EXISTS events_workspace_sequence ON events(workspace_id, sequence);
        CREATE INDEX IF NOT EXISTS events_occurred_at ON events(occurred_at);
      `)
      this.db.transaction(() => {
        this.db.query('INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)').run('stream_id', crypto.randomUUID())
        this.db.query('INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)').run('watermark', '0')
        this.db.query('INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)').run('retained_after', '0')
        this.db.query("INSERT OR IGNORE INTO meta (key, value) SELECT 'retained_bytes', CAST(COALESCE(SUM(bytes), 0) AS TEXT) FROM events").run()
      })()
      this.streamId = this.meta('stream_id')
    } catch (cause) {
      database?.close()
      throw new HubEventStoreError('open', cause)
    }
  }

  ingest(runtimeId: string, workspaceId: string, batch: UpstreamEventBatch): { readonly events: readonly StoredHubEvent[]; readonly duplicates: number } {
    readId(runtimeId, 'runtimeId')
    readId(workspaceId, 'workspaceId')
    readId(batch.stream_id, 'stream_id')
    if (typeof batch.through_cursor !== 'string' || batch.through_cursor.length === 0) throw new HubProtocolError('invalid_request', 'through_cursor is required')
    let previousSequence = -1
    for (const event of batch.events) {
      nonnegativeSequence(event.sequence)
      if (event.sequence < previousSequence) throw new HubProtocolError('invalid_request', 'upstream events must be non-decreasing')
      previousSequence = event.sequence
    }
    if (batch.through_sequence !== undefined) {
      nonnegativeSequence(batch.through_sequence)
      if (batch.through_sequence < previousSequence) throw new HubProtocolError('invalid_request', 'through_sequence precedes the batch')
    }
    const result = this.persist('ingest', () => this.db.transaction(() => {
      const previous = this.cursorRow(runtimeId)
      const sameStream = previous?.stream_id === batch.stream_id
      const pruned = this.db.query<{ value: string }, [string]>('SELECT value FROM meta WHERE key = ?')
        .get(prunedKey(runtimeId, batch.stream_id))
      const prunedThrough = pruned === null ? -1 : Number(pruned.value)
      const events: StoredHubEvent[] = []
      let duplicates = 0
      for (const upstream of batch.events) {
        // A resume checkpoint is not a dedupe set: another workspace's batch
        // can contain lower, previously unseen sequence numbers from this runtime.
        if (upstream.sequence <= prunedThrough) { duplicates += 1; continue }
        const input: HubEventInput = { ...upstream, workspace_id: workspaceId }
        const event = this.insert(input, runtimeId, batch.stream_id, upstream.sequence)
        if (event) events.push(event)
        else duplicates += 1
      }
      const lastSequence = batch.through_sequence ?? batch.events.at(-1)?.sequence ?? (sameStream ? previous.last_sequence : 0)
      if (!sameStream || lastSequence >= previous.last_sequence) {
        this.db.query(`INSERT INTO upstream_cursors (runtime_id, stream_id, last_sequence, last_cursor) VALUES (?, ?, ?, ?)
          ON CONFLICT(runtime_id) DO UPDATE SET stream_id = excluded.stream_id,
          last_sequence = excluded.last_sequence, last_cursor = excluded.last_cursor`).run(runtimeId, batch.stream_id, lastSequence, batch.through_cursor)
      }
      this.pruneRows()
      return { events, duplicates }
    })())
    this.notify(result.events)
    return result
  }

  append(input: HubEventInput): StoredHubEvent {
    if (input.type === 'hub.presence') throw new HubProtocolError('invalid_request', 'presence is ephemeral and must not be journaled')
    const event = this.persist('append', () => this.db.transaction(() => {
      const stored = this.insert(input, null, null, null)
      if (!stored) throw new HubProtocolError('conflict', 'event_id is already stored')
      this.pruneRows()
      return stored
    })())
    this.notify([event])
    return event
  }

  /** Read (after, through] with a scope-specific cursor on every returned event. */
  read(afterSequence: number, filter: CursorScope, limit: number = HUB_LIMITS.maxBatchEvents, maxBytes: number = HUB_LIMITS.normalFrameBytes, throughSequence?: number): HubEventRead {
    nonnegativeSequence(afterSequence)
    const workspaceIds = readIdList(filter.workspaceIds, 'workspaceIds', { required: true })
    const sessionIds = filter.sessionIds === undefined ? undefined : readIdList(filter.sessionIds, 'sessionIds')
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > HUB_LIMITS.maxBatchEvents) throw new HubProtocolError('invalid_request', 'invalid event limit')
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 2 || maxBytes > HUB_LIMITS.maxFrameBytes) throw new HubProtocolError('invalid_request', 'invalid event byte limit')
    if (throughSequence !== undefined) nonnegativeSequence(throughSequence)
    return this.persist('read', () => {
      const watermark = Math.min(this.sequenceMeta('watermark'), throughSequence ?? Number.MAX_SAFE_INTEGER)
      const oldestSequence = this.oldestRow()
      if (afterSequence < oldestSequence - 1) {
        return { events: [], throughSequence: afterSequence, watermark, oldestSequence, bytes: 2, hasMore: false, gap: 'retention' }
      }
      const scopeKey = cursorScopeKey(filter)
      if (workspaceIds.length === 0 || sessionIds?.length === 0) {
        return { events: [], throughSequence: Math.max(afterSequence, watermark), watermark, oldestSequence, bytes: 2, hasMore: false }
      }
      const workspaceSql = workspaceIds.map(() => '?').join(',')
      const sessionSql = sessionIds === undefined ? '' : ` AND session_id IN (${sessionIds.map(() => '?').join(',')})`
      const rows = this.db.query<EventRow, (string | number)[]>(`SELECT * FROM events WHERE sequence > ? AND sequence <= ?
        AND workspace_id IN (${workspaceSql})${sessionSql} ORDER BY sequence LIMIT ?`).all(afterSequence, watermark, ...workspaceIds, ...(sessionIds ?? []), limit + 1)
      const events: HubEvent[] = []
      let bytes = 2
      let consumed = 0
      for (const row of rows.slice(0, limit)) {
        let event: HubEvent = { ...fromRow(row), cursor: encodeCursor({ streamId: this.streamId, sequence: row.sequence, scopeKey }) }
        const originalBytes = utf8ByteLength(JSON.stringify(event))
        // A single event must never wedge replay or exceed either page or wire limits.
        if (originalBytes + 2 > maxBytes || originalBytes > HUB_LIMITS.maxFrameBytes) {
          event = { ...event, payload: { truncated: true, original_bytes: originalBytes } }
        }
        const eventBytes = utf8ByteLength(JSON.stringify(event))
        if (eventBytes + 2 > maxBytes) throw new HubProtocolError('frame_too_large', 'event metadata cannot fit the requested byte limit')
        const nextBytes = bytes + eventBytes + (events.length === 0 ? 0 : 1)
        if (nextBytes > maxBytes) break
        events.push(event)
        bytes = nextBytes
        consumed += 1
      }
      const hasMore = rows.length > consumed
      return { events, throughSequence: hasMore ? events.at(-1)!.sequence : Math.max(afterSequence, watermark), watermark, oldestSequence, bytes, hasMore }
    })
  }

  watermark(): number { return this.persist('watermark', () => this.sequenceMeta('watermark')) }
  /** First retained position, or watermark + 1 if the journal is empty. */
  oldestSequence(): number { return this.persist('oldest sequence', () => this.oldestRow()) }
  upstreamCursor(runtimeId: string): UpstreamCursor | undefined { return this.persist('upstream cursor', () => this.cursorRow(runtimeId) ?? undefined) }
  prune(): void { this.persist('prune', () => this.db.transaction(() => this.pruneRows())()) }

  onEvents(observer: (events: readonly StoredHubEvent[]) => void): () => void {
    this.observers.add(observer)
    return () => { this.observers.delete(observer) }
  }

  close(): void {
    if (this.closed) return
    this.db.close()
    this.closed = true
    this.observers.clear()
  }

  private meta(key: string): string {
    const row = this.db.query<{ value: string }, [string]>('SELECT value FROM meta WHERE key = ?').get(key)
    if (!row) throw new Error(`missing Hub event-store metadata: ${key}`)
    return row.value
  }

  private sequenceMeta(key: string): number { return Number(this.meta(key)) }
  private oldestRow(): number {
    return this.db.query<{ sequence: number | null }, []>('SELECT MIN(sequence) AS sequence FROM events').get()?.sequence ?? this.sequenceMeta('watermark') + 1
  }
  private cursorRow(runtimeId: string): UpstreamCursor | null {
    return this.db.query<UpstreamCursor, [string]>('SELECT * FROM upstream_cursors WHERE runtime_id = ?').get(runtimeId)
  }

  private insert(input: HubEventInput, runtimeId: string | null, streamId: string | null, upstreamSequence: number | null): StoredHubEvent | undefined {
    const eventId = readId(input.event_id ?? crypto.randomUUID(), 'event_id')
    readId(input.workspace_id, 'workspace_id')
    if (input.session_id !== undefined) readId(input.session_id, 'session_id')
    readId(input.type, 'type')
    if (input.type === 'hub.presence') throw new HubProtocolError('invalid_request', 'presence is ephemeral and must not be journaled')
    if (!isRecord(input.payload)) throw new HubProtocolError('invalid_request', 'event payload must be an object')
    const time = input.occurred_at === undefined ? this.clock() : Date.parse(input.occurred_at)
    if (!Number.isFinite(time)) throw new HubProtocolError('invalid_request', 'event occurred_at must be a timestamp')
    const sequence = this.sequenceMeta('watermark') + 1
    nonnegativeSequence(sequence)
    const event: StoredHubEvent = {
      event_id: eventId, sequence, workspace_id: input.workspace_id, type: input.type,
      payload: input.payload, occurred_at: new Date(time).toISOString(),
      ...(input.session_id === undefined ? {} : { session_id: input.session_id }),
      ...(input.actor === undefined ? {} : { actor: input.actor }),
    }
    const payload = JSON.stringify(event.payload)
    const actor = event.actor === undefined ? null : JSON.stringify(event.actor)
    const bytes = utf8ByteLength(JSON.stringify(event))
    const inserted = this.db.query(`INSERT INTO events
      (sequence, event_id, runtime_id, upstream_stream_id, upstream_sequence, workspace_id, session_id, type, payload, occurred_at, actor, bytes)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(event_id) DO NOTHING
      ON CONFLICT(runtime_id, upstream_stream_id, upstream_sequence) DO NOTHING`).run(
      sequence, eventId, runtimeId, streamId, upstreamSequence, event.workspace_id, event.session_id ?? null, event.type, payload, event.occurred_at, actor, bytes,
    )
    if (inserted.changes === 0) return undefined
    this.db.query("UPDATE meta SET value = ? WHERE key = 'watermark'").run(String(sequence))
    this.db.query("UPDATE meta SET value = ? WHERE key = 'retained_bytes'").run(String(this.sequenceMeta('retained_bytes') + bytes))
    // Return the same JSON value a later reader observes (no caller-owned mutable payload).
    return { ...event, payload: JSON.parse(payload) as Record<string, unknown>, ...(actor === null ? {} : { actor: JSON.parse(actor) as HubActor }) }
  }

  private pruneRows(): void {
    const age = this.db.query<{ sequence: number | null }, [string]>('SELECT MAX(sequence) AS sequence FROM events WHERE occurred_at < ?')
      .get(new Date(this.clock() - this.retention.maxAgeMs).toISOString())?.sequence ?? 0
    const retainedBytes = this.sequenceMeta('retained_bytes')
    // Avoid a full window scan on every token when the journal is below quota.
    const size = retainedBytes <= this.retention.maxBytes ? 0 : this.db.query<{ sequence: number | null }, [number]>(`SELECT MAX(sequence) AS sequence FROM
      (SELECT sequence, SUM(bytes) OVER (ORDER BY sequence DESC) AS retained_bytes FROM events)
      WHERE retained_bytes > ?`).get(this.retention.maxBytes)?.sequence ?? 0
    const through = Math.max(age, size)
    if (through === 0) return
    const retired = this.db.query<{ runtime_id: string; upstream_stream_id: string; sequence: number }, [number]>(`SELECT runtime_id, upstream_stream_id,
      MAX(upstream_sequence) AS sequence FROM events WHERE sequence <= ? AND runtime_id IS NOT NULL
      GROUP BY runtime_id, upstream_stream_id`).all(through)
    for (const row of retired) {
      this.db.query(`INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE
        SET value = CAST(MAX(CAST(value AS INTEGER), CAST(excluded.value AS INTEGER)) AS TEXT)`)
        .run(prunedKey(row.runtime_id, row.upstream_stream_id), String(row.sequence))
    }
    const removedBytes = this.db.query<{ bytes: number }, [number]>('SELECT COALESCE(SUM(bytes), 0) AS bytes FROM events WHERE sequence <= ?').get(through)!.bytes
    this.db.query('DELETE FROM events WHERE sequence <= ?').run(through)
    this.db.query("UPDATE meta SET value = ? WHERE key = 'retained_after'").run(String(Math.max(through, this.sequenceMeta('retained_after'))))
    this.db.query("UPDATE meta SET value = ? WHERE key = 'retained_bytes'").run(String(retainedBytes - removedBytes))
  }

  private persist<T>(operation: string, run: () => T): T {
    try {
      if (this.closed) throw new Error('store is closed')
      return run()
    } catch (cause) {
      if (cause instanceof HubProtocolError || cause instanceof HubEventStoreError) throw cause
      throw new HubEventStoreError(operation, cause)
    }
  }

  private notify(events: readonly StoredHubEvent[]): void {
    if (events.length === 0) return
    for (const observer of this.observers) {
      try { observer(events) } catch (error) {
        // COMMIT already succeeded; observer failures must never masquerade as failed persistence.
        try {
          if (this.options.onObserverError) this.options.onObserverError(error)
          else console.error('Hub event-store observer failed after commit', error)
        } catch (reportError) { console.error('Hub event-store observer error reporter failed', reportError) }
      }
    }
  }
}

function prunedKey(runtimeId: string, streamId: string): string {
  return `pruned_upstream:${JSON.stringify([runtimeId, streamId])}`
}

function nonnegativeSequence(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) throw new HubProtocolError('invalid_request', 'sequence must be a non-negative safe integer')
}

function fromRow(row: EventRow): StoredHubEvent {
  return {
    event_id: row.event_id, sequence: row.sequence, workspace_id: row.workspace_id, type: row.type,
    payload: JSON.parse(row.payload) as Record<string, unknown>, occurred_at: row.occurred_at,
    ...(row.session_id === null ? {} : { session_id: row.session_id }),
    ...(row.actor === null ? {} : { actor: JSON.parse(row.actor) as HubActor }),
  }
}
