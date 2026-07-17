// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { randomUUID } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve, sep } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'
import { migrateSessionRecord } from './migrations.js'
import {
  CURRENT_SESSION_SCHEMA_VERSION,
  SessionRecord,
  type AgentTransitionRecord,
  type SessionRecordData,
  type TurnRecord,
} from './models.js'
import { linearSessionSearch, SessionIndex, type SearchHit, type SearchOptions } from './search.js'
import type { Embedder } from '../memory/embedders.js'

/** Common synchronous persistence API for durable and in-memory session collections. */
export interface SessionStore {
  deleteSession(sessionId: string): boolean
  listSessions(workspaceId?: string): string[]
  loadSession(sessionId: string): SessionRecord | undefined
  saveSession(session: SessionRecord): void
  search(query: string, options?: SearchOptions): SearchHit[]
}

/** Process-local session store for tests and ephemeral runs. */
export class InMemorySessionStore implements SessionStore {
  private readonly sessions = new Map<string, SessionRecord>()

  deleteSession(sessionId: string): boolean {
    return this.sessions.delete(sessionId)
  }

  listSessions(workspaceId?: string): string[] {
    return [...this.sessions.values()]
      .filter(session => workspaceId === undefined || session.workspaceId === workspaceId)
      .map(session => session.sessionId)
  }

  loadSession(sessionId: string): SessionRecord | undefined {
    return this.sessions.get(sessionId)
  }

  saveSession(session: SessionRecord): void {
    this.sessions.set(session.sessionId, session)
  }

  search(query: string, options: SearchOptions = {}): SearchHit[] {
    return linearSessionSearch(this.sessions.values(), query, options)
  }
}

export interface FileSessionStoreOptions {
  readonly baseDirectory: string
  readonly schemaVersion?: number
}

/**
 * JSON session store retained for transparent on-disk records and simple
 * migrations. New runtimes should prefer {@link SQLiteSessionStore} for
 * atomic indexed persistence.
 */
export class FileSessionStore implements SessionStore {
  readonly baseDirectory: string
  readonly schemaVersion: number

  constructor(baseDirectoryOrOptions: string | FileSessionStoreOptions) {
    const options = typeof baseDirectoryOrOptions === 'string'
      ? { baseDirectory: baseDirectoryOrOptions }
      : baseDirectoryOrOptions
    this.baseDirectory = resolve(options.baseDirectory)
    this.schemaVersion = options.schemaVersion ?? CURRENT_SESSION_SCHEMA_VERSION
    mkdirSync(this.baseDirectory, { recursive: true })
  }

  deleteSession(sessionId: string): boolean {
    const path = this.findSessionPath(sessionId)
    if (!path) return false
    rmSync(path, { force: true })
    return true
  }

  listSessions(workspaceId?: string): string[] {
    const directory = workspaceId === undefined ? undefined : this.workspaceDirectory(workspaceId)
    if (directory !== undefined) return sessionIdsInDirectory(directory)

    const ids = sessionIdsInDirectory(this.baseDirectory)
    for (const entry of readdirSync(this.baseDirectory, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue
      ids.push(...sessionIdsInDirectory(join(this.baseDirectory, entry.name)))
    }
    return ids
  }

  loadSession(sessionId: string): SessionRecord | undefined {
    const path = this.findSessionPath(sessionId)
    if (!path) return undefined
    const raw = parseSessionRecordText(readFileSync(path, 'utf8'))
    const needsMigration = recordVersion(raw) < this.schemaVersion
    const record = this.migrateIfNeeded(raw)
    const session = SessionRecord.fromRecord(record)
    if (needsMigration) this.writeRecord(this.pathFor(session), record)
    return session
  }

  saveSession(session: SessionRecord): void {
    this.writeRecord(this.pathFor(session), session.toRecord())
  }

  search(query: string, options: SearchOptions = {}): SearchHit[] {
    return linearSessionSearch(this.listSessionRecords(), query, options)
  }

  listSessionRecords(workspaceId?: string): SessionRecord[] {
    return this.listSessions(workspaceId).flatMap(sessionId => {
      const session = this.tryLoadSession(sessionId)
      return session ? [session] : []
    })
  }

  /** Bulk-path load that skips (and reports) corrupt records instead of poisoning the whole batch. */
  private tryLoadSession(sessionId: string): SessionRecord | undefined {
    try {
      return this.loadSession(sessionId)
    } catch (error) {
      console.warn(`Skipping corrupt session record ${sessionId}:`, error)
      return undefined
    }
  }

  private findSessionPath(sessionId: string): string | undefined {
    const safeId = safeSegment(sessionId, 'session_id')
    const flat = join(this.baseDirectory, `${safeId}.json`)
    if (existsSync(flat)) return flat
    for (const entry of readdirSync(this.baseDirectory, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue
      const candidate = join(this.baseDirectory, entry.name, `${safeId}.json`)
      if (existsSync(candidate)) return candidate
    }
    return undefined
  }

  private migrateIfNeeded(record: SessionRecordData): SessionRecordData {
    const version = recordVersion(record)
    if (version >= this.schemaVersion) return record
    return migrateSessionRecord(record, this.schemaVersion)
  }

  private pathFor(session: SessionRecord): string {
    const sessionId = safeSegment(session.sessionId, 'session_id')
    return session.workspaceId === null
      ? join(this.baseDirectory, `${sessionId}.json`)
      : join(this.workspaceDirectory(session.workspaceId), `${sessionId}.json`)
  }

  private workspaceDirectory(workspaceId: string): string {
    return join(this.baseDirectory, safeSegment(workspaceId, 'workspace_id'))
  }

  private writeRecord(path: string, record: SessionRecordData): void {
    mkdirSync(dirname(path), { recursive: true })
    const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`
    try {
      writeFileSync(temporary, JSON.stringify(record, null, 2), 'utf8')
      renameSync(temporary, path)
    } catch (error) {
      rmSync(temporary, { force: true })
      throw error
    }
  }
}

export interface SQLiteSessionStoreOptions {
  readonly dbPath?: string
  readonly embedder?: Embedder
  readonly schemaVersion?: number
}

/**
 * Bun-native durable session store backed by SQLite and a coupled FTS index.
 *
 * Every row retains the full JSON-compatible session record so unknown fields
 * survive migrations, while indexed columns make listing and workspace scopes
 * inexpensive. Turns are indexed immediately on save for history search.
 */
export class SQLiteSessionStore implements SessionStore {
  readonly dbPath: string
  readonly schemaVersion: number
  private readonly database: Database
  private readonly index: SessionIndex

  constructor(options: SQLiteSessionStoreOptions = {}) {
    this.dbPath = options.dbPath ?? join(xerxesHome(), 'sessions', 'sessions.db')
    this.schemaVersion = options.schemaVersion ?? CURRENT_SESSION_SCHEMA_VERSION
    if (this.dbPath !== ':memory:') mkdirSync(dirname(this.dbPath), { recursive: true })
    this.database = new Database(this.dbPath)
    try {
      this.database.run('PRAGMA journal_mode = WAL')
      this.database.run(`
        CREATE TABLE IF NOT EXISTS sessions (
          session_id TEXT PRIMARY KEY,
          workspace_id TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          agent_id TEXT,
          parent_session_id TEXT,
          schema_version INTEGER NOT NULL,
          metadata TEXT NOT NULL,
          record TEXT NOT NULL
        )
      `)
      this.database.run('CREATE INDEX IF NOT EXISTS sessions_workspace_updated ON sessions(workspace_id, updated_at DESC)')
      this.database.run('CREATE INDEX IF NOT EXISTS sessions_updated ON sessions(updated_at DESC)')
      // The index shares this connection so session row writes and turn
      // re-indexing commit inside one transaction.
      this.index = new SessionIndex({
        database: this.database,
        ...(options.embedder === undefined ? {} : { embedder: options.embedder }),
      })
    } catch (error) {
      this.database.close()
      throw error
    }
  }

  close(): void {
    this.index.close()
    this.database.close()
  }

  deleteSession(sessionId: string): boolean {
    let deleted = 0
    this.index.transaction(() => {
      deleted = this.database.query('DELETE FROM sessions WHERE session_id = ?').run(sessionId).changes
      this.index.removeSession(sessionId)
    })
    return deleted > 0
  }

  listSessions(workspaceId?: string): string[] {
    const statement = workspaceId === undefined
      ? this.database.query('SELECT session_id FROM sessions ORDER BY updated_at DESC, session_id ASC')
      : this.database.query('SELECT session_id FROM sessions WHERE workspace_id = ? ORDER BY updated_at DESC, session_id ASC')
    const rows = (workspaceId === undefined ? statement.all() : statement.all(workspaceId)) as unknown as Array<{ session_id?: unknown }>
    return rows.flatMap(row => typeof row.session_id === 'string' ? [row.session_id] : [])
  }

  loadSession(sessionId: string): SessionRecord | undefined {
    const row = this.database.query('SELECT record FROM sessions WHERE session_id = ?').get(sessionId) as { record?: unknown } | null
    if (!row || typeof row.record !== 'string') return undefined
    const raw = parseSessionRecordText(row.record)
    const needsMigration = recordVersion(raw) < this.schemaVersion
    const record = this.migrateIfNeeded(raw)
    const session = SessionRecord.fromRecord(record)
    if (needsMigration) this.saveSession(session)
    return session
  }

  saveSession(session: SessionRecord): void {
    const record = session.toRecord()
    this.index.transaction(() => {
      this.database.query(`
        INSERT INTO sessions
          (session_id, workspace_id, created_at, updated_at, agent_id, parent_session_id, schema_version, metadata, record)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
          workspace_id = excluded.workspace_id,
          created_at = excluded.created_at,
          updated_at = excluded.updated_at,
          agent_id = excluded.agent_id,
          parent_session_id = excluded.parent_session_id,
          schema_version = excluded.schema_version,
          metadata = excluded.metadata,
          record = excluded.record
      `).run(
        session.sessionId,
        session.workspaceId,
        session.createdAt,
        session.updatedAt,
        session.agentId,
        session.parentSessionId,
        session.schemaVersion,
        JSON.stringify(session.metadata),
        JSON.stringify(record),
      )
      this.index.indexSessionIncremental(session)
    })
  }

  search(query: string, options: SearchOptions = {}): SearchHit[] {
    const indexed = this.index.search(query, options)
    if (indexed.length > 0 || query.trim().length === 0) return indexed
    return linearSessionSearch(this.listSessionRecords(), query, options)
  }

  listSessionRecords(workspaceId?: string): SessionRecord[] {
    return this.listSessions(workspaceId).flatMap(sessionId => {
      const session = this.tryLoadSession(sessionId)
      return session ? [session] : []
    })
  }

  rebuildSearchIndex(): number {
    let turns = 0
    for (const session of this.listSessionRecords()) turns += this.index.indexSession(session)
    return turns
  }

  /** Bulk-path load that skips (and reports) corrupt records instead of poisoning the whole batch. */
  private tryLoadSession(sessionId: string): SessionRecord | undefined {
    try {
      return this.loadSession(sessionId)
    } catch (error) {
      console.warn(`Skipping corrupt session record ${sessionId}:`, error)
      return undefined
    }
  }

  private migrateIfNeeded(record: SessionRecordData): SessionRecordData {
    const version = recordVersion(record)
    if (version >= this.schemaVersion) return record
    return migrateSessionRecord(record, this.schemaVersion)
  }
}

/** Lifecycle helper that owns timestamps, identifiers, and durable mutations. */
export class SessionManager {
  constructor(readonly store: SessionStore) {}

  endSession(sessionId: string): void {
    const session = this.requiredSession(sessionId)
    session.updatedAt = timestamp()
    session.metadata.ended = true
    this.store.saveSession(session)
  }

  getSession(sessionId: string): SessionRecord | undefined {
    return this.store.loadSession(sessionId)
  }

  listSessions(workspaceId?: string): string[] {
    return this.store.listSessions(workspaceId)
  }

  recordAgentTransition(sessionId: string, transition: AgentTransitionRecord): void {
    const session = this.requiredSession(sessionId)
    session.agentTransitions.push(transition)
    session.updatedAt = timestamp()
    this.store.saveSession(session)
  }

  recordTurn(sessionId: string, turn: TurnRecord): void {
    const session = this.requiredSession(sessionId)
    session.turns.push(turn)
    session.updatedAt = timestamp()
    this.store.saveSession(session)
  }

  startSession(options: StartSessionOptions = {}): SessionRecord {
    const now = timestamp()
    const session = new SessionRecord({
      sessionId: options.sessionId ?? compactUuid(),
      ...(options.workspaceId === undefined ? {} : { workspaceId: options.workspaceId }),
      ...(options.agentId === undefined ? {} : { agentId: options.agentId }),
      createdAt: now,
      updatedAt: now,
      metadata: { ...options.metadata },
    })
    this.store.saveSession(session)
    return session
  }

  private requiredSession(sessionId: string): SessionRecord {
    const session = this.store.loadSession(sessionId)
    if (!session) throw new Error(`Session not found: ${sessionId}`)
    return session
  }
}

export interface StartSessionOptions {
  readonly agentId?: string | null
  readonly metadata?: Record<string, unknown>
  readonly sessionId?: string
  readonly workspaceId?: string | null
}

function compactUuid(): string {
  return randomUUID().replaceAll('-', '')
}

function parseSessionRecordText(text: string): SessionRecordData {
  const parsed = JSON.parse(text) as unknown
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('Stored session record must be a JSON object')
  }
  return parsed as SessionRecordData
}

function recordVersion(record: SessionRecordData): number {
  return typeof record.schema_version === 'number' && Number.isInteger(record.schema_version) && record.schema_version >= 1
    ? record.schema_version
    : 1
}

function safeSegment(value: string, field: string): string {
  if (value.length === 0 || value === '.' || value === '..' || value.includes('/') || value.includes('\\') || value.includes(sep)) {
    throw new Error(`${field} must be a single path segment`)
  }
  return value
}

function sessionIdsInDirectory(directory: string): string[] {
  if (!existsSync(directory) || !statSync(directory).isDirectory()) return []
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    if (!entry.isFile() || !entry.name.endsWith('.json')) return []
    return [entry.name.slice(0, -'.json'.length)]
  })
}

function timestamp(): string {
  return new Date().toISOString()
}
