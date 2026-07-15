// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { mkdirSync } from 'node:fs'
import { dirname } from 'node:path'

import { cosineSimilarity, type Embedder } from '../memory/embedders.js'
import { type SessionRecord, type TurnRecord } from './models.js'

export interface SearchOptions {
  readonly agentId?: string
  readonly k?: number
  readonly sessionId?: string
}

/** A ranked turn match suitable for history and session-search UIs. */
export interface SearchHit {
  readonly agentId: string | null
  readonly bm25Score: number
  readonly metadata: Readonly<Record<string, unknown>>
  readonly prompt: string
  readonly response: string
  readonly score: number
  readonly semanticScore: number
  readonly sessionId: string
  readonly timestamp: string
  readonly turnId: string
}

export interface FtsSearchResult {
  readonly agentId: string
  readonly content: string
  readonly rank: number
  readonly sessionId: string
  readonly turnId: string
}

/**
 * SQLite FTS5 index over prompt and response text.
 *
 * It deliberately has no dependency on a session store. A caller can index a
 * durable store's records after restore, or use it as the lightweight search
 * companion for a JSON-backed session collection.
 */
export class SessionFTSIndex {
  readonly dbPath: string
  readonly ftsAvailable: boolean
  private readonly database: Database

  constructor(dbPath = ':memory:') {
    this.dbPath = dbPath
    if (dbPath !== ':memory:') mkdirSync(dirname(dbPath), { recursive: true })
    this.database = new Database(dbPath)
    this.ftsAvailable = this.initialize()
  }

  close(): void {
    this.database.close()
  }

  deleteSession(sessionId: string): void {
    if (!this.ftsAvailable) return
    this.database.query('DELETE FROM session_fts WHERE session_id = ?').run(sessionId)
  }

  indexSession(session: SessionRecord): void {
    if (!this.ftsAvailable) return
    this.transaction(() => {
      this.database.query('DELETE FROM session_fts WHERE session_id = ?').run(session.sessionId)
      const insert = this.database.query(
        'INSERT INTO session_fts (session_id, turn_id, agent_id, content) VALUES (?, ?, ?, ?)',
      )
      for (const turn of session.turns) {
        const content = `${turn.prompt}\n${turn.responseContent ?? ''}`.trim()
        if (content.length === 0) continue
        insert.run(session.sessionId, turn.turnId, turn.agentId ?? '', content)
      }
    })
  }

  search(query: string, options: SearchOptions = {}): FtsSearchResult[] {
    if (!this.ftsAvailable || query.trim().length === 0) return []
    const k = positiveLimit(options.k)
    const filters: string[] = []
    const parameters: Array<string | number> = [query]
    if (options.agentId !== undefined) {
      filters.push('agent_id = ?')
      parameters.push(options.agentId)
    }
    if (options.sessionId !== undefined) {
      filters.push('session_id = ?')
      parameters.push(options.sessionId)
    }
    parameters.push(k)
    const where = filters.length === 0 ? '' : ` AND ${filters.join(' AND ')}`
    try {
      const rows = this.database.query(
        `SELECT session_id, turn_id, agent_id, content, rank
         FROM session_fts
         WHERE session_fts MATCH ?${where}
         ORDER BY rank LIMIT ?`,
      ).all(...parameters) as unknown as FtsRow[]
      return rows.map(row => ftsResult(row))
    } catch {
      return this.searchLike(query, options, k)
    }
  }

  private initialize(): boolean {
    try {
      this.database.run(`
        CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5(
          session_id UNINDEXED,
          turn_id UNINDEXED,
          agent_id UNINDEXED,
          content
        )
      `)
      return true
    } catch {
      return false
    }
  }

  private searchLike(query: string, options: SearchOptions, k: number): FtsSearchResult[] {
    const filters = ['content LIKE ?']
    const parameters: Array<string | number> = [`%${query}%`]
    if (options.agentId !== undefined) {
      filters.push('agent_id = ?')
      parameters.push(options.agentId)
    }
    if (options.sessionId !== undefined) {
      filters.push('session_id = ?')
      parameters.push(options.sessionId)
    }
    parameters.push(k)
    const rows = this.database.query(
      `SELECT session_id, turn_id, agent_id, content, 0 AS rank
       FROM session_fts WHERE ${filters.join(' AND ')} LIMIT ?`,
    ).all(...parameters) as unknown as FtsRow[]
    return rows.map(row => ftsResult(row))
  }

  private transaction(work: () => void): void {
    this.database.run('BEGIN')
    try {
      work()
      this.database.run('COMMIT')
    } catch (error) {
      this.database.run('ROLLBACK')
      throw error
    }
  }
}

export interface SessionIndexOptions {
  readonly dbPath?: string
  readonly embedder?: Embedder
}

/**
 * SQLite session-turn index with FTS5 ranking and optional local embeddings.
 *
 * The implementation intentionally keeps embeddings as JSON rather than
 * requiring a vector extension, so it is portable across Bun installations.
 */
export class SessionIndex {
  readonly dbPath: string
  readonly embedder: Embedder | undefined
  readonly ftsAvailable: boolean
  private readonly database: Database

  constructor(options: SessionIndexOptions = {}) {
    this.dbPath = options.dbPath ?? ':memory:'
    this.embedder = options.embedder
    if (this.dbPath !== ':memory:') mkdirSync(dirname(this.dbPath), { recursive: true })
    this.database = new Database(this.dbPath)
    this.ftsAvailable = this.initialize()
  }

  close(): void {
    this.database.close()
  }

  indexSession(session: SessionRecord): number {
    this.transaction(() => {
      this.removeSessionRows(session.sessionId)
      for (const turn of session.turns) this.insertTurn(session.sessionId, turn)
    })
    return session.turns.length
  }

  indexTurn(sessionId: string, turn: TurnRecord): void {
    this.transaction(() => this.insertTurn(sessionId, turn))
  }

  removeSession(sessionId: string): number {
    const row = this.database.query('SELECT COUNT(*) AS count FROM session_turns WHERE session_id = ?').get(sessionId) as {
      count?: number
    } | null
    this.transaction(() => this.removeSessionRows(sessionId))
    return typeof row?.count === 'number' ? row.count : 0
  }

  search(
    query: string,
    options: SearchOptions & { readonly weights?: readonly [number, number] } = {},
  ): SearchHit[] {
    if (query.trim().length === 0) return []
    const k = positiveLimit(options.k)
    const weights = normalizedWeights(options.weights)
    const rows = this.candidates(query, k * 4, options)
    if (rows.length === 0) return []

    const maxBm25 = Math.max(...rows.map(row => row.bm25), 1)
    let queryEmbedding: number[] | undefined
    if (this.embedder && weights.semantic > 0) {
      try {
        queryEmbedding = this.embedder.embed(query)
      } catch {
        queryEmbedding = undefined
      }
    }

    const hits = rows.map(row => {
      const bm25Score = row.bm25 / maxBm25
      const semanticScore = queryEmbedding && row.embedding
        ? Math.max(0, cosineSimilarity(queryEmbedding, parseEmbedding(row.embedding)))
        : 0
      return {
        sessionId: row.sessionId,
        turnId: row.turnId,
        agentId: row.agentId,
        prompt: row.prompt.slice(0, 500),
        response: row.response.slice(0, 1000),
        score: weights.bm25 * bm25Score + weights.semantic * semanticScore,
        bm25Score,
        semanticScore,
        timestamp: row.startedAt,
        metadata: parseMetadata(row.metadata),
      } satisfies SearchHit
    })
    return hits.sort((left, right) => right.score - left.score).slice(0, k)
  }

  private candidates(query: string, k: number, options: SearchOptions): IndexedTurnRow[] {
    if (this.ftsAvailable) {
      const filters: string[] = []
      const params: Array<string | number> = [query]
      if (options.agentId !== undefined) {
        filters.push('t.agent_id = ?')
        params.push(options.agentId)
      }
      if (options.sessionId !== undefined) {
        filters.push('t.session_id = ?')
        params.push(options.sessionId)
      }
      params.push(k)
      const where = filters.length === 0 ? '' : ` AND ${filters.join(' AND ')}`
      try {
        const rows = this.database.query(
          `SELECT t.session_id, t.turn_id, t.agent_id, t.prompt, t.response, t.started_at, t.metadata, t.embedding,
                  bm25(session_turns_fts) AS rank
           FROM session_turns_fts
           JOIN session_turns t
             ON t.session_id = session_turns_fts.session_id AND t.turn_id = session_turns_fts.turn_id
           WHERE session_turns_fts MATCH ?${where}
           ORDER BY rank LIMIT ?`,
        ).all(...params) as unknown as IndexedFtsRow[]
        if (rows.length > 0) return rows.map(indexedFtsRow)
      } catch {
        // Raw FTS expressions can be malformed. The LIKE path below remains available.
      }
    }
    return this.likeCandidates(query, k, options)
  }

  private initialize(): boolean {
    this.database.run(`
      CREATE TABLE IF NOT EXISTS session_turns (
        session_id TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        agent_id TEXT,
        prompt TEXT NOT NULL,
        response TEXT NOT NULL,
        started_at TEXT NOT NULL,
        metadata TEXT NOT NULL,
        embedding TEXT NOT NULL,
        PRIMARY KEY (session_id, turn_id)
      )
    `)
    try {
      this.database.run(`
        CREATE VIRTUAL TABLE IF NOT EXISTS session_turns_fts USING fts5(
          prompt,
          response,
          session_id UNINDEXED,
          turn_id UNINDEXED,
          tokenize = 'porter unicode61'
        )
      `)
      return true
    } catch {
      return false
    }
  }

  private insertTurn(sessionId: string, turn: TurnRecord): void {
    const prompt = turn.prompt
    const response = turn.responseContent ?? ''
    let embedding = ''
    if (this.embedder && (prompt || response)) {
      try {
        embedding = JSON.stringify(this.embedder.embed(`${prompt}\n${response}`))
      } catch {
        embedding = ''
      }
    }
    this.database.query(`
      INSERT INTO session_turns
        (session_id, turn_id, agent_id, prompt, response, started_at, metadata, embedding)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(session_id, turn_id) DO UPDATE SET
        agent_id = excluded.agent_id,
        prompt = excluded.prompt,
        response = excluded.response,
        started_at = excluded.started_at,
        metadata = excluded.metadata,
        embedding = excluded.embedding
    `).run(
      sessionId,
      turn.turnId,
      turn.agentId,
      prompt,
      response,
      turn.startedAt || new Date().toISOString(),
      JSON.stringify(turn.metadata),
      embedding,
    )
    if (!this.ftsAvailable) return
    this.database.query('DELETE FROM session_turns_fts WHERE session_id = ? AND turn_id = ?').run(sessionId, turn.turnId)
    this.database.query(
      'INSERT INTO session_turns_fts (prompt, response, session_id, turn_id) VALUES (?, ?, ?, ?)',
    ).run(prompt, response, sessionId, turn.turnId)
  }

  private likeCandidates(query: string, k: number, options: SearchOptions): IndexedTurnRow[] {
    const clauses = ['(prompt LIKE ? OR response LIKE ?)']
    const params: Array<string | number> = [`%${query}%`, `%${query}%`]
    if (options.agentId !== undefined) {
      clauses.push('agent_id = ?')
      params.push(options.agentId)
    }
    if (options.sessionId !== undefined) {
      clauses.push('session_id = ?')
      params.push(options.sessionId)
    }
    params.push(k)
    const rows = this.database.query(`
      SELECT session_id, turn_id, agent_id, prompt, response, started_at, metadata, embedding
      FROM session_turns
      WHERE ${clauses.join(' AND ')}
      ORDER BY started_at DESC LIMIT ?
    `).all(...params) as unknown as IndexedRow[]
    return rows.map(row => ({ ...indexedRow(row), bm25: 1 }))
  }

  private removeSessionRows(sessionId: string): void {
    this.database.query('DELETE FROM session_turns WHERE session_id = ?').run(sessionId)
    if (this.ftsAvailable) this.database.query('DELETE FROM session_turns_fts WHERE session_id = ?').run(sessionId)
  }

  private transaction(work: () => void): void {
    this.database.run('BEGIN')
    try {
      work()
      this.database.run('COMMIT')
    } catch (error) {
      this.database.run('ROLLBACK')
      throw error
    }
  }
}

/** Linear, dependency-free fallback shared by in-memory and non-FTS stores. */
export function linearSessionSearch(
  sessions: Iterable<SessionRecord>,
  query: string,
  options: SearchOptions = {},
): SearchHit[] {
  const normalized = query.trim().toLocaleLowerCase()
  if (normalized.length === 0) return []
  const limit = positiveLimit(options.k)
  const hits: SearchHit[] = []
  for (const session of sessions) {
    if (options.sessionId !== undefined && session.sessionId !== options.sessionId) continue
    for (const turn of session.turns) {
      if (options.agentId !== undefined && turn.agentId !== options.agentId) continue
      const blob = `${turn.prompt}\n${turn.responseContent ?? ''}`.toLocaleLowerCase()
      if (!blob.includes(normalized)) continue
      hits.push({
        sessionId: session.sessionId,
        turnId: turn.turnId,
        agentId: turn.agentId,
        prompt: turn.prompt.slice(0, 500),
        response: (turn.responseContent ?? '').slice(0, 1000),
        score: 1,
        bm25Score: 1,
        semanticScore: 0,
        timestamp: turn.startedAt,
        metadata: { ...turn.metadata },
      })
      if (hits.length >= limit) return hits
    }
  }
  return hits
}

interface FtsRow {
  readonly agent_id: unknown
  readonly content: unknown
  readonly rank: unknown
  readonly session_id: unknown
  readonly turn_id: unknown
}

interface IndexedRow {
  readonly agent_id: unknown
  readonly embedding: unknown
  readonly metadata: unknown
  readonly prompt: unknown
  readonly response: unknown
  readonly session_id: unknown
  readonly started_at: unknown
  readonly turn_id: unknown
}

interface IndexedFtsRow extends IndexedRow {
  readonly rank: unknown
}

interface IndexedTurnRow {
  readonly agentId: string | null
  readonly bm25: number
  readonly embedding: string
  readonly metadata: string
  readonly prompt: string
  readonly response: string
  readonly sessionId: string
  readonly startedAt: string
  readonly turnId: string
}

function ftsResult(row: FtsRow): FtsSearchResult {
  return {
    sessionId: stringCell(row.session_id),
    turnId: stringCell(row.turn_id),
    agentId: stringCell(row.agent_id),
    content: stringCell(row.content),
    rank: numberCell(row.rank),
  }
}

function indexedFtsRow(row: IndexedFtsRow): IndexedTurnRow {
  return { ...indexedRow(row), bm25: Math.max(-numberCell(row.rank), 0) }
}

function indexedRow(row: IndexedRow): Omit<IndexedTurnRow, 'bm25'> {
  return {
    sessionId: stringCell(row.session_id),
    turnId: stringCell(row.turn_id),
    agentId: nullableStringCell(row.agent_id),
    prompt: stringCell(row.prompt),
    response: stringCell(row.response),
    startedAt: stringCell(row.started_at),
    metadata: stringCell(row.metadata),
    embedding: stringCell(row.embedding),
  }
}

function normalizedWeights(weights: readonly [number, number] | undefined): { bm25: number; semantic: number } {
  const [rawBm25, rawSemantic] = weights ?? [0.6, 0.4]
  const total = rawBm25 + rawSemantic
  if (total <= 0) return { bm25: 0.5, semantic: 0.5 }
  return { bm25: rawBm25 / total, semantic: rawSemantic / total }
}

function parseEmbedding(value: string): number[] {
  try {
    const parsed = JSON.parse(value) as unknown
    return Array.isArray(parsed) && parsed.every(item => typeof item === 'number') ? parsed : []
  } catch {
    return []
  }
}

function parseMetadata(value: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(value) as unknown
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed) ? parsed as Record<string, unknown> : {}
  } catch {
    return {}
  }
}

function positiveLimit(value: number | undefined): number {
  return value === undefined || !Number.isFinite(value) ? 10 : Math.max(1, Math.floor(value))
}

function nullableStringCell(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function numberCell(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function stringCell(value: unknown): string {
  return typeof value === 'string' ? value : ''
}
