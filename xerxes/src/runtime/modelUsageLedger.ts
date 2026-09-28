// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Tokens spent per model, across every task and subagent this machine runs.
 *
 * Providers rarely say what each model cost: OpenRouter's per-model activity
 * needs a management key, and subscription plans report windows, not models.
 * The runtime already sees every provider round (`usage_update`), so it keeps
 * its own record: one row per round with the model, the profile that served
 * it and the token counts. The Usage view totals these by model and prices
 * them at report time from published prices, so a price the runtime cannot
 * find stays unknown instead of becoming $0.
 *
 * Recording is best effort. A turn must never fail because this bookkeeping
 * could not be written; the failure is reported once on stderr instead.
 * Nothing is recorded until the daemon opens the ledger, so tests and
 * embedded runtimes never write into a user's home.
 */

import { Database } from 'bun:sqlite'
import { mkdirSync } from 'node:fs'
import { dirname } from 'node:path'

export interface ModelUsageRound {
  readonly model: string
  /** Provider profile that served the round, when the caller knows it. */
  readonly profile?: string | undefined
  readonly inputTokens: number
  readonly outputTokens: number
  readonly cacheReadTokens?: number | undefined
  readonly cacheWriteTokens?: number | undefined
  /** Milliseconds since the epoch; defaults to now. */
  readonly at?: number
}

export interface ModelUsageTotal {
  readonly model: string
  readonly profile: string
  readonly rounds: number
  readonly inputTokens: number
  readonly outputTokens: number
  readonly cacheReadTokens: number
  readonly cacheWriteTokens: number
  readonly lastAt: number
}

const whole = (value: number | undefined): number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.round(value) : 0

export class ModelUsageLedger {
  private readonly db: Database

  constructor(path: string) {
    if (path !== ':memory:') mkdirSync(dirname(path), { recursive: true })
    this.db = new Database(path, { create: true })
    this.db.run('PRAGMA journal_mode = WAL')
    this.db.run(`CREATE TABLE IF NOT EXISTS rounds (
      at INTEGER NOT NULL,
      model TEXT NOT NULL,
      profile TEXT NOT NULL DEFAULT '',
      input_tokens INTEGER NOT NULL,
      output_tokens INTEGER NOT NULL,
      cache_read_tokens INTEGER NOT NULL DEFAULT 0,
      cache_write_tokens INTEGER NOT NULL DEFAULT 0
    )`)
    this.db.run('CREATE INDEX IF NOT EXISTS rounds_at ON rounds (at)')
  }

  record(round: ModelUsageRound): void {
    const model = round.model.trim()
    if (!model) return
    const input = whole(round.inputTokens)
    const output = whole(round.outputTokens)
    const cacheRead = whole(round.cacheReadTokens)
    const cacheWrite = whole(round.cacheWriteTokens)
    if (!input && !output && !cacheRead && !cacheWrite) return
    this.db.query('INSERT INTO rounds (at, model, profile, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens) VALUES (?, ?, ?, ?, ?, ?, ?)')
      .run(round.at ?? Date.now(), model, round.profile?.trim() ?? '', input, output, cacheRead, cacheWrite)
  }

  /** Totals per (model, profile) since `since`, most expensive in tokens first. */
  totals(since: number): ModelUsageTotal[] {
    const rows = this.db.query(`SELECT model, profile, COUNT(*) AS rounds,
        SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,
        SUM(cache_read_tokens) AS cache_read_tokens, SUM(cache_write_tokens) AS cache_write_tokens,
        MAX(at) AS last_at
      FROM rounds WHERE at >= ? GROUP BY model, profile
      ORDER BY SUM(input_tokens + output_tokens + cache_read_tokens + cache_write_tokens) DESC`)
      .all(since) as Array<Record<string, number | string>>
    return rows.map(row => ({
      model: String(row.model),
      profile: String(row.profile ?? ''),
      rounds: Number(row.rounds),
      inputTokens: Number(row.input_tokens),
      outputTokens: Number(row.output_tokens),
      cacheReadTokens: Number(row.cache_read_tokens),
      cacheWriteTokens: Number(row.cache_write_tokens),
      lastAt: Number(row.last_at),
    }))
  }

  close(): void {
    this.db.close()
  }
}

let shared: ModelUsageLedger | null = null
let warned = false

/** Open the process-wide ledger (the daemon does, at startup); null when it cannot be opened. */
export function openModelUsageLedger(path: string): ModelUsageLedger | null {
  shared?.close()
  try {
    shared = new ModelUsageLedger(path)
  } catch (error) {
    shared = null
    warnOnce(error)
  }
  return shared
}

/** The ledger the daemon opened, if any. */
export function modelUsageLedger(): ModelUsageLedger | null {
  return shared
}

export function closeModelUsageLedger(): void {
  shared?.close()
  shared = null
}

/** Record one provider round without ever failing the caller; a no-op until a ledger is open. */
export function recordModelUsage(round: ModelUsageRound): void {
  if (!shared) return
  try {
    shared.record(round)
  } catch (error) {
    warnOnce(error)
  }
}

function warnOnce(error: unknown): void {
  if (warned) return
  warned = true
  console.warn(`Per-model usage is not being recorded: ${error instanceof Error ? error.message : String(error)}`)
}
