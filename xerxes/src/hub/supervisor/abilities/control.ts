// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomUUID } from 'node:crypto'
import type { Database } from 'bun:sqlite'
import { isRecord } from '../../protocol.js'
import type { SupervisorModelPort } from '../runner.js'

export type ControlMetric = 'actions' | 'wakes' | 'modelCalls' | 'tokens' | 'usd'
export type ControlKind = 'command' | 'edit' | 'risk' | 'wake' | 'model'
export type ControlLimits = Readonly<Partial<Record<ControlMetric, number>>>
export interface ControlCharge { readonly tokens?: number; readonly usd?: number }
export interface ControlState { readonly revision: number; readonly paused: boolean; readonly killed: boolean; readonly limits: ControlLimits }
export interface ControlUsage { readonly actions: number; readonly wakes: number; readonly modelCalls: number; readonly tokens: number | null; readonly usd: number | null }
export interface ControlAudit { readonly kind: 'configure' | 'reserve'; readonly hash: string; readonly ownerId?: string }
export interface ControlOptions {
  readonly db: Database
  readonly clock?: () => number
  readonly isOwner: (id: string) => boolean
  readonly audit: (event: ControlAudit) => Promise<void>
}
interface StateRow { revision: number; paused: number; killed: number; limits: string }
interface ChargeRow { id: string; day: string; kind: ControlKind; tokens: number | null; usd: number | null; settled: number }
const metrics: readonly ControlMetric[] = ['actions', 'wakes', 'modelCalls', 'tokens', 'usd']
function metric(value: string): value is ControlMetric { return metrics.some(item => item === value) }
function amount(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || (name !== 'usd' && !Number.isSafeInteger(value))) throw new Error(`Invalid ${name} budget`)
  return value
}
function limits(input: unknown): ControlLimits {
  if (!isRecord(input)) throw new Error('Invalid daily limits')
  const result: Partial<Record<ControlMetric, number>> = {}
  for (const [key, value] of Object.entries(input)) {
    if (!metric(key)) throw new Error('Unknown daily limit')
    result[key] = amount(value, key)
  }
  return result
}
function digest(input: unknown): string { return createHash('sha256').update(JSON.stringify(input)).digest('hex') }
function quote(input: unknown): ControlCharge {
  if (!isRecord(input) || Object.keys(input).some(key => key !== 'tokens' && key !== 'usd')) throw new Error('Invalid usage quote')
  return { ...(input.tokens === undefined ? {} : { tokens: amount(input.tokens, 'tokens') }), ...(input.usd === undefined ? {} : { usd: amount(input.usd, 'usd') }) }
}

/** Durable admission ledger. Pending reservations are conservative charges, not measured billing.
 * Expiry/restart never refunds an uncertain operation. Owner control is not a model tool.
 */
export class SuperagentControl {
  private controller = new AbortController()
  private disposed = false
  private readonly clock: () => number
  constructor(private readonly options: ControlOptions) {
    this.clock = options.clock ?? Date.now
    options.db.exec(`CREATE TABLE IF NOT EXISTS hub_superagent_control (
      singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL, paused INTEGER NOT NULL, killed INTEGER NOT NULL, limits TEXT NOT NULL);
      INSERT OR IGNORE INTO hub_superagent_control VALUES(1,0,0,0,'{}');
      CREATE TABLE IF NOT EXISTS hub_superagent_charges (
      id TEXT PRIMARY KEY, day TEXT NOT NULL, kind TEXT NOT NULL, tokens REAL, usd REAL, settled INTEGER NOT NULL DEFAULT 0);
      CREATE INDEX IF NOT EXISTS hub_superagent_charges_day ON hub_superagent_charges(day);`)
    if (this.status().killed) this.controller.abort(new Error('Superagent is killed'))
  }
  get signal(): AbortSignal { if (!this.disposed && this.status().killed && !this.controller.signal.aborted) this.controller.abort(new Error('Superagent is killed')); return this.controller.signal }
  dispose(): void { this.disposed = true; this.controller.abort(new Error('Superagent control disposed')) }
  /** Wraps every model path (including digest calls) in the same durable budget and kill fence. */
  model(port: SupervisorModelPort): SupervisorModelPort {
    return { get id() { return port.id }, get reportsCost() { return port.reportsCost },
      get captures() { return port.captures },
      ...(port.quote === undefined ? {} : { quote: request => port.quote!(request) }),
      complete: async request => {
        const signal = request.signal === undefined ? this.signal : AbortSignal.any([request.signal, this.signal])
        signal.throwIfAborted()
        const reservation = await this.reserve('model', `model-${randomUUID()}`, port.quote?.(request))
        signal.throwIfAborted()
        this.assertActive()
        const response = await port.complete({ ...request, signal })
        if (!this.disposed) {
          const usage = response.usage
          this.settle(reservation.id, usage === undefined ? {} : { tokens: usage.inputTokens + usage.outputTokens,
            ...(usage.costUsd === undefined ? {} : { usd: usage.costUsd }) })
        }
        signal.throwIfAborted()
        this.assertActive()
        return response
      },
    }
  }
  status(): ControlState {
    if (this.disposed) throw new Error('Superagent control disposed')
    const row = this.options.db.query<StateRow, []>('SELECT revision,paused,killed,limits FROM hub_superagent_control WHERE singleton=1').get()
    if (!row) throw new Error('Control state unavailable')
    const parsed: unknown = JSON.parse(row.limits)
    return { revision: row.revision, paused: row.paused !== 0, killed: row.killed !== 0, limits: limits(parsed) }
  }
  private day(): string { return new Date(this.clock()).toISOString().slice(0, 10) }
  usage(day = this.day()): ControlUsage {
    const row = this.options.db.query<ControlUsage, [string]>(`SELECT
      COALESCE(SUM(CASE WHEN kind NOT IN ('wake','model') THEN 1 ELSE 0 END),0) AS actions,
      COALESCE(SUM(CASE WHEN kind='wake' THEN 1 ELSE 0 END),0) AS wakes,
      COALESCE(SUM(CASE WHEN kind='model' THEN 1 ELSE 0 END),0) AS modelCalls,
      CASE WHEN COUNT(*)=COUNT(tokens) THEN COALESCE(SUM(tokens),0) ELSE NULL END AS tokens,
      CASE WHEN COUNT(*)=COUNT(usd) THEN COALESCE(SUM(usd),0) ELSE NULL END AS usd
      FROM hub_superagent_charges WHERE day=?`).get(day)
    if (!row) throw new Error('Usage ledger unavailable')
    return row
  }
  async configure(ownerId: string, input: unknown): Promise<ControlState> {
    if (!this.options.isOwner(ownerId)) throw new Error('Only the current Hub owner can change control settings')
    if (!isRecord(input) || Object.keys(input).some(key => !['revision', 'paused', 'killed', 'limits'].includes(key)) || !Number.isSafeInteger(input.revision) || typeof input.revision !== 'number' || input.revision < 0) throw new Error('Invalid control settings')
    for (const key of ['paused', 'killed']) if (input[key] !== undefined && typeof input[key] !== 'boolean') throw new Error('Invalid control flag')
    const revision = input.revision
    const paused = input.paused
    const killed = input.killed
    const changedLimits = input.limits === undefined ? undefined : limits(input.limits)
    await this.options.audit({ kind: 'configure', ownerId, hash: digest({ ownerId, revision, paused, killed, limits: changedLimits }) })
    if (!this.options.isOwner(ownerId)) throw new Error('Hub ownership changed')
    const state = this.options.db.transaction(() => {
      const current = this.status()
      if (current.revision !== revision) throw new Error('Stale control revision')
      const next = { revision: revision + 1, paused: killed === true ? true : typeof paused === 'boolean' ? paused : current.paused,
        killed: typeof killed === 'boolean' ? killed : current.killed, limits: changedLimits ?? current.limits }
      this.options.db.query('UPDATE hub_superagent_control SET revision=?,paused=?,killed=?,limits=? WHERE singleton=1 AND revision=?')
        .run(next.revision, next.paused ? 1 : 0, next.killed ? 1 : 0, JSON.stringify(next.limits), revision)
      return next
    }).immediate()
    if (state.killed) this.controller.abort(new Error('Superagent killed by owner'))
    else if (this.controller.signal.aborted) this.controller = new AbortController()
    return state
  }
  assertActive(): void { const state = this.status(); if (state.paused || state.killed) throw new Error('Superagent admission is paused or killed') }
  /** Audit before admission; synchronously recheck flags, ownership-independent policy and UTC day after await. */
  async reserve(kind: ControlKind, id: string, bound: ControlCharge = {}): Promise<{ readonly id: string; readonly day: string }> {
    if (!['command', 'edit', 'risk', 'wake', 'model'].includes(kind) || typeof id !== 'string' || id.length === 0 || id.length > 200) throw new Error('Invalid reservation')
    const checked = quote(bound)
    this.assertActive()
    await this.options.audit({ kind: 'reserve', hash: digest({ kind, id, bound: checked }) })
    return this.options.db.transaction(() => {
      this.assertActive()
      const state = this.status(); const day = this.day(); const used = this.usage(day)
      if (this.options.db.query<{ id: string }, [string]>('SELECT id FROM hub_superagent_charges WHERE id=?').get(id)) throw new Error('Reservation already exists; never replay uncertain work')
      const tokens = kind === 'model' ? checked.tokens ?? null : checked.tokens ?? 0
      const usd = kind === 'model' || kind === 'risk' ? checked.usd ?? null : checked.usd ?? 0
      const requested: ControlUsage = { actions: kind === 'wake' || kind === 'model' ? 0 : 1, wakes: kind === 'wake' ? 1 : 0, modelCalls: kind === 'model' ? 1 : 0, tokens, usd }
      for (const key of metrics) {
        const cap = state.limits[key]; const prior = used[key]; const next = requested[key]
        if (cap !== undefined && (prior === null || next === null || prior + next > cap)) throw new Error(`Daily ${key} budget exhausted or unknown`)
      }
      this.options.db.query('INSERT INTO hub_superagent_charges(id,day,kind,tokens,usd) VALUES(?,?,?,?,?)').run(id, day, kind, tokens, usd)
      return { id, day }
    }).immediate()
  }
  /** Settle once. Missing measurements retain the reservation; exceeding a bound preserves actual charges and pauses admission. */
  settle(id: string, actual: ControlCharge): void {
    const checked = quote(actual)
    const overrun = this.options.db.transaction(() => {
      const row = this.options.db.query<ChargeRow, [string]>('SELECT * FROM hub_superagent_charges WHERE id=?').get(id)
      if (!row || row.settled !== 0) throw new Error('Missing or already settled reservation')
      const tokens = checked.tokens ?? row.tokens; const usd = checked.usd ?? row.usd
      const over = (row.tokens !== null && tokens !== null && tokens > row.tokens) || (row.usd !== null && usd !== null && usd > row.usd)
      this.options.db.query('UPDATE hub_superagent_charges SET tokens=?,usd=?,settled=1 WHERE id=? AND settled=0').run(tokens, usd, id)
      if (over) this.options.db.query('UPDATE hub_superagent_control SET paused=1,revision=revision+1 WHERE singleton=1').run()
      return over
    }).immediate()
    if (overrun) throw new Error('Usage exceeded reserved upper bound; admission paused')
  }
}
