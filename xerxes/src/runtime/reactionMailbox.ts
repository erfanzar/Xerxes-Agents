// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { Database } from 'bun:sqlite'
import { chmodSync, mkdirSync } from 'node:fs'
import { dirname } from 'node:path'
import { processIsAlive } from '../core/processLiveness.js'

export interface ReactionUsage { readonly inputTokens: number; readonly outputTokens: number; readonly complete: boolean }
export interface ReactionPolicyEdit { readonly revision: string; readonly maxReactions: number; readonly maxDurationMs: number; readonly maxTotalTokens: number | null }
export interface ReactionPolicy {
  readonly owner: string
  readonly runId: string
  readonly expiresAt: number
  readonly maxReactions: number
  readonly maxTotalTokens?: number
  readonly maxDurationMs: number
}
export interface ReactionHealth {
  readonly state: 'waiting' | 'queued' | 'running' | 'cancelling' | 'awaiting-cleanup' | 'cancelled' | 'expired' | 'exhausted'
  readonly attempts: number
  readonly maxReactions: number
  readonly pendingEvents: number
  readonly expiresAt: number
  readonly activeClaimId: string | null
  readonly lastOutcome: string | null
  readonly lastError: string | null
  readonly tokenBudget?: { maximum: number; blocked: boolean }
  readonly usage: ReactionUsage
  readonly policy?: ReactionPolicyEdit
}
export interface ReactionClaim {
  readonly id: string
  readonly owner: string
  readonly runId: string
  readonly fromSequence: number
  readonly throughSequence: number
  readonly tokenBudget?: { maximum: number; priorTokens: number; priorComplete: boolean }
  readonly deadline: number
}
export type ReactionOutcome = 'completed' | 'failed' | 'cancelled' | 'interrupted'
interface PolicyRow {
  owner: string; run_id: string; expires: number; max_reactions: number; duration: number
  max_total_tokens: number | null
  offered: number; consumed: number; attempts: number; cancelled: number
}
interface ClaimRow {
  id: string; owner: string; run_id: string; first: number; last: number; deadline: number
  outcome: 'claimed' | ReactionOutcome; error: string | null; executor_pid: number | null
  input_tokens: number | null; output_tokens: number | null
}

/** Durable admission only. The daemon owns turn execution and cancellation. */
export class ReactionMailbox {
  private readonly cancellations = new Set<(owner: string, runId?: string) => void>()
  private readonly db: Database
  constructor(path: string, private readonly now: () => number = Date.now,
    private readonly executor: { pid?: number; isAlive?: (pid: number) => boolean } = {}) {
    if (executor.pid !== undefined && (!Number.isSafeInteger(executor.pid) || executor.pid <= 0)) throw new Error('Invalid executor PID')
    if (path !== ':memory:') mkdirSync(dirname(path), { recursive: true, mode: 0o700 })
    this.db = new Database(path, { create: true, strict: true })
    if (path !== ':memory:') chmodSync(path, 0o600)
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;
      CREATE TABLE IF NOT EXISTS reaction_policies (
        owner TEXT NOT NULL, run_id TEXT NOT NULL, expires INTEGER NOT NULL,
        max_reactions INTEGER NOT NULL, duration INTEGER NOT NULL,
        offered INTEGER NOT NULL DEFAULT 0, consumed INTEGER NOT NULL DEFAULT 0,
        attempts INTEGER NOT NULL DEFAULT 0, cancelled INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(owner,run_id));
      CREATE TABLE IF NOT EXISTS reaction_claims (
        id TEXT PRIMARY KEY, owner TEXT NOT NULL, run_id TEXT NOT NULL,
        first INTEGER NOT NULL, last INTEGER NOT NULL, deadline INTEGER NOT NULL,
        outcome TEXT NOT NULL, error TEXT);
      CREATE UNIQUE INDEX IF NOT EXISTS reaction_owner_claim ON reaction_claims(owner) WHERE outcome='claimed';`)
    this.db.transaction(() => {
      const columns = this.db.query<{ name: string }, []>('PRAGMA table_info(reaction_claims)').all()
      if (!columns.some(column => column.name === 'executor_pid')) this.db.exec('ALTER TABLE reaction_claims ADD COLUMN executor_pid INTEGER')
      for (const name of ['input_tokens', 'output_tokens', 'usage_complete']) {
        if (!columns.some(column => column.name === name)) this.db.exec('ALTER TABLE reaction_claims ADD COLUMN ' + name + ' INTEGER')
      }
    }).immediate()
    this.db.transaction(() => {
      const columns = this.db.query<{ name: string }, []>('PRAGMA table_info(reaction_policies)').all()
      if (!columns.some(column => column.name === 'max_total_tokens')) this.db.exec('ALTER TABLE reaction_policies ADD COLUMN max_total_tokens INTEGER')
    }).immediate()
    this.recoverExitedExecutors()
  }

  /** Never retry uncertain side effects after the executor process disappears. */
  private recoverExitedExecutors(): void {
    const alive = this.executor.isAlive ?? processIsAlive
    this.db.transaction(() => {
      const claims = this.db.query<ClaimRow, []>("SELECT * FROM reaction_claims WHERE outcome='claimed'").all()
      for (const claim of claims) {
        // Legacy claims have no proof of ownership. Live or unreadable PIDs,
        // including recycled PIDs, remain fenced rather than guessed dead.
        if (claim.executor_pid === null || !Number.isSafeInteger(claim.executor_pid) || claim.executor_pid <= 0 || alive(claim.executor_pid)) continue
        this.db.query("UPDATE reaction_claims SET outcome='interrupted',usage_complete=0,error=? WHERE id=? AND outcome='claimed'")
          .run('Executor process exited; effects may be incomplete. Automatic reactions for this session were disabled.', claim.id)
        this.db.query('UPDATE reaction_policies SET cancelled=1 WHERE owner=?').run(claim.owner)
      }
    }).immediate()
  }

  /** Create one authority grant; later limit edits require updatePolicy's guard. */
  configure(policy: ReactionPolicy): void {
    if (!policy.owner.trim() || !policy.runId.trim() || policy.owner.length > 8192 || policy.runId.length > 8192 ||
        !Number.isSafeInteger(policy.expiresAt) || policy.expiresAt <= this.now() || policy.expiresAt > this.now() + 86_400_000 ||
        !Number.isSafeInteger(policy.maxReactions) || policy.maxReactions < 1 || policy.maxReactions > 100 ||
        !Number.isSafeInteger(policy.maxDurationMs) || policy.maxDurationMs < 100 || policy.maxDurationMs > 86_400_000) {
      throw new Error('Invalid reaction policy')
    }
    if (policy.maxTotalTokens !== undefined && (!Number.isSafeInteger(policy.maxTotalTokens) || policy.maxTotalTokens < 1)) throw new Error('Invalid reaction token threshold')
    this.db.transaction(() => {
      // Spent grants are retained as evidence, not active capacity. An
      // unresolved executor still occupies a slot after expiry or cancellation.
      const count = this.db.query<PolicyRow & { active: number }, [string, number]>(
        `SELECT p.*, EXISTS (SELECT 1 FROM reaction_claims c WHERE c.owner=p.owner AND c.run_id=p.run_id AND c.outcome='claimed') AS active FROM reaction_policies p WHERE p.owner=? AND (
          (p.cancelled=0 AND p.expires>? AND p.attempts<p.max_reactions)
          OR EXISTS (SELECT 1 FROM reaction_claims c WHERE c.owner=p.owner AND c.run_id=p.run_id AND c.outcome='claimed')
        )`).all(policy.owner, this.now()).filter(row => row.active || !this.tokenBlocked(row, this.totalUsage(row.owner, row.run_id))).length
      if (count >= 16) throw new Error('Session reaction policy limit reached')
      this.db.query('INSERT INTO reaction_policies(owner,run_id,expires,max_reactions,duration,max_total_tokens) VALUES(?,?,?,?,?,?)')
        .run(policy.owner, policy.runId, policy.expiresAt, policy.maxReactions, policy.maxDurationMs, policy.maxTotalTokens ?? null)
    }).immediate()
  }

  /** Eligible grants whose owner has no unresolved executor; does not mutate cursors. */
  recoverableRuns(owner: string): string[] {
    return this.db.query<{ run_id: string }, [string, number]>(`SELECT run_id FROM reaction_policies
      WHERE owner=? AND cancelled=0 AND expires>? AND attempts<max_reactions
      AND NOT EXISTS (SELECT 1 FROM reaction_claims WHERE owner=reaction_policies.owner AND outcome='claimed')
      ORDER BY expires,run_id`).all(owner, this.now()).map(row => row.run_id)
  }

  /** Edit limits atomically without rearming cancellation or erasing consumed evidence. */
  updatePolicy(owner: string, runId: string, edit: ReactionPolicyEdit): void {
    if (typeof edit.revision !== 'string' || !edit.revision || !Number.isSafeInteger(edit.maxReactions) || edit.maxReactions < 1 || edit.maxReactions > 100
      || !Number.isSafeInteger(edit.maxDurationMs) || edit.maxDurationMs < 100 || edit.maxDurationMs > 86_400_000
      || (edit.maxTotalTokens !== null && (!Number.isSafeInteger(edit.maxTotalTokens) || edit.maxTotalTokens < 1))) throw new Error('Invalid reaction policy edit')
    this.db.transaction(() => {
      const row = this.db.query<PolicyRow, [string, string]>('SELECT * FROM reaction_policies WHERE owner=? AND run_id=?').get(owner, runId)
      if (!row) throw new Error('Unknown reaction policy')
      if (Bun.hash(JSON.stringify(row)).toString(16) !== edit.revision) throw new Error('Reaction policy changed; refresh before editing')
      if (row.cancelled || row.expires <= this.now()) throw new Error('Cancelled or expired reactions require a new watch')
      if (this.db.query("SELECT id FROM reaction_claims WHERE owner=? AND run_id=? AND outcome='claimed'").get(owner, runId)) throw new Error('Wait for the active reaction to settle before editing')
      if (edit.maxReactions < row.attempts) throw new Error('Attempt limit cannot be below attempts already used')
      const updated = { ...row, max_reactions: edit.maxReactions, max_total_tokens: edit.maxTotalTokens }
      if (row.attempts < edit.maxReactions && !this.tokenBlocked(updated, this.totalUsage(owner, runId))) {
        const others = this.db.query<PolicyRow & { active: number }, [string, string, number]>(`SELECT p.*,
          EXISTS (SELECT 1 FROM reaction_claims c WHERE c.owner=p.owner AND c.run_id=p.run_id AND c.outcome='claimed') AS active
          FROM reaction_policies p WHERE owner=? AND run_id<>? AND ((cancelled=0 AND expires>? AND attempts<max_reactions)
          OR EXISTS (SELECT 1 FROM reaction_claims c WHERE c.owner=p.owner AND c.run_id=p.run_id AND c.outcome='claimed'))`)
          .all(owner, runId, this.now())
        if (others.filter(other => other.active || !this.tokenBlocked(other, this.totalUsage(owner, other.run_id))).length >= 16) throw new Error('Session reaction policy limit reached')
      }
      this.db.query('UPDATE reaction_policies SET max_reactions=?,duration=?,max_total_tokens=? WHERE owner=? AND run_id=?')
        .run(edit.maxReactions, edit.maxDurationMs, edit.maxTotalTokens, owner, runId)
    }).immediate()
  }

  /** Offer the highest durable evidence cursor; bursts coalesce before claim. */
  offer(owner: string, runId: string, throughSequence: number): boolean {
    if (!Number.isSafeInteger(throughSequence) || throughSequence < 1 || throughSequence > 1000) throw new Error('Invalid reaction cursor')
    return this.db.query(`UPDATE reaction_policies SET offered=MAX(offered,?)
      WHERE owner=? AND run_id=? AND cancelled=0 AND expires>? AND attempts<max_reactions`)
      .run(throughSequence, owner, runId, this.now()).changes > 0
  }

  /** Call only after acquiring session turn admission. One claim per owner. */
  claim(owner: string): ReactionClaim | undefined {
    return this.db.transaction(() => {
      // An expired claim is uncertain external work, not permission to retry it.
      // Keep it fenced until the executor confirms termination via settle().
      if (this.db.query('SELECT id FROM reaction_claims WHERE owner=? AND outcome=\'claimed\'').get(owner)) return undefined
      const policy = this.db.query<PolicyRow, [string, number]>(`SELECT * FROM reaction_policies
        WHERE owner=? AND cancelled=0 AND expires>? AND attempts<max_reactions AND offered>consumed
        ORDER BY expires,run_id`).all(owner, this.now()).find(row => !this.tokenBlocked(row, this.totalUsage(owner, row.run_id)))
      if (!policy) return undefined
      const usage = this.totalUsage(owner, policy.run_id)
      const claim: ReactionClaim = { id: crypto.randomUUID(), owner, runId: policy.run_id,
        ...(policy.max_total_tokens == null ? {} : { tokenBudget: { maximum: policy.max_total_tokens, priorTokens: usage.inputTokens + usage.outputTokens, priorComplete: usage.complete } }),
        fromSequence: policy.consumed + 1, throughSequence: policy.offered,
        deadline: Math.min(policy.expires, this.now() + policy.duration) }
      this.db.query(`INSERT INTO reaction_claims(id,owner,run_id,first,last,deadline,outcome,executor_pid)
        VALUES(?,?,?,?,?,?,'claimed',?)`).run(claim.id, owner, claim.runId, claim.fromSequence, claim.throughSequence, claim.deadline, this.executor.pid ?? process.pid)
      this.db.query('UPDATE reaction_policies SET consumed=?,attempts=attempts+1 WHERE owner=? AND run_id=?')
        .run(claim.throughSequence, owner, claim.runId)
      return claim
    }).immediate()
  }

  /** Check again immediately before starting a provider or tool operation. */
  isAuthorized(claim: ReactionClaim): boolean {
    const row = this.db.query<{ id: string }, [string, string, string, number]>(`SELECT c.id FROM reaction_claims c
      JOIN reaction_policies p ON c.owner=p.owner AND c.run_id=p.run_id
      WHERE c.id=? AND c.owner=? AND c.run_id=? AND c.outcome='claimed'
        AND p.cancelled=0 AND c.deadline>?`).get(claim.id, claim.owner, claim.runId, this.now())
    return Boolean(row)
  }

  /** Revoke future admission, retaining active claims until execution settles. */
  cancel(owner: string, runId?: string): void {
    if (runId === undefined) this.db.query('UPDATE reaction_policies SET cancelled=1 WHERE owner=?').run(owner)
    else this.db.query('UPDATE reaction_policies SET cancelled=1 WHERE owner=? AND run_id=?').run(owner, runId)
    for (const listener of this.cancellations) listener(owner, runId)
  }

  subscribeCancellation(listener: (owner: string, runId?: string) => void): () => void {
    this.cancellations.add(listener)
    return () => { this.cancellations.delete(listener) }
  }

  settle(claim: ReactionClaim, outcome: ReactionOutcome, error?: string, usage?: ReactionUsage): void {
    if (usage && (!Number.isSafeInteger(usage.inputTokens) || usage.inputTokens < 0 || !Number.isSafeInteger(usage.outputTokens) || usage.outputTokens < 0 || typeof usage.complete !== 'boolean')) throw new Error('Invalid reaction usage')
    if (!['completed', 'failed', 'cancelled', 'interrupted'].includes(outcome)) throw new Error('Invalid reaction outcome')
    this.db.transaction(() => {
    const row = this.db.query<ClaimRow, [string, string, string]>(
      'SELECT * FROM reaction_claims WHERE id=? AND owner=? AND run_id=?').get(claim.id, claim.owner, claim.runId)
    if (!row) throw new Error('Unknown reaction claim')
    if (row.outcome !== 'claimed' && row.outcome !== outcome) throw new Error('Reaction already settled differently')
    if (row.outcome !== 'claimed') return
    if (usage && (usage.inputTokens < (row.input_tokens ?? 0) || usage.outputTokens < (row.output_tokens ?? 0))) throw new Error('Reaction usage cannot decrease')
    this.db.query(`UPDATE reaction_claims SET outcome=?,error=?,input_tokens=?,output_tokens=?,usage_complete=? WHERE id=? AND owner=? AND outcome='claimed'`)
      .run(outcome, error?.slice(0, 8192) ?? null, usage?.inputTokens ?? row.input_tokens, usage?.outputTokens ?? row.output_tokens, usage ? Number(usage.complete) : 0, claim.id, claim.owner)
    }).immediate()
  }

  /** Absolute attempt counters; repeated checkpoints must never double-charge. */
  checkpointUsage(claim: ReactionClaim, usage: ReactionUsage): void {
    if (!Number.isSafeInteger(usage.inputTokens) || usage.inputTokens < 0 || !Number.isSafeInteger(usage.outputTokens) || usage.outputTokens < 0 || typeof usage.complete !== 'boolean') throw new Error('Invalid reaction usage')
    const result = this.db.query(`UPDATE reaction_claims SET input_tokens=?,output_tokens=?,usage_complete=0
      WHERE id=? AND owner=? AND run_id=? AND first=? AND last=? AND deadline=? AND outcome='claimed'
        AND COALESCE(input_tokens,0)<=? AND COALESCE(output_tokens,0)<=?`)
      .run(usage.inputTokens, usage.outputTokens, claim.id, claim.owner, claim.runId, claim.fromSequence, claim.throughSequence, claim.deadline, usage.inputTokens, usage.outputTokens)
    if (!result.changes) throw new Error('Reaction checkpoint is stale, decreasing, or not owned by this claim')
  }

  inspect(owner: string, runId: string): ReactionHealth | undefined {
    return this.db.transaction(() => {
      const policy = this.db.query<PolicyRow, [string, string]>(
        'SELECT * FROM reaction_policies WHERE owner=? AND run_id=?').get(owner, runId)
      if (!policy) return undefined
      const active = this.db.query<ClaimRow, [string, string]>(
        "SELECT * FROM reaction_claims WHERE owner=? AND run_id=? AND outcome='claimed'").get(owner, runId)
      const latest = this.db.query<ClaimRow, [string, string]>(
        'SELECT * FROM reaction_claims WHERE owner=? AND run_id=? ORDER BY rowid DESC LIMIT 1').get(owner, runId)
      const usage = this.totalUsage(owner, runId)
      const tokenBlocked = this.tokenBlocked(policy, usage)
      const state: ReactionHealth['state'] = active
        ? policy.cancelled ? 'cancelling' : active.deadline <= this.now() ? 'awaiting-cleanup' : 'running'
        : policy.cancelled ? 'cancelled' : policy.expires <= this.now() ? 'expired'
        : policy.attempts >= policy.max_reactions || tokenBlocked ? 'exhausted'
        : policy.offered > policy.consumed ? 'queued' : 'waiting'
      return { state, attempts: policy.attempts, maxReactions: policy.max_reactions,
        policy: { revision: Bun.hash(JSON.stringify(policy)).toString(16), maxReactions: policy.max_reactions, maxDurationMs: policy.duration, maxTotalTokens: policy.max_total_tokens ?? null },
        pendingEvents: policy.offered - policy.consumed, expiresAt: policy.expires,
        usage, ...(policy.max_total_tokens == null ? {} : { tokenBudget: { maximum: policy.max_total_tokens, blocked: tokenBlocked } }),
        activeClaimId: active?.id ?? null, lastOutcome: latest?.outcome ?? null, lastError: latest?.error ?? null }
    })()
  }

  private totalUsage(owner: string, runId: string): ReactionUsage {
    const rows = this.db.query<{ input_tokens: number | null; output_tokens: number | null; usage_complete: number | null }, [string, string]>(
      'SELECT input_tokens,output_tokens,usage_complete FROM reaction_claims WHERE owner=? AND run_id=?').all(owner, runId)
    let inputTokens = 0, outputTokens = 0, complete = true
    for (const row of rows) {
      complete &&= row.usage_complete === 1
      inputTokens += row.input_tokens ?? 0
      outputTokens += row.output_tokens ?? 0
      if (!Number.isSafeInteger(inputTokens + outputTokens)) complete = false
      inputTokens = Math.min(Number.MAX_SAFE_INTEGER, inputTokens)
      outputTokens = Math.min(Number.MAX_SAFE_INTEGER, outputTokens)
    }
    return { inputTokens, outputTokens, complete }
  }
  private tokenBlocked(policy: PolicyRow, usage: ReactionUsage): boolean {
    return policy.max_total_tokens != null && (!usage.complete || usage.inputTokens + usage.outputTokens >= policy.max_total_tokens)
  }

  /** Inspect unresolved claims after restart; recovery requires executor evidence. */
  unresolved(owner: string): ReactionClaim[] {
    return this.db.query<ClaimRow, [string]>("SELECT * FROM reaction_claims WHERE owner=? AND outcome='claimed'").all(owner)
      .map(row => ({ id: row.id, owner: row.owner, runId: row.run_id, fromSequence: row.first, throughSequence: row.last, deadline: row.deadline }))
  }
  close(): void { this.cancellations.clear(); this.db.close() }
}
