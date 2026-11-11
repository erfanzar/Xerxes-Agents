// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Database } from 'bun:sqlite'
import { randomUUID } from 'node:crypto'
import type { NotebookScope } from './memory.js'

export type WakeKind = 'task_idle' | 'goal_blocked' | 'question_waiting' | 'tests_failing' | 'disk_threshold'
export interface Wake extends NotebookScope { readonly id: string; readonly at: number; readonly reason: string; readonly state: string }
interface WakeRow { id: string; owner: string; thread: string; workspace: string; at: number; reason: string; state: string }
interface Subscription extends NotebookScope { readonly id: string; readonly kind: WakeKind; readonly threshold: number | null }
interface WakeOptions {
  readonly clock: () => number
  readonly paused: () => boolean
  readonly authorized: (scope: NotebookScope) => boolean
  readonly deliver: (wake: Wake) => Promise<void>
  readonly onError?: (error: unknown) => void
}
const KINDS: ReadonlySet<string> = new Set(['task_idle', 'goal_blocked', 'question_waiting', 'tests_failing', 'disk_threshold'])

/** Durable timers and event receipts. Wakes are data-only observations, never C/D mandates. */
export class SupervisorWakes {
  private timer: ReturnType<typeof setTimeout> | undefined
  private disposed = false
  private active: Promise<void> | undefined
  constructor(private readonly db: Database, private readonly options: WakeOptions) {
    db.exec(`CREATE TABLE IF NOT EXISTS hub_superagent_wakes (
      id TEXT PRIMARY KEY, owner TEXT NOT NULL, thread TEXT NOT NULL, workspace TEXT NOT NULL,
      at INTEGER NOT NULL, reason TEXT NOT NULL, state TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS hub_superagent_wakes_due ON hub_superagent_wakes(state, at);
      CREATE TABLE IF NOT EXISTS hub_superagent_subscriptions (
      id TEXT PRIMARY KEY, owner TEXT NOT NULL, thread TEXT NOT NULL, workspace TEXT NOT NULL,
      kind TEXT NOT NULL, threshold REAL, UNIQUE(owner,thread,workspace,kind));
      CREATE TABLE IF NOT EXISTS hub_superagent_event_receipts (subscription TEXT NOT NULL, event TEXT NOT NULL, PRIMARY KEY(subscription,event));
      CREATE TABLE IF NOT EXISTS hub_superagent_checkins (owner TEXT NOT NULL, thread TEXT NOT NULL,
      workspace TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(owner,thread,workspace));`)
    this.refresh()
  }

  schedule(scope: NotebookScope, input: { at?: number; after?: number; reason: string }): Wake {
    this.check(scope)
    if ((input.at === undefined) === (input.after === undefined)) throw new Error('Specify exactly one of at or after (milliseconds)')
    const now = this.options.clock()
    const at = input.at ?? now + (input.after ?? 0)
    if (!Number.isSafeInteger(at) || at < now || at > now + 30 * 86400_000 || (input.after !== undefined && input.after < 0)) throw new Error('Wake deadline must be within the next 30 days')
    if (input.reason.trim() === '' || Buffer.byteLength(input.reason) > 2000) throw new Error('Wake reason must contain 1..2000 bytes')
    if (!this.options.authorized(scope)) throw new Error('Owner no longer authorized')
    const row: Wake = { ...scope, id: `wake-${randomUUID()}`, at, reason: input.reason, state: 'pending' }
    this.db.transaction(() => {
      const count = this.db.query<{ n: number }, [string]>("SELECT COUNT(*) AS n FROM hub_superagent_wakes WHERE owner=? AND state='pending'").get(scope.owner)?.n ?? 0
      if (count >= 100) throw new Error('Pending wake budget exceeded')
      this.db.query('INSERT INTO hub_superagent_wakes VALUES (?,?,?,?,?,?,?)').run(row.id, row.owner, row.thread, row.workspace, row.at, row.reason, row.state)
    }).immediate()
    this.refresh()
    return row
  }

  list(scope: NotebookScope): readonly Wake[] {
    this.check(scope)
    return this.db.query<WakeRow, [string, string, string]>(`SELECT * FROM hub_superagent_wakes
      WHERE owner=? AND thread=? AND workspace=? ORDER BY at DESC LIMIT 200`).all(scope.owner, scope.thread, scope.workspace)
  }

  cancel(scope: NotebookScope, id: string): boolean {
    this.check(scope)
    const changed = this.db.query("UPDATE hub_superagent_wakes SET state='cancelled' WHERE id=? AND owner=? AND thread=? AND workspace=? AND state='pending'")
      .run(id, scope.owner, scope.thread, scope.workspace).changes === 1
    this.refresh(); return changed
  }

  subscribe(scope: NotebookScope, input: { kind: WakeKind; threshold?: number }): Subscription {
    this.check(scope)
    if (!this.options.authorized(scope) || !KINDS.has(input.kind)) throw new Error('Subscription not authorized or invalid')
    if (input.kind === 'disk_threshold' && (typeof input.threshold !== 'number' || !Number.isFinite(input.threshold) || input.threshold < 1 || input.threshold > 100)) throw new Error('Disk threshold must be a percentage from 1 to 100')
    if (input.kind !== 'disk_threshold' && input.threshold !== undefined) throw new Error('Only disk subscriptions have a threshold')
    const row = { ...scope, id: `sub-${randomUUID()}`, kind: input.kind, threshold: input.threshold ?? null }
    this.db.query(`INSERT INTO hub_superagent_subscriptions VALUES (?,?,?,?,?,?) ON CONFLICT(owner,thread,workspace,kind)
      DO UPDATE SET threshold=excluded.threshold`).run(row.id, row.owner, row.thread, row.workspace, row.kind, row.threshold)
    return this.db.query<Subscription, [string, string, string, string]>(`SELECT * FROM hub_superagent_subscriptions
      WHERE owner=? AND thread=? AND workspace=? AND kind=?`).get(scope.owner, scope.thread, scope.workspace, input.kind) ?? row
  }

  unsubscribe(scope: NotebookScope, kind: WakeKind): boolean {
    this.check(scope)
    return this.db.query('DELETE FROM hub_superagent_subscriptions WHERE owner=? AND thread=? AND workspace=? AND kind=?')
      .run(scope.owner, scope.thread, scope.workspace, kind).changes === 1
  }

  async observe(event: { id: string; workspace: string; kind: WakeKind; digest: string; value?: number }): Promise<void> {
    if (this.disposed || this.options.paused()) return
    if (event.id.length < 1 || event.id.length > 200 || event.digest.length > 1000 || !KINDS.has(event.kind)) throw new Error('Invalid subscription event')
    const subscriptions = this.db.query<Subscription, [string, string]>('SELECT * FROM hub_superagent_subscriptions WHERE workspace=? AND kind=?').all(event.workspace, event.kind)
    for (const sub of subscriptions) {
      if (!this.options.authorized(sub)) continue
      if (sub.kind === 'disk_threshold' && (event.value === undefined || !Number.isFinite(event.value) || event.value < (sub.threshold ?? 100))) continue
      this.db.transaction(() => {
        const seen = this.db.query('SELECT 1 FROM hub_superagent_event_receipts WHERE subscription=? AND event=?').get(sub.id, event.id)
        if (seen) return
        const count = this.db.query<{ n: number }, []>('SELECT COUNT(*) AS n FROM hub_superagent_event_receipts').get()?.n ?? 0
        if (count >= 100000) throw new Error('Event receipt budget exhausted; owner maintenance required')
        // Receipt and wake are one transaction: failure to enqueue never loses an event silently.
        let reason = `${event.kind}: `
        for (const character of event.digest) {
          if (Buffer.byteLength(reason + character) > 2000) break
          reason += character
        }
        this.schedule(sub, { after: 0, reason })
        this.db.query('INSERT INTO hub_superagent_event_receipts VALUES (?,?)').run(sub.id, event.id)
      }).immediate()
    }
  }

  changed(scope: NotebookScope, digest: string, commit = true): boolean {
    this.check(scope)
    if (digest.length > 1000) throw new Error('Check-in digest exceeds limit')
    const previous = this.db.query<{ digest: string }, [string, string, string]>('SELECT digest FROM hub_superagent_checkins WHERE owner=? AND thread=? AND workspace=?')
      .get(scope.owner, scope.thread, scope.workspace)
    if (previous?.digest === digest) return false
    if (!commit) return true
    this.db.query(`INSERT INTO hub_superagent_checkins VALUES (?,?,?,?) ON CONFLICT(owner,thread,workspace)
      DO UPDATE SET digest=excluded.digest`).run(scope.owner, scope.thread, scope.workspace, digest)
    return true
  }

  drain(): Promise<void> {
    if (this.active !== undefined) return this.active
    const run = async (): Promise<void> => {
      if (this.disposed || this.options.paused()) return
      const rows = this.db.query<WakeRow, [number]>("SELECT * FROM hub_superagent_wakes WHERE state='pending' AND at<=? ORDER BY at,id LIMIT 100").all(this.options.clock())
      for (const row of rows) {
        if (this.disposed || this.options.paused()) break
        if (!this.options.authorized(row)) {
          this.db.query("UPDATE hub_superagent_wakes SET state='blocked' WHERE id=? AND state='pending'").run(row.id); continue
        }
        if (this.db.query("UPDATE hub_superagent_wakes SET state='claimed' WHERE id=? AND state='pending'").run(row.id).changes !== 1) continue
        // A claimed wake after a crash remains visibly claimed; it is never replayed as a side effect.
        try {
          await this.options.deliver({ ...row, state: 'claimed' })
          // Shutdown may close SQLite while delivery awaits the runner. Keep
          // the durable claim (unknown outcome), never touch the closed host.
          if (this.disposed) return
          this.db.query("UPDATE hub_superagent_wakes SET state='delivered' WHERE id=?").run(row.id)
        } catch (error) {
          if (!this.disposed) this.db.query("UPDATE hub_superagent_wakes SET state='failed' WHERE id=?").run(row.id)
          this.options.onError?.(error)
        }
      }
    }
    this.active = run().finally(() => { this.active = undefined; this.refresh() })
    return this.active
  }

  refresh(): void {
    clearTimeout(this.timer); this.timer = undefined
    if (this.disposed || this.options.paused()) return
    const next = this.db.query<{ at: number }, []>("SELECT at FROM hub_superagent_wakes WHERE state='pending' ORDER BY at LIMIT 1").get()
    if (next === null) return
    this.timer = setTimeout(() => { void this.drain().catch(error => this.options.onError?.(error)) }, Math.max(1, Math.min(2147483647, next.at - this.options.clock())))
    this.timer.unref?.()
  }

  dispose(): void { this.disposed = true; clearTimeout(this.timer); this.timer = undefined }
  private check(scope: NotebookScope): void {
    if (this.disposed || [scope.owner, scope.thread, scope.workspace].some(value => value.trim() === '' || value.length > 200)) throw new Error('Invalid or disposed wake scope')
  }
}
