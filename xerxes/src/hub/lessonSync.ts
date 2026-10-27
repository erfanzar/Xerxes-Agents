// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { Database } from 'bun:sqlite'

import {
  mergeSharedLesson, parseSharedLessonWire, REPO_KEY_PATTERN, sharedLessonChanges,
  SHARED_LESSON_ID_PATTERN, SHARED_LESSON_LIMITS, type SharedLessonWire,
} from '../memory/sharedLessonWire.js'
import { canonicalJson, paramsHash, type HubAuthClaims } from '../security/hubAuthEnvelope.js'
import { HUB_AUTH_LIFETIME_MS } from './commandRouter.js'
import type { DaemonCallOptions } from './daemonClient.js'
import { isRecord } from './protocol.js'
import { LESSON_CALL_TIMEOUT_MS, type HubLessonDaemonMethod, type PoolCallResult, type PoolLessonWorkspace } from './runtimePool.js'
import type { HubSigner } from './signer.js'

export const HUB_LESSONS_PER_REPO = 5000
export const HUB_LESSON_MAX_LIST_PAGES = 10
/** Bounded run-local suppression: saturation disables pushes rather than evicting conflicts and retrying them. */
const MAX_CONFLICTS = 10_000
const SYNC_CONCURRENCY = 4
const DEFAULT_INTERVAL_MS = 60_000
const MIN_INTERVAL_MS = 10_000
/** Larger native timer delays overflow to 1 ms; never turn a typo into a busy loop. */
const MAX_INTERVAL_MS = 2_147_483_647

export interface HubLessonRepoCount {
  readonly repo: string
  readonly lessons: number
  readonly active: number
  readonly retired: number
}

/** Canonical field comparison excludes ONLY the three mutable lifecycle fields. */
function immutableLessonContent(lesson: SharedLessonWire): string {
  const { status: _status, updated_at: _updatedAt, retired_reason: _reason, ...content } = lesson
  return canonicalJson(content)
}
function contentFingerprint(lesson: SharedLessonWire): string {
  return createHash('sha256').update(immutableLessonContent(lesson)).digest('hex')
}

/** Append/retire only: never evict a tombstone to make room for another id. */
export class HubLessonStore {
  private readonly db: Database
  constructor(databasePath: string) {
    this.db = new Database(databasePath, { create: true, strict: true })
    this.db.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA synchronous = FULL;
      PRAGMA busy_timeout = 5000;
      CREATE TABLE IF NOT EXISTS shared_lessons (
        repo TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('active', 'retired')), updated_at TEXT NOT NULL,
        PRIMARY KEY (repo, id)
      );
    `)
  }

  lessons(repo: string): readonly SharedLessonWire[] {
    return this.db.query<{ body: string }, [string, number]>('SELECT body FROM shared_lessons WHERE repo = ? ORDER BY id LIMIT ?').all(repo, HUB_LESSONS_PER_REPO)
      .map(row => this.parse(row.body, repo))
  }

  merge(repo: string, lessons: readonly SharedLessonWire[]): { readonly changed: number; readonly capped: number; readonly conflicts: readonly string[] } {
    if (!REPO_KEY_PATTERN.test(repo)) throw new Error('Invalid shared lesson repo')
    // The caller batches list replies. Keep the public store boundary bounded too.
    if (lessons.length > SHARED_LESSON_LIMITS.listLessons) throw new Error('Too many shared lessons in a merge')
    return this.db.transaction(() => {
      let count = this.db.query<{ count: number }, [string]>('SELECT COUNT(*) AS count FROM shared_lessons WHERE repo = ?').get(repo)!.count
      let changed = 0, capped = 0
      const conflicts = new Set<string>()
      const find = this.db.query<{ body: string }, [string, string]>('SELECT body FROM shared_lessons WHERE repo = ? AND id = ?')
      const write = this.db.query('INSERT INTO shared_lessons (repo, id, body, status, updated_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(repo, id) DO UPDATE SET body = excluded.body, status = excluded.status, updated_at = excluded.updated_at')
      for (const value of lessons) {
        const incoming = parseSharedLessonWire(value)
        if (!incoming.ok || incoming.lesson.repo !== repo) throw new Error('Invalid shared lesson in store merge')
        const row = find.get(repo, incoming.lesson.id)
        const current = row === null ? undefined : this.parse(row.body, repo)
        if (current !== undefined && immutableLessonContent(current) !== immutableLessonContent(incoming.lesson)) {
          conflicts.add(incoming.lesson.id); continue
        }
        if (current === undefined && count >= HUB_LESSONS_PER_REPO) { capped++; continue }
        const merged = mergeSharedLesson(current, incoming.lesson)
        if (merged === current) continue
        write.run(repo, merged.id, JSON.stringify(merged), merged.status, merged.updated_at)
        if (current === undefined) count++
        changed++
      }
      return { changed, capped, conflicts: [...conflicts] }
    }).immediate()
  }

  counts(): readonly HubLessonRepoCount[] {
    return this.db.query<HubLessonRepoCount, []>(`SELECT repo, COUNT(*) AS lessons,
      SUM(status = 'active') AS active, SUM(status = 'retired') AS retired
      FROM shared_lessons GROUP BY repo ORDER BY repo`).all()
  }

  close(): void { this.db.close() }

  private parse(body: string, repo: string): SharedLessonWire {
    const parsed = parseSharedLessonWire(JSON.parse(body) as unknown)
    if (!parsed.ok || parsed.lesson.repo !== repo) throw new Error('Invalid stored shared lesson; refusing to sync')
    return parsed.lesson
  }
}

export interface HubLessonBackend {
  /** Only online, verified, signed-command lanes advertising lessons_sync_v1_supported. */
  lessonWorkspaces(): readonly PoolLessonWorkspace[]
  callLessons(workspaceId: string, method: string, params: Readonly<Record<string, unknown>>, options?: DaemonCallOptions): Promise<PoolCallResult>
}
export interface HubLessonSyncOptions {
  readonly databasePath: string
  readonly signer: Pick<HubSigner, 'hubId' | 'sign'>
  readonly backend: HubLessonBackend
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly clock?: () => number
  readonly onError?: (error: unknown, context: string) => void
}
export interface HubLessonSyncStatus {
  readonly enabled: boolean
  readonly running: boolean
  readonly interval_ms: number
  /** Last completed pass, including passes with isolated workspace failures. */
  readonly last_run: string | null
  readonly repos: readonly HubLessonRepoCount[]
  /** Distinct (repo, runtime, id) conflicts observed in this Hub run. */
  readonly conflicts: number
  /** Conflicting ids excluded from pushes for the remainder of this Hub run. */
  readonly suppressed_conflicts: number
  readonly conflict_limit_reached: boolean
  /** Workspace listings that reached the page bound in the last completed pass. */
  readonly incomplete_listings: number
}

// Keep compact immutable-content fingerprints, not up to 10 MiB of text per
// workspace, between gather and push. Lifecycle fields still drive merge/change.
type DaemonVersion = Pick<SharedLessonWire, 'status' | 'updated_at'> & { readonly content: string }
interface DaemonSnapshot {
  readonly workspace: PoolLessonWorkspace
  readonly repo: string
  readonly versions: ReadonlyMap<string, DaemonVersion>
  readonly complete: boolean
  readonly lastId: string | undefined
}
interface DaemonPage {
  readonly repo: string
  readonly lessons: readonly SharedLessonWire[]
  readonly nextAfterId: string | undefined
  readonly lastId: string | undefined
}

export class HubLessonSync {
  private readonly store: HubLessonStore
  private readonly enabled: boolean
  private readonly intervalMs: number
  private readonly clock: () => number
  private readonly abort = new AbortController()
  private timer: ReturnType<typeof setInterval> | undefined
  private inFlight: Promise<void> | undefined
  private pendingOnline = false
  private stopped = false
  private lastRun: string | null = null
  private readonly conflicts = new Set<string>()
  private conflictLimitReached = false
  private incompleteListings = 0

  constructor(private readonly options: HubLessonSyncOptions) {
    const env = options.environment ?? process.env
    this.enabled = env.XERXES_HUB_LESSON_SYNC !== 'off'
    const interval = Number(env.XERXES_HUB_LESSON_SYNC_MS ?? DEFAULT_INTERVAL_MS)
    this.intervalMs = Number.isSafeInteger(interval) && interval > 0 && interval <= MAX_INTERVAL_MS
      ? Math.max(MIN_INTERVAL_MS, interval) : DEFAULT_INTERVAL_MS
    this.clock = options.clock ?? Date.now
    this.store = new HubLessonStore(options.databasePath)
  }

  start(): void {
    if (!this.enabled || this.stopped || this.timer !== undefined) return
    this.timer = setInterval(() => { void this.syncNow() }, this.intervalMs)
    this.timer.unref?.()
    void this.syncNow()
  }

  /** Coalesced: callers during a pass share it, never start another concurrently. */
  syncNow(): Promise<void> {
    if (this.inFlight !== undefined) return this.inFlight
    if (!this.enabled || this.stopped) return Promise.resolve()
    // Defer one microtask so inFlight is installed before any backend callback.
    const running = Promise.resolve().then(() => this.pass()).catch(error => this.report(error, 'pass')).finally(() => {
      this.inFlight = undefined
      if (this.pendingOnline && !this.stopped) { this.pendingOnline = false; void this.syncNow() }
    })
    this.inFlight = running
    return running
  }

  /** Online arrivals during a pass request at most one follow-up pass. */
  workspaceOnline(): void {
    if (!this.enabled || this.stopped) return
    if (this.inFlight !== undefined) this.pendingOnline = true
    else void this.syncNow()
  }

  status(): HubLessonSyncStatus {
    return {
      enabled: this.enabled, running: this.inFlight !== undefined, interval_ms: this.intervalMs, last_run: this.lastRun, repos: this.store.counts(),
      conflicts: this.conflicts.size, suppressed_conflicts: this.conflicts.size, conflict_limit_reached: this.conflictLimitReached,
      incomplete_listings: this.incompleteListings,
    }
  }

  async close(): Promise<void> {
    if (this.stopped) return
    this.stopped = true
    clearInterval(this.timer); this.timer = undefined
    this.abort.abort()
    await this.inFlight
    this.store.close()
  }

  private report(error: unknown, context: string): void {
    if (!this.stopped) {
      if (this.options.onError) this.options.onError(error, `lesson sync ${context}`)
      else console.error(`Hub lesson sync ${context} failed`, error)
    }
  }

  private async pass(): Promise<void> {
    const seen = new Set<string>()
    const workspaces = this.options.backend.lessonWorkspaces().filter(workspace => {
      const key = JSON.stringify([workspace.runtimeId, workspace.projectDir])
      if (seen.has(key)) return false
      seen.add(key); return true
    })
    // Gather from ALL machines before pushing: a retirement discovered on B can
    // retire A in this same pass, independent of workspace ordering.
    const snapshots: DaemonSnapshot[] = []
    await this.each(workspaces, async workspace => {
      const snapshot = await this.readSnapshot(workspace)
      if (snapshot !== undefined) snapshots.push(snapshot)
    }, workspace => workspace.id)
    await this.each(snapshots, snapshot => this.push(snapshot), snapshot => snapshot.workspace.id)
    if (!this.stopped) {
      this.lastRun = new Date(this.clock()).toISOString()
      this.incompleteListings = snapshots.filter(snapshot => !snapshot.complete).length
    }
  }

  private async each<T>(rows: readonly T[], run: (row: T) => Promise<void>, id: (row: T) => string): Promise<void> {
    let index = 0
    await Promise.all(Array.from({ length: Math.min(SYNC_CONCURRENCY, rows.length) }, async () => {
      while (!this.stopped) {
        const row = rows[index++]
        if (row === undefined) return
        try { await run(row) } catch (error) { this.report(error, `workspace ${id(row)}`) }
      }
    }))
  }

  private conflictKey(workspace: PoolLessonWorkspace, repo: string, id: string): string {
    return createHash('sha256').update(JSON.stringify([repo, workspace.runtimeId, id])).digest('hex')
  }

  private rememberConflict(workspace: PoolLessonWorkspace, repo: string, id: string): void {
    const key = this.conflictKey(workspace, repo, id)
    if (this.conflicts.has(key) || this.conflictLimitReached) return
    if (this.conflicts.size >= MAX_CONFLICTS) {
      this.conflictLimitReached = true
      this.report(new Error('Conflict memory bound reached; all lesson pushes disabled for this Hub run'), `workspace ${workspace.id}`)
      return
    }
    this.conflicts.add(key)
    this.report(new Error(`Shared lesson conflict for ${id}; suppressed for this repo/runtime in this Hub run`), `workspace ${workspace.id}`)
  }

  private readPage(workspace: PoolLessonWorkspace, result: Readonly<Record<string, unknown>>, afterId: string | undefined): DaemonPage | undefined {
    if (result.ok !== true || !Array.isArray(result.lessons) || result.lessons.length > SHARED_LESSON_LIMITS.listLessons
      || Buffer.byteLength(JSON.stringify(result)) > SHARED_LESSON_LIMITS.payloadBytes) throw new Error('Invalid shared lesson list reply')
    const next = result.next_after_id
    if (next !== undefined && next !== null && (typeof next !== 'string' || !SHARED_LESSON_ID_PATTERN.test(next))) throw new Error('Invalid shared lesson pagination cursor')
    if (result.repo === null && result.lessons.length === 0 && (next === undefined || next === null)) return undefined
    if (typeof result.repo !== 'string' || !REPO_KEY_PATTERN.test(result.repo)) throw new Error('Invalid repo in shared lesson list reply')
    const paginated = next !== undefined || afterId !== undefined
    const lessons: SharedLessonWire[] = []
    let dropped = 0, previous = afterId, lastId: string | undefined
    let ordered = true
    for (const row of result.lessons) {
      const rawId = isRecord(row) && typeof row.id === 'string' && SHARED_LESSON_ID_PATTERN.test(row.id) ? row.id : undefined
      if (rawId !== undefined) {
        if (previous !== undefined && rawId <= previous) ordered = false
        previous = rawId; lastId = lastId === undefined || rawId > lastId ? rawId : lastId
      }
      const parsed = parseSharedLessonWire(row)
      if (!parsed.ok || parsed.lesson.repo !== result.repo) { dropped++; continue }
      lessons.push(parsed.lesson)
    }
    if (paginated && !ordered) throw new Error('Shared lesson pagination must list ids in ascending order')
    if (typeof next === 'string' && (next !== lastId || (afterId !== undefined && next <= afterId))) throw new Error('Shared lesson pagination did not advance to the last listed id')
    if (dropped > 0) this.report(new Error(`Dropped ${dropped} invalid or wrong-repo shared lesson rows`), `workspace ${workspace.id}`)
    return { repo: result.repo, lessons, nextAfterId: typeof next === 'string' ? next : undefined, lastId }
  }

  private async readSnapshot(workspace: PoolLessonWorkspace): Promise<DaemonSnapshot | undefined> {
    const versions = new Map<string, DaemonVersion>()
    let repo: string | undefined, afterId: string | undefined, lastId: string | undefined
    for (let index = 0; index < HUB_LESSON_MAX_LIST_PAGES; index++) {
      const result = await this.call(workspace, 'lessons.shared.list', afterId === undefined ? {} : { after_id: afterId })
      const page = this.readPage(workspace, result, afterId)
      if (page === undefined) {
        if (repo !== undefined) throw new Error('Shared lesson repo changed during pagination')
        return undefined
      }
      if (repo !== undefined && repo !== page.repo) throw new Error('Shared lesson repo changed during pagination')
      repo = page.repo
      const accepted: SharedLessonWire[] = []
      for (const lesson of page.lessons) {
        const version = versions.get(lesson.id), content = contentFingerprint(lesson)
        if (version !== undefined && version.content !== content) { this.rememberConflict(workspace, repo, lesson.id); continue }
        if (version === undefined || sharedLessonChanges({ ...lesson, status: version.status, updated_at: version.updated_at }, lesson)) {
          versions.set(lesson.id, { status: lesson.status, updated_at: lesson.updated_at, content })
        }
        accepted.push(lesson)
      }
      const merged = this.store.merge(repo, accepted)
      for (const id of merged.conflicts) this.rememberConflict(workspace, repo, id)
      if (merged.capped > 0) this.report(new Error(`Repo lesson cap reached; ${merged.capped} new ids skipped`), `workspace ${workspace.id}`)
      lastId = page.lastId ?? lastId
      if (page.nextAfterId === undefined) return { workspace, repo, versions, complete: true, lastId }
      afterId = page.nextAfterId
      if (index + 1 === HUB_LESSON_MAX_LIST_PAGES) {
        this.report(new Error(`Shared lesson page bound (${HUB_LESSON_MAX_LIST_PAGES}) reached; only the listed prefix may receive pushes`), `workspace ${workspace.id}`)
        return { workspace, repo, versions, complete: false, lastId }
      }
    }
    return undefined
  }

  private signedParams(workspace: PoolLessonWorkspace, method: HubLessonDaemonMethod, params: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> {
    const wire = { ...params, project_dir: workspace.projectDir }
    const now = this.clock()
    const claims: HubAuthClaims = {
      hub_id: this.options.signer.hubId, principal_id: 'hub-lesson-sync', principal_kind: 'agent', display_name: 'Hub lesson sync',
      workspace_id: workspace.id, method, params_hash: paramsHash(wire), request_id: crypto.randomUUID(), nonce: crypto.randomUUID(),
      issued_at: now, expires_at: now + HUB_AUTH_LIFETIME_MS,
      scopes: method === 'lessons.shared.list' ? ['session.read'] : ['lessons.sync'],
    }
    return { ...wire, hub_auth: this.options.signer.sign(claims) }
  }

  private async call(workspace: PoolLessonWorkspace, method: HubLessonDaemonMethod, params: Readonly<Record<string, unknown>>): Promise<Readonly<Record<string, unknown>>> {
    this.abort.signal.throwIfAborted()
    const wire = this.signedParams(workspace, method, params)
    if (Buffer.byteLength(JSON.stringify(wire)) > SHARED_LESSON_LIMITS.payloadBytes) throw new Error('Shared lesson request exceeds 1 MiB')
    const outcome = await this.options.backend.callLessons(workspace.id, method, wire, { timeoutMs: LESSON_CALL_TIMEOUT_MS, signal: this.abort.signal })
    this.abort.signal.throwIfAborted()
    if (outcome.state !== 'accepted') throw new Error(`Daemon refused shared lesson sync (${outcome.error.code.slice(0, 100)})`)
    if (Buffer.byteLength(JSON.stringify(outcome.result)) > SHARED_LESSON_LIMITS.payloadBytes) throw new Error('Shared lesson reply exceeds 1 MiB')
    return outcome.result
  }

  private async push(snapshot: DaemonSnapshot): Promise<void> {
    const { workspace, repo, versions } = snapshot
    if (this.conflictLimitReached) return
    let batch: SharedLessonWire[] = []
    // Empty signed request gives exact fixed envelope overhead; hashes, UUIDs and
    // Ed25519 signatures have fixed lengths. Also check the final signed bytes.
    const overhead = Buffer.byteLength(JSON.stringify(this.signedParams(workspace, 'lessons.shared.apply', { repo, lessons: [] })))
    let bytes = overhead
    const flush = async (): Promise<void> => {
      if (batch.length === 0) return
      const result = await this.call(workspace, 'lessons.shared.apply', { repo, lessons: batch })
      if (result.ok !== true || typeof result.applied !== 'number' || !Number.isSafeInteger(result.applied)
        || typeof result.retired !== 'number' || !Number.isSafeInteger(result.retired)
        || result.applied < 0 || result.retired < 0 || !Array.isArray(result.skipped)
        || result.applied + result.retired + result.skipped.length > batch.length
        || !result.skipped.every(row => isRecord(row) && typeof row.id === 'string' && SHARED_LESSON_ID_PATTERN.test(row.id)
          && batch.some(lesson => lesson.id === row.id) && typeof row.reason === 'string' && row.reason.length <= SHARED_LESSON_LIMITS.reason)) {
        throw new Error('Invalid shared lesson apply reply')
      }
      for (const row of result.skipped) {
        if (isRecord(row) && typeof row.id === 'string' && row.reason === 'conflict') this.rememberConflict(workspace, repo, row.id)
      }
      if (result.skipped.length > 0) this.report(new Error(`Daemon skipped ${result.skipped.length} shared lessons`), `workspace ${workspace.id}`)
      batch = []; bytes = overhead
    }
    for (const lesson of this.store.lessons(repo)) {
      if (this.conflictLimitReached) return
      if (!snapshot.complete && (snapshot.lastId === undefined || lesson.id > snapshot.lastId)) continue
      if (this.conflicts.has(this.conflictKey(workspace, repo, lesson.id))) continue
      const version = versions.get(lesson.id)
      if (version !== undefined && version.content !== contentFingerprint(lesson)) { this.rememberConflict(workspace, repo, lesson.id); continue }
      const current = version === undefined ? undefined : { ...lesson, status: version.status, updated_at: version.updated_at }
      if (!sharedLessonChanges(current, lesson)) continue
      const size = Buffer.byteLength(JSON.stringify(lesson))
      if (batch.length >= SHARED_LESSON_LIMITS.applyLessons || bytes + size + (batch.length > 0 ? 1 : 0) > SHARED_LESSON_LIMITS.payloadBytes) await flush()
      if (overhead + size > SHARED_LESSON_LIMITS.payloadBytes) throw new Error('Shared lesson cannot fit in one apply request')
      bytes += size + (batch.length > 0 ? 1 : 0); batch.push(lesson)
    }
    await flush()
  }
}
