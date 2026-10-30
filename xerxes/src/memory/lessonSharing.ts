// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lstat, mkdir, readdir, realpath } from 'node:fs/promises'
import { hostname } from 'node:os'
import { dirname, join } from 'node:path'
import { scanContextContent } from '../security/promptScanner.js'
import type { JsonObject } from '../types/toolCalls.js'
import { GovernedMemoryStore, type MemoryHistoryRecord } from './governedMemoryStore.js'
import { canonicalWorkspace, lessonProvenance, LessonStore, storedEvidence } from './lessons.js'
import { lessonGit, lessonGitRoot } from './repoIdentity.js'
import { atomicLessonWrite, isMissingLessonFile, LessonSharingSettings, readLessonFile } from './lessonSharingFiles.js'
import { mergeSharedLesson, parseSharedLessonWire, SHARED_LESSON_ID_PATTERN, SHARED_LESSON_LIMITS, type SharedLessonWire } from './sharedLessonWire.js'

export interface LessonSyncResult {
  readonly applied: number
  readonly retired: number
  readonly skipped: readonly { readonly id: string; readonly reason: string }[]
}
export interface LessonFolderSyncResult extends LessonSyncResult {
  readonly repo: string | null
  readonly exported: number
  readonly written: number
  readonly enabled: boolean
}
export class LessonSyncError extends Error {
  constructor(readonly code: 'disabled' | 'repo_mismatch' | 'invalid_request', message: string) { super(message); this.name = 'LessonSyncError' }
}
interface SharingOptions {
  readonly memory: GovernedMemoryStore
  readonly lessons: LessonStore
  readonly report?: (message: string) => void
  readonly settingsPath?: string
  readonly resolveRoot?: (workspace: string) => Promise<string | undefined>
  readonly hostname?: () => string
  readonly preTurnDeadlineMs?: number
}

/** One queue covers imports, stable export metadata, and local trust configuration. */
export class LessonSharing {
  private queue: Promise<unknown> = Promise.resolve()
  private pending = 0
  private readonly settings: LessonSharingSettings
  private readonly synced = new Map<string, { at: number; signature: string }>()
  private revision = 0
  private readonly unsubscribe: () => void
  constructor(private readonly options: SharingOptions) {
    this.settings = new LessonSharingSettings(options.settingsPath ?? join(dirname(options.memory.logPath), 'lesson-sharing.json'))
    this.unsubscribe = options.lessons.subscribe(async workspace => {
      this.revision++
      if (disabled()) return
      try { await this.serial(async () => { await this.exportFolder(workspace) }) }
      catch (error) { this.report(`Lesson export failed: ${message(error)}`) }
    })
  }
  close(): void { this.unsubscribe() }
  private report(text: string): void { (this.options.report ?? console.error)(text) }
  private enabled(): void {
    if (disabled()) throw new LessonSyncError('disabled', 'Lesson sharing is disabled by XERXES_LESSON_SYNC=off.')
  }
  private serial<T>(work: () => Promise<T>): Promise<T> {
    this.pending++
    const result = this.queue.then(work).finally(() => { this.pending-- })
    this.queue = result.catch(() => {})
    return result
  }
  async repoForWorkspace(workspace: string): Promise<string | undefined> {
    this.enabled()
    return this.options.lessons.repoForWorkspace(await canonicalWorkspace(workspace))
  }

  /** Hub export is independent of local Git-folder opt-in. Superseded ids remain represented. */
  async snapshot(workspace: string, options: { readonly markShared?: boolean } = {}): Promise<{ repo: string | null; lessons: SharedLessonWire[] }> {
    this.enabled()
    return this.serial(() => this.snapshotRecords(workspace, options.markShared !== false))
  }
  /**
   * `persist` writes what an export makes true (the repo key of a legacy lesson, the frozen
   * wire); only a preview passes false. `markShared` additionally records publication.
   */
  private async snapshotRecords(workspace: string, markShared: boolean, persist = markShared): Promise<{ repo: string | null; lessons: SharedLessonWire[] }> {
    const repo = await this.repoForWorkspace(workspace)
    if (!repo) return { repo: null, lessons: [] }
    const adopted = await this.adoptLegacyLessons(repo, persist)
    const lessons: SharedLessonWire[] = []
    for (const stored of (await this.options.memory.loadHistory()).records.values()) {
      // A preview (markShared false) adopts in memory only; it never writes the store.
      const record: MemoryHistoryRecord = adopted.has(stored.id) && stored.payload?.repo === undefined ? { ...stored, payload: { ...stored.payload, repo } } : stored
      if (record.payload?.repo !== repo || record.payload.kind !== 'lesson' || sensitive(record)) continue
      const dead = retired(record, this.options.memory.now())
      if (dead ? !wasShared(record) : record.reviewStatus !== 'approved') continue
      const wire = await this.wire(record, persist)
      if (!wire) continue
      lessons.push(wire)
      if (markShared && wire.status === 'active' && record.payload.shared_once !== true) await this.options.memory.updatePayload(record.id, { shared_once: true })
    }
    return { repo, lessons: lessons.sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0) }
  }
  /**
   * Lessons recorded before lessons carried a repository key (pre-0.8.14) only name their
   * workspace path, so export skipped every one of them. Give each such lesson the key of the
   * repository its recorded workspace belongs to, once, as an append-only payload update.
   * A workspace that no longer exists or is not in this repository is left alone.
   */
  private async adoptLegacyLessons(repo: string, write: boolean): Promise<Set<string>> {
    const adopted = new Set<string>()
    const keys = new Map<string, string | undefined>()
    for (const record of (await this.options.memory.loadHistory()).records.values()) {
      const payload = record.payload
      if (payload?.kind !== 'lesson' || typeof payload.repo === 'string' || typeof payload.workspace !== 'string' || sensitive(record)) continue
      const workspace = payload.workspace
      if (!keys.has(workspace)) {
        let key: string | undefined
        try { key = await this.options.lessons.repoForWorkspace(workspace) } catch { key = undefined }
        keys.set(workspace, key)
      }
      if (keys.get(workspace) !== repo) continue
      adopted.add(record.id)
      if (write) await this.options.memory.updatePayload(record.id, { repo })
    }
    return adopted
  }

  /** Mark only rows actually emitted in a bounded RPC page. */
  async markShared(lessons: readonly SharedLessonWire[]): Promise<void> {
    await this.serial(async () => {
      const records = (await this.options.memory.loadHistory()).records
      for (const wire of lessons) {
        const record = records.get(wire.id)
        if (wire.status === 'active' && record && !sensitive(record) && record.payload?.shared_once !== true) await this.options.memory.updatePayload(wire.id, { shared_once: true })
      }
    })
  }

  /** `write: false` (a preview) builds the same wire without freezing legacy fields to disk. */
  private async wire(record: MemoryHistoryRecord, write = true): Promise<SharedLessonWire | undefined> {
    const retirement = parseSharedLessonWire(record.expiredPayload?.shared)
    if (retirement.ok) return retirement.lesson
    const frozen = parseSharedLessonWire(record.payload?.shared)
    let original: SharedLessonWire
    if (frozen.ok) original = frozen.lesson
    else {
      const evidence = storedEvidence(record.payload?.evidence)
      if (!evidence) { this.report(`Lesson ${record.id} cannot be shared: invalid evidence`); return undefined }
      const host = typeof record.payload?.origin_host === 'string' ? record.payload.origin_host : (this.options.hostname ?? hostname)()
      const parsed = parseSharedLessonWire({ id: record.id, repo: record.payload?.repo, status: 'active', cue: record.payload?.cue,
        lesson: record.content, confidence: record.payload?.confidence, evidence: lessonProvenance(evidence).slice(0, SHARED_LESSON_LIMITS.evidence),
        supersedes: ancestry(record), origin: { host, agent_id: record.agentId },
        recorded_at: new Date(record.createdAt).toISOString(), updated_at: new Date(record.createdAt).toISOString() })
      if (!parsed.ok) { this.report(`Lesson ${record.id} cannot be shared: ${parsed.error}`); return undefined }
      original = parsed.lesson
      // Freeze legacy records once, without re-recording or resurrecting the id.
      if (write) await this.options.memory.updatePayload(record.id, { origin_host: host, shared: jsonWire(original) })
    }
    if (!retired(record, this.options.memory.now()) || original.status === 'retired') return original
    const at = record.expiredAt ?? (record.expiresAt !== undefined && record.expiresAt <= this.options.memory.now() ? record.expiresAt : record.reviewedAt) ?? record.createdAt
    return { ...original, status: 'retired', updated_at: new Date(at).toISOString(), retired_reason: (record.expiredReason ?? record.reviewReason ?? 'retired locally').slice(0, SHARED_LESSON_LIMITS.reason) }
  }

  async apply(workspace: string, input: readonly unknown[], source: 'repo' | 'hub'): Promise<LessonSyncResult> {
    this.enabled()
    const root = await canonicalWorkspace(workspace)
    const result = await this.serial(async () => {
      if (source === 'repo' && !await this.optedIn(root)) return { applied: 0, retired: 0, skipped: [] }
      return this.applyBatch(root, input, source)
    })
    if (result.applied || result.retired) await this.options.lessons.notifySharedChange(root)
    return result
  }
  private async applyBatch(workspace: string, input: readonly unknown[], source: 'repo' | 'hub', signal?: AbortSignal): Promise<LessonSyncResult> {
    this.enabled()
    const repo = await this.repoForWorkspace(workspace)
    if (!repo) throw new LessonSyncError('repo_mismatch', 'This workspace has no Git repository identity; lesson sharing is unavailable.')
    let applied = 0; let countRetired = 0
    const skipped: { id: string; reason: string }[] = []
    // Lessons recorded before repo keys must carry the key before comparison; otherwise this
    // machine's own exported files look like conflicting copies of its own lessons.
    await this.adoptLegacyLessons(repo, true)
    const records = new Map((await this.options.memory.loadHistory()).records)
    for (const value of input) {
      signal?.throwIfAborted()
      const parsed = parseSharedLessonWire(value)
      const id = typeof value === 'object' && value !== null && 'id' in value && typeof value.id === 'string' ? value.id : '(invalid)'
      if (!parsed.ok) { skipped.push({ id, reason: parsed.error }); continue }
      const incoming = parsed.lesson
      if (incoming.repo !== repo) { skipped.push({ id, reason: 'repo_mismatch' }); continue }
      const current = records.get(id)
      let currentWire: SharedLessonWire | undefined
      let freezeLegacyOrigin = false
      if (current) {
        // An old export is the only durable source of a legacy id's original host.
        // Any record not yet frozen whose fields match the file (including old cut evidence) adopts the file's wire.
        if (!parseSharedLessonWire(current.payload?.shared).ok && matchesRecord(current, incoming)) {
          freezeLegacyOrigin = true
          currentWire = incoming
        } else currentWire = current.payload?.kind === 'lesson' ? await this.wire(current) : undefined
        if (!currentWire || immutable(currentWire) !== immutable(incoming)) { skipped.push({ id, reason: 'conflict' }); continue }
      }
      // Existing immutable text is already local: its retirement must not be blocked by the scanner.
      if (!(current && incoming.status === 'retired') && scanned(incoming)) { skipped.push({ id, reason: 'scanner_modified' }); continue }
      if (freezeLegacyOrigin && current) {
        const patch = { origin_host: incoming.origin.host, shared: jsonWire(incoming), shared_once: true }
        await this.options.memory.updatePayload(id, patch)
        records.set(id, { ...current, payload: { ...current.payload, ...patch } })
      }
      if (current) {
        if (retired(current, this.options.memory.now())) {
          if (incoming.status !== 'retired' || !currentWire || incoming.updated_at >= currentWire.updated_at) continue
          // Earlier retirement wins and is persisted, so the Hub will converge.
        } else if (incoming.status === 'active') {
          const stored = parseSharedLessonWire(current.payload?.shared)
          if (current.reviewStatus === 'pending' && current.reviewedAt === undefined && stored.ok && stored.lesson.status === 'active' && storedEvidence(current.payload?.evidence)?.kind === 'shared') {
            await this.options.memory.review(id, 'approved', `Automatically approved lesson shared via ${source}`)
            records.set(id, { ...current, reviewStatus: 'approved', everApproved: true, reviewedAt: this.options.memory.now() }); applied++
          }
          continue
        }
      } else {
        await this.options.memory.record({ id, content: incoming.lesson, source: 'lesson', agentId: incoming.origin.agent_id,
          payload: { kind: 'lesson', version: 1, cue: incoming.cue, confidence: incoming.confidence, workspace, repo, origin_host: incoming.origin.host,
            evidence: { kind: 'shared', host: incoming.origin.host, source, summary: incoming.evidence }, supersedes: [...incoming.supersedes], shared: jsonWire(incoming), shared_once: true } })
        if (incoming.status === 'active') { await this.options.memory.review(id, 'approved', `Automatically approved lesson shared via ${source}`); applied++ }
      }
      if (incoming.status === 'retired') {
        await this.options.memory.expire(id, `retired on ${incoming.origin.host}: ${incoming.retired_reason ?? 'retired'}`, { at: Date.parse(incoming.updated_at), payload: { shared: jsonWire(incoming) } })
        countRetired++
      }
      const saved = (await this.options.memory.loadHistory()).records.get(id)
      if (saved) records.set(id, saved)
    }
    return { applied, retired: countRetired, skipped }
  }

  /** Both explicit commands are the local trust decision; repo files cannot make it. */
  async optIn(workspace: string): Promise<void> {
    const repo = await this.repoForWorkspace(workspace)
    if (!repo) throw new LessonSyncError('repo_mismatch', 'This workspace has no Git repository identity; lesson sharing is unavailable.')
    await this.settings.enable(repo)
  }
  private async optedIn(workspace: string): Promise<boolean> {
    const repo = await this.repoForWorkspace(workspace)
    return repo !== undefined && await this.settings.enabled(repo)
  }
  async sync(workspace: string): Promise<LessonFolderSyncResult> {
    await this.optIn(workspace)
    return this.syncRepoFolder(workspace, { force: true })
  }
  async share(workspace: string): Promise<LessonFolderSyncResult & { ignored: boolean }> {
    this.enabled()
    const root = await this.root(workspace)
    if (!root) throw new LessonSyncError('repo_mismatch', 'No Git toplevel; lesson sharing is unavailable.')
    const settings = join(root, '.xerxes')
    await mkdir(settings, { recursive: true })
    if (await realpath(settings) !== settings) throw new Error('Lesson folder must be inside the real workspace, not a symlinked .xerxes directory.')
    await mkdir(join(settings, 'lessons'), { recursive: true })
    await this.optIn(root)
    const result = await this.syncRepoFolder(root, { force: true })
    const ignored = await lessonGit(root, ['check-ignore', '--quiet', '.xerxes/lessons/'])
    return { ...result, ignored: ignored?.code === 0 }
  }
  private async root(workspace: string): Promise<string | undefined> {
    return (this.options.resolveRoot ?? lessonGitRoot)(await canonicalWorkspace(workspace))
  }

  /** Best-effort enrichment never holds a turn behind another sync or a long disk/Git operation. */
  async syncBeforeTurn(workspace: string, signal: AbortSignal): Promise<void> {
    if (disabled() || this.pending || signal.aborted) return
    const work = this.syncRepoFolder(workspace, { signal }).catch(error => {
      if (!disabled() && !signal.aborted) this.report(`Lesson repository sync failed: ${message(error)}`)
    })
    let timer: ReturnType<typeof setTimeout> | undefined
    let onAbort: (() => void) | undefined
    const wait = new Promise<void>(resolve => {
      timer = setTimeout(resolve, this.options.preTurnDeadlineMs ?? 1500)
      onAbort = () => resolve()
      signal.addEventListener('abort', onAbort, { once: true })
      if (signal.aborted) resolve()
    })
    try { await Promise.race([work, wait]) }
    finally { clearTimeout(timer); if (onAbort) signal.removeEventListener('abort', onAbort) }
  }
  async syncRepoFolder(workspace: string, options: { readonly force?: boolean; readonly signal?: AbortSignal } = {}): Promise<LessonFolderSyncResult> {
    if (disabled()) return emptyResult(null, false)
    const result = await this.serial(async () => {
      const signal = options.signal
      signal?.throwIfAborted()
      const repo = await this.repoForWorkspace(workspace)
      if (!repo) return emptyResult(null, false)
      // Reading a committed `.xerxes/lessons/` needs no opt-in: a pulled repo loads its
      // lessons on the next message. Writing into the repo (export) still needs the
      // explicit `/xerxes-sync` (or `/xerxes lessons share|sync`) on this machine.
      const optedIn = await this.settings.enabled(repo)
      const root = await this.root(workspace)
      signal?.throwIfAborted()
      if (!root) return emptyResult(repo, false)
      const folder = await this.folder(root)
      const empty = emptyResult(repo, optedIn && folder !== undefined)
      if (!folder) return empty
      const allEntries = (await readdir(folder)).filter(name => name.endsWith('.json')).sort()
      const entries = allEntries.slice(0, SHARED_LESSON_LIMITS.listLessons)
      const signature = JSON.stringify([this.revision, allEntries.length, await Promise.all(entries.map(async name => { const s = await lstat(join(folder, name)); return [name, s.mtimeMs, s.size] }))])
      const prior = this.synced.get(root)
      if (!options.force && prior && Date.now() - prior.at < 30_000 && prior.signature === signature) return empty
      const values: unknown[] = []; const skipped: { id: string; reason: string }[] = []
      if (allEntries.length > entries.length) skipped.push({ id: '(folder)', reason: 'folder file limit exceeded (at most 1000 JSON files are read)' })
      for (const name of entries) {
        signal?.throwIfAborted()
        try {
          const parsed = parseSharedLessonWire(JSON.parse(await readLessonFile(join(folder, name))))
          if (!parsed.ok) { skipped.push({ id: name, reason: parsed.error }); continue }
          if (name !== `${parsed.lesson.id}.json`) { skipped.push({ id: name, reason: 'filename does not match lesson id' }); continue }
          values.push(parsed.lesson)
        } catch (error) { skipped.push({ id: name, reason: message(error) }) }
      }
      signal?.throwIfAborted()
      const imported = await this.applyBatch(root, values, 'repo', signal)
      const exported = optedIn ? await this.exportFolder(root, signal) : { exported: 0, written: 0 }
      this.synced.set(root, { at: Date.now(), signature })
      return { ...empty, ...imported, ...exported, skipped: [...skipped, ...imported.skipped] }
    })
    if (result.applied || result.retired) await this.options.lessons.notifySharedChange(workspace)
    for (const row of result.skipped) this.report(`Lesson file ${row.id} skipped: ${row.reason}`)
    return result
  }
  private async folder(root: string): Promise<string | undefined> {
    const path = join(root, '.xerxes', 'lessons')
    try {
      if (await realpath(path) !== path || !(await lstat(path)).isDirectory()) throw new Error('Lesson folder must be a real directory inside the workspace.')
      return path
    } catch (error) { if (isMissingLessonFile(error)) return undefined; throw error }
  }
  private async exportFolder(workspace: string, signal?: AbortSignal): Promise<{ exported: number; written: number }> {
    this.enabled()
    if (!await this.optedIn(workspace)) return { exported: 0, written: 0 }
    const root = await this.root(workspace)
    if (!root) return { exported: 0, written: 0 }
    const folder = await this.folder(root)
    if (!folder) return { exported: 0, written: 0 }
    const repo = await this.repoForWorkspace(root)
    // Existing files are evidence that an identity previously left this machine.
    for (const record of (await this.options.memory.loadHistory()).records.values()) {
      signal?.throwIfAborted()
      if (!SHARED_LESSON_ID_PATTERN.test(record.id) || record.payload?.kind !== 'lesson' || record.payload.repo !== repo || parseSharedLessonWire(record.payload?.shared).ok || sensitive(record)) continue
      try {
        const parsed = parseSharedLessonWire(JSON.parse(await readLessonFile(join(folder, `${record.id}.json`))))
        if (parsed.ok && matchesRecord(record, parsed.lesson)) await this.options.memory.updatePayload(record.id, { shared: jsonWire(parsed.lesson), shared_once: true, origin_host: parsed.lesson.origin.host })
      } catch (error) { if (!isMissingLessonFile(error)) this.report(`Lesson ${record.id} existing file skipped: ${message(error)}`) }
    }
    // A real export: persist the repo key of legacy lessons, or the next import cannot rebuild them.
    const snapshot = await this.snapshotRecords(root, false, true)
    let written = 0; let exported = 0
    for (const lesson of snapshot.lessons.slice(0, SHARED_LESSON_LIMITS.listLessons)) {
      signal?.throwIfAborted()
      const path = join(folder, `${lesson.id}.json`)
      const text = JSON.stringify(lesson, null, 2) + '\n'
      try {
        const previous = await readLessonFile(path)
        const parsed = parseSharedLessonWire(JSON.parse(previous))
        if (!parsed.ok || immutable(parsed.lesson) !== immutable(lesson)) { this.report(`Lesson ${lesson.id} not exported: conflicting existing file`); continue }
        if (previous === text || (parsed.lesson.status === 'retired' && mergeSharedLesson(parsed.lesson, lesson) === parsed.lesson)) {
          if (lesson.status === 'active') await this.options.memory.updatePayload(lesson.id, { shared_once: true })
          exported++; continue
        }
      } catch (error) { if (!isMissingLessonFile(error)) { this.report(`Lesson ${lesson.id} not exported: ${message(error)}`); continue } }
      signal?.throwIfAborted()
      await atomicLessonWrite(path, text, 0o644); written++; exported++
      if (lesson.status === 'active') await this.options.memory.updatePayload(lesson.id, { shared_once: true })
    }
    return { exported, written }
  }
}
function emptyResult(repo: string | null, enabled: boolean): LessonFolderSyncResult { return { repo, enabled, applied: 0, retired: 0, skipped: [], exported: 0, written: 0 } }
function disabled(): boolean { return process.env.XERXES_LESSON_SYNC === 'off' }
function sensitive(record: MemoryHistoryRecord): boolean { return record.sensitivity === 'confidential' || record.sensitivity === 'restricted' }
function wasShared(record: MemoryHistoryRecord): boolean { return record.payload?.shared_once === true || (parseSharedLessonWire(record.payload?.shared).ok && storedEvidence(record.payload?.evidence)?.kind === 'shared') }
function retired(record: MemoryHistoryRecord, now: number): boolean { return record.expiredAt !== undefined || record.reviewStatus === 'rejected' || (record.expiresAt !== undefined && record.expiresAt <= now) }
function ancestry(record: MemoryHistoryRecord): string[] {
  const ids = record.payload?.supersedes
  return [...new Set([...(record.correctionOf ? [record.correctionOf] : []), ...(Array.isArray(ids) ? ids.filter((id): id is string => typeof id === 'string') : [])])]
}
function immutable(wire: SharedLessonWire): string { return JSON.stringify([wire.id, wire.repo, wire.cue, wire.lesson, wire.confidence, wire.evidence, [...wire.supersedes].sort(), wire.origin.host, wire.origin.agent_id, wire.recorded_at]) }
function matchesRecord(record: MemoryHistoryRecord, wire: SharedLessonWire): boolean {
  const frozen = parseSharedLessonWire(record.payload?.shared)
  if (frozen.ok) return immutable(frozen.lesson) === immutable(wire)
  const payload = record.payload
  if (!payload) return false
  const evidence = storedEvidence(payload.evidence)
  return record.id === wire.id && payload.repo === wire.repo && payload.cue === wire.cue
    && record.content === wire.lesson && payload.confidence === wire.confidence && evidence !== undefined
    && sameEvidence(lessonProvenance(evidence), wire.evidence)
    && record.agentId === wire.origin.agent_id && new Date(record.createdAt).toISOString() === wire.recorded_at
    && JSON.stringify(ancestry(record).sort()) === JSON.stringify([...wire.supersedes].sort())
    && (payload.origin_host === undefined || payload.origin_host === wire.origin.host)
}
/** Exports before 0.8.20 cut evidence to this many characters; those files must still match. */
const LEGACY_SHARED_EVIDENCE_CHARS = 1000
function sameEvidence(provenance: string, wireEvidence: string): boolean {
  return provenance.slice(0, SHARED_LESSON_LIMITS.evidence) === wireEvidence || provenance.slice(0, LEGACY_SHARED_EVIDENCE_CHARS) === wireEvidence
}
function jsonWire(wire: SharedLessonWire): JsonObject { return { ...wire, origin: { ...wire.origin }, supersedes: [...wire.supersedes] } }
function scanned(wire: SharedLessonWire): boolean { return [wire.cue, wire.lesson, wire.evidence, wire.origin.host, wire.origin.agent_id, wire.retired_reason ?? ''].some(text => scanContextContent(text, 'shared lesson') !== text) }
function message(error: unknown): string { return error instanceof Error ? error.message : String(error) }
