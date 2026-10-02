// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { appendFileSync, chmodSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { basename, dirname, join } from 'node:path'

import { ValidationError } from '../core/errors.js'

/** The slice of ToolExecutionContext the file tools need; the full context satisfies it structurally. */
export interface FileToolContext {
  readonly sessionId?: string
}

export interface FileReadRecord {
  /** Digest of the whole file as it was when the read happened, not of the returned window. */
  readonly digest: string
  readonly mtimeMs: number
  /** True when limit/max_chars/end_line bounded the window, which disables the change report. */
  readonly partialView: boolean
  readonly size: number
  /** Whole text as read, kept only when a later change report would be cheap to produce. */
  readonly snapshot: string | undefined
}

/**
 * One entry of the read-guard state saved into session metadata. Keys are
 * short on purpose: this rides inside the session blob on every persist.
 */
export interface PersistedFileRead {
  readonly digest: string
  readonly mtime_ms: number
  readonly partial: boolean
  readonly path: string
  readonly size: number
}

/** Metadata key under which the read-guard state is stored. */
export const FILE_READS_METADATA_KEY = 'file_reads'

/** Collect the current session's read records in their persisted form. */
export function fileReadsForMetadata(
  sessionId: string | undefined,
  tracker: FileStateTracker = fileStateTracker,
): PersistedFileRead[] {
  if (sessionId === undefined || sessionId === '') return []
  return tracker.serializeSession(sessionId).map(entry => ({ ...entry }))
}

/** Restore read records saved under {@link FILE_READS_METADATA_KEY}. */
export function hydrateFileReadsFromMetadata(
  sessionId: string,
  metadata: Record<string, unknown> | undefined,
  tracker: FileStateTracker = fileStateTracker,
): number {
  if (metadata === undefined) return 0
  return tracker.hydrateSession(sessionId, metadata[FILE_READS_METADATA_KEY])
}

export interface FileStateTrackerOptions {
  /** Records kept per session before that session's least recently touched one is dropped. */
  readonly maxEntries?: number
  /** Files larger than this are tracked but not snapshotted, so drift reports stay bounded. */
  readonly maxSnapshotBytes?: number
  /** Records kept across every session before the least recently active whole session is dropped. */
  readonly maxTotalEntries?: number
}

export interface RecordFileReadOptions {
  readonly mtimeMs: number
  readonly partialView: boolean
  readonly size: number
}

const DEFAULT_MAX_ENTRIES = 200
const DEFAULT_MAX_TOTAL_ENTRIES = DEFAULT_MAX_ENTRIES * 20
const DEFAULT_MAX_SNAPSHOT_BYTES = 64 * 1_024
const MAX_REPORTED_LINES = 12
const MAX_REPORTED_LINE_CHARS = 160

/**
 * Bounded per-session record of which files were read and what they looked like.
 *
 * Keyed by session because freshness is a property of one conversation's beliefs:
 * a path another session read tells this one nothing about what it is editing.
 * Recency is maintained the same way as ToolOutputCache — delete-then-set on every
 * touch, evict from the head — so a long session cannot grow the heap without limit.
 *
 * The cap is per session rather than one LRU shared by the daemon: with a single
 * shared cap a workflow's subagents, or merely opening a saved session (which
 * hydrates up to a full cap of records), pushed a live session's reads out and
 * its next edit was refused as never-read. The process-wide ceiling is only a
 * memory backstop, and it drops whole idle sessions, never the active one.
 */
export class FileStateTracker {
  /** Session id to that session's records; both levels are kept in recency order. */
  private readonly sessions = new Map<string, Map<string, FileReadRecord>>()
  private readonly maxEntries: number
  private readonly maxSnapshotBytes: number
  private readonly maxTotalEntries: number
  private total = 0

  constructor(options: FileStateTrackerOptions = {}) {
    this.maxEntries = positiveInteger(options.maxEntries, DEFAULT_MAX_ENTRIES)
    this.maxSnapshotBytes = positiveInteger(options.maxSnapshotBytes, DEFAULT_MAX_SNAPSHOT_BYTES)
    this.maxTotalEntries = Math.max(this.maxEntries, positiveInteger(options.maxTotalEntries, DEFAULT_MAX_TOTAL_ENTRIES))
  }

  get size(): number {
    return this.total
  }

  /** Latest record for one session's view of a path, refreshing its recency. */
  peek(sessionId: string, absolutePath: string): FileReadRecord | undefined {
    const records = this.sessions.get(sessionId)
    const record = records?.get(absolutePath)
    if (records === undefined || record === undefined) {
      return undefined
    }
    records.delete(absolutePath)
    records.set(absolutePath, record)
    this.touchSession(sessionId, records)
    return record
  }

  record(sessionId: string, absolutePath: string, content: string, options: RecordFileReadOptions): void {
    this.store(sessionId, absolutePath, {
      digest: digestOf(content),
      mtimeMs: options.mtimeMs,
      partialView: options.partialView,
      size: options.size,
      snapshot: this.snapshotOf(content, options.partialView),
    })
    this.enforceTotal(sessionId)
  }

  forget(sessionId: string, absolutePath: string): boolean {
    const records = this.sessions.get(sessionId)
    if (records === undefined || !records.delete(absolutePath)) {
      return false
    }
    this.total -= 1
    if (records.size === 0) {
      this.sessions.delete(sessionId)
    }
    return true
  }

  clearSession(sessionId: string): number {
    const records = this.sessions.get(sessionId)
    if (records === undefined) {
      return 0
    }
    this.sessions.delete(sessionId)
    this.total -= records.size
    return records.size
  }

  clear(): void {
    this.sessions.clear()
    this.total = 0
  }

  /**
   * Persisted form of one read record — what session metadata stores.
   *
   * The snapshot text is deliberately excluded: metadata rides inside the
   * session blob on every save, and pinning up to 200 file bodies there would
   * bloat each write. A hydrated record still carries digest/mtime/size, so
   * the freshness guard works across restarts; only the drift diff report is
   * lost, and its fallback simply asks the model to re-read.
   */
  serializeSession(sessionId: string): readonly PersistedFileRead[] {
    const entries: PersistedFileRead[] = []
    for (const [path, record] of this.sessions.get(sessionId) ?? []) {
      entries.push({
        path,
        digest: record.digest,
        mtime_ms: record.mtimeMs,
        partial: record.partialView,
        size: record.size,
      })
    }
    return entries
  }

  /**
   * Restore records saved by serializeSession (tolerant of anything a future
   * or corrupted metadata blob might hold). Returns the count restored.
   */
  hydrateSession(sessionId: string, raw: unknown): number {
    if (!Array.isArray(raw)) return 0
    let restored = 0
    for (const entry of raw) {
      if (typeof entry !== 'object' || entry === null) continue
      const record = entry as Record<string, unknown>
      const path = record.path
      const digest = record.digest
      const mtimeMs = record.mtime_ms
      const size = record.size
      if (typeof path !== 'string' || path === '' || typeof digest !== 'string' || digest === ''
        || typeof mtimeMs !== 'number' || !Number.isFinite(mtimeMs)
        || typeof size !== 'number' || !Number.isFinite(size)) {
        continue
      }
      this.store(sessionId, path, {
        digest,
        mtimeMs,
        partialView: record.partial !== false,
        size,
        // No snapshot across a restart: drift refusals fall back to
        // "read it again" instead of a diff, which is the safe direction.
        snapshot: undefined,
      })
      restored += 1
    }
    this.enforceTotal(sessionId)
    return restored
  }

  /**
   * Paths this session has read, least recently touched first.
   *
   * Exported for callers that need to know what the transcript has already shown —
   * @-mention dedup and turn-boundary diffing both need exactly this list.
   */
  pathsForSession(sessionId: string): string[] {
    return [...(this.sessions.get(sessionId)?.keys() ?? [])]
  }

  /** Insert or refresh one record, evicting only from the same session. */
  private store(sessionId: string, absolutePath: string, record: FileReadRecord): void {
    const records = this.sessions.get(sessionId) ?? new Map<string, FileReadRecord>()
    this.touchSession(sessionId, records)
    if (records.delete(absolutePath)) {
      this.total -= 1
    }
    records.set(absolutePath, record)
    this.total += 1
    while (records.size > this.maxEntries) {
      const oldest = records.keys().next().value
      if (oldest === undefined) break
      records.delete(oldest)
      this.total -= 1
    }
  }

  private touchSession(sessionId: string, records: Map<string, FileReadRecord>): void {
    this.sessions.delete(sessionId)
    this.sessions.set(sessionId, records)
  }

  /** Drop whole least-recently-active sessions, never `activeSessionId`, until under the ceiling. */
  private enforceTotal(activeSessionId: string): void {
    for (const sessionId of [...this.sessions.keys()]) {
      if (this.total <= this.maxTotalEntries) return
      if (sessionId !== activeSessionId) this.clearSession(sessionId)
    }
  }

  /**
   * A snapshot is only worth keeping when the later change report would be both
   * cheap and meaningful: a partial view cannot be diffed against the whole file,
   * a large file would pin megabytes per entry, and a binary renders as noise.
   */
  private snapshotOf(content: string, partialView: boolean): string | undefined {
    if (partialView || content.length > this.maxSnapshotBytes || content.includes('\0')) {
      return undefined
    }
    return content
  }
}

/** Process-wide tracker used by the registered file tools. */
export const fileStateTracker = new FileStateTracker()

let configuredFreshnessEnforcement: boolean | undefined

/**
 * Let a host turn the read-before-write requirement off at runtime.
 *
 * The default permission mode is accept-all, so this check is the only place the
 * file tools refuse work a user never asked to have gated; an operator who does
 * not want it must be able to say so without editing the source.
 */
export function setFileFreshnessEnforcement(enabled: boolean | undefined): void {
  configuredFreshnessEnforcement = enabled
}

/** Environment override first, then the runtime setting, then enforced by default. */
export function isFileFreshnessEnforced(
  environment: Readonly<Record<string, string | undefined>> = process.env,
): boolean {
  return booleanFlag(environment.XERXES_FILE_FRESHNESS) ?? configuredFreshnessEnforcement ?? true
}

export type FileWriteMode = 'overwrite' | 'targeted'

export interface GuardedWriteRequest {
  readonly absolutePath: string
  /** Path as the caller named it, so refusals quote back what was asked for. */
  readonly displayPath: string
  readonly mode: FileWriteMode
  readonly sessionId: string | undefined
  readonly toolName: string
  /** Computes the new text from the bytes read inside the guarded region; must not await. */
  readonly transform: (current: string) => string
}

export interface GuardedWriteResult {
  readonly changed: boolean
  readonly next: string
  readonly previous: string
  /** Set when the file had drifted but a targeted edit was allowed through anyway. */
  readonly staleNotice: string | undefined
}

/**
 * Stat, freshness-check, transform and write one file with no await in between.
 *
 * The gap this closes is not theoretical: every edit path here used to read the
 * file, await something, and write back whatever it computed, so a file changed
 * between the two lost the other writer's work silently. Everything below runs in
 * one synchronous region, and the bytes the transform sees are the bytes that were
 * checked and the bytes that get overwritten.
 *
 * Blocking I/O is the price: there is no way to hold the check and the write together
 * across an await. The exposure is bounded because an edit needs a read first, and the
 * read tools open files only up to MAX_WINDOWED_READ_FILE_BYTES (32 MiB).
 */
export function guardedWrite(
  request: GuardedWriteRequest,
  tracker: FileStateTracker = fileStateTracker,
): GuardedWriteResult {
  const stats = statSync(request.absolutePath)
  const previous = readFileSync(request.absolutePath, 'utf8')
  const session = request.sessionId
  let staleNotice: string | undefined
  if (session !== undefined && session !== '' && isFileFreshnessEnforced()) {
    const drift = assessDrift(tracker.peek(session, request.absolutePath), stats, previous)
    if (drift !== undefined) {
      // A targeted edit still has to locate old_string in these very bytes, so the match
      // is a second net and the caller is better served by being told what moved than by
      // a refusal costing another read. A whole-file overwrite has no such net: going
      // ahead would drop the other writer's work with nothing left to recover it from.
      if (request.mode === 'overwrite' || drift.report === undefined) {
        throw new ValidationError('file_path', refusalMessage(drift, request), request.displayPath)
      }
      staleNotice = '[stale-read] ' + request.displayPath + ' changed on disk after you read it; the edit was '
        + 'applied to the current contents. What changed since your read:\n' + drift.report
    }
  }
  const next = request.transform(previous)
  const changed = next !== previous
  if (changed) {
    atomicWriteSync(request.absolutePath, next, stats.mode)
  }
  if (session !== undefined && session !== '') {
    // Record the post-write state, otherwise a second edit in the same turn would be
    // refused for drift the caller itself caused.
    const written = changed ? statSync(request.absolutePath) : stats
    tracker.record(session, request.absolutePath, next, {
      mtimeMs: written.mtimeMs,
      partialView: false,
      size: written.size,
    })
  }
  return { changed, next, previous, staleNotice }
}

export interface GuardedCreateRequest {
  readonly absolutePath: string
  readonly content: string
  readonly displayPath: string
  readonly sessionId: string | undefined
}

/**
 * Create a file that must not already exist, atomically.
 *
 * The exclusive open is the point: an exists-check followed by a plain write is the
 * same lost-update race the freshness check exists to close, only with a file that
 * appeared between the two steps instead of one that changed.
 */
export function guardedCreate(
  request: GuardedCreateRequest,
  tracker: FileStateTracker = fileStateTracker,
): void {
  try {
    writeFileSync(request.absolutePath, request.content, { flag: 'wx' })
  } catch (error) {
    if (isAlreadyExists(error)) {
      throw new ValidationError('file_path', 'already exists; pass overwrite=true to replace it', request.displayPath)
    }
    throw error
  }
  const session = request.sessionId
  if (session === undefined || session === '') {
    return
  }
  const written = statSync(request.absolutePath)
  tracker.record(session, request.absolutePath, request.content, {
    mtimeMs: written.mtimeMs,
    partialView: false,
    size: written.size,
  })
}

export interface RecordedAppendRequest {
  readonly absolutePath: string
  readonly sessionId: string | undefined
  readonly text: string
}

/**
 * Append to a file and keep the session's read record in step with its own write.
 *
 * Appending stays blind — it needs no prior read — but without this the
 * session's next edit of the file saw the append as drift and refused it, or
 * blamed it on another writer. The record is refreshed only when it was
 * current before the append (or the append created the file), so an outside
 * change made since the read is still reported rather than laundered. Like
 * guardedWrite, the check, the append and the record share one synchronous
 * region so nothing can land between them.
 */
export function recordedAppend(
  request: RecordedAppendRequest,
  tracker: FileStateTracker = fileStateTracker,
): void {
  const session = request.sessionId
  let before: string | undefined
  if (session !== undefined && session !== '') {
    const stats = statOrUndefined(request.absolutePath)
    if (stats === undefined) {
      before = ''
    } else {
      const prior = tracker.peek(session, request.absolutePath)
      if (prior !== undefined) {
        const current = readFileSync(request.absolutePath, 'utf8')
        if (assessDrift(prior, stats, current) === undefined) before = current
      }
    }
  }
  appendFileSync(request.absolutePath, request.text, 'utf8')
  if (session === undefined || session === '' || before === undefined) {
    return
  }
  const written = statSync(request.absolutePath)
  tracker.record(session, request.absolutePath, before + request.text, {
    mtimeMs: written.mtimeMs,
    partialView: false,
    size: written.size,
  })
}

function statOrUndefined(path: string): { readonly mtimeMs: number; readonly size: number } | undefined {
  try {
    return statSync(path)
  } catch (error) {
    if (typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT') return undefined
    throw error
  }
}

/** Put the drift report ahead of the tool's own summary so the model reads it first. */
export function withStaleNotice(notice: string | undefined, message: string): string {
  return notice === undefined ? message : notice + '\n\n' + message
}

/**
 * Replace a file's contents atomically: write beside it, then rename over it.
 *
 * An in-place `writeFileSync` truncates the target before the new bytes land,
 * so a failure midway through destroyed the original — the one outcome this
 * guarded path must never cause. The temp file lives in the same directory as
 * the target (rename is only atomic within one filesystem) under a unique
 * dotted name, and is removed if anything fails so no stray `.tmp` files are
 * left in the workspace. The target's permission bits are carried onto the
 * temp file before the rename: rename-over otherwise resets an executable
 * script to the process umask, silently breaking it, and swaps a hardlinked
 * file's identity without its mode.
 */
function atomicWriteSync(targetPath: string, contents: string, mode: number): void {
  const temporary = join(dirname(targetPath), `.${basename(targetPath)}.${crypto.randomUUID()}.tmp`)
  try {
    writeFileSync(temporary, contents)
    chmodSync(temporary, mode)
    renameSync(temporary, targetPath)
  } catch (error) {
    try {
      rmSync(temporary, { force: true })
    } catch {
      // Best effort: the original write error is the one worth surfacing.
    }
    throw error
  }
}

/**
 * Record what a read tool just showed the model.
 *
 * A no-op without a session: a caller with no conversation has no prior beliefs
 * about the file for a later write to be stale against.
 */
export function recordFileRead(
  context: FileToolContext | undefined,
  absolutePath: string,
  content: string,
  options: RecordFileReadOptions,
  tracker: FileStateTracker = fileStateTracker,
): void {
  const session = context?.sessionId
  if (session === undefined || session === '') {
    return
  }
  tracker.record(session, absolutePath, content, options)
}

interface FileDrift {
  readonly reason: 'modified' | 'never-read'
  /** Rendered change summary, absent when the recorded read was too partial or too large to diff. */
  readonly report: string | undefined
}

function assessDrift(
  record: FileReadRecord | undefined,
  stats: { readonly mtimeMs: number; readonly size: number },
  current: string,
): FileDrift | undefined {
  if (record === undefined) {
    return { reason: 'never-read', report: undefined }
  }
  if (record.mtimeMs >= stats.mtimeMs && record.size === stats.size) {
    return undefined
  }
  // Escape hatch: a rewrite that landed the same bytes — a formatter no-op, a checkout
  // of the revision already on disk, a save with no change — moves the mtime and means
  // nothing. The model's picture of the file is still exactly right.
  if (digestOf(current) === record.digest) {
    return undefined
  }
  return {
    reason: 'modified',
    report: record.snapshot === undefined ? undefined : describeChange(record.snapshot, current),
  }
}

function refusalMessage(drift: FileDrift, request: GuardedWriteRequest): string {
  const killSwitch = ' (set XERXES_FILE_FRESHNESS=off to disable this check)'
  if (drift.reason === 'never-read') {
    // Lead with the instruction the model can act on directly: the sentence
    // names the tool, the exact path, and the recovery step in that order.
    return request.toolName + ' requires reading "' + request.displayPath + '" first — read the file, then retry'
      + killSwitch
  }
  if (drift.report === undefined) {
    return 'changed on disk after you read it, and your read covered only part of the file so the changes cannot be '
      + 'summarised here; read it again before rewriting it' + killSwitch
  }
  return 'changed on disk after you read it, and a whole-file write would discard those changes; read it again '
    + 'before rewriting it' + killSwitch + '. What changed since your read:\n' + drift.report
}

/**
 * Line-level summary of two texts, trimming the common head and tail.
 *
 * Deliberately not a real diff: this runs on the failure path of an edit, where a
 * quadratic Myers trace would turn a mistake into a stall. Common-prefix/suffix
 * trimming is linear and pins the changed region accurately enough to retry.
 */
export function describeChange(before: string, after: string): string {
  const beforeLines = before.split('\n')
  const afterLines = after.split('\n')
  let head = 0
  while (head < beforeLines.length && head < afterLines.length && beforeLines[head] === afterLines[head]) {
    head += 1
  }
  let tail = 0
  while (
    tail < beforeLines.length - head
    && tail < afterLines.length - head
    && beforeLines[beforeLines.length - 1 - tail] === afterLines[afterLines.length - 1 - tail]
  ) {
    tail += 1
  }
  const removed = beforeLines.slice(head, beforeLines.length - tail)
  const added = afterLines.slice(head, afterLines.length - tail)
  return [
    'at line ' + (head + 1) + ': -' + removed.length + ' +' + added.length,
    ...renderSide('-', removed),
    ...renderSide('+', added),
  ].join('\n')
}

function renderSide(marker: string, lines: readonly string[]): string[] {
  const shown = lines.slice(0, MAX_REPORTED_LINES).map(line => marker + truncateLine(line))
  if (lines.length > MAX_REPORTED_LINES) {
    shown.push(marker + '… (' + (lines.length - MAX_REPORTED_LINES) + ' more lines)')
  }
  return shown
}

function truncateLine(line: string): string {
  return line.length <= MAX_REPORTED_LINE_CHARS ? line : line.slice(0, MAX_REPORTED_LINE_CHARS) + '…'
}

function isAlreadyExists(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'EEXIST'
}

function digestOf(content: string): string {
  return createHash('sha256').update(content, 'utf8').digest('hex').slice(0, 32)
}

function booleanFlag(raw: string | undefined): boolean | undefined {
  if (raw === undefined) {
    return undefined
  }
  const normalized = raw.trim().toLowerCase()
  if (normalized === '0' || normalized === 'false' || normalized === 'off' || normalized === 'no') {
    return false
  }
  if (normalized === '1' || normalized === 'true' || normalized === 'on' || normalized === 'yes') {
    return true
  }
  return undefined
}

function positiveInteger(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isInteger(value) && value > 0 ? value : fallback
}
