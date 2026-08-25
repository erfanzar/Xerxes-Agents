// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { chmodSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs'
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

export interface FileStateTrackerOptions {
  readonly maxEntries?: number
  /** Files larger than this are tracked but not snapshotted, so drift reports stay bounded. */
  readonly maxSnapshotBytes?: number
}

export interface RecordFileReadOptions {
  readonly mtimeMs: number
  readonly partialView: boolean
  readonly size: number
}

const DEFAULT_MAX_ENTRIES = 200
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
 */
export class FileStateTracker {
  private readonly entries = new Map<string, FileReadRecord>()
  private readonly maxEntries: number
  private readonly maxSnapshotBytes: number

  constructor(options: FileStateTrackerOptions = {}) {
    this.maxEntries = positiveInteger(options.maxEntries, DEFAULT_MAX_ENTRIES)
    this.maxSnapshotBytes = positiveInteger(options.maxSnapshotBytes, DEFAULT_MAX_SNAPSHOT_BYTES)
  }

  get size(): number {
    return this.entries.size
  }

  /** Latest record for one session's view of a path, refreshing its recency. */
  peek(sessionId: string, absolutePath: string): FileReadRecord | undefined {
    const key = entryKey(sessionId, absolutePath)
    const record = this.entries.get(key)
    if (record === undefined) {
      return undefined
    }
    this.entries.delete(key)
    this.entries.set(key, record)
    return record
  }

  record(sessionId: string, absolutePath: string, content: string, options: RecordFileReadOptions): void {
    const key = entryKey(sessionId, absolutePath)
    this.entries.delete(key)
    this.entries.set(key, {
      digest: digestOf(content),
      mtimeMs: options.mtimeMs,
      partialView: options.partialView,
      size: options.size,
      snapshot: this.snapshotOf(content, options.partialView),
    })
    while (this.entries.size > this.maxEntries) {
      const oldest = this.entries.keys().next().value
      if (oldest === undefined) {
        return
      }
      this.entries.delete(oldest)
    }
  }

  forget(sessionId: string, absolutePath: string): boolean {
    return this.entries.delete(entryKey(sessionId, absolutePath))
  }

  clearSession(sessionId: string): number {
    const prefix = entryKey(sessionId, '')
    let removed = 0
    for (const key of this.entries.keys()) {
      if (!key.startsWith(prefix)) {
        continue
      }
      this.entries.delete(key)
      removed += 1
    }
    return removed
  }

  clear(): void {
    this.entries.clear()
  }

  /**
   * Paths this session has read, least recently touched first.
   *
   * Exported for callers that need to know what the transcript has already shown —
   * @-mention dedup and turn-boundary diffing both need exactly this list.
   */
  pathsForSession(sessionId: string): string[] {
    const prefix = entryKey(sessionId, '')
    const paths: string[] = []
    for (const key of this.entries.keys()) {
      if (key.startsWith(prefix)) {
        paths.push(key.slice(prefix.length))
      }
    }
    return paths
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
 * across an await. The exposure is bounded in practice because ReadFile refuses files
 * over its byte ceiling, so a file large enough for the sync read to matter is one the
 * caller could not have read in the first place.
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
    return 'has not been read in this session, so ' + request.toolName + ' could overwrite changes made since you '
      + 'last looked at it; read the file first' + killSwitch
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

/** NUL separates the halves because no path may contain one, so no prefix scan can straddle them. */
function entryKey(sessionId: string, absolutePath: string): string {
  return sessionId + '\u0000' + absolutePath
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
