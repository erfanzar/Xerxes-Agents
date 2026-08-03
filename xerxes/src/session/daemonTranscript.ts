// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile, mkdir, open, readdir, readFile, rename, rm, stat } from 'node:fs/promises'
import { basename, dirname, resolve, sep } from 'node:path'

import { ValidationError } from '../core/errors.js'
import {
  RESUME_REPLAY_SENTINEL,
  repairResumedTranscript,
  type PendingResumeReplay,
} from './resumeRepair.js'

export const DAEMON_SESSION_FORMAT = 'xerxes-daemon-session'
export const DAEMON_SESSION_SCHEMA_VERSION = 2
/** Explicit replay sentinel retained in persisted messages until a host replays the interrupted call. */
export const INTERRUPTED_TOOL_RESULT = RESUME_REPLAY_SENTINEL

export type RawMessage = Record<string, unknown>

/**
 * Bytes read from the front of a transcript when only its summary fields are
 * wanted. Everything a listing renders is serialized before `messages`, so a
 * bounded head read answers the question without paying for the history.
 */
const TRANSCRIPT_HEAD_BYTES = 32 * 1024

/**
 * `JSON.stringify(record, null, 2)` writes every top-level key behind a newline
 * and exactly two spaces. A real newline can never appear inside a JSON string
 * literal, so this sequence identifies the top-level `messages` key and not a
 * nested field or a transcript that merely talks about messages.
 */
const MESSAGES_KEY_MARKER = '\n  "messages": ['

/** Journal mutations are serialized by canonical transcript path across store instances. */
const transcriptWrites = new Map<string, Promise<void>>()

export interface DaemonTranscript {
  readonly agentId: string
  /** False when imported history predates exact cumulative API-call accounting. */
  readonly apiCallsComplete?: boolean
  readonly cwd: string
  readonly extra: Readonly<Record<string, unknown>>
  readonly format: 'bun-v2' | 'legacy-v1'
  readonly interactionMode: string
  readonly key: string
  readonly messages: readonly RawMessage[]
  readonly metadata: Readonly<Record<string, unknown>>
  /** Interrupted calls discovered while repairing the loaded transcript. */
  readonly pendingResumeReplays: readonly PendingResumeReplay[]
  readonly planMode: boolean
  readonly schemaVersion: number | undefined
  readonly sessionId: string
  readonly thinkingContent: readonly unknown[]
  readonly toolExecutions: readonly unknown[]
  /** Exact provider attempts, absent for transcripts written before this metric existed. */
  readonly totalApiCalls?: number
  readonly totalInputTokens: number
  readonly totalOutputTokens: number
  readonly turnCount: number
  readonly updatedAt: string
  /** False when token usage is partial; absent when an imported transcript cannot prove completeness. */
  readonly usageComplete?: boolean
  readonly workspace: string
}

export interface TranscriptLoadOptions {
  readonly currentProjectDirectory: string
  readonly requestedSessionKey: string
  readonly workspaceRoot?: string
}

export interface DaemonTranscriptStoreOptions {
  readonly currentProjectDirectory?: string
  readonly directory: string
  readonly workspaceRoot?: string
}

/** Per-read overrides supplied by a daemon connection during initialization. */
export interface DaemonTranscriptReadOptions {
  readonly currentProjectDirectory?: string
  readonly workspaceRoot?: string
}

/** Only explicit resume IDs may load persisted state; slot keys always start fresh. */
export function looksLikeSessionId(value: string): boolean {
  return /^[0-9a-fA-F]{8,32}$/.test(value)
}

/** Normalize an unversioned Python transcript or a Bun v2 transcript without discarding unknown fields. */
export function normalizeDaemonTranscript(raw: unknown, options: TranscriptLoadOptions): DaemonTranscript | undefined {
  if (!isRecord(raw)) {
    return undefined
  }
  const messages = raw.messages
  if (!Array.isArray(messages)) {
    return undefined
  }
  // One malformed entry must not destroy the whole transcript: drop it and
  // keep the rest, since load() returning undefined would let the next save
  // atomically overwrite the persisted history.
  const validMessages = messages.filter(isRecord)
  const droppedMessages = messages.length - validMessages.length
  if (droppedMessages > 0) {
    console.warn(`Skipping ${droppedMessages} malformed message(s) in transcript ${stringValue(raw.session_id) || options.requestedSessionKey}`)
  }
  const rawSessionId = stringValue(raw.session_id) || options.requestedSessionKey
  if (!rawSessionId) {
    return undefined
  }
  const format = raw.format === DAEMON_SESSION_FORMAT ? 'bun-v2' : 'legacy-v1'
  const knownKeys = new Set([
    'format', 'schema_version', 'session_id', 'key', 'agent_id', 'cwd', 'project_dir', 'workspace', 'updated_at', 'messages',
    'message_count',
    'turn_count', 'interaction_mode', 'mode', 'plan_mode', 'api_calls_complete', 'total_api_calls', 'total_input_tokens',
    'total_output_tokens',
    'usage_complete', 'metadata',
    'thinking_content', 'tool_executions',
  ])
  const extra = Object.fromEntries(Object.entries(raw).filter(([key]) => !knownKeys.has(key)))
  const cwd = normalizeProjectDirectory(
    stringValue(raw.cwd) || stringValue(raw.project_dir) || options.currentProjectDirectory,
    options.currentProjectDirectory,
    options.workspaceRoot,
  )
  const repair = repairResumedTranscript(validMessages)
  const totalApiCalls = optionalIntegerValue(raw.total_api_calls)
  return {
    format,
    schemaVersion: numberValue(raw.schema_version),
    sessionId: rawSessionId,
    // Resume always binds to the caller's requested ID, never stale slot keys stored on disk.
    key: options.requestedSessionKey,
    agentId: stringValue(raw.agent_id) || 'default',
    ...(typeof raw.api_calls_complete === 'boolean' ? { apiCallsComplete: raw.api_calls_complete } : {}),
    cwd,
    workspace: stringValue(raw.workspace),
    updatedAt: stringValue(raw.updated_at),
    messages: repair.messages,
    pendingResumeReplays: repair.pendingReplays,
    turnCount: integerValue(raw.turn_count),
    interactionMode: stringValue(raw.interaction_mode) || stringValue(raw.mode) || 'code',
    planMode: booleanValue(raw.plan_mode),
    ...(totalApiCalls === undefined ? {} : { totalApiCalls }),
    totalInputTokens: integerValue(raw.total_input_tokens),
    totalOutputTokens: integerValue(raw.total_output_tokens),
    ...(typeof raw.usage_complete === 'boolean' ? { usageComplete: raw.usage_complete } : {}),
    metadata: isRecord(raw.metadata) ? raw.metadata : {},
    thinkingContent: Array.isArray(raw.thinking_content) ? raw.thinking_content.slice(-32) : [],
    toolExecutions: Array.isArray(raw.tool_executions) ? raw.tool_executions.slice(-200) : [],
    extra,
  }
}

/** Return repaired messages only when a legacy caller does not need replay descriptors. */
export function repairToolPairs(messages: readonly RawMessage[]): RawMessage[] {
  return [...repairResumedTranscript(messages).messages]
}

/**
 * Serialize v2 as a Python-readable superset of the legacy transcript shape.
 *
 * Every field a session listing renders is emitted before `messages` so a
 * reader that only wants the summary can stop at the head of the file. The
 * order is the only thing that changed for existing readers: they look fields
 * up by key, and `message_count` is additive.
 */
export function daemonTranscriptRecord(transcript: DaemonTranscript): Record<string, unknown> {
  return {
    ...transcript.extra,
    format: DAEMON_SESSION_FORMAT,
    schema_version: DAEMON_SESSION_SCHEMA_VERSION,
    session_id: transcript.sessionId,
    key: transcript.key,
    agent_id: transcript.agentId,
    ...(transcript.apiCallsComplete === undefined ? {} : { api_calls_complete: transcript.apiCallsComplete }),
    cwd: transcript.cwd,
    workspace: transcript.workspace,
    updated_at: transcript.updatedAt || new Date().toISOString(),
    turn_count: transcript.turnCount,
    message_count: transcript.messages.length,
    interaction_mode: transcript.interactionMode,
    plan_mode: transcript.planMode,
    ...(transcript.totalApiCalls === undefined ? {} : { total_api_calls: transcript.totalApiCalls }),
    total_input_tokens: transcript.totalInputTokens,
    total_output_tokens: transcript.totalOutputTokens,
    ...(transcript.usageComplete === undefined ? {} : { usage_complete: transcript.usageComplete }),
    metadata: transcript.metadata,
    messages: transcript.messages,
    thinking_content: transcript.thinkingContent.slice(-32),
    tool_executions: transcript.toolExecutions.slice(-200),
  }
}

/** Summary fields of one transcript, read without parsing its message history. */
export interface DaemonTranscriptHeader {
  readonly agentId: string
  readonly cwd: string
  readonly key: string
  readonly messageCount: number
  readonly metadata: Readonly<Record<string, unknown>>
  readonly sessionId: string
  readonly turnCount: number
  readonly updatedAt: string
}

/**
 * Outcome of a head read.
 *
 * `truncated` is not a failure: the record is well-formed but its summary
 * fields do not all fit in the head — a transcript written before the field
 * reorder, or one whose unknown-field prefix is unusually wide. The caller
 * falls back to a full load for those. `unreadable` means the bytes on disk
 * are not a transcript at all.
 */
export type DaemonTranscriptHeaderResult =
  | { readonly header: DaemonTranscriptHeader; readonly kind: 'header' }
  | { readonly kind: 'truncated' }
  | { readonly kind: 'unreadable' }

/** One transcript file as seen by `stat` alone. */
export interface DaemonTranscriptEntry {
  readonly modifiedAtMillis: number
  readonly path: string
  readonly sessionId: string
  readonly sizeBytes: number
}

/** Record one persisted message in the crash journal. Never throws. */
export type TranscriptMessageJournalAppend = (message: RawMessage, index: number) => void

export function transcriptHasHistory(transcript: Pick<DaemonTranscript, 'messages' | 'turnCount'>): boolean {
  return transcript.messages.length > 0 || transcript.turnCount > 0
}

/** Filesystem store for legacy-compatible daemon transcripts. */
export class DaemonTranscriptStore {
  private readonly currentProjectDirectory: string
  private readonly directory: string
  private readonly workspaceRoot: string | undefined

  constructor(options: DaemonTranscriptStoreOptions) {
    this.directory = options.directory
    this.currentProjectDirectory = options.currentProjectDirectory ?? process.cwd()
    this.workspaceRoot = options.workspaceRoot
  }

  async load(sessionKey: string, options: DaemonTranscriptReadOptions = {}): Promise<DaemonTranscript | undefined> {
    if (!looksLikeSessionId(sessionKey)) {
      return undefined
    }
    const path = this.pathFor(sessionKey)
    let raw: unknown
    try {
      raw = JSON.parse(await readFile(path, 'utf8')) as unknown
    } catch {
      return undefined
    }
    // Replay before normalization so journalled messages go through the same
    // resume repair as persisted ones: a crash between two tool calls leaves
    // an unanswered call in the journal exactly as it would in the snapshot.
    await this.replayMessageJournal(sessionKey, raw)
    const workspaceRoot = options.workspaceRoot ?? this.workspaceRoot
    return normalizeDaemonTranscript(raw, {
      currentProjectDirectory: options.currentProjectDirectory ?? this.currentProjectDirectory,
      requestedSessionKey: sessionKey,
      ...(workspaceRoot ? { workspaceRoot } : {}),
    })
  }

  /**
   * Read only the fields a listing renders, without parsing the history.
   *
   * Reports why it could not answer rather than returning undefined: a caller
   * that cannot tell "wide record" from "corrupt file" either drops real
   * sessions or reloads every file to be safe.
   */
  async readHeader(sessionId: string): Promise<DaemonTranscriptHeaderResult> {
    let head: string
    try {
      const handle = await open(this.pathFor(sessionId), 'r')
      try {
        const buffer = Buffer.allocUnsafe(TRANSCRIPT_HEAD_BYTES)
        const { bytesRead } = await handle.read(buffer, 0, TRANSCRIPT_HEAD_BYTES, 0)
        head = buffer.toString('utf8', 0, bytesRead)
      } finally {
        await handle.close()
      }
    } catch {
      return { kind: 'unreadable' }
    }
    return this.parseHeader(head, sessionId)
  }

  /** Enumerate transcript files by `stat` alone, newest modification first. */
  async listEntries(): Promise<DaemonTranscriptEntry[]> {
    let entries: string[]
    try {
      entries = await readdir(this.directory)
    } catch {
      return []
    }
    const stats = await Promise.all(
      entries
        .filter(entry => entry.endsWith('.json') && looksLikeSessionId(basename(entry, '.json')))
        .map(async (entry): Promise<DaemonTranscriptEntry | undefined> => {
          const sessionId = basename(entry, '.json')
          const path = this.pathFor(sessionId)
          try {
            const info = await stat(path)
            return { modifiedAtMillis: info.mtimeMs, path, sessionId, sizeBytes: info.size }
          } catch {
            return undefined
          }
        }),
    )
    return stats
      .filter((entry): entry is DaemonTranscriptEntry => entry !== undefined)
      .sort((left, right) => right.modifiedAtMillis - left.modifiedAtMillis)
  }

  /** Sidecar holding messages persisted since the last full save. */
  journalPathFor(sessionId: string): string {
    return `${this.pathFor(sessionId)}l`
  }

  /**
   * Append one message to the crash journal.
   *
   * A plain append with no fsync: the point is to survive a process crash for
   * the price of a `write(2)`, not to survive power loss. A torn final line is
   * expected and is discarded on replay.
   */
  async appendMessage(sessionId: string, message: RawMessage, index: number): Promise<void> {
    await this.serializeTranscriptWrite(sessionId, async () => {
      const path = this.journalPathFor(sessionId)
      await mkdir(dirname(path), { recursive: true })
      await appendFile(path, `${JSON.stringify({ index, message })}\n`, 'utf8')
    })
  }

  /**
   * A journal-append callback for a message-producing loop.
   *
   * Handed out as a plain function so the streaming library never imports
   * daemon storage, and swallowing so a failing sidecar can never abort the
   * turn whose durability it exists to improve.
   */
  journalAppender(sessionId: string): TranscriptMessageJournalAppend {
    return (message, index) => {
      void this.appendMessage(sessionId, message, index).catch((error: unknown) => {
        console.warn(`Could not journal message ${index} of session ${sessionId}: ${errorText(error)}`)
      })
    }
  }

  async save(transcript: DaemonTranscript): Promise<void> {
    if (!transcriptHasHistory(transcript)) {
      // Never delete a persisted transcript as a side effect of saving an
      // empty in-memory session. Several paths can briefly hold an empty
      // session bound to a persisted id (a failed or skipped resume, a stale
      // duplicate copy, an undo down to zero turns), and one routine save —
      // after a turn, on `/save`, or during the shutdown flush — would
      // otherwise erase the on-disk history silently. Deletion stays
      // explicit through remove().
      return
    }
    await this.serializeTranscriptWrite(transcript.sessionId, async () => {
      await atomicJsonWrite(this.pathFor(transcript.sessionId), daemonTranscriptRecord(transcript))
      // Appends cannot enter this critical section while covered entries are
      // removed. Keep entries beyond this snapshot's message count: an append
      // may have completed just before a save holding an older transcript.
      await this.discardCoveredJournalEntries(transcript.sessionId, transcript.messages.length)
    })
  }

  /**
   * Load one transcript the way a listing needs it: bound to the slot key
   * stored on disk rather than to a caller's resume id.
   *
   * `load` deliberately does the opposite, because a resume must never adopt a
   * stale slot key.
   */
  async loadForListing(sessionId: string): Promise<DaemonTranscript | undefined> {
    let raw: unknown
    try {
      raw = JSON.parse(await readFile(this.pathFor(sessionId), 'utf8')) as unknown
    } catch {
      return undefined
    }
    await this.replayMessageJournal(sessionId, raw)
    const workspaceRoot = this.workspaceRoot
    return normalizeDaemonTranscript(raw, {
      currentProjectDirectory: this.currentProjectDirectory,
      requestedSessionKey: isRecord(raw) ? stringValue(raw.key) || sessionId : sessionId,
      ...(workspaceRoot ? { workspaceRoot } : {}),
    })
  }

  /**
   * Fully parse every transcript in the directory.
   *
   * Costs one full read, parse and resume repair per file, so callers that
   * only render a bounded number of rows should walk `listEntries` and
   * `readHeader` instead.
   */
  async list(): Promise<DaemonTranscript[]> {
    const entries = await this.listEntries()
    const transcripts = await Promise.all(entries.map(entry => this.loadForListing(entry.sessionId)))
    return transcripts
      .filter((transcript): transcript is DaemonTranscript => transcript !== undefined && transcriptHasHistory(transcript))
      .sort((left, right) => timestampMillis(right.updatedAt) - timestampMillis(left.updatedAt))
  }

  /** Remove one persisted transcript by its canonical resume id. */
  async remove(sessionId: string): Promise<boolean> {
    return this.serializeTranscriptWrite(sessionId, async () => {
      const path = this.pathFor(sessionId)
      // An orphaned journal would resurrect the deleted history the next time
      // this id is opened, so it goes with the snapshot.
      await rm(this.journalPathFor(sessionId), { force: true })
      try {
        await rm(path)
        return true
      } catch (error) {
        if (isMissing(error)) return false
        throw error
      }
    })
  }

  pathFor(sessionId: string): string {
    if (!looksLikeSessionId(sessionId)) {
      throw new ValidationError('session_id', 'must be an 8-32 character hexadecimal resume ID', sessionId)
    }
    return resolve(this.directory, `${sessionId}.json`)
  }

  private async discardCoveredJournalEntries(sessionId: string, messageCount: number): Promise<void> {
    const path = this.journalPathFor(sessionId)
    let contents: string
    try {
      contents = await readFile(path, 'utf8')
    } catch (error) {
      if (isMissing(error)) return
      throw error
    }
    const retained = contents.split('\n').filter(line => {
      if (!line.trim()) return false
      try {
        const entry = JSON.parse(line) as unknown
        return isRecord(entry) && typeof entry.index === 'number' && entry.index >= messageCount
      } catch {
        // A torn final line carries no recoverable entry. Writers are excluded
        // from this critical section, so dropping it cannot race live bytes.
        return false
      }
    })
    if (retained.length === 0) {
      await rm(path, { force: true })
      return
    }
    await atomicTextWrite(path, `${retained.join('\n')}\n`)
  }

  private async serializeTranscriptWrite<T>(sessionId: string, operation: () => Promise<T>): Promise<T> {
    const key = this.pathFor(sessionId)
    const previous = transcriptWrites.get(key) ?? Promise.resolve()
    let release!: () => void
    const current = new Promise<void>(resolveOperation => { release = resolveOperation })
    transcriptWrites.set(key, current)
    await previous.catch(() => undefined)
    try {
      return await operation()
    } finally {
      release()
      if (transcriptWrites.get(key) === current) transcriptWrites.delete(key)
    }
  }

  private parseHeader(head: string, sessionId: string): DaemonTranscriptHeaderResult {
    if (!head.trimStart().startsWith('{')) {
      return { kind: 'unreadable' }
    }
    const marker = head.indexOf(MESSAGES_KEY_MARKER)
    if (marker < 0) {
      return { kind: 'truncated' }
    }
    // The cut always lands right after a `{` or a `,`, so one synthetic member
    // closes the object into parseable JSON.
    let raw: unknown
    try {
      raw = JSON.parse(`${head.slice(0, marker)}\n  "message_count_probe": 0\n}`) as unknown
    } catch {
      return { kind: 'unreadable' }
    }
    if (!isRecord(raw) || typeof raw.message_count !== 'number') {
      // Written before the summary fields were hoisted above `messages`: the
      // metadata and turn count are still behind the history.
      return { kind: 'truncated' }
    }
    return {
      kind: 'header',
      header: {
        agentId: stringValue(raw.agent_id) || 'default',
        cwd: normalizeProjectDirectory(
          stringValue(raw.cwd) || stringValue(raw.project_dir) || this.currentProjectDirectory,
          this.currentProjectDirectory,
          this.workspaceRoot,
        ),
        key: stringValue(raw.key) || sessionId,
        messageCount: integerValue(raw.message_count),
        metadata: isRecord(raw.metadata) ? raw.metadata : {},
        sessionId: stringValue(raw.session_id) || sessionId,
        turnCount: integerValue(raw.turn_count),
        updatedAt: stringValue(raw.updated_at),
      },
    }
  }

  /** Splice journalled messages onto a freshly parsed record, in place. */
  private async replayMessageJournal(sessionId: string, raw: unknown): Promise<void> {
    if (!isRecord(raw) || !Array.isArray(raw.messages)) {
      return
    }
    let contents: string
    try {
      contents = await readFile(this.journalPathFor(sessionId), 'utf8')
    } catch {
      return
    }
    const pending = new Map<number, RawMessage>()
    for (const line of contents.split('\n')) {
      if (!line.trim()) continue
      let entry: unknown
      try {
        entry = JSON.parse(line) as unknown
      } catch {
        // A crash mid-append leaves a partial final line. Everything before it
        // is intact, so stop here instead of discarding the whole journal.
        break
      }
      if (!isRecord(entry) || typeof entry.index !== 'number' || !isRecord(entry.message)) continue
      pending.set(entry.index, entry.message)
    }
    // Indexes are absolute positions in the message list, so entries the saved
    // snapshot already covers are ignored and a gap stops the replay rather
    // than reordering the transcript around a lost write.
    let next = raw.messages.length
    let replayed = 0
    for (let message = pending.get(next); message !== undefined; message = pending.get(next)) {
      raw.messages.push(message)
      replayed += 1
      next += 1
    }
    if (replayed > 0) {
      console.warn(`Recovered ${replayed} unsaved message(s) for session ${sessionId} from its crash journal`)
    }
  }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object'
    && error !== null
    && 'code' in error
    && (error as { readonly code?: unknown }).code === 'ENOENT'
}

async function atomicJsonWrite(path: string, value: Record<string, unknown>): Promise<void> {
  await atomicTextWrite(path, `${JSON.stringify(value, null, 2)}\n`)
}

async function atomicTextWrite(path: string, contents: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  const temporaryPath = resolve(dirname(path), `.${basename(path)}.${crypto.randomUUID()}.tmp`)
  try {
    const handle = await open(temporaryPath, 'w')
    try {
      await handle.writeFile(contents, 'utf8')
      await handle.sync()
    } finally {
      await handle.close()
    }
    await rename(temporaryPath, path)
    // Directory fsync flushes the rename's directory entry on POSIX. Windows
    // cannot fsync a directory handle (EPERM); the rename itself is already
    // durable enough there, so the best-effort flush is POSIX-only.
    if (process.platform !== 'win32') {
      const directoryHandle = await open(dirname(path), 'r')
      try {
        await directoryHandle.sync()
      } finally {
        await directoryHandle.close()
      }
    }
  } catch (error) {
    await rm(temporaryPath, { force: true })
    throw error
  }
}

function normalizeProjectDirectory(value: string, fallback: string, workspaceRoot: string | undefined): string {
  const resolved = resolve(value)
  if (!workspaceRoot) {
    return resolved
  }
  const resolvedWorkspace = resolve(workspaceRoot)
  return resolved === resolvedWorkspace || resolved.startsWith(`${resolvedWorkspace}${sep}`) ? resolve(fallback) : resolved
}

function functionName(value: unknown): string {
  return isRecord(value) ? stringValue(value.name) : ''
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Malformed timestamps sort as the epoch instead of producing NaN orderings. */
function timestampMillis(value: string): number {
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function integerValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.trunc(value) : 0
}

function optionalIntegerValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.trunc(value)) : undefined
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function booleanValue(value: unknown): boolean {
  return value === true
}
