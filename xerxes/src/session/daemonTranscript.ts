// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile, mkdir, open, readdir, readFile, rename, rm, stat, utimes } from 'node:fs/promises'
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
  /** Monotonic persisted revision used to authorize transcript mutations. */
  readonly generation?: number
  readonly interactionMode: string
  readonly key: string
  readonly messages: readonly RawMessage[]
  readonly metadata: Readonly<Record<string, unknown>>
  /** Interrupted calls discovered while repairing the loaded transcript. */
  readonly pendingResumeReplays: readonly PendingResumeReplay[]
  readonly planMode: boolean
  /**
   * Length of the raw message list — including crash-journal replays — before
   * resume repair shrank it. Journal indexes are absolute positions against
   * that raw list, so journal retention is judged against this count rather
   * than the repaired length.
   */
  readonly rawMessageCount?: number
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
  /** Maximum age before an unverifiable lock may be recovered. */
  readonly lockStaleMs?: number
  /** Maximum time a writer waits for an active cross-process owner. */
  readonly lockWaitMs?: number
  readonly workspaceRoot?: string
}

export interface TranscriptSaveOptions {
  /** Append preserves independently-added suffixes; rewrite replaces history exactly. */
  readonly mode: 'append' | 'rewrite'
  /** Generation observed when the caller loaded or last saved this transcript. */
  readonly expectedGeneration: number
  /** Message count at that generation; append treats everything after it as the caller's suffix. */
  readonly expectedMessageCount?: number
  /** Receives the committed generation while the write is still serialized. */
  readonly onSavedGeneration?: (generation: number) => void
}


/** Per-read overrides supplied by a daemon connection during initialization. */
export interface DaemonTranscriptReadOptions {
  readonly currentProjectDirectory?: string
  readonly workspaceRoot?: string
}

/** Typed result that keeps absent history distinct from persisted bytes that cannot be resumed. */
export type DaemonTranscriptLoadResult =
  | { readonly kind: 'loaded'; readonly transcript: DaemonTranscript }
  | { readonly kind: 'missing' }
  | { readonly kind: 'corrupt' }

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
  // Journal indexes are absolute positions into this raw list (load() has
  // already spliced replayable entries onto it). Resume repair below may
  // shrink it, so the pre-repair length is captured here for journal
  // retention decisions; judging coverage by the repaired length would keep
  // already-covered entries and re-splice them as duplicates on the next load.
  const rawMessageCount = messages.length
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
    'message_count', 'generation',
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
    generation: integerValue(raw.generation),
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
    rawMessageCount,
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
    generation: transcript.generation ?? 0,
    turn_count: transcript.turnCount,
    message_count: transcript.messages.length,
    // Journal coverage watermark: the number of raw-list positions this
    // snapshot subsumes. Journal indexes are absolute against that raw list,
    // so a save that shrinks history must publish the new base or later
    // entries numbered from it would be misjudged as covered and deleted
    // before their messages were ever persisted. Unknown to older readers,
    // which look fields up by key.
    journal_base: transcript.messages.length,
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
  private readonly lockStaleMs: number
  private readonly lockWaitMs: number
  private readonly workspaceRoot: string | undefined

  constructor(options: DaemonTranscriptStoreOptions) {
    this.directory = options.directory
    this.currentProjectDirectory = options.currentProjectDirectory ?? process.cwd()
    this.lockStaleMs = positiveDuration(options.lockStaleMs, 30_000)
    this.lockWaitMs = positiveDuration(options.lockWaitMs, 10_000)
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

  /** Load while preserving the security-relevant distinction between absent and corrupt history. */
  async loadResult(
    sessionKey: string,
    options: DaemonTranscriptReadOptions = {},
  ): Promise<DaemonTranscriptLoadResult> {
    const transcript = await this.load(sessionKey, options)
    if (transcript) return { kind: 'loaded', transcript }
    if (!looksLikeSessionId(sessionKey)) return { kind: 'missing' }
    try {
      await stat(this.pathFor(sessionKey))
      return { kind: 'corrupt' }
    } catch (error) {
      return isMissing(error) ? { kind: 'missing' } : { kind: 'corrupt' }
    }
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

  async save(
    transcript: DaemonTranscript,
    options: TranscriptSaveOptions = {
      mode: 'append',
      expectedGeneration: transcript.generation ?? 0,
      expectedMessageCount: transcript.messages.length,
    },
  ): Promise<void> {
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
    return this.serializeTranscriptWrite(transcript.sessionId, async () => {
      const persisted = await this.readPersistedState(transcript.sessionId)
      const messages = this.resolveMessages(transcript, options, persisted)
      const generation = persisted.generation + 1
      await atomicJsonWrite(
        this.pathFor(transcript.sessionId),
        daemonTranscriptRecord({ ...transcript, generation, messages }),
      )
      // Appends cannot enter this critical section while journal coverage is
      // settled. The cutoff is the persisted coverage watermark when one
      // exists: journal indexes are absolute against the raw list of the era
      // that watermark was written in, and a shrinking save publishes a new,
      // lower base. Freezing the load-time raw length here instead would let
      // entries numbered from the shrunken base fall below the threshold and
      // be deleted although their messages were never persisted.
      const coveredThrough = Math.max(
        persisted.journalBase ?? transcript.rawMessageCount ?? messages.length,
        messages.length,
      )
      await this.discardCoveredJournalEntries(transcript.sessionId, coveredThrough, messages.length)
      options.onSavedGeneration?.(generation)
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

  private async readPersistedState(
    sessionId: string,
  ): Promise<{ readonly generation: number; readonly journalBase: number | undefined; readonly messages: readonly RawMessage[] }> {
    try {
      const raw = JSON.parse(await readFile(this.pathFor(sessionId), 'utf8')) as unknown
      return {
        generation: isRecord(raw) ? integerValue(raw.generation) : 0,
        // Snapshots written before the watermark existed have none; callers
        // fall back to their own load-time raw count for those.
        journalBase: isRecord(raw) && typeof raw.journal_base === 'number' && Number.isFinite(raw.journal_base)
          ? raw.journal_base
          : undefined,
        messages: isRecord(raw) && Array.isArray(raw.messages) ? raw.messages.filter(isRecord) : [],
      }
    } catch (error) {
      if (isMissing(error)) return { generation: 0, journalBase: undefined, messages: [] }
      throw error
    }
  }

  private resolveMessages(
    transcript: DaemonTranscript,
    options: TranscriptSaveOptions,
    persisted: { readonly generation: number; readonly messages: readonly RawMessage[] },
  ): readonly RawMessage[] {
    if (persisted.generation === options.expectedGeneration) return transcript.messages
    if (options.mode === 'rewrite') {
      throw new ValidationError('transcript_generation', 'stale rewrite conflicts with persisted history', options.expectedGeneration)
    }
    const baseCount = options.expectedMessageCount ?? transcript.messages.length
    // Both sides are compared after the same resume repair: a loaded
    // transcript has provider markers stripped from assistant content, so a
    // raw persisted prefix differs from it cosmetically while meaning the
    // same history. Stringifying raw bytes here used to fabricate divergent
    // append conflicts for legitimate marker-stripped resumes.
    if (baseCount > transcript.messages.length || !messagesEqual(
      normalizedPrefix(transcript.messages.slice(0, baseCount)),
      normalizedPrefix(persisted.messages.slice(0, baseCount)),
    )) {
      throw new ValidationError('transcript_generation', 'divergent append conflicts with persisted history', options.expectedGeneration)
    }
    const suffix = transcript.messages.slice(baseCount)
    if (suffix.length === 0) return persisted.messages
    return [...persisted.messages, ...suffix]
  }

  /**
   * Drop journal entries this snapshot subsumes and rebase the survivors into
   * the saved snapshot's coordinate system.
   *
   * Entries below `coveredThrough` are either persisted or were dropped by
   * resume repair, so they must never replay again. Survivors — entries
   * numbered from a base the snapshot has not reached — shift by the distance
   * between their era's base and the new message count, keeping the journal
   * anchored to "absolute position against the current snapshot" across
   * shrink saves.
   */
  private async discardCoveredJournalEntries(sessionId: string, coveredThrough: number, rebasedBase: number): Promise<void> {
    const path = this.journalPathFor(sessionId)
    let contents: string
    try {
      contents = await readFile(path, 'utf8')
    } catch (error) {
      if (isMissing(error)) return
      throw error
    }
    const retained = contents.split('\n').flatMap(line => {
      if (!line.trim()) return []
      let entry: unknown
      try {
        entry = JSON.parse(line) as unknown
      } catch {
        // A torn final line carries no recoverable entry. Writers are excluded
        // from this critical section, so dropping it cannot race live bytes.
        return []
      }
      if (!isRecord(entry) || typeof entry.index !== 'number' || entry.index < coveredThrough) return []
      return [JSON.stringify({ index: entry.index - coveredThrough + rebasedBase, message: entry.message })]
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
      return await withFileLock(`${key}.lock`, operation, {
        staleMs: this.lockStaleMs,
        waitMs: this.lockWaitMs,
      })
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

function messagesEqual(left: readonly RawMessage[], right: readonly RawMessage[]): boolean {
  return left.length === right.length
    && left.every((message, index) => JSON.stringify(message) === JSON.stringify(right[index]))
}

/** Normalize a message slice through the same resume repair a load applies. */
function normalizedPrefix(messages: readonly RawMessage[]): readonly RawMessage[] {
  return repairResumedTranscript(messages).messages
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

export interface FileLockOptions {
  /** Owner description used in wait-timeout errors; defaults to 'transcript'. */
  readonly label?: string
  readonly staleMs: number
  readonly waitMs: number
}

interface FileLockMetadata {
  readonly createdAt: number
  readonly pid: number
  readonly token: string
}

/**
 * Exclusive cross-process lock file protocol, shared by the transcript store,
 * the SQLite session store, and agent self-memory: an exclusive `'wx'`
 * creation loop with a bounded wait deadline, stale-owner takeover based on a
 * metadata heartbeat rather than PID liveness, and token-checked cleanup so a
 * deposed owner can never delete a replacement lock.
 */
export async function withFileLock<T>(path: string, operation: () => Promise<T>, options: FileLockOptions): Promise<T> {
  await mkdir(dirname(path), { recursive: true })
  const deadline = performance.now() + options.waitMs
  for (;;) {
    const token = crypto.randomUUID()
    let handle
    try {
      handle = await open(path, 'wx')
    } catch (error) {
      if (!isAlreadyExists(error)) throw error
      await removeAbandonedLock(path, options.staleMs)
      if (performance.now() >= deadline) {
        throw new Error(`Timed out waiting for ${options.label ?? 'transcript'} lock ${path} after ${options.waitMs}ms`)
      }
      await Bun.sleep(Math.min(5, Math.max(1, deadline - performance.now())))
      continue
    }
    const metadata: FileLockMetadata = { createdAt: Date.now(), pid: process.pid, token }
    try {
      await handle.writeFile(`${JSON.stringify(metadata)}\n`, 'utf8')
    } finally {
      await handle.close()
    }
    // Age is based on a heartbeat rather than PID liveness. This handles PID
    // reuse while ensuring a genuinely active long operation is never reaped.
    const heartbeat = setInterval(() => {
      void refreshOwnedLock(path, token)
    }, Math.max(10, Math.floor(options.staleMs / 3)))
    heartbeat.unref?.()
    try {
      return await operation()
    } finally {
      clearInterval(heartbeat)
      await removeOwnedLock(path, token)
    }
  }
}

function isAlreadyExists(error: unknown): boolean {
  return typeof error === 'object'
    && error !== null
    && 'code' in error
    && (error as { readonly code?: unknown }).code === 'EEXIST'
}

async function refreshOwnedLock(path: string, token: string): Promise<void> {
  try {
    if (lockToken(await readFile(path, 'utf8')) !== token) return
    const now = new Date()
    await utimes(path, now, now)
  } catch {
    // A heartbeat is best effort. The token check in cleanup prevents this
    // owner from deleting a replacement if its lock was externally removed.
  }
}

async function removeOwnedLock(path: string, token: string): Promise<void> {
  try {
    if (lockToken(await readFile(path, 'utf8')) === token) await rm(path)
  } catch (error) {
    if (!isMissing(error)) throw error
  }
}

async function removeAbandonedLock(path: string, staleMs: number): Promise<void> {
  let observed: string
  let modifiedAt: number
  try {
    observed = await readFile(path, 'utf8')
    modifiedAt = (await stat(path)).mtimeMs
  } catch {
    return
  }
  const metadata = lockMetadata(observed)
  const lastEvidence = Math.max(modifiedAt, metadata?.createdAt ?? 0)
  if (Date.now() - lastEvidence < staleMs) return

  // Rename first so cleanup applies to the exact lock we inspected. If its
  // contents changed during inspection, put it back rather than deleting a
  // newly initialized or refreshed owner.
  const abandonedPath = `${path}.${crypto.randomUUID()}.abandoned`
  try {
    await rename(path, abandonedPath)
  } catch {
    return
  }
  try {
    if (await readFile(abandonedPath, 'utf8') !== observed) {
      try {
        await rename(abandonedPath, path)
      } catch {
        // Another contender already established a lock; retain neither a
        // misleading lock at the canonical path nor an unbounded side file.
        await rm(abandonedPath, { force: true })
      }
      return
    }
    await rm(abandonedPath, { force: true })
  } catch (error) {
    await rm(abandonedPath, { force: true })
    if (!isMissing(error)) throw error
  }
}

function lockMetadata(contents: string): FileLockMetadata | undefined {
  try {
    const value = JSON.parse(contents) as unknown
    if (!isRecord(value)
      || !Number.isSafeInteger(value.pid) || (value.pid as number) <= 0
      || typeof value.createdAt !== 'number' || !Number.isFinite(value.createdAt)
      || typeof value.token !== 'string' || value.token.length === 0) return undefined
    return { createdAt: value.createdAt, pid: value.pid as number, token: value.token }
  } catch {
    return undefined
  }
}

function lockToken(contents: string): string | undefined {
  return lockMetadata(contents)?.token
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

function positiveDuration(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.trunc(value) : fallback
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
