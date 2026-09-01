// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile, mkdir, open, readdir, readFile, rename, rm, stat, utimes } from 'node:fs/promises'
import { basename, dirname, resolve, sep } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { inspectTranscriptEventLog, type TranscriptEventInspection } from './transcriptEventInspection.js'
import {
  RESUME_REPLAY_SENTINEL,
  repairResumedTranscript,
  type PendingResumeReplay,
} from './resumeRepair.js'
import {
  encodeTranscriptEvent,
  readTranscriptEventRecords,
  transcriptMessageAppendedEvent,
  type TranscriptEvent,
  type TranscriptEventIdentity,
} from './transcriptEventLog.js'

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
  /** Byte prefix of the append-only event log covered by this snapshot. */
  readonly eventLogOffset?: number
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
    'message_count', 'generation', 'event_log_offset', 'journal_base',
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
  const eventLogOffset = optionalNonNegativeInteger(raw.event_log_offset)
  return {
    format,
    schemaVersion: numberValue(raw.schema_version),
    generation: integerValue(raw.generation),
    ...(eventLogOffset === undefined ? {} : { eventLogOffset }),
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
    ...(transcript.eventLogOffset === undefined ? {} : { event_log_offset: transcript.eventLogOffset }),
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
  if (transcript.turnCount > 0) return true
  // A completed exchange requires at least one assistant message. User-only
  // transcripts are dead attempts — a prompt whose turn never produced a
  // reply — and must neither persist nor list as sessions, or every failed
  // launch litters the session list with phantom rows.
  return transcript.messages.some(message => message.role === 'assistant')
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
    await this.appendEvent(sessionId, identity => transcriptMessageAppendedEvent(sessionId, index, message, identity))
  }

  /** Append any typed session event with one lock-authorized identity. */
  async appendEvent(
    sessionId: string,
    create: (identity: TranscriptEventIdentity) => TranscriptEvent,
  ): Promise<TranscriptEvent> {
    return this.serializeTranscriptWrite(sessionId, async () => {
      const path = this.journalPathFor(sessionId)
      await mkdir(dirname(path), { recursive: true })
      const identity = { eventId: crypto.randomUUID(), sequence: await this.nextEventSequence(sessionId) }
      const event = create(identity)
      if (event.sessionId !== sessionId || event.eventId !== identity.eventId || event.sequence !== identity.sequence) {
        throw new ValidationError('transcript_event', 'event factory must preserve its allocated session identity')
      }
      await appendFile(path, encodeTranscriptEvent(event), 'utf8')
      return event
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
      const inheritedOffset = Math.max(persisted.eventLogOffset, transcript.eventLogOffset ?? 0)
      const coveredOffset = options.mode === 'rewrite'
        ? await this.eventLogSize(transcript.sessionId)
        : inheritedOffset
      await atomicJsonWrite(
        this.pathFor(transcript.sessionId),
        daemonTranscriptRecord({ ...transcript, generation, eventLogOffset: coveredOffset, messages }),
      )
      // The snapshot above covers only events the caller actually loaded into
      // `messages`. Events appended concurrently or after that load remain
      // beyond its inherited byte watermark and replay on the next resume.
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

  /** Inspect event-log integrity without mutating persisted history. */
  async inspectEventLog(sessionId: string): Promise<
    | { readonly kind: 'inspected'; readonly report: TranscriptEventInspection }
    | { readonly kind: 'missing'; readonly sessionId: string }
  > {
    let bytes: Uint8Array
    try {
      bytes = await readFile(this.journalPathFor(sessionId))
    } catch (error) {
      if (isMissing(error)) return { kind: 'missing', sessionId }
      throw error
    }
    return { kind: 'inspected', report: inspectTranscriptEventLog(bytes, sessionId) }
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
  ): Promise<{ readonly eventLogOffset: number; readonly generation: number; readonly messages: readonly RawMessage[] }> {
    try {
      const raw = JSON.parse(await readFile(this.pathFor(sessionId), 'utf8')) as unknown
      return {
        eventLogOffset: isRecord(raw) ? optionalNonNegativeInteger(raw.event_log_offset) ?? 0 : 0,
        generation: isRecord(raw) ? integerValue(raw.generation) : 0,
        messages: isRecord(raw) && Array.isArray(raw.messages) ? raw.messages.filter(isRecord) : [],
      }
    } catch (error) {
      if (isMissing(error)) return { eventLogOffset: 0, generation: 0, messages: [] }
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
   * Allocate the next journal sequence by reading the tail, not the whole log.
   *
   * This used to read and JSON-parse the entire journal on every append, so
   * journalling a session was quadratic in its own length and a long chat
   * re-parsed a growing sidecar on every single message. The log is append-only
   * with monotonic sequences, so the highest one is in the last complete line;
   * a bounded tail read answers the same question in constant time.
   *
   * Deliberately still reading from disk rather than caching in memory: the
   * journal is a crash-recovery artifact and another writer on the same file
   * must not be allocated a colliding sequence. The full-log fallback covers a
   * final line torn by a crash mid-append, and a window too small to hold one
   * complete record.
   */
  private async nextEventSequence(sessionId: string): Promise<number> {
    const path = this.journalPathFor(sessionId)
    let size: number
    try {
      size = (await stat(path)).size
    } catch (error) {
      if (isMissing(error)) return 1
      throw error
    }
    if (size === 0) return 1

    const window = Math.min(size, NEXT_SEQUENCE_TAIL_BYTES)
    const handle = await open(path, 'r')
    let tail: Buffer
    try {
      tail = Buffer.alloc(window)
      await handle.read(tail, 0, window, size - window)
    } finally {
      await handle.close()
    }
    const highest = highestSequenceIn(tail, sessionId, window < size)
    if (highest !== undefined) return highest + 1

    // Nothing usable in the tail: fall back to the exhaustive read this
    // replaced, so a damaged or unexpectedly shaped log still allocates safely.
    const contents = await readFile(path)
    const decoded = readTranscriptEventRecords(contents, sessionId)
    let maximum = 0
    for (const event of decoded.events) {
      if (event.sequence !== undefined) maximum = Math.max(maximum, event.sequence)
    }
    return maximum + 1
  }

  private async eventLogSize(sessionId: string): Promise<number> {
    try {
      return (await stat(this.journalPathFor(sessionId))).size
    } catch (error) {
      if (isMissing(error)) return 0
      throw error
    }
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

  /** Splice uncovered append-only message events onto a freshly parsed snapshot. */
  private async replayMessageJournal(sessionId: string, raw: unknown): Promise<void> {
    if (!isRecord(raw) || !Array.isArray(raw.messages)) return
    let contents: Uint8Array
    try {
      contents = await readFile(this.journalPathFor(sessionId))
    } catch {
      return
    }
    const offset = optionalNonNegativeInteger(raw.event_log_offset) ?? 0
    const safeOffset = offset <= contents.byteLength ? offset : 0
    const decoded = readTranscriptEventRecords(contents.subarray(safeOffset), sessionId, safeOffset)

    // The first durable append for an index wins. A retry cannot rewrite
    // logical history merely by appending another row with the same index.
    const pending = new Map<number, { readonly endOffset: number; readonly message: RawMessage }>()
    for (const record of decoded.records) {
      if (record.event.type !== 'message_appended') continue
      if (!pending.has(record.event.index)) {
        pending.set(record.event.index, { endOffset: record.endOffset, message: { ...record.event.message } })
      }
    }

    // Indexes are absolute positions in the message list. The watermark moves
    // only through the contiguous records actually projected into messages;
    // gaps, malformed lines, and torn tails stay uncovered for later repair.
    let next = raw.messages.length
    let replayed = 0
    let coveredOffset = safeOffset
    for (let record = pending.get(next); record !== undefined; record = pending.get(next)) {
      raw.messages.push(record.message)
      coveredOffset = record.endOffset
      replayed += 1
      next += 1
    }
    if (replayed > 0) {
      raw.event_log_offset = coveredOffset
      console.warn(`Recovered ${replayed} unsaved message(s) for session ${sessionId} from its event log`)
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

/** Tail window for sequence allocation; comfortably larger than one record. */
const NEXT_SEQUENCE_TAIL_BYTES = 64 * 1024

/**
 * Highest sequence among the complete lines in a tail buffer.
 *
 * `partialLeadingLine` drops the first line when the window starts mid-record,
 * which would otherwise be parsed as garbage. Scans from the end and returns the
 * first line that yields a sequence, so a torn final line is skipped rather than
 * treated as the answer.
 */
function highestSequenceIn(
  tail: Buffer,
  sessionId: string,
  partialLeadingLine: boolean,
): number | undefined {
  const lines = tail.toString('utf8').split('\n').filter(line => line.trim())
  const usable = partialLeadingLine ? lines.slice(1) : lines
  for (let index = usable.length - 1; index >= 0; index -= 1) {
    const decoded = readTranscriptEventRecords(new TextEncoder().encode(`${usable[index]}\n`), sessionId)
    const sequence = decoded.events[0]?.sequence
    if (sequence !== undefined) return sequence
  }
  return undefined
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

function optionalNonNegativeInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function booleanValue(value: unknown): boolean {
  return value === true
}
