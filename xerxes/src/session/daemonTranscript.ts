// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, open, readdir, readFile, rename, rm } from 'node:fs/promises'
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
  if (!Array.isArray(messages) || !messages.every(isRecord)) {
    return undefined
  }
  const rawSessionId = stringValue(raw.session_id) || options.requestedSessionKey
  if (!rawSessionId) {
    return undefined
  }
  const format = raw.format === DAEMON_SESSION_FORMAT ? 'bun-v2' : 'legacy-v1'
  const knownKeys = new Set([
    'format', 'schema_version', 'session_id', 'key', 'agent_id', 'cwd', 'project_dir', 'workspace', 'updated_at', 'messages',
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
  const repair = repairResumedTranscript(messages)
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

/** Serialize v2 as a Python-readable superset of the legacy transcript shape. */
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
    messages: transcript.messages,
    turn_count: transcript.turnCount,
    interaction_mode: transcript.interactionMode,
    plan_mode: transcript.planMode,
    ...(transcript.totalApiCalls === undefined ? {} : { total_api_calls: transcript.totalApiCalls }),
    total_input_tokens: transcript.totalInputTokens,
    total_output_tokens: transcript.totalOutputTokens,
    ...(transcript.usageComplete === undefined ? {} : { usage_complete: transcript.usageComplete }),
    metadata: transcript.metadata,
    thinking_content: transcript.thinkingContent.slice(-32),
    tool_executions: transcript.toolExecutions.slice(-200),
  }
}

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
    const workspaceRoot = options.workspaceRoot ?? this.workspaceRoot
    return normalizeDaemonTranscript(raw, {
      currentProjectDirectory: options.currentProjectDirectory ?? this.currentProjectDirectory,
      requestedSessionKey: sessionKey,
      ...(workspaceRoot ? { workspaceRoot } : {}),
    })
  }

  async save(transcript: DaemonTranscript): Promise<void> {
    const path = this.pathFor(transcript.sessionId)
    if (!transcriptHasHistory(transcript)) {
      await rm(path, { force: true })
      return
    }
    await atomicJsonWrite(path, daemonTranscriptRecord(transcript))
  }

  async list(): Promise<DaemonTranscript[]> {
    let entries: string[]
    try {
      entries = await readdir(this.directory)
    } catch {
      return []
    }
    const transcripts = await Promise.all(
      entries
        .filter(entry => entry.endsWith('.json') && looksLikeSessionId(basename(entry, '.json')))
        .map(async entry => {
          const sessionId = basename(entry, '.json')
          let raw: unknown
          try {
            raw = JSON.parse(await readFile(this.pathFor(sessionId), 'utf8')) as unknown
          } catch {
            return undefined
          }
          const workspaceRoot = this.workspaceRoot
          return normalizeDaemonTranscript(raw, {
            currentProjectDirectory: this.currentProjectDirectory,
            requestedSessionKey: isRecord(raw) ? stringValue(raw.key) || sessionId : sessionId,
            ...(workspaceRoot ? { workspaceRoot } : {}),
          })
        }),
    )
    return transcripts
      .filter((transcript): transcript is DaemonTranscript => transcript !== undefined && transcriptHasHistory(transcript))
      .sort((left, right) => Date.parse(right.updatedAt || '1970-01-01') - Date.parse(left.updatedAt || '1970-01-01'))
  }

  /** Remove one persisted transcript by its canonical resume id. */
  async remove(sessionId: string): Promise<boolean> {
    const path = this.pathFor(sessionId)
    try {
      await rm(path)
      return true
    } catch (error) {
      if (isMissing(error)) return false
      throw error
    }
  }

  pathFor(sessionId: string): string {
    if (!looksLikeSessionId(sessionId)) {
      throw new ValidationError('session_id', 'must be an 8-32 character hexadecimal resume ID', sessionId)
    }
    return resolve(this.directory, `${sessionId}.json`)
  }
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object'
    && error !== null
    && 'code' in error
    && (error as { readonly code?: unknown }).code === 'ENOENT'
}

async function atomicJsonWrite(path: string, value: Record<string, unknown>): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  const temporaryPath = resolve(dirname(path), `.${basename(path)}.${crypto.randomUUID()}.tmp`)
  try {
    const handle = await open(temporaryPath, 'w')
    try {
      await handle.writeFile(`${JSON.stringify(value, null, 2)}\n`, 'utf8')
      await handle.sync()
    } finally {
      await handle.close()
    }
    await rename(temporaryPath, path)
    const directoryHandle = await open(dirname(path), 'r')
    try {
      await directoryHandle.sync()
    } finally {
      await directoryHandle.close()
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
