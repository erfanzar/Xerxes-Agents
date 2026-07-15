// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  DAEMON_SESSION_SCHEMA_VERSION,
  daemonTranscriptRecord,
  normalizeDaemonTranscript,
  type DaemonTranscript,
  type RawMessage,
} from '../session/daemonTranscript.js'
import {
  repairResumedTranscript,
  type PendingResumeReplay,
  type ResumeRepairResult,
} from '../session/resumeRepair.js'
import { stripAssistantToolCallMarkers } from '../streaming/toolMarkers.js'

/** Explicit storage boundary for bridge-owned transcript records. */
export interface BridgeSessionStore {
  read(sessionId: string): Promise<unknown | undefined> | unknown | undefined
  write(sessionId: string, record: Readonly<Record<string, unknown>>): Promise<void> | void
}

/** Inputs required to create one transport-free bridge session. */
export interface BridgeSessionOptions {
  readonly clock: () => Date
  readonly cwd: string
  readonly sessionId: string
  readonly store: BridgeSessionStore
  readonly agentId?: string
  readonly interactionMode?: string
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly model?: string
  readonly planMode?: boolean
  readonly workspace?: string
  readonly workspaceRoot?: string
}

/** Mutable runtime values that a bridge host may update between turns. */
export interface BridgeSessionUpdate {
  readonly agentId?: string
  readonly apiCallsComplete?: boolean
  readonly cwd?: string
  readonly extra?: Readonly<Record<string, unknown>>
  readonly interactionMode?: string
  readonly messages?: readonly RawMessage[]
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly model?: string
  readonly planMode?: boolean
  readonly thinkingContent?: readonly unknown[]
  readonly toolExecutions?: readonly unknown[]
  readonly totalApiCalls?: number
  readonly totalInputTokens?: number
  readonly totalOutputTokens?: number
  readonly turnCount?: number
  readonly usageComplete?: boolean
  readonly workspace?: string
}

/** Read-only bridge state, including resume descriptors that a host may inspect. */
export interface BridgeSessionSnapshot {
  readonly agentId: string
  readonly apiCallsComplete?: boolean
  readonly createdAt: string
  readonly cwd: string
  readonly extra: Readonly<Record<string, unknown>>
  readonly interactionMode: string
  readonly messages: readonly RawMessage[]
  readonly metadata: Readonly<Record<string, unknown>>
  readonly model: string
  readonly pendingResumeReplays: readonly PendingResumeReplay[]
  readonly planMode: boolean
  readonly sessionId: string
  readonly thinkingContent: readonly unknown[]
  readonly toolExecutions: readonly unknown[]
  readonly totalApiCalls?: number
  readonly totalInputTokens: number
  readonly totalOutputTokens: number
  readonly turnCount: number
  readonly updatedAt: string
  readonly usageComplete?: boolean
  readonly workspace: string
}

/** A transport-neutral projection of the Python bridge's history notifications. */
export interface BridgeHistoryReplayRecord {
  readonly body: string
  readonly category: 'history'
  readonly severity: 'info'
  readonly type: 'replay_assistant' | 'replay_user' | 'resumed'
}

/** Outcome of attempting to load a persisted bridge transcript. */
export type BridgeSessionResumeResult =
  | {
    readonly history: readonly BridgeHistoryReplayRecord[]
    readonly pendingResumeReplays: readonly PendingResumeReplay[]
    readonly status: 'invalid' | 'missing'
  }
  | {
    readonly history: readonly BridgeHistoryReplayRecord[]
    readonly pendingResumeReplays: readonly PendingResumeReplay[]
    readonly status: 'resumed'
  }

interface BridgeSessionState {
  agentId: string
  apiCallsComplete?: boolean
  createdAt: string
  cwd: string
  extra: Record<string, unknown>
  interactionMode: string
  messages: RawMessage[]
  metadata: Record<string, unknown>
  model: string
  pendingResumeReplays: PendingResumeReplay[]
  planMode: boolean
  sessionId: string
  thinkingContent: unknown[]
  toolExecutions: unknown[]
  totalApiCalls?: number
  totalInputTokens: number
  totalOutputTokens: number
  turnCount: number
  updatedAt: string
  usageComplete?: boolean
  workspace: string
}

/**
 * Bridge-local durable conversation state.
 *
 * This owns serialization, loading, history projection, and structural repair
 * only. A transport decides how to deliver history records, and a host must
 * explicitly elect to execute any {@link PendingResumeReplay} descriptors.
 */
export class BridgeSession {
  private readonly clock: () => Date
  private readonly store: BridgeSessionStore
  private readonly workspaceRoot: string | undefined
  private state: BridgeSessionState

  constructor(options: BridgeSessionOptions) {
    const now = timestamp(options.clock)
    this.clock = options.clock
    this.store = options.store
    this.workspaceRoot = options.workspaceRoot
    this.state = {
      agentId: options.agentId ?? 'default',
      apiCallsComplete: true,
      createdAt: now,
      cwd: requiredText(options.cwd, 'cwd'),
      extra: {},
      interactionMode: options.interactionMode ?? 'code',
      messages: [],
      metadata: { ...options.metadata },
      model: options.model ?? '',
      pendingResumeReplays: [],
      planMode: options.planMode ?? false,
      sessionId: requiredText(options.sessionId, 'sessionId'),
      thinkingContent: [],
      toolExecutions: [],
      totalApiCalls: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      turnCount: 0,
      updatedAt: now,
      usageComplete: true,
      workspace: options.workspace ?? '',
    }
  }

  /** Snapshot current bridge state without exposing mutable arrays or records. */
  get snapshot(): BridgeSessionSnapshot {
    return snapshotOf(this.state)
  }

  /** Add one message to the current transcript without executing or repairing tool calls. */
  appendMessage(message: RawMessage): void {
    this.state.messages.push(copyMessage(message))
  }

  /** Replace bridge-owned state fields supplied by the active runtime. */
  update(update: BridgeSessionUpdate): void {
    if (update.agentId !== undefined) this.state.agentId = update.agentId
    if (update.apiCallsComplete !== undefined) this.state.apiCallsComplete = update.apiCallsComplete
    if (update.cwd !== undefined) this.state.cwd = requiredText(update.cwd, 'cwd')
    if (update.extra !== undefined) this.state.extra = { ...update.extra }
    if (update.interactionMode !== undefined) this.state.interactionMode = update.interactionMode
    if (update.messages !== undefined) {
      this.state.messages = copyMessages(update.messages)
      this.state.pendingResumeReplays = []
    }
    if (update.metadata !== undefined) this.state.metadata = { ...update.metadata }
    if (update.model !== undefined) this.state.model = update.model
    if (update.planMode !== undefined) this.state.planMode = update.planMode
    if (update.thinkingContent !== undefined) this.state.thinkingContent = [...update.thinkingContent]
    if (update.toolExecutions !== undefined) this.state.toolExecutions = [...update.toolExecutions]
    if (update.totalApiCalls !== undefined) {
      this.state.totalApiCalls = counter(update.totalApiCalls, 'totalApiCalls')
    }
    if (update.totalInputTokens !== undefined) {
      this.state.totalInputTokens = counter(update.totalInputTokens, 'totalInputTokens')
    }
    if (update.totalOutputTokens !== undefined) {
      this.state.totalOutputTokens = counter(update.totalOutputTokens, 'totalOutputTokens')
    }
    if (update.turnCount !== undefined) this.state.turnCount = counter(update.turnCount, 'turnCount')
    if (update.usageComplete !== undefined) this.state.usageComplete = update.usageComplete
    if (update.workspace !== undefined) this.state.workspace = update.workspace
  }

  /** Repair the current transcript and retain descriptors for explicitly host-owned replay. */
  repairInterruptedToolCalls(): readonly PendingResumeReplay[] {
    this.adoptRepair(repairResumedTranscript(this.state.messages))
    return copyPendingReplays(this.state.pendingResumeReplays)
  }

  /** Apply a repaired transcript returned by an explicit host-side replay workflow. */
  adoptRepair(repair: ResumeRepairResult): void {
    this.state.messages = copyMessages(repair.messages)
    this.state.pendingResumeReplays = copyPendingReplays(repair.pendingReplays)
  }

  /** Build a Python-readable Bun v2 transcript record without performing I/O. */
  toRecord(): Record<string, unknown> {
    return daemonTranscriptRecord(this.toTranscript())
  }

  /** Persist the current transcript through the injected storage port. */
  async save(): Promise<Readonly<Record<string, unknown>>> {
    this.state.updatedAt = timestamp(this.clock)
    const record = this.toRecord()
    await this.store.write(this.state.sessionId, record)
    return record
  }

  /**
   * Load and structurally repair a persisted transcript through the injected storage port.
   *
   * Missing and invalid records leave this session untouched. The result exposes
   * pending tool-call descriptors, but it intentionally does not execute them.
   */
  async resume(sessionId: string): Promise<BridgeSessionResumeResult> {
    const requestedSessionId = requiredText(sessionId, 'sessionId')
    const raw = await this.store.read(requestedSessionId)
    if (raw === undefined) {
      return { status: 'missing', history: [], pendingResumeReplays: [] }
    }
    const transcript = normalizeDaemonTranscript(raw, {
      currentProjectDirectory: this.state.cwd,
      requestedSessionKey: requestedSessionId,
      ...(this.workspaceRoot === undefined ? {} : { workspaceRoot: this.workspaceRoot }),
    })
    if (transcript === undefined) {
      return { status: 'invalid', history: [], pendingResumeReplays: [] }
    }
    this.adoptTranscript(transcript)
    return {
      status: 'resumed',
      history: this.historyReplayRecords(),
      pendingResumeReplays: copyPendingReplays(this.state.pendingResumeReplays),
    }
  }

  /** Return prior user/assistant content plus the final session-resumed marker. */
  historyReplayRecords(): BridgeHistoryReplayRecord[] {
    return bridgeHistoryReplayRecords(this.state.messages, this.state.sessionId)
  }

  private adoptTranscript(transcript: DaemonTranscript): void {
    const extra = { ...transcript.extra }
    this.state = {
      agentId: transcript.agentId,
      ...(transcript.apiCallsComplete === undefined ? {} : { apiCallsComplete: transcript.apiCallsComplete }),
      createdAt: recordText(extra.created_at) || this.state.createdAt,
      cwd: transcript.cwd,
      extra,
      interactionMode: transcript.interactionMode,
      messages: copyMessages(transcript.messages),
      metadata: { ...transcript.metadata },
      model: recordText(extra.model) || this.state.model,
      pendingResumeReplays: copyPendingReplays(transcript.pendingResumeReplays),
      planMode: transcript.planMode,
      sessionId: transcript.sessionId,
      thinkingContent: [...transcript.thinkingContent],
      toolExecutions: [...transcript.toolExecutions],
      ...(transcript.totalApiCalls === undefined ? {} : { totalApiCalls: transcript.totalApiCalls }),
      totalInputTokens: transcript.totalInputTokens,
      totalOutputTokens: transcript.totalOutputTokens,
      turnCount: transcript.turnCount,
      updatedAt: transcript.updatedAt || this.state.updatedAt,
      ...(transcript.usageComplete === undefined ? {} : { usageComplete: transcript.usageComplete }),
      workspace: transcript.workspace,
    }
  }

  private toTranscript(): DaemonTranscript {
    return {
      agentId: this.state.agentId,
      ...(this.state.apiCallsComplete === undefined ? {} : { apiCallsComplete: this.state.apiCallsComplete }),
      cwd: this.state.cwd,
      extra: {
        ...this.state.extra,
        created_at: this.state.createdAt,
        model: this.state.model,
      },
      format: 'bun-v2',
      interactionMode: this.state.interactionMode,
      key: this.state.sessionId,
      messages: copyMessages(this.state.messages),
      metadata: { ...this.state.metadata },
      pendingResumeReplays: copyPendingReplays(this.state.pendingResumeReplays),
      planMode: this.state.planMode,
      schemaVersion: DAEMON_SESSION_SCHEMA_VERSION,
      sessionId: this.state.sessionId,
      thinkingContent: [...this.state.thinkingContent],
      toolExecutions: [...this.state.toolExecutions],
      ...(this.state.totalApiCalls === undefined ? {} : { totalApiCalls: this.state.totalApiCalls }),
      totalInputTokens: this.state.totalInputTokens,
      totalOutputTokens: this.state.totalOutputTokens,
      turnCount: this.state.turnCount,
      updatedAt: this.state.updatedAt,
      ...(this.state.usageComplete === undefined ? {} : { usageComplete: this.state.usageComplete }),
      workspace: this.state.workspace,
    }
  }
}

/** Build transport-neutral history replay records from a saved transcript. */
export function bridgeHistoryReplayRecords(
  messages: readonly RawMessage[],
  sessionId: string,
): BridgeHistoryReplayRecord[] {
  const records: BridgeHistoryReplayRecord[] = []
  for (const message of messages) {
    const role = recordText(message.role).toLowerCase()
    const text = replayText(message.content).trim()
    if (!text) continue
    if (role === 'user') {
      records.push({ category: 'history', type: 'replay_user', severity: 'info', body: `✨ ${text}` })
      continue
    }
    if (role === 'assistant') {
      records.push({
        category: 'history',
        type: 'replay_assistant',
        severity: 'info',
        body: stripAssistantToolCallMarkers(text),
      })
    }
  }
  records.push({
    category: 'history',
    type: 'resumed',
    severity: 'info',
    body: `── resumed session ${sessionId} (${records.length} messages) ──`,
  })
  return records
}

function snapshotOf(state: BridgeSessionState): BridgeSessionSnapshot {
  return {
    agentId: state.agentId,
    ...(state.apiCallsComplete === undefined ? {} : { apiCallsComplete: state.apiCallsComplete }),
    createdAt: state.createdAt,
    cwd: state.cwd,
    extra: { ...state.extra },
    interactionMode: state.interactionMode,
    messages: copyMessages(state.messages),
    metadata: { ...state.metadata },
    model: state.model,
    pendingResumeReplays: copyPendingReplays(state.pendingResumeReplays),
    planMode: state.planMode,
    sessionId: state.sessionId,
    thinkingContent: [...state.thinkingContent],
    toolExecutions: [...state.toolExecutions],
    ...(state.totalApiCalls === undefined ? {} : { totalApiCalls: state.totalApiCalls }),
    totalInputTokens: state.totalInputTokens,
    totalOutputTokens: state.totalOutputTokens,
    turnCount: state.turnCount,
    updatedAt: state.updatedAt,
    ...(state.usageComplete === undefined ? {} : { usageComplete: state.usageComplete }),
    workspace: state.workspace,
  }
}

function copyMessages(messages: readonly RawMessage[]): RawMessage[] {
  return messages.map(copyMessage)
}

function copyMessage(message: RawMessage): RawMessage {
  return { ...message }
}

function copyPendingReplays(replays: readonly PendingResumeReplay[]): PendingResumeReplay[] {
  return replays.map(replay => ({ ...replay }))
}

function counter(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new TypeError(`${name} must be a non-negative integer`)
  }
  return value
}

function recordText(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function replayText(content: unknown): string {
  if (Array.isArray(content)) {
    return content
      .map(part => isRecord(part) ? String(part.text ?? '') : String(part))
      .filter(Boolean)
      .join('\n')
  }
  if (isRecord(content)) return String(content.text ?? '')
  return content === null || content === undefined ? '' : String(content)
}

function requiredText(value: string, name: string): string {
  const normalized = value.trim()
  if (!normalized) throw new TypeError(`${name} must not be empty`)
  return normalized
}

function timestamp(clock: () => Date): string {
  const value = clock()
  if (!(value instanceof Date) || Number.isNaN(value.valueOf())) {
    throw new TypeError('clock must return a valid Date')
  }
  return value.toISOString()
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
