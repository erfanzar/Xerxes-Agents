// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

import { detectProvider } from '../llms/providerRegistry.js'
import { CostEvent, CostTracker, type CostEventRecord } from './costTracker.js'
import { HistoryLog } from './history.js'
import { TranscriptStore } from './transcript.js'

export const DEFAULT_RUNTIME_SESSION_DIRECTORY = '.xerxes_sessions'
export const MAX_PERSISTED_STREAM_EVENTS = 100
export const MAX_TOOL_EXECUTION_RESULT_CHARS = 500
export const MAX_HISTORY_TOOL_RESULT_CHARS = 100

export interface RuntimeContextHost {
  cwd(): string
  gitBranch(cwd: string): string
  now(): Date
  platform(): string
  runtimeVersion(): string
}

/** Explicit synchronous boundary for session JSON persistence. */
export interface RuntimeSessionFileSystem {
  makeDirectory(path: string): void
  readFile(path: string): string
  writeFile(path: string, contents: string): void
}

export interface RuntimeContextOptions {
  readonly cwd?: string
  readonly gitBranch?: string
  readonly model?: string
  readonly platformName?: string
  readonly provider?: string
  readonly runtimeVersion?: string
  readonly timestamp: string
}

export interface RuntimeContextCaptureOptions {
  readonly host: RuntimeContextHost
  readonly model?: string
  readonly provider?: string
}

export interface RuntimeContextRecord {
  readonly cwd: string
  readonly git_branch: string
  readonly model: string
  readonly platform_name: string
  readonly provider: string
  readonly runtime_version: string
  readonly timestamp: string
}

/** Immutable environment snapshot captured at the beginning of a runtime session. */
export class RuntimeContext {
  readonly cwd: string
  readonly gitBranch: string
  readonly model: string
  readonly platformName: string
  readonly provider: string
  readonly runtimeVersion: string
  readonly timestamp: string

  constructor(options: RuntimeContextOptions) {
    this.cwd = textValue(options.cwd ?? '', 'cwd')
    this.runtimeVersion = textValue(options.runtimeVersion ?? '', 'runtimeVersion')
    this.platformName = textValue(options.platformName ?? '', 'platformName')
    this.gitBranch = textValue(options.gitBranch ?? '', 'gitBranch')
    this.model = textValue(options.model ?? '', 'model')
    this.provider = textValue(options.provider ?? '', 'provider')
    this.timestamp = timestampValue(options.timestamp)
    Object.freeze(this)
  }

  static capture(options: RuntimeContextCaptureOptions): RuntimeContext {
    const model = textValue(options.model ?? '', 'model')
    const cwd = textValue(options.host.cwd(), 'host.cwd()')
    const provider = options.provider === undefined
      ? model ? detectProvider(model) : ''
      : textValue(options.provider, 'provider')
    return new RuntimeContext({
      cwd,
      runtimeVersion: textValue(options.host.runtimeVersion(), 'host.runtimeVersion()'),
      platformName: textValue(options.host.platform(), 'host.platform()'),
      gitBranch: capturedGitBranch(options.host, cwd),
      model,
      provider,
      timestamp: isoTimestamp(options.host.now()),
    })
  }

  static fromRecord(value: unknown): RuntimeContext {
    const record = objectValue(value, 'context')
    return new RuntimeContext({
      cwd: requiredText(record, 'cwd'),
      runtimeVersion: requiredText(record, 'runtime_version'),
      platformName: requiredText(record, 'platform_name'),
      gitBranch: requiredText(record, 'git_branch'),
      model: requiredText(record, 'model'),
      provider: requiredText(record, 'provider'),
      timestamp: requiredText(record, 'timestamp'),
    })
  }

  toRecord(): RuntimeContextRecord {
    return Object.freeze({
      cwd: this.cwd,
      runtime_version: this.runtimeVersion,
      platform_name: this.platformName,
      git_branch: this.gitBranch,
      model: this.model,
      provider: this.provider,
      timestamp: this.timestamp,
    })
  }
}

export interface RuntimeStreamEvent {
  readonly timestamp: string
  readonly type: string
  readonly [key: string]: unknown
}

export interface RuntimeToolExecutionRecord {
  readonly duration_ms: number
  readonly inputs: unknown
  readonly permitted: boolean
  readonly result: string
  readonly timestamp: string
  readonly tool: string
}

export interface RuntimeSessionOptions {
  readonly context: RuntimeContext
  readonly costTracker?: CostTracker
  readonly history?: HistoryLog
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly now?: () => Date
  readonly prompt?: string
  readonly sessionId: string
  readonly streamEvents?: readonly RuntimeStreamEvent[]
  readonly toolExecutions?: readonly RuntimeToolExecutionRecord[]
  readonly transcript?: TranscriptStore
}

export interface RuntimeSessionCreateOptions {
  readonly contextHost: RuntimeContextHost
  readonly costTracker?: CostTracker
  readonly history?: HistoryLog
  readonly idFactory?: () => string
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly model?: string
  readonly prompt?: string
  readonly provider?: string
  readonly sessionId?: string
  readonly transcript?: TranscriptStore
}

export interface RuntimeSessionSaveOptions {
  readonly directory?: string
  readonly fileSystem: RuntimeSessionFileSystem
}

export interface RuntimeSessionLoadOptions {
  readonly fileSystem: RuntimeSessionFileSystem
  readonly now?: () => Date
  readonly path: string
}

export interface RuntimeSessionRecord {
  readonly context: RuntimeContextRecord
  readonly costs: readonly CostEventRecord[]
  readonly history: readonly Record<string, unknown>[]
  readonly messages: readonly Record<string, unknown>[]
  readonly metadata: Readonly<Record<string, unknown>>
  readonly prompt: string
  readonly session_id: string
  readonly stream_events: readonly RuntimeStreamEvent[]
  readonly tool_executions: readonly RuntimeToolExecutionRecord[]
}

/**
 * Standalone durable state for a long-lived agent conversation.
 *
 * The session deliberately composes the existing transcript, history, and
 * cost ledgers instead of mirroring daemon session records. Callers supply
 * context and persistence hosts, making it usable in a CLI, worker, or test
 * without ambient daemon state.
 */
export class RuntimeSession {
  readonly context: RuntimeContext
  readonly costTracker: CostTracker
  readonly history: HistoryLog
  readonly metadata: Readonly<Record<string, unknown>>
  readonly prompt: string
  readonly sessionId: string
  readonly transcript: TranscriptStore
  private readonly clock: () => Date
  private readonly streamEventLedger: RuntimeStreamEvent[]
  private readonly toolExecutionLedger: RuntimeToolExecutionRecord[]

  constructor(options: RuntimeSessionOptions) {
    if (!(options.context instanceof RuntimeContext)) throw new TypeError('context must be a RuntimeContext')
    this.sessionId = sessionIdValue(options.sessionId)
    this.prompt = textValue(options.prompt ?? '', 'prompt')
    this.context = options.context
    this.clock = options.now ?? (() => new Date())
    this.transcript = options.transcript ?? new TranscriptStore({ now: this.clock })
    this.history = options.history ?? new HistoryLog({ now: this.clock })
    this.costTracker = options.costTracker ?? new CostTracker({ now: this.clock, sessionId: this.sessionId })
    this.metadata = freezeRecord(options.metadata ?? {})
    this.streamEventLedger = (options.streamEvents ?? []).map(copyStreamEvent)
    this.toolExecutionLedger = (options.toolExecutions ?? []).map(copyToolExecution)
  }

  static create(options: RuntimeSessionCreateOptions): RuntimeSession {
    const model = textValue(options.model ?? '', 'model')
    const sessionId = options.sessionId ?? (options.idFactory ?? defaultSessionId)()
    const context = RuntimeContext.capture({
      host: options.contextHost,
      model,
      ...(options.provider === undefined ? {} : { provider: options.provider }),
    })
    return new RuntimeSession({
      sessionId,
      context,
      now: options.contextHost.now,
      ...(options.prompt === undefined ? {} : { prompt: options.prompt }),
      ...(options.transcript === undefined ? {} : { transcript: options.transcript }),
      ...(options.history === undefined ? {} : { history: options.history }),
      ...(options.costTracker === undefined ? {} : { costTracker: options.costTracker }),
      ...(options.metadata === undefined ? {} : { metadata: options.metadata }),
    })
  }

  static load(options: RuntimeSessionLoadOptions): RuntimeSession {
    const parsed = parseSessionRecord(options.fileSystem.readFile(options.path))
    const context = RuntimeContext.fromRecord(parsed.context)
    const clock = options.now ?? (() => new Date())
    const sessionId = sessionIdValue(requiredText(parsed, 'session_id'))
    const transcript = restoreTranscript(parsed.messages, clock)
    const history = restoreHistory(parsed.history, context.timestamp)
    const costTracker = restoreCostTracker(parsed.costs, clock, sessionId)
    return new RuntimeSession({
      sessionId,
      prompt: requiredText(parsed, 'prompt'),
      context,
      transcript,
      history,
      costTracker,
      metadata: objectValue(parsed.metadata, 'metadata'),
      streamEvents: streamEventsFrom(parsed.stream_events),
      toolExecutions: toolExecutionsFrom(parsed.tool_executions),
      now: clock,
    })
  }

  get streamEvents(): readonly RuntimeStreamEvent[] {
    return this.streamEventLedger.map(copyStreamEvent)
  }

  get toolExecutions(): readonly RuntimeToolExecutionRecord[] {
    return this.toolExecutionLedger.map(copyToolExecution)
  }

  /** Record one tool attempt with bounded output in both durable ledgers. */
  recordToolExecution(
    toolName: string,
    inputs: unknown = undefined,
    result: unknown = '',
    durationMs = 0,
    permitted = true,
  ): RuntimeToolExecutionRecord {
    const record = freezeToolExecution({
      tool: textValue(toolName, 'toolName'),
      inputs,
      result: String(result).slice(0, MAX_TOOL_EXECUTION_RESULT_CHARS),
      duration_ms: nonNegativeFinite(durationMs, 'durationMs'),
      permitted: booleanValue(permitted, 'permitted'),
      timestamp: this.nowTimestamp(),
    })
    this.toolExecutionLedger.push(record)
    this.history.addToolCall(record.tool, record.result.slice(0, MAX_HISTORY_TOOL_RESULT_CHARS), record.duration_ms)
    return copyToolExecution(record)
  }

  /** Record a timestamped raw streaming event for replay or audit. */
  recordStreamEvent(type: string, data: Readonly<Record<string, unknown>> = {}): RuntimeStreamEvent {
    const event = freezeStreamEvent({
      ...data,
      type: textValue(type, 'type'),
      timestamp: this.nowTimestamp(),
    })
    this.streamEventLedger.push(event)
    return copyStreamEvent(event)
  }

  /** Mirror a completed model turn into the cost and compact history ledgers. */
  recordTurn(model: string, inputTokens: number, outputTokens: number): CostEvent {
    const event = this.costTracker.recordTurn(model, inputTokens, outputTokens)
    this.history.addTurn(model, inputTokens, outputTokens)
    return event
  }

  /** Render the independent session's context, activity, cost, and transcript. */
  asMarkdown(): string {
    const lines = [
      '# Runtime Session',
      '',
      'Session ID: `' + this.sessionId + '`',
      'Prompt: ' + this.prompt,
      'Started: ' + this.context.timestamp,
      '',
      '## Context',
      '- CWD: `' + this.context.cwd + '`',
      '- Runtime: ' + this.context.runtimeVersion,
      '- Platform: ' + this.context.platformName,
      '- Git branch: ' + (this.context.gitBranch || 'N/A'),
      '- Model: ' + this.context.model,
      '- Provider: ' + this.context.provider,
      '',
      '## Tool Executions',
    ]
    if (this.toolExecutionLedger.length) {
      for (const execution of this.toolExecutionLedger) {
        const status = execution.permitted ? 'OK' : 'DENIED'
        lines.push('- `' + execution.tool + '` [' + status + '] (' + execution.duration_ms.toFixed(1) + 'ms)')
      }
    } else {
      lines.push('- none')
    }
    lines.push(
      '',
      '## Stream Events',
      'Total events: ' + this.streamEventLedger.length,
      '',
      this.costTracker.summary(),
      '',
      this.history.asMarkdown(),
      '',
      this.transcript.asMarkdown(),
    )
    return lines.join('\n')
  }

  /** Persist a JSON snapshot and mark the shared transcript as flushed after a successful write. */
  save(options: RuntimeSessionSaveOptions): string {
    const directory = options.directory ?? DEFAULT_RUNTIME_SESSION_DIRECTORY
    if (!directory) throw new TypeError('directory must be non-empty')
    const path = runtimeSessionPath(directory, this.sessionId)
    options.fileSystem.makeDirectory(directory)
    options.fileSystem.writeFile(path, JSON.stringify(this.toRecord(), null, 2))
    this.transcript.flush()
    return path
  }

  toRecord(): RuntimeSessionRecord {
    return Object.freeze({
      session_id: this.sessionId,
      prompt: this.prompt,
      context: this.context.toRecord(),
      messages: this.transcript.toMessages(),
      history: this.history.asDicts(),
      costs: this.costTracker.asRecords(),
      tool_executions: this.toolExecutionLedger.slice().map(copyToolExecution),
      stream_events: this.streamEventLedger.slice(0, MAX_PERSISTED_STREAM_EVENTS).map(copyStreamEvent),
      metadata: this.metadata,
    })
  }

  private nowTimestamp(): string {
    return isoTimestamp(this.clock())
  }
}

/** Construct the expected JSON snapshot path without doing filesystem I/O. */
export function runtimeSessionPath(directory: string, sessionId: string): string {
  if (!directory) throw new TypeError('directory must be non-empty')
  return join(directory, sessionIdValue(sessionId) + '.json')
}

/** Opt-in Bun adapter for standalone callers that want native process context. */
export function createSystemRuntimeContextHost(): RuntimeContextHost {
  return {
    cwd: () => process.cwd(),
    gitBranch: cwd => {
      try {
        const result = Bun.spawnSync(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], {
          cwd,
          stdin: 'ignore',
          stdout: 'pipe',
          stderr: 'ignore',
        })
        return result.exitCode === 0 ? new TextDecoder().decode(result.stdout).trim() : ''
      } catch {
        return ''
      }
    },
    now: () => new Date(),
    platform: () => process.platform,
    runtimeVersion: () => 'Bun ' + Bun.version,
  }
}

/** Opt-in Bun filesystem adapter for standalone JSON snapshots. */
export function createSystemRuntimeSessionFileSystem(): RuntimeSessionFileSystem {
  return {
    makeDirectory: path => mkdirSync(path, { recursive: true }),
    readFile: path => readFileSync(path, 'utf8'),
    writeFile: (path, contents) => writeFileSync(path, contents, 'utf8'),
  }
}

function defaultSessionId(): string {
  return randomUUID().replaceAll('-', '')
}

function capturedGitBranch(host: RuntimeContextHost, cwd: string): string {
  try {
    return textValue(host.gitBranch(cwd), 'host.gitBranch()')
  } catch {
    return ''
  }
}

function parseSessionRecord(text: string): Record<string, unknown> {
  const parsed: unknown = JSON.parse(text)
  return objectValue(parsed, 'runtime session record')
}

function restoreTranscript(value: unknown, clock: () => Date): TranscriptStore {
  const transcript = new TranscriptStore({ now: clock })
  for (const message of arrayValue(value, 'messages')) {
    const record = objectValue(message, 'message')
    const { role, content, ...metadata } = record
    transcript.append(requiredText({ role }, 'role'), requiredText({ content }, 'content'), metadata)
  }
  return transcript
}

function restoreHistory(value: unknown, fallbackTimestamp: string): HistoryLog {
  const records = arrayValue(value, 'history').map(item => objectValue(item, 'history entry'))
  const timestamps = records.map(record => timestampValue(requiredText(record, 'timestamp')))
  let index = 0
  const history = new HistoryLog({
    now: () => new Date(timestamps[index++] ?? fallbackTimestamp),
  })
  for (const record of records) {
    const { kind, title, detail, timestamp: _timestamp, ...metadata } = record
    history.add(
      textValue(kind, 'history.kind'),
      textValue(title, 'history.title'),
      textValue(detail, 'history.detail'),
      metadata,
    )
  }
  return history
}

function restoreCostTracker(value: unknown, clock: () => Date, sessionId: string): CostTracker {
  const tracker = new CostTracker({ now: clock, sessionId })
  for (const item of arrayValue(value, 'costs')) {
    const record = objectValue(item, 'cost entry')
    const cacheReadTokens = optionalNonNegativeInteger(record.cache_read_tokens, 'cost.cache_read_tokens')
    const cacheCreationTokens = optionalNonNegativeInteger(record.cache_creation_tokens, 'cost.cache_creation_tokens')
    const storedSessionId = optionalScope(record.session_id, 'cost.session_id')
    const agentId = optionalScope(record.agent_id, 'cost.agent_id')
    tracker.record(new CostEvent({
      model: requiredText(record, 'model'),
      inputTokens: nonNegativeInteger(record.in_tokens, 'cost.in_tokens'),
      outputTokens: nonNegativeInteger(record.out_tokens, 'cost.out_tokens'),
      costUsd: finiteNumber(record.cost_usd, 'cost.cost_usd'),
      label: requiredText(record, 'label'),
      timestamp: requiredText(record, 'timestamp'),
      ...(cacheReadTokens === undefined ? {} : { cacheReadTokens }),
      ...(cacheCreationTokens === undefined ? {} : { cacheCreationTokens }),
      ...(storedSessionId === undefined ? {} : { sessionId: storedSessionId }),
      ...(agentId === undefined ? {} : { agentId }),
    }))
  }
  return tracker
}

function streamEventsFrom(value: unknown): RuntimeStreamEvent[] {
  return arrayValue(value, 'stream_events').map(item => freezeStreamEvent(objectValue(item, 'stream event')))
}

function toolExecutionsFrom(value: unknown): RuntimeToolExecutionRecord[] {
  return arrayValue(value, 'tool_executions').map(item => {
    const record = objectValue(item, 'tool execution')
    return freezeToolExecution({
      tool: requiredText(record, 'tool'),
      inputs: record.inputs,
      result: requiredText(record, 'result'),
      duration_ms: nonNegativeFinite(record.duration_ms, 'tool execution.duration_ms'),
      permitted: booleanValue(record.permitted, 'tool execution.permitted'),
      timestamp: requiredText(record, 'timestamp'),
    })
  })
}

function copyStreamEvent(event: RuntimeStreamEvent): RuntimeStreamEvent {
  return freezeStreamEvent(event)
}

function freezeStreamEvent(event: Readonly<Record<string, unknown>>): RuntimeStreamEvent {
  const type = requiredText(event, 'type')
  const timestamp = timestampValue(requiredText(event, 'timestamp'))
  return Object.freeze({ ...freezeRecord(event), type, timestamp })
}

function copyToolExecution(record: RuntimeToolExecutionRecord): RuntimeToolExecutionRecord {
  return freezeToolExecution(record)
}

function freezeToolExecution(record: RuntimeToolExecutionRecord): RuntimeToolExecutionRecord {
  return Object.freeze({
    tool: textValue(record.tool, 'tool'),
    inputs: freezeValue(record.inputs),
    result: textValue(record.result, 'result'),
    duration_ms: nonNegativeFinite(record.duration_ms, 'duration_ms'),
    permitted: booleanValue(record.permitted, 'permitted'),
    timestamp: timestampValue(record.timestamp),
  })
}

function objectValue(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(name + ' must be an object')
  return value as Record<string, unknown>
}

function arrayValue(value: unknown, name: string): unknown[] {
  if (!Array.isArray(value)) throw new TypeError(name + ' must be an array')
  return value
}

function requiredText(record: Readonly<Record<string, unknown>>, name: string): string {
  return textValue(record[name], name)
}

function textValue(value: unknown, name: string): string {
  if (typeof value !== 'string') throw new TypeError(name + ' must be a string')
  return value
}

function booleanValue(value: unknown, name: string): boolean {
  if (typeof value !== 'boolean') throw new TypeError(name + ' must be a boolean')
  return value
}

function sessionIdValue(value: unknown): string {
  const sessionId = textValue(value, 'sessionId')
  if (!sessionId || sessionId === '.' || sessionId === '..' || /[\\/]/.test(sessionId)) {
    throw new TypeError('sessionId must be a single non-empty path segment')
  }
  return sessionId
}

function timestampValue(value: unknown): string {
  const timestamp = textValue(value, 'timestamp')
  if (Number.isNaN(new Date(timestamp).valueOf())) throw new RangeError('timestamp must be a valid ISO timestamp')
  return timestamp
}

function isoTimestamp(date: Date): string {
  if (!(date instanceof Date) || Number.isNaN(date.valueOf())) throw new RangeError('now must return a valid Date')
  return date.toISOString()
}

function nonNegativeFinite(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative finite number')
  }
  return value
}

function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new RangeError(name + ' must be a finite number')
  return value
}

function nonNegativeInteger(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative safe integer')
  }
  return value
}

function optionalNonNegativeInteger(value: unknown, name: string): number | undefined {
  return value === undefined ? undefined : nonNegativeInteger(value, name)
}

function optionalScope(value: unknown, name: string): string | undefined {
  if (value === undefined || value === null) return undefined
  const scope = textValue(value, name)
  if (!scope.trim()) throw new TypeError(name + ' must be a non-empty string')
  return scope
}

function freezeRecord(record: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> {
  const copy: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(record)) copy[key] = freezeValue(value)
  return Object.freeze(copy)
}

function freezeValue(value: unknown): unknown {
  if (Array.isArray(value)) return Object.freeze(value.map(freezeValue))
  if (value instanceof Date) return value.toISOString()
  if (value && typeof value === 'object') return freezeRecord(value as Record<string, unknown>)
  return value
}
