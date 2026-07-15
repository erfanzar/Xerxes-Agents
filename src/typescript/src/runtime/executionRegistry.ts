// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const EntryKind = Object.freeze({
  COMMAND: 'command',
  TOOL: 'tool',
} as const)

export type EntryKind = (typeof EntryKind)[keyof typeof EntryKind]

export const ExecutionStatus = Object.freeze({
  PENDING: 'pending',
  RUNNING: 'running',
  SUCCESS: 'success',
  FAILURE: 'failure',
  FAILED: 'failure',
  PARTIAL: 'partial',
  CANCELLED: 'cancelled',
} as const)

export type ExecutionStatus = (typeof ExecutionStatus)[keyof typeof ExecutionStatus]

export type EntryHandler = (
  inputs: Readonly<Record<string, unknown>>,
  signal?: AbortSignal,
) => unknown | Promise<unknown>

export interface RegistryEntry {
  readonly category: string
  readonly description: string
  readonly handler: EntryHandler | undefined
  readonly kind: EntryKind
  readonly name: string
  readonly safe: boolean
  readonly schema: Readonly<Record<string, unknown>> | undefined
  readonly sourceHint: string
}

export interface RegistryEntryOptions {
  readonly category?: string
  readonly description?: string
  readonly safe?: boolean
  readonly schema?: Readonly<Record<string, unknown>>
  readonly sourceHint?: string
}

export interface ExecutionResult {
  readonly durationMs: number
  readonly error: string
  readonly handled: boolean
  readonly kind: EntryKind
  readonly name: string
  readonly result: string
}

export interface RouteMatch {
  readonly description: string
  readonly kind: EntryKind
  readonly name: string
  readonly score: number
  readonly sourceHint: string
}

export interface ExecutionRecord {
  readonly createdAt: number
  readonly durationMs: number
  readonly entryKind: EntryKind
  readonly entryName: string
  readonly error: string
  readonly finishedAt: number
  readonly id: string
  readonly inputs: Readonly<Record<string, unknown>>
  readonly metadata: Readonly<Record<string, unknown>>
  readonly result: string
  readonly startedAt: number
  readonly status: ExecutionStatus
}

/** Optional host runner for detached records; omit it to invoke registered handlers directly. */
export type ExecutionRunner = (
  record: ExecutionRecord,
  signal: AbortSignal,
) => unknown | Promise<unknown>

export interface ExecutionRegistryOptions {
  readonly idFactory?: () => string
  readonly maxCompleted?: number
  readonly maxConcurrent?: number
  /** Epoch-seconds wall clock for record timestamps. */
  readonly now?: () => number
  /** Monotonic milliseconds clock for direct and detached execution durations. */
  readonly monotonicNow?: () => number
  readonly runner?: ExecutionRunner
}

export interface SubmitExecutionOptions {
  readonly executionId?: string
  readonly metadata?: Readonly<Record<string, unknown>>
}

export interface WaitForExecutionOptions {
  readonly settled?: boolean
  readonly timeoutMs?: number
}

export interface ShutdownExecutionRegistryOptions {
  readonly cancelRunning?: boolean
  readonly timeoutMs?: number
}

interface Deferred {
  readonly promise: Promise<void>
  resolve(): void
}

interface ActiveExecution {
  readonly controller: AbortController
  readonly promise: Promise<void>
  readonly token: number
}

interface MutableExecutionRecord {
  readonly completion: Deferred
  readonly createdAt: number
  durationMs: number
  readonly entryKind: EntryKind
  readonly entryName: string
  error: string
  finishedAt: number
  readonly id: string
  readonly inputs: Readonly<Record<string, unknown>>
  readonly metadata: Readonly<Record<string, unknown>>
  result: string
  readonly sequence: number
  readonly settled: Deferred
  startedAt: number
  startedMonotonic: number
  status: ExecutionStatus
}

const DEFAULT_MAX_COMPLETED = 100
const DEFAULT_MAX_CONCURRENT = 4
const DEFAULT_SHUTDOWN_TIMEOUT_MS = 5_000
const MAX_ID_ATTEMPTS = 100

/**
 * Unified command/tool catalog and detached execution ledger.
 *
 * The catalog preserves runtime/execution_registry.py lookup, routing, and
 * schema behavior. Detached calls add bounded FIFO execution, cooperative
 * cancellation, immutable record snapshots, and settled-only retention.
 */
export class ExecutionRegistry {
  private readonly active = new Map<string, ActiveExecution>()
  private readonly commands = new Map<string, RegistryEntry>()
  private readonly idFactory: () => string
  private readonly maxCompleted: number
  private readonly maxConcurrent: number
  private readonly monotonicNow: () => number
  private readonly now: () => number
  private readonly queue: string[] = []
  private readonly records = new Map<string, MutableExecutionRecord>()
  private readonly runner: ExecutionRunner | undefined
  private sequence = 0
  private shuttingDown = false
  private token = 0
  private readonly tools = new Map<string, RegistryEntry>()

  constructor(options: ExecutionRegistryOptions = {}) {
    this.idFactory = options.idFactory ?? defaultExecutionId
    this.maxConcurrent = boundedInteger(options.maxConcurrent ?? DEFAULT_MAX_CONCURRENT, 'maxConcurrent', 1)
    this.maxCompleted = boundedInteger(options.maxCompleted ?? DEFAULT_MAX_COMPLETED, 'maxCompleted', 0)
    this.now = options.now ?? (() => Date.now() / 1_000)
    this.monotonicNow = options.monotonicNow ?? (() => performance.now())
    this.runner = options.runner
  }

  get commandCount(): number {
    return this.commands.size
  }

  get toolCount(): number {
    return this.tools.size
  }

  get executionCount(): number {
    return this.records.size
  }

  get runningCount(): number {
    return this.active.size
  }

  /** Register or replace a case-insensitive slash command. */
  registerCommand(name: string, handler: EntryHandler | undefined = undefined, options: RegistryEntryOptions = {}): void {
    const entry = makeEntry(name, EntryKind.COMMAND, handler, options)
    this.commands.set(entry.name.toLowerCase(), entry)
  }

  /** Register or replace a case-sensitive LLM-callable tool. */
  registerTool(name: string, handler: EntryHandler | undefined = undefined, options: RegistryEntryOptions = {}): void {
    const entry = makeEntry(name, EntryKind.TOOL, handler, options)
    this.tools.set(entry.name, entry)
  }

  /** Ingest duck-compatible agent functions or OpenAI-style function definitions. */
  registerFromAgentFunctions(functions: readonly unknown[]): void {
    for (const value of functions) {
      const record = objectValue(value)
      const functionRecord = objectValue(record?.function)
      const name = stringField(record?.name) ?? stringField(functionRecord?.name)
      if (!name) continue
      const description = stringField(record?.description) ?? stringField(functionRecord?.description) ?? ''
      const callable = typeof record?.callable_func === 'function'
        ? record.callable_func
        : typeof record?.handler === 'function'
          ? record.handler
          : undefined
      const handler = callable === undefined
        ? undefined
        : (inputs: Readonly<Record<string, unknown>>) => callable(inputs)
      this.registerTool(name, handler, { description })
    }
  }

  getCommand(name: string): RegistryEntry | undefined {
    const entry = this.commands.get(name.toLowerCase())
    return entry === undefined ? undefined : copyEntry(entry)
  }

  getTool(name: string): RegistryEntry | undefined {
    const entry = this.tools.get(name)
    return entry === undefined ? undefined : copyEntry(entry)
  }

  listCommands(category?: string): RegistryEntry[] {
    return [...this.commands.values()]
      .filter(entry => category === undefined || entry.category === category)
      .sort((left, right) => left.name.localeCompare(right.name))
      .map(copyEntry)
  }

  listTools(options: { readonly category?: string; readonly safeOnly?: boolean } = {}): RegistryEntry[] {
    return [...this.tools.values()]
      .filter(entry => options.category === undefined || entry.category === options.category)
      .filter(entry => !options.safeOnly || entry.safe)
      .sort((left, right) => left.name.localeCompare(right.name))
      .map(copyEntry)
  }

  /** Suggest relevant registered entries using the source registry's token-overlap score. */
  route(prompt: string, limit = 5): RouteMatch[] {
    const bounded = nonNegativeInteger(limit, 'limit')
    if (bounded === 0) return []
    const tokens = new Set(prompt
      .replaceAll('/', ' ')
      .replaceAll('-', ' ')
      .replaceAll('_', ' ')
      .split(/\s+/)
      .map(token => token.toLowerCase())
      .filter(token => token.length > 1))
    if (!tokens.size) return []
    return [...this.commands.values(), ...this.tools.values()]
      .map(entry => ({
        name: entry.name,
        kind: entry.kind,
        score: scoreEntry(entry, tokens),
        sourceHint: entry.sourceHint,
        description: entry.description,
      }))
      .filter(match => match.score > 0)
      .sort((left, right) => right.score - left.score || left.name.localeCompare(right.name))
      .slice(0, bounded)
      .map(match => Object.freeze(match))
  }

  /** Execute a slash command immediately without creating a retained detached record. */
  async executeCommand(name: string, inputs: Readonly<Record<string, unknown>> = {}): Promise<ExecutionResult> {
    return this.executeEntry(this.commands.get(name.toLowerCase()), name, EntryKind.COMMAND, inputs)
  }

  /** Execute a tool immediately without creating a retained detached record. */
  async executeTool(name: string, inputs: Readonly<Record<string, unknown>> = {}): Promise<ExecutionResult> {
    return this.executeEntry(this.tools.get(name), name, EntryKind.TOOL, inputs)
  }

  /** Submit a command for detached execution and return its initial immutable record snapshot. */
  submitCommand(
    name: string,
    inputs: Readonly<Record<string, unknown>> = {},
    options: SubmitExecutionOptions = {},
  ): ExecutionRecord {
    return this.submit(EntryKind.COMMAND, name, inputs, options)
  }

  /** Submit a tool for detached execution and return its initial immutable record snapshot. */
  submitTool(
    name: string,
    inputs: Readonly<Record<string, unknown>> = {},
    options: SubmitExecutionOptions = {},
  ): ExecutionRecord {
    return this.submit(EntryKind.TOOL, name, inputs, options)
  }

  /** Submit any catalog target, or an injected-runner target, for detached execution. */
  submit(
    kind: EntryKind,
    name: string,
    inputs: Readonly<Record<string, unknown>> = {},
    options: SubmitExecutionOptions = {},
  ): ExecutionRecord {
    if (this.shuttingDown) throw new Error('execution registry is shutting down')
    const requestedName = requiredText(name, 'name')
    const entry = this.lookup(kind, requestedName)
    const id = options.executionId === undefined ? this.nextId() : requiredText(options.executionId, 'executionId')
    if (this.records.has(id)) throw new Error('execution record already exists: ' + id)
    const createdAt = this.timestamp()
    const record: MutableExecutionRecord = {
      id,
      entryKind: kind,
      entryName: entry?.name ?? requestedName,
      inputs: freezeRecord(inputs),
      metadata: freezeRecord(options.metadata ?? {}),
      status: ExecutionStatus.PENDING,
      result: '',
      error: '',
      createdAt,
      startedAt: 0,
      finishedAt: 0,
      durationMs: 0,
      startedMonotonic: -1,
      completion: deferred(),
      settled: deferred(),
      sequence: this.sequence,
    }
    this.sequence += 1
    this.records.set(id, record)
    this.queue.push(id)
    this.drain()
    return executionSnapshot(record)
  }

  getExecution(executionId: string): ExecutionRecord | undefined {
    const record = this.records.get(executionId)
    return record === undefined ? undefined : executionSnapshot(record)
  }

  listExecutions(): ExecutionRecord[] {
    return [...this.records.values()].map(executionSnapshot)
  }

  /**
   * Mark a record cancelled and signal a running runner. Its slot remains
   * occupied until physical cleanup settles, preventing concurrency leakage.
   */
  cancelExecution(executionId: string): boolean {
    const record = this.records.get(executionId)
    if (!record || !isCancellable(record.status)) return false
    this.markCancelled(record)
    const active = this.active.get(executionId)
    if (active) active.controller.abort(new Error('Execution cancelled'))
    if (!active) {
      record.settled.resolve()
      this.pruneCompleted()
      this.drain()
    }
    return true
  }

  async waitExecution(
    executionId: string,
    options: WaitForExecutionOptions = {},
  ): Promise<ExecutionRecord | undefined> {
    const record = this.records.get(executionId)
    if (!record) return undefined
    await waitWithTimeout(options.settled ? record.settled.promise : record.completion.promise, options.timeoutMs)
    return executionSnapshot(record)
  }

  /**
   * Stop accepting records, cancel queued work, optionally abort running work,
   * and wait up to the configured bound for active cleanup.
   */
  async shutdown(options: ShutdownExecutionRegistryOptions = {}): Promise<void> {
    if (this.shuttingDown) return
    this.shuttingDown = true
    for (const record of [...this.records.values()]) {
      if (record.status === ExecutionStatus.PENDING) this.cancelExecution(record.id)
    }
    if (options.cancelRunning) {
      for (const record of [...this.records.values()]) {
        if (record.status === ExecutionStatus.RUNNING) this.cancelExecution(record.id)
      }
    }
    const active = [...this.active.values()].map(value => value.promise)
    await waitWithTimeout(
      Promise.allSettled(active).then(() => undefined),
      nonNegativeInteger(options.timeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS, 'timeoutMs'),
    )
    this.pruneCompleted()
  }

  /** Return source-compatible tool schemas, auto-generating the minimal input schema when absent. */
  toolSchemas(): ReadonlyArray<Readonly<Record<string, unknown>>> {
    return [...this.tools.values()].map(entry => {
      if (entry.schema !== undefined) return freezeRecord(entry.schema)
      return Object.freeze({
        name: entry.name,
        description: entry.description || 'Execute ' + entry.name,
        input_schema: Object.freeze({
          type: 'object',
          properties: Object.freeze({}),
        }),
      })
    })
  }

  /** Render the source registry's concise Markdown overview. */
  summary(): string {
    const lines = [
      '# Execution Registry',
      '',
      'Commands: ' + this.commandCount,
      'Tools: ' + this.toolCount,
      '',
    ]
    if (this.commands.size) {
      lines.push('## Commands')
      const code = String.fromCharCode(96)
      for (const entry of this.listCommands()) lines.push('- ' + code + '/' + entry.name + code + ' — ' + entry.description)
      lines.push('')
    }
    if (this.tools.size) {
      lines.push('## Tools')
      const code = String.fromCharCode(96)
      for (const entry of this.listTools()) {
        lines.push('- ' + code + entry.name + code + (entry.safe ? ' [safe]' : '') + ' — ' + entry.description)
      }
    }
    return lines.join('\n')
  }

  private async executeEntry(
    entry: RegistryEntry | undefined,
    requestedName: string,
    kind: EntryKind,
    inputs: Readonly<Record<string, unknown>>,
    signal?: AbortSignal,
  ): Promise<ExecutionResult> {
    if (entry === undefined) {
      return Object.freeze({
        name: requestedName,
        kind,
        handled: false,
        result: 'Unknown ' + kind + ': ' + requestedName,
        durationMs: 0,
        error: '',
      })
    }
    if (entry.handler === undefined) {
      return Object.freeze({
        name: entry.name,
        kind: entry.kind,
        handled: false,
        result: 'No handler registered for ' + entry.kind + ': ' + entry.name,
        durationMs: 0,
        error: '',
      })
    }
    const start = this.monotonicTimestamp()
    try {
      if (signal?.aborted) throw signal.reason ?? new Error('Execution cancelled')
      const result = await entry.handler(freezeRecord(inputs), signal)
      return Object.freeze({
        name: entry.name,
        kind: entry.kind,
        handled: true,
        result: stringifyResult(result),
        durationMs: elapsedMilliseconds(start, this.monotonicTimestamp()),
        error: '',
      })
    } catch (error) {
      return Object.freeze({
        name: entry.name,
        kind: entry.kind,
        handled: false,
        result: '',
        durationMs: elapsedMilliseconds(start, this.monotonicTimestamp()),
        error: errorMessage(error),
      })
    }
  }

  private lookup(kind: EntryKind, name: string): RegistryEntry | undefined {
    return kind === EntryKind.COMMAND ? this.commands.get(name.toLowerCase()) : this.tools.get(name)
  }

  private drain(): void {
    if (this.shuttingDown) return
    while (this.active.size < this.maxConcurrent) {
      const record = this.nextPending()
      if (!record) return
      this.start(record)
    }
  }

  private nextPending(): MutableExecutionRecord | undefined {
    while (this.queue.length) {
      const id = this.queue.shift()
      if (!id) continue
      const record = this.records.get(id)
      if (record?.status === ExecutionStatus.PENDING) return record
    }
    return undefined
  }

  private start(record: MutableExecutionRecord): void {
    record.status = ExecutionStatus.RUNNING
    record.startedAt = this.timestamp()
    record.startedMonotonic = this.monotonicTimestamp()
    const controller = new AbortController()
    const token = this.token + 1
    this.token = token
    const snapshot = executionSnapshot(record)
    const promise = Promise.resolve()
      .then(() => this.runner ? this.runner(snapshot, controller.signal) : this.runRegistered(snapshot, controller.signal))
      .then(
        result => this.succeed(record.id, token, result),
        error => this.fail(record.id, token, error),
      )
      .catch(error => this.fail(record.id, token, error))
      .finally(() => this.settle(record.id, token))
    this.active.set(record.id, { controller, promise, token })
  }

  private async runRegistered(record: ExecutionRecord, signal: AbortSignal): Promise<string> {
    const result = await this.executeEntry(
      this.lookup(record.entryKind, record.entryName),
      record.entryName,
      record.entryKind,
      record.inputs,
      signal,
    )
    if (!result.handled) throw new Error(result.error || result.result)
    return result.result
  }

  private succeed(id: string, token: number, value: unknown): void {
    const record = this.currentRecord(id, token)
    if (!record || record.status === ExecutionStatus.CANCELLED) return
    record.result = stringifyResult(value)
    record.error = ''
    record.status = ExecutionStatus.SUCCESS
    this.finish(record)
    record.completion.resolve()
  }

  private fail(id: string, token: number, error: unknown): void {
    const record = this.currentRecord(id, token)
    if (!record || record.status === ExecutionStatus.CANCELLED) return
    record.result = ''
    record.error = lifecycleError(error)
    record.status = ExecutionStatus.FAILURE
    this.finish(record)
    record.completion.resolve()
  }

  private settle(id: string, token: number): void {
    const active = this.active.get(id)
    if (!active || active.token !== token) return
    this.active.delete(id)
    const record = this.records.get(id)
    if (record) {
      if (record.status === ExecutionStatus.RUNNING) {
        record.status = ExecutionStatus.FAILURE
        record.error = 'Error: execution runner settled without a result'
        this.finish(record)
        record.completion.resolve()
      }
      if (isTerminal(record.status) && record.finishedAt === 0) this.finish(record)
      record.settled.resolve()
    }
    this.pruneCompleted()
    this.drain()
  }

  private currentRecord(id: string, token: number): MutableExecutionRecord | undefined {
    const active = this.active.get(id)
    if (!active || active.token !== token) return undefined
    return this.records.get(id)
  }

  private markCancelled(record: MutableExecutionRecord): void {
    record.status = ExecutionStatus.CANCELLED
    record.result = ''
    record.error = ''
    this.finish(record)
    record.completion.resolve()
  }

  private finish(record: MutableExecutionRecord): void {
    record.finishedAt = this.timestamp()
    record.durationMs = record.startedMonotonic < 0
      ? 0
      : elapsedMilliseconds(record.startedMonotonic, this.monotonicTimestamp())
  }

  private pruneCompleted(): void {
    const completed = [...this.records.values()]
      .filter(record => isTerminal(record.status) && !this.active.has(record.id))
      .sort((left, right) => left.finishedAt - right.finishedAt || left.sequence - right.sequence)
    const count = Math.max(0, completed.length - this.maxCompleted)
    for (const record of completed.slice(0, count)) this.records.delete(record.id)
  }

  private nextId(): string {
    for (let attempt = 0; attempt < MAX_ID_ATTEMPTS; attempt += 1) {
      const id = requiredText(this.idFactory(), 'execution id')
      if (!this.records.has(id)) return id
    }
    throw new Error('execution id factory produced too many duplicate identifiers')
  }

  private timestamp(): number {
    const value = this.now()
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError('execution registry clock must return a non-negative finite epoch timestamp')
    }
    return value
  }

  private monotonicTimestamp(): number {
    const value = this.monotonicNow()
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError('execution registry monotonic clock must return a non-negative finite value')
    }
    return value
  }
}

function makeEntry(
  name: string,
  kind: EntryKind,
  handler: EntryHandler | undefined,
  options: RegistryEntryOptions,
): RegistryEntry {
  return Object.freeze({
    name: requiredText(name, 'name'),
    kind,
    description: options.description ?? '',
    handler,
    category: options.category ?? '',
    safe: options.safe ?? false,
    sourceHint: options.sourceHint ?? '',
    schema: options.schema === undefined ? undefined : freezeRecord(options.schema),
  })
}

function copyEntry(entry: RegistryEntry): RegistryEntry {
  return Object.freeze({
    ...entry,
    schema: entry.schema === undefined ? undefined : freezeRecord(entry.schema),
  })
}

function executionSnapshot(record: MutableExecutionRecord): ExecutionRecord {
  return Object.freeze({
    id: record.id,
    entryName: record.entryName,
    entryKind: record.entryKind,
    inputs: freezeRecord(record.inputs),
    metadata: freezeRecord(record.metadata),
    status: record.status,
    result: record.result,
    error: record.error,
    createdAt: record.createdAt,
    startedAt: record.startedAt,
    finishedAt: record.finishedAt,
    durationMs: record.durationMs,
  })
}

function scoreEntry(entry: RegistryEntry, tokens: ReadonlySet<string>): number {
  let score = 0
  const name = entry.name.toLowerCase()
  const description = entry.description.toLowerCase()
  const category = entry.category.toLowerCase()
  for (const token of tokens) {
    if (name.includes(token)) score += 3
    if (description.includes(token)) score += 1
    if (category.includes(token)) score += 2
  }
  if (tokens.has(name)) score += 5
  return score
}

function deferred(): Deferred {
  let resolvePromise: (() => void) | undefined
  const promise = new Promise<void>(resolve => {
    resolvePromise = resolve
  })
  return {
    promise,
    resolve: () => resolvePromise?.(),
  }
}

function isCancellable(status: ExecutionStatus): boolean {
  return status === ExecutionStatus.PENDING || status === ExecutionStatus.RUNNING
}

function isTerminal(status: ExecutionStatus): boolean {
  return status === ExecutionStatus.SUCCESS
    || status === ExecutionStatus.FAILURE
    || status === ExecutionStatus.CANCELLED
}

function defaultExecutionId(): string {
  return crypto.randomUUID().replaceAll('-', '').slice(0, 12)
}

function boundedInteger(value: number, name: string, minimum: number): number {
  if (!Number.isFinite(value)) throw new RangeError(name + ' must be finite')
  return Math.max(minimum, Math.trunc(value))
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) throw new RangeError(name + ' must be a non-negative integer')
  return value
}

function requiredText(value: unknown, name: string): string {
  if (typeof value !== 'string' || !value) throw new TypeError(name + ' must be a non-empty string')
  return value
}

function stringifyResult(value: unknown): string {
  if (value === undefined || value === null) return ''
  return typeof value === 'string' ? value : String(value)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function lifecycleError(error: unknown): string {
  return error instanceof Error ? error.name + ': ' + error.message : 'Error: ' + String(error)
}

function elapsedMilliseconds(start: number, end: number): number {
  return Math.max(0, end - start)
}

async function waitWithTimeout(promise: Promise<void>, timeoutMs: number | undefined): Promise<void> {
  if (timeoutMs === undefined) {
    await promise
    return
  }
  const timeout = nonNegativeInteger(timeoutMs, 'timeoutMs')
  if (timeout === 0) return
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    await Promise.race([
      promise,
      new Promise<void>(resolve => {
        timer = setTimeout(resolve, timeout)
      }),
    ])
  } finally {
    if (timer !== undefined) clearTimeout(timer)
  }
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

function objectValue(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

function stringField(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined
}
