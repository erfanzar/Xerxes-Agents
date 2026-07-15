// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFileSync, existsSync, mkdirSync, renameSync, rmSync, statSync } from 'node:fs'
import { dirname } from 'node:path'

import {
  LoggingLevel,
  defaultLogOutput,
  isLoggingEnabled,
  loggingTimestamp,
  mergeLogContext,
  normalizeLoggingLevel,
  type LogContext,
  type LogOutput,
  type LoggingLevel as LoggingLevelName,
} from './levels.js'

const DEFAULT_LOG_FILE_BACKUPS = 5
const DEFAULT_LOG_FILE_BYTES = 10 * 1024 * 1024
const DEFAULT_HISTOGRAM_BUCKETS = Object.freeze([
  0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10,
])

export type StructuredTraceAttribute = boolean | number | string
export type StructuredTraceAttributes = Readonly<Record<string, StructuredTraceAttribute>>

/** Structural OpenTelemetry boundary; a host supplies the real tracer when it wants tracing. */
export interface StructuredTraceSpan {
  end(): void
  recordException?(error: unknown): void
  setAttribute?(name: string, value: StructuredTraceAttribute): void
  setStatus?(status: Readonly<{ code: 'error' | 'ok'; message?: string }>): void
}

/** The logger owns no telemetry SDK or endpoint connection. */
export interface StructuredTracer {
  startSpan(name: string, options?: Readonly<{ attributes?: StructuredTraceAttributes }>): StructuredTraceSpan
}

export interface StructuredLogRecord {
  readonly event: string
  readonly level: Lowercase<LoggingLevelName>
  readonly logger: string
  readonly timestamp: string
  readonly [field: string]: unknown
}

export interface StructuredLoggerOptions {
  readonly clock?: () => Date
  readonly context?: LogContext
  readonly enableJson?: boolean
  /** Optional local rotating file sink, matching the Python logger's 10 MiB/5-backup defaults. */
  readonly logFile?: string
  readonly maxLogFileBackups?: number
  readonly maxLogFileBytes?: number
  /** Injectable metrics registry; bound child loggers retain the same registry. */
  readonly metrics?: StructuredMetricsRegistry
  readonly level?: string
  readonly name?: string
  readonly stream?: LogOutput
  /** Enables spans only when an explicit host tracer is also supplied. */
  readonly enableTracing?: boolean
  readonly traceEndpoint?: string
  readonly tracer?: StructuredTracer
}

export interface FunctionCallLogInput {
  readonly agentId: string
  readonly arguments: Readonly<Record<string, unknown>>
  readonly duration?: number
  readonly error?: unknown
  readonly functionName: string
  readonly result?: unknown
}

export interface AgentSwitchLogInput {
  readonly fromAgent: string
  readonly reason?: string
  readonly toAgent: string
}

export interface LlmRequestLogInput {
  readonly completionTokens: number
  readonly duration: number
  readonly error?: unknown
  readonly model: string
  readonly promptTokens: number
  readonly provider: string
}

export interface MemoryOperationLogInput {
  readonly agentId: string
  readonly entryCount?: number
  readonly error?: unknown
  readonly memoryType: string
  readonly operation: string
}

/**
 * Dependency-free structured logger with structlog-compatible JSON event keys.
 *
 * Prometheus-style metrics are accumulated locally and OpenTelemetry remains
 * an explicit host injection, so logging never imports a telemetry SDK or
 * opens a network connection on its own.
 */
export class XerxesLogger {
  readonly enableJson: boolean
  readonly enableTracing: boolean
  readonly logFile: string | undefined
  readonly metrics: StructuredMetricsRegistry
  readonly name: string
  readonly traceEndpoint: string | undefined
  private readonly clock: () => Date
  private readonly context: LogContext
  private currentLevel: LoggingLevelName
  private readonly fileSink: RotatingFileSink | undefined
  private readonly stream: LogOutput
  private readonly tracer: StructuredTracer | undefined

  constructor(options: StructuredLoggerOptions = {}) {
    this.name = options.name?.trim() || 'xerxes'
    this.currentLevel = normalizeLoggingLevel(options.level)
    this.enableJson = options.enableJson ?? true
    this.enableTracing = options.enableTracing ?? false
    this.logFile = options.logFile
    this.traceEndpoint = options.traceEndpoint
    this.tracer = options.tracer
    this.clock = options.clock ?? (() => new Date())
    this.stream = options.stream ?? defaultLogOutput()
    this.context = mergeLogContext(options.context)
    this.metrics = options.metrics ?? new StructuredMetricsRegistry()
    this.fileSink = options.logFile === undefined
      ? undefined
      : new RotatingFileSink(options.logFile, options.maxLogFileBytes, options.maxLogFileBackups)
  }

  get level(): LoggingLevelName {
    return this.currentLevel
  }

  /** Return a sibling logger with immutable merged context bindings. */
  bind(context: LogContext): XerxesLogger {
    return new XerxesLogger({
      name: this.name,
      level: this.currentLevel,
      enableJson: this.enableJson,
      enableTracing: this.enableTracing,
      ...(this.logFile === undefined ? {} : { logFile: this.logFile }),
      ...(this.traceEndpoint === undefined ? {} : { traceEndpoint: this.traceEndpoint }),
      ...(this.tracer === undefined ? {} : { tracer: this.tracer }),
      clock: this.clock,
      stream: this.stream,
      context: mergeLogContext(this.context, context),
      metrics: this.metrics,
    })
  }

  setLevel(level: string): void {
    this.currentLevel = normalizeLoggingLevel(level)
  }

  debug(event: string, fields: LogContext = {}): StructuredLogRecord | undefined {
    return this.log(LoggingLevel.DEBUG, event, fields)
  }

  info(event: string, fields: LogContext = {}): StructuredLogRecord | undefined {
    return this.log(LoggingLevel.INFO, event, fields)
  }

  warning(event: string, fields: LogContext = {}): StructuredLogRecord | undefined {
    return this.log(LoggingLevel.WARNING, event, fields)
  }

  error(event: string, fields: LogContext = {}): StructuredLogRecord | undefined {
    return this.log(LoggingLevel.ERROR, event, fields)
  }

  critical(event: string, fields: LogContext = {}): StructuredLogRecord | undefined {
    return this.log(LoggingLevel.CRITICAL, event, fields)
  }

  log(level: string, event: string, fields: LogContext = {}): StructuredLogRecord | undefined {
    const normalized = normalizeLoggingLevel(level)
    if (!isLoggingEnabled(this.currentLevel, normalized)) return undefined
    const record = Object.freeze({
      ...this.context,
      ...fields,
      timestamp: loggingTimestamp(this.clock),
      level: normalized.toLowerCase() as Lowercase<LoggingLevelName>,
      logger: this.name,
      event: String(event),
    }) as StructuredLogRecord
    this.write(this.enableJson ? jsonLine(record) : humanLine(record))
    return record
  }

  logFunctionCall(input: FunctionCallLogInput): StructuredLogRecord | undefined {
    const failed = input.error !== undefined && input.error !== null
    const status = failed ? 'error' : 'success'
    this.metrics.incrementCounter('xerxes_function_calls_total', {
      agent_id: input.agentId,
      function_name: input.functionName,
      status,
    })
    this.metrics.observeHistogram('xerxes_function_duration_seconds', {
      agent_id: input.agentId,
      function_name: input.functionName,
    }, input.duration ?? 0)
    const fields: Record<string, unknown> = {
      event_name: 'function_call',
      agent_id: input.agentId,
      function_name: input.functionName,
      arguments: { ...input.arguments },
      duration: input.duration ?? 0,
      status,
    }
    if (input.result !== undefined) fields.result = String(input.result).slice(0, 200)
    if (failed) {
      fields.error = errorMessage(input.error)
      this.recordError(input.error, 'function_executor')
    }
    return this.log(failed ? LoggingLevel.ERROR : LoggingLevel.INFO, failed ? 'Function call failed' : 'Function call completed', fields)
  }

  logAgentSwitch(input: AgentSwitchLogInput): StructuredLogRecord | undefined {
    this.metrics.incrementCounter('xerxes_switches_total', {
      from_agent: input.fromAgent,
      to_agent: input.toAgent,
    })
    return this.info('Agent switch', {
      event_name: 'agent_switch',
      from_agent: input.fromAgent,
      to_agent: input.toAgent,
      reason: input.reason ?? null,
    })
  }

  logLlmRequest(input: LlmRequestLogInput): StructuredLogRecord | undefined {
    const failed = input.error !== undefined && input.error !== null
    const status = failed ? 'error' : 'success'
    const labels = { provider: input.provider, model: input.model }
    this.metrics.incrementCounter('xerxes_llm_requests_total', { ...labels, status })
    this.metrics.incrementCounter('xerxes_llm_tokens_total', { ...labels, type: 'prompt' }, input.promptTokens)
    this.metrics.incrementCounter('xerxes_llm_tokens_total', { ...labels, type: 'completion' }, input.completionTokens)
    const fields: Record<string, unknown> = {
      event_name: 'llm_request',
      provider: input.provider,
      model: input.model,
      prompt_tokens: input.promptTokens,
      completion_tokens: input.completionTokens,
      total_tokens: input.promptTokens + input.completionTokens,
      duration: input.duration,
      status,
    }
    if (failed) {
      fields.error = errorMessage(input.error)
      this.recordError(input.error, 'llm_client')
    }
    return this.log(failed ? LoggingLevel.ERROR : LoggingLevel.INFO, failed ? 'LLM request failed' : 'LLM request completed', fields)
  }

  logMemoryOperation(input: MemoryOperationLogInput): StructuredLogRecord | undefined {
    const failed = input.error !== undefined && input.error !== null
    const labels = { memory_type: input.memoryType, agent_id: input.agentId }
    if (input.operation === 'add') this.metrics.adjustGauge('xerxes_memory_entries', labels, input.entryCount ?? 1)
    if (input.operation === 'remove') this.metrics.adjustGauge('xerxes_memory_entries', labels, -(input.entryCount ?? 1))
    const fields: Record<string, unknown> = {
      event_name: 'memory_operation',
      operation: input.operation,
      memory_type: input.memoryType,
      agent_id: input.agentId,
      entry_count: input.entryCount ?? 1,
    }
    if (failed) {
      fields.error = errorMessage(input.error)
      this.recordError(input.error, 'memory_store')
    }
    return this.log(failed ? LoggingLevel.ERROR : LoggingLevel.DEBUG, failed ? 'Memory operation failed' : 'Memory operation completed', fields)
  }

  /** Start one host-owned trace span when tracing and an injected tracer are both enabled. */
  startSpan(name: string, attributes: StructuredTraceAttributes = {}): StructuredTraceSpan | undefined {
    if (!this.enableTracing || this.tracer === undefined) return undefined
    try {
      return this.tracer.startSpan(name, {
        attributes: {
          'service.name': this.name,
          ...attributes,
        },
      })
    } catch {
      return undefined
    }
  }

  /** Execute an operation inside a host span, recording failures without swallowing them. */
  withSpan<T>(
    name: string,
    attributes: StructuredTraceAttributes,
    operation: (span: StructuredTraceSpan | undefined) => T | Promise<T>,
  ): T | Promise<T> {
    const span = this.startSpan(name, attributes)
    try {
      const value = operation(span)
      if (isPromise(value)) {
        return value.then(
          result => {
            endSpan(span)
            return result
          },
          error => {
            recordSpanError(span, error)
            endSpan(span)
            throw error
          },
        )
      }
      endSpan(span)
      return value
    } catch (error) {
      recordSpanError(span, error)
      endSpan(span)
      throw error
    }
  }

  /** Return native Prometheus exposition bytes without requiring a Prometheus SDK. */
  getMetrics(): Uint8Array {
    return this.metrics.render()
  }

  private recordError(error: unknown, component: string): void {
    this.metrics.incrementCounter('xerxes_errors_total', {
      error_type: errorType(error),
      component,
    })
  }

  private write(chunk: string): void {
    this.stream.write(chunk)
    this.fileSink?.write(chunk)
  }
}

let defaultLogger: XerxesLogger | undefined

/** Return the lazily-created structured logger. */
export function getLogger(): XerxesLogger {
  defaultLogger ??= new XerxesLogger()
  return defaultLogger
}

/** Replace the structured singleton explicitly without reconfiguring global runtime handlers. */
export function configureLogging(options: StructuredLoggerOptions = {}): XerxesLogger {
  defaultLogger = new XerxesLogger(options)
  return defaultLogger
}

export const getStructuredLogger = getLogger

interface MetricSeries {
  readonly labels: Readonly<Record<string, string>>
  value: number
}

interface HistogramSeries extends MetricSeries {
  count: number
  readonly buckets: number[]
  sum: number
}

/**
 * Dependency-free Prometheus metric accumulator used by structured log events.
 *
 * It mirrors the Python logger's counter, gauge, and histogram updates while
 * keeping export in-process. An HTTP server can return {@link render} directly.
 */
export class StructuredMetricsRegistry {
  private readonly counters = new Map<string, MetricSeries>()
  private readonly gauges = new Map<string, MetricSeries>()
  private readonly histograms = new Map<string, HistogramSeries>()

  incrementCounter(name: string, labels: Readonly<Record<string, string>>, amount = 1): void {
    const series = this.counter(name, labels)
    series.value += finiteMetricValue(amount, 'counter amount')
  }

  adjustGauge(name: string, labels: Readonly<Record<string, string>>, amount: number): void {
    const series = this.gauge(name, labels)
    series.value += finiteMetricValue(amount, 'gauge amount')
  }

  observeHistogram(name: string, labels: Readonly<Record<string, string>>, value: number): void {
    const observed = finiteMetricValue(value, 'histogram value')
    const series = this.histogram(name, labels)
    series.count += 1
    series.sum += observed
    for (let index = 0; index < DEFAULT_HISTOGRAM_BUCKETS.length; index += 1) {
      const boundary = DEFAULT_HISTOGRAM_BUCKETS[index]
      if (boundary !== undefined && observed <= boundary) {
        series.buckets[index] = (series.buckets[index] ?? 0) + 1
      }
    }
  }

  /** Render every accumulated metric in deterministic Prometheus text exposition format. */
  render(): Uint8Array {
    const lines: string[] = []
    renderScalarMetrics(lines, this.counters, 'counter')
    renderScalarMetrics(lines, this.gauges, 'gauge')
    renderHistogramMetrics(lines, this.histograms)
    return new TextEncoder().encode(lines.length ? `${lines.join('\n')}\n` : '')
  }

  private counter(name: string, labels: Readonly<Record<string, string>>): MetricSeries {
    const key = metricKey(name, labels)
    const existing = this.counters.get(key)
    if (existing !== undefined) return existing
    const created = { labels: normalizedLabels(labels), value: 0 }
    this.counters.set(key, created)
    return created
  }

  private gauge(name: string, labels: Readonly<Record<string, string>>): MetricSeries {
    const key = metricKey(name, labels)
    const existing = this.gauges.get(key)
    if (existing !== undefined) return existing
    const created = { labels: normalizedLabels(labels), value: 0 }
    this.gauges.set(key, created)
    return created
  }

  private histogram(name: string, labels: Readonly<Record<string, string>>): HistogramSeries {
    const key = metricKey(name, labels)
    const existing = this.histograms.get(key)
    if (existing !== undefined) return existing
    const created: HistogramSeries = {
      labels: normalizedLabels(labels),
      value: 0,
      count: 0,
      buckets: DEFAULT_HISTOGRAM_BUCKETS.map(() => 0),
      sum: 0,
    }
    this.histograms.set(key, created)
    return created
  }
}

/** Local rotating sink with the same default retention limits as Python's RotatingFileHandler. */
class RotatingFileSink {
  private readonly backups: number
  private readonly maxBytes: number
  private readonly path: string

  constructor(path: string, maxBytes = DEFAULT_LOG_FILE_BYTES, backups = DEFAULT_LOG_FILE_BACKUPS) {
    if (!path.trim()) throw new TypeError('logFile must be a non-empty path')
    this.path = path
    this.maxBytes = positiveSafeInteger(maxBytes, 'maxLogFileBytes')
    this.backups = nonNegativeSafeInteger(backups, 'maxLogFileBackups')
    mkdirSync(dirname(path), { recursive: true })
    if (!existsSync(path)) appendFileSync(path, '')
  }

  write(chunk: string): void {
    try {
      this.rotateIfNeeded(new TextEncoder().encode(chunk).byteLength)
      appendFileSync(this.path, chunk, 'utf8')
    } catch {
      // A logging destination must not make the agent turn fail.
    }
  }

  private rotateIfNeeded(incomingBytes: number): void {
    if (!existsSync(this.path)) return
    if (statSync(this.path).size + incomingBytes <= this.maxBytes) return
    if (this.backups === 0) {
      rmSync(this.path, { force: true })
      return
    }
    for (let index = this.backups; index >= 1; index -= 1) {
      const destination = `${this.path}.${index}`
      const source = index === 1 ? this.path : `${this.path}.${index - 1}`
      if (!existsSync(source)) continue
      rmSync(destination, { force: true })
      renameSync(source, destination)
    }
  }
}

function humanLine(record: StructuredLogRecord): string {
  const fields = Object.fromEntries(Object.entries(record).filter(([key]) => !['timestamp', 'level', 'logger', 'event'].includes(key)))
  const suffix = Object.keys(fields).length ? ' ' + safeJson(fields) : ''
  return `${record.timestamp} - ${record.logger} - ${record.level.toUpperCase()} - ${record.event}${suffix}\n`
}

function jsonLine(record: StructuredLogRecord): string {
  return safeJson(record) + '\n'
}

function safeJson(value: unknown): string {
  const seen = new WeakSet<object>()
  return JSON.stringify(value, (_key, nested: unknown) => {
    if (nested instanceof Error) return { name: nested.name, message: nested.message }
    if (typeof nested === 'bigint') return nested.toString()
    if (typeof nested === 'function' || typeof nested === 'symbol') return String(nested)
    if (typeof nested === 'undefined') return null
    if (typeof nested === 'object' && nested !== null) {
      if (seen.has(nested)) return '[Circular]'
      seen.add(nested)
    }
    return nested
  })
}

function errorMessage(value: unknown): string {
  return value instanceof Error ? value.message : String(value)
}

function errorType(value: unknown): string {
  if (value instanceof Error) return value.name || 'Error'
  return typeof value === 'string' ? 'Error' : 'UnknownError'
}

function endSpan(span: StructuredTraceSpan | undefined): void {
  try {
    span?.end()
  } catch {
    // Telemetry must not alter normal logging control flow.
  }
}

function recordSpanError(span: StructuredTraceSpan | undefined, error: unknown): void {
  try {
    span?.recordException?.(error)
    span?.setStatus?.({ code: 'error', message: errorMessage(error) })
  } catch {
    // A host tracing implementation is isolated from application failures.
  }
}

function isPromise<T>(value: T | Promise<T>): value is Promise<T> {
  return value instanceof Promise
}

function finiteMetricValue(value: number, name: string): number {
  if (!Number.isFinite(value)) throw new TypeError(`${name} must be finite`)
  return value
}

function positiveSafeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive safe integer`)
  return value
}

function nonNegativeSafeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`${name} must be a non-negative safe integer`)
  return value
}

function normalizedLabels(labels: Readonly<Record<string, string>>): Readonly<Record<string, string>> {
  return Object.freeze(Object.fromEntries(Object.entries(labels)
    .map(([key, value]) => [key, String(value)] as const)
    .sort(([left], [right]) => left.localeCompare(right))))
}

function metricKey(name: string, labels: Readonly<Record<string, string>>): string {
  return `${name}|${JSON.stringify(normalizedLabels(labels))}`
}

function renderScalarMetrics(
  lines: string[],
  values: ReadonlyMap<string, MetricSeries>,
  type: 'counter' | 'gauge',
): void {
  const grouped = metricGroups(values)
  for (const [name, entries] of grouped) {
    lines.push(`# TYPE ${name} ${type}`)
    for (const entry of entries) lines.push(`${name}${formatMetricLabels(entry.labels)} ${entry.value}`)
  }
}

function renderHistogramMetrics(lines: string[], values: ReadonlyMap<string, HistogramSeries>): void {
  const grouped = metricGroups(values)
  for (const [name, entries] of grouped) {
    lines.push(`# TYPE ${name} histogram`)
    for (const entry of entries) {
      for (let index = 0; index < DEFAULT_HISTOGRAM_BUCKETS.length; index += 1) {
        const boundary = DEFAULT_HISTOGRAM_BUCKETS[index]
        if (boundary === undefined) continue
        lines.push(`${name}_bucket${formatMetricLabels({ ...entry.labels, le: String(boundary) })} ${entry.buckets[index] ?? 0}`)
      }
      lines.push(`${name}_bucket${formatMetricLabels({ ...entry.labels, le: '+Inf' })} ${entry.count}`)
      lines.push(`${name}_sum${formatMetricLabels(entry.labels)} ${entry.sum}`)
      lines.push(`${name}_count${formatMetricLabels(entry.labels)} ${entry.count}`)
    }
  }
}

function metricGroups<T extends MetricSeries>(values: ReadonlyMap<string, T>): Map<string, T[]> {
  const grouped = new Map<string, T[]>()
  for (const [key, value] of values) {
    const name = key.slice(0, key.indexOf('|'))
    const entries = grouped.get(name) ?? []
    entries.push(value)
    grouped.set(name, entries)
  }
  return new Map([...grouped].sort(([left], [right]) => left.localeCompare(right)).map(([name, entries]) => [
    name,
    [...entries].sort((left, right) => JSON.stringify(left.labels).localeCompare(JSON.stringify(right.labels))),
  ]))
}

function formatMetricLabels(labels: Readonly<Record<string, string>>): string {
  const entries = Object.entries(labels).sort(([left], [right]) => left.localeCompare(right))
  if (!entries.length) return ''
  return `{${entries.map(([name, value]) => `${name}="${escapeMetricLabel(value)}"`).join(',')}}`
}

function escapeMetricLabel(value: string): string {
  return value.replaceAll('\\', '\\\\').replaceAll('\n', '\\n').replaceAll('"', '\\"')
}
