// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { calcCost } from '../llms/providerRegistry.js'

const CACHE_READ_MULTIPLIER = 0.1
const CACHE_CREATION_MULTIPLIER = 1.25
const PRICING_PROBE_TOKENS = 1_000

/** Bucket used by aggregate views for events without an explicit scope. */
export const UNSCOPED_COST_SCOPE = '(unscoped)'

export type CostCalculator = (model: string, inputTokens: number, outputTokens: number) => number

export interface CostEventOptions {
  readonly agentId?: string
  readonly cacheCreationTokens?: number
  readonly cacheReadTokens?: number
  readonly costUsd: number
  readonly inputTokens: number
  readonly label?: string
  readonly model: string
  readonly outputTokens: number
  readonly sessionId?: string
  readonly timestamp?: string
}

export interface CostEventRecord {
  readonly agent_id: string | null
  readonly cache_creation_tokens: number
  readonly cache_read_tokens: number
  readonly cost_usd: number
  readonly in_tokens: number
  readonly label: string
  readonly model: string
  readonly out_tokens: number
  readonly session_id: string | null
  readonly timestamp: string
}

export interface LegacyCostEventRecord {
  readonly cost_usd: number
  readonly in_tokens: number
  readonly label: string
  readonly model: string
  readonly out_tokens: number
  readonly timestamp: string
}

export interface RecordTurnOptions {
  readonly agentId?: string
  readonly cacheCreationTokens?: number
  readonly cacheReadTokens?: number
  readonly sessionId?: string
  readonly timestamp?: string
}

export interface RecordRawOptions {
  readonly agentId?: string
  readonly sessionId?: string
  readonly timestamp?: string
}

export interface CostTrackerOptions {
  /** Default identity applied to events that omit an agent ID. */
  readonly agentId?: string
  /** Provider/model cost function; defaults to the shared pricing registry. */
  readonly costCalculator?: CostCalculator
  /** Injectable wall-clock for deterministic timestamps. */
  readonly now?: () => Date
  /** Default identity applied to events that omit a session ID. */
  readonly sessionId?: string
}

export interface CostAggregate {
  readonly cacheCreationTokens: number
  readonly cacheHitRate: number
  readonly cacheReadTokens: number
  readonly costUsd: number
  readonly inputTokens: number
  readonly outputTokens: number
  readonly tokens: number
  readonly turns: number
}

/**
 * Immutable record for one priced LLM or externally-priced operation.
 *
 * Token counts deliberately exclude cache reads from inputTokens, matching
 * the Python ledger. Cache reads and writes are retained separately so
 * pricing can be recalculated without losing raw provider usage.
 */
export class CostEvent {
  readonly agentId: string | undefined
  readonly cacheCreationTokens: number
  readonly cacheReadTokens: number
  readonly costUsd: number
  readonly inputTokens: number
  readonly label: string
  readonly model: string
  readonly outputTokens: number
  readonly sessionId: string | undefined
  readonly timestamp: string

  constructor(options: CostEventOptions) {
    this.model = stringValue(options.model, 'model')
    this.inputTokens = tokenCount(options.inputTokens, 'inputTokens')
    this.outputTokens = tokenCount(options.outputTokens, 'outputTokens')
    this.costUsd = finiteNumber(options.costUsd, 'costUsd')
    this.label = stringValue(options.label ?? '', 'label')
    this.timestamp = timestampValue(options.timestamp ?? new Date().toISOString())
    this.cacheReadTokens = tokenCount(options.cacheReadTokens ?? 0, 'cacheReadTokens')
    this.cacheCreationTokens = tokenCount(options.cacheCreationTokens ?? 0, 'cacheCreationTokens')
    this.sessionId = scopeValue(options.sessionId, 'sessionId')
    this.agentId = scopeValue(options.agentId, 'agentId')
    Object.freeze(this)
  }

  /** Full persistence form including cache usage and session/agent attribution. */
  toRecord(): CostEventRecord {
    return {
      model: this.model,
      in_tokens: this.inputTokens,
      out_tokens: this.outputTokens,
      cost_usd: this.costUsd,
      label: this.label,
      timestamp: this.timestamp,
      cache_read_tokens: this.cacheReadTokens,
      cache_creation_tokens: this.cacheCreationTokens,
      session_id: this.sessionId ?? null,
      agent_id: this.agentId ?? null,
    }
  }

  /** Python cost_tracker.py serialization shape, retained for old ledgers. */
  toLegacyRecord(): LegacyCostEventRecord {
    return {
      model: this.model,
      in_tokens: this.inputTokens,
      out_tokens: this.outputTokens,
      cost_usd: this.costUsd,
      label: this.label,
      timestamp: this.timestamp,
    }
  }
}

/**
 * Append-only LLM cost ledger with cache-aware pricing and scope aggregates.
 *
 * A tracker can serve one session by supplying sessionId at construction, or
 * a host-wide ledger by attaching sessionId and agentId to individual events.
 */
export class CostTracker {
  private readonly calculator: CostCalculator
  private readonly clock: () => Date
  private readonly defaultAgentId: string | undefined
  private readonly defaultSessionId: string | undefined
  private readonly ledger: CostEvent[] = []

  constructor(options: CostTrackerOptions = {}) {
    this.calculator = options.costCalculator ?? calcCost
    this.clock = options.now ?? (() => new Date())
    this.defaultSessionId = scopeValue(options.sessionId, 'sessionId')
    this.defaultAgentId = scopeValue(options.agentId, 'agentId')
  }

  /** Snapshot event list; callers cannot mutate the underlying ledger array. */
  get events(): readonly CostEvent[] {
    return [...this.ledger]
  }

  get eventCount(): number {
    return this.ledger.length
  }

  get totalCostUsd(): number {
    return aggregate(this.ledger).costUsd
  }

  get totalInputTokens(): number {
    return aggregate(this.ledger).inputTokens
  }

  get totalOutputTokens(): number {
    return aggregate(this.ledger).outputTokens
  }

  get totalTokens(): number {
    return this.totalInputTokens + this.totalOutputTokens
  }

  get totalCacheReadTokens(): number {
    return aggregate(this.ledger).cacheReadTokens
  }

  get totalCacheCreationTokens(): number {
    return aggregate(this.ledger).cacheCreationTokens
  }

  /** Convenience alias for callers that only need total spend. */
  get totalCost(): number {
    return this.totalCostUsd
  }

  /**
   * Price and append one LLM completion event.
   *
   * Cache reads cost 10% of normal input-token price and cache creation costs
   * 125%, matching the Python implementation. The shared calcCost registry
   * keeps provider-prefixed model names and unknown-model zero-pricing intact.
   */
  recordTurn(
    model: string,
    inputTokens: number,
    outputTokens: number,
    label = '',
    options: RecordTurnOptions = {},
  ): CostEvent {
    const validatedModel = stringValue(model, 'model')
    const validatedInput = tokenCount(inputTokens, 'inputTokens')
    const validatedOutput = tokenCount(outputTokens, 'outputTokens')
    const cacheReadTokens = tokenCount(options.cacheReadTokens ?? 0, 'cacheReadTokens')
    const cacheCreationTokens = tokenCount(options.cacheCreationTokens ?? 0, 'cacheCreationTokens')
    const baseCost = finiteNumber(this.calculator(validatedModel, validatedInput, validatedOutput), 'calculated cost')
    let cacheCost = 0
    if (cacheReadTokens || cacheCreationTokens) {
      const inputProbe = finiteNumber(this.calculator(validatedModel, PRICING_PROBE_TOKENS, 0), 'input pricing')
      const inputRate = inputProbe > 0 ? inputProbe / PRICING_PROBE_TOKENS : 0
      cacheCost = cacheReadTokens * inputRate * CACHE_READ_MULTIPLIER
        + cacheCreationTokens * inputRate * CACHE_CREATION_MULTIPLIER
    }
    return this.append(new CostEvent({
      model: validatedModel,
      inputTokens: validatedInput,
      outputTokens: validatedOutput,
      costUsd: baseCost + cacheCost,
      label: stringValue(label, 'label'),
      timestamp: options.timestamp ?? this.nowTimestamp(),
      cacheReadTokens,
      cacheCreationTokens,
      ...(this.scopeOptions(options)),
    }))
  }

  /** Append a separately-priced operation such as embeddings or image generation. */
  recordRaw(label: string, costUsd: number, model = '', options: RecordRawOptions = {}): CostEvent {
    return this.append(new CostEvent({
      model: stringValue(model, 'model'),
      inputTokens: 0,
      outputTokens: 0,
      costUsd: finiteNumber(costUsd, 'costUsd'),
      label: stringValue(label, 'label'),
      timestamp: options.timestamp ?? this.nowTimestamp(),
      ...(this.scopeOptions(options)),
    }))
  }

  /** Append a pre-built immutable event, preserving its original timestamp and scopes. */
  record(event: CostEvent): CostEvent {
    if (!(event instanceof CostEvent)) throw new TypeError('event must be a CostEvent')
    return this.append(event)
  }

  /** Fraction of all served input tokens supplied by a prompt-cache hit. */
  cacheHitRate(): number {
    return cacheHitRate(this.ledger)
  }

  /** Aggregate events by billed model in insertion order. */
  byModel(): Readonly<Record<string, CostAggregate>> {
    return groupBy(this.ledger, event => event.model)
  }

  /** Aggregate events by session; events without a session use the exported unscoped bucket. */
  bySession(): Readonly<Record<string, CostAggregate>> {
    return groupBy(this.ledger, event => event.sessionId ?? UNSCOPED_COST_SCOPE)
  }

  /** Aggregate events by agent; events without an agent use the exported unscoped bucket. */
  byAgent(): Readonly<Record<string, CostAggregate>> {
    return groupBy(this.ledger, event => event.agentId ?? UNSCOPED_COST_SCOPE)
  }

  /** Aggregate exactly one scoped session, returning zeroes when it has no events. */
  forSession(sessionId: string): CostAggregate {
    const expected = scopeValue(sessionId, 'sessionId')
    return aggregate(this.ledger.filter(event => event.sessionId === expected))
  }

  /** Aggregate exactly one scoped agent, returning zeroes when it has no events. */
  forAgent(agentId: string): CostAggregate {
    const expected = scopeValue(agentId, 'agentId')
    return aggregate(this.ledger.filter(event => event.agentId === expected))
  }

  /** Drop all ledger records. */
  clear(): void {
    this.ledger.length = 0
  }

  /** Markdown view matching the Python total and model-breakdown summary. */
  summary(): string {
    const totals = aggregate(this.ledger)
    const lines = [
      '# Cost Summary',
      '',
      'Total cost: $' + totals.costUsd.toFixed(4),
      'Total tokens: ' + formatInteger(totals.tokens)
        + ' (in: ' + formatInteger(totals.inputTokens)
        + ', out: ' + formatInteger(totals.outputTokens) + ')',
      'Events: ' + totals.turns,
      '',
    ]
    const models = this.byModel()
    if (Object.keys(models).length) {
      lines.push('## By Model')
      for (const [model, stats] of Object.entries(models).sort(([left], [right]) => left.localeCompare(right))) {
        lines.push('- **' + model + '**: $' + stats.costUsd.toFixed(4)
          + ' (' + stats.turns + ' turns, ' + formatInteger(stats.tokens) + ' tokens)')
      }
    }
    return lines.join('\n')
  }

  /** Full JSON-safe ledger records, including cache and scope fields. */
  asRecords(): readonly CostEventRecord[] {
    return this.ledger.map(event => event.toRecord())
  }

  /** Python cost_tracker.py persistence records, omitting cache and scope extensions. */
  asDicts(): readonly LegacyCostEventRecord[] {
    return this.ledger.map(event => event.toLegacyRecord())
  }

  private append(event: CostEvent): CostEvent {
    this.ledger.push(event)
    return event
  }

  private nowTimestamp(): string {
    const now = this.clock()
    if (!(now instanceof Date) || Number.isNaN(now.valueOf())) {
      throw new RangeError('now must return a valid Date')
    }
    return now.toISOString()
  }

  private scopeOptions(options: RecordTurnOptions | RecordRawOptions): {
    readonly agentId?: string
    readonly sessionId?: string
  } {
    const sessionId = scopeValue(options.sessionId, 'sessionId') ?? this.defaultSessionId
    const agentId = scopeValue(options.agentId, 'agentId') ?? this.defaultAgentId
    return {
      ...(sessionId !== undefined ? { sessionId } : {}),
      ...(agentId !== undefined ? { agentId } : {}),
    }
  }
}

function aggregate(events: readonly CostEvent[]): CostAggregate {
  let inputTokens = 0
  let outputTokens = 0
  let cacheReadTokens = 0
  let cacheCreationTokens = 0
  let costUsd = 0
  for (const event of events) {
    inputTokens += event.inputTokens
    outputTokens += event.outputTokens
    cacheReadTokens += event.cacheReadTokens
    cacheCreationTokens += event.cacheCreationTokens
    costUsd += event.costUsd
  }
  return Object.freeze({
    turns: events.length,
    inputTokens,
    outputTokens,
    tokens: inputTokens + outputTokens,
    cacheReadTokens,
    cacheCreationTokens,
    cacheHitRate: cacheHitRateValues(inputTokens, cacheReadTokens),
    costUsd,
  })
}

function groupBy(
  events: readonly CostEvent[],
  keyFor: (event: CostEvent) => string,
): Readonly<Record<string, CostAggregate>> {
  const groups = new Map<string, CostEvent[]>()
  for (const event of events) {
    const key = keyFor(event)
    const values = groups.get(key) ?? []
    values.push(event)
    groups.set(key, values)
  }
  const result: Record<string, CostAggregate> = {}
  for (const [key, values] of groups) result[key] = aggregate(values)
  return Object.freeze(result)
}

function cacheHitRate(events: readonly CostEvent[]): number {
  const totals = aggregateTokenCounts(events)
  return cacheHitRateValues(totals.inputTokens, totals.cacheReadTokens)
}

function aggregateTokenCounts(events: readonly CostEvent[]): {
  readonly cacheReadTokens: number
  readonly inputTokens: number
} {
  let inputTokens = 0
  let cacheReadTokens = 0
  for (const event of events) {
    inputTokens += event.inputTokens
    cacheReadTokens += event.cacheReadTokens
  }
  return { inputTokens, cacheReadTokens }
}

function cacheHitRateValues(inputTokens: number, cacheReadTokens: number): number {
  const served = inputTokens + cacheReadTokens
  return served > 0 ? cacheReadTokens / served : 0
}

function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(name + ' must be a finite number')
  }
  return value
}

function tokenCount(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative safe integer')
  }
  return value
}

function stringValue(value: unknown, name: string): string {
  if (typeof value !== 'string') throw new TypeError(name + ' must be a string')
  return value
}

function scopeValue(value: unknown, name: string): string | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(name + ' must be a non-empty string')
  return value
}

function timestampValue(value: unknown): string {
  if (typeof value !== 'string' || Number.isNaN(new Date(value).valueOf())) {
    throw new RangeError('timestamp must be a valid ISO timestamp')
  }
  return value
}

function formatInteger(value: number): string {
  return value.toLocaleString('en-US')
}
