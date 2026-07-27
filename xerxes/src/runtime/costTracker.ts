// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { QuerySource } from '../llms/client.js'
import { calcCost } from '../llms/providerRegistry.js'

const CACHE_READ_MULTIPLIER = 0.1
const CACHE_CREATION_MULTIPLIER = 1.25
const PRICING_PROBE_TOKENS = 1_000

/** Bucket used by aggregate views for events without an explicit scope. */
export const UNSCOPED_COST_SCOPE = '(unscoped)'

/**
 * The one source that is not housekeeping.
 *
 * Compared as a literal instead of importing the llms helper so the ledger
 * keeps a type-only dependency on the provider layer; the constant is typed as
 * {@link QuerySource}, so a typo or a renamed union member fails to compile.
 */
const MAIN_COST_SOURCE: QuerySource = 'main'

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
  /** Why the call was made; absent on legacy events recorded before the dimension existed. */
  readonly source?: QuerySource
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
  readonly source: string | null
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
  readonly source?: QuerySource
  readonly timestamp?: string
}

export interface RecordRawOptions {
  readonly agentId?: string
  readonly sessionId?: string
  readonly source?: QuerySource
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
  /** Default source applied to events that omit one, e.g. a compaction-owned tracker. */
  readonly source?: QuerySource
}

/** Filters narrowing a source breakdown to one session and/or agent. */
export interface CostSourceBreakdownOptions {
  readonly agentId?: string
  readonly sessionId?: string
}

/**
 * Answer to "how much of this spend was housekeeping".
 *
 * `untagged` is kept separate from both halves rather than folded into
 * housekeeping: events recorded before a call site was taught to pass a source
 * are unknown, not proven housekeeping, and merging them would overstate the
 * saving an auxiliary route appears to deliver.
 */
export interface CostSourceBreakdown {
  readonly bySource: Readonly<Record<string, CostAggregate>>
  readonly housekeeping: CostAggregate
  /** Housekeeping share of priced spend in [0, 1]; zero when nothing cost anything. */
  readonly housekeepingCostShare: number
  readonly main: CostAggregate
  readonly total: CostAggregate
  readonly untagged: CostAggregate
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
  readonly source: QuerySource | undefined
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
    this.source = sourceValue(options.source)
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
      source: this.source ?? null,
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
  private readonly defaultSource: QuerySource | undefined
  private readonly ledger: CostEvent[] = []

  constructor(options: CostTrackerOptions = {}) {
    this.calculator = options.costCalculator ?? calcCost
    this.clock = options.now ?? (() => new Date())
    this.defaultSessionId = scopeValue(options.sessionId, 'sessionId')
    this.defaultAgentId = scopeValue(options.agentId, 'agentId')
    this.defaultSource = sourceValue(options.source)
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

  /** Aggregate events by call source; events recorded without one use the unscoped bucket. */
  bySource(): Readonly<Record<string, CostAggregate>> {
    return groupBy(this.ledger, event => event.source ?? UNSCOPED_COST_SCOPE)
  }

  /** Aggregate exactly one call source, returning zeroes when it has no events. */
  forSource(source: QuerySource): CostAggregate {
    return aggregate(this.ledger.filter(event => event.source === source))
  }

  /**
   * Split spend into main-loop, housekeeping, and untagged buckets.
   *
   * This is the accessor that makes an auxiliary route observable: without it
   * there is no way to tell whether compaction, titling, and memory extraction
   * moved off the main model or are still billing at main-model rates.
   * Housekeeping is defined as "tagged with any source other than main", so a
   * newly added source counts immediately instead of silently landing in the
   * main bucket until this file is updated.
   */
  sourceBreakdown(options: CostSourceBreakdownOptions = {}): CostSourceBreakdown {
    const sessionId = scopeValue(options.sessionId, 'sessionId')
    const agentId = scopeValue(options.agentId, 'agentId')
    const scoped = this.ledger.filter(event =>
      (sessionId === undefined || event.sessionId === sessionId)
      && (agentId === undefined || event.agentId === agentId))
    const total = aggregate(scoped)
    const housekeeping = aggregate(scoped.filter(event =>
      event.source !== undefined && event.source !== MAIN_COST_SOURCE))
    return Object.freeze({
      total,
      main: aggregate(scoped.filter(event => event.source === MAIN_COST_SOURCE)),
      housekeeping,
      untagged: aggregate(scoped.filter(event => event.source === undefined)),
      bySource: groupBy(scoped, event => event.source ?? UNSCOPED_COST_SCOPE),
      housekeepingCostShare: total.costUsd > 0 ? housekeeping.costUsd / total.costUsd : 0,
    })
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
    // Only ledgers that actually carry sources gain the section, so summaries
    // from callers that have not adopted the dimension stay byte-identical.
    const breakdown = this.sourceBreakdown()
    if (breakdown.housekeeping.turns || breakdown.main.turns) {
      lines.push('', '## By Source')
      for (const [source, stats] of Object.entries(breakdown.bySource)
        .sort(([left], [right]) => left.localeCompare(right))) {
        lines.push('- **' + source + '**: $' + stats.costUsd.toFixed(4)
          + ' (' + stats.turns + ' turns, ' + formatInteger(stats.tokens) + ' tokens)')
      }
      lines.push('Housekeeping share: ' + (breakdown.housekeepingCostShare * 100).toFixed(1) + '%')
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
    readonly source?: QuerySource
  } {
    const sessionId = scopeValue(options.sessionId, 'sessionId') ?? this.defaultSessionId
    const agentId = scopeValue(options.agentId, 'agentId') ?? this.defaultAgentId
    const source = sourceValue(options.source) ?? this.defaultSource
    return {
      ...(sessionId !== undefined ? { sessionId } : {}),
      ...(agentId !== undefined ? { agentId } : {}),
      ...(source !== undefined ? { source } : {}),
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

/**
 * Accept a source or its absence.
 *
 * The union is enforced by the compiler at every typed call site, so this only
 * has to reject the shapes a plain `string` cast could smuggle in — an empty
 * bucket key would silently merge tagged spend into a nameless group.
 */
function sourceValue(value: QuerySource | undefined): QuerySource | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'string' || !value.trim()) throw new TypeError('source must be a non-empty string')
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
