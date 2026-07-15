// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface CostEventLike {
  readonly cacheCreationTokens?: number
  readonly cacheReadTokens?: number
  readonly cache_creation_tokens?: number
  readonly cache_read_tokens?: number
  readonly costUsd?: number
  readonly cost_usd?: number
  readonly in_tokens?: number
  readonly inputTokens?: number
  readonly label?: string
  readonly model: string
  readonly out_tokens?: number
  readonly outputTokens?: number
  readonly timestamp: string
}

export interface ModelInsights {
  readonly costUsd: number
  readonly events: number
  readonly inputTokens: number
  readonly outputTokens: number
}

export interface DayInsights {
  readonly costUsd: number
  readonly events: number
}

export interface InsightsReport {
  readonly byDay: Readonly<Record<string, DayInsights>>
  readonly byLabel: Readonly<Record<string, number>>
  readonly byModel: Readonly<Record<string, ModelInsights>>
  readonly cacheHitRate: number
  readonly projectedMonthlyCost: number
  readonly totalCacheCreationTokens: number
  readonly totalCacheReadTokens: number
  readonly totalCostUsd: number
  readonly totalEvents: number
  readonly totalInputTokens: number
  readonly totalOutputTokens: number
}

export interface BuildInsightsOptions {
  readonly days?: number
  readonly now?: Date | (() => Date)
}

/** Aggregate cost events into model, day, label, cache, and burn-rate insights. */
export function buildInsightsReport(
  events: Iterable<CostEventLike>,
  options: BuildInsightsOptions = {},
): InsightsReport {
  const now = resolveNow(options.now)
  const days = options.days
  if (days !== undefined && (!Number.isFinite(days) || days < 0)) throw new RangeError('days must be non-negative')
  const horizon = days && days > 0 ? new Date(now.valueOf() - days * 86_400_000) : undefined
  const byModel: Record<string, MutableModelInsights> = {}
  const byDay: Record<string, MutableDayInsights> = {}
  const byLabel: Record<string, number> = {}
  let totalEvents = 0
  let totalCostUsd = 0
  let totalInputTokens = 0
  let totalOutputTokens = 0
  let totalCacheReadTokens = 0
  let totalCacheCreationTokens = 0

  for (const event of events) {
    const timestamp = parseTimestamp(event.timestamp, now)
    if (horizon !== undefined && timestamp.valueOf() < horizon.valueOf()) continue
    const costUsd = numberValue(event.costUsd ?? event.cost_usd)
    const inputTokens = numberValue(event.inputTokens ?? event.in_tokens)
    const outputTokens = numberValue(event.outputTokens ?? event.out_tokens)
    const cacheReadTokens = numberValue(event.cacheReadTokens ?? event.cache_read_tokens)
    const cacheCreationTokens = numberValue(event.cacheCreationTokens ?? event.cache_creation_tokens)
    totalEvents += 1
    totalCostUsd += costUsd
    totalInputTokens += inputTokens
    totalOutputTokens += outputTokens
    totalCacheReadTokens += cacheReadTokens
    totalCacheCreationTokens += cacheCreationTokens

    const model = event.model
    const modelBucket = byModel[model] ?? { events: 0, costUsd: 0, inputTokens: 0, outputTokens: 0 }
    modelBucket.events += 1
    modelBucket.costUsd += costUsd
    modelBucket.inputTokens += inputTokens
    modelBucket.outputTokens += outputTokens
    byModel[model] = modelBucket

    const day = timestamp.toISOString().slice(0, 10)
    const dayBucket = byDay[day] ?? { events: 0, costUsd: 0 }
    dayBucket.events += 1
    dayBucket.costUsd += costUsd
    byDay[day] = dayBucket

    if (event.label) byLabel[event.label] = (byLabel[event.label] ?? 0) + 1
  }

  const served = totalInputTokens + totalCacheReadTokens
  const projectedMonthlyCost = Object.keys(byDay).length
    ? totalCostUsd / Object.keys(byDay).length * 30
    : 0
  return Object.freeze({
    totalEvents,
    totalCostUsd,
    totalInputTokens,
    totalOutputTokens,
    totalCacheReadTokens,
    totalCacheCreationTokens,
    byModel: freezeModelBuckets(byModel),
    byDay: freezeDayBuckets(byDay),
    byLabel: Object.freeze({ ...byLabel }),
    cacheHitRate: served ? totalCacheReadTokens / served : 0,
    projectedMonthlyCost,
  })
}

/** Source-compatible concise alias. */
export const buildReport = buildInsightsReport

/** Render an insights report as the source Markdown overview. */
export function formatInsightsReport(report: InsightsReport): string {
  const lines = [
    '# Xerxes Insights',
    '',
    'Events: ' + report.totalEvents,
    'Total cost: $' + report.totalCostUsd.toFixed(4),
    'Input tokens: ' + formatInteger(report.totalInputTokens),
    'Output tokens: ' + formatInteger(report.totalOutputTokens),
    'Cache read tokens: ' + formatInteger(report.totalCacheReadTokens),
    'Cache hit rate: ' + (report.cacheHitRate * 100).toFixed(1) + '%',
    'Projected monthly cost: $' + report.projectedMonthlyCost.toFixed(2),
  ]
  const models = Object.entries(report.byModel)
    .sort(([, left], [, right]) => right.costUsd - left.costUsd)
  if (models.length) {
    lines.push('\n## Top models')
    for (const [model, stats] of models) {
      lines.push('- ' + model + ': $' + stats.costUsd.toFixed(4)
        + ' (' + stats.events + ' events, ' + formatInteger(stats.inputTokens)
        + ' in / ' + formatInteger(stats.outputTokens) + ' out)')
    }
  }
  return lines.join('\n')
}

/** Source-compatible concise alias. */
export const formatReport = formatInsightsReport

interface MutableModelInsights {
  costUsd: number
  events: number
  inputTokens: number
  outputTokens: number
}

interface MutableDayInsights {
  costUsd: number
  events: number
}

function resolveNow(value: Date | (() => Date) | undefined): Date {
  const now = typeof value === 'function' ? value() : value ?? new Date()
  if (!(now instanceof Date) || Number.isNaN(now.valueOf())) throw new RangeError('now must be a valid Date')
  return now
}

function parseTimestamp(value: string, fallback: Date): Date {
  const parsed = new Date(value)
  return Number.isNaN(parsed.valueOf()) ? fallback : parsed
}

function numberValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function freezeModelBuckets(values: Readonly<Record<string, MutableModelInsights>>): Readonly<Record<string, ModelInsights>> {
  const result: Record<string, ModelInsights> = {}
  for (const [key, value] of Object.entries(values)) result[key] = Object.freeze({ ...value })
  return Object.freeze(result)
}

function freezeDayBuckets(values: Readonly<Record<string, MutableDayInsights>>): Readonly<Record<string, DayInsights>> {
  const result: Record<string, DayInsights> = {}
  for (const [key, value] of Object.entries(values)) result[key] = Object.freeze({ ...value })
  return Object.freeze(result)
}

function formatInteger(value: number): string {
  return value.toLocaleString('en-US')
}
