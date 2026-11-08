// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { BenchmarkId, GradeStatus } from './benchmarks/types.js'

export type MatrixPrivacy = 'constrained' | 'unrestricted'

/** One task run by one config. Unreported measurements stay absent; nothing is invented. */
export interface TaskRecord {
  readonly configId: string
  readonly taskId: string
  readonly benchmark: BenchmarkId
  readonly status: GradeStatus | 'error'
  readonly accepted: boolean
  readonly score: number
  readonly details: string
  readonly disclosureFailure: boolean
  readonly wallMs: number
  readonly inputTokens?: number
  readonly outputTokens?: number
  readonly costUsd?: number
  readonly error?: string
}

export interface ExcludedTask {
  readonly benchmark: BenchmarkId
  readonly id: string
  readonly reason: string
}

export interface ConfigSummary {
  readonly id: string
  readonly privacy: MatrixPrivacy
  /** Runs attempted, including errors. Excluded tasks are not counted here. */
  readonly tasks: number
  readonly accepted: number
  readonly rejected: number
  readonly errors: number
  readonly needsReview: number
  readonly excluded: number
  /** accepted / (tasks - needsReview); null when nothing was gradable. */
  readonly accuracy: number | null
  readonly medianWallMs: number | null
  /** Sum of the costs that were reported; null when no run reported a cost. */
  readonly totalCostUsd: number | null
  readonly costReportedTasks: number
  /** totalCostUsd / accepted; null when there is no cost or no accepted result. */
  readonly costPerAcceptedUsd: number | null
  readonly inputTokens: number | null
  readonly outputTokens: number | null
  readonly disclosureFailures: number
}

export interface MatrixReport {
  readonly generatedAt: string
  readonly constrained: readonly ConfigSummary[]
  readonly unrestricted: readonly ConfigSummary[]
  readonly records: readonly TaskRecord[]
  readonly excluded: readonly ExcludedTask[]
}

export function median(values: readonly number[]): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  const high = sorted[middle]
  const low = sorted[middle - 1]
  if (high === undefined) return null
  return sorted.length % 2 === 1 || low === undefined ? high : (low + high) / 2
}

function sumReported(values: readonly (number | undefined)[]): { readonly total: number | null; readonly count: number } {
  let total = 0
  let count = 0
  for (const value of values) {
    if (value === undefined) continue
    total += value
    count += 1
  }
  return { count, total: count === 0 ? null : total }
}

export function summarizeConfig(
  config: { readonly id: string; readonly privacy: MatrixPrivacy },
  records: readonly TaskRecord[],
  excludedCount: number,
): ConfigSummary {
  const mine = records.filter(record => record.configId === config.id)
  const accepted = mine.filter(record => record.accepted).length
  const needsReview = mine.filter(record => record.status === 'needs_review').length
  const errors = mine.filter(record => record.status === 'error').length
  const rejected = mine.filter(record => record.status === 'rejected').length
  const gradable = mine.length - needsReview
  const cost = sumReported(mine.map(record => record.costUsd))
  return {
    accepted,
    accuracy: gradable > 0 ? accepted / gradable : null,
    costPerAcceptedUsd: cost.total !== null && accepted > 0 ? cost.total / accepted : null,
    costReportedTasks: cost.count,
    disclosureFailures: mine.filter(record => record.disclosureFailure).length,
    errors,
    excluded: excludedCount,
    id: config.id,
    inputTokens: sumReported(mine.map(record => record.inputTokens)).total,
    medianWallMs: median(mine.map(record => record.wallMs)),
    needsReview,
    outputTokens: sumReported(mine.map(record => record.outputTokens)).total,
    privacy: config.privacy,
    rejected,
    tasks: mine.length,
    totalCostUsd: cost.total,
  }
}

export function buildReport(input: {
  readonly configs: readonly { readonly id: string; readonly privacy: MatrixPrivacy }[]
  readonly records: readonly TaskRecord[]
  readonly excluded: readonly ExcludedTask[]
  readonly generatedAt: string
}): MatrixReport {
  const summaries = input.configs.map(config => summarizeConfig(config, input.records, input.excluded.length))
  return {
    constrained: summaries.filter(summary => summary.privacy === 'constrained'),
    excluded: input.excluded,
    generatedAt: input.generatedAt,
    records: input.records,
    unrestricted: summaries.filter(summary => summary.privacy === 'unrestricted'),
  }
}

const NOT_AVAILABLE = 'n/a'

function percent(value: number | null): string {
  return value === null ? NOT_AVAILABLE : `${(value * 100).toFixed(1)}%`
}

function seconds(value: number | null): string {
  return value === null ? NOT_AVAILABLE : (value / 1000).toFixed(1)
}

function dollars(value: number | null): string {
  return value === null ? NOT_AVAILABLE : `$${value.toFixed(4)}`
}

function table(summaries: readonly ConfigSummary[]): string[] {
  if (summaries.length === 0) return ['_none_']
  const lines = [
    '| Config | Tasks | Accepted | Accuracy | Needs review | Excluded | Median wall (s) | Total cost | Cost / accepted | Disclosure failures |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
  ]
  for (const summary of summaries) {
    lines.push(
      `| ${summary.id} | ${summary.tasks} | ${summary.accepted} | ${percent(summary.accuracy)} | ${summary.needsReview} `
      + `| ${summary.excluded} | ${seconds(summary.medianWallMs)} | ${dollars(summary.totalCostUsd)} `
      + `| ${dollars(summary.costPerAcceptedUsd)} | ${summary.disclosureFailures} |`,
    )
  }
  return lines
}

/** Markdown with the privacy-constrained and unrestricted configs in separate sections. */
export function renderMarkdown(report: MatrixReport): string {
  const lines = [
    '# Benchmark matrix',
    '',
    `Generated ${report.generatedAt}. "n/a" means the run did not report the measurement.`,
    'Accuracy = accepted / (tasks - needs review). Cost / accepted is n/a when no cost was reported or nothing was accepted.',
    '',
    '## Privacy-constrained configs',
    '',
    ...table(report.constrained),
    '',
    '## Unrestricted configs',
    '',
    ...table(report.unrestricted),
  ]
  if (report.excluded.length > 0) {
    lines.push('', '## Excluded tasks', '')
    for (const entry of report.excluded) lines.push(`- ${entry.benchmark} / ${entry.id}: ${entry.reason}`)
  }
  return `${lines.join('\n')}\n`
}
