// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isAbsolute, posix, relative, resolve } from 'node:path'

/** One observed tool invocation during an SWE task run. */
export interface SweToolInvocationTrace {
  readonly name: string
  /** 1-based turn index in which the tool call occurred. */
  readonly turnIndex: number
  /** False when the tool call failed (for example target-not-found or patch hunk mismatch). Defaults to true. */
  readonly ok?: boolean
  /** Workspace-relative or absolute file path inspected or edited by this tool call. */
  readonly filePath?: string
  /** Cumulative session tokens consumed up to and including this tool call's turn, when known. */
  readonly cumulativeTokens?: number
}

/** Per-turn token and tool telemetry emitted during an SWE task run. */
export interface SweTurnTelemetry {
  /** 1-based turn index. */
  readonly turnIndex: number
  /** Tokens consumed during this turn (or cumulative delta for this turn). */
  readonly tokens: number
  readonly latencyMs?: number
  readonly tools?: readonly SweToolInvocationTrace[]
}

/** Optional rich telemetry payload returned by an SWE solver or `EvaluationSessionPort`. */
export interface SweRunTelemetry {
  readonly turns?: readonly SweTurnTelemetry[]
  readonly toolCalls?: readonly SweToolInvocationTrace[]
  readonly inputTokens?: number
  readonly outputTokens?: number
  readonly totalTokens?: number
  readonly costUsd?: number
}

/** Optional host session-port extension exposing structured SWE tool/token traces. */
export interface SweTelemetryReportingPort {
  getSweTelemetry(): SweRunTelemetry | Promise<SweRunTelemetry>
}

/** Per-task grading and telemetry record produced by `runSweBenchmark`. */
export interface SweTaskRecord {
  readonly taskId: string
  readonly category: string
  readonly difficulty: number
  readonly groundTruthTargetFiles: readonly string[]
  readonly baselineTestsPassed: boolean
  readonly visibleTestsPassed: boolean
  readonly holdoutTestsPassed: boolean
  readonly resolved: boolean
  readonly holdoutRegression: boolean
  readonly baselineRegression: boolean
  readonly turnsToFirstRelevantRead: number | null
  readonly tokensToFirstRelevantRead: number | null
  readonly editAttempts: number
  readonly failedEditAttempts: number
  readonly wallMs: number
  readonly totalTokens?: number
  readonly costUsd?: number
  readonly detail: string
  readonly error?: string
}

/** KPI 2: Ground-truth file localization efficiency across SWE tasks. */
export interface SweLocalizationEfficiency {
  readonly localizedTaskCount: number
  readonly localizationHitRate: number
  readonly turnsToFirstRelevantRead: number | null
  readonly tokensToFirstRelevantRead: number | null
  readonly meanTurnsToFirstRelevantRead: number | null
  readonly meanTokensToFirstRelevantRead: number | null
}

/** KPI 5: Latency, token, and dollar efficiency among resolved tasks. */
export interface SweCostAndLatencyPerResolved {
  readonly resolvedCount: number
  readonly meanWallMsPerResolved: number | null
  readonly meanTokensPerResolved: number | null
  readonly meanCostUsdPerResolved: number | null
  readonly amortizedWallMsPerResolved: number | null
  readonly amortizedTokensPerResolved: number | null
  readonly amortizedCostUsdPerResolved: number | null
}

/** The 5 Core SWE KPIs computed over a benchmark run. */
export interface SweBenchmarkMetrics {
  readonly totalTasks: number
  readonly resolvedCount: number
  readonly holdoutRegressionCount: number
  readonly totalEditAttempts: number
  readonly failedEditAttempts: number
  /** 1. Pass@1 requiring baseline + visible reproduction + hidden holdout tests to pass. */
  readonly resolvedRate: number
  /** 2. Turns and tokens needed to read the first ground-truth target file. */
  readonly localizationEfficiency: SweLocalizationEfficiency
  /** 3. Failed FileEditTool / ApplyPatchTool calls divided by total edit attempts. */
  readonly editMismatchRate: number
  /** 4. Fraction of tasks where visible reproduction tests passed but hidden holdout tests failed. */
  readonly holdoutRegressionRate: number
  /** 5. Mean wall time, token usage, and USD cost per resolved task. */
  readonly costAndLatencyPerResolved: SweCostAndLatencyPerResolved
}

/** Complete report returned by `runSweBenchmark`. */
export interface SweBenchmarkReport {
  readonly kind: 'swe'
  readonly model: string
  readonly generatedAt: string
  readonly records: readonly SweTaskRecord[]
  readonly metrics: SweBenchmarkMetrics
}

const EDIT_TOOL_NAMES = new Set([
  'FileEditTool',
  'ApplyPatchTool',
  'file_edit',
  'edit_file',
  'apply_patch',
])

const LOCALIZATION_READ_TOOL_NAMES = new Set([
  'ReadFile',
  'FileOutlineTool',
  'GrepTool',
  'LSP',
  'read_file',
  'file_outline',
  'grep',
  'lsp',
])

export function isSweEditToolName(toolName: string): boolean {
  return EDIT_TOOL_NAMES.has(toolName.trim())
}

export function isSweReadLocalizationToolName(toolName: string): boolean {
  return LOCALIZATION_READ_TOOL_NAMES.has(toolName.trim())
}

export function normalizeSweRelativePath(filePath: string, workspaceDirectory?: string): string {
  const trimmed = filePath.trim()
  if (!trimmed) return ''
  if (workspaceDirectory !== undefined && isAbsolute(trimmed)) {
    const rel = relative(resolve(workspaceDirectory), resolve(trimmed)).replaceAll('\\', '/')
    if (rel && !rel.startsWith('../') && rel !== '..') {
      return posix.normalize(rel)
    }
  }
  const normalized = posix.normalize(trimmed.replaceAll('\\', '/').replace(/^\.\//, ''))
  return normalized === '.' ? '' : normalized
}

export interface ComputedTaskTraceStats {
  readonly turnsToFirstRelevantRead: number | null
  readonly tokensToFirstRelevantRead: number | null
  readonly editAttempts: number
  readonly failedEditAttempts: number
  readonly totalTokens?: number
  readonly costUsd?: number
}

/**
 * Derive per-task localization efficiency (`turnsToFirstRelevantRead`, `tokensToFirstRelevantRead`),
 * edit mismatch counts (`FileEditTool` / `ApplyPatchTool`), and token/cost totals from telemetry.
 */
export function computeTaskTelemetryStats(params: {
  readonly groundTruthTargetFiles: readonly string[]
  readonly workspaceDirectory?: string
  readonly telemetry?: SweRunTelemetry
}): ComputedTaskTraceStats {
  const targetSet = new Set(
    params.groundTruthTargetFiles
      .map(path => normalizeSweRelativePath(path, params.workspaceDirectory))
      .filter(Boolean),
  )

  const telemetry = params.telemetry
  const cumulativeByTurn = new Map<number, number>()
  let runningTurnTokens = 0
  const sortedTurns = [...(telemetry?.turns ?? [])].sort((a, b) => a.turnIndex - b.turnIndex)
  for (const turn of sortedTurns) {
    if (Number.isFinite(turn.tokens) && turn.tokens >= 0) {
      runningTurnTokens += turn.tokens
      cumulativeByTurn.set(turn.turnIndex, runningTurnTokens)
    }
  }

  const allCalls: SweToolInvocationTrace[] = []
  if (telemetry?.toolCalls && telemetry.toolCalls.length > 0) {
    allCalls.push(...telemetry.toolCalls)
  } else {
    for (const turn of sortedTurns) {
      for (const call of turn.tools ?? []) {
        allCalls.push(call)
      }
    }
  }

  let turnsToFirstRelevantRead: number | null = null
  let tokensToFirstRelevantRead: number | null = null
  let editAttempts = 0
  let failedEditAttempts = 0

  for (const call of allCalls) {
    if (isSweEditToolName(call.name)) {
      editAttempts += 1
      if (call.ok === false) {
        failedEditAttempts += 1
      }
    }

    if (
      turnsToFirstRelevantRead === null
      && isSweReadLocalizationToolName(call.name)
      && call.filePath !== undefined
    ) {
      const normalized = normalizeSweRelativePath(call.filePath, params.workspaceDirectory)
      if (normalized && targetSet.has(normalized)) {
        const turnIdx = Number.isInteger(call.turnIndex) && call.turnIndex >= 1 ? call.turnIndex : 1
        turnsToFirstRelevantRead = turnIdx
        if (call.cumulativeTokens !== undefined && Number.isFinite(call.cumulativeTokens) && call.cumulativeTokens >= 0) {
          tokensToFirstRelevantRead = call.cumulativeTokens
        } else {
          const fromTurn = cumulativeByTurn.get(turnIdx)
          tokensToFirstRelevantRead = fromTurn !== undefined ? fromTurn : null
        }
      }
    }
  }

  let totalTokens: number | undefined
  if (telemetry?.totalTokens !== undefined && Number.isFinite(telemetry.totalTokens) && telemetry.totalTokens >= 0) {
    totalTokens = telemetry.totalTokens
  } else if (telemetry?.inputTokens !== undefined || telemetry?.outputTokens !== undefined) {
    const input = telemetry?.inputTokens !== undefined && Number.isFinite(telemetry.inputTokens) ? telemetry.inputTokens : 0
    const output = telemetry?.outputTokens !== undefined && Number.isFinite(telemetry.outputTokens) ? telemetry.outputTokens : 0
    totalTokens = Math.max(0, input + output)
  } else if (sortedTurns.length > 0) {
    totalTokens = runningTurnTokens
  }

  const costUsd = telemetry?.costUsd !== undefined && Number.isFinite(telemetry.costUsd) && telemetry.costUsd >= 0
    ? telemetry.costUsd
    : undefined

  return {
    turnsToFirstRelevantRead,
    tokensToFirstRelevantRead,
    editAttempts,
    failedEditAttempts,
    ...(totalTokens === undefined ? {} : { totalTokens }),
    ...(costUsd === undefined ? {} : { costUsd }),
  }
}

function meanOrNull(values: readonly number[]): number | null {
  if (values.length === 0) return null
  const sum = values.reduce((acc, item) => acc + item, 0)
  return sum / values.length
}

/**
 * Compute the 5 Core SWE KPIs across a suite of `SweTaskRecord` entries:
 * 1. `resolvedRate`
 * 2. `localizationEfficiency` (`turnsToFirstRelevantRead`, `tokensToFirstRelevantRead`)
 * 3. `editMismatchRate`
 * 4. `holdoutRegressionRate`
 * 5. `costAndLatencyPerResolved` (`meanWallMsPerResolved`, `meanTokensPerResolved`, `meanCostUsdPerResolved`)
 */
export function computeSweMetrics(records: readonly SweTaskRecord[]): SweBenchmarkMetrics {
  const totalTasks = records.length
  const resolvedRecords = records.filter(record => record.resolved)
  const resolvedCount = resolvedRecords.length
  const holdoutRegressionCount = records.filter(
    record => record.visibleTestsPassed && !record.holdoutTestsPassed,
  ).length

  const totalEditAttempts = records.reduce((sum, record) => sum + record.editAttempts, 0)
  const failedEditAttempts = records.reduce((sum, record) => sum + record.failedEditAttempts, 0)

  const localizedTurns = records
    .map(record => record.turnsToFirstRelevantRead)
    .filter((value): value is number => value !== null && Number.isFinite(value))
  const localizedTokens = records
    .map(record => record.tokensToFirstRelevantRead)
    .filter((value): value is number => value !== null && Number.isFinite(value))

  const meanTurns = meanOrNull(localizedTurns)
  const meanTokensToRead = meanOrNull(localizedTokens)

  const resolvedWallMs = resolvedRecords.map(record => record.wallMs)
  const resolvedTokens = resolvedRecords
    .map(record => record.totalTokens)
    .filter((value): value is number => value !== undefined && Number.isFinite(value))
  const resolvedCosts = resolvedRecords
    .map(record => record.costUsd)
    .filter((value): value is number => value !== undefined && Number.isFinite(value))

  const allTokens = records
    .map(record => record.totalTokens)
    .filter((value): value is number => value !== undefined && Number.isFinite(value))
  const allCosts = records
    .map(record => record.costUsd)
    .filter((value): value is number => value !== undefined && Number.isFinite(value))

  const totalWallMsAll = records.reduce((sum, record) => sum + record.wallMs, 0)
  const totalTokensAll = allTokens.reduce((sum, value) => sum + value, 0)
  const totalCostAll = allCosts.reduce((sum, value) => sum + value, 0)

  return {
    totalTasks,
    resolvedCount,
    holdoutRegressionCount,
    totalEditAttempts,
    failedEditAttempts,
    resolvedRate: totalTasks > 0 ? resolvedCount / totalTasks : 0,
    localizationEfficiency: {
      localizedTaskCount: localizedTurns.length,
      localizationHitRate: totalTasks > 0 ? localizedTurns.length / totalTasks : 0,
      turnsToFirstRelevantRead: meanTurns,
      tokensToFirstRelevantRead: meanTokensToRead,
      meanTurnsToFirstRelevantRead: meanTurns,
      meanTokensToFirstRelevantRead: meanTokensToRead,
    },
    editMismatchRate: totalEditAttempts > 0 ? failedEditAttempts / totalEditAttempts : 0,
    holdoutRegressionRate: totalTasks > 0 ? holdoutRegressionCount / totalTasks : 0,
    costAndLatencyPerResolved: {
      resolvedCount,
      meanWallMsPerResolved: meanOrNull(resolvedWallMs),
      meanTokensPerResolved: meanOrNull(resolvedTokens),
      meanCostUsdPerResolved: meanOrNull(resolvedCosts),
      amortizedWallMsPerResolved: resolvedCount > 0 ? totalWallMsAll / resolvedCount : null,
      amortizedTokensPerResolved: resolvedCount > 0 && allTokens.length > 0 ? totalTokensAll / resolvedCount : null,
      amortizedCostUsdPerResolved: resolvedCount > 0 && allCosts.length > 0 ? totalCostAll / resolvedCount : null,
    },
  }
}

function pct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`
}

function fmtNum(value: number | null, digits = 1, suffix = ''): string {
  if (value === null || !Number.isFinite(value)) return 'n/a'
  return `${value.toFixed(digits)}${suffix}`
}

/** Render a concise, human-readable scorecard for an SWE benchmark report. */
export function formatSweReport(report: SweBenchmarkReport): string {
  const { metrics } = report
  const rows = report.records.map((record) => {
    const badge = record.resolved
      ? 'RESOLVED'
      : record.holdoutRegression
        ? 'HOLDOUT-FAIL'
        : 'FAILED'
    return `  [${badge.padEnd(12)}] d${record.difficulty} ${record.taskId.padEnd(32)} ${record.category.padEnd(13)} ${(record.wallMs / 1_000).toFixed(2)}s  ${record.detail}`
  })

  return [
    '',
    `  Xerxes SWE Benchmark — model: ${report.model}   tasks: ${metrics.totalTasks}   generated: ${report.generatedAt}`,
    '',
    ...rows,
    '',
    `  ${'-'.repeat(78)}`,
    `  1. resolvedRate (Pass@1):        ${metrics.resolvedCount}/${metrics.totalTasks} (${pct(metrics.resolvedRate)})`,
    `  2. localizationEfficiency:       turns=${fmtNum(metrics.localizationEfficiency.turnsToFirstRelevantRead, 2)}  tokens=${fmtNum(metrics.localizationEfficiency.tokensToFirstRelevantRead, 0)}  hit=${pct(metrics.localizationEfficiency.localizationHitRate)}`,
    `  3. editMismatchRate:             ${metrics.failedEditAttempts}/${metrics.totalEditAttempts} (${pct(metrics.editMismatchRate)})`,
    `  4. holdoutRegressionRate:        ${metrics.holdoutRegressionCount}/${metrics.totalTasks} (${pct(metrics.holdoutRegressionRate)})`,
    `  5. costAndLatencyPerResolved:    wallMs=${fmtNum(metrics.costAndLatencyPerResolved.meanWallMsPerResolved, 1, 'ms')}  tokens=${fmtNum(metrics.costAndLatencyPerResolved.meanTokensPerResolved, 0)}  costUsd=${metrics.costAndLatencyPerResolved.meanCostUsdPerResolved === null ? 'n/a' : `$${metrics.costAndLatencyPerResolved.meanCostUsdPerResolved.toFixed(4)}`}`,
    '',
  ].join('\n')
}
