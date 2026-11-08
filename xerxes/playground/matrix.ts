// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

import {
  BENCHMARK_IDS,
  createBenchmarkAdapter,
  materializeTaskFiles,
  type BenchmarkAdapter,
  type BenchmarkId,
  type BenchmarkTask,
  type GradeResult,
} from './benchmarks/index.js'
import { isRecord, isStringArray, errorMessage, pathExists, safeJoin } from './benchmarks/util.js'
import { NativeEvaluationAgent } from './harness.js'
import { createEvaluationIsolation, type EvaluationIsolation } from './isolation.js'
import {
  buildReport,
  renderMarkdown,
  type ExcludedTask,
  type MatrixPrivacy,
  type MatrixReport,
  type TaskRecord,
} from './matrixReport.js'
import type { EvaluationSessionPort, EvaluationSessionPortFactory, EvaluationTransportContext } from './types.js'

/** Config id pattern: also used as a directory name for review packets. */
const CONFIG_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*$/
const DEFAULT_TURN_TIMEOUT_MS = 900_000

export interface XerxesMatrixConfig {
  readonly id: string
  readonly kind: 'xerxes'
  /** Absolute module path exporting `createEvaluationSessionPort(context)`. */
  readonly transport: string
  readonly privacy: MatrixPrivacy
}

export interface ExternalMatrixConfig {
  readonly id: string
  readonly kind: 'external'
  /** Argv template. `{prompt_file}`, `{workdir}` and `{metrics_file}` are substituted. No shell is involved. */
  readonly argv: readonly string[]
  readonly timeoutMs: number
  readonly privacy: MatrixPrivacy
}

export type MatrixConfig = XerxesMatrixConfig | ExternalMatrixConfig

export interface MatrixBenchmarkEntry {
  readonly adapter: BenchmarkId
  readonly datasetDir?: string
  readonly limit?: number
}

export interface MatrixFile {
  readonly configs: readonly MatrixConfig[]
  readonly benchmarks: readonly MatrixBenchmarkEntry[]
}

/** What a run may report about itself. Anything it does not report stays absent. */
export interface RunMetrics {
  readonly inputTokens?: number
  readonly outputTokens?: number
  readonly costUsd?: number
  /** Text the run sent to an external model or service; used to detect private-data disclosure. */
  readonly outboundTexts?: readonly string[]
}

/** Optional method a transport's session port may implement to report its measurements. */
export interface MetricsReportingPort {
  getRunMetrics(): RunMetrics | Promise<RunMetrics>
}

function nonNegative(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

/** Keep only well-formed measurements; nothing is defaulted or invented. */
export function parseRunMetrics(value: unknown): RunMetrics {
  if (!isRecord(value)) return {}
  const inputTokens = nonNegative(value.inputTokens ?? value.input_tokens)
  const outputTokens = nonNegative(value.outputTokens ?? value.output_tokens)
  const costUsd = nonNegative(value.costUsd ?? value.cost_usd)
  const outbound = value.outboundTexts ?? value.outbound_texts
  return {
    ...(inputTokens === undefined ? {} : { inputTokens }),
    ...(outputTokens === undefined ? {} : { outputTokens }),
    ...(costUsd === undefined ? {} : { costUsd }),
    ...(isStringArray(outbound) ? { outboundTexts: outbound } : {}),
  }
}

/** Validate a decoded matrix file. Relative paths resolve against `baseDir`. */
export function parseMatrixFile(value: unknown, baseDir: string): MatrixFile {
  if (!isRecord(value)) throw new TypeError('matrix file must be a JSON object with "configs" and "benchmarks"')
  if (!Array.isArray(value.configs) || value.configs.length === 0) throw new TypeError('matrix "configs" must be a non-empty array')
  if (!Array.isArray(value.benchmarks) || value.benchmarks.length === 0) throw new TypeError('matrix "benchmarks" must be a non-empty array')

  const seen = new Set<string>()
  const configs: MatrixConfig[] = value.configs.map((entry: unknown, index: number): MatrixConfig => {
    const where = `configs[${index}]`
    if (!isRecord(entry)) throw new TypeError(`${where} must be an object`)
    const id = entry.id
    if (typeof id !== 'string' || !CONFIG_ID.test(id)) throw new TypeError(`${where}.id must match ${CONFIG_ID}`)
    if (seen.has(id)) throw new TypeError(`${where}.id "${id}" is duplicated`)
    seen.add(id)
    const privacy = entry.privacy === undefined ? 'unrestricted' : entry.privacy
    if (privacy !== 'constrained' && privacy !== 'unrestricted') throw new TypeError(`${where}.privacy must be "constrained" or "unrestricted"`)
    if (entry.kind === 'xerxes') {
      if (typeof entry.transport !== 'string' || !entry.transport.trim()) throw new TypeError(`${where}.transport must be a module path`)
      return { id, kind: 'xerxes', privacy, transport: resolve(baseDir, entry.transport) }
    }
    if (entry.kind === 'external') {
      if (!isStringArray(entry.argv) || entry.argv.length === 0 || !entry.argv[0]?.trim()) {
        throw new TypeError(`${where}.argv must be a non-empty array of strings`)
      }
      const timeoutMs = entry.timeoutMs
      if (typeof timeoutMs !== 'number' || !Number.isInteger(timeoutMs) || timeoutMs <= 0) {
        throw new TypeError(`${where}.timeoutMs must be a positive integer`)
      }
      return { argv: entry.argv, id, kind: 'external', privacy, timeoutMs }
    }
    throw new TypeError(`${where}.kind must be "xerxes" or "external"`)
  })

  const benchmarks: MatrixBenchmarkEntry[] = value.benchmarks.map((entry: unknown, index: number): MatrixBenchmarkEntry => {
    const where = `benchmarks[${index}]`
    if (!isRecord(entry)) throw new TypeError(`${where} must be an object`)
    const adapter = BENCHMARK_IDS.find(id => id === entry.adapter)
    if (adapter === undefined) throw new TypeError(`${where}.adapter must be one of ${BENCHMARK_IDS.join(', ')}`)
    const datasetDir = entry.datasetDir
    if (datasetDir !== undefined && (typeof datasetDir !== 'string' || !datasetDir.trim())) throw new TypeError(`${where}.datasetDir must be a path`)
    if (datasetDir === undefined && adapter !== 'synthetic-private') throw new TypeError(`${where}.datasetDir is required for ${adapter}`)
    const limit = entry.limit
    if (limit !== undefined && (typeof limit !== 'number' || !Number.isInteger(limit) || limit <= 0)) {
      throw new TypeError(`${where}.limit must be a positive integer`)
    }
    return {
      adapter,
      ...(typeof datasetDir === 'string' ? { datasetDir: resolve(baseDir, datasetDir) } : {}),
      ...(typeof limit === 'number' ? { limit } : {}),
    }
  })
  return { benchmarks, configs }
}

export async function loadMatrixFile(path: string): Promise<MatrixFile> {
  const absolute = resolve(path)
  let parsed: unknown
  try {
    parsed = JSON.parse(await readFile(absolute, 'utf8'))
  } catch (error) {
    throw new TypeError(`cannot read matrix file ${absolute}: ${errorMessage(error)}`)
  }
  return parseMatrixFile(parsed, resolve(absolute, '..'))
}

/** Injectable boundaries; the defaults are the real ones. */
export interface MatrixDependencies {
  /** Parent directory for the per-run isolated workspaces. */
  readonly runRoot: string
  /** When set, report.json, report.md and review packets are written here. */
  readonly outputDir?: string
  readonly adapterFor?: (id: BenchmarkId) => BenchmarkAdapter
  readonly loadSessionPortFactory?: (modulePath: string) => Promise<EvaluationSessionPortFactory>
  readonly createIsolation?: (options: { rootDirectory: string }) => Promise<EvaluationIsolation>
  readonly clock?: () => number
  readonly now?: () => Date
  readonly turnTimeoutMs?: number
  /** Transient-failure retries for xerxes configs; default is the harness default. */
  readonly retries?: number
  readonly onProgress?: (line: string) => void
}

interface RunOutcome {
  readonly finalText: string
  readonly metrics: RunMetrics
  readonly error?: string
}

export async function loadSessionPortFactory(modulePath: string): Promise<EvaluationSessionPortFactory> {
  const absolute = resolve(modulePath)
  const loaded: unknown = await import(pathToFileURL(absolute).href)
  if (!isRecord(loaded) || typeof loaded.createEvaluationSessionPort !== 'function') {
    throw new TypeError(`${absolute} must export createEvaluationSessionPort(context)`)
  }
  return loaded.createEvaluationSessionPort as EvaluationSessionPortFactory
}

function assertSessionPort(value: unknown): asserts value is EvaluationSessionPort {
  if (
    !isRecord(value)
    || typeof value.approve !== 'function'
    || typeof value.close !== 'function'
    || typeof value.reset !== 'function'
    || typeof value.start !== 'function'
    || typeof value.submit !== 'function'
  ) {
    throw new TypeError('createEvaluationSessionPort(context) must return an EvaluationSessionPort')
  }
}

async function readPortMetrics(port: EvaluationSessionPort): Promise<RunMetrics> {
  const candidate: unknown = port
  if (isRecord(candidate) && typeof candidate.getRunMetrics === 'function') {
    return parseRunMetrics(await (candidate.getRunMetrics as () => unknown)())
  }
  return {}
}

async function runXerxesConfig(
  config: XerxesMatrixConfig,
  task: BenchmarkTask,
  isolation: EvaluationIsolation,
  deps: MatrixDependencies,
): Promise<RunOutcome> {
  const context: EvaluationTransportContext = {
    homeDirectory: isolation.homeDirectory,
    runDirectory: isolation.runDirectory,
    workspaceDirectory: isolation.workspaceDirectory,
  }
  const factory = await (deps.loadSessionPortFactory ?? loadSessionPortFactory)(config.transport)
  const port = await factory(context)
  assertSessionPort(port)
  const agent = new NativeEvaluationAgent({
    start: { homeDirectory: context.homeDirectory, permissionMode: 'accept-all', workspaceDirectory: context.workspaceDirectory },
    transport: port,
  })
  try {
    await agent.start()
    const result = await agent.turn(task.prompt, {
      timeoutMs: deps.turnTimeoutMs ?? DEFAULT_TURN_TIMEOUT_MS,
      ...(deps.retries === undefined ? {} : { retries: deps.retries }),
    })
    const metrics = await readPortMetrics(port)
    return { finalText: result.text, metrics, ...(result.error === undefined ? {} : { error: result.error }) }
  } finally {
    await agent.close()
  }
}

async function runExternalConfig(
  config: ExternalMatrixConfig,
  task: BenchmarkTask,
  isolation: EvaluationIsolation,
): Promise<RunOutcome> {
  const promptFile = join(isolation.runDirectory, 'prompt.txt')
  const metricsFile = join(isolation.runDirectory, 'metrics.json')
  await writeFile(promptFile, task.prompt, 'utf8')
  const argv = config.argv.map(argument => argument
    .replaceAll('{prompt_file}', promptFile)
    .replaceAll('{workdir}', isolation.workspaceDirectory)
    .replaceAll('{metrics_file}', metricsFile))

  const child = Bun.spawn(argv, {
    cwd: isolation.workspaceDirectory,
    stderr: 'pipe',
    stdin: 'ignore',
    stdout: 'pipe',
  })
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    child.kill()
  }, config.timeoutMs)
  let stdout: string
  let stderr: string
  let exitCode: number
  try {
    ;[stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])
  } finally {
    clearTimeout(timer)
  }

  let metrics: RunMetrics = {}
  if (await pathExists(metricsFile)) {
    try {
      metrics = parseRunMetrics(JSON.parse(await readFile(metricsFile, 'utf8')))
    } catch {
      metrics = {}
    }
  }
  if (timedOut) return { error: `timed out after ${config.timeoutMs} ms`, finalText: stdout.trim(), metrics }
  if (exitCode !== 0) return { error: `exited with code ${exitCode}: ${stderr.trim().slice(0, 200)}`, finalText: stdout.trim(), metrics }
  return { finalText: stdout.trim(), metrics }
}

function metricFields(metrics: RunMetrics): Pick<TaskRecord, 'inputTokens' | 'outputTokens' | 'costUsd'> {
  return {
    ...(metrics.inputTokens === undefined ? {} : { inputTokens: metrics.inputTokens }),
    ...(metrics.outputTokens === undefined ? {} : { outputTokens: metrics.outputTokens }),
    ...(metrics.costUsd === undefined ? {} : { costUsd: metrics.costUsd }),
  }
}

interface RunResult {
  readonly record: TaskRecord
  readonly packet?: string
}

async function runOne(
  config: MatrixConfig,
  task: BenchmarkTask,
  adapter: BenchmarkAdapter,
  deps: MatrixDependencies,
): Promise<RunResult> {
  const clock = deps.clock ?? Date.now
  const started = clock()
  const base = { benchmark: task.benchmark, configId: config.id, taskId: task.id }
  const failure = (message: string, metrics: RunMetrics = {}): RunResult => ({
    record: {
      ...base,
      accepted: false,
      details: message,
      disclosureFailure: false,
      error: message,
      score: 0,
      status: 'error',
      wallMs: Math.max(0, clock() - started),
      ...metricFields(metrics),
    },
  })

  let isolation: EvaluationIsolation | undefined
  try {
    isolation = await (deps.createIsolation ?? createEvaluationIsolation)({ rootDirectory: deps.runRoot })
    await materializeTaskFiles(task, isolation.workspaceDirectory)
    const outcome = config.kind === 'xerxes'
      ? await runXerxesConfig(config, task, isolation, deps)
      : await runExternalConfig(config, task, isolation)
    const wallMs = Math.max(0, clock() - started)
    if (outcome.error !== undefined) return failure(outcome.error, outcome.metrics)

    const graded: GradeResult = await adapter.grade(task, {
      finalText: outcome.finalText,
      workdir: isolation.workspaceDirectory,
      ...(outcome.metrics.outboundTexts === undefined ? {} : { outboundTexts: outcome.metrics.outboundTexts }),
    })
    return {
      record: {
        ...base,
        accepted: graded.accepted && !graded.disclosureFailure,
        details: graded.details,
        disclosureFailure: graded.disclosureFailure,
        score: graded.score,
        status: graded.status,
        wallMs,
        ...metricFields(outcome.metrics),
      },
      ...(graded.reviewPacket === undefined ? {} : { packet: graded.reviewPacket }),
    }
  } catch (error) {
    return failure(errorMessage(error))
  } finally {
    if (isolation !== undefined) await isolation.cleanup()
  }
}

/** Run every task of every benchmark through every config and build the report. */
export async function runMatrix(matrix: MatrixFile, deps: MatrixDependencies): Promise<MatrixReport> {
  const adapterFor = deps.adapterFor ?? createBenchmarkAdapter
  const records: TaskRecord[] = []
  const excluded: ExcludedTask[] = []
  const packets: { readonly configId: string; readonly taskId: string; readonly text: string }[] = []

  for (const entry of matrix.benchmarks) {
    const adapter = adapterFor(entry.adapter)
    const loaded = await adapter.load(entry.datasetDir ?? '', entry.limit === undefined ? {} : { limit: entry.limit })
    for (const item of loaded.excluded) excluded.push({ benchmark: entry.adapter, id: item.id, reason: item.reason })
    for (const task of loaded.tasks) {
      for (const config of matrix.configs) {
        deps.onProgress?.(`${config.id} :: ${task.id}`)
        const result = await runOne(config, task, adapter, deps)
        records.push(result.record)
        if (result.packet !== undefined) packets.push({ configId: config.id, taskId: task.id, text: result.packet })
      }
    }
  }

  const report = buildReport({
    configs: matrix.configs.map(config => ({ id: config.id, privacy: config.privacy })),
    excluded,
    generatedAt: (deps.now ?? (() => new Date()))().toISOString(),
    records,
  })
  if (deps.outputDir !== undefined) await writeReportFiles(report, packets, deps.outputDir)
  return report
}

export async function writeReportFiles(
  report: MatrixReport,
  packets: readonly { readonly configId: string; readonly taskId: string; readonly text: string }[],
  outputDir: string,
): Promise<void> {
  await mkdir(outputDir, { recursive: true })
  await writeFile(join(outputDir, 'report.json'), `${JSON.stringify(report, null, 2)}\n`, 'utf8')
  await writeFile(join(outputDir, 'report.md'), renderMarkdown(report), 'utf8')
  for (const packet of packets) {
    const dir = safeJoin(outputDir, join('review', packet.configId))
    await mkdir(dir, { recursive: true })
    await writeFile(safeJoin(dir, `${packet.taskId}.md`), `${packet.text}\n`, 'utf8')
  }
}
