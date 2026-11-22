// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, realpath, rm, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path'
import { pathToFileURL } from 'node:url'

import { NativeEvaluationAgent } from './harness.js'
import { createEvaluationIsolation, type EvaluationIsolation } from './isolation.js'
import { parseRunMetrics, type RunMetrics } from './matrix.js'
import {
  computeSweMetrics,
  computeTaskTelemetryStats,
  type SweBenchmarkReport,
  type SweRunTelemetry,
  type SweTaskRecord,
  type SweToolInvocationTrace,
  type SweTurnTelemetry,
} from './sweMetrics.js'
import type {
  EvaluationSessionPort,
  EvaluationSessionPortFactory,
  EvaluationTaskFile,
  EvaluationTransportContext,
} from './types.js'

const DEFAULT_TURN_TIMEOUT_MS = 300_000
const DEFAULT_TEST_TIMEOUT_MS = 30_000

/** One realistic multi-file TypeScript SWE bugfix task with visible and hidden holdout suites. */
export interface NativeSweTask {
  readonly id: string
  readonly category: string
  readonly difficulty: number
  readonly prompt: string
  readonly whyHard: string
  readonly groundTruthTargetFiles: readonly string[]
  readonly sourceFiles: readonly EvaluationTaskFile[]
  readonly baselineTestFiles: readonly EvaluationTaskFile[]
  readonly visibleReproductionTestFiles: readonly EvaluationTaskFile[]
  readonly holdoutTestFiles: readonly EvaluationTaskFile[]
  /** Symptom-patching files that pass visible tests while failing hidden holdout tests. */
  readonly symptomCheatFiles: readonly EvaluationTaskFile[]
  /** Genuine root-cause fix files that satisfy baseline, visible reproduction, and hidden holdout suites. */
  readonly referenceSolutionFiles: readonly EvaluationTaskFile[]
}

/** Context passed to an offline/scripted `SweTaskSolver` during `runSweBenchmark`. */
export interface SweSolverContext {
  readonly task: NativeSweTask
  readonly workspaceDirectory: string
  readonly homeDirectory: string
  readonly runDirectory: string
  readonly signal: AbortSignal
}

/** Outcome returned by a `SweTaskSolver`. */
export interface SweSolverOutcome {
  readonly finalText?: string
  readonly telemetry?: SweRunTelemetry
  readonly error?: string
}

export type SweTaskSolver = (context: SweSolverContext) => Promise<SweSolverOutcome | void>

/** Execution and isolation options for `runSweBenchmark`. */
export interface SweBenchmarkOptions {
  readonly runRoot: string
  readonly tasks?: readonly NativeSweTask[]
  readonly keyword?: string
  readonly model?: string
  /** Direct task solver (ideal for deterministic offline tests or custom tournament runners). */
  readonly solver?: SweTaskSolver
  /** Explicit host session-port factory for evaluating live `EvaluationSessionPort` transports. */
  readonly createSessionPort?: EvaluationSessionPortFactory
  /** Module path exporting `createEvaluationSessionPort(context)`. */
  readonly transportModule?: string
  readonly profileSourceDirectory?: string
  readonly createIsolation?: (options: Parameters<typeof createEvaluationIsolation>[0]) => Promise<EvaluationIsolation>
  readonly clock?: () => number
  readonly now?: () => Date
  readonly turnTimeoutMs?: number
  readonly testTimeoutMs?: number
  readonly retries?: number
  readonly executable?: string
  readonly onTaskComplete?: (record: SweTaskRecord) => void | Promise<void>
}

export interface SweWorkspaceGradeResult {
  readonly baselinePassed: boolean
  readonly visiblePassed: boolean
  readonly holdoutPassed: boolean
  readonly resolved: boolean
  readonly holdoutRegression: boolean
  readonly baselineRegression: boolean
  readonly detail: string
}

const TASK_PACKAGE_JSON = `${JSON.stringify(
  {
    name: 'xerxes-swe-task-fixture',
    private: true,
    type: 'module',
  },
  null,
  2,
)}\n`

function safeWorkspacePath(workspaceDirectory: string, relativeFilePath: string): string {
  const trimmed = relativeFilePath.trim()
  if (!trimmed) throw new Error('task file path must not be empty')
  const candidate = resolve(workspaceDirectory, trimmed)
  const rel = relative(workspaceDirectory, candidate)
  if (!rel || rel === '..' || rel.startsWith(`..${sep}`) || rel.startsWith('../') || isAbsolute(rel)) {
    throw new Error(`task file path escapes workspace: ${relativeFilePath}`)
  }
  return candidate
}

export async function writeSweWorkspaceFiles(
  workspaceDirectory: string,
  files: readonly EvaluationTaskFile[],
): Promise<void> {
  const root = resolve(workspaceDirectory)
  await mkdir(root, { recursive: true })
  await Promise.all(
    files.map(async (file) => {
      const target = safeWorkspacePath(root, file.path)
      await mkdir(dirname(target), { recursive: true })
      await writeFile(target, file.content, 'utf8')
    }),
  )
}

/**
 * Materialize ONLY the visible workspace files (`package.json`, `sourceFiles`,
 * `baselineTestFiles`, and `visibleReproductionTestFiles`) before the agent runs.
 * Hidden `holdoutTestFiles` are intentionally withheld until post-turn grading.
 */
export async function provisionSweTaskWorkspace(
  workspaceDirectory: string,
  task: NativeSweTask,
): Promise<void> {
  const root = resolve(workspaceDirectory)
  await rm(root, { recursive: true, force: true })
  await mkdir(root, { recursive: true })
  await writeSweWorkspaceFiles(root, [
    { path: 'package.json', content: TASK_PACKAGE_JSON },
    ...task.sourceFiles,
    ...task.baselineTestFiles,
    ...task.visibleReproductionTestFiles,
  ])
}

function sanitizedTestEnv(): Record<string, string | undefined> {
  const env: Record<string, string | undefined> = {
    ...Bun.env,
    CI: '1',
    NO_COLOR: '1',
  }
  return env
}

async function runBunTests(params: {
  readonly workspaceDirectory: string
  readonly testFiles: readonly EvaluationTaskFile[]
  readonly executable: string
  readonly timeoutMs: number
}): Promise<{ readonly passed: boolean; readonly summary: string }> {
  if (params.testFiles.length === 0) {
    return { passed: true, summary: 'no test files' }
  }
  const realWorkspace = await realpath(resolve(params.workspaceDirectory))
  const relativePaths = params.testFiles.map(file => `./${file.path.replaceAll('\\', '/').replace(/^\.\//, '')}`)
  const child = Bun.spawn([params.executable, 'test', ...relativePaths], {
    cwd: realWorkspace,
    env: sanitizedTestEnv(),
    stdin: 'ignore',
    stdout: 'pipe',
    stderr: 'pipe',
  })

  const stdoutPromise = new Response(child.stdout).text().catch(() => '')
  const stderrPromise = new Response(child.stderr).text().catch(() => '')
  let timer: ReturnType<typeof setTimeout> | undefined
  let timedOut = false
  const exitCode = await Promise.race([
    child.exited,
    new Promise<number>((resolveExit) => {
      timer = setTimeout(() => {
        timedOut = true
        child.kill('SIGKILL')
        resolveExit(124)
      }, params.timeoutMs)
    }),
  ]).finally(() => {
    if (timer !== undefined) clearTimeout(timer)
  })

  const [stdout, stderr] = await Promise.all([stdoutPromise, stderrPromise])
  if (timedOut) {
    return { passed: false, summary: `timed out after ${params.timeoutMs}ms` }
  }
  if (exitCode === 0) {
    return { passed: true, summary: 'ok' }
  }
  const combined = `${stderr}\n${stdout}`
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0 && !line.includes('PermissionDenied'))
  const tail = combined.slice(-3).join(' | ').slice(0, 180)
  return {
    passed: false,
    summary: tail || `exit ${exitCode}`,
  }
}

/**
 * Restore canonical baseline + visible test suites (preventing test tampering),
 * inject hidden holdout regression tests, and execute all three suites in Bun.
 */
export async function gradeSweTaskWorkspace(params: {
  readonly workspaceDirectory: string
  readonly task: NativeSweTask
  readonly executable?: string
  readonly timeoutMs?: number
}): Promise<SweWorkspaceGradeResult> {
  const executable = params.executable ?? process.execPath
  const timeoutMs = params.timeoutMs ?? DEFAULT_TEST_TIMEOUT_MS
  const root = resolve(params.workspaceDirectory)

  // Restore canonical baseline + visible reproduction tests and inject hidden holdout tests.
  await writeSweWorkspaceFiles(root, [
    ...params.task.baselineTestFiles,
    ...params.task.visibleReproductionTestFiles,
    ...params.task.holdoutTestFiles,
  ])

  const [baseline, visible, holdout] = await Promise.all([
    runBunTests({ workspaceDirectory: root, testFiles: params.task.baselineTestFiles, executable, timeoutMs }),
    runBunTests({ workspaceDirectory: root, testFiles: params.task.visibleReproductionTestFiles, executable, timeoutMs }),
    runBunTests({ workspaceDirectory: root, testFiles: params.task.holdoutTestFiles, executable, timeoutMs }),
  ])

  const resolved = baseline.passed && visible.passed && holdout.passed
  const holdoutRegression = visible.passed && !holdout.passed
  const baselineRegression = !baseline.passed

  let detail: string
  if (resolved) {
    detail = 'baseline + visible + hidden holdout suites passed'
  } else if (holdoutRegression && baseline.passed) {
    detail = `visible passed, hidden holdout failed: ${holdout.summary}`
  } else if (!baseline.passed) {
    detail = `pre-existing baseline regressed: ${baseline.summary}`
  } else {
    detail = `visible reproduction still failing: ${visible.summary}`
  }

  return {
    baselinePassed: baseline.passed,
    visiblePassed: visible.passed,
    holdoutPassed: holdout.passed,
    resolved,
    holdoutRegression,
    baselineRegression,
    detail,
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

async function loadSweSessionPortFactory(modulePath: string): Promise<EvaluationSessionPortFactory> {
  const absolutePath = resolve(process.cwd(), modulePath.trim())
  const loaded: unknown = await import(pathToFileURL(absolutePath).href)
  if (!isRecord(loaded) || typeof loaded.createEvaluationSessionPort !== 'function') {
    throw new TypeError(`${absolutePath} must export createEvaluationSessionPort(context)`)
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

async function extractSessionPortTelemetry(
  port: EvaluationSessionPort,
  fallbackTools: readonly string[],
  fallbackContextTokens: number,
  fallbackLatencyMs: number,
): Promise<SweRunTelemetry> {
  const candidate: unknown = port
  let runMetrics: RunMetrics = {}
  if (isRecord(candidate) && typeof candidate.getRunMetrics === 'function') {
    runMetrics = parseRunMetrics(await (candidate.getRunMetrics as () => unknown)())
  }
  if (isRecord(candidate) && typeof candidate.getSweTelemetry === 'function') {
    const raw: unknown = await (candidate.getSweTelemetry as () => unknown)()
    if (isRecord(raw)) {
      const custom = raw as SweRunTelemetry
      return {
        ...custom,
        ...(custom.inputTokens === undefined && runMetrics.inputTokens !== undefined
          ? { inputTokens: runMetrics.inputTokens }
          : {}),
        ...(custom.outputTokens === undefined && runMetrics.outputTokens !== undefined
          ? { outputTokens: runMetrics.outputTokens }
          : {}),
        ...(custom.costUsd === undefined && runMetrics.costUsd !== undefined
          ? { costUsd: runMetrics.costUsd }
          : {}),
      }
    }
  }

  const toolCalls: SweToolInvocationTrace[] = fallbackTools.map(name => ({
    name,
    turnIndex: 1,
    ok: true,
  }))
  const turns: SweTurnTelemetry[] = [
    {
      turnIndex: 1,
      tokens: fallbackContextTokens,
      latencyMs: fallbackLatencyMs,
      tools: toolCalls,
    },
  ]

  return {
    turns,
    toolCalls,
    ...(runMetrics.inputTokens === undefined ? {} : { inputTokens: runMetrics.inputTokens }),
    ...(runMetrics.outputTokens === undefined ? {} : { outputTokens: runMetrics.outputTokens }),
    ...(runMetrics.costUsd === undefined ? {} : { costUsd: runMetrics.costUsd }),
  }
}

/**
 * Run the multi-file SWE evaluation suite across isolated workspaces, injecting hidden
 * holdout regression tests only after each task run finishes, and compute the 5 Core SWE KPIs.
 */
export async function runSweBenchmark(options: SweBenchmarkOptions): Promise<SweBenchmarkReport> {
  const runRoot = resolve(options.runRoot.trim())
  if (!options.runRoot.trim()) {
    throw new Error('runRoot must not be empty')
  }
  if (
    options.solver === undefined
    && options.createSessionPort === undefined
    && options.transportModule === undefined
  ) {
    throw new Error('runSweBenchmark requires solver, createSessionPort, or transportModule')
  }

  const allTasks = options.tasks ?? createNativeSweTasks()
  const selectedTasks = options.keyword
    ? allTasks.filter(task => task.id.includes(options.keyword!) || task.category.includes(options.keyword!))
    : allTasks

  const createIsolation = options.createIsolation ?? createEvaluationIsolation
  const clock = options.clock ?? Date.now
  const now = options.now ?? (() => new Date())
  const turnTimeoutMs = options.turnTimeoutMs ?? DEFAULT_TURN_TIMEOUT_MS
  const testTimeoutMs = options.testTimeoutMs ?? DEFAULT_TEST_TIMEOUT_MS
  const executable = options.executable ?? process.execPath

  const sessionFactory = options.solver === undefined
    ? (options.createSessionPort ?? await loadSweSessionPortFactory(options.transportModule!))
    : undefined

  let detectedModel = options.model ?? (options.solver ? 'custom-solver' : '?')
  const records: SweTaskRecord[] = []

  for (const task of selectedTasks) {
    const startedAt = clock()
    let isolation: EvaluationIsolation | undefined
    try {
      isolation = await createIsolation({
        rootDirectory: runRoot,
        ...(options.profileSourceDirectory === undefined
          ? {}
          : { profileSourceDirectory: options.profileSourceDirectory }),
      })
      await provisionSweTaskWorkspace(isolation.workspaceDirectory, task)

      let telemetry: SweRunTelemetry | undefined
      let runError: string | undefined

      if (options.solver !== undefined) {
        const controller = new AbortController()
        const timer = setTimeout(() => controller.abort(new Error('timeout')), turnTimeoutMs)
        try {
          const outcome = await options.solver({
            task,
            workspaceDirectory: isolation.workspaceDirectory,
            homeDirectory: isolation.homeDirectory,
            runDirectory: isolation.runDirectory,
            signal: controller.signal,
          })
          if (outcome) {
            telemetry = outcome.telemetry
            runError = outcome.error
          }
        } finally {
          clearTimeout(timer)
        }
      } else if (sessionFactory !== undefined) {
        const transportContext: EvaluationTransportContext = {
          homeDirectory: isolation.homeDirectory,
          runDirectory: isolation.runDirectory,
          workspaceDirectory: isolation.workspaceDirectory,
        }
        const port = await sessionFactory(transportContext)
        assertSessionPort(port)
        const agent = new NativeEvaluationAgent({
          start: {
            homeDirectory: transportContext.homeDirectory,
            permissionMode: 'accept-all',
            workspaceDirectory: transportContext.workspaceDirectory,
          },
          transport: port,
        })
        try {
          await agent.start()
          if (agent.model && agent.model !== '?') {
            detectedModel = agent.model
          }
          const turnResult = await agent.turn(task.prompt, {
            timeoutMs: turnTimeoutMs,
            ...(options.retries === undefined ? {} : { retries: options.retries }),
          })
          runError = turnResult.error
          telemetry = await extractSessionPortTelemetry(
            port,
            turnResult.tools,
            turnResult.contextTokens,
            turnResult.latencyMs,
          )
        } finally {
          await agent.close()
        }
      }

      const wallMs = Math.max(0, clock() - startedAt)
      const traceStats = computeTaskTelemetryStats({
        groundTruthTargetFiles: task.groundTruthTargetFiles,
        workspaceDirectory: isolation.workspaceDirectory,
        ...(telemetry === undefined ? {} : { telemetry }),
      })

      const grade = runError === undefined
        ? await gradeSweTaskWorkspace({
            workspaceDirectory: isolation.workspaceDirectory,
            task,
            executable,
            timeoutMs: testTimeoutMs,
          })
        : {
            baselinePassed: false,
            visiblePassed: false,
            holdoutPassed: false,
            resolved: false,
            holdoutRegression: false,
            baselineRegression: false,
            detail: `ERROR: ${runError}`,
          }

      const record: SweTaskRecord = {
        taskId: task.id,
        category: task.category,
        difficulty: task.difficulty,
        groundTruthTargetFiles: [...task.groundTruthTargetFiles],
        baselineTestsPassed: grade.baselinePassed,
        visibleTestsPassed: grade.visiblePassed,
        holdoutTestsPassed: grade.holdoutPassed,
        resolved: runError === undefined && grade.resolved,
        holdoutRegression: runError === undefined && grade.holdoutRegression,
        baselineRegression: runError === undefined && grade.baselineRegression,
        turnsToFirstRelevantRead: traceStats.turnsToFirstRelevantRead,
        tokensToFirstRelevantRead: traceStats.tokensToFirstRelevantRead,
        editAttempts: traceStats.editAttempts,
        failedEditAttempts: traceStats.failedEditAttempts,
        wallMs,
        ...(traceStats.totalTokens === undefined ? {} : { totalTokens: traceStats.totalTokens }),
        ...(traceStats.costUsd === undefined ? {} : { costUsd: traceStats.costUsd }),
        detail: grade.detail,
        ...(runError === undefined ? {} : { error: runError }),
      }
      records.push(record)
      await options.onTaskComplete?.(record)
    } finally {
      if (isolation !== undefined) {
        await isolation.cleanup()
      }
    }
  }

  return {
    kind: 'swe',
    model: detectedModel,
    generatedAt: now().toISOString(),
    records,
    metrics: computeSweMetrics(records),
  }
}

function buildStreamCancelRaceTask(): NativeSweTask {
  return {
    id: 'async-stream-cancel-race',
    category: 'concurrency',
    difficulty: 4,
    groundTruthTargetFiles: ['src/stream/abortLatch.ts', 'src/stream/pipeline.ts'],
    whyHard: 'Requires coordinating pre-aborted AbortSignal state in abortLatch.ts AND try/finally iterator + resource cleanup in pipeline.ts across both mid-stream aborts and early consumer breaks.',
    prompt: [
      'Fix the async stream cancellation resource leak across `src/stream/`.',
      '`bun test` currently fails `test/repro.test.ts`: aborting an in-flight stream pipeline leaves',
      'acquired resources open and fails to close the source async iterator.',
      'Ensure pre-existing baseline tests keep passing and handle all cancellation / early-termination paths cleanly.',
    ].join(' '),
    sourceFiles: [
      {
        path: 'src/stream/abortLatch.ts',
        content: [
          'export interface AbortLatch {',
          '  readonly aborted: boolean',
          '  onCleanup(fn: () => Promise<void> | void): void',
          '  runCleanups(): Promise<void>',
          '}',
          '',
          'export function createAbortLatch(signal?: AbortSignal): AbortLatch {',
          '  let isAborted = false',
          '  let cleaned = false',
          '  const cleanups: Array<() => Promise<void> | void> = []',
          '',
          "  signal?.addEventListener('abort', () => {",
          '    isAborted = true',
          '  }, { once: true })',
          '',
          '  return {',
          '    get aborted(): boolean {',
          '      return isAborted',
          '    },',
          '    onCleanup(fn: () => Promise<void> | void): void {',
          '      cleanups.push(fn)',
          '    },',
          '    async runCleanups(): Promise<void> {',
          '      if (cleaned) return',
          '      cleaned = true',
          '      for (let index = cleanups.length - 1; index >= 0; index -= 1) {',
          '        await cleanups[index]?.()',
          '      }',
          '    },',
          '  }',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/stream/transformStage.ts',
        content: [
          "import type { AbortLatch } from './abortLatch.js'",
          '',
          'export type TransformStageOutcome<TOut> =',
          '  | { readonly cancelled: true }',
          '  | { readonly cancelled: false; readonly value: TOut }',
          '',
          'export async function runTransformStage<TIn, TOut>(',
          '  item: TIn,',
          '  index: number,',
          '  transform: (item: TIn, index: number) => Promise<TOut> | TOut,',
          '  latch: AbortLatch,',
          '): Promise<TransformStageOutcome<TOut>> {',
          '  if (latch.aborted) return { cancelled: true }',
          '  const value = await transform(item, index)',
          '  if (latch.aborted) return { cancelled: true }',
          '  return { cancelled: false, value }',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/stream/pipeline.ts',
        content: [
          "import { createAbortLatch } from './abortLatch.js'",
          "import { runTransformStage } from './transformStage.js'",
          '',
          'export interface StreamPipelineOptions<TIn, TOut> {',
          '  readonly source: AsyncIterable<TIn>',
          '  readonly transform: (item: TIn, index: number) => Promise<TOut> | TOut',
          '  readonly onAcquireResource?: () => (() => Promise<void> | void)',
          '  readonly signal?: AbortSignal',
          '}',
          '',
          'export async function* streamPipeline<TIn, TOut>(',
          '  options: StreamPipelineOptions<TIn, TOut>,',
          '): AsyncGenerator<TOut> {',
          '  const latch = createAbortLatch(options.signal)',
          '  if (options.onAcquireResource) {',
          '    latch.onCleanup(options.onAcquireResource())',
          '  }',
          '  const iterator = options.source[Symbol.asyncIterator]()',
          '  let index = 0',
          '  while (true) {',
          '    if (latch.aborted) {',
          '      return',
          '    }',
          '    const next = await iterator.next()',
          '    if (next.done) break',
          '    const outcome = await runTransformStage(next.value, index, options.transform, latch)',
          '    index += 1',
          '    if (outcome.cancelled) {',
          '      return',
          '    }',
          '    yield outcome.value',
          '  }',
          '  await latch.runCleanups()',
          '}',
          '',
          'export async function collectPipeline<TIn, TOut>(',
          '  options: StreamPipelineOptions<TIn, TOut>,',
          '): Promise<TOut[]> {',
          '  const output: TOut[] = []',
          '  for await (const item of streamPipeline(options)) {',
          '    output.push(item)',
          '  }',
          '  return output',
          '}',
          '',
        ].join('\n'),
      },
    ],
    baselineTestFiles: [
      {
        path: 'test/baseline.test.ts',
        content: [
          "import { expect, test } from 'bun:test'",
          "import { collectPipeline } from '../src/stream/pipeline.js'",
          '',
          "test('transforms full stream and releases acquired resources on completion', async () => {",
          '  let activeHandles = 0',
          '  async function* words() {',
          "    yield 'one'",
          "    yield 'two'",
          "    yield 'three'",
          '  }',
          '  const out = await collectPipeline({',
          '    source: words(),',
          '    onAcquireResource: () => {',
          '      activeHandles += 1',
          '      return () => {',
          '        activeHandles -= 1',
          '      }',
          '    },',
          '    transform: item => item.toUpperCase(),',
          '  })',
          "  expect(out).toEqual(['ONE', 'TWO', 'THREE'])",
          '  expect(activeHandles).toBe(0)',
          '})',
          '',
        ].join('\n'),
      },
    ],
    visibleReproductionTestFiles: [
      {
        path: 'test/repro.test.ts',
        content: [
          "import { expect, test } from 'bun:test'",
          "import { collectPipeline } from '../src/stream/pipeline.js'",
          '',
          "test('mid-stream abort closes source iterator and releases resources', async () => {",
          '  const controller = new AbortController()',
          '  let activeHandles = 0',
          '  let sourceClosed = false',
          '',
          '  async function* items() {',
          '    try {',
          "      yield 'alpha'",
          "      yield 'beta'",
          "      yield 'gamma'",
          '    } finally {',
          '      sourceClosed = true',
          '    }',
          '  }',
          '',
          '  const out = await collectPipeline({',
          '    source: items(),',
          '    signal: controller.signal,',
          '    onAcquireResource: () => {',
          '      activeHandles += 1',
          '      return () => {',
          '        activeHandles -= 1',
          '      }',
          '    },',
          '    transform: async (value) => {',
          "      if (value === 'beta') {",
          '        controller.abort()',
          '      }',
          '      return value.toUpperCase()',
          '    },',
          '  })',
          '',
          "  expect(out).toEqual(['ALPHA'])",
          '  expect(sourceClosed).toBe(true)',
          '  expect(activeHandles).toBe(0)',
          '})',
          '',
        ].join('\n'),
      },
    ],
    holdoutTestFiles: [
      {
        path: 'test/__holdout__/stream.holdout.test.ts',
        content: [
          "import { expect, test } from 'bun:test'",
          "import { collectPipeline, streamPipeline } from '../../src/stream/pipeline.js'",
          '',
          "test('already-aborted signal never pulls items and still releases resources cleanly', async () => {",
          '  const preAborted = AbortSignal.abort()',
          '  let pulled = 0',
          '  let activeHandles = 0',
          '  async function* source() {',
          '    pulled += 1',
          '    yield 1',
          '  }',
          '  const out = await collectPipeline({',
          '    source: source(),',
          '    signal: preAborted,',
          '    onAcquireResource: () => {',
          '      activeHandles += 1',
          '      return () => {',
          '        activeHandles -= 1',
          '      }',
          '    },',
          '    transform: n => n * 2,',
          '  })',
          '  expect(out).toEqual([])',
          '  expect(pulled).toBe(0)',
          '  expect(activeHandles).toBe(0)',
          '})',
          '',
          "test('early consumer break closes source generator and runs cleanup hooks once', async () => {",
          '  let sourceClosed = false',
          '  let cleanupCalls = 0',
          '  async function* numbers() {',
          '    try {',
          '      yield 10',
          '      yield 20',
          '      yield 30',
          '    } finally {',
          '      sourceClosed = true',
          '    }',
          '  }',
          '  const seen: number[] = []',
          '  for await (const num of streamPipeline({',
          '    source: numbers(),',
          '    onAcquireResource: () => () => {',
          '      cleanupCalls += 1',
          '    },',
          '    transform: n => n,',
          '  })) {',
          '    seen.push(num)',
          '    if (num === 10) break',
          '  }',
          '  expect(seen).toEqual([10])',
          '  expect(sourceClosed).toBe(true)',
          '  expect(cleanupCalls).toBe(1)',
          '})',
          '',
        ].join('\n'),
      },
    ],
    symptomCheatFiles: [
      {
        path: 'src/stream/pipeline.ts',
        content: [
          "import { createAbortLatch } from './abortLatch.js'",
          "import { runTransformStage } from './transformStage.js'",
          '',
          'export interface StreamPipelineOptions<TIn, TOut> {',
          '  readonly source: AsyncIterable<TIn>',
          '  readonly transform: (item: TIn, index: number) => Promise<TOut> | TOut',
          '  readonly onAcquireResource?: () => (() => Promise<void> | void)',
          '  readonly signal?: AbortSignal',
          '}',
          '',
          'export async function* streamPipeline<TIn, TOut>(',
          '  options: StreamPipelineOptions<TIn, TOut>,',
          '): AsyncGenerator<TOut> {',
          '  const latch = createAbortLatch(options.signal)',
          '  if (options.onAcquireResource) {',
          '    latch.onCleanup(options.onAcquireResource())',
          '  }',
          '  const iterator = options.source[Symbol.asyncIterator]()',
          '  let index = 0',
          '  while (true) {',
          '    const next = await iterator.next()',
          '    if (next.done) break',
          '    const outcome = await runTransformStage(next.value, index, options.transform, latch)',
          '    index += 1',
          '    if (outcome.cancelled) {',
          '      await iterator.return?.()',
          '      await latch.runCleanups()',
          '      return',
          '    }',
          '    yield outcome.value',
          '  }',
          '  await latch.runCleanups()',
          '}',
          '',
          'export async function collectPipeline<TIn, TOut>(',
          '  options: StreamPipelineOptions<TIn, TOut>,',
          '): Promise<TOut[]> {',
          '  const output: TOut[] = []',
          '  for await (const item of streamPipeline(options)) {',
          '    output.push(item)',
          '  }',
          '  return output',
          '}',
          '',
        ].join('\n'),
      },
    ],
    referenceSolutionFiles: [
      {
        path: 'src/stream/abortLatch.ts',
        content: [
          'export interface AbortLatch {',
          '  readonly aborted: boolean',
          '  onCleanup(fn: () => Promise<void> | void): void',
          '  runCleanups(): Promise<void>',
          '}',
          '',
          'export function createAbortLatch(signal?: AbortSignal): AbortLatch {',
          '  let isAborted = Boolean(signal?.aborted)',
          '  let cleaned = false',
          '  const cleanups: Array<() => Promise<void> | void> = []',
          '',
          '  if (!isAborted) {',
          "    signal?.addEventListener('abort', () => {",
          '      isAborted = true',
          '    }, { once: true })',
          '  }',
          '',
          '  return {',
          '    get aborted(): boolean {',
          '      return isAborted',
          '    },',
          '    onCleanup(fn: () => Promise<void> | void): void {',
          '      cleanups.push(fn)',
          '    },',
          '    async runCleanups(): Promise<void> {',
          '      if (cleaned) return',
          '      cleaned = true',
          '      for (let index = cleanups.length - 1; index >= 0; index -= 1) {',
          '        await cleanups[index]?.()',
          '      }',
          '    },',
          '  }',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/stream/pipeline.ts',
        content: [
          "import { createAbortLatch } from './abortLatch.js'",
          "import { runTransformStage } from './transformStage.js'",
          '',
          'export interface StreamPipelineOptions<TIn, TOut> {',
          '  readonly source: AsyncIterable<TIn>',
          '  readonly transform: (item: TIn, index: number) => Promise<TOut> | TOut',
          '  readonly onAcquireResource?: () => (() => Promise<void> | void)',
          '  readonly signal?: AbortSignal',
          '}',
          '',
          'export async function* streamPipeline<TIn, TOut>(',
          '  options: StreamPipelineOptions<TIn, TOut>,',
          '): AsyncGenerator<TOut> {',
          '  const latch = createAbortLatch(options.signal)',
          '  if (options.onAcquireResource) {',
          '    latch.onCleanup(options.onAcquireResource())',
          '  }',
          '  let iterator: AsyncIterator<TIn> | undefined',
          '  let exhausted = false',
          '  try {',
          '    if (latch.aborted) return',
          '    iterator = options.source[Symbol.asyncIterator]()',
          '    let index = 0',
          '    while (!latch.aborted) {',
          '      const next = await iterator.next()',
          '      if (next.done) {',
          '        exhausted = true',
          '        break',
          '      }',
          '      const outcome = await runTransformStage(next.value, index, options.transform, latch)',
          '      index += 1',
          '      if (outcome.cancelled) {',
          '        return',
          '      }',
          '      yield outcome.value',
          '    }',
          '  } finally {',
          '    if (!exhausted && iterator?.return) {',
          '      await iterator.return()',
          '    }',
          '    await latch.runCleanups()',
          '  }',
          '}',
          '',
          'export async function collectPipeline<TIn, TOut>(',
          '  options: StreamPipelineOptions<TIn, TOut>,',
          '): Promise<TOut[]> {',
          '  const output: TOut[] = []',
          '  for await (const item of streamPipeline(options)) {',
          '    output.push(item)',
          '  }',
          '  return output',
          '}',
          '',
        ].join('\n'),
      },
    ],
  }
}

function buildCursorPaginationTask(): NativeSweTask {
  return {
    id: 'cursor-pagination-off-by-one',
    category: 'pagination',
    difficulty: 3,
    groundTruthTargetFiles: ['src/pagination/paginator.ts', 'src/services/exportJob.ts'],
    whyHard: 'Combines a lookahead nextCursor off-by-one and composite (timestampMs, id) tie-breaker bug in paginator.ts with a batchSize - 1 caller bug in exportJob.ts.',
    prompt: [
      'Fix the keyset cursor pagination bugs across `src/pagination/` and `src/services/`.',
      '`test/repro.test.ts` fails because multi-page pagination skips boundary items and produces',
      'uneven batch sizes across `auditFeed` and `exportJob`.',
      'Preserve composite `(timestampMs, id)` ordering and all baseline tests.',
    ].join(' '),
    sourceFiles: [
      {
        path: 'src/pagination/cursorCodec.ts',
        content: [
          'export interface KeysetCursor {',
          '  readonly timestampMs: number',
          '  readonly id: string',
          '}',
          '',
          'export function encodeCursor(cursor: KeysetCursor): string {',
          "  return Buffer.from(JSON.stringify(cursor), 'utf8').toString('base64url')",
          '}',
          '',
          'export function decodeCursor(token: string): KeysetCursor {',
          "  const decoded = JSON.parse(Buffer.from(token, 'base64url').toString('utf8')) as {",
          '    readonly timestampMs?: unknown',
          '    readonly id?: unknown',
          '  }',
          "  if (typeof decoded.timestampMs !== 'number' || typeof decoded.id !== 'string') {",
          "    throw new Error('Invalid pagination cursor')",
          '  }',
          '  return { timestampMs: decoded.timestampMs, id: decoded.id }',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/pagination/paginator.ts',
        content: [
          "import { decodeCursor, encodeCursor } from './cursorCodec.js'",
          '',
          'export interface TimestampedRecord {',
          '  readonly id: string',
          '  readonly timestampMs: number',
          '}',
          '',
          'export interface PaginateOptions {',
          '  readonly limit: number',
          '  readonly after?: string',
          '}',
          '',
          'export interface PaginatedPage<T extends TimestampedRecord> {',
          '  readonly items: readonly T[]',
          '  readonly hasMore: boolean',
          '  readonly nextCursor: string | null',
          '}',
          '',
          'export function paginateRecords<T extends TimestampedRecord>(',
          '  records: readonly T[],',
          '  options: PaginateOptions,',
          '): PaginatedPage<T> {',
          '  if (!Number.isInteger(options.limit) || options.limit < 1) {',
          "    throw new Error('limit must be a positive integer')",
          '  }',
          '  const sorted = [...records].sort((left, right) => {',
          '    if (left.timestampMs !== right.timestampMs) return left.timestampMs - right.timestampMs',
          '    return left.id.localeCompare(right.id)',
          '  })',
          '  const cursor = options.after ? decodeCursor(options.after) : undefined',
          '  const remaining = cursor',
          '    ? sorted.filter(item => item.timestampMs > cursor.timestampMs)',
          '    : sorted',
          '',
          '  const window = remaining.slice(0, options.limit + 1)',
          '  const hasMore = window.length > options.limit',
          '  const items = window.slice(0, options.limit)',
          '  const cursorRecord = hasMore ? window[options.limit] : undefined',
          '  return {',
          '    items,',
          '    hasMore,',
          '    nextCursor: cursorRecord',
          '      ? encodeCursor({ timestampMs: cursorRecord.timestampMs, id: cursorRecord.id })',
          '      : null,',
          '  }',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/services/auditFeed.ts',
        content: [
          "import { paginateRecords, type PaginatedPage, type TimestampedRecord } from '../pagination/paginator.js'",
          '',
          'export interface AuditEvent extends TimestampedRecord {',
          '  readonly actor: string',
          '  readonly action: string',
          '}',
          '',
          'export function fetchAuditFeedPage(',
          '  events: readonly AuditEvent[],',
          '  limit: number,',
          '  after?: string,',
          '): PaginatedPage<AuditEvent> {',
          '  return paginateRecords(events, {',
          '    limit,',
          '    ...(after === undefined ? {} : { after }),',
          '  })',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/services/exportJob.ts',
        content: [
          "import { paginateRecords, type TimestampedRecord } from '../pagination/paginator.js'",
          '',
          'export function exportAllBatches<T extends TimestampedRecord>(',
          '  records: readonly T[],',
          '  batchSize: number,',
          '): T[][] {',
          '  const batches: T[][] = []',
          '  let cursor: string | undefined',
          '  while (true) {',
          '    const pageLimit = cursor === undefined ? batchSize : Math.max(1, batchSize - 1)',
          '    const page = paginateRecords(records, {',
          '      limit: pageLimit,',
          '      ...(cursor === undefined ? {} : { after: cursor }),',
          '    })',
          '    if (page.items.length > 0) {',
          '      batches.push([...page.items])',
          '    }',
          '    if (!page.hasMore || page.nextCursor === null) {',
          '      break',
          '    }',
          '    cursor = page.nextCursor',
          '  }',
          '  return batches',
          '}',
          '',
        ].join('\n'),
      },
    ],
    baselineTestFiles: [
      {
        path: 'test/baseline.test.ts',
        content: [
          "import { expect, test } from 'bun:test'",
          "import { decodeCursor, encodeCursor } from '../src/pagination/cursorCodec.js'",
          "import { fetchAuditFeedPage, type AuditEvent } from '../src/services/auditFeed.js'",
          '',
          "test('cursor codec roundtrips timestamp and id', () => {",
          "  const token = encodeCursor({ timestampMs: 1700000000123, id: 'evt-9' })",
          "  expect(decodeCursor(token)).toEqual({ timestampMs: 1700000000123, id: 'evt-9' })",
          '})',
          '',
          "test('single page smaller than limit returns hasMore=false and null cursor', () => {",
          '  const events: AuditEvent[] = [',
          "    { id: 'evt-1', timestampMs: 100, actor: 'u1', action: 'login' },",
          "    { id: 'evt-2', timestampMs: 200, actor: 'u2', action: 'logout' },",
          '  ]',
          '  const page = fetchAuditFeedPage(events, 5)',
          "  expect(page.items.map(i => i.id)).toEqual(['evt-1', 'evt-2'])",
          '  expect(page.hasMore).toBe(false)',
          '  expect(page.nextCursor).toBeNull()',
          '})',
          '',
        ].join('\n'),
      },
    ],
    visibleReproductionTestFiles: [
      {
        path: 'test/repro.test.ts',
        content: [
          "import { expect, test } from 'bun:test'",
          "import { fetchAuditFeedPage, type AuditEvent } from '../src/services/auditFeed.js'",
          "import { exportAllBatches } from '../src/services/exportJob.js'",
          '',
          "test('multi-page cursor walk and batch export visit every record without skipping', () => {",
          '  const events: AuditEvent[] = [',
          "    { id: 'evt-1', timestampMs: 1000, actor: 'a', action: 'create' },",
          "    { id: 'evt-2', timestampMs: 2000, actor: 'a', action: 'update' },",
          "    { id: 'evt-3', timestampMs: 3000, actor: 'b', action: 'update' },",
          "    { id: 'evt-4', timestampMs: 4000, actor: 'b', action: 'verify' },",
          "    { id: 'evt-5', timestampMs: 5000, actor: 'c', action: 'close' },",
          '  ]',
          '',
          '  const first = fetchAuditFeedPage(events, 2)',
          "  expect(first.items.map(i => i.id)).toEqual(['evt-1', 'evt-2'])",
          '  expect(first.nextCursor).not.toBeNull()',
          '',
          '  const second = fetchAuditFeedPage(events, 2, first.nextCursor!)',
          "  expect(second.items.map(i => i.id)).toEqual(['evt-3', 'evt-4'])",
          '',
          '  const batches = exportAllBatches(events, 2)',
          '  expect(batches.map(batch => batch.map(item => item.id))).toEqual([',
          "    ['evt-1', 'evt-2'],",
          "    ['evt-3', 'evt-4'],",
          "    ['evt-5'],",
          '  ])',
          '})',
          '',
        ].join('\n'),
      },
    ],
    holdoutTestFiles: [
      {
        path: 'test/__holdout__/pagination.holdout.test.ts',
        content: [
          "import { expect, test } from 'bun:test'",
          "import { exportAllBatches } from '../../src/services/exportJob.js'",
          "import { fetchAuditFeedPage, type AuditEvent } from '../../src/services/auditFeed.js'",
          '',
          "test('burst events sharing identical millisecond timestamps paginate without drops or duplicates', () => {",
          '  const sameTs = 1700000000000',
          '  const burst: AuditEvent[] = [',
          "    { id: 'req-04', timestampMs: sameTs, actor: 's', action: 'tick' },",
          "    { id: 'req-01', timestampMs: sameTs, actor: 's', action: 'tick' },",
          "    { id: 'req-06', timestampMs: sameTs, actor: 's', action: 'tick' },",
          "    { id: 'req-02', timestampMs: sameTs, actor: 's', action: 'tick' },",
          "    { id: 'req-05', timestampMs: sameTs, actor: 's', action: 'tick' },",
          "    { id: 'req-03', timestampMs: sameTs, actor: 's', action: 'tick' },",
          '  ]',
          '',
          '  const batches = exportAllBatches(burst, 2)',
          '  expect(batches.map(b => b.map(e => e.id))).toEqual([',
          "    ['req-01', 'req-02'],",
          "    ['req-03', 'req-04'],",
          "    ['req-05', 'req-06'],",
          '  ])',
          '',
          '  const exactBoundary = fetchAuditFeedPage(burst, 6)',
          '  expect(exactBoundary.hasMore).toBe(false)',
          '  expect(exactBoundary.nextCursor).toBeNull()',
          '})',
          '',
        ].join('\n'),
      },
    ],
    symptomCheatFiles: [
      {
        path: 'src/pagination/paginator.ts',
        content: [
          "import { decodeCursor, encodeCursor } from './cursorCodec.js'",
          '',
          'export interface TimestampedRecord {',
          '  readonly id: string',
          '  readonly timestampMs: number',
          '}',
          '',
          'export interface PaginateOptions {',
          '  readonly limit: number',
          '  readonly after?: string',
          '}',
          '',
          'export interface PaginatedPage<T extends TimestampedRecord> {',
          '  readonly items: readonly T[]',
          '  readonly hasMore: boolean',
          '  readonly nextCursor: string | null',
          '}',
          '',
          'export function paginateRecords<T extends TimestampedRecord>(',
          '  records: readonly T[],',
          '  options: PaginateOptions,',
          '): PaginatedPage<T> {',
          '  if (!Number.isInteger(options.limit) || options.limit < 1) {',
          "    throw new Error('limit must be a positive integer')",
          '  }',
          '  const sorted = [...records].sort((left, right) => {',
          '    if (left.timestampMs !== right.timestampMs) return left.timestampMs - right.timestampMs',
          '    return left.id.localeCompare(right.id)',
          '  })',
          '  const cursor = options.after ? decodeCursor(options.after) : undefined',
          '  // Still buggy on equal timestamps!',
          '  const remaining = cursor',
          '    ? sorted.filter(item => item.timestampMs > cursor.timestampMs)',
          '    : sorted',
          '',
          '  const window = remaining.slice(0, options.limit + 1)',
          '  const hasMore = window.length > options.limit',
          '  const items = window.slice(0, options.limit)',
          '  const cursorRecord = hasMore ? items.at(-1) : undefined',
          '  return {',
          '    items,',
          '    hasMore,',
          '    nextCursor: cursorRecord',
          '      ? encodeCursor({ timestampMs: cursorRecord.timestampMs, id: cursorRecord.id })',
          '      : null,',
          '  }',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/services/exportJob.ts',
        content: [
          "import { paginateRecords, type TimestampedRecord } from '../pagination/paginator.js'",
          '',
          'export function exportAllBatches<T extends TimestampedRecord>(',
          '  records: readonly T[],',
          '  batchSize: number,',
          '): T[][] {',
          '  const batches: T[][] = []',
          '  let cursor: string | undefined',
          '  while (true) {',
          '    const page = paginateRecords(records, {',
          '      limit: batchSize,',
          '      ...(cursor === undefined ? {} : { after: cursor }),',
          '    })',
          '    if (page.items.length > 0) {',
          '      batches.push([...page.items])',
          '    }',
          '    if (!page.hasMore || page.nextCursor === null) {',
          '      break',
          '    }',
          '    cursor = page.nextCursor',
          '  }',
          '  return batches',
          '}',
          '',
        ].join('\n'),
      },
    ],
    referenceSolutionFiles: [
      {
        path: 'src/pagination/paginator.ts',
        content: [
          "import { decodeCursor, encodeCursor } from './cursorCodec.js'",
          '',
          'export interface TimestampedRecord {',
          '  readonly id: string',
          '  readonly timestampMs: number',
          '}',
          '',
          'export interface PaginateOptions {',
          '  readonly limit: number',
          '  readonly after?: string',
          '}',
          '',
          'export interface PaginatedPage<T extends TimestampedRecord> {',
          '  readonly items: readonly T[]',
          '  readonly hasMore: boolean',
          '  readonly nextCursor: string | null',
          '}',
          '',
          'export function paginateRecords<T extends TimestampedRecord>(',
          '  records: readonly T[],',
          '  options: PaginateOptions,',
          '): PaginatedPage<T> {',
          '  if (!Number.isInteger(options.limit) || options.limit < 1) {',
          "    throw new Error('limit must be a positive integer')",
          '  }',
          '  const sorted = [...records].sort((left, right) => {',
          '    if (left.timestampMs !== right.timestampMs) return left.timestampMs - right.timestampMs',
          '    return left.id.localeCompare(right.id)',
          '  })',
          '  const cursor = options.after ? decodeCursor(options.after) : undefined',
          '  const remaining = cursor',
          '    ? sorted.filter(',
          '        item => item.timestampMs > cursor.timestampMs',
          '          || (item.timestampMs === cursor.timestampMs && item.id.localeCompare(cursor.id) > 0),',
          '      )',
          '    : sorted',
          '',
          '  const window = remaining.slice(0, options.limit + 1)',
          '  const hasMore = window.length > options.limit',
          '  const items = window.slice(0, options.limit)',
          '  const cursorRecord = hasMore ? items.at(-1) : undefined',
          '  return {',
          '    items,',
          '    hasMore,',
          '    nextCursor: cursorRecord',
          '      ? encodeCursor({ timestampMs: cursorRecord.timestampMs, id: cursorRecord.id })',
          '      : null,',
          '  }',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/services/exportJob.ts',
        content: [
          "import { paginateRecords, type TimestampedRecord } from '../pagination/paginator.js'",
          '',
          'export function exportAllBatches<T extends TimestampedRecord>(',
          '  records: readonly T[],',
          '  batchSize: number,',
          '): T[][] {',
          '  const batches: T[][] = []',
          '  let cursor: string | undefined',
          '  while (true) {',
          '    const page = paginateRecords(records, {',
          '      limit: batchSize,',
          '      ...(cursor === undefined ? {} : { after: cursor }),',
          '    })',
          '    if (page.items.length > 0) {',
          '      batches.push([...page.items])',
          '    }',
          '    if (!page.hasMore || page.nextCursor === null) {',
          '      break',
          '    }',
          '    cursor = page.nextCursor',
          '  }',
          '  return batches',
          '}',
          '',
        ].join('\n'),
      },
    ],
  }
}

function buildSchemaCoercionValidatorTask(): NativeSweTask {
  return {
    id: 'schema-coercion-validator-edge',
    category: 'validation',
    difficulty: 3,
    groundTruthTargetFiles: ['src/schema/coercer.ts', 'src/schema/validator.ts'],
    whyHard: 'Requires fixing string-to-primitive edge cases in coercer.ts (whitespace numbers, case-insensitive/0/1 booleans, rejecting garbage booleans) AND falsy zero/false checks in validator.ts.',
    prompt: [
      'Fix the payload coercion and validation bugs in `src/schema/`.',
      '`test/repro.test.ts` fails when coercing and validating legitimate falsy values such as',
      '`retryCount: "0"` and `dryRun: "false"`.',
      'Ensure invalid strings, whitespace numbers, and numeric bounds (`min: 0`, `max: 0`) behave strictly.',
    ].join(' '),
    sourceFiles: [
      {
        path: 'src/schema/types.ts',
        content: [
          "export type FieldKind = 'string' | 'number' | 'boolean'",
          '',
          'export interface FieldSpec {',
          '  readonly type: FieldKind',
          '  readonly required?: boolean',
          '  readonly min?: number',
          '  readonly max?: number',
          '}',
          '',
          'export type SchemaSpec = Readonly<Record<string, FieldSpec>>',
          '',
          'export type ValidationResult =',
          '  | { readonly ok: true; readonly data: Record<string, string | number | boolean | undefined> }',
          '  | { readonly ok: false; readonly errors: readonly string[] }',
          '',
        ].join('\n'),
      },
      {
        path: 'src/schema/coercer.ts',
        content: [
          "import type { SchemaSpec } from './types.js'",
          '',
          'export function coercePayload(',
          '  raw: Readonly<Record<string, unknown>>,',
          '  schema: SchemaSpec,',
          '): Record<string, unknown> {',
          '  const coerced: Record<string, unknown> = {}',
          '  for (const [key, spec] of Object.entries(schema)) {',
          '    const value = raw[key]',
          '    if (value === undefined || value === null) {',
          '      coerced[key] = undefined',
          '      continue',
          '    }',
          "    if (spec.type === 'boolean' && typeof value === 'string') {",
          '      coerced[key] = Boolean(value)',
          '      continue',
          '    }',
          "    if (spec.type === 'number' && typeof value === 'string') {",
          '      coerced[key] = Number(value)',
          '      continue',
          '    }',
          '    coerced[key] = value',
          '  }',
          '  return coerced',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/schema/validator.ts',
        content: [
          "import { coercePayload } from './coercer.js'",
          "import type { SchemaSpec, ValidationResult } from './types.js'",
          '',
          'export function parseAndValidate(',
          '  raw: Readonly<Record<string, unknown>>,',
          '  schema: SchemaSpec,',
          '): ValidationResult {',
          '  const coerced = coercePayload(raw, schema)',
          '  const errors: string[] = []',
          '  const output: Record<string, string | number | boolean | undefined> = {}',
          '',
          '  for (const [key, spec] of Object.entries(schema)) {',
          '    const value = coerced[key]',
          '    if (!value) {',
          '      if (spec.required) {',
          '        errors.push(`${key} is required`)',
          '      }',
          '      output[key] = undefined',
          '      continue',
          '    }',
          "    if (spec.type === 'number') {",
          "      if (typeof value !== 'number' || Number.isNaN(value)) {",
          '        errors.push(`${key} must be a valid number`)',
          '        continue',
          '      }',
          '      if (spec.min && value < spec.min) {',
          '        errors.push(`${key} must be >= ${spec.min}`)',
          '        continue',
          '      }',
          '      if (spec.max && value > spec.max) {',
          '        errors.push(`${key} must be <= ${spec.max}`)',
          '        continue',
          '      }',
          '      output[key] = value',
          '      continue',
          '    }',
          "    if (spec.type === 'boolean') {",
          "      if (typeof value !== 'boolean') {",
          '        errors.push(`${key} must be a boolean`)',
          '        continue',
          '      }',
          '      output[key] = value',
          '      continue',
          '    }',
          "    if (typeof value !== 'string') {",
          '      errors.push(`${key} must be a string`)',
          '      continue',
          '    }',
          '    output[key] = value',
          '  }',
          '',
          '  return errors.length === 0 ? { ok: true, data: output } : { ok: false, errors }',
          '}',
          '',
        ].join('\n'),
      },
    ],
    baselineTestFiles: [
      {
        path: 'test/baseline.test.ts',
        content: [
          "import { expect, test } from 'bun:test'",
          "import { parseAndValidate } from '../src/schema/validator.js'",
          "import type { SchemaSpec } from '../src/schema/types.js'",
          '',
          'const serverSchema: SchemaSpec = {',
          "  host: { type: 'string', required: true },",
          "  port: { type: 'number', required: true, min: 1024, max: 65535 },",
          "  tls: { type: 'boolean', required: true },",
          '}',
          '',
          "test('coerces and validates valid positive configuration', () => {",
          "  const res = parseAndValidate({ host: 'api.local', port: '8080', tls: 'true' }, serverSchema)",
          "  expect(res).toEqual({ ok: true, data: { host: 'api.local', port: 8080, tls: true } })",
          '})',
          '',
          "test('rejects missing required fields', () => {",
          "  const res = parseAndValidate({ port: '8080', tls: 'true' }, serverSchema)",
          '  expect(res.ok).toBe(false)',
          '})',
          '',
        ].join('\n'),
      },
    ],
    visibleReproductionTestFiles: [
      {
        path: 'test/repro.test.ts',
        content: [
          "import { expect, test } from 'bun:test'",
          "import { parseAndValidate } from '../src/schema/validator.js'",
          '',
          "test('accepts zero retryCount and string false dryRun', () => {",
          '  const result = parseAndValidate(',
          "    { retryCount: '0', dryRun: 'false' },",
          '    {',
          "      retryCount: { type: 'number', required: true, min: 0, max: 10 },",
          "      dryRun: { type: 'boolean', required: true },",
          '    },',
          '  )',
          '  expect(result).toEqual({',
          '    ok: true,',
          '    data: { retryCount: 0, dryRun: false },',
          '  })',
          '})',
          '',
        ].join('\n'),
      },
    ],
    holdoutTestFiles: [
      {
        path: 'test/__holdout__/schema.holdout.test.ts',
        content: [
          "import { expect, test } from 'bun:test'",
          "import { parseAndValidate } from '../../src/schema/validator.js'",
          '',
          "test('enforces min: 0 and max: 0 bounds on negative and positive values', () => {",
          '  const neg = parseAndValidate(',
          "    { retryCount: '-3' },",
          "    { retryCount: { type: 'number', required: true, min: 0 } },",
          '  )',
          '  expect(neg.ok).toBe(false)',
          '',
          '  const posOverZeroMax = parseAndValidate(',
          "    { delta: '2' },",
          "    { delta: { type: 'number', required: true, max: 0 } },",
          '  )',
          '  expect(posOverZeroMax.ok).toBe(false)',
          '})',
          '',
          "test('rejects whitespace-only numeric strings and invalid boolean strings while supporting 0/1/TRUE/FALSE', () => {",
          '  const blankNumber = parseAndValidate(',
          "    { retryCount: '   ' },",
          "    { retryCount: { type: 'number', required: true, min: 0 } },",
          '  )',
          '  expect(blankNumber.ok).toBe(false)',
          '',
          '  const invalidBool = parseAndValidate(',
          "    { enabled: 'definitely' },",
          "    { enabled: { type: 'boolean', required: true } },",
          '  )',
          '  expect(invalidBool.ok).toBe(false)',
          '',
          '  const validBoolVariants = parseAndValidate(',
          "    { a: '0', b: '1', c: 'FALSE', d: 'TRUE' },",
          '    {',
          "      a: { type: 'boolean', required: true },",
          "      b: { type: 'boolean', required: true },",
          "      c: { type: 'boolean', required: true },",
          "      d: { type: 'boolean', required: true },",
          '    },',
          '  )',
          '  expect(validBoolVariants).toEqual({',
          '    ok: true,',
          '    data: { a: false, b: true, c: false, d: true },',
          '  })',
          '})',
          '',
        ].join('\n'),
      },
    ],
    symptomCheatFiles: [
      {
        path: 'src/schema/coercer.ts',
        content: [
          "import type { SchemaSpec } from './types.js'",
          '',
          'export function coercePayload(',
          '  raw: Readonly<Record<string, unknown>>,',
          '  schema: SchemaSpec,',
          '): Record<string, unknown> {',
          '  const coerced: Record<string, unknown> = {}',
          '  for (const [key, spec] of Object.entries(schema)) {',
          '    const value = raw[key]',
          '    if (value === undefined || value === null) {',
          '      coerced[key] = undefined',
          '      continue',
          '    }',
          "    if (spec.type === 'boolean' && typeof value === 'string') {",
          "      coerced[key] = value === 'false' ? false : Boolean(value)",
          '      continue',
          '    }',
          "    if (spec.type === 'number' && typeof value === 'string') {",
          '      coerced[key] = Number(value)',
          '      continue',
          '    }',
          '    coerced[key] = value',
          '  }',
          '  return coerced',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/schema/validator.ts',
        content: [
          "import { coercePayload } from './coercer.js'",
          "import type { SchemaSpec, ValidationResult } from './types.js'",
          '',
          'export function parseAndValidate(',
          '  raw: Readonly<Record<string, unknown>>,',
          '  schema: SchemaSpec,',
          '): ValidationResult {',
          '  const coerced = coercePayload(raw, schema)',
          '  const errors: string[] = []',
          '  const output: Record<string, string | number | boolean | undefined> = {}',
          '',
          '  for (const [key, spec] of Object.entries(schema)) {',
          '    const value = coerced[key]',
          "    if (value === undefined || value === null || value === '') {",
          '      if (spec.required) {',
          '        errors.push(`${key} is required`)',
          '      }',
          '      output[key] = undefined',
          '      continue',
          '    }',
          "    if (spec.type === 'number') {",
          "      if (typeof value !== 'number' || Number.isNaN(value)) {",
          '        errors.push(`${key} must be a valid number`)',
          '        continue',
          '      }',
          '      // Still buggy when min === 0 or max === 0!',
          '      if (spec.min && value < spec.min) {',
          '        errors.push(`${key} must be >= ${spec.min}`)',
          '        continue',
          '      }',
          '      if (spec.max && value > spec.max) {',
          '        errors.push(`${key} must be <= ${spec.max}`)',
          '        continue',
          '      }',
          '      output[key] = value',
          '      continue',
          '    }',
          "    if (spec.type === 'boolean') {",
          "      if (typeof value !== 'boolean') {",
          '        errors.push(`${key} must be a boolean`)',
          '        continue',
          '      }',
          '      output[key] = value',
          '      continue',
          '    }',
          "    if (typeof value !== 'string') {",
          '      errors.push(`${key} must be a string`)',
          '      continue',
          '    }',
          '    output[key] = value',
          '  }',
          '',
          '  return errors.length === 0 ? { ok: true, data: output } : { ok: false, errors }',
          '}',
          '',
        ].join('\n'),
      },
    ],
    referenceSolutionFiles: [
      {
        path: 'src/schema/coercer.ts',
        content: [
          "import type { SchemaSpec } from './types.js'",
          '',
          'const INVALID_COERCION = Symbol.for(\'xerxes.schema.invalid\')',
          '',
          'export function coercePayload(',
          '  raw: Readonly<Record<string, unknown>>,',
          '  schema: SchemaSpec,',
          '): Record<string, unknown> {',
          '  const coerced: Record<string, unknown> = {}',
          '  for (const [key, spec] of Object.entries(schema)) {',
          '    const value = raw[key]',
          '    if (value === undefined || value === null) {',
          '      coerced[key] = undefined',
          '      continue',
          '    }',
          "    if (spec.type === 'boolean' && typeof value === 'string') {",
          '      const normalized = value.trim().toLowerCase()',
          "      if (normalized === 'true' || normalized === '1') {",
          '        coerced[key] = true',
          "      } else if (normalized === 'false' || normalized === '0') {",
          '        coerced[key] = false',
          '      } else {',
          '        coerced[key] = INVALID_COERCION',
          '      }',
          '      continue',
          '    }',
          "    if (spec.type === 'number' && typeof value === 'string') {",
          '      const trimmed = value.trim()',
          '      if (!trimmed) {',
          '        coerced[key] = undefined',
          '        continue',
          '      }',
          '      const parsed = Number(trimmed)',
          '      coerced[key] = Number.isFinite(parsed) ? parsed : Number.NaN',
          '      continue',
          '    }',
          '    coerced[key] = value',
          '  }',
          '  return coerced',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/schema/validator.ts',
        content: [
          "import { coercePayload } from './coercer.js'",
          "import type { SchemaSpec, ValidationResult } from './types.js'",
          '',
          'export function parseAndValidate(',
          '  raw: Readonly<Record<string, unknown>>,',
          '  schema: SchemaSpec,',
          '): ValidationResult {',
          '  const coerced = coercePayload(raw, schema)',
          '  const errors: string[] = []',
          '  const output: Record<string, string | number | boolean | undefined> = {}',
          '',
          '  for (const [key, spec] of Object.entries(schema)) {',
          '    const value = coerced[key]',
          "    if (value === undefined || value === null || (spec.type === 'string' && value === '')) {",
          '      if (spec.required) {',
          '        errors.push(`${key} is required`)',
          '      }',
          '      output[key] = undefined',
          '      continue',
          '    }',
          "    if (spec.type === 'number') {",
          "      if (typeof value !== 'number' || !Number.isFinite(value)) {",
          '        errors.push(`${key} must be a valid number`)',
          '        continue',
          '      }',
          '      if (spec.min !== undefined && value < spec.min) {',
          '        errors.push(`${key} must be >= ${spec.min}`)',
          '        continue',
          '      }',
          '      if (spec.max !== undefined && value > spec.max) {',
          '        errors.push(`${key} must be <= ${spec.max}`)',
          '        continue',
          '      }',
          '      output[key] = value',
          '      continue',
          '    }',
          "    if (spec.type === 'boolean') {",
          "      if (typeof value !== 'boolean') {",
          '        errors.push(`${key} must be a boolean`)',
          '        continue',
          '      }',
          '      output[key] = value',
          '      continue',
          '    }',
          "    if (typeof value !== 'string') {",
          '      errors.push(`${key} must be a string`)',
          '      continue',
          '    }',
          '    output[key] = value',
          '  }',
          '',
          '  return errors.length === 0 ? { ok: true, data: output } : { ok: false, errors }',
          '}',
          '',
        ].join('\n'),
      },
    ],
  }
}

function buildDagToposortCacheTask(): NativeSweTask {
  return {
    id: 'dag-toposort-cache-invalidation',
    category: 'graph-state',
    difficulty: 4,
    groundTruthTargetFiles: ['src/graph/dag.ts', 'src/graph/artifactCache.ts'],
    whyHard: 'Requires 3-color cycle detection + diamond deduplication in dag.ts AND selective transitive downstream cache invalidation in artifactCache.ts (without wiping unrelated cached subgraphs).',
    prompt: [
      'Fix the DAG topological sort and incremental artifact cache bugs in `src/graph/`.',
      '`test/repro.test.ts` fails on diamond build graphs: shared dependencies execute multiple times',
      'and invalidating an upstream node leaves downstream dependents serving stale cached artifacts.',
      'Ensure dependency cycles throw an error mentioning `cycle` and unrelated cached subgraphs remain intact.',
    ].join(' '),
    sourceFiles: [
      {
        path: 'src/graph/dag.ts',
        content: [
          'export interface DagNode {',
          '  readonly id: string',
          '  readonly deps: readonly string[]',
          '}',
          '',
          'export function topologicalSort(nodes: readonly DagNode[]): string[] {',
          '  const byId = new Map(nodes.map(node => [node.id, node]))',
          '  const order: string[] = []',
          '',
          '  const visit = (id: string): void => {',
          '    const node = byId.get(id)',
          '    if (!node) {',
          '      throw new Error(`Unknown dependency node: ${id}`)',
          '    }',
          '    for (const dep of node.deps) {',
          '      visit(dep)',
          '    }',
          '    order.push(id)',
          '  }',
          '',
          '  for (const node of nodes) {',
          '    if (!order.includes(node.id)) {',
          '      visit(node.id)',
          '    }',
          '  }',
          '  return order',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/graph/artifactCache.ts',
        content: [
          'export class ArtifactCache<T> {',
          '  private readonly entries = new Map<string, T>()',
          '  private readonly reverseDeps = new Map<string, Set<string>>()',
          '',
          '  registerDependencies(nodeId: string, deps: readonly string[]): void {',
          '    for (const dep of deps) {',
          '      const dependents = this.reverseDeps.get(dep) ?? new Set<string>()',
          '      dependents.add(nodeId)',
          '      this.reverseDeps.set(dep, dependents)',
          '    }',
          '  }',
          '',
          '  has(nodeId: string): boolean {',
          '    return this.entries.has(nodeId)',
          '  }',
          '',
          '  get(nodeId: string): T | undefined {',
          '    return this.entries.get(nodeId)',
          '  }',
          '',
          '  set(nodeId: string, value: T): void {',
          '    this.entries.set(nodeId, value)',
          '  }',
          '',
          '  invalidate(nodeId: string): void {',
          '    // BUG: only evicts direct node, leaving downstream dependents stale!',
          '    this.entries.delete(nodeId)',
          '  }',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/graph/scheduler.ts',
        content: [
          "import { ArtifactCache } from './artifactCache.js'",
          "import { topologicalSort, type DagNode } from './dag.js'",
          '',
          'export interface BuildTaskNode extends DagNode {',
          '  compute(depOutputs: Readonly<Record<string, string>>): string',
          '}',
          '',
          'export class BuildScheduler {',
          '  private readonly nodes: readonly BuildTaskNode[]',
          '  private readonly byId: ReadonlyMap<string, BuildTaskNode>',
          '  private readonly cache = new ArtifactCache<string>()',
          '',
          '  constructor(nodes: readonly BuildTaskNode[]) {',
          '    this.nodes = nodes',
          '    this.byId = new Map(nodes.map(node => [node.id, node]))',
          '    for (const node of nodes) {',
          '      this.cache.registerDependencies(node.id, node.deps)',
          '    }',
          '  }',
          '',
          '  invalidate(nodeId: string): void {',
          '    this.cache.invalidate(nodeId)',
          '  }',
          '',
          '  buildAll(): Record<string, string> {',
          '    const order = topologicalSort(this.nodes)',
          '    const outputs: Record<string, string> = {}',
          '    for (const id of order) {',
          '      if (this.cache.has(id)) {',
          '        outputs[id] = this.cache.get(id)!',
          '        continue',
          '      }',
          '      const node = this.byId.get(id)!',
          '      const depOutputs: Record<string, string> = {}',
          '      for (const dep of node.deps) {',
          '        depOutputs[dep] = outputs[dep] ?? this.cache.get(dep)!',
          '      }',
          '      const built = node.compute(depOutputs)',
          '      this.cache.set(id, built)',
          '      outputs[id] = built',
          '    }',
          '    return outputs',
          '  }',
          '}',
          '',
        ].join('\n'),
      },
    ],
    baselineTestFiles: [
      {
        path: 'test/baseline.test.ts',
        content: [
          "import { expect, test } from 'bun:test'",
          "import { BuildScheduler } from '../src/graph/scheduler.js'",
          '',
          "test('linear chain builds in order and reuses cached outputs on second build', () => {",
          '  let runs = 0',
          '  const scheduler = new BuildScheduler([',
          "    { id: 'config', deps: [], compute: () => { runs += 1; return 'cfg-v1' } },",
          "    { id: 'core', deps: ['config'], compute: d => { runs += 1; return `core(${d.config})` } },",
          "    { id: 'app', deps: ['core'], compute: d => { runs += 1; return `app(${d.core})` } },",
          '  ])',
          '  const first = scheduler.buildAll()',
          "  expect(first.app).toBe('app(core(cfg-v1))')",
          '  expect(runs).toBe(3)',
          '',
          '  const second = scheduler.buildAll()',
          "  expect(second.app).toBe('app(core(cfg-v1))')",
          '  expect(runs).toBe(3)',
          '})',
          '',
        ].join('\n'),
      },
    ],
    visibleReproductionTestFiles: [
      {
        path: 'test/repro.test.ts',
        content: [
          "import { expect, test } from 'bun:test'",
          "import { topologicalSort } from '../src/graph/dag.js'",
          "import { BuildScheduler } from '../src/graph/scheduler.js'",
          '',
          "test('diamond graph visits shared base once and invalidating base rebuilds bundle', () => {",
          '  const order = topologicalSort([',
          "    { id: 'base', deps: [] },",
          "    { id: 'left', deps: ['base'] },",
          "    { id: 'right', deps: ['base'] },",
          "    { id: 'bundle', deps: ['left', 'right'] },",
          '  ])',
          '  expect(order).toHaveLength(4)',
          "  expect(order[0]).toBe('base')",
          "  expect(order[3]).toBe('bundle')",
          '',
          "  let baseVersion = 'v1'",
          '  let baseRuns = 0',
          '  const scheduler = new BuildScheduler([',
          "    { id: 'base', deps: [], compute: () => { baseRuns += 1; return baseVersion } },",
          "    { id: 'left', deps: ['base'], compute: d => `L:${d.base}` },",
          "    { id: 'right', deps: ['base'], compute: d => `R:${d.base}` },",
          "    { id: 'bundle', deps: ['left', 'right'], compute: d => `${d.left}+${d.right}` },",
          '  ])',
          '',
          '  expect(scheduler.buildAll().bundle).toBe(\'L:v1+R:v1\')',
          '  expect(baseRuns).toBe(1)',
          '',
          "  baseVersion = 'v2'",
          "  scheduler.invalidate('base')",
          '  expect(scheduler.buildAll().bundle).toBe(\'L:v2+R:v2\')',
          '  expect(baseRuns).toBe(2)',
          '})',
          '',
        ].join('\n'),
      },
    ],
    holdoutTestFiles: [
      {
        path: 'test/__holdout__/dag.holdout.test.ts',
        content: [
          "import { expect, test } from 'bun:test'",
          "import { topologicalSort } from '../../src/graph/dag.js'",
          "import { BuildScheduler } from '../../src/graph/scheduler.js'",
          '',
          "test('throws descriptive error when graph contains a dependency cycle', () => {",
          '  expect(() =>',
          '    topologicalSort([',
          "      { id: 'init', deps: [] },",
          "      { id: 'a', deps: ['c'] },",
          "      { id: 'b', deps: ['a'] },",
          "      { id: 'c', deps: ['b'] },",
          '    ]),',
          '  ).toThrow(/cycle/i)',
          '})',
          '',
          "test('selective invalidation preserves cached outputs of unrelated subgraphs', () => {",
          "  let baseRev = '1'",
          '  let docsRuns = 0',
          '  let lintDocsRuns = 0',
          '  const scheduler = new BuildScheduler([',
          "    { id: 'base', deps: [], compute: () => baseRev },",
          "    { id: 'api', deps: ['base'], compute: d => `api:${d.base}` },",
          "    { id: 'cli', deps: ['api'], compute: d => `cli:${d.api}` },",
          "    { id: 'docs', deps: [], compute: () => { docsRuns += 1; return 'docs:ok' } },",
          "    { id: 'lint-docs', deps: ['docs'], compute: d => { lintDocsRuns += 1; return `lint(${d.docs})` } },",
          '  ])',
          '',
          '  const first = scheduler.buildAll()',
          "  expect(first.cli).toBe('cli:api:1')",
          '  expect(docsRuns).toBe(1)',
          '  expect(lintDocsRuns).toBe(1)',
          '',
          "  baseRev = '2'",
          "  scheduler.invalidate('base')",
          '  const second = scheduler.buildAll()',
          "  expect(second.cli).toBe('cli:api:2')",
          '  // Unrelated docs subgraph MUST NOT be recomputed!',
          '  expect(docsRuns).toBe(1)',
          '  expect(lintDocsRuns).toBe(1)',
          '})',
          '',
        ].join('\n'),
      },
    ],
    symptomCheatFiles: [
      {
        path: 'src/graph/dag.ts',
        content: [
          'export interface DagNode {',
          '  readonly id: string',
          '  readonly deps: readonly string[]',
          '}',
          '',
          'export function topologicalSort(nodes: readonly DagNode[]): string[] {',
          '  const byId = new Map(nodes.map(node => [node.id, node]))',
          '  const visited = new Set<string>()',
          '  const order: string[] = []',
          '',
          '  // Cheat: deduplicates diamonds with a single visited set, silently ignoring cycles!',
          '  const visit = (id: string): void => {',
          '    if (visited.has(id)) return',
          '    visited.add(id)',
          '    const node = byId.get(id)',
          '    if (!node) throw new Error(`Unknown dependency node: ${id}`)',
          '    for (const dep of node.deps) {',
          '      visit(dep)',
          '    }',
          '    order.push(id)',
          '  }',
          '',
          '  for (const node of nodes) {',
          '    visit(node.id)',
          '  }',
          '  return order',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/graph/artifactCache.ts',
        content: [
          'export class ArtifactCache<T> {',
          '  private readonly entries = new Map<string, T>()',
          '  private readonly reverseDeps = new Map<string, Set<string>>()',
          '',
          '  registerDependencies(nodeId: string, deps: readonly string[]): void {',
          '    for (const dep of deps) {',
          '      const dependents = this.reverseDeps.get(dep) ?? new Set<string>()',
          '      dependents.add(nodeId)',
          '      this.reverseDeps.set(dep, dependents)',
          '    }',
          '  }',
          '',
          '  has(nodeId: string): boolean {',
          '    return this.entries.has(nodeId)',
          '  }',
          '',
          '  get(nodeId: string): T | undefined {',
          '    return this.entries.get(nodeId)',
          '  }',
          '',
          '  set(nodeId: string, value: T): void {',
          '    this.entries.set(nodeId, value)',
          '  }',
          '',
          '  invalidate(_nodeId: string): void {',
          '    // Cheat: wipes the entire cache instead of transitive downstream invalidation!',
          '    this.entries.clear()',
          '  }',
          '}',
          '',
        ].join('\n'),
      },
    ],
    referenceSolutionFiles: [
      {
        path: 'src/graph/dag.ts',
        content: [
          'export interface DagNode {',
          '  readonly id: string',
          '  readonly deps: readonly string[]',
          '}',
          '',
          'export function topologicalSort(nodes: readonly DagNode[]): string[] {',
          '  const byId = new Map(nodes.map(node => [node.id, node]))',
          '  const state = new Map<string, \'visiting\' | \'visited\'>()',
          '  const order: string[] = []',
          '',
          '  const visit = (id: string): void => {',
          '    const current = state.get(id)',
          "    if (current === 'visited') return",
          "    if (current === 'visiting') {",
          '      throw new Error(`Dependency cycle detected involving "${id}"`)',
          '    }',
          '    const node = byId.get(id)',
          '    if (!node) {',
          '      throw new Error(`Unknown dependency node: ${id}`)',
          '    }',
          "    state.set(id, 'visiting')",
          '    for (const dep of node.deps) {',
          '      visit(dep)',
          '    }',
          "    state.set(id, 'visited')",
          '    order.push(id)',
          '  }',
          '',
          '  for (const node of nodes) {',
          '    visit(node.id)',
          '  }',
          '  return order',
          '}',
          '',
        ].join('\n'),
      },
      {
        path: 'src/graph/artifactCache.ts',
        content: [
          'export class ArtifactCache<T> {',
          '  private readonly entries = new Map<string, T>()',
          '  private readonly reverseDeps = new Map<string, Set<string>>()',
          '',
          '  registerDependencies(nodeId: string, deps: readonly string[]): void {',
          '    for (const dep of deps) {',
          '      const dependents = this.reverseDeps.get(dep) ?? new Set<string>()',
          '      dependents.add(nodeId)',
          '      this.reverseDeps.set(dep, dependents)',
          '    }',
          '  }',
          '',
          '  has(nodeId: string): boolean {',
          '    return this.entries.has(nodeId)',
          '  }',
          '',
          '  get(nodeId: string): T | undefined {',
          '    return this.entries.get(nodeId)',
          '  }',
          '',
          '  set(nodeId: string, value: T): void {',
          '    this.entries.set(nodeId, value)',
          '  }',
          '',
          '  invalidate(nodeId: string): void {',
          '    const queue: string[] = [nodeId]',
          '    const visited = new Set<string>()',
          '    while (queue.length > 0) {',
          '      const current = queue.shift()!',
          '      if (visited.has(current)) continue',
          '      visited.add(current)',
          '      this.entries.delete(current)',
          '      const dependents = this.reverseDeps.get(current)',
          '      if (dependents) {',
          '        for (const dep of dependents) {',
          '          if (!visited.has(dep)) queue.push(dep)',
          '        }',
          '      }',
          '    }',
          '  }',
          '}',
          '',
        ].join('\n'),
      },
    ],
  }
}

/** Construct the deterministic, offline, multi-file SWE bugfix benchmark battery. */
export function createNativeSweTasks(): readonly NativeSweTask[] {
  return [
    buildStreamCancelRaceTask(),
    buildCursorPaginationTask(),
    buildSchemaCoercionValidatorTask(),
    buildDagToposortCacheTask(),
  ]
}
