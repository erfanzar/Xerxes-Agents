// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, rm, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import type {
  EvaluationAgent,
  EvaluationCheck,
  EvaluationReport,
  EvaluationScoreRow,
  EvaluationTaskStep,
  EvaluationTurnResult,
} from './types.js'
import type { WorkspaceModuleEvaluator } from './evaluator.js'

/** A native warm-up task with a behavioral or answer-based checker. */
export interface WarmupTask {
  readonly category: string
  check(results: readonly EvaluationTurnResult[], workspaceDirectory: string): Promise<EvaluationCheck> | EvaluationCheck
  readonly name: string
  readonly steps: readonly EvaluationTaskStep[]
}

/** One completed prompt step, emitted for callers that want Python-compatible verbose progress. */
export interface WarmupStepResult {
  readonly prompt: string
  readonly result: EvaluationTurnResult
  readonly stepIndex: number
  readonly task: Pick<WarmupTask, 'category' | 'name'>
}

/** Inputs for the native warm-up suite. */
export interface WarmupSuiteOptions {
  readonly agent: EvaluationAgent
  readonly keyword?: string
  /** Required only because coding checks deliberately execute through a host-selected evaluator. */
  readonly moduleEvaluator: WorkspaceModuleEvaluator
  readonly onRow?: (row: EvaluationScoreRow) => void
  readonly onStep?: (step: WarmupStepResult) => void | Promise<void>
  readonly sandboxDirectory: string
}

/** Build the controlled files used by the warm-up task battery. */
export async function buildWarmupWorkspace(workspaceDirectory: string): Promise<void> {
  const workspace = requiredDirectory(workspaceDirectory, 'workspaceDirectory')
  await rm(workspace, { force: true, recursive: true })
  await mkdir(resolve(workspace, 'notes'), { recursive: true })
  await Promise.all([
    writeFile(resolve(workspace, 'config.txt'), 'APP_NAME=demo\nAPI_VERSION=4.2.0\nDEBUG=false\n', 'utf8'),
    writeFile(resolve(workspace, 'greeting.ts'), "export function greet(): string {\n  return 'hello'\n}\n", 'utf8'),
    writeFile(resolve(workspace, 'calc.ts'), 'export function add(a: number, b: number): number {\n  return a - b // BUG: should add\n}\n', 'utf8'),
    writeFile(resolve(workspace, 'notes/a.txt'), 'nothing interesting here\n', 'utf8'),
    writeFile(resolve(workspace, 'notes/b.txt'), 'the TREASURE is buried in this file\n', 'utf8'),
    writeFile(resolve(workspace, 'notes/c.txt'), 'just some other text\n', 'utf8'),
  ])
}

/** Return the native TypeScript equivalent of the original warm-up battery. */
export function createWarmupTasks(moduleEvaluator: WorkspaceModuleEvaluator): readonly WarmupTask[] {
  return [
    {
      name: 'reasoning',
      category: 'reason',
      steps: [{ prompt: 'What is 17 times 23? Reply with only the number.', freshSession: true }],
      check: results => answerIncludes(results, '391', 40),
    },
    {
      name: 'file_read',
      category: 'tools',
      steps: [{
        prompt: 'Read the file config.txt in the current directory and reply with ONLY the value of API_VERSION.',
        freshSession: true,
      }],
      check: results => answerIncludes(results, '4.2.0', 40),
    },
    {
      name: 'file_edit',
      category: 'coding',
      steps: [{
        prompt: "Edit greeting.ts so greet() returns 'goodbye' instead of 'hello', then confirm.",
        freshSession: true,
      }],
      check: async (_results, workspaceDirectory) => {
        const output = await invoke(moduleEvaluator, workspaceDirectory, 'greeting.ts', 'greet', [])
        return { ok: output === 'goodbye', detail: `greet() -> ${JSON.stringify(output)}` }
      },
    },
    {
      name: 'bug_fix',
      category: 'coding',
      steps: [{
        prompt: 'There is a bug in calc.ts: add(a, b) subtracts instead of adds. Fix it so it adds.',
        freshSession: true,
      }],
      check: async (_results, workspaceDirectory) => {
        const output = await invoke(moduleEvaluator, workspaceDirectory, 'calc.ts', 'add', [2, 3])
        return { ok: output === 5, detail: `add(2,3) -> ${JSON.stringify(output)}` }
      },
    },
    {
      name: 'tool_search',
      category: 'tools',
      steps: [{
        prompt: 'In the notes/ directory, find which file contains the word TREASURE and reply with ONLY its filename.',
        freshSession: true,
      }],
      check: results => {
        const result = lastResult(results)
        return {
          ok: result.text.includes('b.txt'),
          detail: `tools=${JSON.stringify(result.tools.slice(0, 4))} ans=${JSON.stringify(result.text.slice(0, 24))}`,
        }
      },
    },
    {
      name: 'shell',
      category: 'tools',
      steps: [{
        prompt: 'Run the shell command `echo eval-ok-7` and reply with ONLY the command output.',
        freshSession: true,
      }],
      check: results => answerIncludes(results, 'eval-ok-7', 30),
    },
    {
      name: 'multiturn',
      category: 'context',
      steps: [
        { prompt: 'My favorite number is 7. Acknowledge in one word.', freshSession: true },
        { prompt: 'Multiply my favorite number by 6. Reply with only the number.', freshSession: false },
      ],
      check: results => answerIncludes(results, '42', 30),
    },
    {
      name: 'memory_recall',
      category: 'memory',
      steps: [
        { prompt: 'Note for your memory: the deploy target is fly.io. Acknowledge in one word.', freshSession: true },
        {
          prompt: 'Check your memory/journal: what is the deploy target I noted earlier? Reply with only the value.',
          freshSession: true,
        },
      ],
      check: results => {
        const result = lastResult(results)
        return {
          ok: result.text.toLowerCase().includes('fly.io'),
          detail: `fresh session -> ${JSON.stringify(result.text.slice(0, 24))}`,
        }
      },
    },
  ]
}

/** Run and score warm-up tasks against a caller-owned native evaluation agent. */
export async function runWarmupSuite(options: WarmupSuiteOptions): Promise<EvaluationReport> {
  const sandboxDirectory = requiredDirectory(options.sandboxDirectory, 'sandboxDirectory')
  await buildWarmupWorkspace(sandboxDirectory)
  const tasks = createWarmupTasks(options.moduleEvaluator)
    .filter(task => options.keyword === undefined || task.name.includes(options.keyword))
  const rows: EvaluationScoreRow[] = []

  try {
    await options.agent.start()
    for (const task of tasks) {
      const results: EvaluationTurnResult[] = []
      for (const [stepIndex, step] of task.steps.entries()) {
        if (step.freshSession) await options.agent.freshSession()
        const result = await options.agent.turn(step.prompt)
        results.push(result)
        await options.onStep?.({
          prompt: step.prompt,
          result,
          stepIndex,
          task: { category: task.category, name: task.name },
        })
      }
      const last = lastResult(results)
      const check = last.error === undefined
        ? await checkWarmupTask(task, results, sandboxDirectory)
        : { ok: false, detail: `ERROR: ${last.error}` }
      const row: EvaluationScoreRow = {
        category: task.category,
        detail: check.detail,
        diagnosis: undefined,
        difficulty: undefined,
        latencyMs: totalLatency(results),
        name: task.name,
        ok: check.ok,
      }
      rows.push(row)
      options.onRow?.(row)
    }
  } finally {
    await options.agent.close()
  }

  return {
    kind: 'warmup',
    model: options.agent.model,
    rows,
    sandboxDirectory,
    totalLatencyMs: totalLatency(rows),
  }
}

/** Render a scorecard matching the warm-up playground's concise terminal report. */
export function formatWarmupReport(report: EvaluationReport): string {
  if (report.kind !== 'warmup') throw new Error('formatWarmupReport requires a warmup report')
  const rows = report.rows.map(row => (
    `  [${row.ok ? 'PASS' : 'FAIL'}] ${pad(row.name, 14)} ${pad(row.category, 8)} ${seconds(row.latencyMs)}  ${row.detail}`
  ))
  const passed = report.rows.filter(row => row.ok).length
  const failed = report.rows.filter(row => !row.ok).map(row => row.name)
  return [
    '',
    `  Xerxes eval playground — model: ${report.model}   sandbox: ${report.sandboxDirectory}`,
    '',
    ...rows,
    '',
    `  ${'-'.repeat(60)}`,
    `  SCORE: ${passed}/${report.rows.length} passed   (${percentage(passed, report.rows.length)}%)   total ${seconds(report.totalLatencyMs)}   model=${report.model}`,
    ...(failed.length === 0 ? [] : [`  failed: ${failed.join(', ')}`]),
    '',
  ].join('\n')
}

function answerIncludes(results: readonly EvaluationTurnResult[], expected: string, detailLength: number): EvaluationCheck {
  const text = lastResult(results).text
  return { ok: text.includes(expected), detail: text.slice(0, detailLength) }
}

async function checkWarmupTask(
  task: WarmupTask,
  results: readonly EvaluationTurnResult[],
  workspaceDirectory: string,
): Promise<EvaluationCheck> {
  try {
    return await task.check(results, workspaceDirectory)
  } catch (error) {
    return { detail: `grader raised: ${errorName(error)}`, ok: false }
  }
}

async function invoke(
  evaluator: WorkspaceModuleEvaluator,
  workspaceDirectory: string,
  modulePath: string,
  exportName: string,
  args: readonly unknown[],
): Promise<unknown> {
  try {
    return await evaluator.invoke({ args, exportName, modulePath, workspaceDirectory })
  } catch (error) {
    return `<${errorName(error)}>`
  }
}

function lastResult(results: readonly EvaluationTurnResult[]): EvaluationTurnResult {
  const result = results.at(-1)
  if (result === undefined) throw new Error('task must have at least one turn result')
  return result
}

function totalLatency(items: readonly { readonly latencyMs: number }[]): number {
  return items.reduce((total, item) => total + item.latencyMs, 0)
}

function requiredDirectory(value: string, label: string): string {
  const directory = value.trim()
  if (!directory) throw new Error(`${label} must not be empty`)
  return resolve(directory)
}

function pad(value: string, width: number): string {
  return value.padEnd(width)
}

function seconds(milliseconds: number): string {
  return `${(milliseconds / 1_000).toFixed(1)}s`
}

function percentage(passed: number, total: number): number {
  return Math.floor((100 * passed) / Math.max(1, total))
}

function errorName(error: unknown): string {
  return error instanceof Error && error.name ? error.name : 'Error'
}
