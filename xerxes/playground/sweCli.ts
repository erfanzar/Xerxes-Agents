// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolve } from 'node:path'

import type { EvaluationIsolation } from './isolation.js'
import { formatSweReport } from './sweMetrics.js'
import {
  createNativeSweTasks,
  runSweBenchmark,
  type NativeSweTask,
  type SweTaskSolver,
} from './sweTasks.js'
import type { EvaluationSessionPortFactory } from './types.js'

export const SWE_EVAL_CLI_USAGE = [
  'Usage: bun playground/sweCli.ts --transport <module> [options]',
  '',
  'Options:',
  '  -v, --verbose             Print each completed task row as the benchmark runs.',
  '  -k, --keyword <text>      Run only SWE task IDs or categories containing <text>.',
  '  --json                    Print the structured JSON report instead of the text scorecard.',
  '  --transport <module>      Bun module exporting createEvaluationSessionPort(context).',
  '  --profile-dir <directory> Copy only this directory\'s profiles.json into the private run home.',
  '  --run-root <directory>    Parent directory for isolated per-task homes and workspaces.',
  '  -h, --help                Print this help.',
  '',
  'The SWE suite provisions each multi-file TypeScript bugfix into an isolated workspace',
  'with visible failing reproduction tests, withholds hidden holdout regression tests until',
  'post-turn grading, and computes all 5 Core SWE KPIs.',
].join('\n')

export interface SweCliOptions {
  readonly help: boolean
  readonly json: boolean
  readonly keyword: string | undefined
  readonly profileSourceDirectory: string | undefined
  readonly runRoot: string
  readonly transportModule: string | undefined
  readonly verbose: boolean
}

export interface SweCliDependencies {
  readonly createIsolation?: (options: {
    readonly rootDirectory: string
    readonly profileSourceDirectory?: string
    readonly runId?: string
  }) => Promise<EvaluationIsolation>
  readonly createSessionPort?: EvaluationSessionPortFactory
  readonly solver?: SweTaskSolver
  readonly tasks?: readonly NativeSweTask[]
  readonly write?: (text: string) => void | Promise<void>
  readonly writeError?: (text: string) => void | Promise<void>
}

export function parseSweCliOptions(args: readonly string[]): SweCliOptions {
  let help = false
  let json = false
  let keyword: string | undefined
  let profileSourceDirectory: string | undefined
  let runRoot = resolve(process.cwd(), '.xerxes', 'swe-evaluations')
  let transportModule: string | undefined
  let verbose = false

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]
    if (arg === '--help' || arg === '-h') {
      help = true
      continue
    }
    if (arg === '--json') {
      json = true
      continue
    }
    if (arg === '--verbose' || arg === '-v') {
      verbose = true
      continue
    }
    if (arg === '--keyword' || arg === '-k') {
      keyword = requiredArg(args[index + 1], arg)
      index += 1
      continue
    }
    if (arg === '--profile-dir') {
      profileSourceDirectory = requiredArg(args[index + 1], arg)
      index += 1
      continue
    }
    if (arg === '--run-root') {
      runRoot = resolve(requiredArg(args[index + 1], arg))
      index += 1
      continue
    }
    if (arg === '--transport') {
      transportModule = requiredArg(args[index + 1], arg)
      index += 1
      continue
    }
    throw new TypeError(`unknown argument: ${String(arg)}`)
  }

  return {
    help,
    json,
    keyword,
    profileSourceDirectory,
    runRoot,
    transportModule,
    verbose,
  }
}

export async function runSweCli(
  args: readonly string[],
  dependencies: SweCliDependencies = {},
): Promise<number> {
  const write = dependencies.write ?? (text => process.stdout.write(text))
  const writeError = dependencies.writeError ?? (text => process.stderr.write(text))

  let options: SweCliOptions
  try {
    options = parseSweCliOptions(args)
  } catch (error) {
    await writeError(`swe eval: ${errorMessage(error)}\n\n${SWE_EVAL_CLI_USAGE}\n`)
    return 2
  }

  if (options.help) {
    await write(`${SWE_EVAL_CLI_USAGE}\n`)
    return 0
  }

  if (
    dependencies.solver === undefined
    && dependencies.createSessionPort === undefined
    && options.transportModule === undefined
  ) {
    await writeError(`swe eval: --transport <module> is required\n\n${SWE_EVAL_CLI_USAGE}\n`)
    return 2
  }

  try {
    const report = await runSweBenchmark({
      runRoot: options.runRoot,
      tasks: dependencies.tasks ?? createNativeSweTasks(),
      ...(options.keyword === undefined ? {} : { keyword: options.keyword }),
      ...(options.profileSourceDirectory === undefined
        ? {}
        : { profileSourceDirectory: options.profileSourceDirectory }),
      ...(options.transportModule === undefined ? {} : { transportModule: options.transportModule }),
      ...(dependencies.createSessionPort === undefined
        ? {}
        : { createSessionPort: dependencies.createSessionPort }),
      ...(dependencies.solver === undefined ? {} : { solver: dependencies.solver }),
      ...(dependencies.createIsolation === undefined ? {} : { createIsolation: dependencies.createIsolation }),
      ...(options.verbose && !options.json
        ? {
            onTaskComplete: async (record) => {
              await write(
                `  · [${record.resolved ? 'PASS' : 'FAIL'}] ${record.taskId} (${record.wallMs}ms) — ${record.detail}\n`,
              )
            },
          }
        : {}),
    })

    if (options.json) {
      await write(`${JSON.stringify(report, null, 2)}\n`)
    } else {
      await write(`${formatSweReport(report)}\n`)
    }
    return report.metrics.resolvedCount === report.metrics.totalTasks ? 0 : 1
  } catch (error) {
    await writeError(`swe eval: ${errorMessage(error)}\n`)
    return 1
  }
}

export async function sweMain(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  return runSweCli(args)
}

function requiredArg(value: string | undefined, flag: string): string {
  if (typeof value !== 'string' || !value.trim() || value.startsWith('-')) {
    throw new TypeError(`${flag} requires a non-empty value`)
  }
  return value.trim()
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

if (import.meta.main) process.exitCode = await sweMain()
