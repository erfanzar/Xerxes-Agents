// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

import { BunWorkspaceModuleEvaluator } from './evaluator.js'
import { NativeEvaluationAgent } from './harness.js'
import { createEvaluationIsolation, type EvaluationIsolation } from './isolation.js'
import type {
  EvaluationSessionPort,
  EvaluationSessionPortFactory,
  EvaluationTransportContext,
} from './types.js'
import { formatWarmupReport, runWarmupSuite, type WarmupStepResult } from './warmup.js'

export const WARMUP_EVAL_CLI_USAGE = [
  'Usage: bun playground/cli.ts --transport <module> [options]',
  '',
  'Options:',
  '  -v, --verbose             Print every prompt and normalized reply while the suite runs.',
  '  -k, --keyword <text>      Run only warm-up task names containing <text>.',
  '  --transport <module>      Bun module exporting createEvaluationSessionPort(context).',
  '  --profile-dir <directory> Copy only this directory\'s profiles.json into the private run home.',
  '  --run-root <directory>    Parent directory for isolated run homes and workspaces.',
  '  -h, --help                Print this help.',
  '',
  'The transport module owns runtime/provider construction. Its factory receives:',
  '  { homeDirectory, runDirectory, workspaceDirectory }',
  'and must return an EvaluationSessionPort. The CLI never starts Python, changes',
  'XERXES_HOME, or discovers provider credentials on its own.',
].join('\n')

export interface WarmupCliOptions {
  readonly help: boolean
  readonly keyword: string | undefined
  readonly profileSourceDirectory: string | undefined
  readonly runRoot: string
  readonly transportModule: string | undefined
  readonly verbose: boolean
}

/** Injectable boundaries for test hosts and embedding applications. */
export interface WarmupCliDependencies {
  readonly createIsolation?: (options: Parameters<typeof createEvaluationIsolation>[0]) => Promise<EvaluationIsolation>
  readonly createSessionPort?: EvaluationSessionPortFactory
  readonly write?: (text: string) => void | Promise<void>
  readonly writeError?: (text: string) => void | Promise<void>
}

/** Parse standalone warm-up CLI arguments without loading a runtime or provider. */
export function parseWarmupCliOptions(args: readonly string[]): WarmupCliOptions {
  let help = false
  let keyword: string | undefined
  let profileSourceDirectory: string | undefined
  let runRoot = resolve(process.cwd(), '.xerxes', 'evaluations')
  let transportModule: string | undefined
  let verbose = false

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === '--help' || argument === '-h') {
      help = true
      continue
    }
    if (argument === '--verbose' || argument === '-v') {
      verbose = true
      continue
    }
    if (argument === '--keyword' || argument === '-k') {
      keyword = requiredArgument(args[index + 1], argument)
      index += 1
      continue
    }
    if (argument === '--profile-dir') {
      profileSourceDirectory = requiredArgument(args[index + 1], argument)
      index += 1
      continue
    }
    if (argument === '--run-root') {
      runRoot = resolve(requiredArgument(args[index + 1], argument))
      index += 1
      continue
    }
    if (argument === '--transport') {
      transportModule = requiredArgument(args[index + 1], argument)
      index += 1
      continue
    }
    throw new TypeError(`unknown argument: ${String(argument)}`)
  }

  return { help, keyword, profileSourceDirectory, runRoot, transportModule, verbose }
}

/**
 * Run the normal eight-task native warm-up suite and return a process-style exit code.
 *
 * A host-injected factory is useful for tests; standalone use loads the module supplied
 * by `--transport`. The native agent still performs retries, telemetry collection,
 * session resets, behavioral grading, and deterministic reporting itself.
 */
export async function runWarmupCli(
  args: readonly string[],
  dependencies: WarmupCliDependencies = {},
): Promise<number> {
  const write = dependencies.write ?? (text => process.stdout.write(text))
  const writeError = dependencies.writeError ?? (text => process.stderr.write(text))
  let options: WarmupCliOptions
  try {
    options = parseWarmupCliOptions(args)
  } catch (error) {
    await writeError(`warmup eval: ${errorMessage(error)}\n\n${WARMUP_EVAL_CLI_USAGE}\n`)
    return 2
  }

  if (options.help) {
    await write(`${WARMUP_EVAL_CLI_USAGE}\n`)
    return 0
  }
  if (dependencies.createSessionPort === undefined && options.transportModule === undefined) {
    await writeError(`warmup eval: --transport <module> is required\n\n${WARMUP_EVAL_CLI_USAGE}\n`)
    return 2
  }

  const createIsolation = dependencies.createIsolation ?? createEvaluationIsolation
  let isolation: EvaluationIsolation | undefined
  try {
    isolation = await createIsolation({
      rootDirectory: options.runRoot,
      ...(options.profileSourceDirectory === undefined ? {} : { profileSourceDirectory: options.profileSourceDirectory }),
    })
    const context: EvaluationTransportContext = {
      homeDirectory: isolation.homeDirectory,
      runDirectory: isolation.runDirectory,
      workspaceDirectory: isolation.workspaceDirectory,
    }
    const factory = dependencies.createSessionPort ?? await loadSessionPortFactory(options.transportModule as string)
    const transport = await factory(context)
    assertSessionPort(transport)
    const agent = new NativeEvaluationAgent({
      start: {
        homeDirectory: context.homeDirectory,
        permissionMode: 'accept-all',
        workspaceDirectory: context.workspaceDirectory,
      },
      transport,
    })
    const report = await runWarmupSuite({
      agent,
      ...(options.keyword === undefined ? {} : { keyword: options.keyword }),
      moduleEvaluator: new BunWorkspaceModuleEvaluator(),
      ...(options.verbose ? { onStep: async step => { await write(formatVerboseStep(step)) } } : {}),
      sandboxDirectory: context.workspaceDirectory,
    })
    await write(formatWarmupReport(report))
    return report.rows.every(row => row.ok) ? 0 : 1
  } catch (error) {
    await writeError(`warmup eval: ${errorMessage(error)}\n`)
    return 1
  } finally {
    if (isolation !== undefined) await isolation.cleanup()
  }
}

/** Standalone Bun entry point. */
export async function main(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  return runWarmupCli(args)
}

async function loadSessionPortFactory(modulePath: string): Promise<EvaluationSessionPortFactory> {
  const absolutePath = resolve(process.cwd(), requiredArgument(modulePath, '--transport'))
  const loaded: unknown = await import(pathToFileURL(absolutePath).href)
  if (!isRecord(loaded) || typeof loaded.createEvaluationSessionPort !== 'function') {
    throw new TypeError(`${absolutePath} must export createEvaluationSessionPort(context)`)
  }
  return loaded.createEvaluationSessionPort as EvaluationSessionPortFactory
}

function assertSessionPort(value: unknown): asserts value is EvaluationSessionPort {
  if (!isRecord(value)
    || typeof value.approve !== 'function'
    || typeof value.close !== 'function'
    || typeof value.reset !== 'function'
    || typeof value.start !== 'function'
    || typeof value.submit !== 'function') {
    throw new TypeError('createEvaluationSessionPort(context) must return an EvaluationSessionPort')
  }
}

function formatVerboseStep(step: WarmupStepResult): string {
  return [
    `    · [${step.task.name}] ${JSON.stringify(step.prompt.slice(0, 60))}`,
    `        -> ${JSON.stringify(step.result.text.slice(0, 120))}  (tools=${JSON.stringify(step.result.tools)}, ${(step.result.latencyMs / 1_000).toFixed(1)}s)`,
  ].join('\n') + '\n'
}

function requiredArgument(value: string | undefined, name: string): string {
  if (typeof value !== 'string' || !value.trim() || value.startsWith('-')) {
    throw new TypeError(`${name} requires a non-empty value`)
  }
  return value.trim()
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

if (import.meta.main) process.exitCode = await main()
