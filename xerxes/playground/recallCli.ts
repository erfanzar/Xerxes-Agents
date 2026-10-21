// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { createEvaluationIsolation, type EvaluationIsolation } from './isolation.js'
import {
  createRecallScenario, formatRecallPlan, formatRecallReport, runRecallEvaluation,
  DEFAULT_RECALL_FACTS, DEFAULT_RECALL_FILLER_TOKENS, DEFAULT_RECALL_SEED,
} from './recall.js'
import type { EvaluationSessionPort, EvaluationSessionPortFactory } from './types.js'

export const RECALL_CLI_USAGE = [
  'Usage: bun run --cwd xerxes playground:recall -- --transport <module> [options]',
  '', 'Options:',
  '  --transport <module>   Bun module exporting createEvaluationSessionPort(context).',
  `  --facts N              Number of early synthetic facts (1–1000, default ${DEFAULT_RECALL_FACTS}).`,
  `  --filler-tokens N      Filler prompt volume, estimated characters / 4 (0–2000000, default ${DEFAULT_RECALL_FILLER_TOKENS}).`,
  `  --seed N               Deterministic unsigned 32-bit seed (default ${DEFAULT_RECALL_SEED}).`,
  '  --run-root <dir>       Parent directory for private evaluation runs.',
  '  --dry-run              Print the scenario plan without importing any transport or creating files.',
  '  -h, --help             Show this help.',
  '', 'One session is retained through planting, filler and recall; no forced resets or compactions.',
  'Increase filler volume for larger context windows; a filler estimate is not a compaction count.',
  'Optional transport.getCompactionCount() measures compactions; otherwise the report says unknown.',
  'The explicit transport owns runtime/provider configuration and must honor submit signals and close.',
  'No provider client is created and no credentials are discovered by this command.',
].join('\n')

export interface RecallCliOptions {
  readonly help: boolean
  readonly dryRun: boolean
  readonly transportModule: string | undefined
  readonly facts: number
  readonly fillerTokens: number
  readonly seed: number
  readonly runRoot: string
}

export interface RecallCliDependencies {
  readonly cwd?: string
  readonly signal?: AbortSignal
  readonly createIsolation?: (options: Parameters<typeof createEvaluationIsolation>[0]) => Promise<EvaluationIsolation>
  readonly createSessionPort?: EvaluationSessionPortFactory
  readonly write?: (text: string) => void | Promise<void>
  readonly writeError?: (text: string) => void | Promise<void>
}

/** Parse and bound all spending sizes before loading a transport or creating private run files. */
export function parseRecallCliOptions(args: readonly string[], cwd = process.cwd()): RecallCliOptions {
  let help = false
  let dryRun = false
  let transportModule: string | undefined
  let facts = DEFAULT_RECALL_FACTS
  let fillerTokens = DEFAULT_RECALL_FILLER_TOKENS
  let seed = DEFAULT_RECALL_SEED
  let runRoot = resolve(cwd, '.xerxes', 'evaluations')
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]!
    if (argument === '--') continue
    if (argument === '--help' || argument === '-h') { help = true; continue }
    if (argument === '--dry-run') { dryRun = true; continue }
    if (!['--transport', '--facts', '--filler-tokens', '--seed', '--run-root'].includes(argument)) throw new TypeError(`unknown argument: ${argument}`)
    const value = args[index + 1]
    if (value === undefined || !value.trim() || value.startsWith('-')) throw new TypeError(`${argument} requires a non-empty value`)
    switch (argument) {
      case '--transport': transportModule = resolve(cwd, value); break
      case '--run-root': runRoot = resolve(cwd, value); break
      case '--facts': facts = integer(value, 'facts', 1, 1_000); break
      case '--filler-tokens': fillerTokens = integer(value, 'fillerTokens', 0, 2_000_000); break
      case '--seed': seed = integer(value, 'seed', 0, 0xffff_ffff); break
    }
    index += 1
  }
  return { help, dryRun, transportModule, facts, fillerTokens, seed, runRoot }
}

/** Process-style entry with injectable offline dependencies and an explicit transport factory. */
export async function runRecallCli(args: readonly string[], dependencies: RecallCliDependencies = {}): Promise<number> {
  const write = dependencies.write ?? ((text: string) => { process.stdout.write(text) })
  const writeError = dependencies.writeError ?? ((text: string) => { process.stderr.write(text) })
  let options: RecallCliOptions
  try { options = parseRecallCliOptions(args, dependencies.cwd ?? process.cwd()) } catch (error) {
    await writeError(`recall eval: ${errorMessage(error)}\n\n${RECALL_CLI_USAGE}\n`)
    return 2
  }
  if (options.help) { await write(`${RECALL_CLI_USAGE}\n`); return 0 }
  if (!options.dryRun && options.transportModule === undefined && dependencies.createSessionPort === undefined) {
    await writeError(`recall eval: --transport <module> is required for execution; supply an explicit transport or use --dry-run\n\n${RECALL_CLI_USAGE}\n`)
    return 2
  }
  let isolation: EvaluationIsolation | undefined
  const controller = new AbortController()
  const cancel = () => controller.abort()
  const signal = dependencies.signal === undefined ? controller.signal : AbortSignal.any([controller.signal, dependencies.signal])
  try {
    const scenario = createRecallScenario({ seed: options.seed, facts: options.facts, fillerTokens: options.fillerTokens })
    await write(`${formatRecallPlan(scenario)}\n`)
    if (options.dryRun) return 0
    process.once('SIGINT', cancel)
    signal.throwIfAborted()
    isolation = await (dependencies.createIsolation ?? createEvaluationIsolation)({ rootDirectory: options.runRoot })
    signal.throwIfAborted()
    const factory = dependencies.createSessionPort ?? await loadSessionPortFactory(options.transportModule!)
    signal.throwIfAborted()
    const transport = await factory({
      homeDirectory: isolation.homeDirectory, runDirectory: isolation.runDirectory, workspaceDirectory: isolation.workspaceDirectory,
    })
    assertSessionPort(transport)
    const report = await runRecallEvaluation({
      scenario, transport, signal,
      start: { homeDirectory: isolation.homeDirectory, workspaceDirectory: isolation.workspaceDirectory, permissionMode: 'accept-all' },
    })
    await write(`${formatRecallReport(report)}\n`)
    return report.rows.every(row => row.ok) ? 0 : 1
  } catch (error) {
    await writeError(`recall eval: ${errorMessage(error)}\n`)
    return signal.aborted ? 130 : 1
  } finally {
    process.removeListener('SIGINT', cancel)
    if (isolation !== undefined) await isolation.cleanup()
  }
}

export async function recallMain(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  return runRecallCli(args)
}

async function loadSessionPortFactory(modulePath: string): Promise<EvaluationSessionPortFactory> {
  const loaded: unknown = await import(pathToFileURL(modulePath).href)
  if (!isRecord(loaded) || typeof loaded.createEvaluationSessionPort !== 'function') throw new TypeError(`${modulePath} must export createEvaluationSessionPort(context)`)
  return loaded.createEvaluationSessionPort as EvaluationSessionPortFactory
}

function assertSessionPort(value: unknown): asserts value is EvaluationSessionPort {
  if (!isRecord(value) || !['approve', 'close', 'reset', 'start', 'submit'].every(method => typeof value[method] === 'function')
    || (value.getCompactionCount !== undefined && typeof value.getCompactionCount !== 'function')) {
    throw new TypeError('createEvaluationSessionPort(context) must return an EvaluationSessionPort')
  }
}

function integer(value: string, name: string, minimum: number, maximum: number): number {
  const number = Number(value)
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(number) || number < minimum || number > maximum) throw new TypeError(`${name} must be an integer from ${minimum} to ${maximum}`)
  return number
}
function isRecord(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value) }
function errorMessage(error: unknown): string { return error instanceof Error ? error.message : String(error) }

if (import.meta.main) process.exitCode = await recallMain()
