// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  createBasicGrpoTrainingRequest,
  describeBasicGrpoTrainingRequest,
  runBasicGrpoTraining,
} from './basicGrpoTraining.js'
import { loadGsm8kTrainingDataset, mapGsm8kExamples, parseGsm8kJsonl } from './dataset.js'
import type { GrpoDatasetExample, GrpoTrainingDependencies, Gsm8kDatasetPort } from './types.js'

/** Usage text for the safe native Bun GRPO template CLI. */
export const GRPO_TEMPLATE_CLI_USAGE = `Usage: xerxes skill grpo-rl-training (--dry-run | --run) [options]

Commands:
  --dry-run [--dataset <gsm8k.jsonl>]  Validate data and print a host-execution plan.
  --run --dataset <gsm8k.jsonl>        Run only when explicit host ports are injected programmatically.
  --run --gsm8k [--split <name>]       Use an injected GSM8K dataset port.
  --help                               Print this help.

This Bun template does not install or invoke Python, PyTorch, Transformers, or TRL.
Actual model loading, accelerator allocation, optimization, and artifact storage belong
to explicitly injected Bun/TypeScript host provider and storage ports.`

const MISSING_HOST_PORTS_MESSAGE =
  'Cannot start GRPO training: inject explicit host accelerator and storage ports. '
  + 'Use --dry-run to inspect the request.'

export interface GrpoTemplateCliDependencies {
  readonly dataset?: Gsm8kDatasetPort
  readonly readTextFile?: (path: string) => string | Promise<string>
  readonly training?: GrpoTrainingDependencies
  readonly writeLine?: (line: string) => void | Promise<void>
}

/** Run the safe CLI with explicit dependencies; usable from a Bun host integration or tests. */
export async function runGrpoTemplateCli(
  args: readonly string[],
  dependencies: GrpoTemplateCliDependencies = {},
): Promise<number> {
  const writeLine = dependencies.writeLine ?? (line => console.log(line))
  let parsed: GrpoTemplateCliArgs
  try {
    parsed = parseArgs(args)
  } catch (error) {
    await writeLine(`${errorMessage(error)}\n\n${GRPO_TEMPLATE_CLI_USAGE}`)
    return 2
  }
  if (parsed.help) {
    await writeLine(GRPO_TEMPLATE_CLI_USAGE)
    return 0
  }

  try {
    const dataset = await loadDataset(parsed, dependencies)
    const request = createBasicGrpoTrainingRequest(dataset)
    if (parsed.dryRun) {
      await writeLine(JSON.stringify(describeBasicGrpoTrainingRequest(request), null, 2))
      return 0
    }
    if (dependencies.training === undefined) {
      await writeLine(MISSING_HOST_PORTS_MESSAGE)
      return 2
    }
    const result = await runBasicGrpoTraining(request, dependencies.training)
    await writeLine(JSON.stringify({ kind: 'xerxes.grpo-training-result.v1', ...result }, null, 2))
    return 0
  } catch (error) {
    await writeLine(`GRPO template error: ${errorMessage(error)}`)
    return 1
  }
}

/** Standalone Bun entry point: dry-run planning is safe; execution needs caller-owned ports. */
export async function main(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  return runGrpoTemplateCli(args, {
    readTextFile: path => Bun.file(path).text(),
  })
}

interface GrpoTemplateCliArgs {
  readonly datasetPath?: string
  readonly dryRun: boolean
  readonly help: boolean
  readonly split: string
  readonly useGsm8k: boolean
}

function parseArgs(args: readonly string[]): GrpoTemplateCliArgs {
  let datasetPath: string | undefined
  let dryRun = false
  let help = false
  let run = false
  let split = 'train'
  let useGsm8k = false
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === '--help' || argument === '-h') help = true
    else if (argument === '--dry-run') dryRun = true
    else if (argument === '--run') run = true
    else if (argument === '--gsm8k') useGsm8k = true
    else if (argument === '--dataset') {
      datasetPath = requiredArgument(args[index + 1], '--dataset')
      index += 1
    } else if (argument === '--split') {
      split = requiredArgument(args[index + 1], '--split')
      index += 1
    } else throw new TypeError(`unknown argument: ${String(argument)}`)
  }
  if (help) return { help, dryRun: false, split, useGsm8k: false }
  if (dryRun === run) throw new TypeError('provide exactly one of --dry-run or --run')
  if (datasetPath !== undefined && useGsm8k) throw new TypeError('choose either --dataset or --gsm8k, not both')
  if (datasetPath === undefined && !useGsm8k) throw new TypeError('provide --dataset <gsm8k.jsonl> or --gsm8k')
  return {
    ...(datasetPath === undefined ? {} : { datasetPath }),
    dryRun,
    help: false,
    split,
    useGsm8k,
  }
}

async function loadDataset(
  args: GrpoTemplateCliArgs,
  dependencies: GrpoTemplateCliDependencies,
): Promise<readonly GrpoDatasetExample[]> {
  if (args.datasetPath !== undefined) {
    if (dependencies.readTextFile === undefined) {
      throw new TypeError('a --dataset path requires an explicit text-file reader')
    }
    return mapGsm8kExamples(parseGsm8kJsonl(await dependencies.readTextFile(args.datasetPath)))
  }
  if (dependencies.dataset === undefined) {
    throw new TypeError('--gsm8k requires an explicit caller-owned GSM8K dataset port')
  }
  return loadGsm8kTrainingDataset(dependencies.dataset, args.split)
}

function requiredArgument(value: string | undefined, name: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${name} requires a non-empty value`)
  return value.trim()
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

if (import.meta.main) {
  process.exitCode = await main()
}
