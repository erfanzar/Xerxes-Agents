// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmod, lstat, mkdir, mkdtemp, readFile, realpath, rm, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { NativeEvaluationAgent } from './harness.js'
import { createNativeHardTasks } from './hardTasks.js'
import type { NativeHardTask } from './hard.js'
import { createEvaluationIsolation, type EvaluationIsolation } from './isolation.js'
import {
  formatImprovementPlan, formatImprovementReport, planImprovement, runImprovement,
  type CandidateWorkspacePort, type HarnessCandidate, type HarnessEdit, type ImproveBudget,
} from './improve.js'
import type { EvaluationSessionPort, EvaluationSessionPortFactory } from './types.js'

export const IMPROVE_CLI_USAGE = [
  'Usage: bun run --cwd xerxes playground:improve -- --transport <module> --proposals <json> --max-candidates N --max-task-runs N [options]',
  '', 'Options:',
  '  --transport <module>   Bun module exporting createEvaluationSessionPort(context).',
  '  --proposals <json>     JSON array of HarnessCandidate objects or HarnessEdit objects.',
  '  --max-candidates N     Positive ceiling, excluding baseline, including conditional composition.',
  '  --max-task-runs N      Positive ceiling for all baseline, individual, and composed task runs.',
  '  --max-tokens N         Optional ceiling on reported contextTokens (not billing usage).',
  '  --max-wall-ms N        Optional wall-clock deadline in milliseconds.',
  '  -k, --keyword <text>   Select the same hard task names for every candidate.',
  '  --run-root <dir>       Parent directory for private evaluation runs.',
  '  --dry-run              Print the declared plan; never import a transport or create a runtime.',
  '  -h, --help             Show this help.',
  '', 'Edit format: { id, scope: "INV"|"DOM", target: "agent_yaml"|"skill"|"prompt_addendum",',
  '  description, apply: { path: "relative/harness/file", content: "full replacement" } }',
  'Harness paths resolve against the current directory. Only referenced files are copied.',
  'The explicit transport receives a separate candidate workspace and home; it must load',
  'that workspace\'s harness files and honor cancellation/close. No credentials are discovered.',
  'This command never applies recommendations to the source repository.',
].join('\n')

export interface ImproveCliOptions {
  readonly help: boolean
  readonly dryRun: boolean
  readonly transportModule: string | undefined
  readonly proposalsFile: string | undefined
  readonly maxCandidates: number | undefined
  readonly maxTaskRuns: number | undefined
  readonly maxTokens: number | undefined
  readonly maxWallMs: number | undefined
  readonly keyword: string | undefined
  readonly runRoot: string
}

export interface ImproveCliDependencies {
  readonly cwd?: string
  readonly createIsolation?: (options: Parameters<typeof createEvaluationIsolation>[0]) => Promise<EvaluationIsolation>
  readonly createSessionPort?: EvaluationSessionPortFactory
  readonly tasks?: readonly NativeHardTask[]
  readonly signal?: AbortSignal
  readonly write?: (text: string) => void | Promise<void>
  readonly writeError?: (text: string) => void | Promise<void>
}

export function parseImproveCliOptions(args: readonly string[], cwd = process.cwd()): ImproveCliOptions {
  let help = false
  let dryRun = false
  let transportModule: string | undefined
  let proposalsFile: string | undefined
  let maxCandidates: number | undefined
  let maxTaskRuns: number | undefined
  let maxTokens: number | undefined
  let maxWallMs: number | undefined
  let keyword: string | undefined
  let runRoot = resolve(cwd, '.xerxes', 'evaluations')
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]!
    if (argument === '--') continue
    if (argument === '--help' || argument === '-h') { help = true; continue }
    if (argument === '--dry-run') { dryRun = true; continue }
    const value = requiredArgument(args[index + 1], argument)
    switch (argument) {
      case '--transport': transportModule = resolve(cwd, value); break
      case '--proposals': proposalsFile = resolve(cwd, value); break
      case '--max-candidates': maxCandidates = positiveNumber(value, argument); break
      case '--max-task-runs': maxTaskRuns = positiveNumber(value, argument); break
      case '--max-tokens': maxTokens = positiveNumber(value, argument); break
      case '--max-wall-ms': maxWallMs = positiveNumber(value, argument); break
      case '--keyword': case '-k': keyword = value; break
      case '--run-root': runRoot = resolve(cwd, value); break
      default: throw new TypeError(`unknown argument: ${argument}`)
    }
    index += 1
  }
  return { help, dryRun, transportModule, proposalsFile, maxCandidates, maxTaskRuns, maxTokens, maxWallMs, keyword, runRoot }
}

/** Exported process-style CLI entry for offline tests and embedding, with no credential discovery. */
export async function runImproveCli(args: readonly string[], dependencies: ImproveCliDependencies = {}): Promise<number> {
  const write = dependencies.write ?? ((text: string) => { process.stdout.write(text) })
  const writeError = dependencies.writeError ?? ((text: string) => { process.stderr.write(text) })
  const cwd = resolve(dependencies.cwd ?? process.cwd())
  let options: ImproveCliOptions
  try { options = parseImproveCliOptions(args, cwd) } catch (error) {
    await writeError(`harness improve: ${errorMessage(error)}\n\n${IMPROVE_CLI_USAGE}\n`)
    return 2
  }
  if (options.help) { await write(`${IMPROVE_CLI_USAGE}\n`); return 0 }
  if (!options.dryRun && options.transportModule === undefined && dependencies.createSessionPort === undefined) {
    await writeError(`harness improve: --transport <module> is required for execution; supply an explicit transport or use --dry-run\n\n${IMPROVE_CLI_USAGE}\n`)
    return 2
  }
  if (options.proposalsFile === undefined || options.maxCandidates === undefined || options.maxTaskRuns === undefined) {
    await writeError(`harness improve: --proposals, --max-candidates and --max-task-runs are required; declare the budget before running\n\n${IMPROVE_CLI_USAGE}\n`)
    return 2
  }
  let isolation: EvaluationIsolation | undefined
  try {
    const proposals = parseProposals(JSON.parse(await readFile(options.proposalsFile, 'utf8')))
    const tasks = (dependencies.tasks ?? createNativeHardTasks()).filter(task => options.keyword === undefined || task.name.includes(options.keyword))
    const budget: ImproveBudget = {
      maxCandidates: options.maxCandidates, maxTaskRuns: options.maxTaskRuns,
      ...(options.maxTokens === undefined ? {} : { maxTokens: options.maxTokens }),
      ...(options.maxWallMs === undefined ? {} : { maxWallMs: options.maxWallMs }),
    }
    const plan = planImprovement({ baseline: { id: 'baseline', edits: [] }, proposals, tasks, budget })
    await write(`${formatImprovementPlan(plan)}\n`)
    if (options.dryRun) return 0
    // Snapshot and validate all harness files before importing any provider transport.
    const snapshot = await snapshotHarness(cwd, plan.candidates.flatMap(candidate => candidate.edits))
    isolation = await (dependencies.createIsolation ?? createEvaluationIsolation)({ rootDirectory: options.runRoot })
    await chmod(isolation.runDirectory, 0o700)
    const factory = dependencies.createSessionPort ?? await loadSessionPortFactory(options.transportModule!)
    const workspaces = copyWorkspaces(isolation.workspaceDirectory, snapshot)
    const controller = new AbortController()
    const cancel = () => controller.abort()
    process.once('SIGINT', cancel)
    try {
      const result = await runImprovement(plan, {
        workspaces, tasks,
        signal: dependencies.signal === undefined ? controller.signal : AbortSignal.any([controller.signal, dependencies.signal]),
        agentFor: async directory => {
          const homeDirectory = join(directory, '.improve-home')
          await mkdir(homeDirectory, { mode: 0o700 })
          const transport = await factory({ homeDirectory, runDirectory: directory, workspaceDirectory: directory })
          assertSessionPort(transport)
          return new NativeEvaluationAgent({
            start: { homeDirectory, permissionMode: 'accept-all', workspaceDirectory: directory }, transport,
          })
        },
      })
      await write(`${formatImprovementReport(result)}\n`)
      return result.status === 'completed' ? 0 : result.status === 'cancelled' ? 130 : 1
    } finally { process.removeListener('SIGINT', cancel) }
  } catch (error) {
    await writeError(`harness improve: ${errorMessage(error)}\n`)
    return 1
  } finally { if (isolation !== undefined) await isolation.cleanup() }
}

export async function improveMain(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  return runImproveCli(args)
}

function copyWorkspaces(root: string, snapshot: ReadonlyMap<string, string>): CandidateWorkspacePort {
  return {
    create: async (_candidateId, edits) => {
      const directory = await mkdtemp(join(root, 'candidate-'))
      try {
        for (const [path, content] of snapshot) await writeCandidateFile(directory, path, content)
        for (const edit of edits) await writeCandidateFile(directory, edit.apply.path, edit.apply.content)
        return { directory, dispose: () => rm(directory, { recursive: true, force: true }) }
      } catch (error) {
        await rm(directory, { recursive: true, force: true })
        throw error
      }
    },
  }
}

async function snapshotHarness(sourceDirectory: string, edits: readonly HarnessEdit[]): Promise<ReadonlyMap<string, string>> {
  const root = await realpath(sourceDirectory)
  const snapshot = new Map<string, string>()
  for (const path of new Set(edits.map(edit => edit.apply.path))) {
    const source = childPath(root, path)
    let current = root
    let missing = false
    for (const part of relative(root, source).split(sep)) {
      current = join(current, part)
      try {
        const info = await lstat(current)
        if (info.isSymbolicLink()) throw new Error(`harness file symlink is not allowed: ${path}`)
      } catch (error) {
        if (isMissing(error)) { missing = true; break }
        throw error
      }
    }
    // A missing target is a new file, absent from baseline and created by its edit.
    if (missing) continue
    const canonical = await realpath(source)
    childPath(root, relative(root, canonical))
    snapshot.set(path, await readFile(canonical, 'utf8'))
  }
  return snapshot
}

async function writeCandidateFile(directory: string, path: string, content: string): Promise<void> {
  const destination = childPath(directory, path)
  await mkdir(dirname(destination), { recursive: true, mode: 0o700 })
  await writeFile(destination, content, { encoding: 'utf8', mode: 0o600 })
}

function childPath(root: string, path: string): string {
  const target = resolve(root, path)
  const child = relative(root, target)
  if (!child || child === '..' || child.startsWith(`..${sep}`) || isAbsolute(child) || isAbsolute(path)) throw new Error(`harness path must remain inside its candidate directory: ${path}`)
  return target
}

function parseProposals(value: unknown): HarnessCandidate[] | HarnessEdit[] {
  if (!Array.isArray(value)) throw new Error('--proposals JSON must be an array of HarnessCandidate or HarnessEdit objects')
  if (value.every(item => isRecord(item) && 'edits' in item)) {
    return value.map(item => {
      if (!isRecord(item) || typeof item.id !== 'string' || !Array.isArray(item.edits)) throw new Error('each candidate needs id and edits[]')
      return { id: item.id, edits: item.edits.map(parseEdit) }
    })
  }
  return value.map(parseEdit)
}

function parseEdit(value: unknown): HarnessEdit {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.description !== 'string' || (value.scope !== 'INV' && value.scope !== 'DOM') || (value.target !== 'agent_yaml' && value.target !== 'skill' && value.target !== 'prompt_addendum') || !isRecord(value.apply) || typeof value.apply.path !== 'string' || typeof value.apply.content !== 'string') {
    throw new Error('each edit requires id, scope INV|DOM, target agent_yaml|skill|prompt_addendum, description, and apply: { path, content }')
  }
  return { id: value.id, scope: value.scope, target: value.target, description: value.description, apply: { path: value.apply.path, content: value.apply.content } }
}

async function loadSessionPortFactory(modulePath: string): Promise<EvaluationSessionPortFactory> {
  const loaded: unknown = await import(pathToFileURL(modulePath).href)
  if (!isRecord(loaded) || typeof loaded.createEvaluationSessionPort !== 'function') throw new TypeError(`${modulePath} must export createEvaluationSessionPort(context)`)
  return loaded.createEvaluationSessionPort as EvaluationSessionPortFactory
}

function assertSessionPort(value: unknown): asserts value is EvaluationSessionPort {
  if (!isRecord(value) || !['approve', 'close', 'reset', 'start', 'submit'].every(method => typeof value[method] === 'function')) {
    throw new TypeError('createEvaluationSessionPort(context) must return an EvaluationSessionPort')
  }
}

function positiveNumber(value: string, name: string): number {
  const number = Number(value)
  if (!Number.isSafeInteger(number) || number <= 0) throw new Error(`${name} must be a positive integer`)
  return number
}

function requiredArgument(value: string | undefined, name: string): string {
  if (value === undefined || !value.trim() || value.startsWith('-')) throw new Error(`${name} requires a non-empty value`)
  return value.trim()
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isMissing(error: unknown): boolean { return isRecord(error) && error.code === 'ENOENT' }
function errorMessage(error: unknown): string { return error instanceof Error ? error.message : String(error) }

if (import.meta.main) process.exitCode = await improveMain()
