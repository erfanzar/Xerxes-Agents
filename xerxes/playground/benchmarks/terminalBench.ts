// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'

import { parseYaml, yamlMap } from '../../src/agents/yaml.js'
import {
  BenchmarkDatasetError,
  type BenchmarkAdapter,
  type BenchmarkTask,
  type BenchmarkTaskFile,
  type LoadResult,
  type NativeCheck,
} from './types.js'
import {
  applySelection,
  errorMessage,
  isMissingError,
  isRecord,
  listFilesRecursive,
  pathExists,
  safeJoin,
  verdict,
} from './util.js'

const ID = 'terminal-bench'

export const TERMINAL_BENCH_EXPECTED_LAYOUT: readonly string[] = [
  'tasks/<task>/task.yaml (YAML with an "instruction" string)',
  'tasks/<task>/xerxes-checks.json (optional TS-native checks: [{ kind: file_exists|file_contains|file_equals|answer_contains, ... }])',
  'tasks/<task>/workspace/ (optional files copied into the working directory)',
  'Tasks without xerxes-checks.json are reported excluded: "requires container grader".',
]

/**
 * Terminal Bench 2.1. Official checks run a script inside a container, which this adapter never
 * does. A task is runnable only when a declarative TS-native check file ships beside it.
 */
export function createTerminalBenchAdapter(): BenchmarkAdapter {
  return {
    id: ID,
    async load(datasetDir, opts): Promise<LoadResult> {
      const tasksDir = join(datasetDir, 'tasks')
      let names: string[]
      try {
        const entries = await readdir(tasksDir, { withFileTypes: true })
        names = entries.filter(entry => entry.isDirectory()).map(entry => entry.name).sort()
      } catch (error) {
        if (isMissingError(error)) throw new BenchmarkDatasetError(ID, datasetDir, 'missing tasks/ directory', TERMINAL_BENCH_EXPECTED_LAYOUT)
        throw error
      }

      const tasks: BenchmarkTask[] = []
      const excluded: { id: string; reason: string }[] = []
      for (const name of names) {
        const id = `terminal-bench-${name}`
        const taskDir = join(tasksDir, name)
        let instruction: string
        try {
          const yaml = yamlMap(parseYaml(await readFile(join(taskDir, 'task.yaml'), 'utf8'), `${name}/task.yaml`), `${name}/task.yaml`)
          const value = yaml.instruction
          if (typeof value !== 'string' || !value.trim()) throw new Error('task.yaml has no "instruction" string')
          instruction = value.trim()
        } catch (error) {
          excluded.push({ id, reason: `unreadable task.yaml: ${errorMessage(error)}` })
          continue
        }

        const checks = await readChecks(taskDir)
        if (checks === undefined) {
          excluded.push({ id, reason: 'requires container grader' })
          continue
        }
        if (typeof checks === 'string') {
          excluded.push({ id, reason: checks })
          continue
        }

        const files: BenchmarkTaskFile[] = []
        const workspace = join(taskDir, 'workspace')
        if (await pathExists(workspace)) {
          for (const file of await listFilesRecursive(workspace)) files.push({ path: file, sourcePath: join(workspace, file) })
        }
        tasks.push({
          id,
          benchmark: ID,
          prompt: instruction,
          files,
          grader: { kind: 'native-checks', checks },
          tags: ['terminal'],
        })
      }
      return applySelection({ tasks, excluded }, opts)
    },

    async grade(task, run) {
      if (task.grader.kind !== 'native-checks') {
        throw new Error(`${task.id}: terminal-bench adapter cannot grade a ${task.grader.kind} task`)
      }
      const failures: string[] = []
      for (const check of task.grader.checks) {
        const failure = await runCheck(check, run.workdir, run.finalText)
        if (failure !== undefined) failures.push(failure)
      }
      const total = task.grader.checks.length
      const passed = total - failures.length
      const accepted = total > 0 && failures.length === 0
      return verdict(accepted, total === 0 ? 0 : passed / total, accepted ? `all ${total} checks passed` : failures.join('; ') || 'no checks defined')
    },
  }
}

async function readChecks(taskDir: string): Promise<NativeCheck[] | string | undefined> {
  let text: string
  try {
    text = await readFile(join(taskDir, 'xerxes-checks.json'), 'utf8')
  } catch (error) {
    if (isMissingError(error)) return undefined
    throw error
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    return `invalid xerxes-checks.json: ${errorMessage(error)}`
  }
  if (!Array.isArray(parsed) || parsed.length === 0) return 'invalid xerxes-checks.json: expected a non-empty array'
  const checks: NativeCheck[] = []
  for (const entry of parsed) {
    if (!isRecord(entry)) return 'invalid xerxes-checks.json: entries must be objects'
    const { kind, path, text: value } = entry
    if (kind === 'file_exists' && typeof path === 'string') checks.push({ kind, path })
    else if ((kind === 'file_contains' || kind === 'file_equals') && typeof path === 'string' && typeof value === 'string') {
      checks.push({ kind, path, text: value })
    } else if (kind === 'answer_contains' && typeof value === 'string') checks.push({ kind, text: value })
    else return `invalid xerxes-checks.json: unsupported check ${JSON.stringify(kind)}`
  }
  return checks
}

async function runCheck(check: NativeCheck, workdir: string, finalText: string): Promise<string | undefined> {
  if (check.kind === 'answer_contains') {
    return finalText.toLowerCase().includes(check.text.toLowerCase()) ? undefined : `reply does not contain "${check.text}"`
  }
  let contents: string
  try {
    contents = await readFile(safeJoin(workdir, check.path), 'utf8')
  } catch (error) {
    if (isMissingError(error)) return `${check.path} does not exist`
    throw error
  }
  if (check.kind === 'file_exists') return undefined
  if (check.kind === 'file_contains') return contents.includes(check.text) ? undefined : `${check.path} does not contain "${check.text}"`
  return contents.trimEnd() === check.text.trimEnd() ? undefined : `${check.path} differs from the expected text`
}
