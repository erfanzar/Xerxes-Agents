// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolve } from 'node:path'

import { loadMatrixFile, runMatrix, type MatrixDependencies, type MatrixFile } from './matrix.js'
import { renderMarkdown, type MatrixReport } from './matrixReport.js'

export const MATRIX_CLI_USAGE = [
  'Usage: bun playground/matrixCli.ts --matrix <file.json> [options]',
  '',
  'Developer-only comparison suite. Reads locally downloaded benchmark datasets; never fetches.',
  '',
  'Options:',
  '  --matrix <file>     Matrix file: { configs: [...], benchmarks: [...] } (see playground/benchmarks/README.md).',
  '  --out <directory>   Where report.json, report.md and review packets are written (default .xerxes/evaluations/matrix-report).',
  '  --run-root <dir>    Parent directory for isolated per-run workspaces (default .xerxes/evaluations).',
  '  -h, --help          Print this help.',
].join('\n')

export interface MatrixCliOptions {
  readonly help: boolean
  readonly matrixFile: string | undefined
  readonly outputDir: string
  readonly runRoot: string
}

export interface MatrixCliDependencies {
  readonly loadMatrix?: (path: string) => Promise<MatrixFile>
  readonly run?: (matrix: MatrixFile, deps: MatrixDependencies) => Promise<MatrixReport>
  readonly write?: (text: string) => void | Promise<void>
  readonly writeError?: (text: string) => void | Promise<void>
}

function requiredArgument(value: string | undefined, name: string): string {
  if (typeof value !== 'string' || !value.trim() || value.startsWith('-')) throw new TypeError(`${name} requires a non-empty value`)
  return value.trim()
}

export function parseMatrixCliOptions(args: readonly string[]): MatrixCliOptions {
  let help = false
  let matrixFile: string | undefined
  let outputDir = resolve(process.cwd(), '.xerxes', 'evaluations', 'matrix-report')
  let runRoot = resolve(process.cwd(), '.xerxes', 'evaluations')
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === '--help' || argument === '-h') {
      help = true
    } else if (argument === '--matrix') {
      matrixFile = resolve(requiredArgument(args[index + 1], argument))
      index += 1
    } else if (argument === '--out') {
      outputDir = resolve(requiredArgument(args[index + 1], argument))
      index += 1
    } else if (argument === '--run-root') {
      runRoot = resolve(requiredArgument(args[index + 1], argument))
      index += 1
    } else {
      throw new TypeError(`unknown argument: ${String(argument)}`)
    }
  }
  return { help, matrixFile, outputDir, runRoot }
}

/** Run the comparison matrix. Exit 0 when every run completed, 1 when any run errored, 2 on bad usage. */
export async function runMatrixCli(args: readonly string[], dependencies: MatrixCliDependencies = {}): Promise<number> {
  const write = dependencies.write ?? (text => process.stdout.write(text))
  const writeError = dependencies.writeError ?? (text => process.stderr.write(text))
  let options: MatrixCliOptions
  try {
    options = parseMatrixCliOptions(args)
  } catch (error) {
    await writeError(`matrix: ${error instanceof Error ? error.message : String(error)}\n\n${MATRIX_CLI_USAGE}\n`)
    return 2
  }
  if (options.help) {
    await write(`${MATRIX_CLI_USAGE}\n`)
    return 0
  }
  if (options.matrixFile === undefined) {
    await writeError(`matrix: --matrix <file> is required\n\n${MATRIX_CLI_USAGE}\n`)
    return 2
  }
  try {
    const matrix = await (dependencies.loadMatrix ?? loadMatrixFile)(options.matrixFile)
    const report = await (dependencies.run ?? runMatrix)(matrix, {
      onProgress: line => {
        void writeError(`${line}\n`)
      },
      outputDir: options.outputDir,
      runRoot: options.runRoot,
    })
    await write(renderMarkdown(report))
    await write(`Report files: ${options.outputDir}\n`)
    return report.records.some(record => record.status === 'error') ? 1 : 0
  } catch (error) {
    await writeError(`matrix: ${error instanceof Error ? error.message : String(error)}\n`)
    return 1
  }
}

if (import.meta.main) process.exitCode = await runMatrixCli(process.argv.slice(2))
