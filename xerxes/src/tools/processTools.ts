// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { stat } from 'node:fs/promises'

import { ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { optionalInteger, optionalString, optionalStringArray, requireRange, requiredString } from './inputs.js'
import { WorkspacePathResolver } from './pathSafety.js'

const DEFAULT_TIMEOUT_MS = 30_000
const DEFAULT_MAX_OUTPUT_CHARS = 20_000

export const EXEC_COMMAND_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'exec_command',
    description: 'Run one non-interactive argv command with a workspace-relative working directory. '
      + 'The command is never interpreted by a shell. The working directory is contained by the '
      + 'workspace; the executable itself may be any installed binary and is authorized by the '
      + 'upstream tool-policy/approval gate, not by this tool.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        cmd: { type: 'string', description: 'Executable name or path, without shell syntax.' },
        args: {
          type: 'array',
          items: { type: 'string' },
          default: [],
          description: 'Arguments passed directly to the executable.',
        },
        workdir: { type: 'string', default: '.', description: 'Workspace-relative working directory.' },
        timeout_ms: {
          type: 'integer',
          default: DEFAULT_TIMEOUT_MS,
          description: 'Maximum process runtime in milliseconds.',
        },
        max_output_chars: {
          type: 'integer',
          default: DEFAULT_MAX_OUTPUT_CHARS,
          description: 'Maximum characters returned per output stream.',
        },
      },
      required: ['cmd'],
    },
  },
}

export interface ProcessResult {
  readonly command: readonly string[]
  readonly cwd: string
  readonly exitCode: number
  readonly stderr: string
  readonly stdout: string
  readonly timedOut: boolean
  readonly truncated: boolean
}

/** Register a bounded, direct-argv process tool; persistent PTYs remain a separate port. */
export function registerProcessTools(registry: ToolRegistry, paths: WorkspacePathResolver): void {
  registry.register(EXEC_COMMAND_DEFINITION, (inputs, _context, signal) => executeCommand(inputs, paths, signal))
}

/**
 * Execute one direct-argv command.
 *
 * Policy boundary (intentional): the charset check below blocks shell
 * metacharacters so the command can never be reinterpreted by a shell, and
 * `workdir` is contained by the workspace resolver. The `cmd` executable
 * itself may still be an absolute or `../`-relative path — constraining which
 * binaries may run is the job of the upstream tool-policy/approval gate, not
 * of this executor, so the behavior is deliberately left unchanged here.
 */
export async function executeCommand(
  inputs: JsonObject,
  paths: WorkspacePathResolver,
  signal?: AbortSignal,
): Promise<ProcessResult> {
  const command = requiredString(inputs, 'cmd')
  if (/\s/.test(command) || /[;&|`$<>]/.test(command)) {
    throw new ValidationError(
      'cmd',
      'must contain one executable only; pass arguments separately and do not use shell syntax',
      command,
    )
  }
  const args = optionalStringArray(inputs, 'args')
  const workdir = optionalString(inputs, 'workdir') ?? '.'
  const timeout = requireRange(optionalInteger(inputs, 'timeout_ms', DEFAULT_TIMEOUT_MS), 'timeout_ms', 1, 120_000)
  const maxOutputChars = requireRange(
    optionalInteger(inputs, 'max_output_chars', DEFAULT_MAX_OUTPUT_CHARS),
    'max_output_chars',
    1,
    1_000_000,
  )
  const cwd = await paths.resolve(workdir)
  if (!(await isDirectory(cwd))) {
    throw new ValidationError('workdir', 'must refer to an existing workspace directory', workdir)
  }
  if (signal?.aborted) {
    throw new ValidationError('exec_command', 'was cancelled before execution')
  }

  let timedOut = false
  const controller = new AbortController()
  const cancel = () => controller.abort(signal?.reason)
  signal?.addEventListener('abort', cancel, { once: true })
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort(new Error(`Command timed out after ${timeout}ms`))
  }, timeout)

  try {
    const process = Bun.spawn([command, ...args], {
      cwd,
      stdin: 'ignore',
      stdout: 'pipe',
      stderr: 'pipe',
      signal: controller.signal,
      maxBuffer: maxOutputChars * 8,
    })
    const [exitCode, stdout, stderr] = await Promise.all([
      process.exited,
      new Response(process.stdout).text(),
      new Response(process.stderr).text(),
    ])
    if (signal?.aborted && !timedOut) {
      throw new ValidationError('exec_command', 'was cancelled during execution')
    }
    const stdoutResult = capOutput(stdout, maxOutputChars)
    const stderrResult = capOutput(stderr, maxOutputChars)
    return {
      command: [command, ...args],
      cwd: await paths.relative(cwd),
      exitCode,
      stdout: stdoutResult.text,
      stderr: stderrResult.text,
      timedOut,
      truncated: stdoutResult.truncated || stderrResult.truncated,
    }
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', cancel)
  }
}

function capOutput(output: string, maxChars: number): { text: string; truncated: boolean } {
  if (output.length <= maxChars) {
    return { text: output, truncated: false }
  }
  return { text: `${output.slice(0, maxChars)}\n…[truncated]…`, truncated: true }
}

async function isDirectory(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isDirectory()
  } catch (error) {
    if (typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT') {
      return false
    }
    throw error
  }
}
