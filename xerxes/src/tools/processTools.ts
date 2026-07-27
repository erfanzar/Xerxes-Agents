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
    description: 'Run one executable directly from an argv list. There is no shell anywhere in this path: `cmd` is '
      + 'rejected if it contains whitespace or any of ;&|`$<>, so pipes, redirection, &&, ||, globs, quoting, ~ and '
      + '$VAR expansion do not exist here, and every entry in `args` reaches the program literally — "*.ts" arrives '
      + 'as the three characters, unexpanded. Express filtering with the program\'s own flags rather than piping to '
      + 'head or grep; if a workflow genuinely needs shell semantics, invoke the interpreter as the executable and '
      + 'note that the approval gate then only sees that interpreter. stdin is /dev/null, so a command that prompts '
      + 'receives nothing and burns the whole timeout doing nothing — pass the non-interactive flag (--yes, '
      + '--no-pager, --porcelain, CI mode) up front. A non-zero exitCode is a normal successful call and never an '
      + 'exception: read stderr and decide. The tool itself only fails for shell syntax in `cmd`, a workdir that is '
      + `not an existing directory inside the workspace, or cancellation. A timeout (default ${DEFAULT_TIMEOUT_MS}ms, `
      + 'ceiling 120000) also returns normally, with timedOut:true and whatever output arrived before the kill, so '
      + 'check that flag before trusting empty output. stdout and stderr are capped independently at '
      + `max_output_chars (default ${DEFAULT_MAX_OUTPUT_CHARS}) with truncated:true set. Nothing survives the call: `
      + 'the process is killed when it returns, and no cwd, environment variable, or shell state carries into the '
      + 'next invocation. Keeping a live shell open across calls needs the separate PTY tool, where the host '
      + 'enables it.',
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
  // Deciding concurrency by tool NAME alone would make every shell call a
  // barrier, and the shipped prompt tells the model to batch independent calls —
  // so it complies and gets serialized anyway. The read-only analyzer already
  // gates permissions for this exact tool, so reusing it here adds no new
  // trust surface: `git status` and `ls` may overlap, anything that writes
  // still runs alone.
  registry.register(
    EXEC_COMMAND_DEFINITION,
    (inputs, _context, signal) => executeCommand(inputs, paths, signal),
    'default',
    { concurrencySafe: false, defer: false, destructive: true, openWorld: true, readOnly: false },
  )
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
