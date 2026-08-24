// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { stat } from 'node:fs/promises'

import { ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import { ProcessRegistry } from '../runtime/processRegistry.js'
import type { TerminalRegistry } from '../runtime/terminalRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { optionalBoolean, optionalInteger, optionalString, optionalStringArray, requireRange, requiredString } from './inputs.js'
import { WorkspacePathResolver } from './pathSafety.js'
import {
  BackgroundCommandManager,
  type BackgroundStartResult,
  MAX_CHECK_WAIT_MS,
} from './backgroundCommands.js'
import { BoundedOutputBuffer, capOutput, drainStream, type StreamDrain } from './processOutput.js'

const DEFAULT_TIMEOUT_MS = 30_000
const DEFAULT_MAX_OUTPUT_CHARS = 20_000
/**
 * Grace period for output already written but still in the pipe when the child
 * exits. Short: it is paid on every call, and it is not a wait for EOF.
 */
const OUTPUT_SETTLE_MS = 50

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
      + 'enables it. For work that outlasts the timeout — a build, a server, a training run — do NOT background it '
      + 'with `&` or nohup, whose output goes nowhere you can read and whose failure you cannot see: pass '
      + 'run_in_background:true instead, which returns a proc_id immediately and leaves the process readable through '
      + 'check_command.',
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
        run_in_background: {
          type: 'boolean',
          default: false,
          description: 'Start the command and return a proc_id at once instead of waiting for it. timeout_ms does '
            + 'not apply — the process runs until it exits or you kill it. Read its output with check_command.',
        },
      },
      required: ['cmd'],
    },
  },
}

export const CHECK_COMMAND_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'check_command',
    description: 'Read the progress of a background command started by exec_command with run_in_background:true. '
      + 'Returns running, exitCode (null while running), and the output produced since your last check — output is '
      + 'consumed, so successive calls show new work rather than repeating from the start, and an empty stdout on a '
      + 'running process means nothing new, not nothing at all. Only the most recent ~1M characters per stream are '
      + 'retained; droppedOutput:true means a chatty process outran your polling and the earliest output is gone. '
      + 'truncated:true means more is buffered than max_output_chars carried, so call again. Set wait_ms to give a '
      + `command that is nearly done a chance to finish (ceiling ${MAX_CHECK_WAIT_MS}) rather than returning `
      + 'running:true and being asked again immediately. A finished process stays readable until you kill_command it, '
      + 'so its final output is never lost by checking too late.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        proc_id: { type: 'string', description: 'Identifier returned by exec_command run_in_background.' },
        max_output_chars: {
          type: 'integer',
          default: DEFAULT_MAX_OUTPUT_CHARS,
          description: 'Maximum characters returned per output stream.',
        },
        wait_ms: {
          type: 'integer',
          default: 0,
          description: 'Block up to this long for the process to exit before answering.',
        },
      },
      required: ['proc_id'],
    },
  },
}

export const KILL_COMMAND_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'kill_command',
    description: 'Stop a background command and stop tracking it. SIGTERM by default so it can shut down cleanly; '
      + 'pass SIGKILL for one that ignores it. signalled:false means the process had already exited on its own, '
      + 'which is not an error. Read any output you still want with check_command first — the buffer is released '
      + 'here. Killing is how a background process ends; anything still running is terminated when the session ends.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        proc_id: { type: 'string', description: 'Identifier returned by exec_command run_in_background.' },
        signal: {
          type: 'string',
          enum: ['SIGTERM', 'SIGKILL'],
          default: 'SIGTERM',
          description: 'Signal to deliver.',
        },
      },
      required: ['proc_id'],
    },
  },
}

export const LIST_COMMANDS_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'list_commands',
    description: 'List background commands this session started that have not been killed, with their proc_id, pid, '
      + 'command line and start time. Use it to recover a proc_id you did not record, or to confirm nothing was left '
      + 'running.',
    parameters: { type: 'object', additionalProperties: false, properties: {} },
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
export function registerProcessTools(
  registry: ToolRegistry,
  paths: WorkspacePathResolver,
  backgroundManager?: BackgroundCommandManager,
  terminals?: TerminalRegistry,
): void {
  const background = backgroundManager ?? new BackgroundCommandManager(new ProcessRegistry(), terminals)
  // Deciding concurrency by tool NAME alone would make every shell call a
  // barrier, and the shipped prompt tells the model to batch independent calls —
  // so it complies and gets serialized anyway. The read-only analyzer already
  // gates permissions for this exact tool, so reusing it here adds no new
  // trust surface: `git status` and `ls` may overlap, anything that writes
  // still runs alone.
  registry.register(
    EXEC_COMMAND_DEFINITION,
    (inputs, context, signal) => executeCommand(
      inputs,
      paths,
      signal,
      background,
      terminals,
      requiredOwnerSessionId(context.sessionId),
    ),
    'default',
    { concurrencySafe: false, defer: false, destructive: true, openWorld: true, readOnly: false },
    // Co-located usage policy for the one tool whose shape models most often
    // get wrong: argv-only invocation plus the batching rule the registry
    // actually enforces (read-only commands run concurrently, writers alone).
    'Invoke with a command name and an args array — never a single shell string. '
      + 'Independent read-only commands (git status, ls, cat) may be batched in one round and run '
      + 'concurrently; anything that writes runs alone, so do not batch it with calls that must '
      + 'observe its effect.',
  )
  // Checking and listing only read state that this session already owns, so they
  // are concurrency-safe and read-only: polling three builds at once is the
  // normal case and must not serialize behind an approval gate.
  registry.register(
    CHECK_COMMAND_DEFINITION,
    async (inputs, context) => background.checkForOwner(
      requiredOwnerSessionId(context.sessionId),
      requiredString(inputs, 'proc_id'),
      requireRange(optionalInteger(inputs, 'max_output_chars', DEFAULT_MAX_OUTPUT_CHARS), 'max_output_chars', 1, 1_000_000),
      requireRange(optionalInteger(inputs, 'wait_ms', 0), 'wait_ms', 0, MAX_CHECK_WAIT_MS),
    ),
    'default',
    { concurrencySafe: true, defer: false, destructive: false, openWorld: false, readOnly: true },
  )
  registry.register(
    LIST_COMMANDS_DEFINITION,
    async (_inputs, context) => ({ processes: background.listForOwner(requiredOwnerSessionId(context.sessionId)) }),
    'default',
    { concurrencySafe: true, defer: false, destructive: false, openWorld: false, readOnly: true },
  )
  registry.register(
    KILL_COMMAND_DEFINITION,
    async (inputs, context) => background.killForOwner(
      requiredOwnerSessionId(context.sessionId),
      requiredString(inputs, 'proc_id'),
      killSignal(optionalString(inputs, 'signal')),
    ),
    'default',
    { concurrencySafe: false, defer: false, destructive: true, openWorld: false, readOnly: false },
  )
}

function requiredOwnerSessionId(sessionId: string | undefined): string {
  if (sessionId === undefined || sessionId.trim() === '') {
    throw new ValidationError('sessionId', 'is required for background command tools')
  }
  return sessionId
}

function killSignal(value: string | undefined): 'SIGKILL' | 'SIGTERM' {
  if (value === undefined || value === 'SIGTERM') return 'SIGTERM'
  if (value === 'SIGKILL') return 'SIGKILL'
  throw new ValidationError('signal', 'must be SIGTERM or SIGKILL', value)
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
  background?: BackgroundCommandManager,
  terminals?: TerminalRegistry,
  ownerSessionId?: string,
): Promise<BackgroundStartResult | ProcessResult> {
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

  if (optionalBoolean(inputs, 'run_in_background', false)) {
    if (background === undefined) {
      throw new ValidationError('run_in_background', 'is not enabled by this host', true)
    }
    // No timeout is applied on purpose: outliving the foreground ceiling is the
    // entire reason a caller asks for this.
    const startOptions = { command, args, cwd, name: [command, ...args].join(' ').slice(0, 60) }
    return ownerSessionId === undefined
      ? background.start(startOptions)
      : background.startForOwner(ownerSessionId, startOptions)
  }

  let timedOut = false
  const controller = new AbortController()
  const cancel = () => controller.abort(signal?.reason)
  signal?.addEventListener('abort', cancel, { once: true })
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort(new Error(`Command timed out after ${timeout}ms`))
  }, timeout)

  const stdoutBuffer = new BoundedOutputBuffer(maxOutputChars * 8)
  const stderrBuffer = new BoundedOutputBuffer(maxOutputChars * 8)
  let stdoutDrain: StreamDrain | undefined
  let stderrDrain: StreamDrain | undefined
  let child: Bun.Subprocess | undefined
  let observedExit: number | null = null
  // Mirrored live rather than recorded at the end: a foreground command may run
  // for two minutes, and the whole point of the terminal panel is being able to
  // watch it during those two minutes instead of afterwards. The kill control
  // reads `child` at call time, so it can be published before the spawn.
  const mirror = ownerSessionId === undefined ? undefined : terminals?.open({
    id: `fg_${crypto.randomUUID().replaceAll('-', '').slice(0, 10)}`,
    kind: 'foreground',
    ownerSessionId,
    command: [command, ...args].join(' '),
    cwd,
    control: {
      kill: async killSignal => {
        child?.kill(killSignal)
      },
    },
  })

  try {
    const process = Bun.spawn([command, ...args], {
      cwd,
      stdin: 'ignore',
      stdout: 'pipe',
      stderr: 'pipe',
      signal: controller.signal,
    })
    child = process
    stdoutDrain = drainStream(process.stdout, stdoutBuffer, text => mirror?.append(text))
    stderrDrain = drainStream(process.stderr, stderrBuffer, text => mirror?.append(text))

    // Wait for the process, not for its pipes to reach EOF.
    //
    // EOF requires every holder of the write end to close it, and a command that
    // backgrounds anything (`cmd &`, `nohup cmd`) hands a copy to a process that
    // outlives the one we can kill. Awaiting EOF therefore made the timeout
    // unenforceable: it fired, killed the direct child, and the call still sat
    // forever on a read that could never finish — one stray `&` stalled a turn
    // indefinitely. Exit status is a fact about our child alone, so that is what
    // bounds the call.
    const exitCode = await process.exited
    observedExit = typeof exitCode === 'number' ? exitCode : null
    // Give the already-buffered output a moment to land. Output written just
    // before exit may still be in flight in the pipe, and returning without it
    // would drop the last line of every fast command.
    await settleDrains([stdoutDrain, stderrDrain], OUTPUT_SETTLE_MS)

    if (signal?.aborted && !timedOut) {
      throw new ValidationError('exec_command', 'was cancelled during execution')
    }
    const stdoutResult = capOutput(stdoutBuffer.peek(maxOutputChars + 1).text, maxOutputChars)
    const stderrResult = capOutput(stderrBuffer.peek(maxOutputChars + 1).text, maxOutputChars)
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
    // Release our end unconditionally. A grandchild may still hold the write
    // end; that is its business, and no longer ours.
    stdoutDrain?.cancel()
    stderrDrain?.cancel()
    // Closed on the error and cancellation paths too: a foreground command the
    // panel still listed as running after the turn moved on would be a lie.
    mirror?.close(observedExit)
  }
}

/** Wait briefly for in-flight output, without ever waiting on EOF. */
async function settleDrains(drains: readonly StreamDrain[], settleMs: number): Promise<void> {
  await Promise.race([
    Promise.all(drains.map(drain => drain.done)),
    new Promise<void>(resolve => setTimeout(resolve, settleMs)),
  ])
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
