// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Model-facing persistent PTY sessions.
 *
 * exec_command is deliberately one-shot: fresh process, no shell state. That
 * is the wrong shape for interactive work — an SSH hop you keep open and run
 * many commands on, a REPL, a long interactive installer — where the session
 * IS the state. These tools expose the PtySessionManager: open once, then
 * write/read against the same terminal across turns, with output mirrored to
 * the terminals panel (F8) so the user can watch and type too.
 *
 * Ownership is scoped to the calling Xerxes session, exactly like background
 * commands: a session can only see, write to, and close its own PTYs, and the
 * daemon disposes them when the session ends.
 */

import { ValidationError } from '../core/errors.js'
import type { ToolRegistry } from '../executors/toolRegistry.js'
import type { PtySessionManager } from '../operators/pty.js'
import type { ToolDefinition } from '../types/toolCalls.js'
import { optionalBoolean, optionalInteger, optionalString, requireRange, requiredString } from './inputs.js'

const DEFAULT_YIELD_MS = 1_000
/** Ceiling on a write's yield, so a poll can never become a blocking wait. */
const MAX_YIELD_MS = 60_000
const DEFAULT_MAX_OUTPUT_CHARS = 4_000

export const PTY_OPEN_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'pty_open',
    description: 'Open a persistent interactive terminal (a real PTY) running a shell command, and keep it alive '
      + 'across calls. Unlike exec_command, everything carries over: cwd, environment, shell state, and the process '
      + 'itself. This is the right tool for an SSH connection you run many commands on (`ssh user@host`, then keep '
      + 'sending commands with pty_write), a REPL (python, node, psql), or any interactive program. It is the WRONG '
      + 'tool for a one-shot command — use exec_command for that. The command runs through the login shell, so '
      + 'shell syntax is available here (unlike exec_command). Returns session_id plus the output produced within '
      + 'yield_time_ms; a still-running session says so and is polled by writing more (or empty chars). Open at '
      + 'most the sessions you truly use concurrently, and pty_close them when done.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        command: { type: 'string', description: 'Shell command to run in the new terminal (e.g. "ssh user@host").' },
        workdir: { type: 'string', default: '.', description: 'Workspace-relative working directory.' },
        yield_time_ms: {
          type: 'integer',
          default: DEFAULT_YIELD_MS,
          description: `How long to wait for initial output before answering (ceiling ${MAX_YIELD_MS}).`,
        },
        max_output_chars: {
          type: 'integer',
          default: DEFAULT_MAX_OUTPUT_CHARS,
          description: 'Maximum output characters returned; the rest stays buffered for later reads.',
        },
      },
      required: ['command'],
    },
  },
}

export const PTY_WRITE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'pty_write',
    description: 'Write to a PTY session opened with pty_open and read what it printed since your last call — output '
      + 'is consumed per call, so successive calls show new work. Send a command as chars ending in "\\n". With no '
      + 'chars at all this is a pure poll. interrupt:true delivers Ctrl+C to the foreground command WITHOUT killing '
      + 'the shell, close_stdin:true sends EOF (^D). Answers after yield_time_ms or first output, whichever first.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        session_id: { type: 'string', description: 'Identifier returned by pty_open.' },
        chars: { type: 'string', description: 'Text to type into the terminal; include "\\n" to submit a line.' },
        interrupt: { type: 'boolean', default: false, description: 'Send Ctrl+C instead of typing.' },
        close_stdin: { type: 'boolean', default: false, description: 'Send EOF (^D) after any chars.' },
        yield_time_ms: {
          type: 'integer',
          default: DEFAULT_YIELD_MS,
          description: `How long to wait for output before answering (ceiling ${MAX_YIELD_MS}).`,
        },
        max_output_chars: {
          type: 'integer',
          default: DEFAULT_MAX_OUTPUT_CHARS,
          description: 'Maximum output characters returned; the rest stays buffered.',
        },
      },
      required: ['session_id'],
    },
  },
}

export const PTY_CLOSE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'pty_close',
    description: 'Close a PTY session opened with pty_open: SIGTERM, then SIGKILL after a short grace. Returns the '
      + 'exit code. Sessions still open when your Xerxes session ends are closed automatically.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        session_id: { type: 'string', description: 'Identifier returned by pty_open.' },
      },
      required: ['session_id'],
    },
  },
}

export const PTY_LIST_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'pty_list',
    description: 'List PTY sessions this session opened that are still tracked, with session_id, command, running '
      + 'state and exit code. Use it to recover a session_id you did not record.',
    parameters: { type: 'object', additionalProperties: false, properties: {} },
  },
}

function requireOwner(sessionId: string | undefined): string {
  if (sessionId === undefined || sessionId.trim() === '') {
    throw new ValidationError('sessionId', 'is required for PTY session tools')
  }
  return sessionId
}

function yieldMs(value: number | undefined): number {
  return requireRange(value ?? DEFAULT_YIELD_MS, 'yield_time_ms', 0, MAX_YIELD_MS)
}

function maxOutput(value: number | undefined): number {
  return requireRange(value ?? DEFAULT_MAX_OUTPUT_CHARS, 'max_output_chars', 1, 1_000_000)
}

export function registerPtyTools(registry: ToolRegistry, manager: PtySessionManager): void {
  registry.register(
    PTY_OPEN_DEFINITION,
    async (inputs, context) => {
      const workdir = optionalString(inputs, 'workdir')
      return manager.createSession(requiredString(inputs, 'command'), {
        ownerSessionId: requireOwner(context.sessionId),
        ...(workdir === undefined ? {} : { workdir }),
        yieldTimeMs: yieldMs(optionalInteger(inputs, 'yield_time_ms', DEFAULT_YIELD_MS)),
        maxOutputChars: maxOutput(optionalInteger(inputs, 'max_output_chars', DEFAULT_MAX_OUTPUT_CHARS)),
      })
    },
    'default',
    // A shell running an arbitrary command string: same trust surface as
    // exec_command through an interpreter, so the same flags.
    { concurrencySafe: false, defer: true, destructive: true, openWorld: true, readOnly: false },
    'Open one PTY per interactive context (one SSH host, one REPL) and reuse it with pty_write instead of opening '
      + 'a new terminal per command; pty_close when the work is done.',
  )
  registry.register(
    PTY_WRITE_DEFINITION,
    async (inputs, context) => manager.writeForOwner(
      requireOwner(context.sessionId),
      requiredString(inputs, 'session_id'),
      {
        ...(optionalString(inputs, 'chars') === undefined
          ? {}
          : { chars: optionalString(inputs, 'chars') as string }),
        interrupt: optionalBoolean(inputs, 'interrupt', false),
        closeStdin: optionalBoolean(inputs, 'close_stdin', false),
        yieldTimeMs: yieldMs(optionalInteger(inputs, 'yield_time_ms', DEFAULT_YIELD_MS)),
        maxOutputChars: maxOutput(optionalInteger(inputs, 'max_output_chars', DEFAULT_MAX_OUTPUT_CHARS)),
      },
    ),
    'default',
    // Typing into a session mutates its state, so it is neither read-only nor
    // safe to run concurrently with itself.
    { concurrencySafe: false, defer: true, destructive: false, openWorld: true, readOnly: false },
  )
  registry.register(
    PTY_CLOSE_DEFINITION,
    async (inputs, context) => manager.closeForOwner(
      requireOwner(context.sessionId),
      requiredString(inputs, 'session_id'),
    ),
    'default',
    { concurrencySafe: false, defer: true, destructive: true, openWorld: false, readOnly: false },
  )
  registry.register(
    PTY_LIST_DEFINITION,
    async (_inputs, context) => ({ sessions: manager.listForOwner(requireOwner(context.sessionId)) }),
    'default',
    { concurrencySafe: true, defer: true, destructive: false, openWorld: false, readOnly: true },
  )
}
