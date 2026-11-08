// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import type { ToolDefinition } from '../types/toolCalls.js'

/** Environment variable that turns a protected run on or off. */
export const PROTECTED_RUN_ENV = 'XERXES_PROTECTED_RUN'

const TRUE_WORDS = new Set(['1', 'true', 'yes', 'on'])
const FALSE_WORDS = new Set(['0', 'false', 'no', 'off'])

/**
 * Resolve whether a protected run is requested. Precedence: explicit CLI flag,
 * then `XERXES_PROTECTED_RUN`, then `runtime.protected_run`, then false.
 * An unreadable value is a ValidationError, never a silent `false`.
 */
export function parseProtectedRunConfig(
  runtime: Record<string, unknown>,
  env: Record<string, string | undefined>,
  flag?: boolean,
): boolean {
  if (flag !== undefined) return flag
  const fromEnv = env[PROTECTED_RUN_ENV]?.trim().toLowerCase()
  if (fromEnv) {
    if (TRUE_WORDS.has(fromEnv)) return true
    if (FALSE_WORDS.has(fromEnv)) return false
    throw new ValidationError(PROTECTED_RUN_ENV, 'must be one of 1, true, yes, on, 0, false, no, off', fromEnv)
  }
  const fromFile = runtime['protected_run']
  if (fromFile === undefined || fromFile === null) return false
  if (typeof fromFile !== 'boolean') throw new ValidationError('protected_run', 'must be a boolean', fromFile)
  return fromFile
}

/** Shell, terminal and background-process tools (compared case-insensitively). */
const SHELL_TOOL_NAMES: ReadonlySet<string> = new Set([
  'exec_command',
  'executeshell',
  'bash',
  'shell',
  'run_command',
  'write_stdin',
  // Background-process companions of exec_command: with no shell there is nothing to poll or kill.
  'check_command',
  'kill_command',
  'list_commands',
])

const OTHER_PRIVILEGED_TOOL_NAMES: ReadonlySet<string> = new Set([
  'pluginmanagetool',
  'computer_use',
])

/** Why a tool is not available in a protected run, or undefined when it is allowed. */
export function protectedRunExclusion(name: string): 'shell' | 'mcp' | 'plugin' | 'privileged' | undefined {
  const lower = name.toLowerCase()
  if (SHELL_TOOL_NAMES.has(lower) || lower.startsWith('pty_')) return 'shell'
  if (lower.startsWith('mcp__') || lower.startsWith('mcp_')) return 'mcp'
  if (lower.startsWith('plugin_')) return 'plugin'
  if (OTHER_PRIVILEGED_TOOL_NAMES.has(lower)) return 'privileged'
  return undefined
}

/**
 * Split tool definitions for a protected run: shell/terminal tools, MCP tools,
 * plugin tools, PluginManageTool and computer_use are removed. Order is kept.
 */
export function filterProtectedTools<T extends ToolDefinition>(
  definitions: readonly T[],
): { readonly kept: T[]; readonly removed: T[] } {
  const kept: T[] = []
  const removed: T[] = []
  for (const definition of definitions) {
    if (protectedRunExclusion(definition.function.name) === undefined) kept.push(definition)
    else removed.push(definition)
  }
  return { kept, removed }
}
