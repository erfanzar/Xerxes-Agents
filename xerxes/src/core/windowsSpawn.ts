// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Windows executable resolution for helpers Xerxes spawns by name.
//
// Two Windows facts break the plain `spawn(['name', ...args])` that works
// everywhere else:
//
//   1. Executables carry an extension. `git` is `git.exe`, and the extensions
//      that count come from `PATHEXT`.
//   2. `.cmd` and `.bat` files are not executables at all. `CreateProcess`
//      cannot run them; only `cmd.exe` can. This is not an edge case for an
//      agent runtime — every npm-installed CLI on Windows is a `.cmd` shim, so
//      MCP servers launched as `npx …` land here.
//
// Wrapping in `cmd.exe` re-introduces a shell, so the quoting is deliberate and
// conservative: arguments that cannot be represented safely inside a cmd quoted
// string are rejected rather than escaped-and-hoped. `%` is the reason this is a
// rejection and not a transformation — cmd expands `%VAR%` *inside* double
// quotes, so no amount of quoting makes an argument containing `%` inert (this
// is the same hole as CVE-2024-27980). A clear error beats a command that runs
// something the caller did not write.

import { XerxesError } from './errors.js'
import { isWindows } from './hostPlatform.js'

/** Extensions that must be run through `cmd.exe` rather than executed directly. */
const BATCH_EXTENSIONS = ['.cmd', '.bat'] as const

/** Characters with no safe representation inside a `cmd.exe` quoted argument. */
const UNREPRESENTABLE_IN_CMD = /[%\r\n\0]/

export class WindowsSpawnError extends XerxesError {}

export interface SpawnPlan {
  /** argv to hand to `Bun.spawn`. */
  readonly argv: string[]
  /**
   * Must be forwarded to `Bun.spawn` when true.
   *
   * The wrapped form is a single pre-quoted command line. Letting the runtime
   * re-quote it would escape the inner quotes as `\"`, which `cmd.exe` does not
   * understand, and the command would silently run with mangled arguments.
   */
  readonly windowsVerbatimArguments?: true
}

export interface PlanSpawnOptions {
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly platform?: NodeJS.Platform
  /** Injected for tests; defaults to `Bun.which`, which honours PATHEXT. */
  readonly which?: (command: string) => string | null
}

/**
 * Resolve how to actually spawn `command` with `args` on the host platform.
 *
 * POSIX is returned untouched. Windows resolves the name through `PATHEXT` and,
 * for a batch shim, produces the `cmd.exe /d /s /c "…"` form with
 * `windowsVerbatimArguments` set.
 */
export function planSpawn(
  command: string,
  args: readonly string[] = [],
  options: PlanSpawnOptions = {},
): SpawnPlan {
  const platform = options.platform ?? process.platform
  if (!isWindows(platform)) {
    return { argv: [command, ...args] }
  }

  const env = options.env ?? process.env
  const which = options.which ?? ((name: string) => Bun.which(name))
  // An unresolvable name is passed through unchanged so the spawn fails with the
  // runtime's own ENOENT, which names the command; inventing an error here would
  // only replace a familiar message with an unfamiliar one.
  const resolved = which(command) ?? command

  if (!isBatchScript(resolved)) {
    return { argv: [resolved, ...args] }
  }

  const commandLine = [resolved, ...args].map(quoteForCmd).join(' ')
  const comspec = env.COMSPEC?.trim() || 'cmd.exe'
  // `/d` skips AutoRun commands from the registry — without it, a machine-local
  // AutoRun value would execute inside every tool call Xerxes makes. `/s` makes
  // cmd strip exactly the one outer quote pair below and leave the inner quoting
  // to its normal parser.
  return {
    argv: [comspec, '/d', '/s', '/c', `"${commandLine}"`],
    windowsVerbatimArguments: true,
  }
}

/** Whether a resolved path is a batch script, which only `cmd.exe` can run. */
export function isBatchScript(resolvedPath: string): boolean {
  const lowered = resolvedPath.toLowerCase()
  return BATCH_EXTENSIONS.some(extension => lowered.endsWith(extension))
}

/**
 * Quote one argument for a `cmd.exe` command line.
 *
 * Always quoted, internal `"` doubled — the form `cmd.exe`'s parser accepts.
 * Anything that cannot be neutralized inside those quotes is rejected; see the
 * note on `%` at the top of this file.
 */
function quoteForCmd(argument: string): string {
  if (UNREPRESENTABLE_IN_CMD.test(argument)) {
    throw new WindowsSpawnError(
      'argument cannot be passed to a Windows .cmd/.bat command safely: '
      + 'it contains a percent sign, newline, or null byte',
      { argument },
    )
  }
  return `"${argument.replaceAll('"', '""')}"`
}
