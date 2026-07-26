// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Windows CreateProcess cannot execute `.cmd`/`.bat` batch shims without a
 * shell, yet PATH is full of them on Windows (`npx.cmd`, `uvx.cmd`, …). This
 * helper keeps the repository's direct-argv spawn design intact: executables
 * pass through unchanged and only batch shims are wrapped in
 * `cmd.exe /d /s /c`. All decision inputs are injectable for tests.
 */

export interface ResolvedSpawn {
  readonly command: string
  readonly args: readonly string[]
  /** True when the command was wrapped in `cmd.exe /d /s /c`. */
  readonly wrapped: boolean
}

export interface ResolveWindowsSpawnOptions {
  readonly platform?: NodeJS.Platform
  /** Executable lookup (`Bun.which`); used for bare names without extension. */
  readonly which?: (name: string) => string | null
}

const BATCH_EXTENSION = /\.(cmd|bat)$/i
const HAS_EXTENSION = /\.[A-Za-z0-9]+$/
const IS_PATH = /[\\/]/
/** Characters that force double-quoting on a `cmd.exe` command line. */
const NEEDS_QUOTES = /[\s"&|<>()%!^]/

function quoteForCmd(value: string): string {
  if (value.length === 0) return '""'
  if (!NEEDS_QUOTES.test(value)) return value
  return `"${value.replace(/"/g, '\\"')}"`
}

/**
 * Resolve the argv that actually spawns `command` on this host.
 *
 * POSIX is the identity function. On win32 a `.cmd`/`.bat` command — given
 * directly, or discovered by resolving a bare name through `which` — is
 * wrapped in `cmd.exe /d /s /c` with cross-spawn-style quoting; everything
 * else passes through so real `.exe` spawns keep their exact argv.
 */
export function resolveWindowsSpawn(
  command: string,
  args: readonly string[],
  options: ResolveWindowsSpawnOptions = {},
): ResolvedSpawn {
  const platform = options.platform ?? process.platform
  if (platform !== 'win32') {
    return { command, args, wrapped: false }
  }
  let resolved = command
  if (BATCH_EXTENSION.test(resolved)) {
    return wrapInCmd(resolved, args)
  }
  // A path with any other extension (.exe, .ps1, …) spawns directly, matching
  // CreateProcess semantics as closely as Bun.spawn allows.
  if (HAS_EXTENSION.test(resolved) || IS_PATH.test(resolved)) {
    return { command, args, wrapped: false }
  }
  const found = (options.which ?? Bun.which)(resolved)
  if (found && BATCH_EXTENSION.test(found)) {
    return wrapInCmd(found, args)
  }
  return { command, args, wrapped: false }
}

function wrapInCmd(batch: string, args: readonly string[]): ResolvedSpawn {
  const line = [batch, ...args].map(quoteForCmd).join(' ')
  // The outer quotes follow the cross-spawn/libuv idiom: with /s, cmd strips
  // the first and last quote and executes the remainder verbatim.
  return { command: 'cmd.exe', args: ['/d', '/s', '/c', `"${line}"`], wrapped: true }
}
