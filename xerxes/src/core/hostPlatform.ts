// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Single source of truth for the host-OS facts that the runtime branches on.
// Everything here takes `platform` and `env` as explicit parameters so a Windows
// path can be exercised from a macOS or Linux test run: the CI matrix cannot
// cover every branch on every OS, and a branch that only ever executes on the
// host that happens to be running is a branch nobody reviews.
//
// The TUI keeps a parallel copy in `ui/lib/hostPlatform.ts`. That duplication is
// deliberate and matches the existing precedent in `core/processLiveness.ts`:
// the TUI bundle compiles standalone under `rootDir: src/ui` and must not reach
// into runtime-side modules. `test/windowsSupport.test.ts` asserts the two
// copies stay in agreement, because a control-channel path that disagrees by one
// character means the TUI silently starts a second daemon.

import { basename } from 'node:path'

/** Windows named-pipe prefix accepted by libuv, and therefore by Bun's `node:net`. */
const NAMED_PIPE_PREFIX = '\\\\.\\pipe\\'

/**
 * Environment variables a child process needs to function at all.
 *
 * On Windows this list is not a convenience: a process spawned without
 * `SystemRoot` cannot load `kernel32`-adjacent DLLs, one without `PATHEXT`
 * cannot resolve `git` to `git.exe`, and one without `TEMP` fails inside
 * anything that writes a scratch file. POSIX needs far less.
 */
const POSIX_SAFE_ENVIRONMENT_NAMES = ['PATH', 'HOME', 'LANG', 'LC_ALL', 'TERM'] as const

const WINDOWS_SAFE_ENVIRONMENT_NAMES = [
  'PATH',
  'PATHEXT',
  'COMSPEC',
  'SystemRoot',
  'SystemDrive',
  'windir',
  'TEMP',
  'TMP',
  'USERPROFILE',
  'HOMEDRIVE',
  'HOMEPATH',
  'APPDATA',
  'LOCALAPPDATA',
  'NUMBER_OF_PROCESSORS',
  'PROCESSOR_ARCHITECTURE',
] as const

/** Fallback `PATH` when the parent environment has none. */
const POSIX_FALLBACK_PATH = '/usr/bin:/bin'

/** Default `PATHEXT`, matching the Windows shell default. */
export const DEFAULT_PATHEXT = '.COM;.EXE;.BAT;.CMD;.VBS;.JS;.WS;.MSC'

export function isWindows(platform: NodeJS.Platform = process.platform): boolean {
  return platform === 'win32'
}

/**
 * True for a Windows named pipe such as `\\.\pipe\xerxes-abc`.
 *
 * A pipe is a kernel object, not a filesystem entry, so callers must not
 * `mkdir` its parent or `rm` it before listening — both fail, and the second
 * one used to abort daemon startup.
 */
export function isNamedPipePath(path: string): boolean {
  return /^\\\\[.?]\\pipe\\/i.test(path)
}

/**
 * Derive the per-project control-channel address from a project digest.
 *
 * POSIX gets a Unix socket under the Xerxes home; Windows gets a named pipe,
 * whose name is flat (no directories) and global to the machine, so the digest
 * carries the whole per-project distinction.
 */
export function controlChannelPath(
  socketDirectory: string,
  digest: string,
  platform: NodeJS.Platform = process.platform,
): string {
  if (isWindows(platform)) {
    return `${NAMED_PIPE_PREFIX}xerxes-${digest}`
  }
  // Kept as string concatenation rather than path.join so a Windows-hosted test
  // of the POSIX branch still produces a POSIX separator.
  return `${socketDirectory}/${digest}.sock`
}

/**
 * Machine-wide default control-channel address.
 *
 * This is the daemon-config fallback for hosts that do not pass a per-project
 * channel on the command line; the per-project address from
 * {@link controlChannelPath} is what the TUI and daemon actually use.
 */
export function defaultControlChannelPath(
  daemonDirectory: string,
  platform: NodeJS.Platform = process.platform,
): string {
  return isWindows(platform) ? `${NAMED_PIPE_PREFIX}xerxes-daemon` : `${daemonDirectory}/xerxes.sock`
}

/** The interactive shell a PTY session should launch when the caller names none. */
export function defaultInteractiveShell(
  env: Readonly<Record<string, string | undefined>> = process.env,
  platform: NodeJS.Platform = process.platform,
): string {
  if (isWindows(platform)) {
    // COMSPEC is set on every supported Windows host; the literal is the floor.
    return env.COMSPEC?.trim() || 'cmd.exe'
  }
  return env.SHELL?.trim() || '/bin/sh'
}

/**
 * Build the argv that runs `command` inside `shell`.
 *
 * `cmd.exe` takes `/d /s /c` (skip AutoRun, keep the outer quotes intact, then
 * run) and PowerShell takes `-Command`; neither understands `-c`, and passing
 * `-c` to `cmd.exe` silently starts an interactive shell that never exits.
 */
export function shellCommandArgv(
  shell: string,
  command: string,
  login: boolean,
  platform: NodeJS.Platform = process.platform,
): string[] {
  if (isWindows(platform)) {
    const name = basename(shell).toLowerCase()
    // A login shell has no Windows equivalent, so `login` is intentionally ignored.
    if (name.startsWith('powershell') || name.startsWith('pwsh')) {
      return command.trim() ? [shell, '-NoLogo', '-NoProfile', '-Command', command] : [shell, '-NoLogo', '-NoProfile']
    }
    return command.trim() ? [shell, '/d', '/s', '/c', command] : [shell, '/d']
  }
  const name = basename(shell)
  const supportsLogin = login && (name.endsWith('bash') || name.endsWith('zsh'))
  if (!command.trim()) {
    return [shell, ...(supportsLogin ? ['-l'] : [])]
  }
  return [shell, ...(supportsLogin ? ['-l'] : []), '-c', command]
}

/** Names copied from the parent environment into a sandboxed child. */
export function safeEnvironmentNames(platform: NodeJS.Platform = process.platform): readonly string[] {
  return isWindows(platform) ? WINDOWS_SAFE_ENVIRONMENT_NAMES : POSIX_SAFE_ENVIRONMENT_NAMES
}

/**
 * A usable `PATH` for a sandboxed child when the parent has none.
 *
 * The previous Windows value was the empty string, which resolved nothing at
 * all: every allow-listed command failed with ENOENT rather than being denied,
 * so the sandbox looked like it was working while executing nothing.
 */
export function fallbackExecutablePath(
  env: Readonly<Record<string, string | undefined>> = process.env,
  platform: NodeJS.Platform = process.platform,
): string {
  if (!isWindows(platform)) {
    return POSIX_FALLBACK_PATH
  }
  const root = env.SystemRoot?.trim() || env.windir?.trim() || 'C:\\Windows'
  return [`${root}\\system32`, root, `${root}\\system32\\Wbem`].join(';')
}

/**
 * Compare two environment-variable names the way the host OS does.
 *
 * Windows environment blocks are case-insensitive, so a block-list that only
 * rejects `NODE_OPTIONS` still lets `node_options` through and re-opens the
 * exact injection the list exists to close.
 */
export function environmentNamesMatch(
  left: string,
  right: string,
  platform: NodeJS.Platform = process.platform,
): boolean {
  return isWindows(platform) ? left.toLowerCase() === right.toLowerCase() : left === right
}

/** Normalize an environment-variable name into its block-list lookup key. */
export function environmentNameKey(name: string, platform: NodeJS.Platform = process.platform): string {
  return isWindows(platform) ? name.toLowerCase() : name
}

/**
 * Signal used to interrupt a foreground job in an interactive terminal.
 *
 * Windows has no SIGINT delivery to another process group — `process.kill(pid,
 * 'SIGINT')` is mapped to an immediate terminate, which kills the shell itself
 * rather than the command running inside it. Writing the Ctrl+C control
 * character into the terminal is the equivalent that leaves the shell alive.
 */
export const CTRL_C = '\u0003'

/** Whether an interrupt must be delivered as a terminal keystroke instead of a signal. */
export function interruptViaTerminalWrite(platform: NodeJS.Platform = process.platform): boolean {
  return isWindows(platform)
}
