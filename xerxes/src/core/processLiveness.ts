// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Pid-liveness discipline for daemon-side subsystems that recover a resource
// from a process that may or may not still exist (currently the cron lease).
// A bare `process.kill(pid, 0)` is not enough on its own: pids are recycled, so
// an unrelated program can inherit the pid recorded in a stale file and either
// block recovery forever or, worse, be mistaken for ours. Proving identity
// always takes two steps — the process is alive, and its command line is the
// one we recorded.
//
// The TUI keeps its own copy of this discipline in `ui/gatewayClient.ts`
// alongside its daemon-signature matcher. That duplication is deliberate: the
// TUI bundle compiles standalone under `rootDir: src/ui` and must not reach
// into daemon-side modules, the same reason `ui/protocol` shadows `protocol`.

import { execFileSync } from 'node:child_process'

import { isWindows } from './hostPlatform.js'

/** True unless the kernel reports the pid as gone. */
export function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    return !isMissingProcessError(error)
  }
}

/**
 * ESRCH, the only errno that proves a pid no longer exists.
 *
 * EPERM deliberately does not count: a live process owned by another user
 * rejects the signal, and treating that as "gone" would let recovery steal a
 * resource that is still in use. This holds on Windows too, where a pid outside
 * the caller's rights surfaces as EPERM rather than ESRCH.
 */
export function isMissingProcessError(error: unknown): boolean {
  return error instanceof Error && 'code' in error && error.code === 'ESRCH'
}

/**
 * The full command line of a pid, or '' when it cannot be read.
 *
 * Windows has no `ps`. `Get-CimInstance Win32_Process` is the closest
 * equivalent that reports the command line, and it is queried by pid so the
 * output needs no parsing. An empty result is a normal outcome on both
 * platforms — the caller treats an unreadable command line as "identity not
 * proven" rather than as an error.
 */
export function processCommand(pid: number, platform: NodeJS.Platform = process.platform): string {
  const [command, args] = processCommandProbe(pid, platform)
  try {
    return execFileSync(command, args, {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      ...(isWindows(platform) ? { windowsHide: true } : {})
    }).trim()
  } catch {
    return ''
  }
}

/**
 * Argv that prints one pid's command line, with no shell involved.
 *
 * Exported for tests: the Windows form cannot be exercised by running it on a
 * POSIX CI host, so the argv itself is what gets asserted.
 */
export function processCommandProbe(
  pid: number,
  platform: NodeJS.Platform = process.platform
): readonly [string, string[]] {
  if (isWindows(platform)) {
    // The pid is interpolated into a PowerShell string, so it must not be able
    // to carry syntax. It cannot: the value is typed `number`, and no JavaScript
    // number stringifies to anything containing a quote, backtick, or
    // semicolon. Non-finite values are still normalized to a literal that
    // simply matches no process rather than reaching PowerShell as `NaN`.
    const target = Number.isFinite(pid) ? Math.trunc(pid) : -1
    return [
      'powershell.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `(Get-CimInstance Win32_Process -Filter "ProcessId=${target}").CommandLine`
      ]
    ]
  }
  return ['ps', ['-p', String(pid), '-o', 'command=']]
}
