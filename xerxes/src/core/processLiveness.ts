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

/** True unless the kernel reports the pid as gone. */
export function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    return !isMissingProcessError(error)
  }
}

/** ESRCH, the only errno that proves a pid no longer exists. */
export function isMissingProcessError(error: unknown): boolean {
  return error instanceof Error && 'code' in error && error.code === 'ESRCH'
}

/** The full command line of a pid, or '' when it cannot be read. */
export function processCommand(pid: number): string {
  try {
    return execFileSync('ps', ['-p', String(pid), '-o', 'command='], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore']
    }).trim()
  } catch {
    return ''
  }
}
