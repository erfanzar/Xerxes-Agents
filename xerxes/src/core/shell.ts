// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { basename } from 'node:path'

/** Executable lookup port (`Bun.which` by default) kept injectable for tests. */
export type WhichExecutable = (name: string) => string | null

/**
 * Pick the interactive shell for PTY sessions on this host.
 *
 * POSIX keeps the historical `$SHELL` â†’ `/bin/sh` chain. Native Windows has no
 * `$SHELL` or `/bin/sh`, so the chain becomes `$SHELL` â†’ `$COMSPEC` â†’
 * `pwsh` â†’ `powershell` â†’ `cmd.exe` (always present).
 */
export function resolveDefaultShell(
  environment: Readonly<Record<string, string | undefined>> = process.env,
  platform: NodeJS.Platform = process.platform,
  which: WhichExecutable = Bun.which,
): string {
  if (platform === 'win32') {
    const configured = environment.SHELL?.trim() || environment.COMSPEC?.trim()
    if (configured) return configured
    return which('pwsh') ?? which('powershell') ?? 'cmd.exe'
  }
  return environment.SHELL?.trim() || '/bin/sh'
}

function isPowerShell(shell: string): boolean {
  const name = basename(shell).toLowerCase()
  return name === 'pwsh' || name === 'pwsh.exe' || name === 'powershell' || name === 'powershell.exe'
}

/**
 * Build the argv used to run `command` (or open an interactive shell when the
 * command is blank) under the given shell.
 *
 * Windows shells do not understand POSIX `-l`/`-c`: `cmd.exe` takes
 * `/d /s /c`, PowerShell takes `-NoLogo -NoProfile -Command`. Login-shell
 * flags remain a POSIX-only concept.
 */
export function shellInvocation(
  shell: string,
  command: string,
  login: boolean,
  platform: NodeJS.Platform = process.platform,
): string[] {
  if (platform === 'win32') {
    if (!command.trim()) return [shell]
    return isPowerShell(shell)
      ? [shell, '-NoLogo', '-NoProfile', '-Command', command]
      : [shell, '/d', '/s', '/c', command]
  }
  const name = basename(shell)
  const supportsLogin = login && (name.endsWith('bash') || name.endsWith('zsh'))
  if (!command.trim()) return [shell, ...(supportsLogin ? ['-l'] : [])]
  return [shell, ...(supportsLogin ? ['-l'] : []), '-c', command]
}

/**
 * The keystroke that asks the foreground process of a terminal to interrupt.
 * POSIX sends SIGINT to the child process; ConPTY instead receives the Ctrl+C
 * control character so the console interrupts its foreground process group,
 * because Windows has no cross-process SIGINT.
 */
export function interruptTerminalInput(platform: NodeJS.Platform = process.platform): string {
  return platform === 'win32' ? '\u0003' : ''
}

/**
 * The input that ends an interactive shell session. POSIX shells honor Ctrl-D
 * (EOT); `cmd.exe` and PowerShell expect an explicit `exit` command.
 */
export function exitShellInput(platform: NodeJS.Platform = process.platform): string {
  return platform === 'win32' ? 'exit\r' : '\u0004'
}
