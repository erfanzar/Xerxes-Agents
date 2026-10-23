// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A narrow argv-only process port for `hub setup` and `hub link`. No shell is
 * involved locally; callers that run a command on a remote host through ssh
 * quote every argument with `shellQuote`. Tests inject a fake runner.
 */

export interface CommandResult {
  readonly code: number
  readonly stdout: string
  readonly stderr: string
  readonly timedOut: boolean
}
export interface CommandOptions { readonly timeoutMs?: number }
export type CommandRunner = (argv: readonly string[], options?: CommandOptions) => Promise<CommandResult>

export const spawnCommand: CommandRunner = async (argv, options = {}) => {
  const child = (() => {
    try { return Bun.spawn([...argv], { stdin: 'ignore', stdout: 'pipe', stderr: 'pipe' }) } catch (error) { return error instanceof Error ? error : new Error(String(error)) }
  })()
  if (child instanceof Error) return { code: 127, stdout: '', stderr: child.message, timedOut: false }
  let timedOut = false
  const timer = options.timeoutMs === undefined ? undefined : setTimeout(() => { timedOut = true; child.kill() }, options.timeoutMs)
  try {
    const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited])
    return { code, stdout, stderr, timedOut }
  } finally {
    clearTimeout(timer)
  }
}

/** POSIX single-quote one argument for a remote `sh`. */
export function shellQuote(value: string): string {
  return `'${value.replaceAll("'", `'\\''`)}'`
}

export function firstLine(text: string): string {
  return text.split('\n').map(line => line.trim()).find(Boolean)?.slice(0, 300) ?? 'no output'
}

export const realSleep = (milliseconds: number): Promise<void> => Bun.sleep(milliseconds)
