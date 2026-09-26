// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Keep the Claude Code CLI installed and current, once per runtime start.
 *
 * The `claude-code` provider runs the CLI as its model, so an old CLI means
 * old models and fixed bugs still hitting you — on this machine and on every
 * SSH host whose runtime starts the same way. Only runs when a profile uses
 * Claude Code; never blocks startup; each outcome is reported, not hidden.
 * `XERXES_CLAUDE_CODE_AUTOUPDATE=0` turns it off.
 *
 * (Codex needs nothing here: Xerxes talks to the ChatGPT backend itself and
 * only reads the Codex CLI's saved sign-in.)
 */

import { claudeExecutable } from '../auth/claudeCodeLogin.js'

/** Anthropic's official installer (https://docs.claude.com/en/docs/claude-code/setup). */
export const CLAUDE_CODE_INSTALLER = 'https://claude.ai/install.sh'

export type ClaudeCodeMaintenance =
  | { readonly action: 'skipped'; readonly reason: string }
  | { readonly action: 'updated' | 'installed'; readonly output: string }
  | { readonly action: 'failed'; readonly error: string }

/** Run a command with a deadline; resolve its exit code and combined output. */
export type CommandRunner = (argv: readonly string[], environment: Record<string, string>, timeoutMs: number) => Promise<{ code: number; output: string }>

export interface ClaudeCodeMaintenanceOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
  /** Whether any provider profile uses Claude Code. */
  readonly inUse: boolean
  readonly platform?: NodeJS.Platform
  readonly run?: CommandRunner
  /** Where to look for `claude`; defaults to the same lookup the provider uses. */
  readonly executable?: (environment: Readonly<Record<string, string | undefined>>) => string | undefined
}

const defaultRunner: CommandRunner = async (argv, environment, timeoutMs) => {
  const child = Bun.spawn([...argv], { env: environment, stdin: 'ignore', stdout: 'pipe', stderr: 'pipe' })
  const timer = setTimeout(() => child.kill(), timeoutMs)
  try {
    const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited])
    return { code, output: `${stdout}${stderr}`.trim() }
  } finally { clearTimeout(timer) }
}

const lastLines = (output: string) => output.split('\n').filter(line => line.trim()).slice(-3).join(' · ').slice(0, 400)

export async function maintainClaudeCode(options: ClaudeCodeMaintenanceOptions): Promise<ClaudeCodeMaintenance> {
  const environment = options.environment ?? process.env
  const setting = environment.XERXES_CLAUDE_CODE_AUTOUPDATE?.trim() ?? ''
  if (/^(0|off|false|no)$/i.test(setting)) return { action: 'skipped', reason: 'turned off by XERXES_CLAUDE_CODE_AUTOUPDATE' }
  if (!options.inUse) return { action: 'skipped', reason: 'no provider profile uses Claude Code' }
  const platform = options.platform ?? process.platform
  const run = options.run ?? defaultRunner
  const env = Object.fromEntries(Object.entries(environment).filter((entry): entry is [string, string] => typeof entry[1] === 'string'))
  const executable = (options.executable ?? claudeExecutable)(environment)
  try {
    if (executable) {
      const result = await run([executable, 'update'], env, 180_000)
      return result.code === 0 ? { action: 'updated', output: lastLines(result.output) } : { action: 'failed', error: `claude update exited ${result.code}: ${lastLines(result.output)}` }
    }
    if (platform === 'win32') return { action: 'skipped', reason: 'Claude Code is not installed; install it from https://docs.claude.com/en/docs/claude-code/setup' }
    const result = await run(['bash', '-c', `set -o pipefail; curl -fsSL ${CLAUDE_CODE_INSTALLER} | bash`], env, 300_000)
    return result.code === 0 ? { action: 'installed', output: lastLines(result.output) } : { action: 'failed', error: `Claude Code installer exited ${result.code}: ${lastLines(result.output)}` }
  } catch (error) {
    return { action: 'failed', error: error instanceof Error ? error.message : String(error) }
  }
}
