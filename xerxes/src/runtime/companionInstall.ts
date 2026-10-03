// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { homedir } from 'node:os'
import { join } from 'node:path'

export const CLAUDE_CODE_PACKAGE = '@anthropic-ai/claude-code'

export const INSTALL_HELP = `Install optional Xerxes companion tools.

Usage:
  xerxes install --cloud-code [--force] [--dry-run]
  xerxes install --claude-code [--force] [--dry-run]

The Bun runtime replaces Xerxes' former managed Node.js runtime. The --node
option is therefore no longer supported.`

export interface InstallCommandOptions {
  readonly dryRun: boolean
  readonly force: boolean
}

export interface InstallCommandHost {
  readonly bunExecutable?: string
  readonly findExecutable?: (name: string) => string | null
  readonly run?: (argv: readonly string[]) => Promise<number>
  readonly write?: (message: string) => void
}

export interface InstallCommandResult {
  readonly command: readonly string[]
  readonly status: 'already-installed' | 'dry-run' | 'installed'
}

/** Raised when the install command cannot be parsed or its child process fails. */
export class InstallCommandError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'InstallCommandError'
  }
}

/** Parse the Bun CLI's supported companion-install options. */
export function parseInstallCommandOptions(args: readonly string[]): InstallCommandOptions {
  let hasInstallTarget = false
  let dryRun = false
  let force = false

  for (const argument of args) {
    switch (argument) {
      case '--claude-code':
      case '--cloud-code':
        hasInstallTarget = true
        break
      case '--dry-run':
        dryRun = true
        break
      case '--force':
        force = true
        break
      case '--node':
        throw new InstallCommandError('The Bun runtime replaces the managed Node.js installer; --node is no longer supported.')
      default:
        throw new InstallCommandError('Unknown install option: ' + argument)
    }
  }

  if (!hasInstallTarget) {
    throw new InstallCommandError('Choose an install target, for example `xerxes install --cloud-code`.')
  }
  return { dryRun, force }
}

/** Install the optional Claude Code companion through Bun, never through Python or npm. */
export async function runInstallCommand(
  args: readonly string[],
  host: InstallCommandHost = {},
): Promise<InstallCommandResult> {
  const options = parseInstallCommandOptions(args)
  const write = host.write ?? console.log
  const findExecutable = host.findExecutable ?? (name => Bun.which(name))
  const existing = findExecutable('claude')

  if (existing && !options.force) {
    write('Claude Code is already installed: ' + existing)
    write('Login or refresh credentials with: claude auth login')
    return { command: [], status: 'already-installed' }
  }

  const command = [
    host.bunExecutable ?? process.execPath,
    'add',
    '--global',
    ...(options.force ? ['--force'] : []),
    CLAUDE_CODE_PACKAGE,
  ]
  if (options.dryRun) {
    write('Would run: ' + shellCommand(command))
    return { command, status: 'dry-run' }
  }

  const run = host.run ?? runBunInstall
  const exitCode = await run(command)
  if (exitCode !== 0) {
    throw new InstallCommandError('Claude Code installation failed with exit code ' + exitCode + '.')
  }
  write('Claude Code installed.')
  write('Login with: claude auth login')
  return { command, status: 'installed' }
}

async function runBunInstall(argv: readonly string[]): Promise<number> {
  const child = Bun.spawn([...argv], {
    stdin: 'inherit',
    stderr: 'inherit',
    stdout: 'inherit',
  })
  return child.exited
}

function shellCommand(argv: readonly string[]): string {
  return argv.map(shellQuote).join(' ')
}

function shellQuote(value: string): string {
  return /^[A-Za-z0-9_@./:-]+$/.test(value) ? value : JSON.stringify(value)
}

/** Where Bun puts globally installed commands, so a Claude Code Xerxes installed is found. */
export function bunGlobalBin(environment: Readonly<Record<string, string | undefined>>, home: string): string {
  return join(environment.BUN_INSTALL?.trim() || join(home, '.bun'), 'bin')
}

export interface EnsureClaudeCodeHost {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly bunExecutable?: string
  /** Where `claude` is now, or undefined. */
  readonly find: () => string | undefined
  readonly run?: (argv: readonly string[], timeoutMs: number) => Promise<{ code: number; output: string }>
  readonly now?: () => number
  /** Records when a Xerxes-installed Claude Code was last updated. */
  readonly stateFile?: string
  /** Overrides XERXES_AUTO_INSTALL_CLAUDE_CODE (tests set it explicitly). */
  readonly autoInstall?: boolean
}

export type EnsureClaudeCodeResult = { readonly path: string; readonly status: 'present' | 'installed' }

const INSTALL_TIMEOUT_MS = 5 * 60_000
const UPDATE_EVERY_MS = 24 * 60 * 60_000
let installing: Promise<EnsureClaudeCodeResult> | undefined

/**
 * The `claude` command, installed through Bun when it is missing: choosing
 * Claude Code as the model is enough, with no separate install step. A copy
 * Xerxes installed (in Bun's global bin) is kept current in the background,
 * at most daily; one from Anthropic's installer or Homebrew updates itself
 * and is left alone. Set XERXES_AUTO_INSTALL_CLAUDE_CODE=0 to turn this off.
 * Concurrent callers share one install.
 */
export function ensureClaudeCode(host: EnsureClaudeCodeHost): Promise<EnsureClaudeCodeResult> {
  const environment = host.environment ?? process.env
  const found = host.find()
  if (found) {
    void updateManagedClaudeCode(found, host, environment).catch(error => console.warn(`Claude Code update check failed: ${error instanceof Error ? error.message : String(error)}`))
    return Promise.resolve({ path: found, status: 'present' })
  }
  // The switch is read from the process too: a caller's own environment (a
  // test's, a profile's) must not turn installs back on.
  const off = (value: string | undefined) => /^(0|off|false|no)$/i.test(value?.trim() ?? '')
  if (!(host.autoInstall ?? !(off(environment.XERXES_AUTO_INSTALL_CLAUDE_CODE) || off(process.env.XERXES_AUTO_INSTALL_CLAUDE_CODE)))) {
    return Promise.reject(new InstallCommandError("Claude Code is not installed and automatic install is off (XERXES_AUTO_INSTALL_CLAUDE_CODE). Install it from claude.com/code, sign in with 'claude', then retry."))
  }
  installing ??= (async () => {
    const run = host.run ?? runCaptured
    const result = await run([host.bunExecutable ?? process.execPath, 'add', '--global', CLAUDE_CODE_PACKAGE], INSTALL_TIMEOUT_MS)
    if (result.code !== 0) {
      throw new InstallCommandError(`Could not install Claude Code automatically (bun add exited ${result.code}): ${result.output.trim().split('\n').slice(-3).join(' ') || 'no output'}. Install it from claude.com/code, sign in with 'claude', then retry.`)
    }
    const path = host.find()
    if (!path) throw new InstallCommandError("Claude Code was installed but its 'claude' command was not found. Set CLAUDE_CODE_CLI to its path, then retry.")
    if (host.stateFile) await Bun.write(host.stateFile, JSON.stringify({ updatedAt: (host.now ?? Date.now)() }))
    return { path, status: 'installed' as const }
  })().finally(() => { installing = undefined })
  return installing
}

async function updateManagedClaudeCode(path: string, host: EnsureClaudeCodeHost, environment: Readonly<Record<string, string | undefined>>): Promise<void> {
  if (!host.stateFile || !path.startsWith(bunGlobalBin(environment, environment.HOME?.trim() || homedir()) + '/')) return
  const now = (host.now ?? Date.now)()
  const state: unknown = await Bun.file(host.stateFile).json().catch(() => ({}))
  const updatedAt = state && typeof state === 'object' && typeof (state as Record<string, unknown>).updatedAt === 'number' ? (state as { updatedAt: number }).updatedAt : 0
  if (now - updatedAt < UPDATE_EVERY_MS) return
  // Recorded first, so a slow or failing update is not retried on every request.
  await Bun.write(host.stateFile, JSON.stringify({ updatedAt: now }))
  const result = await (host.run ?? runCaptured)([host.bunExecutable ?? process.execPath, 'add', '--global', `${CLAUDE_CODE_PACKAGE}@latest`], INSTALL_TIMEOUT_MS)
  if (result.code !== 0) throw new Error(`bun add exited ${result.code}`)
}

async function runCaptured(argv: readonly string[], timeoutMs: number): Promise<{ code: number; output: string }> {
  const child = Bun.spawn([...argv], { stdin: 'ignore', stdout: 'pipe', stderr: 'pipe', timeout: timeoutMs })
  const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited])
  return { code, output: (stdout + '\n' + stderr).slice(-4000) }
}
