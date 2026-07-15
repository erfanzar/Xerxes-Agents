// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

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
