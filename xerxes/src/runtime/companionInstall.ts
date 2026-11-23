// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { accessSync, constants as fsConstants } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { homedir, tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'

export const CLAUDE_CODE_PACKAGE = '@anthropic-ai/claude-code'
export const CODEX_PACKAGE = '@openai/codex'
export const AGY_INSTALL_SCRIPT_URL = 'https://antigravity.google/cli/install.sh'
export const AGY_WINDOWS_INSTALL_SCRIPT_URL = 'https://antigravity.google/cli/install.ps1'

export type CompanionInstallTarget = 'claude-code' | 'codex' | 'agy'

export const INSTALL_HELP = `Install optional Xerxes companion tools.

Usage:
  xerxes install --cloud-code [--force] [--dry-run]
  xerxes install --claude-code [--force] [--dry-run]
  xerxes install --codex [--force] [--dry-run]
  xerxes install --agy [--force] [--dry-run]
  xerxes install --antigravity [--force] [--dry-run]
  xerxes install --all [--force] [--dry-run]

The Bun runtime replaces Xerxes' former managed Node.js runtime. The --node
option is therefore no longer supported.`

export interface InstallCommandOptions {
  readonly dryRun: boolean
  readonly force: boolean
  readonly targets?: readonly CompanionInstallTarget[]
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
  const targets = new Set<CompanionInstallTarget>()
  let dryRun = false
  let force = false

  for (const argument of args) {
    switch (argument) {
      case '--claude-code':
      case '--cloud-code':
        targets.add('claude-code')
        break
      case '--codex':
        targets.add('codex')
        break
      case '--agy':
      case '--antigravity':
        targets.add('agy')
        break
      case '--all':
        targets.add('claude-code')
        targets.add('codex')
        targets.add('agy')
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

  if (targets.size === 0) {
    throw new InstallCommandError('Choose an install target, for example `xerxes install --cloud-code` or `xerxes install --agy`.')
  }
  const list = [...targets]
  // Preserve exact option shape for the default single-target Claude Code call.
  if (list.length === 1 && list[0] === 'claude-code') {
    return { dryRun, force }
  }
  return { dryRun, force, targets: list }
}

function agyInstallArgv(): readonly string[] {
  if (process.platform === 'win32') {
    return ['powershell', '-NoProfile', '-Command', `irm ${AGY_WINDOWS_INSTALL_SCRIPT_URL} | iex`]
  }
  return [
    'sh',
    '-c',
    `tmp="$(mktemp "\${TMPDIR:-/tmp}/xerxes-agy-install.XXXXXX")"; trap 'rm -f "$tmp"' EXIT INT TERM; curl -fsSL ${AGY_INSTALL_SCRIPT_URL} -o "$tmp" && bash "$tmp"`,
  ]
}

/** Install optional companion tools (Claude Code, Codex, Antigravity CLI `agy`) without Python or npm. */
export async function runInstallCommand(
  args: readonly string[],
  host: InstallCommandHost = {},
): Promise<InstallCommandResult> {
  const options = parseInstallCommandOptions(args)
  const targets: readonly CompanionInstallTarget[] = options.targets ?? ['claude-code']
  const write = host.write ?? console.log
  const findExecutable = host.findExecutable ?? (name => Bun.which(name))
  const run = host.run ?? runBunInstall
  let lastCommand: readonly string[] = []
  let installedAny = false
  let dryRanAny = false

  for (const target of targets) {
    const binaryName = target === 'claude-code' ? 'claude' : target === 'codex' ? 'codex' : 'agy'
    const label = target === 'claude-code' ? 'Claude Code' : target === 'codex' ? 'Codex' : 'Antigravity CLI (agy)'
    const loginHint = target === 'claude-code' ? 'claude auth login' : target === 'codex' ? 'codex login' : 'agy acp --login'
    const existing = findExecutable(binaryName)

    if (existing && !options.force) {
      write(`${label} is already installed: ${existing}`)
      write(`Login or refresh credentials with: ${loginHint}`)
      continue
    }

    const command: readonly string[] = target === 'agy'
      ? agyInstallArgv()
      : [
          host.bunExecutable ?? process.execPath,
          'add',
          '--global',
          ...(options.force ? ['--force'] : []),
          target === 'claude-code' ? CLAUDE_CODE_PACKAGE : CODEX_PACKAGE,
        ]
    lastCommand = command

    if (options.dryRun) {
      write('Would run: ' + shellCommand(command))
      dryRanAny = true
      continue
    }

    const exitCode = await run(command)
    if (exitCode !== 0) {
      throw new InstallCommandError(`${label} installation failed with exit code ${exitCode}.`)
    }
    write(`${label} installed.`)
    write(`Login with: ${loginHint}`)
    installedAny = true
  }

  if (dryRanAny) return { command: lastCommand, status: 'dry-run' }
  if (installedAny) return { command: lastCommand, status: 'installed' }
  return { command: [], status: 'already-installed' }
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
  const disabled =
    off(environment.XERXES_AUTO_INSTALL_CLAUDE_CODE)
    || off(process.env.XERXES_AUTO_INSTALL_CLAUDE_CODE)
    || off(environment.XERXES_AUTO_INSTALL_COMPANIONS)
    || off(process.env.XERXES_AUTO_INSTALL_COMPANIONS)
  if (!(host.autoInstall ?? !disabled)) {
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

export interface EnsureAgyCliHost {
  readonly environment?: Readonly<Record<string, string | undefined>>
  /** Where `agy` is now, or undefined. */
  readonly find: () => string | undefined
  readonly run?: (argv: readonly string[], timeoutMs: number) => Promise<{ code: number; output: string }>
  /** Overrides XERXES_AUTO_INSTALL_AGY (tests set it explicitly). */
  readonly autoInstall?: boolean
}

export type EnsureAgyCliResult = { readonly path: string; readonly status: 'present' | 'installed' }

let installingAgy: Promise<EnsureAgyCliResult> | undefined

/**
 * The `agy` (Google Antigravity CLI) command, installed via Google's official
 * installer when missing so choosing Antigravity works without a manual install
 * step. Set XERXES_AUTO_INSTALL_AGY=0 to turn automatic installs off.
 * Concurrent callers share one install.
 */
export function ensureAgyCli(host: EnsureAgyCliHost): Promise<EnsureAgyCliResult> {
  const environment = host.environment ?? process.env
  const found = host.find()
  if (found) {
    return Promise.resolve({ path: found, status: 'present' })
  }
  const off = (value: string | undefined) => /^(0|off|false|no)$/i.test(value?.trim() ?? '')
  const disabled =
    off(environment.XERXES_AUTO_INSTALL_AGY)
    || off(process.env.XERXES_AUTO_INSTALL_AGY)
    || off(environment.XERXES_AUTO_INSTALL_COMPANIONS)
    || off(process.env.XERXES_AUTO_INSTALL_COMPANIONS)
  if (!(host.autoInstall ?? !disabled)) {
    return Promise.reject(new InstallCommandError("Antigravity CLI (agy) is not installed and automatic install is off (XERXES_AUTO_INSTALL_AGY). Install it from https://antigravity.google/docs/install, then retry."))
  }
  installingAgy ??= (async () => {
    const run = host.run ?? runAgyInstallerCaptured
    const result = await run(agyInstallArgv(), INSTALL_TIMEOUT_MS)
    if (result.code !== 0) {
      throw new InstallCommandError(`Could not install Antigravity CLI (agy) automatically (exited ${result.code}): ${result.output.trim().split('\n').slice(-3).join(' ') || 'no output'}. Install it from https://antigravity.google/docs/install, then retry.`)
    }
    const path = host.find()
    if (!path) {
      throw new InstallCommandError("Antigravity CLI (agy) was installed but its 'agy' command was not found on PATH. Ensure ~/.local/bin is on PATH or set XERXES_AGY_BIN, then retry.")
    }
    return { path, status: 'installed' as const }
  })().finally(() => { installingAgy = undefined })
  return installingAgy
}

export interface EnsureCodexCliHost {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly bunExecutable?: string
  /** Where `codex` is now, or undefined. */
  readonly find: () => string | undefined
  readonly run?: (argv: readonly string[], timeoutMs: number) => Promise<{ code: number; output: string }>
  /** Overrides XERXES_AUTO_INSTALL_CODEX (tests set it explicitly). */
  readonly autoInstall?: boolean
}

export type EnsureCodexCliResult = { readonly path: string; readonly status: 'present' | 'installed' }

let installingCodex: Promise<EnsureCodexCliResult> | undefined

/**
 * The `codex` (OpenAI Codex CLI) command, installed through Bun when missing
 * so Codex sessions work everywhere the Xerxes daemon runs. Set
 * XERXES_AUTO_INSTALL_CODEX=0 (or XERXES_AUTO_INSTALL_COMPANIONS=0) to opt out.
 * Concurrent callers share one install.
 */
export function ensureCodexCli(host: EnsureCodexCliHost): Promise<EnsureCodexCliResult> {
  const environment = host.environment ?? process.env
  const found = host.find()
  if (found) {
    return Promise.resolve({ path: found, status: 'present' })
  }
  const off = (value: string | undefined) => /^(0|off|false|no)$/i.test(value?.trim() ?? '')
  const disabled =
    off(environment.XERXES_AUTO_INSTALL_CODEX)
    || off(process.env.XERXES_AUTO_INSTALL_CODEX)
    || off(environment.XERXES_AUTO_INSTALL_COMPANIONS)
    || off(process.env.XERXES_AUTO_INSTALL_COMPANIONS)
  if (!(host.autoInstall ?? !disabled)) {
    return Promise.reject(new InstallCommandError("Codex CLI is not installed and automatic install is off (XERXES_AUTO_INSTALL_CODEX). Install @openai/codex, sign in with 'codex login', then retry."))
  }
  installingCodex ??= (async () => {
    const run = host.run ?? runCaptured
    const result = await run([host.bunExecutable ?? process.execPath, 'add', '--global', CODEX_PACKAGE], INSTALL_TIMEOUT_MS)
    if (result.code !== 0) {
      throw new InstallCommandError(`Could not install Codex CLI automatically (bun add exited ${result.code}): ${result.output.trim().split('\n').slice(-3).join(' ') || 'no output'}. Install @openai/codex and retry.`)
    }
    const path = host.find()
    if (!path) {
      throw new InstallCommandError("Codex CLI was installed but its 'codex' command was not found on PATH. Ensure ~/.bun/bin is on PATH, then retry.")
    }
    return { path, status: 'installed' as const }
  })().finally(() => { installingCodex = undefined })
  return installingCodex
}

export type DaemonCompanionCliOutcome =
  | { readonly status: 'present' | 'installed'; readonly path: string }
  | { readonly status: 'skipped'; readonly reason: string }
  | { readonly status: 'failed'; readonly error: string }

export interface DaemonCompanionClisResult {
  readonly claude: DaemonCompanionCliOutcome
  readonly codex: DaemonCompanionCliOutcome
  readonly agy: DaemonCompanionCliOutcome
}

export interface EnsureDaemonCompanionClisHost {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly bunExecutable?: string
  readonly home?: string
  readonly find?: (name: 'claude' | 'codex' | 'agy') => string | undefined
  readonly run?: (argv: readonly string[], timeoutMs: number) => Promise<{ code: number; output: string }>
  readonly autoInstall?: boolean
}

/** Standard directories where `claude`, `codex`, and `agy` land even when omitted from daemon PATH. */
export function companionCliSearchDirectories(
  environment: Readonly<Record<string, string | undefined>> = process.env,
  home: string = environment.HOME?.trim() || homedir(),
): readonly string[] {
  return [
    bunGlobalBin(environment, home),
    join(home, '.local', 'bin'),
    join(home, '.claude', 'local'),
    join(home, '.gemini', 'antigravity', 'bin'),
    '/opt/homebrew/bin',
    '/usr/local/bin',
    '/usr/bin',
    '/bin',
  ]
}

export function findCompanionCliBinary(
  name: 'claude' | 'codex' | 'agy',
  environment: Readonly<Record<string, string | undefined>> = process.env,
  home: string = environment.HOME?.trim() || homedir(),
): string | undefined {
  const pathDirs = (environment.PATH ?? '').split(delimiter).map(dir => dir.trim()).filter(Boolean)
  const searchDirs = [...new Set([...pathDirs, ...companionCliSearchDirectories(environment, home)])]
  const extensions = process.platform === 'win32' ? ['', '.exe', '.cmd', '.bat'] : ['']
  const mode = process.platform === 'win32' ? fsConstants.F_OK : fsConstants.X_OK
  for (const dir of searchDirs) {
    for (const ext of extensions) {
      const candidate = join(dir, `${name}${ext}`)
      try {
        accessSync(candidate, mode)
        return candidate
      } catch {
        // Keep searching standard install directories.
      }
    }
  }
  return undefined
}

/**
 * Ensure `claude`, `codex`, and `agy` are installed wherever a Xerxes daemon runs.
 * Runs each missing install independently so one failure never blocks the others,
 * and shares the single-flight installers with on-demand ACP / provider calls.
 */
export async function ensureDaemonCompanionClis(
  host: EnsureDaemonCompanionClisHost = {},
): Promise<DaemonCompanionClisResult> {
  const environment = host.environment ?? process.env
  const home = host.home ?? (environment.HOME?.trim() || homedir())
  const find = host.find ?? ((name: 'claude' | 'codex' | 'agy') => findCompanionCliBinary(name, environment, home))

  const runOne = async (
    task: () => Promise<{ readonly path: string; readonly status: 'present' | 'installed' }>,
  ): Promise<DaemonCompanionCliOutcome> => {
    try {
      const res = await task()
      return { status: res.status, path: res.path }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      if (/automatic install is off/i.test(message)) {
        return { status: 'skipped', reason: message }
      }
      return { status: 'failed', error: message }
    }
  }

  const [claude, codex, agy] = await Promise.all([
    runOne(() => ensureClaudeCode({
      environment,
      ...(host.bunExecutable ? { bunExecutable: host.bunExecutable } : {}),
      ...(host.run ? { run: host.run } : {}),
      ...(host.autoInstall !== undefined ? { autoInstall: host.autoInstall } : {}),
      find: () => find('claude'),
    })),
    runOne(() => ensureCodexCli({
      environment,
      ...(host.bunExecutable ? { bunExecutable: host.bunExecutable } : {}),
      ...(host.run ? { run: host.run } : {}),
      ...(host.autoInstall !== undefined ? { autoInstall: host.autoInstall } : {}),
      find: () => find('codex'),
    })),
    runOne(() => ensureAgyCli({
      environment,
      ...(host.run ? { run: host.run } : {}),
      ...(host.autoInstall !== undefined ? { autoInstall: host.autoInstall } : {}),
      find: () => find('agy'),
    })),
  ])

  return { claude, codex, agy }
}

async function runAgyInstallerCaptured(argv: readonly string[], timeoutMs: number): Promise<{ code: number; output: string }> {
  if (process.platform === 'win32') {
    const dir = await mkdtemp(join(tmpdir(), 'xerxes-agy-install-'))
    try {
      const script = join(dir, 'install.ps1')
      const response = await fetch(AGY_WINDOWS_INSTALL_SCRIPT_URL)
      if (!response.ok) return { code: 1, output: `download failed (${response.status})` }
      await Bun.write(script, await response.text())
      return await runCaptured(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script], timeoutMs)
    } finally {
      await rm(dir, { recursive: true, force: true }).catch(() => undefined)
    }
  }
  return runCaptured(argv, timeoutMs)
}
