// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { spawn } from 'node:child_process'
import { access, constants, rm, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { mkdtempSafe } from './safeTmp.js'
import { bunBinaryOf } from './spawn.js'

export interface LocalProviderCliStatus { readonly claude: boolean; readonly codex: boolean; readonly agy: boolean }
export type LocalCliName = 'claude' | 'codex' | 'agy'
export const LOCAL_CLI_PACKAGES = { claude: '@anthropic-ai/claude-code', codex: '@openai/codex' } as const
export const AGY_INSTALL_SCRIPT_URL = 'https://antigravity.google/cli/install.sh'
export const AGY_WINDOWS_INSTALL_SCRIPT_URL = 'https://antigravity.google/cli/install.ps1'
const LOCAL_CLI_LABELS: Readonly<Record<LocalCliName, string>> = {
  claude: LOCAL_CLI_PACKAGES.claude,
  codex: LOCAL_CLI_PACKAGES.codex,
  agy: 'Antigravity CLI (agy)',
}

/** Directories a Bun-installed or native CLI lands in, searched even when this process's PATH omits them. */
export function cliSearchDirectories(home = homedir()): readonly string[] {
  return [
    join(home, '.bun', 'bin'),
    join(home, '.local', 'bin'),
    join(home, '.claude', 'local'),
    join(home, '.gemini', 'antigravity', 'bin'),
    '/opt/homebrew/bin',
    '/usr/local/bin',
    '/usr/bin',
    '/bin',
  ]
}

async function executable(path: string): Promise<boolean> {
  try {
    await access(path, constants.X_OK)
    return true
  } catch {
    return false
  }
}

async function resolveBunBinary(env: NodeJS.ProcessEnv, home = homedir()): Promise<string> {
  const configured = bunBinaryOf(env)
  if (configured !== 'bun' && await executable(configured)) return configured
  const dirs = [...(env.PATH ?? '').split(':').filter(Boolean), ...cliSearchDirectories(home)]
  for (const dir of dirs) {
    const candidate = join(dir, 'bun')
    if (await executable(candidate)) return candidate
  }
  return configured
}

function runSubprocess(
  binary: string,
  args: readonly string[],
  env: NodeJS.ProcessEnv,
  timeoutMs = 300_000,
): Promise<{ readonly code: number; readonly stderr: string }> {
  return new Promise(resolve => {
    let stderr = ''
    let settled = false
    const child = spawn(binary, [...args], { env, stdio: ['ignore', 'ignore', 'pipe'] })
    const finish = (code: number, fallbackError?: string) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve({ code, stderr: (stderr.trim() || fallbackError || '').slice(0, 300) })
    }
    const timer = setTimeout(() => {
      child.kill('SIGKILL')
      finish(-1, 'timed out')
    }, timeoutMs)
    child.stderr?.on('data', chunk => {
      if (stderr.length < 4096) stderr += String(chunk)
    })
    child.once('error', error => finish(-1, error.message))
    child.once('close', code => finish(code ?? -1))
  })
}

/**
 * Where `claude`, `codex`, and `agy` actually are on this computer. PATH alone
 * is not enough: the desktop process rarely inherits ~/.bun/bin or ~/.local/bin,
 * so a CLI installed for the person would otherwise read as missing forever.
 */
export async function probeLocalProviderClis(environment: NodeJS.ProcessEnv = process.env, home = homedir()): Promise<LocalProviderCliStatus> {
  const directories = [...(environment.PATH ?? '').split(':').filter(Boolean), ...cliSearchDirectories(home)]
  const found = async (name: LocalCliName): Promise<boolean> => {
    for (const directory of directories) if (await executable(join(directory, name))) return true
    return false
  }
  return { claude: await found('claude'), codex: await found('codex'), agy: await found('agy') }
}

export interface LocalProviderSetupOptions {
  readonly probe?: () => Promise<LocalProviderCliStatus>
  /** Runs one install and resolves when it finished; rejects on a non-zero exit. */
  readonly install?: (name: LocalCliName) => Promise<void>
  readonly onProgress?: (message: string) => void
}

/**
 * Install only the provider CLIs this computer is missing, then verify them.
 * A missing CLI means the profile or ACP runtime cannot run here at all, so
 * this is done without asking; each install is reported individually and a
 * failure never blocks the others.
 */
export async function ensureLocalProviderClis(options: LocalProviderSetupOptions = {}): Promise<LocalProviderCliStatus> {
  const probe = options.probe ?? (() => probeLocalProviderClis())
  const install = options.install ?? defaultInstall
  const before = await probe()
  const failures: string[] = []
  for (const name of ['claude', 'codex', 'agy'] as const) {
    if (before[name]) continue
    options.onProgress?.(`Installing ${LOCAL_CLI_LABELS[name]}…`)
    try {
      await install(name)
    } catch (error) {
      failures.push(`${name}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  const after = await probe()
  if (failures.length) {
    throw new Error(`Provider CLI setup did not finish (${failures.join('; ')}). Install ${failures.length > 1 ? 'them' : 'it'} manually or check the network.`)
  }
  return after
}

async function defaultInstall(name: LocalCliName): Promise<void> {
  const env: NodeJS.ProcessEnv = { ...process.env, PATH: [...cliSearchDirectories(), process.env.PATH ?? ''].join(':') }
  if (name === 'agy') {
    await defaultInstallAgy(env)
    return
  }
  const bun = await resolveBunBinary(env)
  const result = await runSubprocess(bun, ['add', '--global', LOCAL_CLI_PACKAGES[name]], env)
  if (result.code !== 0) {
    throw new Error(result.stderr || `install exited ${result.code}`)
  }
}

/**
 * Install Google Antigravity CLI (`agy`) via its official installer script.
 * Downloads to a temporary file before invoking bash (never piping curl to sh).
 */
export async function defaultInstallAgy(env: NodeJS.ProcessEnv = process.env): Promise<void> {
  const dir = await mkdtempSafe('xerxes-agy-install-')
  try {
    if (process.platform === 'win32') {
      const script = join(dir, 'install.ps1')
      const response = await fetch(AGY_WINDOWS_INSTALL_SCRIPT_URL)
      if (!response.ok) throw new Error(`download failed (${response.status})`)
      await writeFile(script, await response.text(), 'utf8')
      const result = await runSubprocess('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script], env)
      if (result.code !== 0) {
        throw new Error(result.stderr || `agy installer exited ${result.code}`)
      }
      return
    }
    const script = join(dir, 'install.sh')
    const fetchResult = await runSubprocess('curl', ['-fsSL', AGY_INSTALL_SCRIPT_URL, '-o', script], env, 60_000)
    if (fetchResult.code !== 0) {
      throw new Error(fetchResult.stderr || `could not download agy installer (exit ${fetchResult.code})`)
    }
    const runResult = await runSubprocess('bash', [script], env)
    if (runResult.code !== 0) {
      throw new Error(runResult.stderr || `agy installer exited ${runResult.code}`)
    }
  } finally {
    await rm(dir, { recursive: true, force: true }).catch(() => undefined)
  }
}