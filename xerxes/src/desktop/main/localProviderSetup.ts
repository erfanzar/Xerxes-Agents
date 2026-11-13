// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { access, constants } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

export interface LocalProviderCliStatus { readonly claude: boolean; readonly codex: boolean }
export type LocalCliName = 'claude' | 'codex'
export const LOCAL_CLI_PACKAGES = { claude: '@anthropic-ai/claude-code', codex: '@openai/codex' } as const

/** Directories a Bun-installed CLI lands in, searched even when this process's PATH omits them. */
export function cliSearchDirectories(home = homedir()): readonly string[] {
  return [join(home, '.bun', 'bin'), join(home, '.local', 'bin'), '/opt/homebrew/bin', '/usr/local/bin', '/usr/bin', '/bin']
}

async function executable(path: string): Promise<boolean> {
  try {
    await access(path, constants.X_OK)
    return true
  } catch {
    return false
  }
}

/**
 * Where `claude`/`codex` actually are on this computer. PATH alone is not
 * enough: the desktop process rarely inherits ~/.bun/bin, so a CLI installed
 * for the person would otherwise read as missing forever.
 */
export async function probeLocalProviderClis(environment: NodeJS.ProcessEnv = process.env, home = homedir()): Promise<LocalProviderCliStatus> {
  const directories = [...(environment.PATH ?? '').split(':').filter(Boolean), ...cliSearchDirectories(home)]
  const found = async (name: LocalCliName): Promise<boolean> => {
    for (const directory of directories) if (await executable(join(directory, name))) return true
    return false
  }
  return { claude: await found('claude'), codex: await found('codex') }
}

export interface LocalProviderSetupOptions {
  readonly probe?: () => Promise<LocalProviderCliStatus>
  /** Runs one install and resolves when it finished; rejects on a non-zero exit. */
  readonly install?: (name: LocalCliName) => Promise<void>
  readonly onProgress?: (message: string) => void
}

/**
 * Install only the provider CLIs this computer is missing, then verify them.
 * A missing CLI means the profile cannot run here at all, so this is done
 * without asking; each install is reported individually and a failure never
 * blocks the other one.
 */
export async function ensureLocalProviderClis(options: LocalProviderSetupOptions = {}): Promise<LocalProviderCliStatus> {
  const probe = options.probe ?? (() => probeLocalProviderClis())
  const install = options.install ?? defaultInstall
  const before = await probe()
  const failures: string[] = []
  for (const name of ['claude', 'codex'] as const) {
    if (before[name]) continue
    options.onProgress?.(`Installing ${LOCAL_CLI_PACKAGES[name]}…`)
    try {
      await install(name)
    } catch (error) {
      failures.push(`${name}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  const after = await probe()
  if (failures.length) {
    throw new Error(`Provider CLI setup did not finish (${failures.join('; ')}). Install ${failures.length === 2 ? 'them' : 'it'} manually or check the network.`)
  }
  return after
}

async function defaultInstall(name: LocalCliName): Promise<void> {
  const bun = process.execPath
  const child = Bun.spawn([bun, 'install', '-g', LOCAL_CLI_PACKAGES[name]], {
    env: { ...process.env, PATH: [...cliSearchDirectories(), process.env.PATH ?? ''].join(':') },
    stdout: 'ignore',
    stderr: 'pipe',
  })
  const code = await child.exited
  if (code !== 0) {
    const detail = (await new Response(child.stderr).text()).trim().slice(0, 300)
    throw new Error(detail || `install exited ${code}`)
  }
}