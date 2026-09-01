// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Daemon address derivation and launch for the desktop app.
 *
 * The address algorithm mirrors `daemonPaths()` in ui/gatewayClient.ts
 * byte-for-byte (sha256 of the canonical project dir, hex[:16], under
 * `$XERXES_HOME/daemon/projects`, `XERXES_DAEMON_SOCKET` wins) — drift means
 * the app finds no daemon and launches a second one nobody else will ever
 * talk to. `test/desktopSocketParity.test.ts` pins the mirror against the
 * original.
 *
 * The launch contract mirrors the gateway client's: bun from
 * `XERXES_TUI_BUN`/`XERXES_BUN`, entry from `XERXES_TUI_BUN_DAEMON`/
 * `XERXES_BUN_DAEMON` else colocated cli, `daemon --project-dir … --socket …
 * --pid-file …`, detached, stderr captured to a ring.
 */

import { type ChildProcess, execFileSync, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, realpathSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { controlChannelPath } from '../../ui/lib/hostPlatform.js'

export type Env = Readonly<Record<string, string | undefined>>

export interface DaemonAddress {
  readonly socketPath: string
  readonly pidPath: string
}

export function xerxesHome(env: Env): string {
  const configured = env.XERXES_HOME?.trim()
  return configured ? resolve(configured) : join(homedir(), '.xerxes')
}

/** Nearest git root when available, else realpath — the exact digest input. */
export function canonicalProjectDir(projectDir?: string): string {
  const raw = resolve(projectDir ?? process.cwd())
  try {
    const root = execFileSync('git', ['-C', raw, 'rev-parse', '--show-toplevel'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
    if (root) return realpathSync(root)
  } catch {
    // Not a git checkout; fall through.
  }
  try {
    return realpathSync(raw)
  } catch {
    return raw
  }
}

export function daemonAddress(
  projectDir: string,
  env: Env = process.env,
  platform: NodeJS.Platform = process.platform,
): DaemonAddress {
  const digest = createHash('sha256').update(projectDir, 'utf8').digest('hex').slice(0, 16)
  const base = join(xerxesHome(env), 'daemon', 'projects')
  const override = env.XERXES_DAEMON_SOCKET?.trim()
  return {
    socketPath: override || controlChannelPath(base, digest, platform),
    pidPath: join(base, `${digest}.pid`),
  }
}

export function bunBinaryOf(env: Env): string {
  return env.XERXES_TUI_BUN?.trim() || env.XERXES_BUN?.trim() || 'bun'
}

/** Directory of this module — the app root the runtime checkout hangs off. */
function appDirOf(): string {
  try {
    return dirname(fileURLToPath(import.meta.url))
  } catch {
    return process.cwd()
  }
}

export function daemonEntryOf(projectDir: string, env: Env, appDir: string = appDirOf()): string {
  const configured = env.XERXES_TUI_BUN_DAEMON?.trim() || env.XERXES_BUN_DAEMON?.trim()
  if (configured) {
    const entry = isAbsolute(configured) ? configured : resolve(projectDir, configured)
    if (!existsSync(entry)) {
      throw new Error(`Configured Bun daemon entry does not exist: ${entry}`)
    }
    return entry
  }
  // Workspace-relative first: a workspace carrying its own runtime wins.
  // App-relative next: the desktop shell ships inside the runtime checkout
  // (appDir is <checkout>/dist/desktop when built, <checkout>/src/desktop/main
  // from source), so the checkout that built the app can serve a daemon for
  // ANY workspace — without this, picking a workspace outside the checkout
  // leaves the app unable to launch a daemon for it.
  const candidates = [
    join(projectDir, 'xerxes', 'src', 'cli.ts'),
    join(projectDir, 'xerxes', 'dist', 'cli.js'),
    join(projectDir, 'src', 'cli.ts'),
    join(projectDir, 'dist', 'cli.js'),
    join(appDir, '..', '..', 'src', 'cli.ts'),
    join(appDir, '..', '..', 'dist', 'cli.js'),
    join(appDir, '..', '..', '..', 'src', 'cli.ts'),
    join(appDir, '..', '..', '..', 'dist', 'cli.js'),
    join(appDir, '..', '..', '..', '..', 'xerxes', 'src', 'cli.ts'),
    join(appDir, '..', '..', '..', '..', 'xerxes', 'dist', 'cli.js'),
    // Packaged bundle: the runtime dist is copied under Resources/runtime.
    join(appDir, '..', 'runtime', 'cli.js'),
  ]
  const found = [...new Set(candidates)].find(existsSync)
  if (!found) {
    throw new Error(
      'Could not locate the Bun daemon entry. Set XERXES_TUI_BUN_DAEMON (or XERXES_BUN_DAEMON) to the runtime cli path.',
    )
  }
  return found
}

export function daemonArgv(
  projectDir: string,
  socketPath: string,
  pidPath: string,
  env: Env,
): { binary: string; args: readonly string[] } {
  return {
    binary: bunBinaryOf(env),
    args: [
      daemonEntryOf(projectDir, env),
      'daemon',
      '--project-dir',
      projectDir,
      '--socket',
      socketPath,
      '--pid-file',
      pidPath,
    ],
  }
}

/** Launch the daemon detached; the child outlives this app by contract. */
export function launchDaemon(
  projectDir: string,
  socketPath: string,
  pidPath: string,
  env: Env,
  onStderr: (line: string) => void,
): ChildProcess {
  const { binary, args } = daemonArgv(projectDir, socketPath, pidPath, env)
  const child = spawn(binary, args, {
    stdio: ['ignore', 'ignore', 'pipe'],
    detached: true,
    env: env as NodeJS.ProcessEnv | undefined,
  })
  child.unref()
  child.once('error', () => {
    /* connect polling surfaces the failure; stderr ring keeps the why */
  })
  child.stderr?.setEncoding('utf8')
  child.stderr?.on('data', chunk => {
    for (const line of chunk.split('\n')) if (line) onStderr(line)
  })
  return child
}
