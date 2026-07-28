// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { realpathSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'

import { controlChannelPath } from '../core/hostPlatform.js'

export interface DaemonPaths {
  readonly pidPath: string
  readonly socketPath: string
}

export function xerxesHome(environment = process.env): string {
  const configured = environment.XERXES_HOME?.trim()
  if (!configured) return join(homedir(), '.xerxes')
  if (configured === '~') return homedir()
  if (configured.startsWith('~/') || configured.startsWith('~\\')) {
    return resolve(homedir(), configured.slice(2))
  }
  return resolve(configured)
}

export function resolveProjectDirectory(projectDirectory = process.cwd()): string {
  const raw = resolve(projectDirectory)
  try {
    return realpathSync(raw)
  } catch {
    return raw
  }
}

/**
 * Match the current TypeScript gateway's per-project control-channel algorithm.
 *
 * POSIX hosts get a Unix socket under the Xerxes home. Windows has no Unix
 * sockets in this position, so the control channel is a named pipe instead —
 * same `node:net` API, different address form. The pid file stays on disk on
 * both, because pipe names are not enumerable the way a directory is.
 *
 * `platform` is a parameter so the Windows branch is reachable from a POSIX test
 * run; `ui/lib/hostPlatform.ts` must derive the identical address.
 */
export function daemonPaths(
  projectDirectory = process.cwd(),
  environment = process.env,
  platform: NodeJS.Platform = process.platform,
): DaemonPaths {
  const project = resolveProjectDirectory(projectDirectory)
  const digest = createHash('sha256').update(project, 'utf8').digest('hex').slice(0, 16)
  const base = join(xerxesHome(environment), 'daemon', 'projects')
  const configuredSocket = environment.XERXES_DAEMON_SOCKET?.trim()
  return {
    socketPath: configuredSocket || controlChannelPath(base, digest, platform),
    pidPath: join(base, `${digest}.pid`),
  }
}
