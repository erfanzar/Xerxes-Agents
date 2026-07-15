// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { realpathSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'

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

/** Match the current TypeScript gateway's per-project Unix socket algorithm. */
export function daemonPaths(projectDirectory = process.cwd(), environment = process.env): DaemonPaths {
  const project = resolveProjectDirectory(projectDirectory)
  const digest = createHash('sha256').update(project, 'utf8').digest('hex').slice(0, 16)
  const base = join(xerxesHome(environment), 'daemon', 'projects')
  const configuredSocket = environment.XERXES_DAEMON_SOCKET?.trim()
  return {
    socketPath: configuredSocket || join(base, `${digest}.sock`),
    pidPath: join(base, `${digest}.pid`),
  }
}
