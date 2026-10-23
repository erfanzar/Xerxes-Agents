// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmod, mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { xerxesHome } from '../core/paths.js'

export interface HubPaths {
  readonly stateDir: string
  readonly controlSocketPath: string
  readonly pidPath: string
  readonly logsDir: string
  readonly workspacesPath: string
  readonly databasePath: string
  readonly ownershipLockPath: string
  readonly startLockPath: string
  readonly machinesPath: string
}

/** Resolve only; no files are created and daemon addresses are never reused. */
export function hubPaths(environment: Record<string, string | undefined> = process.env): HubPaths {
  const home = xerxesHome(environment)
  const stateDir = join(home, 'hub')
  return {
    stateDir,
    controlSocketPath: join(stateDir, 'control.sock'),
    pidPath: join(stateDir, 'hub.pid.json'),
    logsDir: join(stateDir, 'logs'),
    workspacesPath: join(stateDir, 'workspaces.json'),
    databasePath: join(stateDir, 'hub.sqlite'),
    ownershipLockPath: join(stateDir, 'ownership.lock'),
    startLockPath: join(stateDir, 'start.lock'),
    machinesPath: join(home, 'machines.json'),
  }
}

export async function ensureHubDirectories(paths: HubPaths = hubPaths()): Promise<void> {
  for (const directory of [paths.stateDir, paths.logsDir]) {
    await mkdir(directory, { recursive: true, mode: 0o700 })
    await chmod(directory, 0o700)
  }
}
