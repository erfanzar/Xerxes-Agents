// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { exists, readdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'

export interface StatusOptions {
  readonly directory?: string | undefined
}

export interface StatusResult {
  readonly ok: boolean
  readonly message?: string
  readonly error?: string
}

export async function runStatusCommand(options: StatusOptions = {}): Promise<StatusResult> {
  try {
    const base = options.directory ?? xerxesHome()
    const [scheduler, memory, capabilities, telemetry, workspaces] = await Promise.all([
      countFiles(join(base, 'scheduler')),
      countFiles(join(base, 'governed-memory')),
      countManifests(join(base, 'capabilities')),
      countLines(join(base, 'telemetry', 'events.jsonl')),
      countWorkspaces(join(base, 'workspaces')),
    ])
    const lines = [
      `scheduler triggers: ${scheduler}`,
      `memory records: ${memory}`,
      `capability manifests: ${capabilities}`,
      `telemetry events: ${telemetry}`,
      `workspaces: ${workspaces}`,
    ]
    return { ok: true, message: lines.join('\n') }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}

async function countFiles(directory: string): Promise<number> {
  if (!(await exists(directory))) return 0
  const entries = await readdir(directory)
  return entries.length
}

async function countManifests(directory: string): Promise<number> {
  const path = join(directory, 'manifests.json')
  if (!(await exists(path))) return 0
  try {
    const parsed = JSON.parse(await readFile(path, 'utf8')) as unknown
    return Array.isArray(parsed) ? parsed.length : 0
  } catch {
    return 0
  }
}

async function countLines(path: string): Promise<number> {
  if (!(await exists(path))) return 0
  const contents = await readFile(path, 'utf8')
  return contents.split('\n').filter(line => line.length > 0).length
}

async function countWorkspaces(directory: string): Promise<number> {
  if (!(await exists(directory))) return 0
  const entries = await readdir(directory, { withFileTypes: true })
  return entries.filter(entry => entry.isDirectory()).length
}
