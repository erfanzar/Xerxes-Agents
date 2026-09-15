// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { randomUUID } from 'node:crypto'
import { dirname, isAbsolute } from 'node:path'
import { remoteTarget, type RemoteTarget } from './remote.js'

export interface WindowBounds { x: number; y: number; width: number; height: number }
export interface SavedWindow {
  workspace: string | null
  sessionId: string | null
  remote: RemoteTarget | null
  bounds: WindowBounds
  maximized: boolean
  fullscreen: boolean
}

export function parseWindowLayout(value: unknown): SavedWindow[] {
  if (!value || typeof value !== 'object' || !('version' in value) || value.version !== 1
    || !('windows' in value) || !Array.isArray(value.windows) || value.windows.length > 100) throw new Error('Invalid saved window layout')
  return value.windows.map((value: unknown) => {
    if (!value || typeof value !== 'object') throw new Error('Invalid saved window')
    const row = value as Record<string, unknown>, bounds = row.bounds as Record<string, unknown> | undefined
    if (!bounds || ['x', 'y', 'width', 'height'].some(key => typeof bounds[key] !== 'number' || !Number.isSafeInteger(bounds[key]) || Math.abs(bounds[key] as number) > 100_000)
      || (bounds.width as number) < 1 || (bounds.height as number) < 1) throw new Error('Invalid saved window bounds')
    if (row.workspace !== null && (typeof row.workspace !== 'string' || !isAbsolute(row.workspace) || /[\x00-\x1f]/.test(row.workspace))) throw new Error('Invalid saved workspace')
    if (row.sessionId !== null && (typeof row.sessionId !== 'string' || !/^[a-zA-Z0-9_-]{1,256}$/.test(row.sessionId))) throw new Error('Invalid saved session')
    if (typeof row.maximized !== 'boolean' || typeof row.fullscreen !== 'boolean') throw new Error('Invalid saved window mode')
    const remote = row.remote === null ? null : remoteTarget(row.remote)
    return { workspace: remote ? null : row.workspace as string | null, sessionId: row.sessionId as string | null, remote,
      bounds: { x: bounds.x as number, y: bounds.y as number, width: bounds.width as number, height: bounds.height as number }, maximized: row.maximized, fullscreen: row.fullscreen }
  })
}

/** Keep the entire restored frame on a connected work area, including a moved dock. */
export function visibleWindowBounds(bounds: WindowBounds, areas: readonly WindowBounds[]): WindowBounds {
  const overlap = (area: WindowBounds) => Math.max(0, Math.min(bounds.x + bounds.width, area.x + area.width) - Math.max(bounds.x, area.x))
    * Math.max(0, Math.min(bounds.y + bounds.height, area.y + area.height) - Math.max(bounds.y, area.y))
  const area = [...areas].sort((a, b) => overlap(b) - overlap(a))[0]
  if (!area) return bounds
  const width = Math.min(area.width, Math.max(760, bounds.width)), height = Math.min(area.height, Math.max(560, bounds.height))
  return { width, height, x: Math.min(Math.max(bounds.x, area.x), area.x + area.width - width), y: Math.min(Math.max(bounds.y, area.y), area.y + area.height - height) }
}

export function loadWindowLayout(file: string): SavedWindow[] | null {
  try { return parseWindowLayout(JSON.parse(readFileSync(file, 'utf8'))) }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') console.error('Could not restore desktop windows:', error)
    return null
  }
}

export function saveWindowLayout(file: string, windows: readonly SavedWindow[]): void {
  const validated = parseWindowLayout({ version: 1, windows })
  mkdirSync(dirname(file), { recursive: true, mode: 0o700 })
  const temporary = `${file}.${randomUUID()}.tmp`
  try {
    writeFileSync(temporary, JSON.stringify({ version: 1, windows: validated }) + '\n', { mode: 0o600 })
    renameSync(temporary, file)
  } finally { rmSync(temporary, { force: true }) }
}
