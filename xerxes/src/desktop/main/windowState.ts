// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { randomUUID } from 'node:crypto'
import { dirname, isAbsolute } from 'node:path'
import { remoteTarget, type RemoteTarget } from './remote.js'

export interface WindowBounds { x: number; y: number; width: number; height: number }
export interface SavedWindow {
  windowGroup?: string
  active?: boolean
  workspace: string | null
  sessionId: string | null
  remote: RemoteTarget | null
  bounds: WindowBounds
  maximized: boolean
  fullscreen: boolean
}

/** The most rows one layout file holds; parseWindowLayout rejects more. */
export const MAX_SAVED_WINDOWS = 100
/**
 * Views one window brings back on launch. Every view is a renderer process
 * and a daemon connection, and navigation opens a new one whenever no view
 * shows the requested session, so an uncapped layout grew across launches
 * until, past MAX_SAVED_WINDOWS, it stopped saving at all. The least recently
 * shown views beyond this are not restored; their sessions stay in the sidebar.
 */
export const MAX_RESTORED_VIEWS = 10

export function parseWindowLayout(value: unknown): SavedWindow[] {
  if (!value || typeof value !== 'object' || !('version' in value) || value.version !== 1
    || !('windows' in value) || !Array.isArray(value.windows) || value.windows.length > MAX_SAVED_WINDOWS) throw new Error('Invalid saved window layout')
  return value.windows.map((value: unknown) => {
    if (!value || typeof value !== 'object') throw new Error('Invalid saved window')
    const row = value as Record<string, unknown>, bounds = row.bounds as Record<string, unknown> | undefined
    if (!bounds || ['x', 'y', 'width', 'height'].some(key => typeof bounds[key] !== 'number' || !Number.isSafeInteger(bounds[key]) || Math.abs(bounds[key] as number) > 100_000)
      || (bounds.width as number) < 1 || (bounds.height as number) < 1) throw new Error('Invalid saved window bounds')
    if (row.workspace !== null && (typeof row.workspace !== 'string' || !isAbsolute(row.workspace) || /[\x00-\x1f]/.test(row.workspace))) throw new Error('Invalid saved workspace')
    if (row.sessionId !== null && (typeof row.sessionId !== 'string' || !/^[a-zA-Z0-9_-]{1,256}$/.test(row.sessionId))) throw new Error('Invalid saved session')
    if (typeof row.maximized !== 'boolean' || typeof row.fullscreen !== 'boolean') throw new Error('Invalid saved window mode')
    if (row.windowGroup !== undefined && (typeof row.windowGroup !== 'string' || !/^[a-zA-Z0-9_-]{1,64}$/.test(row.windowGroup))) throw new Error('Invalid window group')
    if (row.active !== undefined && typeof row.active !== 'boolean') throw new Error('Invalid active view')
    const remote = row.remote === null ? null : remoteTarget(row.remote)
    return { ...(typeof row.windowGroup === 'string' ? { windowGroup: row.windowGroup } : {}), ...(typeof row.active === 'boolean' ? { active: row.active } : {}), workspace: remote ? null : row.workspace as string | null, sessionId: row.sessionId as string | null, remote,
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

interface RecordedSurface { read: () => SavedWindow; usedAt: number }

/**
 * The layout the app writes: every open surface, ranked by when it was last
 * shown. It also remembers the last window to close, because the live set is
 * already empty when the app quits after that window closed (the normal quit
 * on Windows and Linux), and writing it as-is erased the whole layout.
 */
export class WindowLayoutRecorder {
  private readonly live = new Map<number, RecordedSurface>()
  private lastClosed: SavedWindow[] | null = null
  private clock = 0

  set(id: number, read: () => SavedWindow): void {
    this.live.set(id, { read, usedAt: ++this.clock })
    // A window opened afterwards is the new layout; the closed one is not.
    this.lastClosed = null
  }
  get(id: number): (() => SavedWindow) | undefined { return this.live.get(id)?.read }
  delete(id: number): void { this.live.delete(id) }
  /** Record that a surface was shown, for restore ranking and close fallback. */
  used(id: number): void {
    const entry = this.live.get(id)
    if (entry) entry.usedAt = ++this.clock
  }
  /** The most recently shown of `ids`, or undefined when none is recorded. */
  mostRecent(ids: Iterable<number>): number | undefined {
    let best: number | undefined, bestAt = -1
    for (const id of ids) {
      const usedAt = this.live.get(id)?.usedAt
      if (usedAt !== undefined && usedAt > bestAt) { best = id; bestAt = usedAt }
    }
    return best
  }
  /** Call while the window still exists, before its surfaces are torn down. */
  closing(windowGroup: string): void {
    this.lastClosed = this.rows().filter(row => row.windowGroup === windowGroup)
  }
  rows(): SavedWindow[] {
    if (!this.live.size) return this.lastClosed ?? []
    const entries = [...this.live.values()].map(entry => ({ row: entry.read(), usedAt: entry.usedAt }))
    type Entry = typeof entries[number]
    const rank = (a: Entry, b: Entry) => Number(b.row.active === true) - Number(a.row.active === true) || b.usedAt - a.usedAt
    const groups = new Map<string | undefined, Entry[]>()
    for (const entry of entries) groups.set(entry.row.windowGroup, [...groups.get(entry.row.windowGroup) ?? [], entry])
    const kept = new Set([...groups.values()].flatMap(group => group.sort(rank).slice(0, MAX_RESTORED_VIEWS))
      .sort(rank).slice(0, MAX_SAVED_WINDOWS))
    // Creation order is restoration order; ranking only decides what is kept.
    return entries.filter(entry => kept.has(entry)).map(entry => entry.row)
  }
}
