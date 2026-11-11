// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The files a task wrote, kept with the task.
 *
 * At the end of each turn the run's safety floor lists what is in the task's
 * folder that was not there when it started, whichever tool or script wrote
 * it. The paths (and the turn each first appeared in) are kept in the task's
 * metadata, so the list outlives a runtime restart; sizes and times are read
 * from disk when the list is asked for, and a file that is gone drops out.
 */
import { stat } from 'node:fs/promises'
import { isAbsolute, join, normalize } from 'node:path'

import type { RunOutput } from '../runtime/runSafety.js'

export const TASK_OUTPUTS_KEY = 'task_outputs'
/** Beyond this the oldest entries are dropped: a task that writes thousands of files is not reading a list. */
export const MAX_TASK_OUTPUTS = 500

export interface TaskOutputEntry {
  /** Relative to the task's folder. */
  readonly path: string
  /** The turn the file first appeared after. */
  readonly turn: number
}

export interface TaskOutputFile extends TaskOutputEntry {
  readonly size: number
  readonly modifiedAt: number
}

function safeRelative(path: unknown): path is string {
  if (typeof path !== 'string' || !path || path.length > 4096 || isAbsolute(path) || /[\x00-\x1f\x7f]/.test(path)) return false
  return !normalize(path).split(/[\\/]/).includes('..')
}

/** The saved list, keeping only well-formed entries (metadata is persisted and may be old or edited). */
export function taskOutputsOf(metadata: Readonly<Record<string, unknown>>): TaskOutputEntry[] {
  const raw = metadata[TASK_OUTPUTS_KEY]
  if (!Array.isArray(raw)) return []
  return raw.flatMap(item => {
    if (!item || typeof item !== 'object') return []
    const { path, turn } = item as Record<string, unknown>
    return safeRelative(path) && typeof turn === 'number' && Number.isInteger(turn) && turn >= 0 ? [{ path, turn }] : []
  })
}

/**
 * Files nobody reads as a result: hidden files and folders (Xerxes's own
 * `.xerxes/`, editor and OS files) and Python's bytecode caches.
 */
function housekeeping(path: string): boolean {
  return path.split(/[\\/]/).some(segment => segment.startsWith('.') || segment === '__pycache__')
}

/** The list after a turn: new paths added with that turn, earlier ones keeping theirs. */
export function withTurnOutputs(previous: readonly TaskOutputEntry[], written: readonly RunOutput[], turn: number): TaskOutputEntry[] {
  const known = new Map(previous.map(entry => [entry.path, entry]))
  for (const file of written) if (safeRelative(file.path) && !housekeeping(file.path) && !known.has(file.path)) known.set(file.path, { path: file.path, turn })
  return [...known.values()].slice(-MAX_TASK_OUTPUTS)
}

/** The saved entries that still exist under `cwd`, with their size and time, newest first. */
export async function taskOutputFiles(cwd: string, entries: readonly TaskOutputEntry[]): Promise<TaskOutputFile[]> {
  const files = await Promise.all(entries.map(async entry => {
    try {
      const info = await stat(join(cwd, entry.path))
      return info.isFile() ? { ...entry, size: info.size, modifiedAt: Math.round(info.mtimeMs) } : undefined
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT' || (error as NodeJS.ErrnoException).code === 'ENOTDIR') return undefined
      throw error
    }
  }))
  return files.filter((file): file is TaskOutputFile => file !== undefined)
    .sort((a, b) => b.modifiedAt - a.modifiedAt || a.path.localeCompare(b.path))
}

export function taskOutputsPayload(cwd: string, files: readonly TaskOutputFile[]): Record<string, unknown> {
  return {
    ok: true,
    cwd,
    outputs: files.map(file => ({ path: file.path, turn: file.turn, size: file.size, modified_at: file.modifiedAt })),
  }
}
