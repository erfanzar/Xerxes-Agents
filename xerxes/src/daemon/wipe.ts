// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Global memory and history wipes behind the daemon's destructive RPCs.
 *
 * Both wipes are deliberately global: the user asked to forget everything, so
 * every scope — global, every project, and the live in-memory tiers — is
 * cleared in one call. Counts are returned so the confirmation can report
 * what was actually removed. Live sessions keep running after a history wipe;
 * their transcripts simply re-save on the next turn.
 */

import { readdir, rm, stat } from 'node:fs/promises'
import { join } from 'node:path'

export interface WipeCounts {
  bytes: number
  files: number
}

export interface WipeResult {
  ok: true
  removed: WipeCounts
}

/** Human-readable byte size for the wipe confirmation message. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KiB', 'MiB', 'GiB']
  let value = bytes
  let unit = 'B'
  for (const next of units) {
    if (value < 1024) break
    value /= 1024
    unit = next
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${unit}`
}

/** Recursively sum file count and byte size of a path that may not exist. */
async function measure(path: string): Promise<WipeCounts> {
  const entry = await stat(path).catch(() => undefined)
  if (!entry) return { bytes: 0, files: 0 }
  if (entry.isFile()) return { bytes: entry.size, files: 1 }
  if (!entry.isDirectory()) return { bytes: 0, files: 0 }

  let counts: WipeCounts = { bytes: 0, files: 0 }
  for (const name of await readdir(path)) {
    const child = await measure(join(path, name))
    counts = { bytes: counts.bytes + child.bytes, files: counts.files + child.files }
  }
  return counts
}

function combine(results: readonly WipeCounts[]): WipeCounts {
  return results.reduce<WipeCounts>(
    (total, item) => ({ bytes: total.bytes + item.bytes, files: total.files + item.files }),
    { bytes: 0, files: 0 },
  )
}

/** Remove a path and report what it held; a missing path is a zero-count no-op. */
async function removeTree(path: string): Promise<WipeCounts> {
  const counts = await measure(path)
  await rm(path, { force: true, recursive: true })
  return counts
}

/**
 * Every directory that holds agent memory: the cross-project global store, the
 * per-agent self-memory root, every known project memory root, and the SQLite
 * tiers (`.xerxes_memory/memory.db`, `.xerxes_memory/vectors.db`) beside each
 * of those roots and under the current project.
 */
export function memoryWipePaths(home: string, projectRoot: string | undefined): string[] {
  const paths = [join(home, 'memory'), join(home, 'agent_memory'), join(home, 'projects')]

  const sqliteRoots = [home]
  if (projectRoot) sqliteRoots.push(projectRoot)
  for (const root of sqliteRoots) {
    paths.push(join(root, '.xerxes_memory'))
  }
  return [...new Set(paths)]
}

/** Run the memory wipe, returning aggregate removal counts. */
export async function wipeMemoryStores(
  home: string,
  projectRoot: string | undefined,
): Promise<WipeResult> {
  const counts = await Promise.all(memoryWipePaths(home, projectRoot).map(removeTree))
  return { ok: true, removed: combine(counts) }
}

/**
 * History wipe targets: the persisted transcript store and the snapshot
 * shadow copies. The search index is rebuilt from transcripts on the next
 * search, so removing the transcripts is sufficient.
 */
export function historyWipePaths(sessionsDirectory: string, snapshotsDirectory: string): string[] {
  return [sessionsDirectory, snapshotsDirectory]
}

/** Run the history wipe, returning aggregate removal counts. */
export async function wipeHistoryStores(
  sessionsDirectory: string,
  snapshotsDirectory: string,
): Promise<WipeResult> {
  const counts = await Promise.all(
    historyWipePaths(sessionsDirectory, snapshotsDirectory).map(removeTree),
  )
  return { ok: true, removed: combine(counts) }
}
