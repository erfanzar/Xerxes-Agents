// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, realpathSync, readFileSync, mkdirSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, isAbsolute } from 'node:path'
import { randomUUID } from 'node:crypto'

/** Absent file and unparseable file are different answers; only the caller knows which matters. */
function readWorkspaceRecord(file: string): { directories?: unknown; workspace?: unknown } | null {
  if (!existsSync(file)) return {}
  try {
    const value: unknown = JSON.parse(readFileSync(file, 'utf8'))
    return value && typeof value === 'object' ? (value as { directories?: unknown; workspace?: unknown }) : null
  } catch {
    return null
  }
}

/**
 * The recent-workspace list, or an empty one when the file cannot be read.
 *
 * Reading degrades instead of throwing, because this runs on the startup path.
 * `createWorkspaceWindow` saves BEFORE constructing the window, inside the
 * `app.whenReady()` restore loop, so a truncated or `null` `desktop.json` —
 * a torn write, a full disk, a hand-edit — used to throw during restore and
 * leave the user with no windows at all. The same throw escaped the
 * `desktop:workspaces` IPC handler. The sibling reader for this very file
 * (`loadWorkspace` in main.ts) already guards this way.
 *
 * Refusing to CLOBBER unparseable content is a separate, deliberate guarantee
 * and still belongs to `saveDesktopWorkspace` below.
 */
export function loadDesktopWorkspaces(file: string): string[] {
  const record = readWorkspaceRecord(file)
  if (!record) return []
  const paths: unknown[] = [...(Array.isArray(record.directories) ? record.directories : []), record.workspace]
  return [...new Set(paths.filter((path): path is string => typeof path === 'string' && isAbsolute(path) && !/[\x00-\x1f]/.test(path)))]
}

/**
 * The saved folders that exist on this computer: the sidebar's "This Mac"
 * sections. A server path (`/home/…`) that reached the list showed as an
 * empty "This Mac · prismcore" beside the real one. The file keeps every
 * entry, so a folder on an unplugged disk comes back with the disk.
 */
export function localDesktopWorkspaces(file: string, exists: (path: string) => boolean = existsSync): string[] {
  return loadDesktopWorkspaces(file).filter(path => exists(path))
}

export function saveDesktopWorkspace(file: string, directory: string): void {
  if (!isAbsolute(directory) || /[\0\r\n]/.test(directory)) throw new Error('Choose an absolute workspace folder')
  // A file we cannot parse is not an empty file: merging into it would silently
  // drop every workspace the user had. Refuse rather than rewrite, and let the
  // caller decide whether that is fatal — for window restore it is not.
  if (!readWorkspaceRecord(file)) throw new Error('Could not read the saved workspace list; leaving it untouched')
  if (existsSync(directory)) directory = realpathSync(directory)
  mkdirSync(dirname(file), { recursive: true, mode: 0o700 })
  const temporary = `${file}.${randomUUID()}.tmp`
  try {
    writeFileSync(temporary, `${JSON.stringify({workspace:directory, directories:[...new Set([...loadDesktopWorkspaces(file), directory])]})}\n`, {mode:0o600})
    renameSync(temporary, file)
  } finally { rmSync(temporary, {force:true}) }
}
