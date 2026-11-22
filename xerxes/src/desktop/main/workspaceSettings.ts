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

function canonicalWorkspaceKey(directory: string): string {
  const trimmed = directory.replace(/\/+$/, '') || '/'
  try {
    return existsSync(trimmed) ? realpathSync(trimmed).replace(/\/+$/, '') || '/' : trimmed
  } catch {
    return trimmed
  }
}

/**
 * Remove a remembered local workspace folder from the desktop sidebar list.
 * Never touches or deletes any user files on disk.
 */
export function removeDesktopWorkspace(file: string, directory: string, exists: (path: string) => boolean = existsSync): string[] {
  if (!isAbsolute(directory) || /[\0\r\n\x00-\x1f]/.test(directory)) throw new Error('Choose an absolute workspace folder')
  const record = readWorkspaceRecord(file)
  if (!record) throw new Error('Could not read the saved workspace list; leaving it untouched')
  const rawTarget = directory.replace(/\/+$/, '') || '/'
  const resolvedTarget = canonicalWorkspaceKey(directory)
  const matches = (candidate: string): boolean => {
    const clean = candidate.replace(/\/+$/, '') || '/'
    return clean === rawTarget || canonicalWorkspaceKey(candidate) === resolvedTarget
  }
  const remaining = loadDesktopWorkspaces(file).filter(path => !matches(path))
  const currentWorkspace = typeof record.workspace === 'string' && !matches(record.workspace)
    ? record.workspace
    : remaining[0]
  if (existsSync(file) || remaining.length > 0) {
    mkdirSync(dirname(file), { recursive: true, mode: 0o700 })
    const temporary = `${file}.${randomUUID()}.tmp`
    try {
      const payload = currentWorkspace
        ? { workspace: currentWorkspace, directories: remaining }
        : { directories: remaining }
      writeFileSync(temporary, `${JSON.stringify(payload)}\n`, { mode: 0o600 })
      renameSync(temporary, file)
    } finally {
      rmSync(temporary, { force: true })
    }
  }
  return remaining.filter(path => exists(path))
}

/**
 * Re-add a local workspace folder to the remembered sidebar list without
 * changing the default launch workspace (used when undoing sidebar removal).
 */
export function rememberDesktopWorkspace(file: string, directory: string, exists: (path: string) => boolean = existsSync): string[] {
  if (!isAbsolute(directory) || /[\0\r\n\x00-\x1f]/.test(directory)) throw new Error('Choose an absolute workspace folder')
  const record = readWorkspaceRecord(file)
  if (!record) throw new Error('Could not read the saved workspace list; leaving it untouched')
  const target = exists(directory) ? canonicalWorkspaceKey(directory) : (directory.replace(/\/+$/, '') || '/')
  const directories = [...new Set([...loadDesktopWorkspaces(file), target])]
  const workspace = typeof record.workspace === 'string' && isAbsolute(record.workspace) ? record.workspace : target
  mkdirSync(dirname(file), { recursive: true, mode: 0o700 })
  const temporary = `${file}.${randomUUID()}.tmp`
  try {
    writeFileSync(temporary, `${JSON.stringify({ workspace, directories })}\n`, { mode: 0o600 })
    renameSync(temporary, file)
  } finally {
    rmSync(temporary, { force: true })
  }
  return directories.filter(path => exists(path))
}
