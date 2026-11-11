// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Open a workspace file in its default app, or show it in the file manager.
 * Node only: the Electron main process and the VS Code extension host call
 * it with their own port. The path resolves inside the task's folder and
 * must stay there, symlinks included.
 *
 * Opening is what double-clicking the file in Finder does, so a file that
 * would run as a program instead of opening as a document (a script, an app
 * or installer, an executable, a link that launches something) is never
 * opened: the agent can write such a file and name it like a report. It can
 * still be shown in Finder.
 */

import { realpath, stat } from 'node:fs/promises'
import { extname, isAbsolute, relative, resolve, sep } from 'node:path'

import type { FileActionRequest, FileActionResult } from '../fileActionShared.js'

/** What the host does with a checked, absolute path. */
export interface FileActionPort {
  /** Open in the default app; resolves to '' on success or the host's error text (Electron's `shell.openPath`). */
  openPath(path: string): Promise<string>
  showItemInFolder(path: string): void | Promise<void>
}

/** Extensions macOS, Windows or Linux run (or hand to an interpreter) rather than open as a document. */
const LAUNCHABLE = new Set([
  '.app', '.command', '.tool', '.terminal', '.workflow', '.action', '.scpt', '.scptd', '.applescript', '.osax',
  '.pkg', '.mpkg', '.dmg', '.prefpane', '.kext', '.webloc', '.inetloc', '.fileloc', '.url', '.desktop', '.appimage',
  '.sh', '.bash', '.zsh', '.csh', '.ksh', '.fish', '.py', '.pyw', '.pyc', '.rb', '.pl', '.php', '.lua', '.jar',
  '.exe', '.com', '.bat', '.cmd', '.ps1', '.psm1', '.vbs', '.vbe', '.js', '.jse', '.wsf', '.wsh', '.hta', '.msi', '.msp', '.lnk', '.scr', '.cpl', '.reg',
  '.deb', '.rpm', '.run', '.bin',
])

const escapes = (root: string, target: string): boolean => {
  const inside = relative(root, target)
  return !inside || isAbsolute(inside) || inside.split(sep)[0] === '..'
}

function reasonOf(error: unknown): string {
  const code = typeof error === 'object' && error !== null && 'code' in error ? error.code : undefined
  if (code === 'ENOENT' || code === 'ENOTDIR') return 'the file no longer exists'
  if (code === 'EACCES' || code === 'EPERM') return 'permission to read the file was denied'
  return 'the file could not be found'
}

/**
 * Document types the system opens by type whatever the file's mode. Files
 * copied from FAT or exFAT drives are often marked executable, and a report
 * should still open; an executable file of any other type would run.
 */
const DOCUMENTS = new Set([
  '.pdf', '.csv', '.tsv', '.txt', '.md', '.json', '.xml', '.yaml', '.yml', '.log', '.rtf', '.html', '.htm',
  '.doc', '.docx', '.xls', '.xlsx', '.xlsm', '.ppt', '.pptx', '.odt', '.ods', '.odp', '.pages', '.numbers', '.key',
  '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.heic', '.tif', '.tiff', '.bmp',
  '.mp3', '.m4a', '.wav', '.mp4', '.mov', '.zip',
])

/** True when opening the file would run it rather than show it. */
export function isLaunchable(path: string, mode: number): boolean {
  const extension = extname(path).toLowerCase()
  if (LAUNCHABLE.has(extension)) return true
  return process.platform !== 'win32' && (mode & 0o111) !== 0 && !DOCUMENTS.has(extension)
}

export async function performFileAction(request: FileActionRequest, port: FileActionPort): Promise<FileActionResult> {
  let target: string
  let mode: number
  try {
    const root = await realpath(request.cwd)
    target = await realpath(resolve(root, request.path))
    if (escapes(root, target)) return { ok: false, error: 'Only files inside this task’s folder can be opened from here.' }
    const info = await stat(target)
    if (!info.isFile()) return { ok: false, error: 'Only files can be opened from here.' }
    mode = info.mode
  } catch (error) {
    return { ok: false, error: `Could not find the file: ${reasonOf(error)}.` }
  }
  if (request.action === 'reveal') {
    await port.showItemInFolder(target)
    return { ok: true }
  }
  if (isLaunchable(target, mode)) {
    return { ok: false, error: 'This file can run as a program, so Xerxes does not open it. Use Show in Finder and open it yourself if you trust it.' }
  }
  const failure = await port.openPath(target)
  return failure ? { ok: false, error: `The file could not be opened: ${failure}` } : { ok: true }
}
