// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { realpath, stat } from 'node:fs/promises'
import { isAbsolute, relative, resolve } from 'node:path'

const MAX_BYTES = 128 * 1024

/** The version a save must match: a hash of the file's bytes. */
const versionOf = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex')

/** Canonical file inside the workspace; symlinks out of it and non-files are refused. */
async function workspaceFile(cwd: string, path: unknown, action: 'preview' | 'editing'): Promise<{ file: string; local: string }> {
  if (typeof path !== 'string' || !path || path.length > 4096 || path.includes('\0')) {
    throw new TypeError('Choose a valid workspace file path')
  }
  const root = await realpath(cwd)
  const file = await realpath(resolve(root, path))
  const local = relative(root, file)
  if (!local || local === '..' || local.startsWith('../') || isAbsolute(local)) {
    throw new Error(`File ${action} is limited to the selected workspace`)
  }
  if (!(await stat(file)).isFile()) throw new Error(`Choose a regular file to ${action === 'preview' ? 'preview' : 'edit'}`)
  return { file, local }
}

/** Explicit workspace reads never follow a symlink outside the selected workspace. */
export async function previewWorkspaceFile(cwd: string, path: unknown) {
  const { file, local } = await workspaceFile(cwd, path, 'preview')
  const blob = Bun.file(file)
  const truncated = blob.size > MAX_BYTES
  const bytes = new Uint8Array(await blob.slice(0, MAX_BYTES).arrayBuffer())
  let content: string
  try { content = new TextDecoder('utf-8', { fatal: true }).decode(bytes, { stream: truncated }) }
  catch { throw new Error('This file is not UTF-8 text and cannot be previewed') }
  if (content.includes('\0')) throw new Error('This binary file cannot be previewed as text')
  // Only a whole file can be edited, so only a whole file has a version.
  return { ok: true as const, path: local, content, truncated, ...(truncated ? {} : { version: versionOf(bytes) }) }
}

/**
 * Save an edit made in the app. The file must already exist inside the
 * workspace, be text the preview showed whole, and still be the version the
 * editor opened: if anything (the agent, another editor) changed it since,
 * nothing is written and the caller is told to reload. Writing in place keeps
 * the file's mode and identity.
 */
export async function writeWorkspaceFile(cwd: string, path: unknown, content: unknown, baseVersion: unknown) {
  if (typeof content !== 'string') throw new TypeError('content must be text')
  if (typeof baseVersion !== 'string' || !/^[0-9a-f]{64}$/.test(baseVersion)) throw new TypeError('base_version must be the version the file was opened at')
  const encoded = new TextEncoder().encode(content)
  if (encoded.byteLength > MAX_BYTES) throw new Error('Files over 128 KiB cannot be edited here')
  if (content.includes('\0')) throw new Error('Text files cannot contain NUL characters')
  const { file, local } = await workspaceFile(cwd, path, 'editing')
  const current = new Uint8Array(await Bun.file(file).arrayBuffer())
  if (versionOf(current) !== baseVersion) {
    return { ok: false as const, conflict: true, path: local, error: 'This file changed on disk since you opened it. Reload it to see the new version, then edit again.' }
  }
  await Bun.write(file, encoded)
  return { ok: true as const, path: local, version: versionOf(encoded) }
}
