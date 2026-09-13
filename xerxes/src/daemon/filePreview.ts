// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { realpath, stat } from 'node:fs/promises'
import { isAbsolute, relative, resolve } from 'node:path'

const MAX_BYTES = 128 * 1024

/** Explicit workspace reads never follow a symlink outside the selected workspace. */
export async function previewWorkspaceFile(cwd: string, path: unknown) {
  if (typeof path !== 'string' || !path || path.length > 4096 || path.includes('\0')) {
    throw new TypeError('Choose a valid workspace file path')
  }
  const root = await realpath(cwd)
  const file = await realpath(resolve(root, path))
  const local = relative(root, file)
  if (!local || local === '..' || local.startsWith('../') || isAbsolute(local)) {
    throw new Error('File preview is limited to the selected workspace')
  }
  if (!(await stat(file)).isFile()) throw new Error('Choose a regular file to preview')
  const blob = Bun.file(file)
  const truncated = blob.size > MAX_BYTES
  const bytes = new Uint8Array(await blob.slice(0, MAX_BYTES).arrayBuffer())
  let content: string
  try { content = new TextDecoder('utf-8', { fatal: true }).decode(bytes, { stream: truncated }) }
  catch { throw new Error('This file is not UTF-8 text and cannot be previewed') }
  if (content.includes('\0')) throw new Error('This binary file cannot be previewed as text')
  return { ok: true as const, path: local, content, truncated }
}
