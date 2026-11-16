// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Format Python with Ruff on the machine that holds the workspace. The
 * project's own Ruff (its venv) wins so its pyproject settings and version
 * apply; then a user-wide one on PATH. Nothing is installed: without Ruff
 * the caller gets an actionable error.
 */

import { stat } from 'node:fs/promises'
import { join } from 'node:path'

import { WorkspacePathResolver } from '../tools/pathSafety.js'

export interface FormatRequest { readonly filePath: string; readonly text: string }
export interface FormatResult { readonly text: string; readonly formatter: string }
const OUTPUT_LIMIT = 4 * 1024 * 1024

async function isFile(path: string): Promise<boolean> {
  try { return (await stat(path)).isFile() } catch { return false }
}

export async function findRuff(root: string, which: (name: string) => string | null = Bun.which): Promise<string | undefined> {
  for (const candidate of ['.venv/bin/ruff', 'venv/bin/ruff', '.venv/Scripts/ruff.exe', 'venv/Scripts/ruff.exe']) {
    const path = join(root, candidate)
    if (await isFile(path)) return path
  }
  const home = process.env.HOME
  for (const path of home ? [join(home, '.local', 'bin', 'ruff')] : []) if (await isFile(path)) return path
  return which('ruff') ?? undefined
}

export async function ruffFormat(root: string, request: FormatRequest, signal?: AbortSignal, ruffPath?: string): Promise<FormatResult> {
  if (typeof request.text !== 'string' || request.text.length > 1_048_576) throw new Error('Text to format must be at most 1 MiB')
  // The path only names the file for Ruff's config lookup; it must stay inside the workspace.
  const path = await new WorkspacePathResolver(root).resolve(request.filePath)
  if (!/\.pyi?$/.test(path)) throw new Error('Ruff formats Python files only')
  const ruff = ruffPath ?? await findRuff(root)
  if (!ruff) throw new Error('Ruff is not installed for this project. Install it in the project venv (pip install ruff) or on PATH, then format again.')
  signal?.throwIfAborted()
  const child = Bun.spawn([ruff, 'format', '--stdin-filename', path, '-'], {
    cwd: root, stdin: new Blob([request.text]), stdout: 'pipe', stderr: 'pipe',
    env: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: process.env.HOME ?? '' },
  })
  const kill = () => child.kill()
  signal?.addEventListener('abort', kill, { once: true })
  const timer = setTimeout(kill, 20_000)
  try {
    const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited])
    signal?.throwIfAborted()
    if (code !== 0) {
      const reason = stderr.trim().split('\n').filter(Boolean).at(-1) ?? `exit ${code}`
      throw new Error(`Ruff could not format this file: ${reason.slice(0, 400)}`)
    }
    if (stdout.length > OUTPUT_LIMIT) throw new Error('Ruff output is too large')
    return { text: stdout, formatter: ruff }
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', kill)
  }
}
