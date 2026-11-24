// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync } from 'node:fs'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { AGY_OUTER_HOST_ENV_KEYS, hasOuterAgyHostEnv } from '../../auth/agyRuntimeEnv.js'

/**
 * Strip stale TMPDIR/TEMP/TMP variables inherited from ephemeral launcher shells
 * whose temporary directory was deleted after the launcher exited.
 */
export function sanitizeTmpdirEnv(env: NodeJS.ProcessEnv = process.env): void {
  for (const key of ['TMPDIR', 'TEMP', 'TMP'] as const) {
    const value = env[key]
    if (value && !existsSync(value)) delete env[key]
  }
}

/**
 * Strip a parent Antigravity sandbox's private variables (its CSRF token,
 * language-server address, app-data dir) from the Electron desktop main process
 * so a child `agy acp` never borrows another agy instance's internals. SSH
 * session variables are left alone: how agy stores its sign-in is agy's call.
 */
export function sanitizeDesktopInheritedEnv(env: NodeJS.ProcessEnv = process.env): void {
  const outerHostDetected = hasOuterAgyHostEnv(env)
  for (const key of AGY_OUTER_HOST_ENV_KEYS) {
    delete env[key]
  }
  if (outerHostDetected) {
    delete env.GEMINI_HOME
  }
}

/**
 * Create a temporary directory resilient to a deleted TMPDIR: if the primary
 * `os.tmpdir()` directory no longer exists on disk, fall back to `/tmp` on
 * POSIX instead of crashing with ENOENT.
 */
export async function mkdtempSafe(prefix: string): Promise<string> {
  const envTmp = process.env.TMPDIR?.trim()
  const primary = envTmp || tmpdir()
  try {
    return await mkdtemp(join(primary, prefix))
  } catch (error) {
    const code = (error as NodeJS.ErrnoException | undefined)?.code
    if (code !== 'ENOENT' || process.platform === 'win32' || primary === '/tmp') throw error
    sanitizeTmpdirEnv()
    return await mkdtemp(join('/tmp', prefix))
  }
}
