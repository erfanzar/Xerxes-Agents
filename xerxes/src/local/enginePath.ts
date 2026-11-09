// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { homedir } from 'node:os'
import { delimiter, join } from 'node:path'

type Environment = Readonly<Record<string, string | undefined>>

/**
 * PATH plus the folders engine installers use. An app opened from Finder or
 * the Dock gets launchd's PATH (/usr/bin:/bin:/usr/sbin:/sbin), so on PATH
 * alone the desktop's daemon would not find Homebrew's `llama-server`,
 * `uv tool install mlx-lm`'s `mlx_lm.server` or LM Studio's `lms`.
 */
export function engineSearchPath(environment: Environment = process.env): string {
  const home = environment.HOME?.trim() || homedir()
  const extra = ['/opt/homebrew/bin', '/usr/local/bin', join(home, '.local', 'bin'), join(home, '.lmstudio', 'bin')]
  return [...new Set([...(environment.PATH ?? '').split(delimiter), ...extra].filter(Boolean))].join(delimiter)
}

/** An engine binary on `engineSearchPath`. */
export function whichEngine(name: string, environment: Environment = process.env): string | undefined {
  return Bun.which(name, { PATH: engineSearchPath(environment) }) ?? undefined
}
