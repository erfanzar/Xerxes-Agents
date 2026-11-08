// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Whether an installed engine is new enough for a model. A llama.cpp build
 * older than the one that added a model's runtime can load the file and write
 * gibberish without an error, so setup checks the version before it starts one.
 */

/** What `llama-server --version` says, as far as it can be read. */
export interface LlamaCppVersion {
  readonly build?: number
  /** The release version (`0.9.0`), on builds that report one. */
  readonly version?: string
}

/**
 * Read `llama-server --version`. Releases print
 * `version: 0.9.0 (build 11683, commit a1b2c3d)`; older builds print
 * `version: 11236 (a1b2c3d)`. A build number of 0 (built without git
 * history) is treated as unknown.
 */
export function parseLlamaCppVersion(text: string): LlamaCppVersion {
  const release = /version:\s*v?(\d+\.\d+\.\d+)\s*\(build\s+(\d+)/i.exec(text)
  if (release) {
    const build = Number(release[2])
    return { version: release[1]!, ...(build > 0 ? { build } : {}) }
  }
  const legacy = /version:\s*(\d+)\s*\(/i.exec(text)
  if (legacy) {
    const build = Number(legacy[1])
    return build > 0 ? { build } : {}
  }
  const bare = /version:\s*v?(\d+\.\d+\.\d+)/i.exec(text)
  return bare ? { version: bare[1]! } : {}
}

export type VersionCheck =
  | { readonly ok: true; readonly found: string }
  | { readonly ok: false; readonly reason: 'too-old'; readonly found: string }
  | { readonly ok: false; readonly reason: 'unknown' }

/** Compare dotted versions numerically: negative when `a` is older. */
export function compareVersions(a: string, b: string): number {
  const left = a.split('.').map(part => Number.parseInt(part, 10) || 0)
  const right = b.split('.').map(part => Number.parseInt(part, 10) || 0)
  for (let index = 0; index < Math.max(left.length, right.length); index += 1) {
    const difference = (left[index] ?? 0) - (right[index] ?? 0)
    if (difference !== 0) return difference
  }
  return 0
}

/** Check a llama.cpp version against a minimum build, using the release version when no build is reported. */
export function checkLlamaCpp(found: LlamaCppVersion, minimum: { readonly build: number; readonly version: string }): VersionCheck {
  if (found.build !== undefined) {
    const label = `build ${found.build}${found.version === undefined ? '' : ` (${found.version})`}`
    return found.build >= minimum.build ? { ok: true, found: label } : { ok: false, reason: 'too-old', found: label }
  }
  if (found.version !== undefined) {
    return compareVersions(found.version, minimum.version) >= 0
      ? { ok: true, found: found.version }
      : { ok: false, reason: 'too-old', found: found.version }
  }
  return { ok: false, reason: 'unknown' }
}

/** Read `mlx_lm --version`, which prints the bare version (`0.32.2`). */
export function parseMlxLmVersion(text: string): string | undefined {
  return /(?:^|\s)v?(\d+\.\d+\.\d+)/.exec(text.trim())?.[1]
}

export function checkMlxLm(found: string | undefined, minimum: string): VersionCheck {
  if (found === undefined) return { ok: false, reason: 'unknown' }
  return compareVersions(found, minimum) >= 0 ? { ok: true, found } : { ok: false, reason: 'too-old', found }
}
