// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { homedir } from 'node:os'
import { isAbsolute, resolve, sep } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'

export { xerxesHome }

export const XERXES_HOME_ENV = 'XERXES_HOME'

/** Resolve a path below the configured Xerxes home without creating it. */
export function xerxesSubdir(...parts: string[]): string {
  return xerxesSubdirFor(process.env, ...parts)
}

/** Testable/environment-explicit variant of {@link xerxesSubdir}. */
export function xerxesSubdirFor(environment: Record<string, string | undefined>, ...parts: string[]): string {
  return resolveSubdir(xerxesHome(environment), parts)
}

/** Resolve the shared agents home used by agent skills and specifications. */
export function agentsHome(home = homedir()): string {
  return resolve(home, '.agents')
}

/** Resolve a path below the shared agents home without creating it. */
export function agentsSubdir(...parts: string[]): string {
  return agentsSubdirFor(homedir(), ...parts)
}

/** Testable/home-explicit variant of {@link agentsSubdir}. */
export function agentsSubdirFor(home: string, ...parts: string[]): string {
  return resolveSubdir(agentsHome(home), parts)
}

/**
 * Join free-form segments under a base directory with traversal protection.
 *
 * Absolute segments and `..` traversals are rejected up front, and the fully
 * resolved result must remain inside the resolved base directory.
 */
function resolveSubdir(base: string, parts: readonly string[]): string {
  const resolvedBase = resolve(base)
  for (const part of parts) {
    if (typeof part !== 'string' || part.length === 0) {
      throw new TypeError(`path segments must be non-empty strings, received: ${String(part)}`)
    }
    if (isAbsolute(part) || part.split(/[\\/]/).includes('..')) {
      throw new Error(`unsafe path segment '${part}': segments must stay below ${resolvedBase}`)
    }
  }
  const resolved = resolve(resolvedBase, ...parts)
  if (resolved !== resolvedBase && !resolved.startsWith(resolvedBase + sep)) {
    throw new Error(`resolved path '${resolved}' escapes the base directory ${resolvedBase}`)
  }
  return resolved
}
