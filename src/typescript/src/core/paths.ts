// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { homedir } from 'node:os'
import { join, resolve } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'

export { xerxesHome }

export const XERXES_HOME_ENV = 'XERXES_HOME'

/** Resolve a path below the configured Xerxes home without creating it. */
export function xerxesSubdir(...parts: string[]): string {
  return xerxesSubdirFor(process.env, ...parts)
}

/** Testable/environment-explicit variant of {@link xerxesSubdir}. */
export function xerxesSubdirFor(environment: Record<string, string | undefined>, ...parts: string[]): string {
  return join(xerxesHome(environment), ...parts)
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
  return join(agentsHome(home), ...parts)
}
