// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isAbsolute, relative, resolve, sep } from 'node:path'

import { ValidationError } from '../core/errors.js'

/** An explicitly supplied environment map used to select sandbox credentials. */
export type CredentialEnvironment = Readonly<Record<string, string | undefined>>

/** Configuration for an isolated credential-file allow-list. */
export interface CredentialFileRegistryOptions {
  /** Absolute host directory used to resolve relative credential paths. */
  readonly baseDirectory: string
  /** Optional explicit replacement for `~`; no process-global home lookup occurs. */
  readonly homeDirectory?: string
  /** Absolute host directories in which credential files may reside. */
  readonly allowedRoots: readonly string[]
}

/** Raised when a credential path is malformed or escapes every configured root. */
export class CredentialPathError extends ValidationError {
  constructor(candidate: string, allowedRoots: readonly string[], reason: string) {
    super('credentialPath', reason, candidate, { allowedRoots })
  }
}

/**
 * Keeps credential-file authorization scoped to one caller-owned registry.
 *
 * The registry deliberately has no singleton and never reads `process.env` or a
 * process-global home directory. Callers must provide every root and, when using
 * `~`, the home directory to which it expands.
 */
export class CredentialFileRegistry {
  readonly allowedRoots: readonly string[]
  readonly baseDirectory: string
  readonly homeDirectory: string | undefined

  readonly #paths = new Set<string>()

  constructor(options: CredentialFileRegistryOptions) {
    const normalized = normalizeOptions(options)
    this.allowedRoots = normalized.allowedRoots
    this.baseDirectory = normalized.baseDirectory
    this.homeDirectory = normalized.homeDirectory
  }

  /** Add a credential file after resolving it inside one configured root. */
  register(path: string): string {
    const resolvedPath = this.#resolve(path)
    this.#paths.add(resolvedPath)
    return resolvedPath
  }

  /** Remove a registered credential file. Returns false for an invalid or unknown path. */
  unregister(path: string): boolean {
    const resolvedPath = this.#safeResolve(path)
    return resolvedPath === undefined ? false : this.#paths.delete(resolvedPath)
  }

  /** Return a stable snapshot of registered credential paths. */
  allowedPaths(): string[] {
    return [...this.#paths].sort()
  }

  /** Check whether a path is both contained and registered. */
  isAllowed(path: string): boolean {
    const resolvedPath = this.#safeResolve(path)
    return resolvedPath !== undefined && this.#paths.has(resolvedPath)
  }

  /** Remove every registered path from this caller-owned registry. */
  clear(): void {
    this.#paths.clear()
  }

  #resolve(path: string): string {
    return resolveCredentialPath(path, this.baseDirectory, this.homeDirectory, this.allowedRoots)
  }

  #safeResolve(path: string): string | undefined {
    try {
      return this.#resolve(path)
    } catch (error) {
      if (error instanceof CredentialPathError) {
        return undefined
      }
      throw error
    }
  }
}

/**
 * Select requested credential variables from an explicit environment map.
 *
 * Missing names are omitted. Values from unrequested names, inherited fields,
 * and the host process environment are never consulted.
 */
export function selectCredentialEnvironment(
  names: readonly string[],
  environment: CredentialEnvironment,
): Readonly<Record<string, string>> {
  if (!Array.isArray(names)) {
    throw new ValidationError('credentialNames', 'must be an array of environment variable names', names)
  }
  if (environment === null || typeof environment !== 'object' || Array.isArray(environment)) {
    throw new ValidationError('credentialEnvironment', 'must be an explicit environment map', environment)
  }

  const selected: Record<string, string> = {}
  for (const name of names) {
    validateEnvironmentName(name)
    if (!Object.prototype.hasOwnProperty.call(environment, name)) {
      continue
    }

    const value = environment[name]
    if (value === undefined) {
      continue
    }
    if (typeof value !== 'string') {
      throw new ValidationError('credentialEnvironment', `value for ${name} must be a string`, value)
    }
    Object.defineProperty(selected, name, {
      configurable: true,
      enumerable: true,
      value,
      writable: true,
    })
  }
  return Object.freeze(selected)
}

interface NormalizedCredentialFileRegistryOptions {
  readonly allowedRoots: readonly string[]
  readonly baseDirectory: string
  readonly homeDirectory: string | undefined
}

function normalizeOptions(options: CredentialFileRegistryOptions): NormalizedCredentialFileRegistryOptions {
  if (options === null || typeof options !== 'object') {
    throw new ValidationError('credentialFileRegistry', 'requires explicit registry options', options)
  }

  const homeDirectory = options.homeDirectory === undefined
    ? undefined
    : normalizeAbsoluteDirectory(options.homeDirectory, 'homeDirectory', undefined)
  const baseDirectory = normalizeAbsoluteDirectory(options.baseDirectory, 'baseDirectory', homeDirectory)
  if (!Array.isArray(options.allowedRoots) || options.allowedRoots.length === 0) {
    throw new ValidationError(
      'allowedRoots',
      'must contain at least one absolute credential directory',
      options.allowedRoots,
    )
  }

  const allowedRoots = Object.freeze(options.allowedRoots.map((root, index) => (
    normalizeAbsoluteDirectory(root, `allowedRoots[${index}]`, homeDirectory)
  )))
  return { allowedRoots, baseDirectory, homeDirectory }
}

function resolveCredentialPath(
  candidate: string,
  baseDirectory: string,
  homeDirectory: string | undefined,
  allowedRoots: readonly string[],
): string {
  const expandedCandidate = expandHome(candidate, homeDirectory, 'credentialPath', allowedRoots)
  const resolvedPath = isAbsolute(expandedCandidate)
    ? resolve(expandedCandidate)
    : resolve(baseDirectory, expandedCandidate)
  if (!allowedRoots.some(root => isContained(root, resolvedPath))) {
    throw new CredentialPathError(
      candidate,
      allowedRoots,
      `path ${JSON.stringify(candidate)} must remain inside a configured credential root`,
    )
  }
  return resolvedPath
}

function normalizeAbsoluteDirectory(value: string, field: string, homeDirectory: string | undefined): string {
  const expandedValue = expandHome(value, homeDirectory, field, [])
  if (!isAbsolute(expandedValue)) {
    throw new CredentialPathError(value, [], `${field} must be an absolute path`)
  }
  return resolve(expandedValue)
}

function expandHome(
  value: string,
  homeDirectory: string | undefined,
  field: string,
  allowedRoots: readonly string[],
): string {
  if (typeof value !== 'string' || value.trim() === '' || value.includes('\0')) {
    throw new CredentialPathError(String(value), allowedRoots, `${field} must be a non-empty path without NUL bytes`)
  }
  if (value === '~' || value.startsWith('~/') || value.startsWith(`~${sep}`)) {
    if (homeDirectory === undefined) {
      throw new CredentialPathError(value, allowedRoots, `${field} uses ~ without an explicit homeDirectory`)
    }
    return value === '~' ? homeDirectory : resolve(homeDirectory, value.slice(2))
  }
  if (value.startsWith('~')) {
    throw new CredentialPathError(value, allowedRoots, `${field} supports only ~ or ~/ path expansion`)
  }
  return value
}

function isContained(root: string, candidate: string): boolean {
  const relativePath = relative(root, candidate)
  return relativePath === '' || (
    relativePath !== '..'
    && !relativePath.startsWith(`..${sep}`)
    && !isAbsolute(relativePath)
  )
}

function validateEnvironmentName(name: string): void {
  if (typeof name !== 'string' || name.trim() === '' || name.includes('\0')) {
    throw new ValidationError('credentialName', 'must be a non-empty name without NUL bytes', name)
  }
}
