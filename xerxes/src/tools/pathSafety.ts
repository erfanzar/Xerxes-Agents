// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lstat, realpath } from 'node:fs/promises'
import { basename, dirname, isAbsolute, relative, resolve, sep } from 'node:path'

import { ValidationError } from '../core/errors.js'

/** Raised when a tool-supplied path is not contained by its workspace root. */
export class WorkspacePathError extends ValidationError {
  constructor(candidate: string, message: string) {
    super('path', message, candidate)
  }
}

/**
 * Resolve paths for filesystem tools without allowing traversal or symlink escapes.
 *
 * The resolver follows existing symlinks before checking containment. For a path that
 * does not exist yet, it resolves the nearest existing ancestor so writes cannot be
 * redirected through an existing symlink outside the workspace.
 */
export class WorkspacePathResolver {
  readonly root: string

  constructor(root: string = process.cwd()) {
    if (!root.trim()) {
      throw new ValidationError('workspace_root', 'must not be empty', root)
    }
    this.root = resolve(root)
  }

  async resolve(candidate: string): Promise<string> {
    const normalized = validateCandidate(candidate)
    const workspaceRoot = await this.canonicalRoot()
    const lexicalTarget = isAbsolute(normalized) ? resolve(normalized) : resolve(workspaceRoot, normalized)
    if (!isWithin(workspaceRoot, lexicalTarget)) {
      throw new WorkspacePathError(candidate, `escapes workspace root ${workspaceRoot}`)
    }

    const physicalTarget = await resolveExistingAncestor(lexicalTarget)
    if (!isWithin(workspaceRoot, physicalTarget)) {
      throw new WorkspacePathError(candidate, `resolves outside workspace root ${workspaceRoot}`)
    }
    return physicalTarget
  }

  async relative(candidate: string): Promise<string> {
    const workspaceRoot = await this.canonicalRoot()
    const target = await this.resolve(candidate)
    return relative(workspaceRoot, target) || '.'
  }

  /**
   * Best-effort re-validation immediately before a mutation. resolve() checks
   * containment at resolution time; a symlink swapped in afterwards can still
   * redirect the mutation outside the workspace. Callers re-check the already
   * resolved target right before write/delete/move and operate on the returned
   * physical path. Residual race: a path component can still be swapped between
   * this check and the kernel operation — closing it fully needs dirfd-relative
   * syscalls, which the runtime does not expose.
   */
  async recheck(resolved: string): Promise<string> {
    const workspaceRoot = await this.canonicalRoot()
    const physicalTarget = await resolveExistingAncestor(resolved)
    if (!isWithin(workspaceRoot, physicalTarget)) {
      throw new WorkspacePathError(resolved, `resolves outside workspace root ${workspaceRoot}`)
    }
    return physicalTarget
  }

  private async canonicalRoot(): Promise<string> {
    try {
      return await realpath(this.root)
    } catch (error) {
      throw new WorkspacePathError(this.root, `workspace root is unavailable: ${errorMessage(error)}`)
    }
  }
}

function validateCandidate(candidate: string): string {
  if (!candidate.trim()) {
    throw new WorkspacePathError(candidate, 'must not be empty')
  }
  if (candidate.includes('\0')) {
    throw new WorkspacePathError(candidate, 'must not contain a null byte')
  }
  return candidate
}

function isWithin(root: string, candidate: string): boolean {
  const pathFromRoot = relative(root, candidate)
  return pathFromRoot === ''
    || (!pathFromRoot.startsWith(`..${sep}`) && pathFromRoot !== '..' && !isAbsolute(pathFromRoot))
}

async function resolveExistingAncestor(candidate: string): Promise<string> {
  let current = candidate
  const missingSegments: string[] = []

  while (true) {
    try {
      await lstat(current)
    } catch (error) {
      if (!isNotFound(error)) {
        throw error
      }
      const parent = dirname(current)
      if (parent === current) {
        throw new WorkspacePathError(candidate, 'does not have an existing ancestor')
      }
      missingSegments.unshift(basename(current))
      current = parent
      continue
    }

    try {
      return resolve(await realpath(current), ...missingSegments)
    } catch (error) {
      throw new WorkspacePathError(candidate, `contains an unresolved symlink: ${errorMessage(error)}`)
    }
  }
}

function isNotFound(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
