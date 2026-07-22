// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { realpath } from 'node:fs/promises'
import { isAbsolute, parse, relative, resolve, sep } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { WorkspacePathError, WorkspacePathResolver } from '../tools/pathSafety.js'

/** Raised when a candidate path would escape its workspace. */
export class PathEscape extends ValidationError {
  constructor(workspace: string, candidate: string, reason?: string) {
    super(
      'path',
      reason ?? `path ${JSON.stringify(candidate)} escapes workspace root ${workspace}`,
      candidate,
      { workspace },
    )
  }
}

/**
 * Resolve a candidate within a workspace, including through existing symlinks.
 *
 * An absolute candidate that already resolves inside the workspace is kept as-is.
 * An absolute candidate pointing outside the workspace is re-rooted under the
 * workspace to match Xerxes' historical file-tool contract, preserving the
 * no-outside-access property without mangling valid in-workspace paths.
 */
export async function resolveWithin(workspace: string, candidate: string): Promise<string> {
  const resolver = new WorkspacePathResolver(workspace)
  try {
    return await resolver.resolve(await rerootAbsoluteCandidate(resolver.root, candidate))
  } catch (error) {
    if (error instanceof WorkspacePathError) {
      throw new PathEscape(workspace, candidate, error.message)
    }
    throw error
  }
}

/** Soft-denial companion to {@link resolveWithin}. */
export async function safePath(workspace: string, candidate: string): Promise<string | undefined> {
  try {
    return await resolveWithin(workspace, candidate)
  } catch (error) {
    if (error instanceof PathEscape) {
      return undefined
    }
    throw error
  }
}

async function rerootAbsoluteCandidate(workspaceRoot: string, candidate: string): Promise<string> {
  if (!isAbsolute(candidate)) {
    return candidate
  }
  const resolvedCandidate = resolve(candidate)
  let canonicalRoot: string | undefined
  try {
    canonicalRoot = await realpath(workspaceRoot)
  } catch {
    // An unreadable workspace root is surfaced by the resolver below.
    canonicalRoot = undefined
  }
  // The workspace root itself may sit behind a symlink (for example /var on
  // macOS), so accept candidates expressed against either root spelling.
  const roots = canonicalRoot === undefined || canonicalRoot === workspaceRoot
    ? [workspaceRoot]
    : [canonicalRoot, workspaceRoot]
  for (const root of roots) {
    if (isWithinRoot(root, resolvedCandidate)) {
      // Re-express against the canonical root so the resolver's lexical
      // containment check accepts the path even behind a symlinked root.
      return resolve(canonicalRoot ?? root, relative(root, resolvedCandidate))
    }
  }
  return relative(parse(candidate).root, candidate) || '.'
}

function isWithinRoot(root: string, candidate: string): boolean {
  const pathFromRoot = relative(root, candidate)
  return pathFromRoot === ''
    || (!pathFromRoot.startsWith(`..${sep}`) && pathFromRoot !== '..' && !isAbsolute(pathFromRoot))
}
