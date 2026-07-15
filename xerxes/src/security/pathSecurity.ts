// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isAbsolute, parse, relative } from 'node:path'

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
 * Absolute candidates are deliberately re-rooted under the workspace to match
 * Xerxes' historical file-tool contract rather than being interpreted as host paths.
 */
export async function resolveWithin(workspace: string, candidate: string): Promise<string> {
  try {
    return await new WorkspacePathResolver(workspace).resolve(rerootAbsoluteCandidate(candidate))
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

function rerootAbsoluteCandidate(candidate: string): string {
  if (!isAbsolute(candidate)) {
    return candidate
  }
  return relative(parse(candidate).root, candidate) || '.'
}
