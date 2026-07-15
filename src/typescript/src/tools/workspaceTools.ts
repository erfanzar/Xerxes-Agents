// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile, lstat, mkdir, readdir, stat } from 'node:fs/promises'
import { dirname, join, relative, sep } from 'node:path'

import { MarkdownAgentWorkspace } from '../channels/workspace.js'
import { ValidationError } from '../core/errors.js'
import { createUnifiedDiff } from './codingTools.js'
import { WorkspacePathResolver } from './pathSafety.js'

export interface WorkspaceFileMetadata {
  readonly bytes: number
  readonly modified: number
  readonly path: string
}

export interface WorkspaceReadOptions {
  /** Maximum UTF-8 bytes to include before adding an explicit truncation marker. */
  readonly maxBytes?: number
}

export interface WorkspaceWriteOptions {
  /** Create missing parent directories. Defaults to true. */
  readonly createDirs?: boolean
}

export interface WorkspaceAppendOptions extends WorkspaceWriteOptions {
  /** Insert one newline before appended content when the existing file has none. Defaults to true. */
  readonly ensureNewline?: boolean
}

export interface WorkspaceWriteResult {
  readonly bytes: number
  readonly created: boolean
  readonly path: string
}

export interface WorkspaceAppendResult {
  readonly appendedBytes: number
  readonly created: boolean
  readonly path: string
}

/** List regular files in a channel workspace without following child symlinks. */
export async function workspaceList(
  workspace: MarkdownAgentWorkspace = new MarkdownAgentWorkspace(),
): Promise<readonly WorkspaceFileMetadata[]> {
  const { root } = await resolveWorkspace(workspace)
  const files: WorkspaceFileMetadata[] = []
  await collectFiles(root, root, files)
  return files.sort((left, right) => left.path.localeCompare(right.path))
}

/** Read one workspace file while preserving workspace containment. */
export async function workspaceRead(
  path: string,
  workspace: MarkdownAgentWorkspace = new MarkdownAgentWorkspace(),
  options: WorkspaceReadOptions = {},
): Promise<string> {
  const maxBytes = options.maxBytes
  if (maxBytes !== undefined && (!Number.isInteger(maxBytes) || maxBytes < 0)) {
    throw new ValidationError('maxBytes', 'must be a non-negative integer', maxBytes)
  }
  const { paths } = await resolveWorkspace(workspace)
  const target = await paths.resolve(path)
  await requireRegularFile(target, path)
  const content = await Bun.file(target).text()
  if (maxBytes === undefined || utf8Length(content) <= maxBytes) return content

  const visible = truncateUtf8(content, maxBytes)
  return `${visible}\n[... truncated; ${utf8Length(content) - utf8Length(visible)} bytes elided]`
}

/** Replace one workspace file and return its canonical relative path and byte count. */
export async function workspaceWrite(
  path: string,
  content: string,
  workspace: MarkdownAgentWorkspace = new MarkdownAgentWorkspace(),
  options: WorkspaceWriteOptions = {},
): Promise<WorkspaceWriteResult> {
  const { paths } = await resolveWorkspace(workspace)
  const target = await paths.resolve(path)
  const existing = await existingMetadata(target)
  if (existing !== undefined && !existing.isFile()) {
    throw new ValidationError('path', 'must refer to a regular file', path)
  }
  await ensureParent(target, options.createDirs ?? true)
  await Bun.write(target, content)
  return {
    path: await paths.relative(target),
    bytes: utf8Length(content),
    created: existing === undefined,
  }
}

/** Append content to a workspace file, optionally inserting exactly one separator newline. */
export async function workspaceAppend(
  path: string,
  content: string,
  workspace: MarkdownAgentWorkspace = new MarkdownAgentWorkspace(),
  options: WorkspaceAppendOptions = {},
): Promise<WorkspaceAppendResult> {
  const { paths } = await resolveWorkspace(workspace)
  const target = await paths.resolve(path)
  const existing = await existingMetadata(target)
  if (existing !== undefined && !existing.isFile()) {
    throw new ValidationError('path', 'must refer to a regular file', path)
  }
  await ensureParent(target, options.createDirs ?? true)

  let prefix = ''
  if (existing !== undefined && options.ensureNewline !== false) {
    const previous = await Bun.file(target).text()
    if (previous && !previous.endsWith('\n')) prefix = '\n'
  }
  await appendFile(target, prefix + content, 'utf8')
  return {
    path: await paths.relative(target),
    appendedBytes: utf8Length(prefix + content),
    created: existing === undefined,
  }
}

/** Return a unified preview between a workspace file and proposed text without changing disk state. */
export async function workspaceDiff(
  path: string,
  newContent: string,
  workspace: MarkdownAgentWorkspace = new MarkdownAgentWorkspace(),
): Promise<string> {
  const { paths } = await resolveWorkspace(workspace)
  const target = await paths.resolve(path)
  const existing = await existingMetadata(target)
  if (existing !== undefined && !existing.isFile()) {
    throw new ValidationError('path', 'must refer to a regular file', path)
  }
  const previous = existing === undefined ? '' : await Bun.file(target).text()
  return createUnifiedDiff(previous, newContent, path, path)
}

async function resolveWorkspace(workspace: MarkdownAgentWorkspace): Promise<{ readonly paths: WorkspacePathResolver; readonly root: string }> {
  await workspace.ensure()
  const paths = new WorkspacePathResolver(workspace.path)
  return { paths, root: await paths.resolve('.') }
}

async function collectFiles(root: string, directory: string, files: WorkspaceFileMetadata[]): Promise<void> {
  const entries = await readdir(directory, { withFileTypes: true })
  for (const entry of entries) {
    const target = join(directory, entry.name)
    const metadata = await lstat(target)
    if (metadata.isSymbolicLink()) continue
    if (metadata.isDirectory()) {
      await collectFiles(root, target, files)
      continue
    }
    if (!metadata.isFile()) continue
    files.push({
      path: relative(root, target).split(sep).join('/'),
      bytes: metadata.size,
      modified: metadata.mtimeMs / 1_000,
    })
  }
}

async function ensureParent(target: string, createDirs: boolean): Promise<void> {
  const parent = dirname(target)
  if (createDirs) {
    await mkdir(parent, { recursive: true })
    return
  }
  let metadata
  try {
    metadata = await stat(parent)
  } catch (error) {
    if (isMissing(error)) {
      throw new ValidationError('path', 'parent directory does not exist and createDirs is false', target)
    }
    throw error
  }
  if (!metadata.isDirectory()) {
    throw new ValidationError('path', 'parent path must be a directory', parent)
  }
}

async function requireRegularFile(target: string, sourcePath: string): Promise<void> {
  const metadata = await existingMetadata(target)
  if (metadata?.isFile()) return
  if (metadata === undefined) throw new WorkspaceFileNotFoundError(sourcePath)
  throw new ValidationError('path', 'must refer to a regular file', sourcePath)
}

async function existingMetadata(target: string): Promise<Awaited<ReturnType<typeof lstat>> | undefined> {
  try {
    return await lstat(target)
  } catch (error) {
    if (isMissing(error)) return undefined
    throw error
  }
}

function utf8Length(text: string): number {
  return new TextEncoder().encode(text).length
}

function truncateUtf8(text: string, maxBytes: number): string {
  const encoded = new TextEncoder().encode(text)
  let end = Math.min(maxBytes, encoded.length)
  while (end > 0 && ((encoded[end] ?? 0) & 0b1100_0000) === 0b1000_0000) end -= 1
  return new TextDecoder().decode(encoded.subarray(0, end))
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

/** Raised when a requested contained workspace file does not exist. */
export class WorkspaceFileNotFoundError extends Error {
  constructor(path: string) {
    super(`workspace file not found: ${path}`)
    this.name = 'FileNotFoundError'
  }
}
