// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lstat, mkdir, readFile, realpath, rename, rm, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

/** Raised when a channel workspace path is missing, malformed, or unsafe to use. */
export class WorkspaceFilesystemError extends Error {
  readonly path: string

  constructor(path: string, reason: string) {
    super(`workspace path ${JSON.stringify(path)} is unsafe: ${reason}`)
    this.name = 'WorkspaceFilesystemError'
    this.path = path
  }
}

export interface WorkspaceFile {
  readonly path: string
  readonly size: number
}

export type WorkspaceChild =
  | { readonly kind: 'directory'; readonly path: string }
  | { readonly kind: 'file'; readonly file: WorkspaceFile }
  | { readonly kind: 'missing' }
  | { readonly kind: 'other'; readonly path: string }

/** Resolve a user-supplied workspace root once, including a conventional home shortcut. */
export function normalizeWorkspacePath(path: string): string {
  if (typeof path !== 'string') {
    throw new TypeError('workspace path must be a string')
  }
  const trimmed = path.trim()
  if (!trimmed) {
    throw new WorkspaceFilesystemError(path, 'must not be empty')
  }
  if (trimmed.includes('\0')) {
    throw new WorkspaceFilesystemError(path, 'must not contain a null byte')
  }
  return resolve(expandHome(trimmed))
}

/** Create a workspace root when necessary, then return its canonical directory path. */
export async function ensureWorkspaceDirectory(path: string, label: string): Promise<string> {
  const normalized = normalizeWorkspacePath(path)
  try {
    await lstat(normalized)
  } catch (error) {
    if (!isMissing(error)) {
      throw filesystemError(normalized, `cannot inspect ${label}`, error)
    }
    try {
      await mkdir(normalized, { recursive: true })
    } catch (mkdirError) {
      throw filesystemError(normalized, `cannot create ${label}`, mkdirError)
    }
  }
  return requireWorkspaceDirectory(normalized, label, { allowRootSymlink: true })
}

/** Return an existing workspace root's canonical path, or undefined when it does not exist. */
export async function existingWorkspaceDirectory(path: string, label: string): Promise<string | undefined> {
  const normalized = normalizeWorkspacePath(path)
  try {
    await lstat(normalized)
  } catch (error) {
    if (isMissing(error)) return undefined
    throw filesystemError(normalized, `cannot inspect ${label}`, error)
  }
  return requireWorkspaceDirectory(normalized, label, { allowRootSymlink: true })
}

/** Require a workspace root to be a directory. A root symlink is allowed because it is explicit configuration. */
export async function requireWorkspaceDirectory(
  path: string,
  label: string,
  options: { readonly allowRootSymlink?: boolean } = {},
): Promise<string> {
  const normalized = normalizeWorkspacePath(path)
  let metadata
  try {
    metadata = await lstat(normalized)
  } catch (error) {
    if (isMissing(error)) {
      throw new WorkspaceFilesystemError(normalized, `${label} does not exist`)
    }
    throw filesystemError(normalized, `cannot inspect ${label}`, error)
  }
  if (metadata.isSymbolicLink() && !options.allowRootSymlink) {
    throw new WorkspaceFilesystemError(normalized, `${label} must not be a symbolic link`)
  }

  let canonical: string
  try {
    canonical = await realpath(normalized)
  } catch (error) {
    throw filesystemError(normalized, `cannot resolve ${label}`, error)
  }

  try {
    metadata = await lstat(canonical)
  } catch (error) {
    throw filesystemError(canonical, `cannot inspect resolved ${label}`, error)
  }
  if (!metadata.isDirectory()) {
    throw new WorkspaceFilesystemError(normalized, `${label} must be a directory`)
  }
  return canonical
}

/** Ensure a direct child directory exists below a canonical workspace root without accepting symlink children. */
export async function ensureWorkspaceChildDirectory(root: string, name: string, label: string): Promise<string> {
  const child = await inspectWorkspaceChild(root, name, label)
  if (child.kind === 'directory') return child.path
  if (child.kind !== 'missing') {
    throw new WorkspaceFilesystemError(workspaceChildPath(root, name, label), `${label} must be a directory`)
  }

  const target = workspaceChildPath(root, name, label)
  try {
    await mkdir(target)
  } catch (error) {
    if (!isAlreadyExists(error)) {
      throw filesystemError(target, `cannot create ${label}`, error)
    }
  }

  const created = await inspectWorkspaceChild(root, name, label)
  if (created.kind !== 'directory') {
    throw new WorkspaceFilesystemError(target, `${label} was not created as a directory`)
  }
  return created.path
}

/** Inspect a direct child below a canonical workspace root, rejecting symlink traversal. */
export async function inspectWorkspaceChild(root: string, name: string, label: string): Promise<WorkspaceChild> {
  const candidate = workspaceChildPath(root, name, label)
  let metadata
  try {
    metadata = await lstat(candidate)
  } catch (error) {
    if (isMissing(error)) return { kind: 'missing' }
    throw filesystemError(candidate, `cannot inspect ${label}`, error)
  }
  if (metadata.isSymbolicLink()) {
    throw new WorkspaceFilesystemError(candidate, `${label} must not be a symbolic link`)
  }

  let canonical: string
  try {
    canonical = await realpath(candidate)
  } catch (error) {
    throw filesystemError(candidate, `cannot resolve ${label}`, error)
  }
  assertWithinWorkspace(root, canonical, label)

  if (metadata.isFile()) {
    return { kind: 'file', file: { path: canonical, size: metadata.size } }
  }
  if (metadata.isDirectory()) {
    return { kind: 'directory', path: canonical }
  }
  return { kind: 'other', path: canonical }
}

/** Read a previously inspected regular workspace file, preserving a useful error on failure. */
export async function readWorkspaceFile(file: WorkspaceFile, label: string): Promise<string> {
  try {
    return await readFile(file.path, 'utf8')
  } catch (error) {
    throw filesystemError(file.path, `cannot read ${label}`, error)
  }
}

/** Seed a direct workspace file if it is absent. Existing user files are preserved. */
export async function writeWorkspaceFileIfMissing(
  root: string,
  name: string,
  content: string,
  label: string,
): Promise<boolean> {
  const existing = await inspectWorkspaceChild(root, name, label)
  if (existing.kind === 'directory' || existing.kind === 'other') {
    throw new WorkspaceFilesystemError(workspaceChildPath(root, name, label), `${label} must be a regular file`)
  }
  if (existing.kind === 'file') return false
  const target = workspaceChildPath(root, name, label)
  try {
    await writeFile(target, content, { encoding: 'utf8', flag: 'wx' })
    return true
  } catch (error) {
    if (isAlreadyExists(error)) return false
    throw filesystemError(target, `cannot seed ${label}`, error)
  }
}

/** Atomically replace a direct workspace file after rejecting unsafe existing targets. */
export async function writeWorkspaceFile(root: string, name: string, content: string, label: string): Promise<string> {
  const existing = await inspectWorkspaceChild(root, name, label)
  if (existing.kind === 'directory' || existing.kind === 'other') {
    throw new WorkspaceFilesystemError(workspaceChildPath(root, name, label), `${label} must be a regular file`)
  }
  const target = workspaceChildPath(root, name, label)
  await writeAtomically(target, content, label)
  return target
}

/** Move a direct workspace file without following symlinks or replacing an existing destination. */
export async function moveWorkspaceFile(
  root: string,
  sourceName: string,
  targetName: string,
  label: string,
): Promise<string> {
  const source = await inspectWorkspaceChild(root, sourceName, label)
  if (source.kind !== 'file') {
    throw new WorkspaceFilesystemError(workspaceChildPath(root, sourceName, label), `${label} must be a regular file`)
  }
  const destination = await inspectWorkspaceChild(root, targetName, `${label} archive`)
  if (destination.kind !== 'missing') {
    throw new WorkspaceFilesystemError(workspaceChildPath(root, targetName, label), `${label} archive already exists`)
  }
  const target = workspaceChildPath(root, targetName, `${label} archive`)
  try {
    await rename(source.file.path, target)
  } catch (error) {
    throw filesystemError(source.file.path, `cannot archive ${label}`, error)
  }
  return target
}

/** Check a regular workspace file's byte count before it is read into channel context. */
export function requireWorkspaceFileSize(file: WorkspaceFile, maximum: number, label: string): void {
  if (file.size > maximum) {
    throw new WorkspaceFilesystemError(file.path, `${label} exceeds the ${maximum}-byte limit`)
  }
}

function workspaceChildPath(root: string, name: string, label: string): string {
  if (!name || name.includes('\0') || basename(name) !== name || name === '.' || name === '..') {
    throw new WorkspaceFilesystemError(name, `${label} must use a single file or directory name`)
  }
  const candidate = resolve(root, name)
  assertWithinWorkspace(root, candidate, label)
  return candidate
}

function assertWithinWorkspace(root: string, candidate: string, label: string): void {
  const pathFromRoot = relative(root, candidate)
  const isContained = pathFromRoot === ''
    || (!pathFromRoot.startsWith(`..${sep}`) && pathFromRoot !== '..' && !isAbsolute(pathFromRoot))
  if (isContained) {
    return
  }
  throw new WorkspaceFilesystemError(candidate, `${label} resolves outside workspace root ${root}`)
}

async function writeAtomically(target: string, content: string, label: string): Promise<void> {
  const temporary = join(dirname(target), `.${basename(target)}.${crypto.randomUUID()}.tmp`)
  try {
    await Bun.write(temporary, content)
    await rename(temporary, target)
  } catch (error) {
    throw filesystemError(target, `cannot write ${label}`, error)
  } finally {
    try {
      await rm(temporary, { force: true })
    } catch {
      // The primary write error above is more useful, and a failed best-effort cleanup cannot expose workspace content.
    }
  }
}

function expandHome(path: string): string {
  if (path === '~') return homedir()
  if (path.startsWith('~/') || path.startsWith('~\\')) return join(homedir(), path.slice(2))
  return path
}

function filesystemError(path: string, action: string, error: unknown): WorkspaceFilesystemError {
  const detail = error instanceof Error ? error.message : String(error)
  return new WorkspaceFilesystemError(path, `${action}: ${detail}`)
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function isAlreadyExists(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'EEXIST'
}
