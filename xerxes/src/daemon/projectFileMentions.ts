// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readdir } from 'node:fs/promises'
import { relative, resolve } from 'node:path'

export const PROJECT_FILE_INDEX_LIMIT = 5_000
export const PROJECT_FILE_MENTION_LIMIT = 50
export const PROJECT_FILE_INDEX_CACHE_TTL_MS = 30_000

const BINARY_EXTENSIONS = new Set([
  '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.bmp', '.tiff',
  '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.flac', '.ogg',
  '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar', '.xz',
  '.woff', '.woff2', '.ttf', '.eot', '.otf',
  '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
  '.exe', '.dll', '.so', '.dylib', '.o', '.a', '.pyc', '.class', '.wasm',
])

export type ProjectFileIndexSource = 'filesystem' | 'git'

export interface ProjectFileMention {
  readonly absolutePath: string
  readonly basename: string
  readonly relativePath: string
}

export interface ProjectFileIndex {
  readonly files: readonly ProjectFileMention[]
  readonly rootDirectory: string
  readonly source: ProjectFileIndexSource
}

export interface ProjectFileMentionSearchResult {
  readonly matches: readonly ProjectFileMention[]
  /** Root against which every match's `relativePath` is resolved. */
  readonly rootDirectory: string
  readonly source: ProjectFileIndexSource
}

export interface ProjectFileMentionIndexOptions {
  /** Cache lifetime for a project index. Defaults to 30 seconds. */
  readonly cacheTtlMs?: number
  /** Testable lower ceiling; values above 5,000 are clamped. */
  readonly maxFiles?: number
}

interface CachedProjectFileIndex {
  readonly createdAt: number
  readonly pending: Promise<ProjectFileIndex>
}

interface ScoredProjectFile {
  readonly file: ProjectFileMention
  readonly normalizedPath: string
  readonly rank: number
}

/**
 * Project-wide file index used by `@file` completion.
 *
 * Index promises are cached as well as completed indexes so rapid keystrokes do
 * not start duplicate Git processes. Cache keys are resolved session project
 * directories; callers can explicitly invalidate after a known workspace change.
 */
export class ProjectFileMentionIndexCache {
  readonly #cache = new Map<string, CachedProjectFileIndex>()
  readonly #cacheTtlMs: number
  readonly #maxFiles: number

  constructor(options: ProjectFileMentionIndexOptions = {}) {
    this.#cacheTtlMs = nonNegativeInteger(
      options.cacheTtlMs ?? PROJECT_FILE_INDEX_CACHE_TTL_MS,
      'cacheTtlMs',
    )
    this.#maxFiles = Math.min(
      positiveInteger(options.maxFiles ?? PROJECT_FILE_INDEX_LIMIT, 'maxFiles'),
      PROJECT_FILE_INDEX_LIMIT,
    )
  }

  async search(projectDirectory: string, query: string): Promise<ProjectFileMentionSearchResult> {
    const index = await this.#index(projectDirectory)
    return {
      rootDirectory: index.rootDirectory,
      source: index.source,
      matches: searchProjectFileIndex(index, query),
    }
  }

  invalidate(projectDirectory?: string): void {
    if (projectDirectory === undefined) {
      this.#cache.clear()
      return
    }
    this.#cache.delete(resolve(projectDirectory))
  }

  async #index(projectDirectory: string): Promise<ProjectFileIndex> {
    const cacheKey = resolve(projectDirectory)
    const now = Date.now()
    for (const [key, entry] of this.#cache) {
      if (now - entry.createdAt >= this.#cacheTtlMs) this.#cache.delete(key)
    }
    const cached = this.#cache.get(cacheKey)
    if (cached !== undefined && now - cached.createdAt < this.#cacheTtlMs) {
      return await cached.pending
    }

    if (cached !== undefined) this.#cache.delete(cacheKey)
    const pending = loadProjectFileIndex(cacheKey, this.#maxFiles)
    const entry = { createdAt: now, pending }
    this.#cache.set(cacheKey, entry)
    try {
      return await pending
    } catch (error) {
      if (this.#cache.get(cacheKey) === entry) this.#cache.delete(cacheKey)
      throw error
    }
  }
}

const defaultProjectFileMentionIndex = new ProjectFileMentionIndexCache()

/** Search a cached project-wide index for a case-insensitive path substring. */
export async function searchProjectFileMentions(
  projectDirectory: string,
  query: string,
): Promise<ProjectFileMentionSearchResult> {
  return await defaultProjectFileMentionIndex.search(projectDirectory, query)
}

/** Drop the shared project index after a known filesystem change. */
export function invalidateProjectFileMentionIndex(projectDirectory?: string): void {
  defaultProjectFileMentionIndex.invalidate(projectDirectory)
}

/** Resolve the index root without collecting any project files. */
export async function resolveProjectFileMentionRoot(projectDirectory: string): Promise<string> {
  const cwd = resolve(projectDirectory)
  return await findGitRoot(cwd) ?? cwd
}

/**
 * Rank an already loaded index without I/O.
 *
 * Matching is a contiguous, case-insensitive substring over the normalized
 * project-relative path. The result is always capped at 50 entries.
 */
export function searchProjectFileIndex(
  index: ProjectFileIndex,
  query: string,
): readonly ProjectFileMention[] {
  const normalizedQuery = normalizeQuery(query)
  const scored: ScoredProjectFile[] = []

  for (const file of index.files) {
    const normalizedPath = file.relativePath.toLowerCase()
    if (!normalizedPath.includes(normalizedQuery)) continue
    const normalizedBasename = file.basename.toLowerCase()
    scored.push({
      file,
      normalizedPath,
      rank: matchRank(normalizedPath, normalizedBasename, normalizedQuery),
    })
  }

  scored.sort(compareScoredFiles)
  return scored.slice(0, PROJECT_FILE_MENTION_LIMIT).map(({ file }) => file)
}

async function loadProjectFileIndex(
  projectDirectory: string,
  maximum: number,
): Promise<ProjectFileIndex> {
  const gitRoot = await findGitRoot(projectDirectory)
  if (gitRoot !== undefined) {
    const files = await listGitProjectFiles(gitRoot, maximum)
    if (files !== undefined) return { rootDirectory: gitRoot, source: 'git', files }
  }

  return {
    rootDirectory: projectDirectory,
    source: 'filesystem',
    files: await walkProjectFiles(projectDirectory, maximum),
  }
}

async function findGitRoot(projectDirectory: string): Promise<string | undefined> {
  const output = await runGit(projectDirectory, ['rev-parse', '--show-toplevel'])
  const root = output?.trim()
  return root ? resolve(root) : undefined
}

async function listGitProjectFiles(
  gitRoot: string,
  maximum: number,
): Promise<readonly ProjectFileMention[] | undefined> {
  const output = await runGit(gitRoot, [
    'ls-files',
    '--cached',
    '--others',
    '--exclude-standard',
    '-z',
  ])
  if (output === undefined) return undefined

  const files: ProjectFileMention[] = []
  const seen = new Set<string>()
  for (const rawPath of output.split('\0')) {
    if (files.length >= maximum) break
    const relativePath = normalizeRelativePath(rawPath)
    if (!relativePath || relativePath === '..' || relativePath.startsWith('../')) continue
    if (isBinaryPath(relativePath)) continue
    if (seen.has(relativePath)) continue
    seen.add(relativePath)
    const file = projectFile(gitRoot, relativePath)
    if (!await Bun.file(file.absolutePath).exists()) continue
    files.push(file)
  }
  return files
}

async function runGit(cwd: string, args: readonly string[]): Promise<string | undefined> {
  try {
    const process = Bun.spawn(['git', ...args], {
      cwd,
      stdin: 'ignore',
      stdout: 'pipe',
      stderr: 'ignore',
    })
    const [exitCode, output] = await Promise.all([
      process.exited,
      new Response(process.stdout).text(),
    ])
    return exitCode === 0 ? output : undefined
  } catch {
    return undefined
  }
}

async function walkProjectFiles(
  rootDirectory: string,
  maximum: number,
): Promise<readonly ProjectFileMention[]> {
  const files: ProjectFileMention[] = []
  await walkDirectory(rootDirectory, rootDirectory, files, maximum)
  return files
}

async function walkDirectory(
  rootDirectory: string,
  directory: string,
  files: ProjectFileMention[],
  maximum: number,
): Promise<void> {
  let entries
  try {
    entries = await readdir(directory, { withFileTypes: true })
  } catch (error) {
    if (ignorableDirectoryReadError(error)) return
    throw error
  }
  entries.sort((left, right) => compareText(left.name, right.name))

  for (const entry of entries) {
    if (files.length >= maximum) return
    const absolutePath = resolve(directory, entry.name)
    if (entry.isDirectory()) {
      if (skipFallbackDirectory(entry.name)) continue
      await walkDirectory(rootDirectory, absolutePath, files, maximum)
      continue
    }
    if (!entry.isFile()) continue
    const relativePath = normalizeRelativePath(relative(rootDirectory, absolutePath))
    if (!relativePath || isBinaryPath(relativePath)) continue
    files.push(projectFile(rootDirectory, relativePath))
  }
}

function projectFile(rootDirectory: string, relativePath: string): ProjectFileMention {
  const separator = relativePath.lastIndexOf('/')
  return {
    relativePath,
    absolutePath: resolve(rootDirectory, ...relativePath.split('/')),
    basename: separator < 0 ? relativePath : relativePath.slice(separator + 1),
  }
}

function normalizeRelativePath(path: string): string {
  return path.replaceAll('\\', '/').replace(/^\.\/+/, '').replace(/\/{2,}/g, '/')
}

function normalizeQuery(query: string): string {
  const mention = query.startsWith('@') ? query.slice(1) : query
  return normalizeRelativePath(mention).toLowerCase()
}

function isBinaryPath(path: string): boolean {
  const extension = path.lastIndexOf('.')
  return extension >= 0 && BINARY_EXTENSIONS.has(path.slice(extension).toLowerCase())
}

function matchRank(path: string, basename: string, query: string): number {
  if (!query) return 4
  if (basename === query) return 0
  if (basename.startsWith(query)) return 1
  if (basename.includes(query)) return 2
  if (path.startsWith(query)) return 3
  return 4
}

function compareScoredFiles(left: ScoredProjectFile, right: ScoredProjectFile): number {
  if (left.rank !== right.rank) return left.rank - right.rank
  if (left.file.relativePath.length !== right.file.relativePath.length) {
    return left.file.relativePath.length - right.file.relativePath.length
  }
  const normalized = compareText(left.normalizedPath, right.normalizedPath)
  return normalized || compareText(left.file.relativePath, right.file.relativePath)
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}

function skipFallbackDirectory(name: string): boolean {
  return name === 'node_modules' || name.startsWith('.')
}

function ignorableDirectoryReadError(error: unknown): boolean {
  if (typeof error !== 'object' || error === null || !('code' in error)) return false
  const code = (error as { readonly code?: unknown }).code
  return code === 'EACCES' || code === 'ENOENT' || code === 'EPERM'
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive integer`)
  }
  return value
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative integer`)
  }
  return value
}
