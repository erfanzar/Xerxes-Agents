// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { copyFile, mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

import {
  BenchmarkDatasetError,
  type BenchmarkId,
  type BenchmarkTask,
  type GradeResult,
  type LoadOptions,
  type LoadResult,
} from './types.js'

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(entry => typeof entry === 'string')
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function errorCode(error: unknown): string | undefined {
  return typeof error === 'object' && error !== null && 'code' in error && typeof error.code === 'string'
    ? error.code
    : undefined
}

export function isMissingError(error: unknown): boolean {
  const code = errorCode(error)
  return code === 'ENOENT' || code === 'ENOTDIR'
}

export async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path)
    return true
  } catch (error) {
    if (isMissingError(error)) return false
    throw error
  }
}

/** Join `rel` under `root`, refusing absolute paths and anything that escapes the root. */
export function safeJoin(root: string, rel: string): string {
  if (!rel || isAbsolute(rel) || rel.includes('\0')) throw new Error(`unsafe relative path: ${JSON.stringify(rel)}`)
  const resolved = resolve(root, rel)
  const fromRoot = relative(root, resolved)
  if (!fromRoot || fromRoot === '..' || fromRoot.startsWith(`..${sep}`) || isAbsolute(fromRoot)) {
    throw new Error(`path escapes its root: ${JSON.stringify(rel)}`)
  }
  return resolved
}

/** Read a required dataset file; a missing one is a typed layout error naming the expected files. */
export async function readDatasetText(
  adapter: BenchmarkId,
  root: string,
  rel: string,
  expected: readonly string[],
): Promise<string> {
  try {
    return await readFile(join(root, rel), 'utf8')
  } catch (error) {
    if (isMissingError(error)) throw new BenchmarkDatasetError(adapter, root, `missing ${rel}`, expected)
    throw error
  }
}

export function parseJsonl(
  adapter: BenchmarkId,
  root: string,
  rel: string,
  text: string,
  expected: readonly string[],
): Record<string, unknown>[] {
  const records: Record<string, unknown>[] = []
  const lines = text.split('\n')
  for (let index = 0; index < lines.length; index += 1) {
    const line = (lines[index] ?? '').trim()
    if (!line) continue
    let parsed: unknown
    try {
      parsed = JSON.parse(line)
    } catch (error) {
      throw new BenchmarkDatasetError(adapter, root, `${rel} line ${index + 1} is not JSON (${errorMessage(error)})`, expected)
    }
    if (!isRecord(parsed)) {
      throw new BenchmarkDatasetError(adapter, root, `${rel} line ${index + 1} is not a JSON object`, expected)
    }
    records.push(parsed)
  }
  return records
}

/** A required non-empty string field; numbers are accepted and stringified (task ids are often numeric). */
export function requiredText(
  record: Record<string, unknown>,
  key: string,
  fail: (problem: string) => BenchmarkDatasetError,
): string {
  const value = record[key]
  if (typeof value === 'string' && value.trim()) return value
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  throw fail(`record is missing a non-empty "${key}"`)
}

export function optionalText(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key]
  return typeof value === 'string' && value.trim() ? value : undefined
}

/** Regular files under `dir` (relative, `/`-separated, sorted). Symlinks are skipped, never followed. */
export async function listFilesRecursive(dir: string, maxFiles = 2_000): Promise<string[]> {
  const found: string[] = []
  const walk = async (current: string, prefix: string): Promise<void> => {
    const entries = await readdir(current, { withFileTypes: true })
    entries.sort((left, right) => left.name.localeCompare(right.name))
    for (const entry of entries) {
      if (found.length >= maxFiles) return
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name
      if (entry.isDirectory()) await walk(join(current, entry.name), rel)
      else if (entry.isFile()) found.push(rel)
    }
  }
  await walk(dir, '')
  return found
}

export function applySelection(result: LoadResult, opts: LoadOptions): LoadResult {
  const ids = opts.ids === undefined ? undefined : new Set(opts.ids)
  let tasks = ids === undefined ? [...result.tasks] : result.tasks.filter(task => ids.has(task.id))
  if (opts.limit !== undefined) tasks = tasks.slice(0, Math.max(0, opts.limit))
  const excluded = ids === undefined ? result.excluded : result.excluded.filter(entry => ids.has(entry.id))
  return { tasks, excluded }
}

export function verdict(accepted: boolean, score: number, details: string): GradeResult {
  return { accepted, score, details, status: accepted ? 'accepted' : 'rejected', disclosureFailure: false }
}

/** Write a task's files into a fresh working directory. */
export async function materializeTaskFiles(task: BenchmarkTask, workdir: string): Promise<void> {
  for (const file of task.files) {
    const target = safeJoin(workdir, file.path)
    await mkdir(dirname(target), { recursive: true })
    if ('contents' in file) await writeFile(target, file.contents, 'utf8')
    else await copyFile(file.sourcePath, target)
  }
}
