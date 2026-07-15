// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { copyFile, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import type { Dirent } from 'node:fs'
import { isAbsolute, relative, resolve } from 'node:path'

const OWNER_FILE = '.evaluation-owner.json'
const PROFILE_FILE = 'profiles.json'
const RUN_PREFIX = 'run-'

/** Configuration for one isolated native evaluation run. */
export interface EvaluationIsolationOptions {
  /** Parent directory for all per-run homes and workspaces. */
  readonly rootDirectory: string
  /** Copy only this explicit profile file into the isolated home, when present. */
  readonly profileSourceDirectory?: string
  /** Deterministic identifier for tests or a caller-owned run coordinator. */
  readonly runId?: string
}

/** Filesystem paths owned exclusively by one evaluation run. */
export interface EvaluationIsolation {
  readonly homeDirectory: string
  readonly runDirectory: string
  readonly runId: string
  readonly workspaceDirectory: string
  cleanup(): Promise<void>
}

interface OwnerRecord {
  readonly pid: number
}

/**
 * Create a private home and workspace without mutating process environment.
 *
 * The embedding runtime receives these paths through `EvaluationStartRequest`.
 * That makes profile and daemon selection explicit instead of allowing a test
 * import to silently redirect the host's default home directory.
 */
export async function createEvaluationIsolation(options: EvaluationIsolationOptions): Promise<EvaluationIsolation> {
  const rootDirectory = resolveRequiredDirectory(options.rootDirectory, 'rootDirectory')
  await mkdir(rootDirectory, { recursive: true })
  await sweepStaleEvaluationRuns(rootDirectory)

  const runId = options.runId === undefined ? generatedRunId() : validateRunId(options.runId)
  const runDirectory = childPath(rootDirectory, runId)
  const existingOwner = await readOwner(runDirectory)
  if (existingOwner !== undefined && processIsAlive(existingOwner.pid)) {
    throw new Error(`evaluation run ${runId} is still owned by process ${existingOwner.pid}`)
  }
  await rm(runDirectory, { force: true, recursive: true })
  await mkdir(runDirectory, { recursive: true })

  const homeDirectory = childPath(runDirectory, 'home')
  const workspaceDirectory = childPath(runDirectory, 'workspace')
  await Promise.all([
    mkdir(homeDirectory, { recursive: true }),
    mkdir(workspaceDirectory, { recursive: true }),
    writeOwner(runDirectory),
  ])

  if (options.profileSourceDirectory !== undefined) {
    await copyExplicitProfile(options.profileSourceDirectory, homeDirectory)
  }

  return {
    homeDirectory,
    runDirectory,
    runId,
    workspaceDirectory,
    cleanup: () => rm(runDirectory, { force: true, recursive: true }),
  }
}

/** Remove abandoned evaluation directories whose recorded owner is gone. */
export async function sweepStaleEvaluationRuns(rootDirectory: string): Promise<void> {
  const resolvedRoot = resolveRequiredDirectory(rootDirectory, 'rootDirectory')
  let entries: Dirent<string>[]
  try {
    entries = await readdir(resolvedRoot, { encoding: 'utf8', withFileTypes: true })
  } catch (error) {
    if (isMissing(error)) return
    throw error
  }

  await Promise.all(entries
    .filter(entry => entry.isDirectory() && entry.name.startsWith(RUN_PREFIX))
    .map(async entry => {
      const runDirectory = childPath(resolvedRoot, entry.name)
      const owner = await readOwner(runDirectory)
      if (owner !== undefined && processIsAlive(owner.pid)) return
      await rm(runDirectory, { force: true, recursive: true })
    }))
}

/** Return whether an OS process still owns a run directory. */
export function processIsAlive(pid: number): boolean {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    if (isPermissionError(error)) return true
    if (isNodeError(error, 'ESRCH')) return false
    throw error
  }
}

async function copyExplicitProfile(sourceDirectory: string, homeDirectory: string): Promise<void> {
  const source = resolveRequiredDirectory(sourceDirectory, 'profileSourceDirectory')
  const profile = childPath(source, PROFILE_FILE)
  try {
    await copyFile(profile, childPath(homeDirectory, PROFILE_FILE))
  } catch (error) {
    if (isMissing(error)) return
    throw error
  }
}

async function writeOwner(runDirectory: string): Promise<void> {
  const owner: OwnerRecord = { pid: process.pid }
  await writeFile(childPath(runDirectory, OWNER_FILE), JSON.stringify(owner) + '\n', 'utf8')
}

async function readOwner(runDirectory: string): Promise<OwnerRecord | undefined> {
  try {
    const raw = await readFile(childPath(runDirectory, OWNER_FILE), 'utf8')
    const parsed: unknown = JSON.parse(raw)
    if (!isOwnerRecord(parsed)) return undefined
    return parsed
  } catch (error) {
    if (isMissing(error) || error instanceof SyntaxError) return undefined
    throw error
  }
}

function isOwnerRecord(value: unknown): value is OwnerRecord {
  return typeof value === 'object'
    && value !== null
    && 'pid' in value
    && typeof value.pid === 'number'
}

function generatedRunId(): string {
  return `${RUN_PREFIX}${process.pid}-${crypto.randomUUID()}`
}

function validateRunId(value: string): string {
  if (!/^run-[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value)) {
    throw new Error('runId must begin with "run-" and contain only letters, digits, dot, underscore, or dash')
  }
  return value
}

function resolveRequiredDirectory(value: string, label: string): string {
  const trimmed = value.trim()
  if (!trimmed) throw new Error(`${label} must not be empty`)
  return resolve(trimmed)
}

function childPath(parent: string, child: string): string {
  const candidate = resolve(parent, child)
  const fromParent = relative(parent, candidate)
  if (!fromParent || fromParent === '..' || fromParent.startsWith('../') || isAbsolute(fromParent)) {
    throw new Error(`unsafe child path: ${child}`)
  }
  return candidate
}

function isMissing(error: unknown): boolean {
  return isNodeError(error, 'ENOENT')
}

function isPermissionError(error: unknown): boolean {
  return isNodeError(error, 'EPERM') || isNodeError(error, 'EACCES')
}

function isNodeError(error: unknown, code: string): boolean {
  return typeof error === 'object'
    && error !== null
    && 'code' in error
    && error.code === code
}
