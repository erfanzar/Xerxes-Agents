// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { realpathSync } from 'node:fs'
import { chmod, mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { basename, dirname, isAbsolute, relative, resolve } from 'node:path'

import { fileStateTracker, type FileStateTracker } from '../tools/fileState.js'
import { changedWorkspacePaths, type CommandResult, type CommandRunner } from './changeGuard.js'
import {
  captureCodeCheckBaseline,
  extractPatchTargetPaths,
  type CheckFinding,
} from './resultChecks.js'

const DEFAULT_MAX_SNAPSHOT_FILE_BYTES = 16 * 1024 * 1024

/** Tools that inspect workspace files without mutating them. */
export const READ_ONLY_INSPECTION_TOOL_NAMES: ReadonlySet<string> = new Set([
  'FileOutlineTool',
  'GlobTool',
  'GrepTool',
  'ListDir',
  'ReadFile',
  'XlsxReader',
  'file_outline',
  'file_outline_tool',
  'glob_tool',
  'grep_tool',
  'list_dir',
  'read_file',
])

const PATCH_MUTATING_TOOL_NAMES: ReadonlySet<string> = new Set([
  'ApplyPatchTool',
  'apply_patch',
  'apply_patch_tool',
  'git_apply_patch',
])

const FILE_MUTATING_TOOL_NAMES: ReadonlySet<string> = new Set([
  'AppendFile',
  'ApplyPatchTool',
  'Edit',
  'FileEditTool',
  'MultiEdit',
  'NotebookEdit',
  'NotebookEditTool',
  'Write',
  'WriteFile',
  'XlsxConvert',
  'XlsxWriter',
  'append_file',
  'apply_patch',
  'apply_patch_tool',
  'edit_file',
  'file_edit_tool',
  'git_apply_patch',
  'notebook_edit_tool',
  'replace_in_file',
  'write_file',
])

const CSV_TOOL_NAMES: ReadonlySet<string> = new Set(['CSVProcessor', 'csv_processor'])

/** True when `toolName` is a read-only codebase inspection tool (such as `FileOutlineTool`). */
export function isReadOnlyInspectionTool(toolName: string): boolean {
  return READ_ONLY_INSPECTION_TOOL_NAMES.has(toolName)
}

/**
 * Extract target paths that a mutating tool call will create, modify, move, or delete.
 * Read-only tools (including `FileOutlineTool` and `ReadFile`) always return `[]`.
 */
export function extractMutatedToolPaths(
  toolName: string,
  inputs: Readonly<Record<string, unknown>>,
): string[] {
  if (READ_ONLY_INSPECTION_TOOL_NAMES.has(toolName)) return []

  const paths: string[] = []
  const seen = new Set<string>()
  const push = (candidate: unknown): void => {
    if (typeof candidate !== 'string') return
    const trimmed = candidate.trim()
    if (!trimmed || seen.has(trimmed)) return
    seen.add(trimmed)
    paths.push(trimmed)
  }

  if (PATCH_MUTATING_TOOL_NAMES.has(toolName)) {
    const patchPayload = inputs['patch'] ?? inputs['diff'] ?? inputs['patch_text']
    if (typeof patchPayload === 'string' && patchPayload) {
      for (const target of extractPatchTargetPaths(patchPayload)) push(target)
    }
    push(inputs['file_path'])
    push(inputs['path'])
    return paths
  }

  if (toolName === 'XlsxConvert') {
    push(inputs['output_path'])
    return paths
  }

  if (CSV_TOOL_NAMES.has(toolName)) {
    const operation = typeof inputs['operation'] === 'string' ? inputs['operation'].trim() : ''
    if (/^(?:write|convert)$/iu.test(operation)) {
      push(inputs['output_path'] ?? inputs['file_path'] ?? inputs['path'])
    }
    return paths
  }

  if (!FILE_MUTATING_TOOL_NAMES.has(toolName)) return []

  push(inputs['file_path'] ?? inputs['output_path'] ?? inputs['path'] ?? inputs['notebook_path'])
  if (toolName === 'MultiEdit' && Array.isArray(inputs['edits'])) {
    for (const entry of inputs['edits']) {
      if (typeof entry === 'object' && entry !== null) {
        const record = entry as Record<string, unknown>
        push(record['file_path'] ?? record['path'])
      }
    }
  }
  return paths
}

export interface FileStateSnapshot {
  readonly exists: boolean
  readonly bytes?: Uint8Array
  readonly mode?: number
  /** True when the file exceeded `maxSnapshotFileBytes` and should not be overwritten on rollback. */
  readonly oversized?: boolean
}

export interface CascadeAttemptSnapshot {
  readonly attempt: number
  readonly errorCount: number
  readonly warnCount: number
  readonly findings: readonly CheckFinding[]
  readonly files: ReadonlyMap<string, FileStateSnapshot>
}

export interface CascadeRollbackResult {
  readonly restoredAttempt: number
  readonly bestErrors: number
  readonly lastErrors: number
  readonly revertedPaths: readonly string[]
  readonly bestFindings: readonly CheckFinding[]
  readonly notice: string
}

export interface WorkspaceCheckpointOptions {
  readonly sessionId?: string
  readonly maxSnapshotFileBytes?: number
  readonly commandRunner?: CommandRunner
  readonly fileTracker?: FileStateTracker
}

export interface RecordAttemptOptions {
  readonly additionalTouchedPaths?: readonly string[]
  readonly escalationAction?: 'retry_local' | 'change_model' | 'ask_advisor' | 'stop'
  readonly rollbackEnabled?: boolean
}

/** Build the canonical harness notice after restoring a regressed attempt. */
export function formatCascadeRollbackNotice(
  bestErrors: number,
  lastErrors: number,
  revertedPaths: readonly string[],
): string {
  return `[harness] Workspace restored to best prior attempt (had ${bestErrors} error(s) vs ${lastErrors} in last attempt; reverted ${revertedPaths.join(', ')})`
}

/**
 * Per-turn workspace checkpoint & baseline tracker for `cascade.ts`.
 *
 * Snapshots only files touched during the cascade in memory (never touching
 * `.git/index` and never clobbering pre-existing uncommitted user files outside
 * the turn's touched set), and restores the best prior attempt whenever a retry
 * regresses or escalates from a worse error state.
 */
export class WorkspaceCheckpointManager {
  readonly workspaceRoot: string
  readonly #sessionId: string | undefined
  readonly #maxSnapshotBytes: number
  readonly #commandRunner: CommandRunner | undefined
  readonly #fileTracker: FileStateTracker
  readonly #preTurnFiles = new Map<string, FileStateSnapshot>()
  readonly #touchedPaths = new Set<string>()
  readonly #baselineFindings = new Set<string>()
  #beganTurn = false
  #attemptCount = 0
  #bestAttempt: CascadeAttemptSnapshot | undefined

  constructor(workspaceRoot: string, options: WorkspaceCheckpointOptions = {}) {
    this.workspaceRoot = canonicalPath(workspaceRoot)
    this.#sessionId = options.sessionId
    this.#maxSnapshotBytes = options.maxSnapshotFileBytes ?? DEFAULT_MAX_SNAPSHOT_FILE_BYTES
    this.#commandRunner = options.commandRunner ?? readOnlyGitRunner
    this.#fileTracker = options.fileTracker ?? fileStateTracker
  }

  /** Pre-turn code-check fingerprints captured before files were mutated by this turn. */
  get baselineFindings(): ReadonlySet<string> {
    return this.#baselineFindings
  }

  /** Workspace-absolute paths touched during this cascade turn. */
  get touchedPaths(): readonly string[] {
    return [...this.#touchedPaths].sort((left, right) => left.localeCompare(right))
  }

  /** Best prior failing attempt (`lowest errorCount > 0`) recorded so far. */
  get bestAttempt(): CascadeAttemptSnapshot | undefined {
    return this.#bestAttempt
  }

  /**
   * Anchor pre-turn dirty files' baselines and initial byte snapshots via
   * read-only git status (if in a git repository), without marking them touched.
   */
  async beginTurn(): Promise<void> {
    if (this.#beganTurn) return
    this.#beganTurn = true
    const dirtyPaths = await changedWorkspacePaths(
      this.workspaceRoot,
      this.#commandRunner === undefined ? {} : { commandRunner: this.#commandRunner },
    )
    if (dirtyPaths.length > 0) {
      await this.capturePreTurnPaths(dirtyPaths, { markTouched: false })
    }
  }

  /**
   * Capture pre-mutation file bytes + existence + code-check baseline for
   * `candidatePaths`. Safe to call multiple times: only the first snapshot per
   * path is retained as its pre-turn state.
   */
  async capturePreTurnPaths(
    candidatePaths: readonly string[],
    options: { readonly markTouched?: boolean } = {},
  ): Promise<void> {
    const newBaselineTargets: string[] = []
    for (const candidate of candidatePaths) {
      if (!candidate) continue
      const absolute = this.#resolveInsideWorkspace(candidate)
      if (absolute === undefined) continue
      if (!this.#preTurnFiles.has(absolute)) {
        const snapshot = await this.#snapshotFile(absolute)
        this.#preTurnFiles.set(absolute, snapshot)
        if (snapshot.exists && !snapshot.oversized) {
          newBaselineTargets.push(absolute)
        }
      }
      if (options.markTouched === true) {
        this.#touchedPaths.add(absolute)
      }
    }
    if (newBaselineTargets.length > 0) {
      const captured = await captureCodeCheckBaseline(this.workspaceRoot, newBaselineTargets)
      for (const fingerprint of captured) {
        this.#baselineFindings.add(fingerprint)
      }
    }
  }

  /** Hook invoked immediately before a tool executes during a cascade round. */
  async noteBeforeToolExecution(
    toolName: string,
    inputs: Readonly<Record<string, unknown>>,
  ): Promise<void> {
    const mutated = extractMutatedToolPaths(toolName, inputs)
    if (mutated.length === 0) return
    await this.capturePreTurnPaths(mutated, { markTouched: true })
  }

  /**
   * Record a completed attempt's findings, update the best-attempt watermark,
   * and — when `rollbackEnabled !== false` and the attempt regressed or
   * escalated from a worse state — restore touched files to the best attempt.
   */
  async recordAttemptAndMaybeRollback(
    findings: readonly CheckFinding[],
    options: RecordAttemptOptions = {},
  ): Promise<CascadeRollbackResult | undefined> {
    this.#attemptCount += 1
    for (const rawPath of options.additionalTouchedPaths ?? []) {
      const absolute = this.#resolveInsideWorkspace(rawPath)
      if (absolute !== undefined) this.#touchedPaths.add(absolute)
    }
    await this.#detectAdditionalDirtyTouchedPaths()

    let errorCount = 0
    let warnCount = 0
    for (const finding of findings) {
      if (finding.severity === 'error') errorCount += 1
      else warnCount += 1
    }

    if (errorCount === 0) return undefined

    const currentFiles = await this.#snapshotTouchedSet()
    const currentSnapshot: CascadeAttemptSnapshot = {
      attempt: this.#attemptCount,
      errorCount,
      warnCount,
      findings: [...findings],
      files: currentFiles,
    }

    const best = this.#bestAttempt
    if (best === undefined) {
      this.#bestAttempt = currentSnapshot
      return undefined
    }

    const strictlyBetter = errorCount < best.errorCount
      || (errorCount === best.errorCount && warnCount < best.warnCount)
    if (strictlyBetter) {
      this.#bestAttempt = currentSnapshot
      return undefined
    }

    const rollbackEnabled = options.rollbackEnabled ?? true
    if (!rollbackEnabled) return undefined

    const isEscalating = options.escalationAction === 'change_model' || options.escalationAction === 'ask_advisor'
    const regressedErrors = errorCount > best.errorCount
    const worseOnEscalation = isEscalating
      && (errorCount > best.errorCount || (errorCount === best.errorCount && warnCount > best.warnCount))

    if (!regressedErrors && !worseOnEscalation) return undefined

    const revertedPaths = await this.#restoreToAttempt(best, currentFiles)
    if (revertedPaths.length === 0) return undefined

    return {
      restoredAttempt: best.attempt,
      bestErrors: best.errorCount,
      lastErrors: errorCount,
      revertedPaths,
      bestFindings: best.findings,
      notice: formatCascadeRollbackNotice(best.errorCount, errorCount, revertedPaths),
    }
  }

  async #detectAdditionalDirtyTouchedPaths(): Promise<void> {
    if (!this.#beganTurn) return
    const dirtyNow = await changedWorkspacePaths(
      this.workspaceRoot,
      this.#commandRunner === undefined ? {} : { commandRunner: this.#commandRunner },
    )
    for (const candidate of dirtyNow) {
      const absolute = this.#resolveInsideWorkspace(candidate)
      if (absolute === undefined || this.#touchedPaths.has(absolute)) continue
      const preTurn = this.#preTurnFiles.get(absolute)
      if (preTurn === undefined) {
        // Newly modified or untracked since beginTurn() (e.g. via exec_command).
        this.#touchedPaths.add(absolute)
        continue
      }
      const current = await this.#snapshotFile(absolute)
      if (!snapshotsEqual(preTurn, current)) {
        this.#touchedPaths.add(absolute)
      }
    }
  }

  async #snapshotTouchedSet(): Promise<Map<string, FileStateSnapshot>> {
    const map = new Map<string, FileStateSnapshot>()
    for (const absolute of this.#touchedPaths) {
      map.set(absolute, await this.#snapshotFile(absolute))
    }
    return map
  }

  async #restoreToAttempt(
    targetAttempt: CascadeAttemptSnapshot,
    currentFiles: ReadonlyMap<string, FileStateSnapshot>,
  ): Promise<string[]> {
    const reverted: string[] = []
    for (const absolute of this.#touchedPaths) {
      const desired = targetAttempt.files.get(absolute) ?? this.#preTurnFiles.get(absolute)
      if (desired === undefined || desired.oversized === true) continue
      const current = currentFiles.get(absolute) ?? await this.#snapshotFile(absolute)
      if (snapshotsEqual(desired, current)) continue
      await this.#applyFileSnapshot(absolute, desired)
      reverted.push(this.#displayPath(absolute))
    }
    reverted.sort((left, right) => left.localeCompare(right))
    return reverted
  }

  async #applyFileSnapshot(absolute: string, desired: FileStateSnapshot): Promise<void> {
    if (!desired.exists || desired.bytes === undefined) {
      await rm(absolute, { force: true })
      if (this.#sessionId !== undefined && this.#sessionId !== '') {
        this.#fileTracker.forget(this.#sessionId, absolute)
      }
      return
    }

    await mkdir(dirname(absolute), { recursive: true })
    const temporary = `${dirname(absolute)}/.${basename(absolute)}.${crypto.randomUUID()}.cascade-tmp`
    try {
      await writeFile(temporary, desired.bytes)
      if (desired.mode !== undefined) {
        await chmod(temporary, desired.mode)
      }
      await rename(temporary, absolute)
    } catch (error) {
      await rm(temporary, { force: true }).catch(() => undefined)
      throw error
    }

    if (this.#sessionId !== undefined && this.#sessionId !== '' && this.#fileTracker.peek(this.#sessionId, absolute) !== undefined) {
      try {
        const writtenStats = await stat(absolute)
        const text = new TextDecoder('utf-8', { fatal: true }).decode(desired.bytes)
        this.#fileTracker.record(this.#sessionId, absolute, text, {
          mtimeMs: writtenStats.mtimeMs,
          partialView: false,
          size: writtenStats.size,
        })
      } catch {
        this.#fileTracker.forget(this.#sessionId, absolute)
      }
    }
  }

  async #snapshotFile(absolute: string): Promise<FileStateSnapshot> {
    try {
      const info = await stat(absolute)
      if (!info.isFile()) return { exists: false }
      if (info.size > this.#maxSnapshotBytes) {
        return { exists: true, mode: info.mode, oversized: true }
      }
      const bytes = new Uint8Array(await readFile(absolute))
      return { exists: true, bytes, mode: info.mode }
    } catch (error) {
      if (typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT') {
        return { exists: false }
      }
      throw error
    }
  }

  #resolveInsideWorkspace(candidate: string): string | undefined {
    const resolved = canonicalPath(isAbsolute(candidate) ? candidate : resolve(this.workspaceRoot, candidate))
    if (resolved === this.workspaceRoot) return undefined
    const rel = relative(this.workspaceRoot, resolved)
    if (!rel || rel.startsWith('..') || isAbsolute(rel)) return undefined
    const firstSegment = rel.split(/[\\/]/u)[0]
    if (firstSegment === '.git') return undefined
    return resolved
  }

  #displayPath(absolute: string): string {
    const rel = relative(this.workspaceRoot, absolute).replaceAll('\\', '/')
    return rel || basename(absolute)
  }
}

async function readOnlyGitRunner(
  args: readonly string[],
  options: { readonly cwd: string },
): Promise<CommandResult> {
  const argv = args[0] === 'git' ? ['git', '--no-optional-locks', ...args.slice(1)] : [...args]
  const proc = Bun.spawn(argv, {
    cwd: options.cwd,
    env: { ...process.env, GIT_OPTIONAL_LOCKS: '0' },
    stdin: 'ignore',
    stdout: 'pipe',
    stderr: 'pipe',
  })
  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ])
  const exitCode = await proc.exited
  return { exitCode, stdout, stderr }
}

function canonicalPath(target: string): string {
  const abs = resolve(target)
  try {
    return realpathSync(abs)
  } catch {
    const parent = dirname(abs)
    if (parent === abs) return abs
    return resolve(canonicalPath(parent), basename(abs))
  }
}

function snapshotsEqual(left: FileStateSnapshot, right: FileStateSnapshot): boolean {
  if (left.exists !== right.exists) return false
  if (!left.exists) return true
  if (left.oversized || right.oversized) return left.oversized === right.oversized
  const leftBytes = left.bytes
  const rightBytes = right.bytes
  if (leftBytes === undefined || rightBytes === undefined) return leftBytes === rightBytes
  if (leftBytes.byteLength !== rightBytes.byteLength) return false
  for (let index = 0; index < leftBytes.byteLength; index += 1) {
    if (leftBytes[index] !== rightBytes[index]) return false
  }
  return true
}
