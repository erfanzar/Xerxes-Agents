// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { cp, mkdir, mkdtemp, readdir, readFile, realpath, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, isAbsolute, join, posix, relative, resolve } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { xerxesHome } from '../daemon/paths.js'
import type {
  SwarmWorkspaceAllocation,
  SwarmWorkspaceCleanupRequest,
  SwarmWorkspaceCreateRequest,
  SwarmWorkspacePatchApplyRequest,
  SwarmWorkspacePatchInspection,
  SwarmWorkspacePatchInspectRequest,
  SwarmWorkspacePort,
} from './types.js'

const gitWorktreeLocks = new Map<string, Promise<void>>()

const CONFLICT_MARKER_LINE_REGEX = /^(?:<{7}(?:\s|$)|={7}\s*$|>{7}(?:\s|$)|\|{7}(?:\s|$))/m
const ADDED_CONFLICT_MARKER_LINE_REGEX = /^\+(?!\+\+\s)(?:<{7}(?:\s|$)|={7}\s*$|>{7}(?:\s|$)|\|{7}(?:\s|$))/m

async function withWorktreeLock<T>(lockKey: string, task: () => Promise<T>): Promise<T> {
  const pending = (gitWorktreeLocks.get(lockKey) ?? Promise.resolve()).then(task, task)
  const settled = pending.then(() => undefined, () => undefined)
  gitWorktreeLocks.set(lockKey, settled)
  try {
    return await pending
  } finally {
    if (gitWorktreeLocks.get(lockKey) === settled) {
      gitWorktreeLocks.delete(lockKey)
    }
  }
}

function sanitizedGitEnv(): Record<string, string | undefined> {
  const env: Record<string, string | undefined> = {
    ...Bun.env,
    GIT_OPTIONAL_LOCKS: '0',
  }
  for (const key of [
    'GIT_DIR',
    'GIT_WORK_TREE',
    'GIT_COMMON_DIR',
    'GIT_INDEX_FILE',
    'GIT_OBJECT_DIRECTORY',
    'GIT_ALTERNATE_OBJECT_DIRECTORIES',
    'GIT_PREFIX',
  ]) {
    delete env[key]
  }
  return env
}

async function runGitRaw(
  args: readonly string[],
  cwd: string,
  signal?: AbortSignal,
): Promise<{ readonly exitCode: number; readonly stdout: string; readonly stderr: string }> {
  signal?.throwIfAborted()
  const timeoutSignal = AbortSignal.timeout(60_000)
  const combinedSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal
  const proc = Bun.spawn(['git', '-c', 'core.hooksPath=/dev/null', ...args], {
    cwd,
    env: sanitizedGitEnv(),
    stdout: 'pipe',
    stderr: 'pipe',
    signal: combinedSignal,
  })
  const [exitCode, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ])
  return {
    exitCode,
    stdout,
    stderr: stderr.trim(),
  }
}

async function runGit(
  args: readonly string[],
  cwd: string,
  signal?: AbortSignal,
): Promise<{ readonly exitCode: number; readonly stdout: string; readonly stderr: string }> {
  const result = await runGitRaw(args, cwd, signal)
  return {
    exitCode: result.exitCode,
    stdout: result.stdout.trim(),
    stderr: result.stderr,
  }
}

function ensureTrailingNewline(text: string): string {
  if (!text) return ''
  return text.endsWith('\n') ? text : `${text}\n`
}

function normalizeRelativePosixPath(value: string): string {
  return value.replaceAll('\\', '/').replace(/^\.\//, '')
}

function parseNullDelimitedPaths(raw: string): string[] {
  if (!raw) return []
  const result: string[] = []
  for (const entry of raw.split('\0')) {
    if (!entry) continue
    result.push(normalizeRelativePosixPath(entry))
  }
  return result
}

function parseNumstatRecords(raw: string, nullDelimited: boolean): { insertions: number; deletions: number } {
  let insertions = 0
  let deletions = 0
  if (!raw) return { insertions, deletions }
  const records = nullDelimited ? raw.split('\0') : raw.split(/\r?\n/)
  for (const record of records) {
    if (!record) continue
    const parts = record.split('\t')
    const rawIns = parts[0]
    const rawDel = parts[1]
    if (rawIns && rawIns !== '-') {
      const parsedIns = Number.parseInt(rawIns, 10)
      if (Number.isFinite(parsedIns) && parsedIns > 0) insertions += parsedIns
    }
    if (rawDel && rawDel !== '-') {
      const parsedDel = Number.parseInt(rawDel, 10)
      if (Number.isFinite(parsedDel) && parsedDel > 0) deletions += parsedDel
    }
  }
  return { insertions, deletions }
}

function validateSafeRelativePatchPath(rawPath: string): void {
  const cleaned = rawPath.replaceAll('\\', '/')
  if (!cleaned || cleaned === '/dev/null') return
  const stripped = cleaned.startsWith('a/') || cleaned.startsWith('b/') ? cleaned.slice(2) : cleaned
  if (!stripped || isAbsolute(stripped) || stripped.startsWith('/')) {
    throw new ValidationError('patchText', `unsafe absolute path in candidate patch: "${rawPath}"`, rawPath)
  }
  const normalized = posix.normalize(stripped)
  if (normalized === '..' || normalized.startsWith('../') || normalized.includes('/../')) {
    throw new ValidationError('patchText', `path traversal in candidate patch: "${rawPath}"`, rawPath)
  }
  const segments = normalized.split('/')
  if (segments.includes('.git')) {
    throw new ValidationError('patchText', `candidate patch cannot target .git metadata: "${rawPath}"`, rawPath)
  }
}

export function assertSafePatchPaths(patchText: string): void {
  for (const line of patchText.split(/\r?\n/)) {
    if (line.startsWith('diff --git ')) {
      const match = /^diff --git "?a\/([^"\t\r\n]+)"? "?b\/([^"\t\r\n]+)"?$/.exec(line)
      if (match) {
        if (match[1]) validateSafeRelativePatchPath(match[1])
        if (match[2]) validateSafeRelativePatchPath(match[2])
      }
    } else if (line.startsWith('--- ') || line.startsWith('+++ ')) {
      const rawTarget = line.slice(4).split('\t')[0]?.trim().replace(/^"|"$/g, '') ?? ''
      if (rawTarget && rawTarget !== '/dev/null') {
        validateSafeRelativePatchPath(rawTarget)
      }
    }
  }
}

export function hasConflictMarkersInText(content: string): boolean {
  if (!content || content.includes('\0')) return false
  return CONFLICT_MARKER_LINE_REGEX.test(content)
}

async function detectWorkspaceConflictMarkers(
  workspacePath: string,
  changedFiles: readonly string[],
  patchText: string,
): Promise<boolean> {
  if (patchText && ADDED_CONFLICT_MARKER_LINE_REGEX.test(patchText)) {
    return true
  }
  for (const relPath of changedFiles) {
    validateSafeRelativePatchPath(relPath)
    const fullPath = join(workspacePath, relPath)
    try {
      const content = await readFile(fullPath, 'utf8')
      if (hasConflictMarkersInText(content)) {
        return true
      }
    } catch {
      // Deleted or unreadable file in working tree; patchText check already inspected hunks.
    }
  }
  return false
}

async function withTempPatchFile<T>(
  patchText: string,
  task: (patchFilePath: string) => Promise<T>,
): Promise<T> {
  const tempDir = await mkdtemp(join(tmpdir(), 'xerxes-swarm-patch-'))
  const patchFilePath = join(tempDir, 'candidate.patch')
  try {
    await writeFile(patchFilePath, ensureTrailingNewline(patchText), { encoding: 'utf8', mode: 0o600 })
    return await task(patchFilePath)
  } finally {
    await rm(tempDir, { recursive: true, force: true }).catch(() => undefined)
  }
}

export async function checkWorkspacePatchApplicable(
  request: SwarmWorkspacePatchApplyRequest,
): Promise<boolean> {
  if (!request.patchText.trim()) return false
  try {
    assertSafePatchPaths(request.patchText)
  } catch {
    return false
  }
  const resolvedTarget = await realpath(resolve(request.targetDir)).catch(() => resolve(request.targetDir))
  return withTempPatchFile(request.patchText, async (patchFilePath) => {
    const check = await runGit(['apply', '--check', '--binary', patchFilePath], resolvedTarget, request.signal)
    return check.exitCode === 0
  })
}

export async function applyWorkspacePatch(
  request: SwarmWorkspacePatchApplyRequest,
): Promise<void> {
  if (!request.patchText.trim()) {
    throw new ValidationError('patchText', 'cannot apply an empty patch')
  }
  assertSafePatchPaths(request.patchText)
  const resolvedTarget = await realpath(resolve(request.targetDir)).catch(() => resolve(request.targetDir))
  await withTempPatchFile(request.patchText, async (patchFilePath) => {
    const check = await runGit(['apply', '--check', '--binary', patchFilePath], resolvedTarget, request.signal)
    if (check.exitCode !== 0) {
      throw new ValidationError(
        'patchText',
        `candidate patch failed git apply --check in "${resolvedTarget}": ${check.stderr || `exit ${check.exitCode}`}`,
      )
    }
    const applied = await runGit(['apply', '--binary', patchFilePath], resolvedTarget, request.signal)
    if (applied.exitCode !== 0) {
      throw new ValidationError(
        'patchText',
        `candidate patch failed git apply in "${resolvedTarget}": ${applied.stderr || `exit ${applied.exitCode}`}`,
      )
    }
  })
}

async function resolveBaseHeadInGitWorkspace(
  workspacePath: string,
  baseDir: string,
  mode: 'share' | 'branch',
  explicitBaseHead: string | undefined,
  signal: AbortSignal | undefined,
): Promise<string | undefined> {
  const commitExists = async (rev: string): Promise<string | undefined> => {
    const clean = rev.trim()
    if (!clean) return undefined
    const resolved = await runGit(['rev-parse', '--verify', `${clean}^{commit}`], workspacePath, signal)
    return resolved.exitCode === 0 && resolved.stdout ? resolved.stdout : undefined
  }

  if (explicitBaseHead) {
    const verified = await commitExists(explicitBaseHead)
    if (verified) return verified
  }
  if (mode === 'branch') {
    const originHead = await commitExists('origin/HEAD')
    if (originHead) return originHead
  }
  const parentHead = await runGit(['rev-parse', '--verify', 'HEAD'], baseDir, signal)
  if (parentHead.exitCode === 0 && parentHead.stdout) {
    const verifiedParent = await commitExists(parentHead.stdout)
    if (verifiedParent) return verifiedParent
  }
  return commitExists('HEAD')
}

async function walkNonGitFiles(rootDir: string, currentDir = rootDir, acc: string[] = []): Promise<string[]> {
  const entries = await readdir(currentDir, { withFileTypes: true })
  for (const entry of entries) {
    if (entry.name === '.git' || entry.name === 'node_modules') continue
    const fullPath = join(currentDir, entry.name)
    if (entry.isDirectory()) {
      await walkNonGitFiles(rootDir, fullPath, acc)
    } else if (entry.isFile()) {
      acc.push(normalizeRelativePosixPath(relative(rootDir, fullPath)))
    }
  }
  return acc
}

function rewriteNoIndexHeaderPaths(rawDiff: string, relPath: string): string {
  const lines = rawDiff.split('\n')
  const rewritten = lines.map((line) => {
    if (line.startsWith('diff --git ')) {
      return `diff --git a/${relPath} b/${relPath}`
    }
    if (line.startsWith('--- ') && !line.startsWith('--- /dev/null')) {
      return `--- a/${relPath}`
    }
    if (line.startsWith('+++ ') && !line.startsWith('+++ /dev/null')) {
      return `+++ b/${relPath}`
    }
    return line
  })
  return ensureTrailingNewline(rewritten.join('\n').trimEnd())
}

async function inspectNonGitDirectoryDiff(
  baseDir: string,
  workspacePath: string,
  targetDir: string,
  signal?: AbortSignal,
): Promise<SwarmWorkspacePatchInspection> {
  const [baseFiles, workFiles] = await Promise.all([
    walkNonGitFiles(baseDir),
    walkNonGitFiles(workspacePath),
  ])
  const baseSet = new Set(baseFiles)
  const workSet = new Set(workFiles)
  const allCandidates = [...new Set([...baseFiles, ...workFiles])].sort()

  const changedFiles: string[] = []
  const patchChunks: string[] = []
  let insertions = 0
  let deletions = 0

  for (const relPath of allCandidates) {
    const inBase = baseSet.has(relPath)
    const inWork = workSet.has(relPath)
    if (!inBase && inWork) {
      const diffRes = await runGitRaw(
        ['diff', '--binary', '--full-index', '--no-index', '--', '/dev/null', relPath],
        workspacePath,
        signal,
      )
      const statRes = await runGit(
        ['diff', '--numstat', '--no-index', '--', '/dev/null', relPath],
        workspacePath,
        signal,
      )
      const stats = parseNumstatRecords(statRes.stdout, false)
      changedFiles.push(relPath)
      insertions += stats.insertions
      deletions += stats.deletions
      if (diffRes.stdout) patchChunks.push(ensureTrailingNewline(diffRes.stdout))
    } else if (inBase && !inWork) {
      const diffRes = await runGitRaw(
        ['diff', '--binary', '--full-index', '--no-index', '--', relPath, '/dev/null'],
        baseDir,
        signal,
      )
      const statRes = await runGit(
        ['diff', '--numstat', '--no-index', '--', relPath, '/dev/null'],
        baseDir,
        signal,
      )
      const stats = parseNumstatRecords(statRes.stdout, false)
      changedFiles.push(relPath)
      insertions += stats.insertions
      deletions += stats.deletions
      if (diffRes.stdout) patchChunks.push(ensureTrailingNewline(diffRes.stdout))
    } else {
      const absBase = join(baseDir, relPath)
      const absWork = join(workspacePath, relPath)
      const [baseBuf, workBuf] = await Promise.all([readFile(absBase), readFile(absWork)])
      if (baseBuf.equals(workBuf)) continue
      const diffRes = await runGitRaw(
        ['diff', '--binary', '--full-index', '--no-index', '--', absBase, absWork],
        workspacePath,
        signal,
      )
      const statRes = await runGit(
        ['diff', '--numstat', '--no-index', '--', absBase, absWork],
        workspacePath,
        signal,
      )
      const stats = parseNumstatRecords(statRes.stdout, false)
      changedFiles.push(relPath)
      insertions += stats.insertions
      deletions += stats.deletions
      if (diffRes.stdout) patchChunks.push(rewriteNoIndexHeaderPaths(diffRes.stdout, relPath))
    }
  }

  const patchText = patchChunks.join('')
  const hasConflictMarkers = await detectWorkspaceConflictMarkers(workspacePath, changedFiles, patchText)
  const applicable = changedFiles.length > 0 && patchText.trim().length > 0
    ? await checkWorkspacePatchApplicable({
        patchText,
        targetDir,
        ...(signal === undefined ? {} : { signal }),
      })
    : false

  return {
    changedFiles,
    insertions,
    deletions,
    patchText,
    cleanWorkingTree: changedFiles.length === 0,
    hasConflictMarkers,
    applicable,
  }
}

export async function inspectIsolatedWorkspacePatch(
  request: SwarmWorkspacePatchInspectRequest,
): Promise<SwarmWorkspacePatchInspection> {
  if (request.mode === 'inherit') {
    throw new ValidationError(
      'workspace',
      `cannot inspect isolated candidate patch for "${request.memberName}" because its workspace mode is "inherit"; use "share" or "branch"`,
      request.mode,
    )
  }

  const resolvedBase = await realpath(resolve(request.baseDir)).catch(() => resolve(request.baseDir))
  const resolvedWork = await realpath(resolve(request.workspacePath)).catch(() => resolve(request.workspacePath))
  const resolvedTarget = await realpath(resolve(request.targetDir ?? request.baseDir)).catch(() =>
    resolve(request.targetDir ?? request.baseDir),
  )

  const insideGit = await runGit(['rev-parse', '--is-inside-work-tree'], resolvedWork, request.signal)
  if (insideGit.exitCode !== 0 || insideGit.stdout !== 'true') {
    return inspectNonGitDirectoryDiff(resolvedBase, resolvedWork, resolvedTarget, request.signal)
  }

  const baseHead = await resolveBaseHeadInGitWorkspace(
    resolvedWork,
    resolvedBase,
    request.mode,
    request.baseHead,
    request.signal,
  )
  const headRes = await runGit(['rev-parse', '--verify', 'HEAD'], resolvedWork, request.signal)
  const head = headRes.exitCode === 0 && headRes.stdout ? headRes.stdout : undefined
  const baseRef = baseHead ?? 'HEAD'

  const [statusRes, trackedNamesRes, trackedStatRes, trackedDiffRes, untrackedRes] = await Promise.all([
    runGit(['status', '--porcelain', '--untracked-files=normal'], resolvedWork, request.signal),
    runGitRaw(['diff', '--no-renames', '--name-only', '-z', baseRef], resolvedWork, request.signal),
    runGitRaw(['diff', '--no-renames', '--numstat', '-z', baseRef], resolvedWork, request.signal),
    runGitRaw(['diff', '--no-renames', '--binary', '--full-index', baseRef], resolvedWork, request.signal),
    runGitRaw(['ls-files', '--others', '--exclude-standard', '-z'], resolvedWork, request.signal),
  ])

  if (trackedDiffRes.exitCode !== 0) {
    throw new Error(
      `Failed to inspect git diff in workspace "${resolvedWork}": ${trackedDiffRes.stderr || `exit ${trackedDiffRes.exitCode}`}`,
    )
  }

  const trackedFiles = parseNullDelimitedPaths(trackedNamesRes.stdout)
  const trackedSet = new Set(trackedFiles)
  const untrackedFiles = parseNullDelimitedPaths(untrackedRes.stdout)
    .filter(path => !trackedSet.has(path))
    .sort()

  const trackedStats = parseNumstatRecords(trackedStatRes.stdout, true)
  let insertions = trackedStats.insertions
  let deletions = trackedStats.deletions

  const patchChunks: string[] = []
  if (trackedDiffRes.stdout.trim().length > 0) {
    patchChunks.push(ensureTrailingNewline(trackedDiffRes.stdout))
  }

  for (const relPath of untrackedFiles) {
    const [untrackedDiff, untrackedStat] = await Promise.all([
      runGitRaw(
        ['diff', '--binary', '--full-index', '--no-index', '--', '/dev/null', relPath],
        resolvedWork,
        request.signal,
      ),
      runGit(
        ['diff', '--numstat', '--no-index', '--', '/dev/null', relPath],
        resolvedWork,
        request.signal,
      ),
    ])
    if (untrackedDiff.exitCode > 1) {
      throw new Error(
        `Failed to diff untracked file "${relPath}" in "${resolvedWork}": ${untrackedDiff.stderr || `exit ${untrackedDiff.exitCode}`}`,
      )
    }
    const stats = parseNumstatRecords(untrackedStat.stdout, false)
    insertions += stats.insertions
    deletions += stats.deletions
    if (untrackedDiff.stdout.length > 0) {
      patchChunks.push(ensureTrailingNewline(untrackedDiff.stdout))
    }
  }

  const changedFiles = [...new Set([...trackedFiles, ...untrackedFiles])].sort()
  const patchText = patchChunks.join('')
  const cleanWorkingTree = statusRes.exitCode === 0 && statusRes.stdout.length === 0
  const hasConflictMarkers = await detectWorkspaceConflictMarkers(resolvedWork, changedFiles, patchText)
  const applicable = changedFiles.length > 0 && patchText.trim().length > 0
    ? await checkWorkspacePatchApplicable({
        patchText,
        targetDir: resolvedTarget,
        ...(request.signal === undefined ? {} : { signal: request.signal }),
      })
    : false

  return {
    ...(baseHead === undefined ? {} : { baseHead }),
    ...(head === undefined ? {} : { head }),
    changedFiles,
    insertions,
    deletions,
    patchText,
    cleanWorkingTree,
    hasConflictMarkers,
    applicable,
  }
}

export interface DefaultSwarmWorkspaceOptions {
  readonly rootDir?: string
}

/**
 * Default Bun-native workspace allocator and candidate patch inspector for Swarm workers:
 * - `inherit`: shares the adder's workspace path directly.
 * - `share`: creates an isolated detached Git worktree (`git worktree add --detach`).
 * - `branch`: creates an isolated local Git clone (`git clone --local`), falling back
 *   to recursive directory copy when the source directory is not a Git repository.
 */
export function createDefaultSwarmWorkspacePort(
  options: DefaultSwarmWorkspaceOptions = {},
): SwarmWorkspacePort {
  const rootBase = resolve(options.rootDir ?? join(xerxesHome(), 'swarm-workspaces'))

  const allocationPathFor = (sessionId: string, baseDir: string, memberName: string, mode: 'share' | 'branch'): string => {
    const scopeHash = createHash('sha256')
      .update(`${resolve(baseDir)}\u0000${sessionId}`, 'utf8')
      .digest('hex')
      .slice(0, 16)
    const token = crypto.randomUUID().slice(0, 8)
    return join(rootBase, scopeHash, `${mode}-${memberName}-${token}`)
  }

  return {
    async createWorkspace(request: SwarmWorkspaceCreateRequest): Promise<SwarmWorkspaceAllocation> {
      const baseDir = resolve(request.baseDir)
      if (request.mode === 'inherit') {
        return {
          mode: 'inherit',
          path: baseDir,
        }
      }

      const targetPath = allocationPathFor(request.sessionId, baseDir, request.memberName, request.mode)
      await mkdir(dirname(targetPath), { recursive: true, mode: 0o700 })

      if (request.mode === 'share') {
        const commonProbe = await runGit(['rev-parse', '--path-format=absolute', '--git-common-dir'], baseDir, request.signal)
        if (commonProbe.exitCode !== 0 || !commonProbe.stdout) {
          throw new Error(
            `Cannot allocate "share" git worktree for "${request.memberName}": "${baseDir}" is not a Git repository (${commonProbe.stderr || 'git rev-parse failed'})`,
          )
        }
        const canonicalCommon = await realpath(commonProbe.stdout).catch(() => commonProbe.stdout)
        await withWorktreeLock(canonicalCommon, async () => {
          const added = await runGit(['worktree', 'add', '--detach', targetPath, 'HEAD'], baseDir, request.signal)
          if (added.exitCode !== 0) {
            throw new Error(
              `Failed to create "share" git worktree for "${request.memberName}": ${added.stderr || `exit ${added.exitCode}`}`,
            )
          }
        })
        const resolvedPath = await realpath(targetPath).catch(() => targetPath)
        const headProbe = await runGit(['rev-parse', '--verify', 'HEAD'], resolvedPath, request.signal)
        return {
          mode: 'share',
          path: resolvedPath,
          handle_id: canonicalCommon,
          ...(headProbe.exitCode === 0 && headProbe.stdout ? { base_head: headProbe.stdout } : {}),
        }
      }

      // mode === 'branch': prefer `git clone --local`, fall back to clean recursive copy for non-git workspaces.
      const gitTopProbe = await runGit(['rev-parse', '--is-inside-work-tree'], baseDir, request.signal)
      if (gitTopProbe.exitCode === 0 && gitTopProbe.stdout === 'true') {
        const cloned = await runGit(['clone', '--local', '--quiet', baseDir, targetPath], baseDir, request.signal)
        if (cloned.exitCode === 0) {
          const resolvedPath = await realpath(targetPath).catch(() => targetPath)
          const headProbe = await runGit(['rev-parse', '--verify', 'HEAD'], resolvedPath, request.signal)
          return {
            mode: 'branch',
            path: resolvedPath,
            ...(headProbe.exitCode === 0 && headProbe.stdout ? { base_head: headProbe.stdout } : {}),
          }
        }
      }

      await cp(baseDir, targetPath, { recursive: true })
      const resolvedPath = await realpath(targetPath).catch(() => targetPath)
      return {
        mode: 'branch',
        path: resolvedPath,
      }
    },

    async cleanupWorkspace(request: SwarmWorkspaceCleanupRequest): Promise<void> {
      if (request.mode === 'inherit') {
        return
      }
      const baseDir = resolve(request.baseDir)
      const targetPath = resolve(request.workspacePath)
      if (!targetPath || targetPath === baseDir) {
        return
      }

      if (request.mode === 'share') {
        const lockKey = request.handleId ?? baseDir
        await withWorktreeLock(lockKey, async () => {
          await runGit(['worktree', 'remove', '--force', targetPath], baseDir).catch(() => undefined)
          await rm(targetPath, { recursive: true, force: true }).catch(() => undefined)
          await runGit(['worktree', 'prune'], baseDir).catch(() => undefined)
        })
        return
      }

      await rm(targetPath, { recursive: true, force: true }).catch(() => undefined)
    },

    async inspectWorkspacePatch(request: SwarmWorkspacePatchInspectRequest): Promise<SwarmWorkspacePatchInspection> {
      return inspectIsolatedWorkspacePatch(request)
    },

    async checkPatchApplicable(request: SwarmWorkspacePatchApplyRequest): Promise<boolean> {
      return checkWorkspacePatchApplicable(request)
    },

    async applyPatchToWorkspace(request: SwarmWorkspacePatchApplyRequest): Promise<void> {
      await applyWorkspacePatch(request)
    },
  }
}
