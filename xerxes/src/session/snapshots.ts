// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomUUID } from 'node:crypto'
import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'

const GIT_COMMAND_TIMEOUT_MS = 30_000

/** Mutating shadow-Git operations share an index and record log across manager instances. */
const repositoryOperations = new Map<string, Promise<void>>()

/** Paths the shadow repository never tracks or deletes: Xerxes state and common secret files. */
const SHADOW_EXCLUDE_PATTERNS = [
  '.xerxes/snapshots/**',
  '.env*',
  '*.pem',
  '*.key',
  '*credentials*',
  '*secret*',
  'id_rsa*',
  '.ssh/**',
  '.npmrc',
  '.netrc',
  '*.p12',
  '*.keystore',
  'kubeconfig*',
] as const

/** Number of tab-separated fields written before the session/turn link existed. */
const LEGACY_RECORD_FIELDS = 5

export interface SnapshotRecord {
  readonly commitSha: string
  readonly createdAt: string
  readonly id: string
  readonly label: string
  /** Session that owns this snapshot; absent for manual or pre-link records. */
  readonly sessionId?: string
  /**
   * Index of the turn this snapshot precedes. Without it "take me back to
   * before turn 7" is unexpressible: a bare timestamp cannot be matched to a
   * point in the conversation the user actually remembers.
   */
  readonly turnIndex?: number
  readonly workspaceDir: string
}

/** Conversation coordinates attached to an automatic snapshot. */
export interface SnapshotLink {
  readonly sessionId?: string
  readonly turnIndex?: number
}

/**
 * Creates git snapshots in a bare shadow repository without modifying the
 * workspace's own git metadata or history.
 *
 * Git commands run asynchronously with a bounded lifetime so snapshot work
 * never blocks the daemon's event loop, and the shadow directory is created
 * private (0o700) because it mirrors workspace contents.
 */
export class SnapshotManager {
  readonly workspaceDirectory: string
  private readonly recordsPath: string
  private readonly shadowRoot: string

  constructor(workspaceDirectory: string, options: SnapshotManagerOptions = {}) {
    this.workspaceDirectory = resolve(workspaceDirectory)
    this.shadowRoot = resolve(options.shadowRoot ?? join(xerxesHome(), 'snapshots'))
    this.recordsPath = join(this.shadowDirectory, '_records.txt')
  }

  get shadowDirectory(): string {
    return join(this.shadowRoot, workspaceHash(this.workspaceDirectory))
  }

  get(ref: string): SnapshotRecord | undefined {
    if (ref.length === 0) return undefined
    const records = this.list()
    const exact = records.find(record => record.id === ref || record.label === ref)
    if (exact) return exact
    // Empty or short SHA prefixes silently matched the first record, letting
    // rollback('') restore an arbitrary snapshot. Require enough entropy and
    // refuse ambiguous prefixes instead.
    if (ref.length < 4) return undefined
    const matches = records.filter(record => record.commitSha.startsWith(ref))
    if (matches.length > 1) {
      throw new Error(`ambiguous snapshot ref: ${ref} matches ${matches.length} snapshots`)
    }
    return matches[0]
  }

  /**
   * Read the record log, tolerating rows written before snapshots carried a
   * session/turn link.
   *
   * Those rows have five fields instead of seven. Requiring the new width
   * would silently discard every snapshot a user had already taken, so a short
   * row is read as an unlinked record and a longer one keeps only the fields
   * this version understands.
   */
  list(): SnapshotRecord[] {
    if (!existsSync(this.recordsPath)) return []
    return readFileSync(this.recordsPath, 'utf8').split(/\r?\n/).flatMap(line => {
      if (!line.trim()) return []
      const parts = line.split('\t')
      if (parts.length < LEGACY_RECORD_FIELDS) return []
      const [id, label, commitSha, createdAt, workspaceDir, sessionId, turnIndex] = parts
      if (!id || label === undefined || !commitSha || !createdAt || !workspaceDir) return []
      const turn = parseTurnIndex(turnIndex)
      return [{
        id,
        label,
        commitSha,
        createdAt,
        workspaceDir,
        ...(sessionId ? { sessionId } : {}),
        ...(turn === undefined ? {} : { turnIndex: turn }),
      }]
    })
  }

  /** Snapshots taken for one session, oldest first. */
  listForSession(sessionId: string): SnapshotRecord[] {
    if (!sessionId) return []
    return this.list().filter(record => record.sessionId === sessionId)
  }

  /**
   * The snapshot capturing the workspace as it stood before a given turn.
   *
   * The newest match wins: a retried turn snapshots the same index again, and
   * the later capture is the one that precedes the attempt still in the
   * transcript.
   */
  getForTurn(sessionId: string, turnIndex: number): SnapshotRecord | undefined {
    return this.listForSession(sessionId).filter(record => record.turnIndex === turnIndex).at(-1)
  }

  async prune(options: SnapshotPruneOptions = {}): Promise<number> {
    return this.serializeRepositoryOperation(() => this.pruneUnlocked(options))
  }

  private async pruneUnlocked(options: SnapshotPruneOptions): Promise<number> {
    const keep = options.keep ?? 100
    if (!Number.isInteger(keep) || keep < 0) throw new RangeError('keep must be a non-negative integer')
    const records = this.list()
    if (records.length <= keep) return 0
    const retained = keep === 0 ? [] : records.slice(-keep)
    if (retained.length === 0) {
      this.resetUnlocked()
      return records.length
    }
    // Re-anchor retained history on a fresh root commit so the pruned commits
    // become unreachable and `git gc` can collect them; otherwise the bare
    // repo would grow with every snapshot forever. Retained records keep their
    // ids and labels while their rewritten commit SHAs are stored back.
    const rewritten: SnapshotRecord[] = []
    let parent: string | undefined
    for (const record of retained) {
      const tree = (await this.runGitUnlocked(['rev-parse', `${record.commitSha}^{tree}`])).trim()
      const args = ['commit-tree', tree, '-m', record.label || `snapshot-${record.createdAt}`]
      if (parent) args.push('-p', parent)
      parent = (await this.runGitUnlocked(args)).trim()
      rewritten.push({ ...record, commitSha: parent })
    }
    if (parent) await this.runGitUnlocked(['update-ref', 'HEAD', parent])
    // Bare repositories normally keep no reflogs; expiry is best-effort.
    await this.runGitUnlocked(['reflog', 'expire', '--expire=now', '--all']).catch(() => '')
    await this.runGitUnlocked(['gc', '--prune=now', '--quiet'])
    this.writeRecords(rewritten)
    return records.length - retained.length
  }

  async reset(): Promise<void> {
    await this.serializeRepositoryOperation(async () => this.resetUnlocked())
  }

  private resetUnlocked(): void {
    rmSync(this.shadowDirectory, { recursive: true, force: true })
  }

  async rollback(ref: string): Promise<SnapshotRecord> {
    return this.serializeRepositoryOperation(() => this.rollbackUnlocked(ref))
  }

  private async rollbackUnlocked(ref: string): Promise<SnapshotRecord> {
    const record = this.get(ref)
    if (!record) throw new Error(`snapshot not found: ${ref}`)
    // checkout-index overwrites modified files without a backup, so capture
    // the current tree first; the pre-rollback snapshot can itself be
    // rolled back to undo a mistaken restore.
    await this.snapshotUnlocked(`pre-rollback:${record.id}`)
    // Full-tree restore: point the index at the snapshot tree, rewrite every
    // tracked file, then delete files the snapshot does not track. `-x` also
    // removes ignored build outputs created after the snapshot (plain `-fd`
    // would honor the workspace .gitignore and leave a mixed tree), while the
    // explicit `-e` patterns keep shadow-excluded secrets and Xerxes state.
    await this.runGitUnlocked(['read-tree', record.commitSha])
    await this.runGitUnlocked(['checkout-index', '-f', '-a'])
    await this.runGitUnlocked(['clean', '-fdx', ...SHADOW_EXCLUDE_PATTERNS.flatMap(pattern => ['-e', pattern])])
    return record
  }

  async snapshot(label = '', link: SnapshotLink = {}): Promise<SnapshotRecord> {
    return this.serializeRepositoryOperation(() => this.snapshotUnlocked(label, link))
  }

  private async snapshotUnlocked(label = '', link: SnapshotLink = {}): Promise<SnapshotRecord> {
    await this.ensureRepository()
    await this.runGitUnlocked(['add', '-A'])
    const message = label || `snapshot-${new Date().toISOString()}`
    await this.runGitUnlocked(['commit', '--allow-empty', '-m', message])
    const commitSha = (await this.runGitUnlocked(['rev-parse', 'HEAD'])).trim()
    const turnIndex = link.turnIndex
    const record: SnapshotRecord = {
      id: randomUUID().replaceAll('-', '').slice(0, 12),
      label,
      commitSha,
      createdAt: new Date().toISOString(),
      workspaceDir: this.workspaceDirectory,
      ...(link.sessionId ? { sessionId: link.sessionId } : {}),
      ...(turnIndex === undefined || !Number.isInteger(turnIndex) || turnIndex < 0 ? {} : { turnIndex }),
    }
    this.appendRecord(record)
    return record
  }

  /**
   * Restore one file from a snapshot without touching the rest of the tree.
   *
   * A full rollback is the wrong tool when a single file was damaged: it also
   * discards every unrelated edit made since. Like rollback, this captures the
   * current tree first, because `git checkout` overwrites the target with no
   * backup of its own.
   */
  async restoreFile(ref: string, filePath: string): Promise<SnapshotRestoreResult> {
    return this.serializeRepositoryOperation(() => this.restoreFileUnlocked(ref, filePath))
  }

  private async restoreFileUnlocked(ref: string, filePath: string): Promise<SnapshotRestoreResult> {
    const record = this.get(ref)
    if (!record) throw new Error(`snapshot not found: ${ref}`)
    const path = this.workspaceRelativePath(filePath)
    await this.ensureRepository()
    // `cat-file -e` distinguishes "the snapshot never tracked this file" from a
    // genuine git failure, which `checkout` alone reports as the same error.
    const tracked = await this.runGitUnlocked(['cat-file', '-e', `${record.commitSha}:${path}`]).then(() => true, () => false)
    if (!tracked) throw new Error(`snapshot ${record.id} does not track ${path}`)
    const previous = await this.snapshotUnlocked(`pre-restore:${record.id}`)
    // `:(literal)` keeps a filename containing glob characters from being
    // expanded into a pathspec that would restore unrelated files.
    await this.runGitUnlocked(['checkout', record.commitSha, '--', `:(literal)${path}`])
    return { path, previous, snapshot: record }
  }

  /** Run a command against the shadow repository for snapshot-diff consumers. */
  async runGit(args: readonly string[]): Promise<string> {
    return this.serializeRepositoryOperation(() => this.runGitUnlocked(args))
  }

  private async runGitUnlocked(args: readonly string[]): Promise<string> {
    return runGitProcess(args, {
      cwd: this.workspaceDirectory,
      env: {
        ...process.env,
        GIT_DIR: join(this.shadowDirectory, '.git'),
        GIT_WORK_TREE: this.workspaceDirectory,
        GIT_AUTHOR_NAME: 'xerxes-snapshot',
        GIT_AUTHOR_EMAIL: 'snapshots@xerxes',
        GIT_COMMITTER_NAME: 'xerxes-snapshot',
        GIT_COMMITTER_EMAIL: 'snapshots@xerxes',
      },
    })
  }

  private async serializeRepositoryOperation<T>(operation: () => Promise<T>): Promise<T> {
    const key = this.shadowDirectory
    const previous = repositoryOperations.get(key) ?? Promise.resolve()
    let release!: () => void
    const current = new Promise<void>(resolveOperation => { release = resolveOperation })
    repositoryOperations.set(key, current)
    await previous.catch(() => undefined)
    try {
      return await operation()
    } finally {
      release()
      if (repositoryOperations.get(key) === current) repositoryOperations.delete(key)
    }
  }

  private appendRecord(record: SnapshotRecord): void {
    const existing = existsSync(this.recordsPath) ? readFileSync(this.recordsPath, 'utf8') : ''
    const content = `${existing}${existing && !existing.endsWith('\n') ? '\n' : ''}${recordLine(record)}\n`
    this.writeTextAtomically(this.recordsPath, content)
  }

  /** Resolve a caller-supplied path to a workspace-relative, git-usable path. */
  private workspaceRelativePath(candidate: string): string {
    const trimmed = candidate.trim()
    if (!trimmed) throw new Error('a file path is required')
    const relativePath = relative(this.workspaceDirectory, resolve(this.workspaceDirectory, trimmed))
    // A `../` path would let a restore write anywhere the daemon can write,
    // driven by nothing more than a snapshot ref and an attacker-chosen path.
    if (!relativePath || relativePath === '..' || relativePath.startsWith(`..${sep}`) || isAbsolute(relativePath)) {
      throw new Error(`path escapes the snapshot workspace: ${candidate}`)
    }
    return relativePath.split(sep).join('/')
  }

  private async ensureRepository(): Promise<void> {
    const gitDirectory = join(this.shadowDirectory, '.git')
    if (!existsSync(gitDirectory)) {
      mkdirSync(this.shadowDirectory, { recursive: true, mode: 0o700 })
      // Normalize permissions even when the directory already existed.
      chmodSync(this.shadowDirectory, 0o700)
      await runGitProcess(['init', '--bare', '--quiet', '--initial-branch', 'main', gitDirectory], {
        cwd: this.workspaceDirectory,
        env: { ...process.env },
      })
    }
    this.ensureExcludePatterns(join(gitDirectory, 'info'))
  }

  private ensureExcludePatterns(infoDirectory: string): void {
    mkdirSync(infoDirectory, { recursive: true })
    const path = join(infoDirectory, 'exclude')
    const existing = existsSync(path) ? readFileSync(path, 'utf8') : ''
    const present = new Set(existing.split(/\r?\n/))
    const missing = SHADOW_EXCLUDE_PATTERNS.filter(pattern => !present.has(pattern))
    if (missing.length === 0) return
    const separator = existing.length > 0 && !existing.endsWith('\n') ? '\n' : ''
    writeFileSync(path, `${existing}${separator}${missing.join('\n')}\n`, 'utf8')
  }

  private writeRecords(records: readonly SnapshotRecord[]): void {
    this.writeTextAtomically(this.recordsPath, records.map(recordLine).join('\n'))
  }

  private writeTextAtomically(path: string, content: string): void {
    mkdirSync(dirname(path), { recursive: true })
    const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`
    try {
      writeFileSync(temporary, content, 'utf8')
      renameSync(temporary, path)
    } catch (error) {
      rmSync(temporary, { force: true })
      throw error
    }
  }
}

export interface SnapshotManagerOptions {
  readonly shadowRoot?: string
}

export interface SnapshotPruneOptions {
  readonly keep?: number
}

/** What a single-file restore replaced, and the snapshot that can undo it. */
export interface SnapshotRestoreResult {
  readonly path: string
  readonly previous: SnapshotRecord
  readonly snapshot: SnapshotRecord
}

/** One tab-separated record row; every field is scrubbed of the row separators. */
function recordLine(record: SnapshotRecord): string {
  return [
    record.id,
    record.label,
    record.commitSha,
    record.createdAt,
    record.workspaceDir,
    record.sessionId ?? '',
    record.turnIndex === undefined ? '' : String(record.turnIndex),
  ].map(field => field.replaceAll(/[\t\r\n]/g, ' ')).join('\t')
}

/** A turn index is only trusted when it survives a round trip as a non-negative integer. */
function parseTurnIndex(value: string | undefined): number | undefined {
  if (value === undefined || value.trim() === '') return undefined
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : undefined
}

/** Run one git invocation with a hard timeout, killing the process when it overruns. */
async function runGitProcess(
  args: readonly string[],
  options: { readonly cwd: string; readonly env: Record<string, string | undefined> },
): Promise<string> {
  const child = Bun.spawn(['git', ...args], {
    cwd: options.cwd,
    env: options.env,
    stdout: 'pipe',
    stderr: 'pipe',
  })
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    child.kill()
  }, GIT_COMMAND_TIMEOUT_MS)
  try {
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])
    if (timedOut) throw new Error(`git ${args.join(' ')} timed out after ${GIT_COMMAND_TIMEOUT_MS}ms`)
    if (exitCode !== 0) {
      throw new Error(`git ${args.join(' ')} failed (exit ${exitCode}): ${stderr.trim()}`)
    }
    return stdout
  } finally {
    clearTimeout(timer)
  }
}

function workspaceHash(workspaceDirectory: string): string {
  return createHash('sha1').update(workspaceDirectory).digest('hex').slice(0, 12)
}
